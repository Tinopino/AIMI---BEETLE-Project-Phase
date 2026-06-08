#!/usr/bin/env bash
set -euo pipefail
umask 002

ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase

BASE_MODEL_DIR=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results/Dataset301_BEETLE/nnUNetTrainer_CutMixStainEMA__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d

BASE_TRAINER_FILE="$ROOT/nnUNet_pathology/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py"

BEETLE_TRAINER_FILE="$ROOT/nnUNet_pathology/nnunetv2/training/nnUNetTrainer/nnUnetTrainerBEETLE.py"

cd "$ROOT"
mkdir -p logs

###############################################################################
# 1. CANCEL ONLY STALE JOBS FROM EARLIER PIPELINE ATTEMPTS
#
# Only clean numeric job IDs are passed to scancel.
###############################################################################
STALE_IDS_FILE=/tmp/tijn_beetle_stale_job_ids.txt

squeue -h -u "$USER" -o "%A|%j" |
awk -F'|' '
    $2 ~ /^(cutmixema-|ctxft-|eval-cema-|eval-ctx-|agg-cema-|agg-final-|test-f0-resume)/ {
        print $1
    }
' |
sort -u > "$STALE_IDS_FILE"

if [[ -s "$STALE_IDS_FILE" ]]; then
    echo "Cancelling stale jobs:"
    cat "$STALE_IDS_FILE"
    xargs -r scancel < "$STALE_IDS_FILE"
else
    echo "No stale pipeline jobs found."
fi

echo
echo "Remaining jobs:"
squeue -u "$USER" \
  -o "%.18i %.32j %.2t %.12M %.45R"

###############################################################################
# 2. VERIFY THAT FOLD 0 STILL HAS A HEALTHY RESUME CHECKPOINT
###############################################################################
CHECKPOINT="$BASE_MODEL_DIR/fold_0/checkpoint_latest.pth"

test -s "$CHECKPOINT" || {
    echo "ERROR: fold-0 checkpoint_latest.pth is missing:" >&2
    echo "  $CHECKPOINT" >&2
    exit 10
}

echo
echo "Fold-0 checkpoint found:"
ls -lh "$CHECKPOINT"

###############################################################################
# 3. REMOVE THE INVALID CONTEXT CLASS IF IT WAS APPENDED TO THE WRONG MODULE
#
# It is not needed for base five-fold training and previously broke imports.
###############################################################################
cp -n "$BEETLE_TRAINER_FILE" \
      "${BEETLE_TRAINER_FILE}.before_resume_fix.bak" \
      || true

python - <<'PY'
from pathlib import Path

path = Path(
    "nnUNet_pathology/nnunetv2/training/"
    "nnUNetTrainer/nnUnetTrainerBEETLE.py"
)

text = path.read_text()

marker = "\nclass nnUNetTrainer_CutMixStainEMA_Context1024FT100"

if marker not in text:
    print("No invalid appended context class found")
else:
    prefix, suffix = text.split(marker, 1)

    # This context class was appended at the end during the earlier attempt.
    # Refuse to truncate if another top-level class follows it.
    if "\nclass " in suffix:
        raise RuntimeError(
            "Unexpected class definitions after the invalid context trainer. "
            "Inspect nnUnetTrainerBEETLE.py manually before continuing."
        )

    path.write_text(prefix.rstrip() + "\n")

    print("Removed invalid appended context class from:")
    print(" ", path)
PY

###############################################################################
# 4. PATCH THE BASE nnUNetTrainer.load_checkpoint() METHOD
#
# Before:
#   strings work, dictionaries leave `checkpoint` undefined.
#
# After:
#   both strings and checkpoint dictionaries work.
###############################################################################
cp -n "$BASE_TRAINER_FILE" \
      "${BASE_TRAINER_FILE}.before_resume_fix.bak" \
      || true

python - <<'PY'
from pathlib import Path

path = Path(
    "nnUNet_pathology/nnunetv2/training/"
    "nnUNetTrainer/nnUNetTrainer.py"
)

text = path.read_text()

already_fixed = """        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
        else:
            checkpoint = filename_or_checkpoint
"""

old = """        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
"""

if already_fixed in text:
    print("Base load_checkpoint() dictionary handling was already fixed")
elif old in text:
    text = text.replace(old, already_fixed, 1)
    path.write_text(text)

    print("Patched base load_checkpoint() dictionary handling:")
    print(" ", path)
else:
    raise RuntimeError(
        "Could not find the expected load_checkpoint() block. "
        "Inspect nnUNetTrainer.py manually before continuing."
    )
PY

###############################################################################
# 5. COMPILE AND IMPORT TEST BEFORE SUBMITTING A GPU JOB
###############################################################################
source /scratch/$USER/virtual_environments/AIMI-BEETLE/bin/activate

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

export nnUNet_raw=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_raw
export nnUNet_preprocessed=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_preprocessed
export nnUNet_results=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results

python -m py_compile \
  "$BASE_TRAINER_FILE" \
  "$BEETLE_TRAINER_FILE" \
  nnUNet_pathology/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_CutMixStainEMA.py

python - <<'PY'
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import (
    nnUNetTrainer,
)

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_CutMixStainEMA import (
    nnUNetTrainer_CutMixStainEMA,
)

assert hasattr(nnUNetTrainer, "load_checkpoint")
assert hasattr(nnUNetTrainer_CutMixStainEMA, "load_checkpoint")

print("Verified base trainer import")
print("Verified CutMix+EMA trainer import")
print("Verified load_checkpoint() methods")
PY

###############################################################################
# 6. CREATE A LOAD-ONLY FOLD-0 SMOKE TEST
#
# This does not train. It only instantiates the trainer and restores:
# - raw model weights;
# - optimizer state;
# - logger state;
# - epoch number;
# - EMA state;
# - global step.
###############################################################################
cat > smoke_test_fold0_resume.py <<'PY'
#!/usr/bin/env python3

from pathlib import Path

from nnunetv2.run.run_training_pathology import get_trainer_from_args


checkpoint = Path(
    "/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/"
    "nnUNet_results/Dataset301_BEETLE/"
    "nnUNetTrainer_CutMixStainEMA__"
    "nnUNetWholeSlideDataPlans__"
    "wsd_None_iterator_nnunet_aug__2d/"
    "fold_0/checkpoint_latest.pth"
)

if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)

trainer = get_trainer_from_args(
    "301",
    "2d",
    0,
    "nnUNetTrainer_CutMixStainEMA",
    "nnUNetWholeSlideDataPlans",
)

print("Loading fold-0 checkpoint:", checkpoint, flush=True)

trainer.load_checkpoint(str(checkpoint))

print("Resume smoke test passed", flush=True)
print("Loaded current_epoch:", trainer.current_epoch, flush=True)
print("Loaded _best_ema:", trainer._best_ema, flush=True)
print("Loaded _global_step:", trainer._global_step, flush=True)
print("EMA initialized:", trainer._ema_initialized, flush=True)
PY

chmod +x smoke_test_fold0_resume.py

cat > smoke_test_fold0_resume.slurm <<'SLURM'
#!/usr/bin/env bash
#SBATCH --account=cseduimc037
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --nodelist=cn48
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --job-name=test-f0-resume
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase

cd "$ROOT"

source /scratch/$USER/virtual_environments/AIMI-BEETLE/bin/activate

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

export nnUNet_raw=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_raw
export nnUNet_preprocessed=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_preprocessed
export nnUNet_results=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results

python -u smoke_test_fold0_resume.py

echo "Fold-0 resume smoke test completed successfully."
SLURM

chmod +x smoke_test_fold0_resume.slurm

###############################################################################
# 7. SUBMIT ONLY THE SMOKE TEST
###############################################################################
SMOKE_JOB=$(
    sbatch --parsable \
      smoke_test_fold0_resume.slurm
)

echo
echo "======================================================================"
echo "SUBMITTED LOAD-ONLY FOLD-0 RESUME TEST"
echo "======================================================================"
echo
echo "Smoke-test job ID: $SMOKE_JOB"
echo
echo "No fold training jobs have been queued yet."
echo

squeue -u "$USER" \
  -o "%.18i %.32j %.2t %.12M %.45R"
