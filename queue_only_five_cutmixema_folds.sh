#!/usr/bin/env bash
set -euo pipefail
umask 002

ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase

NNUNET_RESULTS=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results

BASE_MODEL_DIR="$NNUNET_RESULTS/Dataset301_BEETLE/nnUNetTrainer_CutMixStainEMA__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"

###############################################################################
# Fold 1 is already running correctly from the earlier submission.
###############################################################################
CURRENT_FOLD1_JOB=10425050

cd "$ROOT"
mkdir -p logs

###############################################################################
# 1. Verify fold 0 has a resumable checkpoint.
###############################################################################
test -s "$BASE_MODEL_DIR/fold_0/checkpoint_latest.pth" || {
    echo "ERROR: fold-0 checkpoint_latest.pth is missing:" >&2
    echo "  $BASE_MODEL_DIR/fold_0/checkpoint_latest.pth" >&2
    exit 10
}

echo "Fold-0 latest checkpoint:"
ls -lh "$BASE_MODEL_DIR/fold_0/checkpoint_latest.pth"

###############################################################################
# 2. Verify the existing fold-1 job is still queued or running.
###############################################################################
FOLD1_STATE=$(squeue -h -j "$CURRENT_FOLD1_JOB" -o "%T" || true)

if [[ "$FOLD1_STATE" != "RUNNING" && "$FOLD1_STATE" != "PENDING" ]]; then
    echo "ERROR: expected fold-1 job $CURRENT_FOLD1_JOB to be RUNNING or PENDING." >&2
    echo "Observed state: '${FOLD1_STATE:-not found}'" >&2
    echo "Inspect with: squeue -u \$USER" >&2
    exit 11
fi

echo "Existing fold-1 job:"
echo "  $CURRENT_FOLD1_JOB ($FOLD1_STATE)"

###############################################################################
# 3. Correct resume wrapper:
#    explicitly load the checkpoint dictionary before passing it to the trainer.
###############################################################################
cat > run_training_pathology_resume.py <<'PY'
#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from nnunetv2.run.run_training_pathology import get_trainer_from_args


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume pathology nnU-Net training from checkpoint_latest.pth."
    )

    parser.add_argument("dataset_name_or_id")
    parser.add_argument("fold", type=int)
    parser.add_argument("trainer")

    parser.add_argument(
        "--planner",
        default="nnUNetWholeSlideDataPlans",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--load-only",
        action="store_true",
        help="Load the checkpoint and exit without training.",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    trainer = get_trainer_from_args(
        args.dataset_name_or_id,
        "2d",
        args.fold,
        args.trainer,
        args.planner,
    )

    print(f"Reading checkpoint dictionary: {checkpoint_path}", flush=True)

    checkpoint_dict = torch.load(
        str(checkpoint_path),
        map_location="cpu",
    )

    if not isinstance(checkpoint_dict, dict):
        raise TypeError(
            "Expected torch.load() to return a checkpoint dictionary, "
            f"received {type(checkpoint_dict).__name__}"
        )

    if "network_weights" not in checkpoint_dict:
        raise KeyError(
            "Checkpoint dictionary does not contain 'network_weights'. "
            f"Available keys: {sorted(checkpoint_dict.keys())}"
        )

    print(
        "Checkpoint dictionary loaded successfully. "
        f"Keys: {sorted(checkpoint_dict.keys())}",
        flush=True,
    )

    trainer.load_checkpoint(checkpoint_dict)

    print("Trainer state loaded successfully", flush=True)

    if args.load_only:
        print("Load-only smoke test completed successfully", flush=True)
        return

    print("Continuing training", flush=True)

    trainer.run_training()


if __name__ == "__main__":
    main()
PY

chmod +x run_training_pathology_resume.py

###############################################################################
# 4. Validate Python syntax and imports before submitting anything.
###############################################################################
source /scratch/$USER/virtual_environments/AIMI-BEETLE/bin/activate

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

export nnUNet_raw=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_raw
export nnUNet_preprocessed=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_preprocessed
export nnUNet_results=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results

python -m py_compile run_training_pathology_resume.py

python - <<'PY'
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_CutMixStainEMA import (
    nnUNetTrainer_CutMixStainEMA,
)

assert hasattr(nnUNetTrainer_CutMixStainEMA, "load_checkpoint")

print("Verified CutMix + EMA trainer import")
print("Verified load_checkpoint() availability")
PY

###############################################################################
# 5. Create a short fold-0 checkpoint-loading smoke test.
###############################################################################
cat > test_fold0_resume_load.slurm <<'SLURM'
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

CHECKPOINT="$nnUNet_results/Dataset301_BEETLE/nnUNetTrainer_CutMixStainEMA__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d/fold_0/checkpoint_latest.pth"

python -u run_training_pathology_resume.py \
    301 \
    0 \
    nnUNetTrainer_CutMixStainEMA \
    --planner nnUNetWholeSlideDataPlans \
    --checkpoint "$CHECKPOINT" \
    --load-only

echo "Fold-0 resume smoke test passed."
SLURM

chmod +x test_fold0_resume_load.slurm

###############################################################################
# 6. Create the robust 12-hour base-training launcher.
#
# Each fold gets at most two chunks.
# A chunk:
# - resumes automatically if checkpoint_latest.pth exists;
# - starts fresh if no checkpoint exists;
# - exits safely if checkpoint_final.pth already exists;
# - uses a valid timeout value: 42600 seconds = 11h50m;
# - accepts only natural completion or the expected timeout code.
###############################################################################
cat > run_fivefold_cutmixema_chunk.slurm <<'SLURM'
#!/usr/bin/env bash
#SBATCH --account=cseduimc037
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --nodelist=cn48
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=18G
#SBATCH --time=12:00:00
#SBATCH --job-name=cutmixema-fold
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
umask 002

ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase

FOLD="${1:?Usage: sbatch run_fivefold_cutmixema_chunk.slurm <fold> <intermediate|final>}"
ROLE="${2:?Usage: sbatch run_fivefold_cutmixema_chunk.slurm <fold> <intermediate|final>}"

TRAINER=nnUNetTrainer_CutMixStainEMA
PLANNER=nnUNetWholeSlideDataPlans

cd "$ROOT"
mkdir -p logs

source /scratch/$USER/virtual_environments/AIMI-BEETLE/bin/activate

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

export nnUNet_raw=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_raw
export nnUNet_preprocessed=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_preprocessed
export nnUNet_results=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export WANDB_MODE=disabled
export WANDB_DISABLED=true

MODEL_BASE="$nnUNet_results/Dataset301_BEETLE/${TRAINER}__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"

RESULT_DIR="$MODEL_BASE/fold_${FOLD}"

LATEST="$RESULT_DIR/checkpoint_latest.pth"
FINAL="$RESULT_DIR/checkpoint_final.pth"

mkdir -p "$RESULT_DIR"

echo "======================================================================"
echo "BEETLE CutMix + stain jitter + EMA fold chunk"
echo "Started:      $(date)"
echo "Node:         $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "Fold:         $FOLD"
echo "Role:         $ROLE"
echo "Result dir:   $RESULT_DIR"
echo "======================================================================"

if [[ -s "$FINAL" ]]; then
    echo "Fold $FOLD already completed 1000 epochs."
    exit 0
fi

if [[ -s "$LATEST" ]]; then
    echo "Existing checkpoint_latest.pth detected."
    echo "Resuming fold $FOLD."

    TRAIN_CMD=(
        python -u
        run_training_pathology_resume.py
        301 "$FOLD" "$TRAINER"
        --planner "$PLANNER"
        --checkpoint "$LATEST"
    )
else
    echo "No checkpoint found."
    echo "Starting fold $FOLD fresh."

    TRAIN_CMD=(
        python -u
        nnUNet_pathology/nnunetv2/run/run_training_pathology.py
        301 "$FOLD" "$TRAINER"
        --planner "$PLANNER"
    )
fi

echo
echo "Executing:"
printf '  %q' "${TRAIN_CMD[@]}"
echo
echo

set +e

timeout \
    --signal=TERM \
    --kill-after=60s \
    42600s \
    "${TRAIN_CMD[@]}"

RC=$?

set -e

echo
echo "Training command exit code: $RC"

###############################################################################
# Valid outcomes:
# 0   = training naturally completed;
# 124 = expected timeout after 11h50m.
###############################################################################
if [[ "$RC" -ne 0 && "$RC" -ne 124 ]]; then
    echo "ERROR: unexpected training failure with exit code $RC." >&2
    exit 30
fi

if [[ -s "$FINAL" ]]; then
    echo "SUCCESS: fold $FOLD completed 1000 epochs."
    exit 0
fi

if [[ ! -s "$LATEST" ]]; then
    echo "ERROR: no checkpoint_latest.pth exists after this chunk:" >&2
    echo "  $LATEST" >&2
    exit 31
fi

if [[ "$ROLE" == "intermediate" ]]; then
    echo "SUCCESS: fold $FOLD remains resumable."
    echo "The next queued chunk will continue from checkpoint_latest.pth."
    exit 0
fi

if [[ "$ROLE" == "final" ]]; then
    echo "ERROR: fold $FOLD is still incomplete after its final allocated chunk." >&2
    echo "Queue one additional resume chunk before evaluation." >&2
    exit 32
fi

echo "ERROR: role must be intermediate or final." >&2
exit 33
SLURM

chmod +x run_fivefold_cutmixema_chunk.slurm

###############################################################################
# 7. Submit a smoke test and queue ONLY the five base folds.
#
# Maximum simultaneous GPU jobs: 2.
#
# Lane A:
#   fold-0 smoke test
#   -> fold 0 resume chunks
#   -> fold 2 chunks
#   -> fold 4 chunks
#
# Lane B:
#   existing fold-1 chunk
#   -> fold 1 second chunk
#   -> fold 3 chunks
###############################################################################
SMOKE_JOB=$(
    sbatch --parsable \
        test_fold0_resume_load.slurm
)

F0_C1=$(
    sbatch --parsable \
        --dependency=afterok:${SMOKE_JOB} \
        --job-name=cutmixema-base-f0-c1 \
        run_fivefold_cutmixema_chunk.slurm \
        0 intermediate
)

F0_C2=$(
    sbatch --parsable \
        --dependency=afterok:${F0_C1} \
        --job-name=cutmixema-base-f0-c2 \
        run_fivefold_cutmixema_chunk.slurm \
        0 final
)

F1_C2=$(
    sbatch --parsable \
        --dependency=afterok:${CURRENT_FOLD1_JOB} \
        --job-name=cutmixema-base-f1-c2 \
        run_fivefold_cutmixema_chunk.slurm \
        1 final
)

F2_C1=$(
    sbatch --parsable \
        --dependency=afterok:${F0_C2} \
        --job-name=cutmixema-base-f2-c1 \
        run_fivefold_cutmixema_chunk.slurm \
        2 intermediate
)

F2_C2=$(
    sbatch --parsable \
        --dependency=afterok:${F2_C1} \
        --job-name=cutmixema-base-f2-c2 \
        run_fivefold_cutmixema_chunk.slurm \
        2 final
)

F3_C1=$(
    sbatch --parsable \
        --dependency=afterok:${F1_C2} \
        --job-name=cutmixema-base-f3-c1 \
        run_fivefold_cutmixema_chunk.slurm \
        3 intermediate
)

F3_C2=$(
    sbatch --parsable \
        --dependency=afterok:${F3_C1} \
        --job-name=cutmixema-base-f3-c2 \
        run_fivefold_cutmixema_chunk.slurm \
        3 final
)

F4_C1=$(
    sbatch --parsable \
        --dependency=afterok:${F2_C2} \
        --job-name=cutmixema-base-f4-c1 \
        run_fivefold_cutmixema_chunk.slurm \
        4 intermediate
)

F4_C2=$(
    sbatch --parsable \
        --dependency=afterok:${F4_C1} \
        --job-name=cutmixema-base-f4-c2 \
        run_fivefold_cutmixema_chunk.slurm \
        4 final
)

###############################################################################
# 8. Save readable job manifest.
###############################################################################
MANIFEST=queued_only_five_cutmixema_folds.txt

cat > "$MANIFEST" <<EOF
Submitted: $(date --iso-8601=seconds)

Existing fold-1 job:
  fold 1 chunk 1: $CURRENT_FOLD1_JOB

Fold-0 resume smoke test:
  $SMOKE_JOB

Lane A:
  fold 0 chunk 1: $F0_C1
  fold 0 chunk 2: $F0_C2
  fold 2 chunk 1: $F2_C1
  fold 2 chunk 2: $F2_C2
  fold 4 chunk 1: $F4_C1
  fold 4 chunk 2: $F4_C2

Lane B:
  fold 1 chunk 1: $CURRENT_FOLD1_JOB
  fold 1 chunk 2: $F1_C2
  fold 3 chunk 1: $F3_C1
  fold 3 chunk 2: $F3_C2
EOF

echo
echo "======================================================================"
echo "QUEUED ONLY THE FIVE BASE FOLDS"
echo "======================================================================"
echo
echo "Maximum simultaneous GPU jobs: 2"
echo
echo "Lane A:"
echo "  fold-0 smoke test -> fold 0 -> fold 2 -> fold 4"
echo
echo "Lane B:"
echo "  existing fold 1 -> fold 3"
echo
echo "No evaluations or context fine-tuning jobs were submitted."
echo
echo "Manifest:"
echo "  $ROOT/$MANIFEST"
echo

squeue -u "$USER" \
    -o "%.18i %.32j %.2t %.12M %.45R"
