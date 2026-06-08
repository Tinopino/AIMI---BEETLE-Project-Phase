#!/usr/bin/env bash
set -euo pipefail
umask 002

###############################################################################
# PATHS
###############################################################################
ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase

NNUNET_RESULTS=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results

BASE_MODEL_DIR="$NNUNET_RESULTS/Dataset301_BEETLE/nnUNetTrainer_CutMixStainEMA__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"

BASE_TRAINER_FILE="$ROOT/nnUNet_pathology/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py"

cd "$ROOT"
mkdir -p logs

###############################################################################
# 1. PRE-FLIGHT CHECKS
###############################################################################
echo "======================================================================"
echo "PRE-FLIGHT CHECKS"
echo "======================================================================"

###############################################################################
# Verify that the fold-0 smoke test passed.
###############################################################################
LATEST_SMOKE_OUT="$(
    ls -t logs/test-f0-resume-*.out 2>/dev/null |
    head -n 1
)"

if [[ -z "$LATEST_SMOKE_OUT" ]]; then
    echo "ERROR: could not find a fold-0 smoke-test output file." >&2
    exit 10
fi

if ! grep -q "Fold-0 resume smoke test completed successfully" \
    "$LATEST_SMOKE_OUT"; then
    echo "ERROR: latest fold-0 resume smoke test did not pass:" >&2
    echo "  $LATEST_SMOKE_OUT" >&2
    echo >&2
    tail -n 120 "$LATEST_SMOKE_OUT" >&2
    exit 11
fi

echo "Verified fold-0 resume smoke test:"
echo "  $LATEST_SMOKE_OUT"

###############################################################################
# Verify that the base trainer patch is present.
###############################################################################
python - <<'PY'
from pathlib import Path

path = Path(
    "nnUNet_pathology/nnunetv2/training/"
    "nnUNetTrainer/nnUNetTrainer.py"
)

text = path.read_text()

required = """        else:
            checkpoint = filename_or_checkpoint
"""

if required not in text:
    raise RuntimeError(
        "Base trainer dictionary-checkpoint fix is missing. "
        "Do not queue training."
    )

print("Verified dictionary checkpoint support in nnUNetTrainer.load_checkpoint()")
PY

###############################################################################
# Verify fold-0 checkpoint.
###############################################################################
FOLD0_LATEST="$BASE_MODEL_DIR/fold_0/checkpoint_latest.pth"

test -s "$FOLD0_LATEST" || {
    echo "ERROR: fold-0 checkpoint_latest.pth is missing:" >&2
    echo "  $FOLD0_LATEST" >&2
    exit 12
}

echo
echo "Fold-0 checkpoint:"
ls -lh "$FOLD0_LATEST"

###############################################################################
# Abort if another base-CV queue is already active.
###############################################################################
if squeue -h -u "$USER" -o "%j" |
    grep -Eq '^basecv-f[0-4]-c[12]$'; then
    echo "ERROR: a base five-fold queue already appears to be active." >&2
    echo "Inspect with:" >&2
    echo "  squeue -u \$USER" >&2
    exit 13
fi

###############################################################################
# 2. CLEAN EMPTY FAILED OUTPUT FOLDERS FOR FOLDS 1-4
#
# Fold 0 is deliberately preserved.
#
# Refuse to delete folders containing real checkpoints.
###############################################################################
for FOLD in 1 2 3 4; do
    DIR="$BASE_MODEL_DIR/fold_${FOLD}"

    if [[ -d "$DIR" ]]; then
        if find "$DIR" \
            -maxdepth 1 \
            -type f \
            -name 'checkpoint*.pth' \
            -print -quit |
            grep -q .; then

            echo "ERROR: fold $FOLD unexpectedly contains checkpoint files:" >&2

            find "$DIR" \
                -maxdepth 1 \
                -type f \
                -name 'checkpoint*.pth' \
                -print >&2

            echo >&2
            echo "Inspect this folder manually before queueing." >&2
            exit 14
        fi

        echo "Removing empty failed-run folder:"
        echo "  $DIR"

        rm -rf "$DIR"
    fi
done

###############################################################################
# 3. CREATE A KNOWN-GOOD RESUME WRAPPER
#
# The custom CutMix+EMA trainer supports loading from a filename string.
# The base trainer patch now also supports the dictionary passed internally.
###############################################################################
cat > run_training_pathology_resume.py <<'PY'
#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

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

    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Resume checkpoint does not exist: {checkpoint}"
        )

    trainer = get_trainer_from_args(
        args.dataset_name_or_id,
        "2d",
        args.fold,
        args.trainer,
        args.planner,
    )

    print(f"Loading checkpoint: {checkpoint}", flush=True)

    trainer.load_checkpoint(str(checkpoint))

    print("Checkpoint loaded successfully", flush=True)
    print(f"Resuming from epoch: {trainer.current_epoch}", flush=True)
    print(f"Best EMA: {trainer._best_ema}", flush=True)

    trainer.run_training()


if __name__ == "__main__":
    main()
PY

chmod +x run_training_pathology_resume.py

###############################################################################
# 4. VALIDATE IMPORTS AND PYTHON SYNTAX
###############################################################################
source /scratch/$USER/virtual_environments/AIMI-BEETLE/bin/activate

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

export nnUNet_raw=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_raw
export nnUNet_preprocessed=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_preprocessed
export nnUNet_results=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results

python -m py_compile \
    run_training_pathology_resume.py \
    nnUNet_pathology/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py \
    nnUNet_pathology/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_CutMixStainEMA.py

python - <<'PY'
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_CutMixStainEMA import (
    nnUNetTrainer_CutMixStainEMA,
)

assert hasattr(nnUNetTrainer_CutMixStainEMA, "load_checkpoint")

print("Verified CutMix+EMA trainer import")
print("Verified load_checkpoint() support")
PY

###############################################################################
# 5. CREATE ROBUST 12-HOUR TRAINING CHUNK LAUNCHER
#
# Behavior:
# - fold 0 resumes automatically;
# - folds 1-4 start fresh during their first chunk;
# - second chunks resume automatically;
# - valid GNU timeout syntax is used: 42600 seconds = 11h50m;
# - only a natural completion or timeout is accepted;
# - checkpoint manifests are saved after every chunk;
# - checkpoint_latest, checkpoint_best, class-best and checkpoint_final remain
#   managed by the existing trainer.
###############################################################################
cat > run_basecv_chunk.slurm <<'SLURM'
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
#SBATCH --job-name=basecv
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
umask 002

ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase

FOLD="${1:?Usage: sbatch run_basecv_chunk.slurm <fold> <intermediate|final>}"
ROLE="${2:?Usage: sbatch run_basecv_chunk.slurm <fold> <intermediate|final>}"

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

write_manifest() {
    local LABEL="$1"

    local OUT="$RESULT_DIR/checkpoint_manifest_${LABEL}_${SLURM_JOB_ID:-manual}.txt"

    {
        echo "timestamp=$(date --iso-8601=seconds)"
        echo "fold=$FOLD"
        echo "role=$ROLE"
        echo "result_dir=$RESULT_DIR"
        echo

        find "$RESULT_DIR" \
            -maxdepth 1 \
            -type f \
            \( \
                -name 'checkpoint*.pth' \
                -o -name 'class_metrics.csv' \
                -o -name 'class_metrics.jsonl' \
                -o -name 'training_log_*.txt' \
            \) \
            -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %f\n' |
            sort
    } > "$OUT"

    echo "Wrote checkpoint manifest:"
    echo "  $OUT"
}

echo "======================================================================"
echo "BEETLE CUTMIX + STAIN JITTER + EMA BASE-CV CHUNK"
echo "======================================================================"
echo "Started:      $(date)"
echo "Node:         $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "Fold:         $FOLD"
echo "Role:         $ROLE"
echo "Trainer:      $TRAINER"
echo "Result dir:   $RESULT_DIR"
echo "======================================================================"

if [[ -s "$FINAL" ]]; then
    echo "Fold $FOLD already completed 1000 epochs."
    write_manifest "already_complete"
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
    echo "No checkpoint_latest.pth detected."
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

write_manifest "finished"

###############################################################################
# Expected outcomes:
#   0   natural completion
#   124 expected timeout after 11h50m
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
    echo "The next queued chunk will resume from checkpoint_latest.pth."
    exit 0
fi

if [[ "$ROLE" == "final" ]]; then
    echo "ERROR: fold $FOLD remains incomplete after its allocated chunks." >&2
    echo "Queue one additional resume chunk before evaluation." >&2
    exit 32
fi

echo "ERROR: role must be intermediate or final." >&2
exit 33
SLURM

chmod +x run_basecv_chunk.slurm

###############################################################################
# 6. QUEUE ONLY THE FIVE BASE FOLDS
#
# Maximum simultaneous GPU jobs: 2.
#
# Lane A:
#   fold 0 chunk 1 resume
#   -> fold 0 chunk 2 resume
#   -> fold 2 chunk 1 fresh
#   -> fold 2 chunk 2 resume
#   -> fold 4 chunk 1 fresh
#   -> fold 4 chunk 2 resume
#
# Lane B:
#   fold 1 chunk 1 fresh
#   -> fold 1 chunk 2 resume
#   -> fold 3 chunk 1 fresh
#   -> fold 3 chunk 2 resume
###############################################################################
F0_C1=$(
    sbatch \
        --parsable \
        --job-name=basecv-f0-c1 \
        run_basecv_chunk.slurm \
        0 intermediate
)

F0_C2=$(
    sbatch \
        --parsable \
        --dependency=afterok:${F0_C1} \
        --job-name=basecv-f0-c2 \
        run_basecv_chunk.slurm \
        0 final
)

F1_C1=$(
    sbatch \
        --parsable \
        --job-name=basecv-f1-c1 \
        run_basecv_chunk.slurm \
        1 intermediate
)

F1_C2=$(
    sbatch \
        --parsable \
        --dependency=afterok:${F1_C1} \
        --job-name=basecv-f1-c2 \
        run_basecv_chunk.slurm \
        1 final
)

F2_C1=$(
    sbatch \
        --parsable \
        --dependency=afterok:${F0_C2} \
        --job-name=basecv-f2-c1 \
        run_basecv_chunk.slurm \
        2 intermediate
)

F2_C2=$(
    sbatch \
        --parsable \
        --dependency=afterok:${F2_C1} \
        --job-name=basecv-f2-c2 \
        run_basecv_chunk.slurm \
        2 final
)

F3_C1=$(
    sbatch \
        --parsable \
        --dependency=afterok:${F1_C2} \
        --job-name=basecv-f3-c1 \
        run_basecv_chunk.slurm \
        3 intermediate
)

F3_C2=$(
    sbatch \
        --parsable \
        --dependency=afterok:${F3_C1} \
        --job-name=basecv-f3-c2 \
        run_basecv_chunk.slurm \
        3 final
)

F4_C1=$(
    sbatch \
        --parsable \
        --dependency=afterok:${F2_C2} \
        --job-name=basecv-f4-c1 \
        run_basecv_chunk.slurm \
        4 intermediate
)

F4_C2=$(
    sbatch \
        --parsable \
        --dependency=afterok:${F4_C1} \
        --job-name=basecv-f4-c2 \
        run_basecv_chunk.slurm \
        4 final
)

###############################################################################
# 7. SAVE A READABLE JOB MANIFEST
###############################################################################
MANIFEST=queued_only_five_base_folds.txt

cat > "$MANIFEST" <<EOF
Submitted: $(date --iso-8601=seconds)

Maximum simultaneous GPU jobs: 2

Lane A:
  fold 0 chunk 1: $F0_C1
  fold 0 chunk 2: $F0_C2
  fold 2 chunk 1: $F2_C1
  fold 2 chunk 2: $F2_C2
  fold 4 chunk 1: $F4_C1
  fold 4 chunk 2: $F4_C2

Lane B:
  fold 1 chunk 1: $F1_C1
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
echo "  fold 0 -> fold 2 -> fold 4"
echo
echo "Lane B:"
echo "  fold 1 -> fold 3"
echo
echo "No evaluations, context fine-tunes, or aggregations were queued."
echo
echo "Manifest:"
echo "  $ROOT/$MANIFEST"
echo

squeue -u "$USER" \
    -o "%.18i %.32j %.2t %.12M %.45R"
