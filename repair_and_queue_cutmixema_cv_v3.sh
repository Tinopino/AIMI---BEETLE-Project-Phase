#!/usr/bin/env bash
set -euo pipefail
umask 002

###############################################################################
# PATHS
###############################################################################
ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase

NNUNET_RESULTS=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results

BASE_MODEL_DIR="$NNUNET_RESULTS/Dataset301_BEETLE/nnUNetTrainer_CutMixStainEMA__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"

CONTEXT_MODEL_DIR="$NNUNET_RESULTS/Dataset301_BEETLE/nnUNetTrainer_CutMixStainEMA_Context1024FT100__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d_context1024"

BEETLE_TRAINER="$ROOT/nnUNet_pathology/nnunetv2/training/nnUNetTrainer/nnUnetTrainerBEETLE.py"

CONTEXT_TRAINER="$ROOT/nnUNet_pathology/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_CutMixStainEMA_Context1024FT100.py"

cd "$ROOT"
mkdir -p logs cv_summaries/cutmixema1000_context1024

###############################################################################
# 0. PRE-FLIGHT: FOLD 0 MUST REMAIN RESUMABLE
###############################################################################
echo "======================================================================"
echo "PRE-FLIGHT CHECKS"
echo "======================================================================"

test -s "$BASE_MODEL_DIR/fold_0/checkpoint_latest.pth" || {
    echo "ERROR: fold-0 checkpoint_latest.pth is missing:" >&2
    echo "  $BASE_MODEL_DIR/fold_0/checkpoint_latest.pth" >&2
    exit 10
}

echo "Fold-0 checkpoint found:"
ls -lh "$BASE_MODEL_DIR/fold_0/checkpoint_latest.pth"

###############################################################################
# 1. REMOVE THE INVALID CONTEXT CLASS FROM nnUnetTrainerBEETLE.py
#
# It was appended to the wrong module and caused:
# NameError: nnUNetTrainer_CutMixStainEMA is not defined
###############################################################################
python - <<'PY'
from pathlib import Path

path = Path(
    "nnUNet_pathology/nnunetv2/training/"
    "nnUNetTrainer/nnUnetTrainerBEETLE.py"
)

text = path.read_text()

marker = "\nclass nnUNetTrainer_CutMixStainEMA_Context1024FT100"

if marker in text:
    prefix, tail = text.split(marker, 1)

    # The invalid class was appended at the end of the module.
    # Refuse to truncate if unrelated classes appear after it.
    if "\nclass " in tail:
        raise RuntimeError(
            "Unexpected additional class definitions after the invalid "
            "context trainer. Inspect nnUnetTrainerBEETLE.py manually."
        )

    path.write_text(prefix.rstrip() + "\n")
    print("Removed invalid appended context class from:", path)
else:
    print("Invalid appended context class was already absent")
PY

###############################################################################
# 2. CREATE THE CONTEXT TRAINER IN ITS OWN MODULE
#
# This imports the actual CutMix + EMA base trainer correctly.
###############################################################################
cat > "$CONTEXT_TRAINER" <<'PY'
import torch

from .nnUNetTrainer_CutMixStainEMA import nnUNetTrainer_CutMixStainEMA


class nnUNetTrainer_CutMixStainEMA_Context1024FT100(
    nnUNetTrainer_CutMixStainEMA
):
    """
    Fine-tunes a completed CutMix + stain-jitter + EMA checkpoint using
    larger 1024x1024 WholeSlideData patches.

    Inherited behavior:
    - alpha-weighted Dice + focal loss;
    - CutMix;
    - stain jitter;
    - EMA;
    - checkpoint_latest.pth;
    - checkpoint_best.pth;
    - class-specific best checkpoints;
    - checkpoint_final.pth;
    - per-class metric logging.

    Changed:
    - patch size: 1024x1024;
    - WSD batch size: 2;
    - initial LR: 0.0005;
    - additional epochs: 100.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans,
            configuration,
            fold,
            dataset_json,
            unpack_dataset,
            device,
        )

        expected_patch_size = [1024, 1024]
        actual_patch_size = list(self.configuration_manager.patch_size)

        if actual_patch_size != expected_patch_size:
            raise RuntimeError(
                "Context trainer requires patch_size="
                f"{expected_patch_size}, received {actual_patch_size}. "
                "Use configuration='2d_context1024'."
            )

        self.wsd_batch_size_override = 2
        self.initial_lr = 5e-4
        self.num_epochs = 100
        self.save_every = 1

        self.print_to_log_file(
            "Using CutMix + stain jitter + EMA context fine-tuning: "
            "patch_size=[1024, 1024], WSD batch_size=2, "
            "initial_lr=0.0005, num_epochs=100",
            also_print_to_console=True,
        )
PY

###############################################################################
# 3. CREATE A REAL RESUME WRAPPER
#
# run_training_pathology.py has no -c flag. This wrapper instantiates the
# trainer and calls its native load_checkpoint() method before training.
###############################################################################
cat > run_training_pathology_resume.py <<'PY'
#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from nnunetv2.run.run_training_pathology import get_trainer_from_args


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume pathology nnU-Net training from checkpoint_latest."
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

    nnunet_trainer = get_trainer_from_args(
        args.dataset_name_or_id,
        "2d",
        args.fold,
        args.trainer,
        args.planner,
    )

    print(f"Loading checkpoint: {checkpoint}", flush=True)

    nnunet_trainer.load_checkpoint(str(checkpoint))

    print("Checkpoint loaded successfully", flush=True)
    print("Continuing training", flush=True)

    nnunet_trainer.run_training()


if __name__ == "__main__":
    main()
PY

chmod +x run_training_pathology_resume.py

###############################################################################
# 4. VALIDATE PYTHON IMPORTS BEFORE QUEUEING ANYTHING
###############################################################################
source /scratch/$USER/virtual_environments/AIMI-BEETLE/bin/activate

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

export nnUNet_raw=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_raw
export nnUNet_preprocessed=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_preprocessed
export nnUNet_results=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results

python -m py_compile \
    "$BEETLE_TRAINER" \
    "$CONTEXT_TRAINER" \
    run_training_pathology_resume.py \
    visual_analysis.py \
    aggregate_cv_results_v2.py

python - <<'PY'
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_CutMixStainEMA import (
    nnUNetTrainer_CutMixStainEMA,
)

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_CutMixStainEMA_Context1024FT100 import (
    nnUNetTrainer_CutMixStainEMA_Context1024FT100,
)

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

assert issubclass(
    nnUNetTrainer_CutMixStainEMA_Context1024FT100,
    nnUNetTrainer_CutMixStainEMA,
)

assert hasattr(nnUNetTrainer, "load_checkpoint")

print("Verified base CutMix + EMA trainer import")
print("Verified context trainer import")
print("Verified native load_checkpoint() support")
PY

###############################################################################
# 5. CLEAR ONLY EMPTY FAILED FOLD-1 TO FOLD-4 OUTPUT DIRECTORIES
#
# Fold 0 is deliberately preserved.
###############################################################################
for FOLD in 1 2 3 4; do
    DIR="$BASE_MODEL_DIR/fold_${FOLD}"

    if [[ -d "$DIR" ]]; then
        if find "$DIR" -maxdepth 1 -type f -name 'checkpoint*.pth' \
            -print -quit | grep -q .; then
            echo "ERROR: fold $FOLD unexpectedly contains checkpoints:" >&2
            find "$DIR" -maxdepth 1 -type f -name 'checkpoint*.pth' -print >&2
            exit 20
        fi

        echo "Removing empty failed output folder:"
        echo "  $DIR"
        rm -rf "$DIR"
    fi
done

###############################################################################
# 6. CREATE CORRECTED 12-HOUR CUTMIX + EMA CHUNK LAUNCHER
#
# - Uses native load_checkpoint() wrapper for resume.
# - Uses valid GNU timeout syntax: 42600 seconds = 11h50m.
# - Rejects unexpected non-zero exit codes.
# - Preserves latest, general-best, class-best and final checkpoints.
###############################################################################
cat > run_cv_cutmixema_chunk_v3.slurm <<'SLURM'
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
#SBATCH --job-name=cutmixema-v3
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
umask 002

ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase

FOLD="${1:?Usage: sbatch run_cv_cutmixema_chunk_v3.slurm <fold> <auto|resume> <intermediate|final>}"
MODE="${2:?Usage: sbatch run_cv_cutmixema_chunk_v3.slurm <fold> <auto|resume> <intermediate|final>}"
ROLE="${3:?Usage: sbatch run_cv_cutmixema_chunk_v3.slurm <fold> <auto|resume> <intermediate|final>}"

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
        echo "mode=$MODE"
        echo "role=$ROLE"
        echo "result_dir=$RESULT_DIR"
        echo

        find "$RESULT_DIR" -maxdepth 1 -type f \
            \( -name 'checkpoint*.pth' \
               -o -name 'class_metrics.csv' \
               -o -name 'class_metrics.jsonl' \
               -o -name 'training_log_*.txt' \
            \) \
            -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %f\n' \
            | sort
    } > "$OUT"

    echo "Wrote checkpoint manifest: $OUT"
}

echo "======================================================================"
echo "BEETLE CutMix + stain jitter + EMA training chunk"
echo "Started:      $(date)"
echo "Node:         $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "Fold:         $FOLD"
echo "Mode:         $MODE"
echo "Role:         $ROLE"
echo "Trainer:      $TRAINER"
echo "Result dir:   $RESULT_DIR"
echo "======================================================================"

if [[ -s "$FINAL" ]]; then
    echo "Fold $FOLD already has checkpoint_final.pth."
    write_manifest "already_complete"
    exit 0
fi

case "$MODE" in
    auto)
        if [[ -s "$LATEST" ]]; then
            echo "Existing checkpoint_latest.pth detected. Resuming fold $FOLD."

            TRAIN_CMD=(
                python -u
                run_training_pathology_resume.py
                301 "$FOLD" "$TRAINER"
                --planner "$PLANNER"
                --checkpoint "$LATEST"
            )
        else
            echo "No checkpoint detected. Starting fold $FOLD fresh."

            TRAIN_CMD=(
                python -u
                nnUNet_pathology/nnunetv2/run/run_training_pathology.py
                301 "$FOLD" "$TRAINER"
                --planner "$PLANNER"
            )
        fi
        ;;

    resume)
        test -s "$LATEST" || {
            echo "ERROR: resume requested but checkpoint_latest.pth is missing:" >&2
            echo "  $LATEST" >&2
            exit 30
        }

        TRAIN_CMD=(
            python -u
            run_training_pathology_resume.py
            301 "$FOLD" "$TRAINER"
            --planner "$PLANNER"
            --checkpoint "$LATEST"
        )
        ;;

    *)
        echo "ERROR: mode must be auto or resume" >&2
        exit 31
        ;;
esac

case "$ROLE" in
    intermediate|final)
        ;;
    *)
        echo "ERROR: role must be intermediate or final" >&2
        exit 32
        ;;
esac

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
# Only a natural completion or an expected timeout is acceptable.
###############################################################################
if [[ "$RC" -ne 0 && "$RC" -ne 124 ]]; then
    echo "ERROR: training command failed unexpectedly with exit code $RC." >&2
    exit 33
fi

if [[ -s "$FINAL" ]]; then
    echo "SUCCESS: fold $FOLD completed 1000 epochs."
    exit 0
fi

if [[ ! -s "$LATEST" ]]; then
    echo "ERROR: training chunk ended without checkpoint_latest.pth:" >&2
    echo "  $LATEST" >&2
    exit 34
fi

if [[ "$ROLE" == "intermediate" ]]; then
    echo "SUCCESS: fold $FOLD remains resumable after intermediate chunk."
    echo "The queued final chunk will continue from checkpoint_latest.pth."
    exit 0
fi

echo "ERROR: fold $FOLD is incomplete after its final allocated chunk." >&2
echo "Queue one additional resume chunk before evaluation." >&2
exit 35
SLURM

chmod +x run_cv_cutmixema_chunk_v3.slurm

###############################################################################
# 7. VERIFY THE DOWNSTREAM LAUNCHERS EXIST
###############################################################################
test -x run_cv_context_ft100_v2.slurm || {
    echo "ERROR: missing run_cv_context_ft100_v2.slurm" >&2
    exit 40
}

test -x run_cv_full_eval_v2.slurm || {
    echo "ERROR: missing run_cv_full_eval_v2.slurm" >&2
    exit 41
}

test -x run_cv_aggregate_v2.slurm || {
    echo "ERROR: missing run_cv_aggregate_v2.slurm" >&2
    exit 42
}

###############################################################################
# 8. ENSURE NO DUPLICATE V3 PIPELINE IS ALREADY QUEUED
###############################################################################
if squeue -h -u "$USER" -o "%j" |
    grep -Eq '^(cutmixema-v3-|eval-cema-v3-|ctxft-v3-|eval-ctx-v3-|agg-cema-v3|agg-final-v3)'; then
    echo "ERROR: a v3 pipeline already appears to be queued." >&2
    echo "Inspect with: squeue -u \$USER" >&2
    exit 43
fi

join_colon() {
    local IFS=:
    echo "$*"
}

###############################################################################
# 9. STAGE 1: 1000-EPOCH BASE TRAINING
#
# TWO EXPLICIT GPU LANES:
#
# Lane A:
#   fold 0 resume chunk 1
#   -> fold 0 resume chunk 2
#   -> fold 2 chunks 1 and 2
#   -> fold 4 chunks 1 and 2
#
# Lane B:
#   fold 1 chunks 1 and 2
#   -> fold 3 chunks 1 and 2
###############################################################################
F0_R1=$(
    sbatch --parsable \
        --job-name=cutmixema-v3-f0-r1 \
        run_cv_cutmixema_chunk_v3.slurm \
        0 resume intermediate
)

F0_R2=$(
    sbatch --parsable \
        --dependency=afterok:${F0_R1} \
        --job-name=cutmixema-v3-f0-r2 \
        run_cv_cutmixema_chunk_v3.slurm \
        0 resume final
)

F1_C1=$(
    sbatch --parsable \
        --job-name=cutmixema-v3-f1-c1 \
        run_cv_cutmixema_chunk_v3.slurm \
        1 auto intermediate
)

F1_C2=$(
    sbatch --parsable \
        --dependency=afterok:${F1_C1} \
        --job-name=cutmixema-v3-f1-c2 \
        run_cv_cutmixema_chunk_v3.slurm \
        1 resume final
)

F2_C1=$(
    sbatch --parsable \
        --dependency=afterok:${F0_R2} \
        --job-name=cutmixema-v3-f2-c1 \
        run_cv_cutmixema_chunk_v3.slurm \
        2 auto intermediate
)

F2_C2=$(
    sbatch --parsable \
        --dependency=afterok:${F2_C1} \
        --job-name=cutmixema-v3-f2-c2 \
        run_cv_cutmixema_chunk_v3.slurm \
        2 resume final
)

F3_C1=$(
    sbatch --parsable \
        --dependency=afterok:${F1_C2} \
        --job-name=cutmixema-v3-f3-c1 \
        run_cv_cutmixema_chunk_v3.slurm \
        3 auto intermediate
)

F3_C2=$(
    sbatch --parsable \
        --dependency=afterok:${F3_C1} \
        --job-name=cutmixema-v3-f3-c2 \
        run_cv_cutmixema_chunk_v3.slurm \
        3 resume final
)

F4_C1=$(
    sbatch --parsable \
        --dependency=afterok:${F2_C2} \
        --job-name=cutmixema-v3-f4-c1 \
        run_cv_cutmixema_chunk_v3.slurm \
        4 auto intermediate
)

F4_C2=$(
    sbatch --parsable \
        --dependency=afterok:${F4_C1} \
        --job-name=cutmixema-v3-f4-c2 \
        run_cv_cutmixema_chunk_v3.slurm \
        4 resume final
)

ALL_BASE_FINALS=$(join_colon "$F0_R2" "$F1_C2" "$F2_C2" "$F3_C2" "$F4_C2")

###############################################################################
# 10. STAGE 2: FULL MIRRORED BASE EVALUATIONS
#
# Also capped at two GPUs.
###############################################################################
E0=$(
    sbatch --parsable \
        --dependency=afterok:${ALL_BASE_FINALS} \
        --job-name=eval-cema-v3-f0 \
        run_cv_full_eval_v2.slurm \
        base 0
)

E1=$(
    sbatch --parsable \
        --dependency=afterok:${ALL_BASE_FINALS} \
        --job-name=eval-cema-v3-f1 \
        run_cv_full_eval_v2.slurm \
        base 1
)

E2=$(
    sbatch --parsable \
        --dependency=afterok:${E0} \
        --job-name=eval-cema-v3-f2 \
        run_cv_full_eval_v2.slurm \
        base 2
)

E3=$(
    sbatch --parsable \
        --dependency=afterok:${E1} \
        --job-name=eval-cema-v3-f3 \
        run_cv_full_eval_v2.slurm \
        base 3
)

E4=$(
    sbatch --parsable \
        --dependency=afterok:${E2} \
        --job-name=eval-cema-v3-f4 \
        run_cv_full_eval_v2.slurm \
        base 4
)

ALL_BASE_EVALS=$(join_colon "$E0" "$E1" "$E2" "$E3" "$E4")

BASE_AGG=$(
    sbatch --parsable \
        --dependency=afterok:${ALL_BASE_EVALS} \
        --job-name=agg-cema-v3 \
        run_cv_aggregate_v2.slurm \
        base
)

###############################################################################
# 11. STAGE 3: 100-EPOCH 1024x1024 CONTEXT FINE-TUNING
#
# Capped at two GPUs.
###############################################################################
T0=$(
    sbatch --parsable \
        --dependency=afterok:${BASE_AGG} \
        --job-name=ctxft-v3-f0 \
        run_cv_context_ft100_v2.slurm \
        0
)

T1=$(
    sbatch --parsable \
        --dependency=afterok:${BASE_AGG} \
        --job-name=ctxft-v3-f1 \
        run_cv_context_ft100_v2.slurm \
        1
)

T2=$(
    sbatch --parsable \
        --dependency=afterok:${T0} \
        --job-name=ctxft-v3-f2 \
        run_cv_context_ft100_v2.slurm \
        2
)

T3=$(
    sbatch --parsable \
        --dependency=afterok:${T1} \
        --job-name=ctxft-v3-f3 \
        run_cv_context_ft100_v2.slurm \
        3
)

T4=$(
    sbatch --parsable \
        --dependency=afterok:${T2} \
        --job-name=ctxft-v3-f4 \
        run_cv_context_ft100_v2.slurm \
        4
)

ALL_CONTEXT_TRAINS=$(join_colon "$T0" "$T1" "$T2" "$T3" "$T4")

###############################################################################
# 12. STAGE 4: FULL MIRRORED CONTEXT EVALUATIONS
#
# Capped at two GPUs.
###############################################################################
CE0=$(
    sbatch --parsable \
        --dependency=afterok:${ALL_CONTEXT_TRAINS} \
        --job-name=eval-ctx-v3-f0 \
        run_cv_full_eval_v2.slurm \
        context 0
)

CE1=$(
    sbatch --parsable \
        --dependency=afterok:${ALL_CONTEXT_TRAINS} \
        --job-name=eval-ctx-v3-f1 \
        run_cv_full_eval_v2.slurm \
        context 1
)

CE2=$(
    sbatch --parsable \
        --dependency=afterok:${CE0} \
        --job-name=eval-ctx-v3-f2 \
        run_cv_full_eval_v2.slurm \
        context 2
)

CE3=$(
    sbatch --parsable \
        --dependency=afterok:${CE1} \
        --job-name=eval-ctx-v3-f3 \
        run_cv_full_eval_v2.slurm \
        context 3
)

CE4=$(
    sbatch --parsable \
        --dependency=afterok:${CE2} \
        --job-name=eval-ctx-v3-f4 \
        run_cv_full_eval_v2.slurm \
        context 4
)

ALL_CONTEXT_EVALS=$(join_colon "$CE0" "$CE1" "$CE2" "$CE3" "$CE4")

FINAL_AGG=$(
    sbatch --parsable \
        --dependency=afterok:${ALL_CONTEXT_EVALS} \
        --job-name=agg-final-v3 \
        run_cv_aggregate_v2.slurm \
        both
)

###############################################################################
# 13. WRITE SUBMISSION MANIFEST
###############################################################################
MANIFEST=queued_cutmixema_context_cv_v3.txt

cat > "$MANIFEST" <<EOF
Submitted: $(date --iso-8601=seconds)

Stage 1: CutMix + stain jitter + EMA, 1000 epochs
  fold 0 resume chunk 1: $F0_R1
  fold 0 resume chunk 2: $F0_R2
  fold 1 chunk 1:        $F1_C1
  fold 1 chunk 2:        $F1_C2
  fold 2 chunk 1:        $F2_C1
  fold 2 chunk 2:        $F2_C2
  fold 3 chunk 1:        $F3_C1
  fold 3 chunk 2:        $F3_C2
  fold 4 chunk 1:        $F4_C1
  fold 4 chunk 2:        $F4_C2

Stage 2: mirrored base evaluations
  fold 0: $E0
  fold 1: $E1
  fold 2: $E2
  fold 3: $E3
  fold 4: $E4
  aggregate: $BASE_AGG

Stage 3: context fine-tuning
  fold 0: $T0
  fold 1: $T1
  fold 2: $T2
  fold 3: $T3
  fold 4: $T4

Stage 4: mirrored context evaluations
  fold 0: $CE0
  fold 1: $CE1
  fold 2: $CE2
  fold 3: $CE3
  fold 4: $CE4

Stage 5: final aggregation
  $FINAL_AGG
EOF

echo
echo "======================================================================"
echo "CORRECTED V3 TWO-GPU PIPELINE QUEUED"
echo "======================================================================"
echo
echo "GPU lane A:"
echo "  fold 0 resume -> fold 2 -> fold 4"
echo
echo "GPU lane B:"
echo "  fold 1 -> fold 3"
echo
echo "Maximum simultaneous GPU jobs from this pipeline: 2"
echo
echo "Manifest:"
echo "  $ROOT/$MANIFEST"
echo

squeue -u "$USER" \
    -o "%.18i %.32j %.2t %.12M %.45R"
