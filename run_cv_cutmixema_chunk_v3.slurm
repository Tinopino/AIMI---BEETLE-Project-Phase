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
