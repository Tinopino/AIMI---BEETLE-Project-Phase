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
#SBATCH --signal=B:TERM@60
#SBATCH --job-name=beetle_cema
#SBATCH --output=logs/cutmixema1000-%x-%j.out
#SBATCH --error=logs/cutmixema1000-%x-%j.err

set -euo pipefail
umask 002

ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase

FOLD="${1:?Usage: sbatch run_cv_cutmixema1000_chunk.slurm <fold> <fresh|resume> <chunk_tag>}"
MODE="${2:?Usage: sbatch run_cv_cutmixema1000_chunk.slurm <fold> <fresh|resume> <chunk_tag>}"
CHUNK_TAG="${3:?Usage: sbatch run_cv_cutmixema1000_chunk.slurm <fold> <fresh|resume> <chunk_tag>}"

TRAINER=nnUNetTrainer_CutMixStainEMA

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

mkdir -p "$RESULT_DIR"

write_checkpoint_manifest() {
    local LABEL="$1"
    local OUT="$RESULT_DIR/checkpoint_manifest_${LABEL}.txt"

    {
        echo "timestamp=$(date --iso-8601=seconds)"
        echo "fold=$FOLD"
        echo "mode=$MODE"
        echo "chunk_tag=$CHUNK_TAG"
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
echo "BEETLE 1000-epoch CutMix + stain jitter + EMA chunk"
echo "Started:      $(date)"
echo "Node:         $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "Fold:         $FOLD"
echo "Mode:         $MODE"
echo "Chunk tag:    $CHUNK_TAG"
echo "Trainer:      $TRAINER"
echo "Result dir:   $RESULT_DIR"
echo "======================================================================"

if [[ -s "$RESULT_DIR/checkpoint_final.pth" ]]; then
    echo "Fold $FOLD already completed 1000 epochs."
    write_checkpoint_manifest "${CHUNK_TAG}_already_complete"
    exit 0
fi

case "$MODE" in
    fresh)
        if [[ -s "$RESULT_DIR/checkpoint_latest.pth" ]]; then
            echo "ERROR: fresh run requested but checkpoint_latest.pth already exists:" >&2
            echo "  $RESULT_DIR/checkpoint_latest.pth" >&2
            exit 11
        fi
        TRAIN_CMD=(
            python -u
            nnUNet_pathology/nnunetv2/run/run_training_pathology.py
            301 "$FOLD" "$TRAINER"
        )
        ;;
    resume)
        if [[ ! -s "$RESULT_DIR/checkpoint_latest.pth" ]]; then
            echo "ERROR: resume requested but checkpoint_latest.pth is missing:" >&2
            echo "  $RESULT_DIR/checkpoint_latest.pth" >&2
            echo "A previous chunk may have failed before saving a resumable checkpoint." >&2
            exit 12
        fi
        TRAIN_CMD=(
            python -u
            nnUNet_pathology/nnunetv2/run/run_training_pathology.py
            301 "$FOLD" "$TRAINER" -c
        )
        ;;
    *)
        echo "ERROR: mode must be fresh or resume" >&2
        exit 13
        ;;
esac

set +e
timeout \
  --signal=TERM \
  --kill-after=60s \
  42600s \
  "${TRAIN_CMD[@]}"

RC=$?
set -e

write_checkpoint_manifest "${CHUNK_TAG}_finished"

if [[ -s "$RESULT_DIR/checkpoint_final.pth" ]]; then
    echo "Fold $FOLD completed successfully."
    exit 0
fi

echo
echo "Fold $FOLD has not yet reached checkpoint_final.pth."
echo "Chunk exit code: $RC"
echo "A resume chunk is expected to continue from checkpoint_latest.pth."
echo

# A non-zero status is intentional when the wall-time chunk ends before epoch 1000.
exit 99
