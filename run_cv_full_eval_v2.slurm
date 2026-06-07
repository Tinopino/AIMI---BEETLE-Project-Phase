#!/usr/bin/env bash
#SBATCH --account=cseduimc037
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --nodelist=cn48
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=12:00:00
#SBATCH --job-name=eval-v2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
umask 002

ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase

STAGE="${1:?Usage: sbatch run_cv_full_eval_v2.slurm <base|context> <fold>}"
FOLD="${2:?Usage: sbatch run_cv_full_eval_v2.slurm <base|context> <fold>}"

cd "$ROOT"
mkdir -p logs

source /scratch/$USER/virtual_environments/AIMI-BEETLE/bin/activate

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

export nnUNet_raw=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_raw
export nnUNet_preprocessed=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_preprocessed
export nnUNet_results=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results

case "$STAGE" in
    base)
        MODEL_BASE_PATH="$nnUNet_results/Dataset301_BEETLE/nnUNetTrainer_CutMixStainEMA__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"
        CHECKPOINT_TAG="cutmixema1000_best_mirror_visual"
        MODEL_PATCH_SIZE=512
        VIS_OUT_DIR_BASE="$nnUNet_results/../validation_visuals/cv_cutmixema1000"
        ;;
    context)
        MODEL_BASE_PATH="$nnUNet_results/Dataset301_BEETLE/nnUNetTrainer_CutMixStainEMA_Context1024FT100__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d_context1024"
        CHECKPOINT_TAG="cutmixema1000_context1024_ft100_best_mirror_visual"
        MODEL_PATCH_SIZE=1024
        VIS_OUT_DIR_BASE="$nnUNet_results/../validation_visuals/cv_cutmixema1000_context1024_ft100"
        ;;
    *)
        echo "ERROR: stage must be base or context" >&2
        exit 50
        ;;
esac

CSV_PATH="/home/tijnveldwijk/fold${FOLD}_validation_inference_inputs.csv"
VIS_OUT_DIR="$VIS_OUT_DIR_BASE/fold_${FOLD}/$CHECKPOINT_TAG"

CHECKPOINT="$MODEL_BASE_PATH/fold_${FOLD}/checkpoint_best.pth"

test -s "$CSV_PATH" || {
    echo "ERROR: missing validation CSV:" >&2
    echo "  $CSV_PATH" >&2
    exit 51
}

test -s "$CHECKPOINT" || {
    echo "ERROR: missing checkpoint:" >&2
    echo "  $CHECKPOINT" >&2
    exit 52
}

export MODEL_BASE_PATH
export CSV_PATH
export EVAL_FOLD="$FOLD"
export CHECKPOINT_NAME=checkpoint_best.pth
export CHECKPOINT_TAG
export SAVE_VISUALS=1
export USE_MIRRORING=1
export MODEL_PATCH_SIZE
export VIS_OUT_DIR

echo "======================================================================"
echo "BEETLE full mirrored validation"
echo "Started:          $(date)"
echo "Node:             $(hostname)"
echo "SLURM_JOB_ID:     ${SLURM_JOB_ID:-none}"
echo "Stage:            $STAGE"
echo "Fold:             $FOLD"
echo "MODEL_BASE_PATH:  $MODEL_BASE_PATH"
echo "CSV_PATH:         $CSV_PATH"
echo "CHECKPOINT_TAG:   $CHECKPOINT_TAG"
echo "MODEL_PATCH_SIZE: $MODEL_PATCH_SIZE"
echo "USE_MIRRORING:    $USE_MIRRORING"
echo "VIS_OUT_DIR:      $VIS_OUT_DIR"
echo "======================================================================"

python -u visual_analysis.py

CM_CSV="$MODEL_BASE_PATH/fold_${FOLD}/fold${FOLD}_${CHECKPOINT_TAG}_confusion_matrix_rows_gt_cols_pred.csv"

test -s "$CM_CSV" || {
    echo "ERROR: evaluation returned without confusion matrix CSV:" >&2
    echo "  $CM_CSV" >&2
    exit 53
}

echo "Verified evaluation output:"
echo "  $CM_CSV"

echo "Finished: $(date)"
