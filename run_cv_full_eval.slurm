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
#SBATCH --job-name=beetle_eval
#SBATCH --output=logs/full-eval-%x-%j.out
#SBATCH --error=logs/full-eval-%x-%j.err

set -euo pipefail
umask 002

ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase

STAGE="${1:?Usage: sbatch run_cv_full_eval.slurm <base|context> <fold>}"
FOLD="${2:?Usage: sbatch run_cv_full_eval.slurm <base|context> <fold>}"

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
        exit 40
        ;;
esac

CSV_PATH="/home/tijnveldwijk/fold${FOLD}_validation_inference_inputs.csv"
VIS_OUT_DIR="$VIS_OUT_DIR_BASE/fold_${FOLD}/$CHECKPOINT_TAG"

test -s "$CSV_PATH" || {
    echo "ERROR: missing validation CSV: $CSV_PATH" >&2
    exit 41
}

test -s "$MODEL_BASE_PATH/fold_${FOLD}/checkpoint_best.pth" || {
    echo "ERROR: missing general-best checkpoint:" >&2
    echo "  $MODEL_BASE_PATH/fold_${FOLD}/checkpoint_best.pth" >&2
    exit 42
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
echo "BEETLE full mirrored validation analysis"
echo "Started:          $(date)"
echo "Node:             $(hostname)"
echo "SLURM_JOB_ID:     ${SLURM_JOB_ID:-none}"
echo "Stage:            $STAGE"
echo "Fold:             $FOLD"
echo "MODEL_BASE_PATH:  $MODEL_BASE_PATH"
echo "CSV_PATH:         $CSV_PATH"
echo "CHECKPOINT_TAG:   $CHECKPOINT_TAG"
echo "MODEL_PATCH_SIZE: $MODEL_PATCH_SIZE"
echo "VIS_OUT_DIR:      $VIS_OUT_DIR"
echo "USE_MIRRORING:    $USE_MIRRORING"
echo "======================================================================"

python -u visual_analysis.py

OUT_JSON="$MODEL_BASE_PATH/fold_${FOLD}/fold${FOLD}_${CHECKPOINT_TAG}_full_validation_dice_tiffslide_hybrid_visual_cm.json"

test -s "$OUT_JSON" || {
    echo "ERROR: evaluation returned without expected JSON:" >&2
    echo "  $OUT_JSON" >&2
    exit 43
}

echo "Verified evaluation JSON:"
echo "  $OUT_JSON"

echo "Finished: $(date)"
