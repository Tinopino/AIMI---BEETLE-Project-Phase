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
#SBATCH --job-name=eval-ctx1024
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
umask 002

ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase

FOLD="${1:?Usage: sbatch run_context1024_eval_fold.slurm <fold>}"

cd "$ROOT"
mkdir -p logs

source /scratch/$USER/virtual_environments/AIMI-BEETLE/bin/activate

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

export nnUNet_raw=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_raw

export nnUNet_preprocessed=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_preprocessed

export nnUNet_results=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results

export MODEL_BASE_PATH="$nnUNet_results/Dataset301_BEETLE/nnUNetTrainer_CutMixStainEMA_Context1024FT100__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d_context1024"

export CSV_PATH="/home/tijnveldwijk/fold${FOLD}_validation_inference_inputs.csv"

export EVAL_FOLD="$FOLD"

export CHECKPOINT_NAME=checkpoint_best.pth

export CHECKPOINT_TAG=cutmixema1000_context1024_ft100_best_mirror_visual

export SAVE_VISUALS=1

export USE_MIRRORING=1

export MODEL_PATCH_SIZE=1024

export VIS_OUT_DIR="$nnUNet_results/../validation_visuals/cv_cutmixema1000_context1024_ft100/fold_${FOLD}/${CHECKPOINT_TAG}"

CHECKPOINT="$MODEL_BASE_PATH/fold_${FOLD}/$CHECKPOINT_NAME"

test -s "$CHECKPOINT" || {
    echo "ERROR: missing context-model checkpoint:" >&2
    echo "  $CHECKPOINT" >&2
    exit 30
}

test -s "$CSV_PATH" || {
    echo "ERROR: missing validation CSV:" >&2
    echo "  $CSV_PATH" >&2
    exit 31
}

echo "======================================================================"
echo "BEETLE MIRRORED FULL-VALIDATION CONTEXT EVALUATION"
echo "======================================================================"
echo "Started:          $(date)"
echo "Node:             $(hostname)"
echo "SLURM_JOB_ID:     ${SLURM_JOB_ID:-none}"
echo "Fold:             $FOLD"
echo "MODEL_BASE_PATH:  $MODEL_BASE_PATH"
echo "CHECKPOINT:       $CHECKPOINT"
echo "CSV_PATH:         $CSV_PATH"
echo "CHECKPOINT_TAG:   $CHECKPOINT_TAG"
echo "MODEL_PATCH_SIZE: $MODEL_PATCH_SIZE"
echo "USE_MIRRORING:    $USE_MIRRORING"
echo "VIS_OUT_DIR:      $VIS_OUT_DIR"
echo "======================================================================"

python -u visual_analysis.py

RESULT_DIR="$MODEL_BASE_PATH/fold_${FOLD}"

RESULT_FILE="$(
    find "$RESULT_DIR" \
        -maxdepth 1 \
        -type f \
        \( \
            -name "fold${FOLD}_${CHECKPOINT_TAG}*confusion_matrix*.csv" \
            -o \
            -name "fold${FOLD}_${CHECKPOINT_TAG}*full_validation*.json" \
        \) \
        | head -n 1
)"

if [[ -z "$RESULT_FILE" || ! -s "$RESULT_FILE" ]]; then
    echo "ERROR: evaluation completed without a result file." >&2
    echo "Expected a matching CSV or JSON file in:" >&2
    echo "  $RESULT_DIR" >&2
    exit 32
fi

echo
echo "Verified evaluation result:"
echo "  $RESULT_FILE"

echo
echo "Finished: $(date)"
