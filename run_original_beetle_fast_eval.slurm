#!/bin/bash
#SBATCH --job-name=beetle_orig_eval
#SBATCH --account=cseduimc037
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=12:00:00
#SBATCH --nodelist=cn48
#SBATCH --output=logs/original-beetle-fast-eval-%j.out
#SBATCH --error=logs/original-beetle-fast-eval-%j.out

set -euo pipefail
umask 002

cd /home/tijnveldwijk/AIMI---BEETLE-Project-Phase
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

echo "============================================================"
echo "Original BEETLE fold-0 fast evaluation"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "============================================================"

# Defaults preserve the original baseline behavior, but values can be overridden
# through sbatch --export for other trained models.
: "${MODEL_BASE_PATH:=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase/original_beetle_fold0_model}"
: "${CSV_PATH:=/home/tijnveldwijk/fold0_validation_inference_inputs.csv}"
: "${CHECKPOINT_NAME:=checkpoint_best.pth}"
: "${CHECKPOINT_TAG:=original_beetle_epoch934_fast}"
: "${SAVE_VISUALS:=0}"
: "${USE_MIRRORING:=0}"

export MODEL_BASE_PATH
export CSV_PATH
export CHECKPOINT_NAME
export CHECKPOINT_TAG
export SAVE_VISUALS
export USE_MIRRORING

echo "MODEL_BASE_PATH: $MODEL_BASE_PATH"
echo "CHECKPOINT_NAME: $CHECKPOINT_NAME"
echo "CHECKPOINT_TAG:  $CHECKPOINT_TAG"
echo "SAVE_VISUALS:   $SAVE_VISUALS"
echo "USE_MIRRORING:  $USE_MIRRORING"

python -u visual_analysis.py
echo "Finished: $(date)"
