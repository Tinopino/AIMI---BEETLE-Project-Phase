#!/bin/bash
#SBATCH --account=cseduimc037
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --nodelist=cn48
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --job-name=mine_wf250_hard23
#SBATCH --output=logs/mine-wf250-hard23-%j.out
#SBATCH --error=logs/mine-wf250-hard23-%j.err

set -euo pipefail
umask 002

cd /home/tijnveldwijk/AIMI---BEETLE-Project-Phase
mkdir -p logs

source /scratch/$USER/virtual_environments/AIMI-BEETLE/bin/activate
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

export MODEL_BASE_PATH=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results/Dataset301_BEETLE/nnUNetTrainerPathologyFocalClassMetricsAlpha__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d
export CSV_PATH=/home/tijnveldwijk/fold0_training_inference_inputs.csv
export OUT_CSV=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/hard_mining/wf250_fold0_train_hard_confusions.csv

export CHECKPOINT_NAME=checkpoint_best.pth
export USE_MIRRORING=1

# Conservative first-pass settings.
export HARD_GRID_CELL=512
export MIN_CONFUSION_PIXELS_PER_CELL=500
export MAX_POINTS_PER_SLIDE_DIRECTION=200

mkdir -p "$(dirname "$OUT_CSV")"

echo "Mining weighted-focal training errors with mirroring enabled"
echo "MODEL_BASE_PATH=$MODEL_BASE_PATH"
echo "CSV_PATH=$CSV_PATH"
echo "OUT_CSV=$OUT_CSV"

python -u mine_hard_confusions.py
