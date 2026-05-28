#!/bin/bash
#SBATCH --job-name=focal_valvis_f0
#SBATCH --account=cseduimc037
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --nodelist=cn48
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=logs/focal_valvis_fold0-%j.out
#SBATCH --error=logs/focal_valvis_fold0-%j.err

set -euo pipefail

cd /home/tijnveldwijk/AIMI---BEETLE-Project-Phase
mkdir -p logs

source /scratch/tijnveldwijk/virtual_environments/AIMI-BEETLE/bin/activate

export PYTHONPATH="$PWD/nnUNet_pathology:${PYTHONPATH:-}"

export nnUNet_raw=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_raw
export nnUNet_preprocessed=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_preprocessed
export nnUNet_results=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results

MODEL_BASE="/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results/Dataset301_BEETLE/nnUNetTrainerPathologyFocal__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"
CHECKPOINT="$MODEL_BASE/fold_0/checkpoint_best.pth"

python -c "import tiffslide; print('tiffslide ok')"

echo "Checking model files:"
ls -lh "$MODEL_BASE/plans.json"
ls -lh "$MODEL_BASE/dataset.json"
ls -lh "$CHECKPOINT"

echo "Running full fold-0 Dice + visuals on checkpoint_best..."
python compute_fold0_dice_full_tiffslide_hybrid_visual.py