#!/bin/bash
#SBATCH --job-name=beetle_inf_f0
#SBATCH --account=cseduimc037
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --nodelist=cn48
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --time=03:00:00
#SBATCH --array=0-83
#SBATCH --output=logs/infer_fold0-%A_%a.out
#SBATCH --error=logs/infer_fold0-%A_%a.err

set -euo pipefail

cd /home/tijnveldwijk/AIMI---BEETLE-Project-Phase
mkdir -p logs

source /scratch/tijnveldwijk/virtual_environments/AIMI-BEETLE/bin/activate

export PYTHONPATH="$PWD/nnUNet_pathology:${PYTHONPATH:-}"

export nnUNet_raw=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_raw
export nnUNet_preprocessed=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_preprocessed
export nnUNet_results=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results

CSV=/home/tijnveldwijk/fold0_validation_inference_inputs.csv
LINE=$((SLURM_ARRAY_TASK_ID + 2))

WSI=$(sed -n "${LINE}p" "$CSV" | cut -d',' -f1)
MASK=$(sed -n "${LINE}p" "$CSV" | cut -d',' -f2)

OUTDIR=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/validation_inference/DiceFocal_fold0
mkdir -p "$OUTDIR"

STEM=$(basename "$WSI" .tif)

python nnUNet_pathology/pathology_code_and_utils/nnUNetV2_run_WSI_inference_REWORK_with_config.py \
    beetle_focal_fold0 \
    "$WSI" \
    "$MASK" \
    "$OUTDIR/${STEM}_nnunet.tif" \
    "$OUTDIR/${STEM}_uncertainty.tif" \
    "$OUTDIR/${STEM}_uncertainty_KL.tif" \
    "$OUTDIR/${STEM}_uncertainty_entropy.tif"

