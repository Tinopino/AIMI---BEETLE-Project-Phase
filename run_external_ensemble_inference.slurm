#!/bin/bash
#SBATCH --account=cseduimc037
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --nodelist=cn48
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --job-name=ext-ens-cutmixema
#SBATCH --output=logs/ext-ens-cutmixema-%j.out
#SBATCH --error=logs/ext-ens-cutmixema-%j.err

set -euo pipefail

cd /home/tijnveldwijk/AIMI---BEETLE-Project-Phase
source /scratch/tijnveldwijk/virtual_environments/AIMI-BEETLE/bin/activate

MODEL_BASE_PATH="/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results/Dataset301_BEETLE/nnUNetTrainer_CutMixStainEMA__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"
ROI_FOLDER="/vol/csedu-nobackup/course/IMC037_aimi/group14/aalina/images/evaluation/rois"
OUTPUT_FOLDER="/home/tijnveldwijk/AIMI---BEETLE-Project-Phase/inference_outputs/cutmix_stain_ema_5fold_external_rois"

mkdir -p "${OUTPUT_FOLDER}"

echo "============================================================"
echo "BEETLE external ROI inference: 5-fold CutMix + stain + EMA ensemble"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "MODEL_BASE_PATH: ${MODEL_BASE_PATH}"
echo "ROI_FOLDER: ${ROI_FOLDER}"
echo "OUTPUT_FOLDER: ${OUTPUT_FOLDER}"
echo "============================================================"

echo
echo "Checking fold checkpoints..."
for fold in {0..4}; do
    checkpoint="${MODEL_BASE_PATH}/fold_${fold}/checkpoint_best.pth"
    if [[ ! -f "${checkpoint}" ]]; then
        echo "ERROR: missing checkpoint: ${checkpoint}"
        exit 1
    fi
    echo "OK: ${checkpoint}"
done

echo
echo "Found $(find "${ROI_FOLDER}" -maxdepth 1 -type f -name '*.png' | wc -l) ROI images."

python - <<'PY'
from inference import main

main(
    model_base_path=(
        "/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/"
        "nnUNet_results/Dataset301_BEETLE/"
        "nnUNetTrainer_CutMixStainEMA__nnUNetWholeSlideDataPlans__"
        "wsd_None_iterator_nnunet_aug__2d"
    ),
    checkpoint_name="checkpoint_best.pth",
    folds_to_use=(0, 1, 2, 3, 4),
    roi_folder=(
        "/vol/csedu-nobackup/course/IMC037_aimi/group14/"
        "aalina/images/evaluation/rois"
    ),
    output_folder=(
        "/home/tijnveldwijk/AIMI---BEETLE-Project-Phase/"
        "inference_outputs/cutmix_stain_ema_5fold_external_rois"
    ),
)
PY

echo
echo "============================================================"
echo "Inference finished: $(date)"
echo "Predictions: ${OUTPUT_FOLDER}"
echo "Number of saved PNG files:"
find "${OUTPUT_FOLDER}" -maxdepth 1 -type f -name '*.png' | wc -l
echo "============================================================"
