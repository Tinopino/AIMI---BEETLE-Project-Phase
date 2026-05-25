#!/bin/bash
#SBATCH --job-name=beetle_preprocess
#SBATCH --account=cseduimc037
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --cpus-per-task=6
#SBATCH --mem=14G
#SBATCH --time=02:00:00
#SBATCH --output=logs/preprocess-%j.out
#SBATCH --error=logs/preprocess-%j.out

set -euo pipefail

cd ~/AIMI---BEETLE-Project-Phase

source /vol/csedu-nobackup/course/IMC037_aimi/group14/venvs/tijn/beetle/bin/activate

export nnUNet_raw=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/dicefocal/nnUNet_raw
export nnUNet_preprocessed=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/dicefocal/nnUNet_preprocessed
export nnUNet_results=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/dicefocal/nnUNet_results

export TMPDIR=/vol/csedu-nobackup/course/IMC037_aimi/group14/tmp/tijn
export PIP_CACHE_DIR=/vol/csedu-nobackup/course/IMC037_aimi/group14/pip_cache/tijn

mkdir -p logs "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results" "$TMPDIR" "$PIP_CACHE_DIR"

echo "Converting BEETLE mini dataset..."
rm -rf "$nnUNet_raw/Dataset301_BEETLE"
python convert_beetle_mini.py

echo "Checking conversion..."
find "$nnUNet_raw/Dataset301_BEETLE/imagesTr" -name "*_0000.png" | wc -l
find "$nnUNet_raw/Dataset301_BEETLE/labelsTr" -name "*.png" | wc -l

echo "Planning and preprocessing..."
rm -rf "$nnUNet_preprocessed/Dataset301_BEETLE"

nnUNetv2_plan_and_preprocess \
  -d 301 \
  -c 2d \
  --verify_dataset_integrity

echo "Preprocessing finished."
