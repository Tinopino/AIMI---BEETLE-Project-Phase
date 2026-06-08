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
#SBATCH --job-name=ext-ens-ctx1024
#SBATCH --output=logs/ext-ens-ctx1024-%j.out
#SBATCH --error=logs/ext-ens-ctx1024-%j.err

set -euo pipefail

cd /home/tijnveldwijk/AIMI---BEETLE-Project-Phase

mkdir -p logs inference_outputs

source /scratch/tijnveldwijk/virtual_environments/AIMI-BEETLE/bin/activate

export nnUNet_raw="/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_raw"
export nnUNet_preprocessed="/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_preprocessed"
export nnUNet_results="/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results"

export MODEL_BASE_PATH="/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results/Dataset301_BEETLE/nnUNetTrainer_CutMixStainEMA_Context1024FT100__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d_context1024"
export ROI_FOLDER="/vol/csedu-nobackup/course/IMC037_aimi/group14/aalina/images/evaluation/rois"
export OUTPUT_FOLDER="/home/tijnveldwijk/AIMI---BEETLE-Project-Phase/inference_outputs/context1024_ft100_5fold_external_rois"
export ZIP_PATH="/home/tijnveldwijk/AIMI---BEETLE-Project-Phase/inference_outputs/context1024_ft100_5fold_external_submission.zip"

rm -rf "${OUTPUT_FOLDER}"
mkdir -p "${OUTPUT_FOLDER}"

echo "============================================================"
echo "BEETLE external ROI inference: context-1024 FT100 five-fold ensemble"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "Python: $(which python)"
echo "MODEL_BASE_PATH: ${MODEL_BASE_PATH}"
echo "ROI_FOLDER: ${ROI_FOLDER}"
echo "OUTPUT_FOLDER: ${OUTPUT_FOLDER}"
echo "ZIP_PATH: ${ZIP_PATH}"
echo "============================================================"

echo
echo "Checking model metadata..."

for metadata_file in plans.json dataset.json; do
    metadata_path="${MODEL_BASE_PATH}/${metadata_file}"

    if [[ ! -f "${metadata_path}" ]]; then
        echo "ERROR: missing model metadata file: ${metadata_path}"
        exit 1
    fi

    echo "OK: ${metadata_path}"
done

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
echo "Checking ROI inputs..."

ROI_COUNT=$(find "${ROI_FOLDER}" -maxdepth 1 -type f -name '*.png' | wc -l)

echo "Found ${ROI_COUNT} ROI images."

if [[ "${ROI_COUNT}" -ne 170 ]]; then
    echo "ERROR: expected exactly 170 ROI PNG files, but found ${ROI_COUNT}."
    exit 1
fi

echo
echo "Starting five-fold ensemble inference..."

python - <<'PY'
import os
from inference import main

main(
    model_base_path=os.environ["MODEL_BASE_PATH"],
    checkpoint_name="checkpoint_best.pth",
    folds_to_use=(0, 1, 2, 3, 4),
    roi_folder=os.environ["ROI_FOLDER"],
    output_folder=os.environ["OUTPUT_FOLDER"],
)
PY

echo
echo "Validating submission files and creating ZIP..."

python - <<'PY'
import os
from pathlib import Path
from PIL import Image
import numpy as np
import zipfile

roi_folder = Path(os.environ["ROI_FOLDER"])
output_folder = Path(os.environ["OUTPUT_FOLDER"])
zip_path = Path(os.environ["ZIP_PATH"])

expected_labels = {1, 2, 3, 4}

roi_paths = sorted(roi_folder.glob("*.png"))
prediction_paths = sorted(output_folder.glob("*.png"))

roi_names = {path.name for path in roi_paths}
prediction_names = {path.name for path in prediction_paths}

print(f"Input ROI count:       {len(roi_paths)}")
print(f"Prediction PNG count:  {len(prediction_paths)}")

if len(roi_paths) != 170:
    raise RuntimeError(
        f"Expected exactly 170 ROI inputs, but found {len(roi_paths)}."
    )

if len(prediction_paths) != 170:
    raise RuntimeError(
        f"Expected exactly 170 predictions, but found {len(prediction_paths)}."
    )

missing = sorted(roi_names - prediction_names)
unexpected = sorted(prediction_names - roi_names)

if missing:
    raise RuntimeError(f"Missing prediction files: {missing}")

if unexpected:
    raise RuntimeError(f"Unexpected prediction files: {unexpected}")

observed_labels = set()

for roi_path in roi_paths:
    prediction_path = output_folder / roi_path.name

    with Image.open(roi_path) as roi_image:
        roi_size = roi_image.size

    with Image.open(prediction_path) as prediction_image:
        prediction_size = prediction_image.size
        prediction_mode = prediction_image.mode
        prediction_array = np.asarray(prediction_image)

    if prediction_size != roi_size:
        raise RuntimeError(
            f"Dimension mismatch for {roi_path.name}: "
            f"ROI={roi_size}, prediction={prediction_size}"
        )

    if prediction_array.ndim != 2:
        raise RuntimeError(
            f"{roi_path.name} is not single-channel: "
            f"shape={prediction_array.shape}, mode={prediction_mode}"
        )

    labels = set(np.unique(prediction_array).tolist())
    observed_labels.update(labels)

    invalid_labels = labels - expected_labels

    if invalid_labels:
        raise RuntimeError(
            f"{roi_path.name} contains invalid labels: "
            f"{sorted(invalid_labels)}"
        )

if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(
    zip_path,
    "w",
    compression=zipfile.ZIP_DEFLATED,
) as zip_file:
    for prediction_path in prediction_paths:
        # Store the PNG files directly at the root of the ZIP archive.
        zip_file.write(prediction_path, arcname=prediction_path.name)

print(f"Observed labels:       {sorted(observed_labels)}")
print(f"Validated ZIP created: {zip_path}")
PY

echo
echo "============================================================"
echo "Inference finished successfully: $(date)"
echo "Predictions: ${OUTPUT_FOLDER}"
echo "Submission ZIP: ${ZIP_PATH}"
echo "============================================================"

