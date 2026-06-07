#!/bin/bash
set -euo pipefail

REPO="${1:-$HOME/AIMI---BEETLE-Project-Phase}"
PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VARIANT_DIR="$REPO/nnUNet_pathology/nnunetv2/training/nnUNetTrainer/variants/pathology"
TRAINER_FILE="$REPO/nnUNet_pathology/nnunetv2/training/nnUNetTrainer/nnUnetTrainerBEETLE.py"

if [[ ! -d "$VARIANT_DIR" ]]; then
    echo "ERROR: variant directory not found: $VARIANT_DIR" >&2
    exit 1
fi

if [[ ! -f "$TRAINER_FILE" ]]; then
    echo "ERROR: trainer file not found: $TRAINER_FILE" >&2
    exit 1
fi

cp "$PACKAGE_DIR/hard_mining_batch_reference_sampler.py" "$VARIANT_DIR/"
cp "$PACKAGE_DIR/mine_hard_confusions.py" "$REPO/"
cp "$PACKAGE_DIR/make_fold_training_inference_csv.py" "$REPO/"
cp "$PACKAGE_DIR/run_mine_wf250_hard_manifest.slurm" "$REPO/"
cp "$PACKAGE_DIR/run_pathology_wfhardmine250.slurm" "$REPO/"

if grep -q "class nnUNetTrainerPathologyWFCHardMining250" "$TRAINER_FILE"; then
    echo "Trainer subclass already present; not appending a duplicate."
else
    cp "$TRAINER_FILE" "${TRAINER_FILE}.bak_before_hardmining"
    cat "$PACKAGE_DIR/nnUnetTrainerBEETLE_hardmining_append.py" >> "$TRAINER_FILE"
    echo "Appended nnUNetTrainerPathologyWFCHardMining250 to:"
    echo "  $TRAINER_FILE"
fi

python -m py_compile \
  "$VARIANT_DIR/hard_mining_batch_reference_sampler.py" \
  "$REPO/mine_hard_confusions.py" \
  "$REPO/make_fold_training_inference_csv.py" \
  "$TRAINER_FILE"

echo
echo "Installed hard-mining files successfully."
echo "Review with:"
echo "  cd \"$REPO\""
echo "  git diff --stat"
