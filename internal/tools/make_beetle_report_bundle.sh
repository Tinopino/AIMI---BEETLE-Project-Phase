#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/tijnveldwijk/AIMI---BEETLE-Project-Phase"
RESULTS_ROOT="/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results/Dataset301_BEETLE"
ORIGINAL_BASELINE="$PROJECT/original_beetle_fold0_model"

STAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE="$PROJECT/beetle_report_bundle_$STAMP"
ARCHIVE="$PROJECT/beetle_report_bundle_$STAMP.tar.gz"

mkdir -p \
    "$BUNDLE/results" \
    "$BUNDLE/original_beetle_fold0_model" \
    "$BUNDLE/project_logs" \
    "$BUNDLE/configs" \
    "$BUNDLE/source_code" \
    "$BUNDLE/slurm_scripts" \
    "$BUNDLE/report"

echo "============================================================"
echo "1. Copying non-checkpoint experiment artefacts"
echo "============================================================"

rsync -a \
    --exclude='*.pth' \
    "$RESULTS_ROOT/" \
    "$BUNDLE/results/"

echo
echo "============================================================"
echo "2. Copying released BEETLE baseline artefacts"
echo "============================================================"

if [[ -d "$ORIGINAL_BASELINE" ]]; then
    rsync -a \
        --exclude='*.pth' \
        "$ORIGINAL_BASELINE/" \
        "$BUNDLE/original_beetle_fold0_model/"
else
    echo "Baseline folder not found: $ORIGINAL_BASELINE"
fi

echo
echo "============================================================"
echo "3. Copying project logs"
echo "============================================================"

if [[ -d "$PROJECT/logs" ]]; then
    rsync -a "$PROJECT/logs/" "$BUNDLE/project_logs/"
fi

echo
echo "============================================================"
echo "4. Copying split and dataset configuration"
echo "============================================================"

SPLITS="$RESULTS_ROOT/nnUNetTrainerPathologyFocal__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d/splits.json"

if [[ -f "$SPLITS" ]]; then
    cp "$SPLITS" "$BUNDLE/configs/splits.json"
else
    echo "WARNING: splits.json not found at expected location"
fi

find "/vol/csedu-nobackup/course/IMC037_aimi/group14" \
    -path '*Dataset301_BEETLE*' \
    -name 'dataset.json' \
    -type f \
    -print \
    -exec cp '{}' "$BUNDLE/configs/" \; \
    2>/dev/null || true

echo
echo "============================================================"
echo "5. Copying trainer and evaluation source code"
echo "============================================================"

if [[ -d "$PROJECT/nnUNet_pathology/nnunetv2/training/nnUNetTrainer" ]]; then
    rsync -a \
        "$PROJECT/nnUNet_pathology/nnunetv2/training/nnUNetTrainer/" \
        "$BUNDLE/source_code/nnUNetTrainer/"
fi

find "$PROJECT" \
    -maxdepth 3 \
    -type f \
    \( \
        -name '*.slurm' -o \
        -name 'inference.py' -o \
        -name '*eval*.py' -o \
        -name '*dice*.py' -o \
        -name 'visual_analysis.py' -o \
        -name '*iterator*.json' -o \
        -name '*template*.json' \
    \) \
    -print0 |
while IFS= read -r -d '' FILE; do
    REL="${FILE#$PROJECT/}"
    DEST="$BUNDLE/source_code/$REL"
    mkdir -p "$(dirname "$DEST")"
    cp "$FILE" "$DEST"
done

echo
echo "============================================================"
echo "6. Copying report files when present"
echo "============================================================"

find "$PROJECT" \
    -maxdepth 3 \
    -type f \
    \( \
        -name '*.tex' -o \
        -name '*.bib' \
    \) \
    -print0 |
while IFS= read -r -d '' FILE; do
    cp "$FILE" "$BUNDLE/report/"
done

echo
echo "============================================================"
echo "7. Copying generated inventory files"
echo "============================================================"

for FILE in \
    "$PROJECT/paper_experiment_inventory.txt" \
    "$PROJECT/paper_metric_log_hits.txt"
do
    [[ -f "$FILE" ]] && cp "$FILE" "$BUNDLE/"
done

echo
echo "============================================================"
echo "8. Creating compressed archive"
echo "============================================================"

tar -czf "$ARCHIVE" \
    -C "$PROJECT" \
    "$(basename "$BUNDLE")"

echo
echo "Done."
echo "Bundle folder:"
echo "  $BUNDLE"
echo
echo "Archive to upload:"
echo "  $ARCHIVE"
echo
echo "Archive size:"
du -sh "$ARCHIVE"
