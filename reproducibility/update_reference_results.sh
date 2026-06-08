#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/tijnveldwijk/AIMI---BEETLE-Project-Phase"
GROUP_ROOT="/vol/csedu-nobackup/course/IMC037_aimi/group14"
RESULTS_ROOT="$GROUP_ROOT/nnunet/tijn/pathology/nnUNet_results/Dataset301_BEETLE"

REPRO="$PROJECT/reproducibility"
REFERENCE_RESULTS="$REPRO/reference_results"
REFERENCE_LOGS="$REPRO/reference_logs"

mkdir -p "$REFERENCE_RESULTS" "$REFERENCE_LOGS"

echo "Copying compact result artefacts..."

find "$RESULTS_ROOT" \
    -type f \
    \( \
        -name '*full_validation_dice*.json' -o \
        -name '*confusion_matrix*.csv' -o \
        -name 'class_metrics.csv' -o \
        -name 'class_metrics.jsonl' -o \
        -name 'debug.json' -o \
        -name 'checkpoint_manifest*.txt' -o \
        -name 'training_log*.txt' \
    \) \
    -size -5M \
    -print0 |
while IFS= read -r -d '' file; do
    rel="${file#$RESULTS_ROOT/}"
    dest="$REFERENCE_RESULTS/$rel"
    mkdir -p "$(dirname "$dest")"
    cp -p "$file" "$dest"
done

echo "Copying relevant compact SLURM stdout logs..."

if [[ -d "$PROJECT/logs" ]]; then
    find "$PROJECT/logs" \
        -maxdepth 1 \
        -type f \
        \( \
            -name 'eval-*.out' -o \
            -name '*original*eval*.out' -o \
            -name '*mine*.out' -o \
            -name '*filter*.out' \
        \) \
        -size -5M \
        -print0 |
    while IFS= read -r -d '' file; do
        cp -p "$file" "$REFERENCE_LOGS/"
    done
fi

echo
echo "Final CutMix + stain jitter + EMA fold status:"

MODEL="$RESULTS_ROOT/nnUNetTrainer_CutMixStainEMA__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"

for fold in 0 1 2 3 4; do
    printf 'fold_%s: ' "$fold"
    if find "$MODEL/fold_$fold" \
        -maxdepth 1 \
        -type f \
        -name '*full_validation_dice*.json' \
        | grep -q .; then
        echo "evaluation result present"
    else
        echo "evaluation result missing or still running"
    fi
done
