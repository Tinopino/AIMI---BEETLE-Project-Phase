#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase
LOGDIR="$ROOT/logs"

BROAD=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/hard_mining/wf250_fold0_train_hard_confusions.csv
FILTERED=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/hard_mining/wf250_fold0_train_hard_confusions_filtered.csv
SUMMARY=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/hard_mining/wf250_fold0_train_hard_confusions_filtered.summary.json

cd "$ROOT"
mkdir -p "$LOGDIR"

###############################################################################
# 1. Find the currently running broad mining job
###############################################################################
MINE_JOB="${1:-}"

if [[ -z "$MINE_JOB" ]]; then
    MINE_JOB="$(
        squeue -u "$USER" -h -n mine_wf250_hard23 -o "%A" | head -n 1
    )"
fi

if [[ -z "$MINE_JOB" ]]; then
    echo "ERROR: Could not find the running mining job."
    echo "Run: squeue -u \$USER"
    echo "Then run: ./queue_wfhard_overnight.sh <MINING_JOB_ID>"
    exit 1
fi

echo "Using mining job: $MINE_JOB"

###############################################################################
# 2. Create the manifest-filtering Python script
###############################################################################
cat > filter_wfhard_manifest.py <<'PY'
#!/usr/bin/env python3

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tiffslide import TiffSlide

BROAD = Path(
    "/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/"
    "hard_mining/wf250_fold0_train_hard_confusions.csv"
)

FILTERED = Path(
    "/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/"
    "hard_mining/wf250_fold0_train_hard_confusions_filtered.csv"
)

SUMMARY = Path(
    "/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/"
    "hard_mining/wf250_fold0_train_hard_confusions_filtered.summary.json"
)

CELL_SIZE = 512

TARGET_PER_DIRECTION = 750
MIN_ACCEPTABLE_PER_DIRECTION = 150
MAX_PER_SLIDE_DIRECTION = 10

# Prefer clear and substantial class-2 <-> class-3 mistakes.
# Relax thresholds only if the strict tier does not provide enough diversity.
TIERS = [
    {
        "name": "strict",
        "min_confusion_pixels": 2500,
        "min_source_pixels": 2500,
        "min_source_error_rate": 0.30,
    },
    {
        "name": "moderate",
        "min_confusion_pixels": 1500,
        "min_source_pixels": 1500,
        "min_source_error_rate": 0.25,
    },
    {
        "name": "fallback",
        "min_confusion_pixels": 1000,
        "min_source_pixels": 1000,
        "min_source_error_rate": 0.20,
    },
]

if not BROAD.is_file():
    raise FileNotFoundError(f"Broad manifest not found: {BROAD}")

with BROAD.open(newline="") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise RuntimeError(f"Broad manifest is empty: {BROAD}")

print("Broad rows:", len(rows), flush=True)
print(
    "Broad rows by direction:",
    dict(Counter(row["direction"] for row in rows)),
    flush=True,
)

rows_by_mask = defaultdict(list)

for row in rows:
    rows_by_mask[row["mask_path"]].append(row)

enriched = []

for mask_index, (mask_path, mask_rows) in enumerate(rows_by_mask.items(), start=1):
    mask = TiffSlide(mask_path)
    mask_w, mask_h = mask.dimensions

    for row in mask_rows:
        x = int(row["cell_x_out"])
        y = int(row["cell_y_out"])
        gt_label = int(row["gt_label"])
        confusion_pixels = int(row["confusion_pixels_in_cell"])

        width = min(CELL_SIZE, mask_w - x)
        height = min(CELL_SIZE, mask_h - y)

        if width <= 0 or height <= 0:
            continue

        patch = np.asarray(mask.read_region((x, y), 0, (width, height)))

        if patch.ndim == 3:
            patch = patch[..., 0]

        source_pixels = int((patch == gt_label).sum())

        source_error_rate = (
            confusion_pixels / source_pixels
            if source_pixels > 0
            else 0.0
        )

        hardness_score = confusion_pixels * source_error_rate

        item = dict(row)
        item["gt_source_pixels_in_cell"] = source_pixels
        item["confusion_fraction_of_source_gt"] = f"{source_error_rate:.8f}"
        item["hardness_score"] = f"{hardness_score:.8f}"

        enriched.append(item)

    mask.close()

    if mask_index % 25 == 0:
        print(
            f"Processed masks: {mask_index}/{len(rows_by_mask)}",
            flush=True,
        )

def passes(row, tier):
    return (
        int(row["confusion_pixels_in_cell"]) >= tier["min_confusion_pixels"]
        and int(row["gt_source_pixels_in_cell"]) >= tier["min_source_pixels"]
        and float(row["confusion_fraction_of_source_gt"])
        >= tier["min_source_error_rate"]
    )

selected_all = []
summary_by_direction = {}

for direction in ["gt2_pred3", "gt3_pred2"]:
    direction_rows = [
        row for row in enriched
        if row["direction"] == direction
    ]

    direction_rows.sort(
        key=lambda row: float(row["hardness_score"]),
        reverse=True,
    )

    selected = []
    selected_keys = set()
    per_slide_count = Counter()
    tier_counts = Counter()

    for tier in TIERS:
        for row in direction_rows:
            key = (
                row["slide"],
                row["direction"],
                row["center_x"],
                row["center_y"],
            )

            if key in selected_keys:
                continue

            if per_slide_count[row["slide"]] >= MAX_PER_SLIDE_DIRECTION:
                continue

            if not passes(row, tier):
                continue

            item = dict(row)
            item["selection_tier"] = tier["name"]

            selected.append(item)
            selected_keys.add(key)
            per_slide_count[row["slide"]] += 1
            tier_counts[tier["name"]] += 1

            if len(selected) >= TARGET_PER_DIRECTION:
                break

        if len(selected) >= TARGET_PER_DIRECTION:
            break

    if len(selected) < MIN_ACCEPTABLE_PER_DIRECTION:
        raise RuntimeError(
            f"Too few usable hard examples for {direction}: "
            f"{len(selected)} < {MIN_ACCEPTABLE_PER_DIRECTION}"
        )

    selected_all.extend(selected)

    summary_by_direction[direction] = {
        "broad_rows": len(direction_rows),
        "selected_rows": len(selected),
        "unique_slides": len({row["slide"] for row in selected}),
        "tier_counts": dict(tier_counts),
    }

selected_all.sort(
    key=lambda row: (
        row["direction"],
        -float(row["hardness_score"]),
    )
)

fieldnames = list(selected_all[0].keys())

with FILTERED.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(selected_all)

summary = {
    "broad_manifest": str(BROAD),
    "filtered_manifest": str(FILTERED),
    "broad_rows": len(rows),
    "selected_total": len(selected_all),
    "target_per_direction": TARGET_PER_DIRECTION,
    "max_per_slide_direction": MAX_PER_SLIDE_DIRECTION,
    "sampling_plan": {
        "normal_sampling_fraction": 0.75,
        "hard_sampling_fraction": 0.25,
        "hard_sampling_jitter_pixels": 128,
    },
    "threshold_tiers": TIERS,
    "selected_by_direction": summary_by_direction,
}

with SUMMARY.open("w") as f:
    json.dump(summary, f, indent=4)

print()
print("============================================================")
print("FILTERING COMPLETED")
print("============================================================")
print("Filtered manifest:", FILTERED)
print("Summary:", SUMMARY)
print("Selected total:", len(selected_all))

for direction, values in summary_by_direction.items():
    print(
        f"{direction}: "
        f"selected={values['selected_rows']} | "
        f"slides={values['unique_slides']} | "
        f"tiers={values['tier_counts']}"
    )
PY

###############################################################################
# 3. Create the CPU filtering SLURM job
###############################################################################
cat > filter_wfhard_manifest.slurm <<'FILTERBASH'
#!/usr/bin/env bash
#SBATCH --account=cseduimc037
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --job-name=filter_wfhard
#SBATCH --output=logs/filter-wfhard-%j.out
#SBATCH --error=logs/filter-wfhard-%j.err

set -euo pipefail

cd /home/tijnveldwijk/AIMI---BEETLE-Project-Phase

source /scratch/$USER/virtual_environments/AIMI-BEETLE/bin/activate

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

python -u filter_wfhard_manifest.py
FILTERBASH

chmod +x filter_wfhard_manifest.py
chmod +x filter_wfhard_manifest.slurm

###############################################################################
# 4. Patch final evaluation so USE_MIRRORING=1 truly enables 2D TTA
###############################################################################
python - <<'PY'
from pathlib import Path

path = Path("visual_analysis.py")
text = path.read_text()

if "FORCED_MIRRORING_AXES" not in text:
    needle = '''    predictor.initialize_from_trained_model_folder(
        str(MODEL_BASE_PATH),
        use_folds=FOLDS_TO_USE,
        checkpoint_name=CHECKPOINT_NAME,
    )
'''

    replacement = needle + '''
    # Explicitly enable both spatial axes for 2D test-time augmentation.
    if USE_MIRRORING and predictor.allowed_mirroring_axes is None:
        predictor.allowed_mirroring_axes = (0, 1)
        print("FORCED_MIRRORING_AXES:", predictor.allowed_mirroring_axes)

    print("ALLOWED_MIRRORING_AXES:", predictor.allowed_mirroring_axes)
    print(
        "EFFECTIVE_MIRRORING:",
        bool(USE_MIRRORING and predictor.allowed_mirroring_axes),
    )
'''

    if needle not in text:
        raise RuntimeError(
            "Could not locate predictor initialization in visual_analysis.py"
        )

    path.write_text(text.replace(needle, replacement, 1))

    print("Patched visual_analysis.py for effective mirroring")
else:
    print("visual_analysis.py was already patched")
PY

python -m py_compile visual_analysis.py

###############################################################################
# 5. Queue: current mining -> filter -> training -> mirrored evaluation
###############################################################################
FILTER_JOB="$(
    sbatch --parsable \
      --dependency=afterok:${MINE_JOB} \
      filter_wfhard_manifest.slurm
)"

echo "Queued filtering job: $FILTER_JOB"

TRAIN_JOB="$(
    sbatch --parsable \
      --dependency=afterok:${FILTER_JOB} \
      --export=ALL,HARD_MINING_MANIFEST=${FILTERED},HARD_MINING_FRACTION=0.25,HARD_MINING_JITTER=128 \
      run_pathology_wfhardmine250.slurm \
      0 fresh
)"

echo "Queued hard-mining training job: $TRAIN_JOB"

EVAL_JOB="$(
    sbatch --parsable \
      --dependency=afterok:${TRAIN_JOB} \
      --job-name=eval_wfhardmine250_mirror \
      --output=logs/eval-wfhardmine250-mirror-%j.out \
      --error=logs/eval-wfhardmine250-mirror-%j.err \
      --export=ALL,MODEL_BASE_PATH=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results/Dataset301_BEETLE/nnUNetTrainerPathologyWFCHardMining250__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d,CHECKPOINT_NAME=checkpoint_best.pth,CHECKPOINT_TAG=wfhardmine250_best_mirror_visual,SAVE_VISUALS=1,USE_MIRRORING=1 \
      run_original_beetle_fast_eval.slurm
)"

echo "Queued mirrored evaluation job: $EVAL_JOB"

echo
echo "============================================================"
echo "OVERNIGHT PIPELINE QUEUED"
echo "============================================================"
echo "Mining:    $MINE_JOB"
echo "Filtering: $FILTER_JOB"
echo "Training:  $TRAIN_JOB"
echo "Evaluation:$EVAL_JOB"
echo
echo "Dependency chain:"
echo "mining -> filtering -> training -> mirrored evaluation"
echo

squeue -u "$USER"
