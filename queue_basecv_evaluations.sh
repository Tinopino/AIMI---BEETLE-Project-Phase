#!/usr/bin/env bash
set -euo pipefail
umask 002

###############################################################################
# PATHS
###############################################################################
ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase

NNUNET_PREPROCESSED=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_preprocessed

NNUNET_RESULTS=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results

MODEL_BASE_PATH="$NNUNET_RESULTS/Dataset301_BEETLE/nnUNetTrainer_CutMixStainEMA__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"

SPLITS_JSON="$NNUNET_PREPROCESSED/Dataset301_BEETLE/splits.json"

TRAIN_MANIFEST="$ROOT/queued_only_five_base_folds.txt"

cd "$ROOT"
mkdir -p logs

###############################################################################
# 1. PRE-FLIGHT CHECKS
###############################################################################
echo "======================================================================"
echo "PRE-FLIGHT CHECKS"
echo "======================================================================"

test -s "$TRAIN_MANIFEST" || {
    echo "ERROR: training manifest missing:" >&2
    echo "  $TRAIN_MANIFEST" >&2
    exit 10
}

test -s "$SPLITS_JSON" || {
    echo "ERROR: splits.json missing:" >&2
    echo "  $SPLITS_JSON" >&2
    exit 11
}

test -s visual_analysis.py || {
    echo "ERROR: visual_analysis.py missing" >&2
    exit 12
}

test -s make_fold_training_inference_csv.py || {
    echo "ERROR: make_fold_training_inference_csv.py missing" >&2
    exit 13
}

###############################################################################
# Extract final training-chunk job IDs from the readable manifest.
###############################################################################
get_job_id() {
    local pattern="$1"

    local id

    id="$(
        grep -E "$pattern" "$TRAIN_MANIFEST" |
        awk '{print $NF}' |
        head -n 1
    )"

    if [[ ! "$id" =~ ^[0-9]+$ ]]; then
        echo "ERROR: could not extract numeric job ID for pattern:" >&2
        echo "  $pattern" >&2
        exit 20
    fi

    echo "$id"
}

F0_FINAL=$(get_job_id 'fold 0 chunk 2:')
F1_FINAL=$(get_job_id 'fold 1 chunk 2:')
F2_FIRST=$(get_job_id 'fold 2 chunk 1:')
F2_FINAL=$(get_job_id 'fold 2 chunk 2:')
F3_FIRST=$(get_job_id 'fold 3 chunk 1:')
F3_FINAL=$(get_job_id 'fold 3 chunk 2:')
F4_FIRST=$(get_job_id 'fold 4 chunk 1:')
F4_FINAL=$(get_job_id 'fold 4 chunk 2:')

echo
echo "Training jobs read from manifest:"
echo "  fold 0 final: $F0_FINAL"
echo "  fold 1 final: $F1_FINAL"
echo "  fold 2 first: $F2_FIRST"
echo "  fold 2 final: $F2_FINAL"
echo "  fold 3 first: $F3_FIRST"
echo "  fold 3 final: $F3_FINAL"
echo "  fold 4 first: $F4_FIRST"
echo "  fold 4 final: $F4_FINAL"

###############################################################################
# The next training folds must still be pending before dependencies are changed.
###############################################################################
assert_pending() {
    local id="$1"
    local name="$2"

    local state

    state="$(
        squeue -h -j "$id" -o "%T" |
        head -n 1
    )"

    if [[ "$state" != "PENDING" ]]; then
        echo "ERROR: $name is not pending." >&2
        echo "  job ID: $id" >&2
        echo "  state:  ${state:-not found}" >&2
        echo >&2
        echo "Do not continue automatically. Inspect the queue first." >&2
        exit 21
    fi
}

assert_pending "$F2_FIRST" "fold 2 first chunk"
assert_pending "$F3_FIRST" "fold 3 first chunk"
assert_pending "$F4_FIRST" "fold 4 first chunk"

###############################################################################
# Prevent duplicate evaluation submission.
###############################################################################
if squeue -h -u "$USER" -o "%j" |
    grep -Eq '^eval-basecv-f[0-4]$'; then

    echo "ERROR: base-CV evaluation jobs already appear to be queued." >&2
    echo "Inspect with:" >&2
    echo "  squeue -u \$USER" >&2
    exit 22
fi

###############################################################################
# 2. ENSURE visual_analysis.py SUPPORTS ARBITRARY FOLDS
###############################################################################
python - <<'PY'
from pathlib import Path

path = Path("visual_analysis.py")
text = path.read_text()

if 'EVAL_FOLD = int(os.environ.get("EVAL_FOLD", "0"))' not in text:
    old = "FOLDS_TO_USE = (0,)"

    new = """EVAL_FOLD = int(os.environ.get("EVAL_FOLD", "0"))
FOLDS_TO_USE = (EVAL_FOLD,)"""

    if old not in text:
        raise RuntimeError(
            "Could not locate FOLDS_TO_USE = (0,) in visual_analysis.py"
        )

    text = text.replace(old, new, 1)

if 'MODEL_PATCH_SIZE = int(os.environ.get("MODEL_PATCH_SIZE", "512"))' not in text:
    old = "MODEL_PATCH_SIZE = 512"

    new = 'MODEL_PATCH_SIZE = int(os.environ.get("MODEL_PATCH_SIZE", "512"))'

    if old not in text:
        raise RuntimeError(
            "Could not locate MODEL_PATCH_SIZE = 512 in visual_analysis.py"
        )

    text = text.replace(old, new, 1)

text = text.replace(
    'print("\\n=== Fold 0 full validation Dice over annotated pixels ===")',
    'print(f"\\n=== Fold {EVAL_FOLD} full validation Dice over annotated pixels ===")',
)

text = text.replace(
    'MODEL_BASE_PATH / "fold_0" / f"fold0_{CHECKPOINT_TAG}',
    'MODEL_BASE_PATH / f"fold_{EVAL_FOLD}" / f"fold{EVAL_FOLD}_{CHECKPOINT_TAG}',
)

path.write_text(text)

updated = path.read_text()

required = [
    'EVAL_FOLD = int(os.environ.get("EVAL_FOLD", "0"))',
    'FOLDS_TO_USE = (EVAL_FOLD,)',
]

for marker in required:
    if marker not in updated:
        raise RuntimeError(
            f"Missing required fold-aware marker after patch: {marker}"
        )

if 'MODEL_BASE_PATH / "fold_0"' in updated:
    raise RuntimeError(
        "visual_analysis.py still contains a hard-coded fold_0 model path. "
        "Inspect manually before queueing evaluations."
    )

print("Verified fold-aware visual_analysis.py")
PY

python -m py_compile visual_analysis.py

###############################################################################
# 3. CREATE VALIDATION INPUT CSV FILES FOR ALL FIVE FOLDS
###############################################################################
for FOLD in 0 1 2 3 4; do
    CSV="/home/tijnveldwijk/fold${FOLD}_validation_inference_inputs.csv"

    python make_fold_training_inference_csv.py \
        --splits-json "$SPLITS_JSON" \
        --fold "$FOLD" \
        --subset validation \
        --out-csv "$CSV"

    test -s "$CSV" || {
        echo "ERROR: validation CSV missing or empty:" >&2
        echo "  $CSV" >&2
        exit 30
    }

    echo "Verified validation CSV:"
    echo "  $CSV"
done

###############################################################################
# 4. CREATE FOLD-AWARE MIRRORED EVALUATION LAUNCHER
###############################################################################
cat > run_basecv_eval_fold.slurm <<'SLURM'
#!/usr/bin/env bash
#SBATCH --account=cseduimc037
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --nodelist=cn48
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=12:00:00
#SBATCH --job-name=eval-basecv
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
umask 002

ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase

FOLD="${1:?Usage: sbatch run_basecv_eval_fold.slurm <fold>}"

cd "$ROOT"
mkdir -p logs

source /scratch/$USER/virtual_environments/AIMI-BEETLE/bin/activate

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

export nnUNet_raw=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_raw

export nnUNet_preprocessed=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_preprocessed

export nnUNet_results=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results

export MODEL_BASE_PATH="$nnUNet_results/Dataset301_BEETLE/nnUNetTrainer_CutMixStainEMA__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"

export CSV_PATH="/home/tijnveldwijk/fold${FOLD}_validation_inference_inputs.csv"

export EVAL_FOLD="$FOLD"

export CHECKPOINT_NAME=checkpoint_best.pth

export CHECKPOINT_TAG="cutmixema1000_best_mirror_visual"

export SAVE_VISUALS=1

export USE_MIRRORING=1

export MODEL_PATCH_SIZE=512

export VIS_OUT_DIR="$nnUNet_results/../validation_visuals/cv_cutmixema1000/fold_${FOLD}/${CHECKPOINT_TAG}"

CHECKPOINT="$MODEL_BASE_PATH/fold_${FOLD}/$CHECKPOINT_NAME"

test -s "$CHECKPOINT" || {
    echo "ERROR: missing evaluation checkpoint:" >&2
    echo "  $CHECKPOINT" >&2
    exit 40
}

test -s "$CSV_PATH" || {
    echo "ERROR: missing validation CSV:" >&2
    echo "  $CSV_PATH" >&2
    exit 41
}

echo "======================================================================"
echo "BEETLE MIRRORED FULL-VALIDATION EVALUATION"
echo "======================================================================"
echo "Started:          $(date)"
echo "Node:             $(hostname)"
echo "SLURM_JOB_ID:     ${SLURM_JOB_ID:-none}"
echo "Fold:             $FOLD"
echo "MODEL_BASE_PATH:  $MODEL_BASE_PATH"
echo "CHECKPOINT:       $CHECKPOINT"
echo "CSV_PATH:         $CSV_PATH"
echo "CHECKPOINT_TAG:   $CHECKPOINT_TAG"
echo "MODEL_PATCH_SIZE: $MODEL_PATCH_SIZE"
echo "USE_MIRRORING:    $USE_MIRRORING"
echo "VIS_OUT_DIR:      $VIS_OUT_DIR"
echo "======================================================================"

python -u visual_analysis.py

RESULT_DIR="$MODEL_BASE_PATH/fold_${FOLD}"

CM_FILE="$(
    find "$RESULT_DIR" \
        -maxdepth 1 \
        -type f \
        -name "fold${FOLD}_${CHECKPOINT_TAG}*confusion_matrix*.csv" \
        | head -n 1
)"

if [[ -z "$CM_FILE" || ! -s "$CM_FILE" ]]; then
    echo "ERROR: evaluation completed without a confusion-matrix CSV." >&2
    echo "Expected a matching file in:" >&2
    echo "  $RESULT_DIR" >&2
    exit 42
fi

echo
echo "Verified evaluation result:"
echo "  $CM_FILE"

echo
echo "Finished: $(date)"
SLURM

chmod +x run_basecv_eval_fold.slurm

###############################################################################
# 5. QUEUE EVALUATIONS
#
# E0 and E1 run immediately after their own folds finish.
#
# Fold 2 starts after evaluation 0.
# Fold 3 starts after evaluation 1.
# Fold 4 starts after evaluation 2.
#
# This inserts evaluations into the existing two GPU lanes and guarantees that
# the full pipeline still uses no more than two GPUs simultaneously.
###############################################################################
E0="$(
    sbatch \
        --parsable \
        --dependency=afterok:${F0_FINAL} \
        --job-name=eval-basecv-f0 \
        run_basecv_eval_fold.slurm \
        0
)"

E1="$(
    sbatch \
        --parsable \
        --dependency=afterok:${F1_FINAL} \
        --job-name=eval-basecv-f1 \
        run_basecv_eval_fold.slurm \
        1
)"

E2="$(
    sbatch \
        --parsable \
        --dependency=afterok:${F2_FINAL} \
        --job-name=eval-basecv-f2 \
        run_basecv_eval_fold.slurm \
        2
)"

E3="$(
    sbatch \
        --parsable \
        --dependency=afterok:${F3_FINAL} \
        --job-name=eval-basecv-f3 \
        run_basecv_eval_fold.slurm \
        3
)"

E4="$(
    sbatch \
        --parsable \
        --dependency=afterok:${F4_FINAL} \
        --job-name=eval-basecv-f4 \
        run_basecv_eval_fold.slurm \
        4
)"

###############################################################################
# 6. INSERT EVALUATIONS INTO THE TWO GPU LANES
###############################################################################
scontrol update \
    JobId="$F2_FIRST" \
    Dependency="afterok:$E0"

scontrol update \
    JobId="$F3_FIRST" \
    Dependency="afterok:$E1"

scontrol update \
    JobId="$F4_FIRST" \
    Dependency="afterok:$E2"

###############################################################################
# 7. WRITE EVALUATION MANIFEST
###############################################################################
EVAL_MANIFEST="$ROOT/queued_basecv_evaluations.txt"

cat > "$EVAL_MANIFEST" <<EOF
Submitted: $(date --iso-8601=seconds)

Maximum simultaneous GPU jobs: 2

Lane A:
  fold 0 evaluation: $E0
  fold 2 evaluation: $E2
  fold 4 evaluation: $E4

Lane B:
  fold 1 evaluation: $E1
  fold 3 evaluation: $E3

Inserted dependencies:
  fold 2 first training chunk now waits for evaluation fold 0
  fold 3 first training chunk now waits for evaluation fold 1
  fold 4 first training chunk now waits for evaluation fold 2
EOF

echo
echo "======================================================================"
echo "QUEUED FIVE MIRRORED BASE-CV EVALUATIONS"
echo "======================================================================"
echo
echo "Evaluation jobs:"
echo "  fold 0: $E0"
echo "  fold 1: $E1"
echo "  fold 2: $E2"
echo "  fold 3: $E3"
echo "  fold 4: $E4"
echo
echo "Maximum simultaneous GPU jobs remains: 2"
echo
echo "Manifest:"
echo "  $EVAL_MANIFEST"
echo

squeue -u "$USER" \
    -o "%.18i %.32j %.2t %.12M %.45R"
