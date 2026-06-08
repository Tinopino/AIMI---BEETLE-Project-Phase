#!/usr/bin/env bash
set -euo pipefail
umask 002

###############################################################################
# PATHS
###############################################################################
ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase

NNUNET_RESULTS=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results

MODEL_BASE_PATH="$NNUNET_RESULTS/Dataset301_BEETLE/nnUNetTrainer_CutMixStainEMA_Context1024FT100__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d_context1024"

FT_MANIFEST="$ROOT/queued_context1024_finetuning.txt"

CHECKPOINT_TAG=cutmixema1000_context1024_ft100_best_mirror_visual

cd "$ROOT"
mkdir -p logs

###############################################################################
# 1. PRE-FLIGHT CHECKS
###############################################################################
echo "======================================================================"
echo "CONTEXT-EVALUATION PRE-FLIGHT CHECKS"
echo "======================================================================"

test -s "$FT_MANIFEST" || {
    echo "ERROR: context fine-tuning manifest is missing:" >&2
    echo "  $FT_MANIFEST" >&2
    exit 10
}

test -s visual_analysis.py || {
    echo "ERROR: visual_analysis.py is missing." >&2
    exit 11
}

###############################################################################
# Refuse duplicate submissions.
###############################################################################
if squeue -h -u "$USER" -o "%j" |
    grep -Eq '^eval-ctx1024-f[0-4]$'; then

    echo "ERROR: context-evaluation jobs already appear to be queued." >&2
    echo "Inspect with:" >&2
    echo "  squeue -u \$USER" >&2
    exit 12
fi

###############################################################################
# Extract job IDs from the fine-tuning manifest.
###############################################################################
get_job_id() {
    local fold="$1"
    local chunk="$2"
    local id

    id="$(
        grep -E "^[[:space:]]*fold ${fold} chunk ${chunk}: [0-9]+$" \
            "$FT_MANIFEST" |
        awk '{print $NF}' |
        head -n 1
    )"

    if [[ ! "$id" =~ ^[0-9]+$ ]]; then
        echo "ERROR: could not extract numeric job ID:" >&2
        echo "  fold=$fold chunk=$chunk" >&2
        echo "  manifest=$FT_MANIFEST" >&2
        exit 20
    fi

    echo "$id"
}

T0_FINAL=$(get_job_id 0 2)

T1_FINAL=$(get_job_id 1 2)

T2_FIRST=$(get_job_id 2 1)
T2_FINAL=$(get_job_id 2 2)

T3_FIRST=$(get_job_id 3 1)
T3_FINAL=$(get_job_id 3 2)

T4_FIRST=$(get_job_id 4 1)
T4_FINAL=$(get_job_id 4 2)

echo
echo "Context fine-tuning jobs read from manifest:"
echo "  fold 0 final: $T0_FINAL"
echo "  fold 1 final: $T1_FINAL"
echo "  fold 2 first: $T2_FIRST"
echo "  fold 2 final: $T2_FINAL"
echo "  fold 3 first: $T3_FIRST"
echo "  fold 3 final: $T3_FINAL"
echo "  fold 4 first: $T4_FIRST"
echo "  fold 4 final: $T4_FINAL"

###############################################################################
# Folds 2, 3 and 4 must still be pending so their dependencies can safely be
# updated. They will be made to wait for the earlier evaluations.
###############################################################################
assert_pending() {
    local id="$1"
    local description="$2"
    local state

    state="$(
        squeue -h -j "$id" -o "%T" |
        head -n 1
    )"

    if [[ "$state" != "PENDING" ]]; then
        echo "ERROR: expected a pending job:" >&2
        echo "  description: $description" >&2
        echo "  job ID:      $id" >&2
        echo "  state:       ${state:-not found}" >&2
        echo >&2
        echo "No evaluation jobs were submitted." >&2
        exit 21
    fi
}

assert_pending "$T2_FIRST" "context fine-tuning fold 2 chunk 1"
assert_pending "$T3_FIRST" "context fine-tuning fold 3 chunk 1"
assert_pending "$T4_FIRST" "context fine-tuning fold 4 chunk 1"

###############################################################################
# Verify that fold-aware evaluation support is installed.
###############################################################################
python - <<'PY'
from pathlib import Path

path = Path("visual_analysis.py")
text = path.read_text()

required = [
    'EVAL_FOLD = int(os.environ.get("EVAL_FOLD", "0"))',
    'FOLDS_TO_USE = (EVAL_FOLD,)',
    'MODEL_PATCH_SIZE = int(os.environ.get("MODEL_PATCH_SIZE", "512"))',
]

for marker in required:
    if marker not in text:
        raise RuntimeError(
            f"visual_analysis.py is missing required marker: {marker}"
        )

if 'MODEL_BASE_PATH / "fold_0"' in text:
    raise RuntimeError(
        "visual_analysis.py still contains a hard-coded fold_0 model path."
    )

print("Verified fold-aware visual_analysis.py")
PY

python -m py_compile visual_analysis.py

###############################################################################
# Verify that all fold-specific validation CSVs exist.
###############################################################################
for FOLD in 0 1 2 3 4; do
    CSV="/home/tijnveldwijk/fold${FOLD}_validation_inference_inputs.csv"

    test -s "$CSV" || {
        echo "ERROR: missing validation CSV:" >&2
        echo "  $CSV" >&2
        exit 22
    }

    echo "Verified validation CSV:"
    echo "  $CSV"
done

###############################################################################
# 2. CREATE THE FULL MIRRORED CONTEXT-EVALUATION LAUNCHER
###############################################################################
cat > run_context1024_eval_fold.slurm <<'SLURM'
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
#SBATCH --job-name=eval-ctx1024
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
umask 002

ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase

FOLD="${1:?Usage: sbatch run_context1024_eval_fold.slurm <fold>}"

cd "$ROOT"
mkdir -p logs

source /scratch/$USER/virtual_environments/AIMI-BEETLE/bin/activate

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

export nnUNet_raw=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_raw

export nnUNet_preprocessed=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_preprocessed

export nnUNet_results=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results

export MODEL_BASE_PATH="$nnUNet_results/Dataset301_BEETLE/nnUNetTrainer_CutMixStainEMA_Context1024FT100__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d_context1024"

export CSV_PATH="/home/tijnveldwijk/fold${FOLD}_validation_inference_inputs.csv"

export EVAL_FOLD="$FOLD"

export CHECKPOINT_NAME=checkpoint_best.pth

export CHECKPOINT_TAG=cutmixema1000_context1024_ft100_best_mirror_visual

export SAVE_VISUALS=1

export USE_MIRRORING=1

export MODEL_PATCH_SIZE=1024

export VIS_OUT_DIR="$nnUNet_results/../validation_visuals/cv_cutmixema1000_context1024_ft100/fold_${FOLD}/${CHECKPOINT_TAG}"

CHECKPOINT="$MODEL_BASE_PATH/fold_${FOLD}/$CHECKPOINT_NAME"

test -s "$CHECKPOINT" || {
    echo "ERROR: missing context-model checkpoint:" >&2
    echo "  $CHECKPOINT" >&2
    exit 30
}

test -s "$CSV_PATH" || {
    echo "ERROR: missing validation CSV:" >&2
    echo "  $CSV_PATH" >&2
    exit 31
}

echo "======================================================================"
echo "BEETLE MIRRORED FULL-VALIDATION CONTEXT EVALUATION"
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

RESULT_FILE="$(
    find "$RESULT_DIR" \
        -maxdepth 1 \
        -type f \
        \( \
            -name "fold${FOLD}_${CHECKPOINT_TAG}*confusion_matrix*.csv" \
            -o \
            -name "fold${FOLD}_${CHECKPOINT_TAG}*full_validation*.json" \
        \) \
        | head -n 1
)"

if [[ -z "$RESULT_FILE" || ! -s "$RESULT_FILE" ]]; then
    echo "ERROR: evaluation completed without a result file." >&2
    echo "Expected a matching CSV or JSON file in:" >&2
    echo "  $RESULT_DIR" >&2
    exit 32
fi

echo
echo "Verified evaluation result:"
echo "  $RESULT_FILE"

echo
echo "Finished: $(date)"
SLURM

chmod +x run_context1024_eval_fold.slurm

###############################################################################
# 3. HOLD FUTURE FINE-TUNING JOBS BRIEFLY WHILE UPDATING THE TWO LANES
###############################################################################
TO_PATCH=(
    "$T2_FIRST"
    "$T3_FIRST"
    "$T4_FIRST"
)

release_jobs() {
    for id in "${TO_PATCH[@]}"; do
        scontrol release "$id" 2>/dev/null || true
    done
}

for id in "${TO_PATCH[@]}"; do
    scontrol hold "$id"
done

trap release_jobs EXIT

###############################################################################
# 4. QUEUE THE FIVE CONTEXT EVALUATIONS
###############################################################################
E0="$(
    sbatch \
        --parsable \
        --dependency=afterok:${T0_FINAL} \
        --job-name=eval-ctx1024-f0 \
        run_context1024_eval_fold.slurm \
        0
)"

E1="$(
    sbatch \
        --parsable \
        --dependency=afterok:${T1_FINAL} \
        --job-name=eval-ctx1024-f1 \
        run_context1024_eval_fold.slurm \
        1
)"

E2="$(
    sbatch \
        --parsable \
        --dependency=afterok:${T2_FINAL} \
        --job-name=eval-ctx1024-f2 \
        run_context1024_eval_fold.slurm \
        2
)"

E3="$(
    sbatch \
        --parsable \
        --dependency=afterok:${T3_FINAL} \
        --job-name=eval-ctx1024-f3 \
        run_context1024_eval_fold.slurm \
        3
)"

E4="$(
    sbatch \
        --parsable \
        --dependency=afterok:${T4_FINAL} \
        --job-name=eval-ctx1024-f4 \
        run_context1024_eval_fold.slurm \
        4
)"

###############################################################################
# 5. INSERT THE EVALUATIONS INTO THE TWO GPU LANES
#
# Lane A:
#   fine-tune 0 -> eval 0 -> fine-tune 2 -> eval 2 -> fine-tune 4 -> eval 4
#
# Lane B:
#   fine-tune 1 -> eval 1 -> fine-tune 3 -> eval 3
###############################################################################
scontrol update \
    JobId="$T2_FIRST" \
    Dependency="afterok:$E0"

scontrol update \
    JobId="$T3_FIRST" \
    Dependency="afterok:$E1"

scontrol update \
    JobId="$T4_FIRST" \
    Dependency="afterok:$E2"

release_jobs

trap - EXIT

###############################################################################
# 6. WRITE A READABLE MANIFEST
###############################################################################
EVAL_MANIFEST="$ROOT/queued_context1024_evaluations.txt"

cat > "$EVAL_MANIFEST" <<EOF
Submitted: $(date --iso-8601=seconds)

Maximum simultaneous GPU jobs: 2

Lane A:
  fold 0 context evaluation: $E0
  fold 2 context evaluation: $E2
  fold 4 context evaluation: $E4

Lane B:
  fold 1 context evaluation: $E1
  fold 3 context evaluation: $E3

Inserted dependencies:
  context fine-tuning fold 2 chunk 1 waits for context evaluation fold 0
  context fine-tuning fold 3 chunk 1 waits for context evaluation fold 1
  context fine-tuning fold 4 chunk 1 waits for context evaluation fold 2
EOF

echo
echo "======================================================================"
echo "QUEUED FIVE MIRRORED CONTEXT EVALUATIONS"
echo "======================================================================"
echo
echo "Context-evaluation jobs:"
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
