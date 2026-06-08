#!/usr/bin/env bash
set -euo pipefail
umask 002

###############################################################################
# PROJECT PATHS
###############################################################################
ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase

NNUNET_RAW=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_raw
NNUNET_PREPROCESSED=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_preprocessed
NNUNET_RESULTS=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results

DATASET=Dataset301_BEETLE
PLANNER=nnUNetWholeSlideDataPlans

BASE_TRAINER=nnUNetTrainer_CutMixStainEMA
CONTEXT_TRAINER=nnUNetTrainer_CutMixStainEMA_Context1024FT100

BASE_MODEL_DIR="$NNUNET_RESULTS/$DATASET/${BASE_TRAINER}__${PLANNER}__wsd_None_iterator_nnunet_aug__2d"
CONTEXT_MODEL_DIR="$NNUNET_RESULTS/$DATASET/${CONTEXT_TRAINER}__${PLANNER}__wsd_None_iterator_nnunet_aug__2d_context1024"

SPLITS_JSON="$NNUNET_PREPROCESSED/$DATASET/splits.json"

TRAINER_FILE="$ROOT/nnUNet_pathology/nnunetv2/training/nnUNetTrainer/nnUnetTrainerBEETLE.py"

DATALOADER_FILE="$ROOT/nnUNet_pathology/nnunetv2/training/nnUNetTrainer/variants/pathology/nnUNetTrainer_WSD_undefined_dataloader.py"

cd "$ROOT"
mkdir -p logs cv_summaries/cutmixema1000_context1024

###############################################################################
# 0. PRE-FLIGHT CHECKS
###############################################################################
echo "======================================================================"
echo "PRE-FLIGHT CHECKS"
echo "======================================================================"

test -s "$SPLITS_JSON" || {
    echo "ERROR: missing splits.json:" >&2
    echo "  $SPLITS_JSON" >&2
    exit 10
}

test -s "$TRAINER_FILE" || {
    echo "ERROR: missing trainer file:" >&2
    echo "  $TRAINER_FILE" >&2
    exit 11
}

test -s "$DATALOADER_FILE" || {
    echo "ERROR: missing WSD dataloader file:" >&2
    echo "  $DATALOADER_FILE" >&2
    exit 12
}

test -s "$BASE_MODEL_DIR/fold_0/checkpoint_latest.pth" || \
test -s "$BASE_MODEL_DIR/fold_0/checkpoint_final.pth" || {
    echo "ERROR: fold 0 has no resumable or completed checkpoint:" >&2
    echo "  $BASE_MODEL_DIR/fold_0/checkpoint_latest.pth" >&2
    echo "  $BASE_MODEL_DIR/fold_0/checkpoint_final.pth" >&2
    exit 13
}

if squeue -h -u "$USER" -o "%j" |
    grep -Eq \
    '^(cutmixema-v2-|eval-cema-v2-|ctxft-v2-|eval-ctx-v2-|agg-cema-v2|agg-final-v2)'; then
    echo "ERROR: a v2 pipeline already appears to be queued." >&2
    echo "Inspect with:" >&2
    echo "  squeue -u \$USER" >&2
    exit 14
fi

echo "Fold-0 resumable checkpoint found."
echo

###############################################################################
# 1. PATCH OPTIONAL WSD BATCH-SIZE OVERRIDE
###############################################################################
python - <<'PY'
from pathlib import Path

path = Path(
    "nnUNet_pathology/nnunetv2/training/nnUNetTrainer/"
    "variants/pathology/nnUNetTrainer_WSD_undefined_dataloader.py"
)

text = path.read_text()

if "wsd_batch_size_override" in text:
    print("WSD batch-size override already installed")
else:
    old = """        print('\\n\\n\\nTEMP BATCH SIZE 8\\n\\n\\n')
        # batch_size = self.configuration_manager.batch_size
        batch_size = 8
"""

    new = """        batch_size = int(getattr(self, "wsd_batch_size_override", 8))
        print(f'\\n\\n\\nWSD BATCH SIZE {batch_size}\\n\\n\\n')
"""

    if old not in text:
        raise RuntimeError(
            "Could not locate the WSD batch-size block. "
            "Inspect nnUNetTrainer_WSD_undefined_dataloader.py manually."
        )

    path.write_text(text.replace(old, new, 1))
    print("Installed optional WSD batch-size override")
PY

###############################################################################
# 2. ADD CONTEXT-FINETUNING TRAINER IF MISSING
###############################################################################
python - <<'PY'
from pathlib import Path

path = Path(
    "nnUNet_pathology/nnunetv2/training/nnUNetTrainer/"
    "nnUnetTrainerBEETLE.py"
)

text = path.read_text()

marker = "class nnUNetTrainer_CutMixStainEMA_Context1024FT100"

if marker in text:
    print("Context trainer already installed")
else:
    addition = r'''

# =============================================================================
# FINAL CUTMIX + STAIN JITTER + EMA + 1024x1024 CONTEXT REFINEMENT
# =============================================================================

class nnUNetTrainer_CutMixStainEMA_Context1024FT100(
    nnUNetTrainer_CutMixStainEMA
):
    """
    Fine-tunes the completed CutMix + stain jitter + EMA model with a larger
    1024x1024 tissue-context window.

    Inherited:
    - alpha-weighted Dice + focal loss;
    - CutMix;
    - stain jitter;
    - EMA;
    - checkpoint_latest.pth;
    - checkpoint_best.pth;
    - class-specific best checkpoints;
    - checkpoint_final.pth;
    - per-class metric logging.

    Changed:
    - patch size: 1024x1024;
    - online WSD batch size: 2;
    - initial LR: 0.0005;
    - additional epochs: 100.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans,
            configuration,
            fold,
            dataset_json,
            unpack_dataset,
            device,
        )

        expected_patch_size = [1024, 1024]
        actual_patch_size = list(self.configuration_manager.patch_size)

        if actual_patch_size != expected_patch_size:
            raise RuntimeError(
                "Context trainer requires patch_size="
                f"{expected_patch_size}, received {actual_patch_size}. "
                "Use configuration='2d_context1024'."
            )

        self.wsd_batch_size_override = 2
        self.initial_lr = 5e-4
        self.num_epochs = 100
        self.save_every = 1

        self.print_to_log_file(
            "Using CutMix + stain jitter + EMA context fine-tuning: "
            "patch_size=[1024, 1024], WSD batch_size=2, "
            "initial_lr=0.0005, num_epochs=100",
            also_print_to_console=True,
        )
'''

    path.write_text(text + addition)
    print("Installed context trainer")
PY

python -m py_compile "$TRAINER_FILE"
python -m py_compile "$DATALOADER_FILE"

###############################################################################
# 3. ADD 1024x1024 PLANS CONFIGURATION
###############################################################################
export nnUNet_preprocessed="$NNUNET_PREPROCESSED"

python - <<'PY'
import json
import os
from copy import deepcopy
from pathlib import Path

plans_path = (
    Path(os.environ["nnUNet_preprocessed"])
    / "Dataset301_BEETLE"
    / "nnUNetWholeSlideDataPlans.json"
)

with plans_path.open() as f:
    plans = json.load(f)

base = deepcopy(plans["configurations"]["2d"])
context = deepcopy(base)

context["patch_size"] = [1024, 1024]
context["batch_size"] = 2
context["data_identifier"] = base["data_identifier"]

plans["configurations"]["2d_context1024"] = context

with plans_path.open("w") as f:
    json.dump(plans, f, indent=4)

assert plans["configurations"]["2d_context1024"]["patch_size"] == [1024, 1024]
assert plans["configurations"]["2d_context1024"]["batch_size"] == 2

print("Verified 2d_context1024 configuration")
PY

###############################################################################
# 4. ENSURE visual_analysis.py SUPPORTS ARBITRARY FOLDS
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
        raise RuntimeError("Could not patch FOLDS_TO_USE")

    text = text.replace(old, new, 1)

if 'MODEL_PATCH_SIZE = int(os.environ.get("MODEL_PATCH_SIZE", "512"))' not in text:
    old = "MODEL_PATCH_SIZE = 512"
    new = 'MODEL_PATCH_SIZE = int(os.environ.get("MODEL_PATCH_SIZE", "512"))'

    if old not in text:
        raise RuntimeError("Could not patch MODEL_PATCH_SIZE")

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

required = [
    'EVAL_FOLD = int(os.environ.get("EVAL_FOLD", "0"))',
    'FOLDS_TO_USE = (EVAL_FOLD,)',
]

for marker in required:
    if marker not in path.read_text():
        raise RuntimeError(f"Missing required marker after patch: {marker}")

print("Verified fold-aware visual_analysis.py")
PY

python -m py_compile visual_analysis.py

###############################################################################
# 5. BUILD VALIDATION CSV FILES FOR ALL FIVE FOLDS
###############################################################################
for FOLD in 0 1 2 3 4; do
    OUT_CSV="/home/tijnveldwijk/fold${FOLD}_validation_inference_inputs.csv"

    python make_fold_training_inference_csv.py \
        --splits-json "$SPLITS_JSON" \
        --fold "$FOLD" \
        --subset validation \
        --out-csv "$OUT_CSV"

    test -s "$OUT_CSV" || {
        echo "ERROR: missing validation CSV: $OUT_CSV" >&2
        exit 20
    }

    echo "Verified fold-$FOLD validation CSV:"
    wc -l "$OUT_CSV"
done

###############################################################################
# 6. CREATE ROBUST 12-HOUR CHUNK TRAINER
#
# Intermediate chunks:
# - exit 0 only when a resumable latest checkpoint exists;
# - may finish early if checkpoint_final.pth already exists.
#
# Final chunks:
# - exit 0 only when checkpoint_final.pth exists;
# - fail clearly if two chunks were not enough.
###############################################################################
cat > run_cv_cutmixema_chunk_v2.slurm <<'SLURM'
#!/usr/bin/env bash
#SBATCH --account=cseduimc037
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --nodelist=cn48
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=18G
#SBATCH --time=12:00:00
#SBATCH --job-name=cutmixema-v2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
umask 002

ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase

FOLD="${1:?Usage: sbatch run_cv_cutmixema_chunk_v2.slurm <fold> <auto|resume> <intermediate|final>}"
MODE="${2:?Usage: sbatch run_cv_cutmixema_chunk_v2.slurm <fold> <auto|resume> <intermediate|final>}"
ROLE="${3:?Usage: sbatch run_cv_cutmixema_chunk_v2.slurm <fold> <auto|resume> <intermediate|final>}"

TRAINER=nnUNetTrainer_CutMixStainEMA

cd "$ROOT"
mkdir -p logs

source /scratch/$USER/virtual_environments/AIMI-BEETLE/bin/activate

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

export nnUNet_raw=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_raw
export nnUNet_preprocessed=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_preprocessed
export nnUNet_results=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export WANDB_MODE=disabled
export WANDB_DISABLED=true

MODEL_BASE="$nnUNet_results/Dataset301_BEETLE/${TRAINER}__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"
RESULT_DIR="$MODEL_BASE/fold_${FOLD}"

mkdir -p "$RESULT_DIR"

LATEST="$RESULT_DIR/checkpoint_latest.pth"
FINAL="$RESULT_DIR/checkpoint_final.pth"

write_manifest() {
    local LABEL="$1"
    local OUT="$RESULT_DIR/checkpoint_manifest_${LABEL}_${SLURM_JOB_ID:-manual}.txt"

    {
        echo "timestamp=$(date --iso-8601=seconds)"
        echo "fold=$FOLD"
        echo "mode=$MODE"
        echo "role=$ROLE"
        echo "result_dir=$RESULT_DIR"
        echo
        find "$RESULT_DIR" -maxdepth 1 -type f \
            \( -name 'checkpoint*.pth' \
               -o -name 'class_metrics.csv' \
               -o -name 'class_metrics.jsonl' \
               -o -name 'training_log_*.txt' \
            \) \
            -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %f\n' \
            | sort
    } > "$OUT"

    echo "Wrote checkpoint manifest: $OUT"
}

echo "======================================================================"
echo "BEETLE CutMix + stain jitter + EMA training chunk"
echo "Started:      $(date)"
echo "Node:         $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "Fold:         $FOLD"
echo "Mode:         $MODE"
echo "Role:         $ROLE"
echo "Trainer:      $TRAINER"
echo "Result dir:   $RESULT_DIR"
echo "======================================================================"

if [[ -s "$FINAL" ]]; then
    echo "Fold $FOLD already has checkpoint_final.pth."
    write_manifest "already_complete"
    exit 0
fi

case "$MODE" in
    auto)
        if [[ -s "$LATEST" ]]; then
            echo "Existing checkpoint_latest.pth detected. Resuming fold $FOLD."
            TRAIN_CMD=(
                python -u
                nnUNet_pathology/nnunetv2/run/run_training_pathology.py
                301 "$FOLD" "$TRAINER" -c
            )
        else
            echo "No checkpoint found. Starting fold $FOLD fresh."
            TRAIN_CMD=(
                python -u
                nnUNet_pathology/nnunetv2/run/run_training_pathology.py
                301 "$FOLD" "$TRAINER"
            )
        fi
        ;;
    resume)
        test -s "$LATEST" || {
            echo "ERROR: resume requested but checkpoint_latest.pth is missing:" >&2
            echo "  $LATEST" >&2
            exit 31
        }

        TRAIN_CMD=(
            python -u
            nnUNet_pathology/nnunetv2/run/run_training_pathology.py
            301 "$FOLD" "$TRAINER" -c
        )
        ;;
    *)
        echo "ERROR: mode must be auto or resume" >&2
        exit 32
        ;;
esac

case "$ROLE" in
    intermediate|final)
        ;;
    *)
        echo "ERROR: role must be intermediate or final" >&2
        exit 33
        ;;
esac

echo
echo "Executing:"
printf '  %q' "${TRAIN_CMD[@]}"
echo
echo

set +e

timeout \
    --signal=TERM \
    --kill-after=60s \
    42600s \
    "${TRAIN_CMD[@]}"

RC=$?

set -e

echo
echo "Training command exit code: $RC"

write_manifest "finished"

if [[ -s "$FINAL" ]]; then
    echo "SUCCESS: fold $FOLD completed 1000 epochs."
    exit 0
fi

if [[ ! -s "$LATEST" ]]; then
    echo "ERROR: training chunk ended without checkpoint_latest.pth:" >&2
    echo "  $LATEST" >&2
    exit 34
fi

if [[ "$ROLE" == "intermediate" ]]; then
    echo "SUCCESS: fold $FOLD remains resumable after intermediate chunk."
    echo "The queued final chunk will continue from checkpoint_latest.pth."
    exit 0
fi

echo "ERROR: fold $FOLD remains incomplete after its final allocated chunk." >&2
echo "A further resume chunk is required before evaluation." >&2
echo "Latest checkpoint:" >&2
echo "  $LATEST" >&2
exit 35
SLURM

chmod +x run_cv_cutmixema_chunk_v2.slurm

###############################################################################
# 7. CREATE 100-EPOCH CONTEXT-FINETUNING LAUNCHER
###############################################################################
cat > run_cv_context_ft100_v2.slurm <<'SLURM'
#!/usr/bin/env bash
#SBATCH --account=cseduimc037
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --nodelist=cn48
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=18G
#SBATCH --time=12:00:00
#SBATCH --job-name=ctxft-v2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
umask 002

ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase

FOLD="${1:?Usage: sbatch run_cv_context_ft100_v2.slurm <fold>}"

BASE_TRAINER=nnUNetTrainer_CutMixStainEMA
TRAINER=nnUNetTrainer_CutMixStainEMA_Context1024FT100

PLANNER=nnUNetWholeSlideDataPlans
CONFIG=2d_context1024

cd "$ROOT"
mkdir -p logs

source /scratch/$USER/virtual_environments/AIMI-BEETLE/bin/activate

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

export nnUNet_raw=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_raw
export nnUNet_preprocessed=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_preprocessed
export nnUNet_results=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export WANDB_MODE=disabled
export WANDB_DISABLED=true

BASE_MODEL="$nnUNet_results/Dataset301_BEETLE/${BASE_TRAINER}__${PLANNER}__wsd_None_iterator_nnunet_aug__2d"

MODEL_BASE="$nnUNet_results/Dataset301_BEETLE/${TRAINER}__${PLANNER}__wsd_None_iterator_nnunet_aug__${CONFIG}"

PRETRAINED="$BASE_MODEL/fold_${FOLD}/checkpoint_best.pth"
RESULT_DIR="$MODEL_BASE/fold_${FOLD}"

echo "======================================================================"
echo "BEETLE CutMix + EMA larger-context refinement"
echo "Started:               $(date)"
echo "Node:                  $(hostname)"
echo "SLURM_JOB_ID:          ${SLURM_JOB_ID:-none}"
echo "Fold:                  $FOLD"
echo "Trainer:               $TRAINER"
echo "Configuration:         $CONFIG"
echo "Patch size:            1024 x 1024"
echo "WSD batch size:        2"
echo "Additional epochs:     100"
echo "Initial LR:            0.0005"
echo "Pretrained checkpoint: $PRETRAINED"
echo "Result dir:            $RESULT_DIR"
echo "======================================================================"

test -s "$PRETRAINED" || {
    echo "ERROR: pretrained checkpoint missing:" >&2
    echo "  $PRETRAINED" >&2
    exit 40
}

if [[ -s "$RESULT_DIR/checkpoint_final.pth" ]]; then
    echo "Context fold $FOLD already completed."
    exit 0
fi

if [[ -e "$RESULT_DIR/checkpoint_latest.pth" || \
      -e "$RESULT_DIR/checkpoint_best.pth" ]]; then
    echo "ERROR: partial context output already exists:" >&2
    echo "  $RESULT_DIR" >&2
    echo "Inspect deliberately before rerunning." >&2
    exit 41
fi

mkdir -p "$RESULT_DIR"

set +e

timeout \
    --signal=TERM \
    --kill-after=60s \
    42600s \
    python -u - "$FOLD" "$TRAINER" "$PLANNER" "$CONFIG" "$PRETRAINED" <<'PY'
import os
import sys
from os.path import join

import torch
import nnunetv2

from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

fold = int(sys.argv[1])
trainer_name = sys.argv[2]
plans_identifier = sys.argv[3]
configuration = sys.argv[4]
pretrained_path = sys.argv[5]

trainer_class = recursive_find_python_class(
    join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
    trainer_name,
    "nnunetv2.training.nnUNetTrainer",
)

if trainer_class is None:
    raise RuntimeError(f"Could not locate trainer class: {trainer_name}")

dataset_folder = join(
    os.environ["nnUNet_preprocessed"],
    "Dataset301_BEETLE",
)

plans = load_json(join(dataset_folder, plans_identifier + ".json"))
dataset_json = load_json(join(dataset_folder, "dataset.json"))

trainer = trainer_class(
    plans=plans,
    configuration=configuration,
    fold=fold,
    dataset_json=dataset_json,
    unpack_dataset=True,
    device=torch.device("cuda"),
)

trainer.initialize()

checkpoint = torch.load(
    pretrained_path,
    map_location=torch.device("cpu"),
)

if "network_weights" not in checkpoint:
    raise RuntimeError(
        "Expected network_weights in checkpoint. "
        f"Available keys: {sorted(checkpoint.keys())}"
    )

network = trainer.network

if hasattr(network, "_orig_mod"):
    network = network._orig_mod

result = network.load_state_dict(
    checkpoint["network_weights"],
    strict=True,
)

print("Loaded pretrained network parameters")
print("Missing keys:", result.missing_keys)
print("Unexpected keys:", result.unexpected_keys)

ema_synced = False

for attr_name in (
    "ema_model",
    "network_ema",
    "ema_network",
    "model_ema",
    "_ema_model",
    "ema",
):
    candidate = getattr(trainer, attr_name, None)

    if candidate is None:
        continue

    target = getattr(candidate, "module", candidate)
    target = getattr(target, "model", target)

    if hasattr(target, "_orig_mod"):
        target = target._orig_mod

    if hasattr(target, "load_state_dict"):
        target.load_state_dict(network.state_dict(), strict=True)
        print(f"Synchronized EMA state through trainer.{attr_name}")
        ema_synced = True
        break

if not ema_synced:
    print(
        "WARNING: no directly loadable EMA object was found. "
        "Inspect the initial log lines carefully."
    )

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

trainer.run_training()
PY

RC=$?

set -e

echo "Context training exit code: $RC"

test -s "$MODEL_BASE/dataset.json" || {
    echo "ERROR: context training returned without dataset.json:" >&2
    echo "  $MODEL_BASE/dataset.json" >&2
    exit 42
}

test -s "$RESULT_DIR/checkpoint_final.pth" || {
    echo "ERROR: context training returned without checkpoint_final.pth:" >&2
    echo "  $RESULT_DIR/checkpoint_final.pth" >&2
    exit 43
}

echo "Verified context fold $FOLD outputs."

find "$RESULT_DIR" -maxdepth 1 -type f \
    \( -name 'checkpoint*.pth' \
       -o -name 'class_metrics.csv' \
       -o -name 'class_metrics.jsonl' \
    \) \
    -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %f\n' \
    | sort \
    > "$RESULT_DIR/checkpoint_manifest_context1024_ft100.txt"

echo "Finished: $(date)"
SLURM

chmod +x run_cv_context_ft100_v2.slurm

###############################################################################
# 8. CREATE FOLD-AWARE FULL MIRRORED EVALUATION LAUNCHER
###############################################################################
cat > run_cv_full_eval_v2.slurm <<'SLURM'
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
#SBATCH --job-name=eval-v2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
umask 002

ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase

STAGE="${1:?Usage: sbatch run_cv_full_eval_v2.slurm <base|context> <fold>}"
FOLD="${2:?Usage: sbatch run_cv_full_eval_v2.slurm <base|context> <fold>}"

cd "$ROOT"
mkdir -p logs

source /scratch/$USER/virtual_environments/AIMI-BEETLE/bin/activate

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

export nnUNet_raw=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_raw
export nnUNet_preprocessed=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_preprocessed
export nnUNet_results=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results

case "$STAGE" in
    base)
        MODEL_BASE_PATH="$nnUNet_results/Dataset301_BEETLE/nnUNetTrainer_CutMixStainEMA__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"
        CHECKPOINT_TAG="cutmixema1000_best_mirror_visual"
        MODEL_PATCH_SIZE=512
        VIS_OUT_DIR_BASE="$nnUNet_results/../validation_visuals/cv_cutmixema1000"
        ;;
    context)
        MODEL_BASE_PATH="$nnUNet_results/Dataset301_BEETLE/nnUNetTrainer_CutMixStainEMA_Context1024FT100__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d_context1024"
        CHECKPOINT_TAG="cutmixema1000_context1024_ft100_best_mirror_visual"
        MODEL_PATCH_SIZE=1024
        VIS_OUT_DIR_BASE="$nnUNet_results/../validation_visuals/cv_cutmixema1000_context1024_ft100"
        ;;
    *)
        echo "ERROR: stage must be base or context" >&2
        exit 50
        ;;
esac

CSV_PATH="/home/tijnveldwijk/fold${FOLD}_validation_inference_inputs.csv"
VIS_OUT_DIR="$VIS_OUT_DIR_BASE/fold_${FOLD}/$CHECKPOINT_TAG"

CHECKPOINT="$MODEL_BASE_PATH/fold_${FOLD}/checkpoint_best.pth"

test -s "$CSV_PATH" || {
    echo "ERROR: missing validation CSV:" >&2
    echo "  $CSV_PATH" >&2
    exit 51
}

test -s "$CHECKPOINT" || {
    echo "ERROR: missing checkpoint:" >&2
    echo "  $CHECKPOINT" >&2
    exit 52
}

export MODEL_BASE_PATH
export CSV_PATH
export EVAL_FOLD="$FOLD"
export CHECKPOINT_NAME=checkpoint_best.pth
export CHECKPOINT_TAG
export SAVE_VISUALS=1
export USE_MIRRORING=1
export MODEL_PATCH_SIZE
export VIS_OUT_DIR

echo "======================================================================"
echo "BEETLE full mirrored validation"
echo "Started:          $(date)"
echo "Node:             $(hostname)"
echo "SLURM_JOB_ID:     ${SLURM_JOB_ID:-none}"
echo "Stage:            $STAGE"
echo "Fold:             $FOLD"
echo "MODEL_BASE_PATH:  $MODEL_BASE_PATH"
echo "CSV_PATH:         $CSV_PATH"
echo "CHECKPOINT_TAG:   $CHECKPOINT_TAG"
echo "MODEL_PATCH_SIZE: $MODEL_PATCH_SIZE"
echo "USE_MIRRORING:    $USE_MIRRORING"
echo "VIS_OUT_DIR:      $VIS_OUT_DIR"
echo "======================================================================"

python -u visual_analysis.py

CM_CSV="$MODEL_BASE_PATH/fold_${FOLD}/fold${FOLD}_${CHECKPOINT_TAG}_confusion_matrix_rows_gt_cols_pred.csv"

test -s "$CM_CSV" || {
    echo "ERROR: evaluation returned without confusion matrix CSV:" >&2
    echo "  $CM_CSV" >&2
    exit 53
}

echo "Verified evaluation output:"
echo "  $CM_CSV"

echo "Finished: $(date)"
SLURM

chmod +x run_cv_full_eval_v2.slurm

###############################################################################
# 9. CREATE ROBUST FIVE-FOLD AGGREGATOR
#
# Metrics are recomputed directly from the confusion matrices, avoiding
# assumptions about the internal JSON schema.
###############################################################################
cat > aggregate_cv_results_v2.py <<'PY'
#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

LABELS = {
    1: "other",
    2: "non-invasive epithelium",
    3: "invasive epithelium",
    4: "necrosis",
}

ROOT = Path(
    "/vol/csedu-nobackup/course/IMC037_aimi/group14/"
    "nnunet/tijn/pathology/nnUNet_results/Dataset301_BEETLE"
)

OUT = Path(
    "/home/tijnveldwijk/AIMI---BEETLE-Project-Phase/"
    "cv_summaries/cutmixema1000_context1024"
)


def stage_config(stage: str) -> tuple[Path, str]:
    if stage == "base":
        return (
            ROOT
            / "nnUNetTrainer_CutMixStainEMA__"
              "nnUNetWholeSlideDataPlans__"
              "wsd_None_iterator_nnunet_aug__2d",
            "cutmixema1000_best_mirror_visual",
        )

    if stage == "context":
        return (
            ROOT
            / "nnUNetTrainer_CutMixStainEMA_Context1024FT100__"
              "nnUNetWholeSlideDataPlans__"
              "wsd_None_iterator_nnunet_aug__2d_context1024",
            "cutmixema1000_context1024_ft100_best_mirror_visual",
        )

    raise ValueError("stage must be base or context")


def read_cm(path: Path) -> list[list[int]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(newline="") as f:
        rows = [
            [int(float(value)) for value in row]
            for row in csv.reader(f)
            if row
        ]

    if len(rows) != 5 or any(len(row) != 5 for row in rows):
        raise ValueError(f"Expected a 5x5 confusion matrix: {path}")

    return rows


def add_cm(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    return [
        [x + y for x, y in zip(row_a, row_b)]
        for row_a, row_b in zip(a, b)
    ]


def dice_by_class(cm: list[list[int]]) -> dict[str, float]:
    result: dict[str, float] = {}

    for label, name in LABELS.items():
        tp = cm[label][label]
        fp = sum(cm[row][label] for row in LABELS if row != label)
        fn = sum(cm[label][col] for col in LABELS if col != label)

        denom = 2 * tp + fp + fn
        result[name] = 2 * tp / denom if denom else float("nan")

    return result


def micro_dice(cm: list[list[int]]) -> float:
    tp = sum(cm[label][label] for label in LABELS)

    fp = sum(
        cm[row][label]
        for label in LABELS
        for row in LABELS
        if row != label
    )

    fn = sum(
        cm[label][col]
        for label in LABELS
        for col in LABELS
        if col != label
    )

    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else float("nan")


def macro_dice(dices: dict[str, float]) -> float:
    values = [value for value in dices.values() if not math.isnan(value)]
    return statistics.mean(values)


def confusion_rate(cm: list[list[int]], source: int, target: int) -> float:
    denom = sum(cm[source][col] for col in LABELS)
    return cm[source][target] / denom if denom else float("nan")


def write_cm(path: Path, cm: list[list[int]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(cm)


def summarise(stage: str) -> dict[str, Any]:
    model_dir, tag = stage_config(stage)

    folds: list[dict[str, Any]] = []
    pooled = [[0 for _ in range(5)] for _ in range(5)]

    for fold in range(5):
        cm_path = (
            model_dir
            / f"fold_{fold}"
            / f"fold{fold}_{tag}_confusion_matrix_rows_gt_cols_pred.csv"
        )

        cm = read_cm(cm_path)
        pooled = add_cm(pooled, cm)

        dices = dice_by_class(cm)

        folds.append(
            {
                "fold": fold,
                "checkpoint": str(
                    model_dir / f"fold_{fold}" / "checkpoint_best.pth"
                ),
                "confusion_matrix_csv": str(cm_path),
                "class_dices": dices,
                "macro_dice": macro_dice(dices),
                "micro_dice": micro_dice(cm),
                "non_invasive_to_invasive": confusion_rate(cm, 2, 3),
                "invasive_to_non_invasive": confusion_rate(cm, 3, 2),
            }
        )

    pooled_dices = dice_by_class(pooled)

    summary = {
        "stage": stage,
        "model_dir": str(model_dir),
        "checkpoint_tag": tag,
        "folds": folds,
        "fold_mean_std": {},
        "pooled": {
            "class_dices": pooled_dices,
            "macro_dice": macro_dice(pooled_dices),
            "micro_dice": micro_dice(pooled),
            "non_invasive_to_invasive": confusion_rate(pooled, 2, 3),
            "invasive_to_non_invasive": confusion_rate(pooled, 3, 2),
            "confusion_matrix_rows_gt_cols_prediction": pooled,
        },
    }

    metrics = [
        ("other", lambda fold: fold["class_dices"]["other"]),
        (
            "non-invasive epithelium",
            lambda fold: fold["class_dices"]["non-invasive epithelium"],
        ),
        (
            "invasive epithelium",
            lambda fold: fold["class_dices"]["invasive epithelium"],
        ),
        ("necrosis", lambda fold: fold["class_dices"]["necrosis"]),
        ("macro_dice", lambda fold: fold["macro_dice"]),
        ("micro_dice", lambda fold: fold["micro_dice"]),
        (
            "non_invasive_to_invasive",
            lambda fold: fold["non_invasive_to_invasive"],
        ),
        (
            "invasive_to_non_invasive",
            lambda fold: fold["invasive_to_non_invasive"],
        ),
    ]

    for name, getter in metrics:
        values = [getter(fold) for fold in folds]

        summary["fold_mean_std"][name] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values),
            "values": values,
        }

    return summary


def save_stage(summary: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    stage = summary["stage"]

    json_path = OUT / f"cv_summary_{stage}.json"
    csv_path = OUT / f"cv_summary_{stage}.csv"
    cm_path = OUT / f"cv_pooled_confusion_matrix_{stage}.csv"
    manifest_path = OUT / f"ensemble_manifest_{stage}.txt"

    with json_path.open("w") as f:
        json.dump(summary, f, indent=4)

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "fold",
                "other",
                "non_invasive_epithelium",
                "invasive_epithelium",
                "necrosis",
                "macro_dice",
                "micro_dice",
                "non_invasive_to_invasive",
                "invasive_to_non_invasive",
            ]
        )

        for fold in summary["folds"]:
            writer.writerow(
                [
                    fold["fold"],
                    fold["class_dices"]["other"],
                    fold["class_dices"]["non-invasive epithelium"],
                    fold["class_dices"]["invasive epithelium"],
                    fold["class_dices"]["necrosis"],
                    fold["macro_dice"],
                    fold["micro_dice"],
                    fold["non_invasive_to_invasive"],
                    fold["invasive_to_non_invasive"],
                ]
            )

        pooled = summary["pooled"]

        writer.writerow(
            [
                "pooled",
                pooled["class_dices"]["other"],
                pooled["class_dices"]["non-invasive epithelium"],
                pooled["class_dices"]["invasive epithelium"],
                pooled["class_dices"]["necrosis"],
                pooled["macro_dice"],
                pooled["micro_dice"],
                pooled["non_invasive_to_invasive"],
                pooled["invasive_to_non_invasive"],
            ]
        )

    write_cm(cm_path, summary["pooled"]["confusion_matrix_rows_gt_cols_prediction"])

    with manifest_path.open("w") as f:
        for fold in summary["folds"]:
            f.write(f"fold_{fold['fold']}={fold['checkpoint']}\n")

    print("Saved:", json_path)
    print("Saved:", csv_path)
    print("Saved:", cm_path)
    print("Saved:", manifest_path)


def save_comparison(base: dict[str, Any], context: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    base_pooled = base["pooled"]
    context_pooled = context["pooled"]

    rows = []

    for name in LABELS.values():
        rows.append(
            (
                f"dice_{name}",
                base_pooled["class_dices"][name],
                context_pooled["class_dices"][name],
            )
        )

    rows.extend(
        [
            (
                "macro_dice",
                base_pooled["macro_dice"],
                context_pooled["macro_dice"],
            ),
            (
                "micro_dice",
                base_pooled["micro_dice"],
                context_pooled["micro_dice"],
            ),
            (
                "non_invasive_to_invasive",
                base_pooled["non_invasive_to_invasive"],
                context_pooled["non_invasive_to_invasive"],
            ),
            (
                "invasive_to_non_invasive",
                base_pooled["invasive_to_non_invasive"],
                context_pooled["invasive_to_non_invasive"],
            ),
        ]
    )

    comparison = {
        "base": base_pooled,
        "context": context_pooled,
        "context_minus_base": {
            name: context_value - base_value
            for name, base_value, context_value in rows
        },
    }

    json_path = OUT / "cv_comparison_context_minus_base.json"
    csv_path = OUT / "cv_comparison_context_minus_base.csv"

    with json_path.open("w") as f:
        json.dump(comparison, f, indent=4)

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "metric",
                "base",
                "context",
                "context_minus_base",
            ]
        )

        for name, base_value, context_value in rows:
            writer.writerow(
                [
                    name,
                    base_value,
                    context_value,
                    context_value - base_value,
                ]
            )

    print("Saved:", json_path)
    print("Saved:", csv_path)


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"base", "context", "both"}:
        raise SystemExit(
            "Usage: python aggregate_cv_results_v2.py <base|context|both>"
        )

    requested = sys.argv[1]

    if requested == "base":
        base = summarise("base")
        save_stage(base)
        return

    if requested == "context":
        context = summarise("context")
        save_stage(context)
        return

    base = summarise("base")
    context = summarise("context")

    save_stage(base)
    save_stage(context)
    save_comparison(base, context)


if __name__ == "__main__":
    main()
PY

chmod +x aggregate_cv_results_v2.py

###############################################################################
# 10. CREATE CPU-ONLY AGGREGATION LAUNCHER
###############################################################################
cat > run_cv_aggregate_v2.slurm <<'SLURM'
#!/usr/bin/env bash
#SBATCH --account=cseduimc037
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --job-name=agg-v2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
umask 002

ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase

STAGE="${1:?Usage: sbatch run_cv_aggregate_v2.slurm <base|both>}"

cd "$ROOT"

source /scratch/$USER/virtual_environments/AIMI-BEETLE/bin/activate

python -u aggregate_cv_results_v2.py "$STAGE"

echo "Finished: $(date)"
SLURM

chmod +x run_cv_aggregate_v2.slurm

###############################################################################
# 11. FINAL STATIC VALIDATION BEFORE SUBMISSION
###############################################################################
echo
echo "======================================================================"
echo "STATIC VALIDATION"
echo "======================================================================"

grep -n "42600s" run_cv_cutmixema_chunk_v2.slurm
grep -n "42600s" run_cv_context_ft100_v2.slurm
grep -n "USE_MIRRORING=1" run_cv_full_eval_v2.slurm

python -m py_compile aggregate_cv_results_v2.py
python -m py_compile visual_analysis.py

echo
echo "Verified launchers and Python files."

###############################################################################
# 12. QUEUE THE PIPELINE WITH TWO EXPLICIT GPU LANES
#
# Fold 0:
#   it already has a checkpoint from the cancelled run;
#   queue two resume chunks, where the second exits immediately if unnecessary.
#
# Lane A:
#   fold 0 resume 1 -> fold 0 resume 2 -> fold 2 -> fold 4
#
# Lane B:
#   fold 1 -> fold 3
###############################################################################
join_colon() {
    local IFS=:
    echo "$*"
}

F0_R1=$(
    sbatch --parsable \
        --job-name=cutmixema-v2-f0-r1 \
        run_cv_cutmixema_chunk_v2.slurm \
        0 resume intermediate
)

F0_R2=$(
    sbatch --parsable \
        --dependency=afterok:${F0_R1} \
        --job-name=cutmixema-v2-f0-r2 \
        run_cv_cutmixema_chunk_v2.slurm \
        0 resume final
)

F1_C1=$(
    sbatch --parsable \
        --job-name=cutmixema-v2-f1-c1 \
        run_cv_cutmixema_chunk_v2.slurm \
        1 auto intermediate
)

F1_C2=$(
    sbatch --parsable \
        --dependency=afterok:${F1_C1} \
        --job-name=cutmixema-v2-f1-c2 \
        run_cv_cutmixema_chunk_v2.slurm \
        1 resume final
)

F2_C1=$(
    sbatch --parsable \
        --dependency=afterok:${F0_R2} \
        --job-name=cutmixema-v2-f2-c1 \
        run_cv_cutmixema_chunk_v2.slurm \
        2 auto intermediate
)

F2_C2=$(
    sbatch --parsable \
        --dependency=afterok:${F2_C1} \
        --job-name=cutmixema-v2-f2-c2 \
        run_cv_cutmixema_chunk_v2.slurm \
        2 resume final
)

F3_C1=$(
    sbatch --parsable \
        --dependency=afterok:${F1_C2} \
        --job-name=cutmixema-v2-f3-c1 \
        run_cv_cutmixema_chunk_v2.slurm \
        3 auto intermediate
)

F3_C2=$(
    sbatch --parsable \
        --dependency=afterok:${F3_C1} \
        --job-name=cutmixema-v2-f3-c2 \
        run_cv_cutmixema_chunk_v2.slurm \
        3 resume final
)

F4_C1=$(
    sbatch --parsable \
        --dependency=afterok:${F2_C2} \
        --job-name=cutmixema-v2-f4-c1 \
        run_cv_cutmixema_chunk_v2.slurm \
        4 auto intermediate
)

F4_C2=$(
    sbatch --parsable \
        --dependency=afterok:${F4_C1} \
        --job-name=cutmixema-v2-f4-c2 \
        run_cv_cutmixema_chunk_v2.slurm \
        4 resume final
)

ALL_BASE_FINALS=$(join_colon "$F0_R2" "$F1_C2" "$F2_C2" "$F3_C2" "$F4_C2")

###############################################################################
# 13. BASE EVALUATIONS: TWO GPU LANES
###############################################################################
E0=$(
    sbatch --parsable \
        --dependency=afterok:${ALL_BASE_FINALS} \
        --job-name=eval-cema-v2-f0 \
        run_cv_full_eval_v2.slurm \
        base 0
)

E1=$(
    sbatch --parsable \
        --dependency=afterok:${ALL_BASE_FINALS} \
        --job-name=eval-cema-v2-f1 \
        run_cv_full_eval_v2.slurm \
        base 1
)

E2=$(
    sbatch --parsable \
        --dependency=afterok:${E0} \
        --job-name=eval-cema-v2-f2 \
        run_cv_full_eval_v2.slurm \
        base 2
)

E3=$(
    sbatch --parsable \
        --dependency=afterok:${E1} \
        --job-name=eval-cema-v2-f3 \
        run_cv_full_eval_v2.slurm \
        base 3
)

E4=$(
    sbatch --parsable \
        --dependency=afterok:${E2} \
        --job-name=eval-cema-v2-f4 \
        run_cv_full_eval_v2.slurm \
        base 4
)

ALL_BASE_EVALS=$(join_colon "$E0" "$E1" "$E2" "$E3" "$E4")

BASE_AGG=$(
    sbatch --parsable \
        --dependency=afterok:${ALL_BASE_EVALS} \
        --job-name=agg-cema-v2 \
        run_cv_aggregate_v2.slurm \
        base
)

###############################################################################
# 14. CONTEXT FINE-TUNES: TWO GPU LANES
###############################################################################
T0=$(
    sbatch --parsable \
        --dependency=afterok:${BASE_AGG} \
        --job-name=ctxft-v2-f0 \
        run_cv_context_ft100_v2.slurm \
        0
)

T1=$(
    sbatch --parsable \
        --dependency=afterok:${BASE_AGG} \
        --job-name=ctxft-v2-f1 \
        run_cv_context_ft100_v2.slurm \
        1
)

T2=$(
    sbatch --parsable \
        --dependency=afterok:${T0} \
        --job-name=ctxft-v2-f2 \
        run_cv_context_ft100_v2.slurm \
        2
)

T3=$(
    sbatch --parsable \
        --dependency=afterok:${T1} \
        --job-name=ctxft-v2-f3 \
        run_cv_context_ft100_v2.slurm \
        3
)

T4=$(
    sbatch --parsable \
        --dependency=afterok:${T2} \
        --job-name=ctxft-v2-f4 \
        run_cv_context_ft100_v2.slurm \
        4
)

ALL_CONTEXT_TRAINS=$(join_colon "$T0" "$T1" "$T2" "$T3" "$T4")

###############################################################################
# 15. CONTEXT EVALUATIONS: TWO GPU LANES
###############################################################################
CE0=$(
    sbatch --parsable \
        --dependency=afterok:${ALL_CONTEXT_TRAINS} \
        --job-name=eval-ctx-v2-f0 \
        run_cv_full_eval_v2.slurm \
        context 0
)

CE1=$(
    sbatch --parsable \
        --dependency=afterok:${ALL_CONTEXT_TRAINS} \
        --job-name=eval-ctx-v2-f1 \
        run_cv_full_eval_v2.slurm \
        context 1
)

CE2=$(
    sbatch --parsable \
        --dependency=afterok:${CE0} \
        --job-name=eval-ctx-v2-f2 \
        run_cv_full_eval_v2.slurm \
        context 2
)

CE3=$(
    sbatch --parsable \
        --dependency=afterok:${CE1} \
        --job-name=eval-ctx-v2-f3 \
        run_cv_full_eval_v2.slurm \
        context 3
)

CE4=$(
    sbatch --parsable \
        --dependency=afterok:${CE2} \
        --job-name=eval-ctx-v2-f4 \
        run_cv_full_eval_v2.slurm \
        context 4
)

ALL_CONTEXT_EVALS=$(join_colon "$CE0" "$CE1" "$CE2" "$CE3" "$CE4")

FINAL_AGG=$(
    sbatch --parsable \
        --dependency=afterok:${ALL_CONTEXT_EVALS} \
        --job-name=agg-final-v2 \
        run_cv_aggregate_v2.slurm \
        both
)

###############################################################################
# 16. WRITE MANIFEST
###############################################################################
MANIFEST=queued_cutmixema_context_cv_v2.txt

cat > "$MANIFEST" <<EOF
Submitted: $(date --iso-8601=seconds)

Stage 1: CutMix + stain jitter + EMA, 1000 epochs
  fold 0 resume chunk 1: $F0_R1
  fold 0 resume chunk 2: $F0_R2
  fold 1 chunk 1:        $F1_C1
  fold 1 chunk 2:        $F1_C2
  fold 2 chunk 1:        $F2_C1
  fold 2 chunk 2:        $F2_C2
  fold 3 chunk 1:        $F3_C1
  fold 3 chunk 2:        $F3_C2
  fold 4 chunk 1:        $F4_C1
  fold 4 chunk 2:        $F4_C2

Stage 2: mirrored base evaluations
  fold 0: $E0
  fold 1: $E1
  fold 2: $E2
  fold 3: $E3
  fold 4: $E4
  aggregate: $BASE_AGG

Stage 3: 1024x1024 context fine-tuning
  fold 0: $T0
  fold 1: $T1
  fold 2: $T2
  fold 3: $T3
  fold 4: $T4

Stage 4: mirrored context evaluations
  fold 0: $CE0
  fold 1: $CE1
  fold 2: $CE2
  fold 3: $CE3
  fold 4: $CE4

Stage 5: final aggregation
  $FINAL_AGG
EOF

echo
echo "======================================================================"
echo "CLEAN TWO-GPU PIPELINE QUEUED"
echo "======================================================================"
echo
echo "Stage-1 GPU lane A:"
echo "  fold 0 resume 1 -> fold 0 resume 2 -> fold 2 -> fold 4"
echo
echo "Stage-1 GPU lane B:"
echo "  fold 1 -> fold 3"
echo
echo "Later training and evaluation stages are also capped at two GPUs."
echo
echo "Manifest:"
echo "  $ROOT/$MANIFEST"
echo

squeue -u "$USER" \
    -o "%.18i %.32j %.2t %.12M %.45R"
