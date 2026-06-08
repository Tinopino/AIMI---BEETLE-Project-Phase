#!/usr/bin/env bash
set -euo pipefail
umask 002

###############################################################################
# USER CONFIGURATION
#
# Fold 0 chunk 1 is already running. Change this only if its job ID changed.
###############################################################################
FOLD0_ACTIVE_JOB_ID="${FOLD0_ACTIVE_JOB_ID:-10406217}"

###############################################################################
# FIXED PROJECT PATHS
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

cd "$ROOT"
mkdir -p logs cv_summaries

###############################################################################
# 1. PATCH THE PATHOLOGY WSD DATALOADER:
#    - existing models still default to WSD batch size 8;
#    - context trainer can explicitly override this to batch size 2.
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
            "Could not locate the hardcoded WSD batch-size block. "
            "Inspect nnUNetTrainer_WSD_undefined_dataloader.py before continuing."
        )

    path.write_text(text.replace(old, new, 1))
    print("Installed optional WSD batch-size override")
PY

###############################################################################
# 2. ADD THE CUTMIX + EMA CONTEXT-FINETUNING TRAINER ONCE.
#
#    This inherits the existing CutMix, stain jitter, EMA, weighted focal loss,
#    class-metric logging and class-specific checkpoint behavior.
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
    print("Context fine-tuning trainer already installed")
else:
    addition = r'''

# =============================================================================
# CUTMIX + STAIN JITTER + EMA + 1024x1024 CONTEXT FINE-TUNING
# =============================================================================

class nnUNetTrainer_CutMixStainEMA_Context1024FT100(
    nnUNetTrainer_CutMixStainEMA
):
    """
    Final BEETLE context-refinement trainer.

    Inherits:
    - alpha-weighted Dice + focal loss;
    - CutMix;
    - stain jitter;
    - EMA weight tracking;
    - normal checkpoint_best.pth behavior;
    - class-specific best checkpoint saving;
    - class_metrics.csv and class_metrics.jsonl.

    Changes:
    - uses configuration 2d_context1024;
    - receives 1024x1024 patches;
    - uses WSD batch size 2;
    - uses a lower LR for refinement;
    - trains for 100 additional epochs.
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

        actual_patch_size = list(self.configuration_manager.patch_size)
        expected_patch_size = [1024, 1024]

        if actual_patch_size != expected_patch_size:
            raise RuntimeError(
                "Context trainer requires patch_size="
                f"{expected_patch_size}, received {actual_patch_size}. "
                "Use configuration='2d_context1024'."
            )

        self.wsd_batch_size_override = 2
        self.initial_lr = 5e-4
        self.num_epochs = 100

        # Save resumable checkpoints frequently during this short refinement.
        self.save_every = 1

        self.print_to_log_file(
            "Using CutMix + stain jitter + EMA context fine-tuning: "
            "patch_size=[1024, 1024], WSD batch_size=2, "
            "initial_lr=0.0005, num_epochs=100",
            also_print_to_console=True,
        )
'''

    path.write_text(text + addition)
    print("Added:", marker)
PY

python -m py_compile \
  nnUNet_pathology/nnunetv2/training/nnUNetTrainer/nnUnetTrainerBEETLE.py

python -m py_compile \
  nnUNet_pathology/nnunetv2/training/nnUNetTrainer/variants/pathology/nnUNetTrainer_WSD_undefined_dataloader.py

###############################################################################
# 3. ADD A 1024x1024 PLANS CONFIGURATION ONCE.
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

# WholeSlideData samples WSI patches online. Reuse the existing identifier.
context["data_identifier"] = base["data_identifier"]

plans["configurations"]["2d_context1024"] = context

with plans_path.open("w") as f:
    json.dump(plans, f, indent=4)

print("Configured 2d_context1024:")
print(json.dumps(plans["configurations"]["2d_context1024"], indent=4))
PY

###############################################################################
# 4. PATCH visual_analysis.py SO IT SUPPORTS ARBITRARY FOLDS.
#
#    Existing fold-0 behavior remains the default.
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
        raise RuntimeError("Could not locate FOLDS_TO_USE = (0,)")

    text = text.replace(old, new, 1)

if 'MODEL_PATCH_SIZE = int(os.environ.get("MODEL_PATCH_SIZE", "512"))' not in text:
    old = "MODEL_PATCH_SIZE = 512"
    new = 'MODEL_PATCH_SIZE = int(os.environ.get("MODEL_PATCH_SIZE", "512"))'

    if old not in text:
        raise RuntimeError("Could not locate MODEL_PATCH_SIZE = 512")

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
print("Patched visual_analysis.py for fold-specific evaluation")
PY

python -m py_compile visual_analysis.py

###############################################################################
# 5. CREATE THE VALIDATION CSV FOR EACH FOLD.
###############################################################################
for FOLD in 0 1 2 3 4; do
    OUT_CSV="/home/tijnveldwijk/fold${FOLD}_validation_inference_inputs.csv"

    python make_fold_training_inference_csv.py \
      --splits-json "$SPLITS_JSON" \
      --fold "$FOLD" \
      --subset validation \
      --out-csv "$OUT_CSV"

    test -s "$OUT_CSV" || {
        echo "ERROR: validation CSV missing or empty: $OUT_CSV" >&2
        exit 10
    }
done

###############################################################################
# 6. CREATE THE 1000-EPOCH CUTMIX + EMA CHUNK LAUNCHER.
#
#    Each chunk:
#    - has a 12-hour SLURM limit;
#    - gives Python 11h50m so the shell can exit cleanly;
#    - uses checkpoint_latest.pth for resume;
#    - preserves the inherited general-best, per-class-best and final checkpoints;
#    - writes a manifest of currently saved checkpoints.
###############################################################################
cat > run_cv_cutmixema1000_chunk.slurm <<'SLURM'
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
#SBATCH --signal=B:TERM@60
#SBATCH --job-name=beetle_cema
#SBATCH --output=logs/cutmixema1000-fold-%a-%j.out
#SBATCH --error=logs/cutmixema1000-fold-%a-%j.err

set -euo pipefail
umask 002

ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase

FOLD="${1:?Usage: sbatch run_cv_cutmixema1000_chunk.slurm <fold> <fresh|resume> <chunk_tag>}"
MODE="${2:?Usage: sbatch run_cv_cutmixema1000_chunk.slurm <fold> <fresh|resume> <chunk_tag>}"
CHUNK_TAG="${3:?Usage: sbatch run_cv_cutmixema1000_chunk.slurm <fold> <fresh|resume> <chunk_tag>}"

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

write_checkpoint_manifest() {
    local LABEL="$1"
    local OUT="$RESULT_DIR/checkpoint_manifest_${LABEL}.txt"

    {
        echo "timestamp=$(date --iso-8601=seconds)"
        echo "fold=$FOLD"
        echo "mode=$MODE"
        echo "chunk_tag=$CHUNK_TAG"
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
echo "BEETLE 1000-epoch CutMix + stain jitter + EMA chunk"
echo "Started:      $(date)"
echo "Node:         $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "Fold:         $FOLD"
echo "Mode:         $MODE"
echo "Chunk tag:    $CHUNK_TAG"
echo "Trainer:      $TRAINER"
echo "Result dir:   $RESULT_DIR"
echo "======================================================================"

if [[ -s "$RESULT_DIR/checkpoint_final.pth" ]]; then
    echo "Fold $FOLD already completed 1000 epochs."
    write_checkpoint_manifest "${CHUNK_TAG}_already_complete"
    exit 0
fi

case "$MODE" in
    fresh)
        if [[ -s "$RESULT_DIR/checkpoint_latest.pth" ]]; then
            echo "ERROR: fresh run requested but checkpoint_latest.pth already exists:" >&2
            echo "  $RESULT_DIR/checkpoint_latest.pth" >&2
            exit 11
        fi
        TRAIN_CMD=(
            python -u
            nnUNet_pathology/nnunetv2/run/run_training_pathology.py
            301 "$FOLD" "$TRAINER"
        )
        ;;
    resume)
        if [[ ! -s "$RESULT_DIR/checkpoint_latest.pth" ]]; then
            echo "ERROR: resume requested but checkpoint_latest.pth is missing:" >&2
            echo "  $RESULT_DIR/checkpoint_latest.pth" >&2
            echo "A previous chunk may have failed before saving a resumable checkpoint." >&2
            exit 12
        fi
        TRAIN_CMD=(
            python -u
            nnUNet_pathology/nnunetv2/run/run_training_pathology.py
            301 "$FOLD" "$TRAINER" -c
        )
        ;;
    *)
        echo "ERROR: mode must be fresh or resume" >&2
        exit 13
        ;;
esac

set +e
timeout \
  --signal=TERM \
  --kill-after=60s \
  11h50m \
  "${TRAIN_CMD[@]}"

RC=$?
set -e

write_checkpoint_manifest "${CHUNK_TAG}_finished"

if [[ -s "$RESULT_DIR/checkpoint_final.pth" ]]; then
    echo "Fold $FOLD completed successfully."
    exit 0
fi

echo
echo "Fold $FOLD has not yet reached checkpoint_final.pth."
echo "Chunk exit code: $RC"
echo "A resume chunk is expected to continue from checkpoint_latest.pth."
echo

# A non-zero status is intentional when the wall-time chunk ends before epoch 1000.
exit 99
SLURM

chmod +x run_cv_cutmixema1000_chunk.slurm

###############################################################################
# 7. CREATE THE 100-EPOCH CONTEXT-FINETUNING LAUNCHER.
###############################################################################
cat > run_cv_cutmixema_context1024_ft100.slurm <<'SLURM'
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
#SBATCH --job-name=beetle_ctxft
#SBATCH --output=logs/context1024-ft100-fold-%a-%j.out
#SBATCH --error=logs/context1024-ft100-fold-%a-%j.err

set -euo pipefail
umask 002

ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase

FOLD="${1:?Usage: sbatch run_cv_cutmixema_context1024_ft100.slurm <fold>}"

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
    exit 20
}

if [[ -s "$RESULT_DIR/checkpoint_final.pth" ]]; then
    echo "Context fold $FOLD already completed."
    exit 0
fi

if [[ -e "$RESULT_DIR/checkpoint_latest.pth" || \
      -e "$RESULT_DIR/checkpoint_best.pth" ]]; then
    echo "ERROR: partial context output already exists:" >&2
    echo "  $RESULT_DIR" >&2
    echo "Inspect it deliberately before rerunning." >&2
    exit 21
fi

mkdir -p "$RESULT_DIR"

set +e
timeout \
  --signal=TERM \
  --kill-after=60s \
  11h50m \
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

load_result = network.load_state_dict(
    checkpoint["network_weights"],
    strict=True,
)

print("Loaded all pretrained CutMix + EMA network parameters successfully")
print("Missing keys:", load_result.missing_keys)
print("Unexpected keys:", load_result.unexpected_keys)

# Synchronize the newly constructed EMA copy with the imported weights.
# The existing trainer is expected to expose ema_model. The fallbacks keep the
# launcher defensive if the attribute name changes.
ema_synced = False

for attr_name in (
    "ema_model",
    "network_ema",
    "ema_network",
    "model_ema",
):
    ema_object = getattr(trainer, attr_name, None)

    if ema_object is None:
        continue

    target = getattr(ema_object, "module", ema_object)

    if hasattr(target, "_orig_mod"):
        target = target._orig_mod

    if hasattr(target, "load_state_dict"):
        target.load_state_dict(network.state_dict(), strict=True)
        print(f"Synchronized EMA weights through trainer.{attr_name}")
        ema_synced = True
        break

if not ema_synced:
    print(
        "WARNING: could not locate a directly loadable EMA model attribute. "
        "Training will still run, but inspect the first log lines carefully."
    )

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

trainer.run_training()
PY

RC=$?
set -e

if [[ ! -s "$MODEL_BASE/dataset.json" ]]; then
    echo "ERROR: context training returned without dataset.json:" >&2
    echo "  $MODEL_BASE/dataset.json" >&2
    exit 30
fi

if [[ ! -s "$RESULT_DIR/checkpoint_final.pth" ]]; then
    echo "ERROR: context training returned without checkpoint_final.pth" >&2
    echo "Exit code: $RC" >&2
    echo "  $RESULT_DIR/checkpoint_final.pth" >&2
    exit 31
fi

echo "Verified completed context training outputs:"
echo "  $MODEL_BASE/dataset.json"
echo "  $RESULT_DIR/checkpoint_final.pth"

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

chmod +x run_cv_cutmixema_context1024_ft100.slurm

###############################################################################
# 8. CREATE A FOLD-AWARE MIRRORED FULL-VALIDATION LAUNCHER.
###############################################################################
cat > run_cv_full_eval.slurm <<'SLURM'
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
#SBATCH --job-name=beetle_eval
#SBATCH --output=logs/full-eval-%x-%j.out
#SBATCH --error=logs/full-eval-%x-%j.err

set -euo pipefail
umask 002

ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase

STAGE="${1:?Usage: sbatch run_cv_full_eval.slurm <base|context> <fold>}"
FOLD="${2:?Usage: sbatch run_cv_full_eval.slurm <base|context> <fold>}"

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
        exit 40
        ;;
esac

CSV_PATH="/home/tijnveldwijk/fold${FOLD}_validation_inference_inputs.csv"
VIS_OUT_DIR="$VIS_OUT_DIR_BASE/fold_${FOLD}/$CHECKPOINT_TAG"

test -s "$CSV_PATH" || {
    echo "ERROR: missing validation CSV: $CSV_PATH" >&2
    exit 41
}

test -s "$MODEL_BASE_PATH/fold_${FOLD}/checkpoint_best.pth" || {
    echo "ERROR: missing general-best checkpoint:" >&2
    echo "  $MODEL_BASE_PATH/fold_${FOLD}/checkpoint_best.pth" >&2
    exit 42
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
echo "BEETLE full mirrored validation analysis"
echo "Started:          $(date)"
echo "Node:             $(hostname)"
echo "SLURM_JOB_ID:     ${SLURM_JOB_ID:-none}"
echo "Stage:            $STAGE"
echo "Fold:             $FOLD"
echo "MODEL_BASE_PATH:  $MODEL_BASE_PATH"
echo "CSV_PATH:         $CSV_PATH"
echo "CHECKPOINT_TAG:   $CHECKPOINT_TAG"
echo "MODEL_PATCH_SIZE: $MODEL_PATCH_SIZE"
echo "VIS_OUT_DIR:      $VIS_OUT_DIR"
echo "USE_MIRRORING:    $USE_MIRRORING"
echo "======================================================================"

python -u visual_analysis.py

OUT_JSON="$MODEL_BASE_PATH/fold_${FOLD}/fold${FOLD}_${CHECKPOINT_TAG}_full_validation_dice_tiffslide_hybrid_visual_cm.json"

test -s "$OUT_JSON" || {
    echo "ERROR: evaluation returned without expected JSON:" >&2
    echo "  $OUT_JSON" >&2
    exit 43
}

echo "Verified evaluation JSON:"
echo "  $OUT_JSON"

echo "Finished: $(date)"
SLURM

chmod +x run_cv_full_eval.slurm

###############################################################################
# 9. CREATE A FIVE-FOLD AGGREGATOR.
#
#    It reports:
#    - fold-level class Dice;
#    - mean and standard deviation across folds;
#    - pooled confusion matrix across all five validation folds;
#    - pooled class Dice, macro Dice and micro Dice;
#    - a checkpoint manifest for later ensemble inference;
#    - base-versus-context comparison.
###############################################################################
cat > aggregate_cv_results.py <<'PY'
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


def safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def safe_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def add_matrices(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    return [
        [int(x) + int(y) for x, y in zip(row_a, row_b)]
        for row_a, row_b in zip(a, b)
    ]


def dice_from_cm(cm: list[list[int]]) -> dict[str, float]:
    dices: dict[str, float] = {}

    for label, name in LABELS.items():
        tp = int(cm[label][label])
        fp = sum(int(cm[row][label]) for row in range(len(cm)) if row != label)
        fn = sum(int(cm[label][col]) for col in range(len(cm[label])) if col != label)

        denom = 2 * tp + fp + fn
        dices[name] = float(2 * tp / denom) if denom else float("nan")

    return dices


def micro_from_cm(cm: list[list[int]]) -> float:
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for label in LABELS:
        tp = int(cm[label][label])
        fp = sum(int(cm[row][label]) for row in range(len(cm)) if row != label)
        fn = sum(int(cm[label][col]) for col in range(len(cm[label])) if col != label)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    denom = 2 * total_tp + total_fp + total_fn
    return float(2 * total_tp / denom) if denom else float("nan")


def macro_from_dices(dices: dict[str, float]) -> float:
    values = [value for value in dices.values() if not math.isnan(value)]
    return safe_mean(values)


def stage_config(stage: str) -> tuple[Path, str]:
    root = Path(
        "/vol/csedu-nobackup/course/IMC037_aimi/group14/"
        "nnunet/tijn/pathology/nnUNet_results/Dataset301_BEETLE"
    )

    if stage == "base":
        return (
            root
            / "nnUNetTrainer_CutMixStainEMA__"
              "nnUNetWholeSlideDataPlans__"
              "wsd_None_iterator_nnunet_aug__2d",
            "cutmixema1000_best_mirror_visual",
        )

    if stage == "context":
        return (
            root
            / "nnUNetTrainer_CutMixStainEMA_Context1024FT100__"
              "nnUNetWholeSlideDataPlans__"
              "wsd_None_iterator_nnunet_aug__2d_context1024",
            "cutmixema1000_context1024_ft100_best_mirror_visual",
        )

    raise ValueError("stage must be base or context")


def load_stage(stage: str) -> dict[str, Any]:
    model_dir, tag = stage_config(stage)

    fold_results: list[dict[str, Any]] = []
    pooled_cm = [[0 for _ in range(5)] for _ in range(5)]

    for fold in range(5):
        result_path = (
            model_dir
            / f"fold_{fold}"
            / f"fold{fold}_{tag}_full_validation_dice_tiffslide_hybrid_visual_cm.json"
        )

        if not result_path.is_file():
            raise FileNotFoundError(f"Missing evaluation JSON: {result_path}")

        with result_path.open() as f:
            result = json.load(f)

        cm = result["confusion_matrix_rows_gt_cols_pred"]
        pooled_cm = add_matrices(pooled_cm, cm)

        fold_dices = {
            str(name): float(value)
            for name, value in result["class_dices"].items()
        }

        fold_results.append(
            {
                "fold": fold,
                "json": str(result_path),
                "checkpoint": str(model_dir / f"fold_{fold}" / "checkpoint_best.pth"),
                "class_dices": fold_dices,
                "macro_mean_dice": float(result["macro_mean_dice"]),
                "micro_overall_dice": float(result["micro_overall_dice"]),
                "processed_annotated_tiles": int(result["processed_annotated_tiles"]),
            }
        )

    per_class_stats: dict[str, dict[str, float]] = {}

    for name in LABELS.values():
        values = [fold["class_dices"][name] for fold in fold_results]

        per_class_stats[name] = {
            "mean": safe_mean(values),
            "std": safe_std(values),
            "values": values,
        }

    pooled_class_dices = dice_from_cm(pooled_cm)
    pooled_macro = macro_from_dices(pooled_class_dices)
    pooled_micro = micro_from_cm(pooled_cm)

    fold_macro_values = [fold["macro_mean_dice"] for fold in fold_results]
    fold_micro_values = [fold["micro_overall_dice"] for fold in fold_results]

    summary = {
        "stage": stage,
        "model_dir": str(model_dir),
        "checkpoint_tag": tag,
        "fold_results": fold_results,
        "fold_mean_std": {
            "class_dices": per_class_stats,
            "macro_mean_dice": {
                "mean": safe_mean(fold_macro_values),
                "std": safe_std(fold_macro_values),
                "values": fold_macro_values,
            },
            "micro_overall_dice": {
                "mean": safe_mean(fold_micro_values),
                "std": safe_std(fold_micro_values),
                "values": fold_micro_values,
            },
        },
        "pooled_across_all_validation_pixels": {
            "class_dices": pooled_class_dices,
            "macro_mean_dice": pooled_macro,
            "micro_overall_dice": pooled_micro,
            "confusion_matrix_rows_gt_cols_prediction": pooled_cm,
        },
    }

    return summary


def save_stage(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    stage = summary["stage"]

    json_path = output_dir / f"cv_summary_{stage}.json"
    csv_path = output_dir / f"cv_summary_{stage}.csv"
    cm_path = output_dir / f"cv_pooled_confusion_matrix_{stage}.csv"
    manifest_path = output_dir / f"ensemble_manifest_{stage}.txt"

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
                "macro_mean_dice",
                "micro_overall_dice",
            ]
        )

        for fold in summary["fold_results"]:
            writer.writerow(
                [
                    fold["fold"],
                    fold["class_dices"]["other"],
                    fold["class_dices"]["non-invasive epithelium"],
                    fold["class_dices"]["invasive epithelium"],
                    fold["class_dices"]["necrosis"],
                    fold["macro_mean_dice"],
                    fold["micro_overall_dice"],
                ]
            )

        stats = summary["fold_mean_std"]

        writer.writerow(
            [
                "mean",
                stats["class_dices"]["other"]["mean"],
                stats["class_dices"]["non-invasive epithelium"]["mean"],
                stats["class_dices"]["invasive epithelium"]["mean"],
                stats["class_dices"]["necrosis"]["mean"],
                stats["macro_mean_dice"]["mean"],
                stats["micro_overall_dice"]["mean"],
            ]
        )

        writer.writerow(
            [
                "std",
                stats["class_dices"]["other"]["std"],
                stats["class_dices"]["non-invasive epithelium"]["std"],
                stats["class_dices"]["invasive epithelium"]["std"],
                stats["class_dices"]["necrosis"]["std"],
                stats["macro_mean_dice"]["std"],
                stats["micro_overall_dice"]["std"],
            ]
        )

        pooled = summary["pooled_across_all_validation_pixels"]

        writer.writerow(
            [
                "pooled_pixels",
                pooled["class_dices"]["other"],
                pooled["class_dices"]["non-invasive epithelium"],
                pooled["class_dices"]["invasive epithelium"],
                pooled["class_dices"]["necrosis"],
                pooled["macro_mean_dice"],
                pooled["micro_overall_dice"],
            ]
        )

    with cm_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(
            summary["pooled_across_all_validation_pixels"][
                "confusion_matrix_rows_gt_cols_prediction"
            ]
        )

    with manifest_path.open("w") as f:
        for fold in summary["fold_results"]:
            f.write(f"fold_{fold['fold']}={fold['checkpoint']}\n")

    print("Saved:", json_path)
    print("Saved:", csv_path)
    print("Saved:", cm_path)
    print("Saved:", manifest_path)


def save_comparison(
    base: dict[str, Any],
    context: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    base_pooled = base["pooled_across_all_validation_pixels"]
    context_pooled = context["pooled_across_all_validation_pixels"]

    comparison = {
        "base": base_pooled,
        "context": context_pooled,
        "context_minus_base": {
            "class_dices": {
                name: (
                    context_pooled["class_dices"][name]
                    - base_pooled["class_dices"][name]
                )
                for name in LABELS.values()
            },
            "macro_mean_dice": (
                context_pooled["macro_mean_dice"]
                - base_pooled["macro_mean_dice"]
            ),
            "micro_overall_dice": (
                context_pooled["micro_overall_dice"]
                - base_pooled["micro_overall_dice"]
            ),
        },
    }

    json_path = output_dir / "cv_comparison_context_minus_base.json"
    csv_path = output_dir / "cv_comparison_context_minus_base.csv"

    with json_path.open("w") as f:
        json.dump(comparison, f, indent=4)

    delta = comparison["context_minus_base"]

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

        for name in LABELS.values():
            writer.writerow(
                [
                    f"dice_{name}",
                    base_pooled["class_dices"][name],
                    context_pooled["class_dices"][name],
                    delta["class_dices"][name],
                ]
            )

        writer.writerow(
            [
                "macro_mean_dice",
                base_pooled["macro_mean_dice"],
                context_pooled["macro_mean_dice"],
                delta["macro_mean_dice"],
            ]
        )

        writer.writerow(
            [
                "micro_overall_dice",
                base_pooled["micro_overall_dice"],
                context_pooled["micro_overall_dice"],
                delta["micro_overall_dice"],
            ]
        )

    print("Saved:", json_path)
    print("Saved:", csv_path)


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"base", "context", "both"}:
        raise SystemExit(
            "Usage: python aggregate_cv_results.py <base|context|both>"
        )

    requested = sys.argv[1]

    output_dir = Path(
        "/home/tijnveldwijk/AIMI---BEETLE-Project-Phase/"
        "cv_summaries/cutmixema1000_context1024"
    )

    if requested == "base":
        base = load_stage("base")
        save_stage(base, output_dir)
        return

    if requested == "context":
        context = load_stage("context")
        save_stage(context, output_dir)
        return

    base = load_stage("base")
    context = load_stage("context")

    save_stage(base, output_dir)
    save_stage(context, output_dir)
    save_comparison(base, context, output_dir)


if __name__ == "__main__":
    main()
PY

chmod +x aggregate_cv_results.py

###############################################################################
# 10. CREATE THE CPU-ONLY AGGREGATION LAUNCHER.
###############################################################################
cat > run_cv_aggregate.slurm <<'SLURM'
#!/usr/bin/env bash
#SBATCH --account=cseduimc037
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --job-name=aggregate_cv
#SBATCH --output=logs/aggregate-cv-%j.out
#SBATCH --error=logs/aggregate-cv-%j.err

set -euo pipefail
umask 002

ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase
STAGE="${1:?Usage: sbatch run_cv_aggregate.slurm <base|both>}"

cd "$ROOT"

source /scratch/$USER/virtual_environments/AIMI-BEETLE/bin/activate

python -u aggregate_cv_results.py "$STAGE"

echo "Finished: $(date)"
SLURM

chmod +x run_cv_aggregate.slurm

###############################################################################
# 11. CANCEL THE OLD STANDALONE FOLD-0 EVALUATION IF IT IS STILL PENDING.
#
#     It was useful before the full CV plan existed, but would now consume a GPU
#     before all five stage-1 folds have completed.
###############################################################################
while read -r OLD_JOB; do
    [[ -z "$OLD_JOB" ]] && continue

    echo "Cancelling old pending standalone evaluation job: $OLD_JOB"
    scancel "$OLD_JOB"
done < <(
    squeue \
      -u "$USER" \
      -h \
      -t PD \
      -n eval_cutmixema_best_mirror \
      -o "%A" \
      || true
)

###############################################################################
# 12. QUEUE STAGE 1:
#
#     Fold 0:
#       current external chunk 1 -> queued resume chunk 2
#
#     Folds 1-4:
#       fresh chunk 1 -> resume chunk 2
#
#     All GPU jobs request one GPU. SLURM will use up to the available two GPUs.
###############################################################################
declare -a BASE_FINAL_JOBS

if squeue -h -j "$FOLD0_ACTIVE_JOB_ID" -o "%A" | grep -q .; then
    echo "Fold 0 active chunk found: $FOLD0_ACTIVE_JOB_ID"

    FOLD0_RESUME_JOB=$(
        sbatch \
          --parsable \
          --dependency=afterany:${FOLD0_ACTIVE_JOB_ID} \
          --job-name=cutmixema-f0-c2 \
          run_cv_cutmixema1000_chunk.slurm \
          0 resume chunk2
    )
else
    echo "Fold 0 active job is no longer in squeue: $FOLD0_ACTIVE_JOB_ID"
    echo "Queueing fold 0 resume chunk immediately."

    FOLD0_RESUME_JOB=$(
        sbatch \
          --parsable \
          --job-name=cutmixema-f0-c2 \
          run_cv_cutmixema1000_chunk.slurm \
          0 resume chunk2
    )
fi

BASE_FINAL_JOBS[0]="$FOLD0_RESUME_JOB"

echo "Queued fold 0 resume chunk: $FOLD0_RESUME_JOB"

for FOLD in 1 2 3 4; do
    CHUNK1_JOB=$(
        sbatch \
          --parsable \
          --job-name=cutmixema-f${FOLD}-c1 \
          run_cv_cutmixema1000_chunk.slurm \
          "$FOLD" fresh chunk1
    )

    CHUNK2_JOB=$(
        sbatch \
          --parsable \
          --dependency=afterany:${CHUNK1_JOB} \
          --job-name=cutmixema-f${FOLD}-c2 \
          run_cv_cutmixema1000_chunk.slurm \
          "$FOLD" resume chunk2
    )

    BASE_FINAL_JOBS[$FOLD]="$CHUNK2_JOB"

    echo "Queued fold $FOLD chunk 1: $CHUNK1_JOB"
    echo "Queued fold $FOLD chunk 2: $CHUNK2_JOB"
done

BASE_TRAIN_DEP=$(
    IFS=:
    echo "${BASE_FINAL_JOBS[*]}"
)

###############################################################################
# 13. QUEUE STAGE 2:
#     Wait for all five 1000-epoch folds, then evaluate all five with mirroring.
###############################################################################
declare -a BASE_EVAL_JOBS

for FOLD in 0 1 2 3 4; do
    JOB=$(
        sbatch \
          --parsable \
          --dependency=afterok:${BASE_TRAIN_DEP} \
          --job-name=eval-cema1000-f${FOLD} \
          run_cv_full_eval.slurm \
          base "$FOLD"
    )

    BASE_EVAL_JOBS[$FOLD]="$JOB"

    echo "Queued CutMix + EMA full evaluation fold $FOLD: $JOB"
done

BASE_EVAL_DEP=$(
    IFS=:
    echo "${BASE_EVAL_JOBS[*]}"
)

BASE_AGG_JOB=$(
    sbatch \
      --parsable \
      --dependency=afterok:${BASE_EVAL_DEP} \
      --job-name=aggregate-cema1000 \
      run_cv_aggregate.slurm \
      base
)

echo "Queued CutMix + EMA five-fold aggregation: $BASE_AGG_JOB"

###############################################################################
# 14. QUEUE STAGE 3:
#     Wait for all stage-2 evaluations and aggregation, then context fine-tune
#     all five folds for 100 extra epochs.
###############################################################################
declare -a CONTEXT_TRAIN_JOBS

for FOLD in 0 1 2 3 4; do
    JOB=$(
        sbatch \
          --parsable \
          --dependency=afterok:${BASE_AGG_JOB} \
          --job-name=ctx1024-ft-f${FOLD} \
          run_cv_cutmixema_context1024_ft100.slurm \
          "$FOLD"
    )

    CONTEXT_TRAIN_JOBS[$FOLD]="$JOB"

    echo "Queued context fine-tune fold $FOLD: $JOB"
done

CONTEXT_TRAIN_DEP=$(
    IFS=:
    echo "${CONTEXT_TRAIN_JOBS[*]}"
)

###############################################################################
# 15. QUEUE STAGE 4:
#     Wait for all five context fine-tunes, then evaluate all five with
#     effective mirroring.
###############################################################################
declare -a CONTEXT_EVAL_JOBS

for FOLD in 0 1 2 3 4; do
    JOB=$(
        sbatch \
          --parsable \
          --dependency=afterok:${CONTEXT_TRAIN_DEP} \
          --job-name=eval-ctx1024-f${FOLD} \
          run_cv_full_eval.slurm \
          context "$FOLD"
    )

    CONTEXT_EVAL_JOBS[$FOLD]="$JOB"

    echo "Queued context full evaluation fold $FOLD: $JOB"
done

CONTEXT_EVAL_DEP=$(
    IFS=:
    echo "${CONTEXT_EVAL_JOBS[*]}"
)

###############################################################################
# 16. QUEUE STAGE 5:
#     Aggregate the final five-fold results and write the comparison.
###############################################################################
FINAL_AGG_JOB=$(
    sbatch \
      --parsable \
      --dependency=afterok:${CONTEXT_EVAL_DEP} \
      --job-name=aggregate-final-cv \
      run_cv_aggregate.slurm \
      both
)

echo "Queued final five-fold aggregation and comparison: $FINAL_AGG_JOB"

###############################################################################
# 17. SAVE THE SUBMISSION MANIFEST.
###############################################################################
MANIFEST=queued_cutmixema_context_cv_jobs.txt

{
    echo "Submitted: $(date --iso-8601=seconds)"
    echo "Fold 0 existing chunk 1: $FOLD0_ACTIVE_JOB_ID"
    echo
    echo "Stage 1 final 1000-epoch jobs:"
    for FOLD in 0 1 2 3 4; do
        echo "  fold $FOLD: ${BASE_FINAL_JOBS[$FOLD]}"
    done
    echo
    echo "Stage 2 base evaluations:"
    for FOLD in 0 1 2 3 4; do
        echo "  fold $FOLD: ${BASE_EVAL_JOBS[$FOLD]}"
    done
    echo "  aggregate: $BASE_AGG_JOB"
    echo
    echo "Stage 3 context fine-tunes:"
    for FOLD in 0 1 2 3 4; do
        echo "  fold $FOLD: ${CONTEXT_TRAIN_JOBS[$FOLD]}"
    done
    echo
    echo "Stage 4 context evaluations:"
    for FOLD in 0 1 2 3 4; do
        echo "  fold $FOLD: ${CONTEXT_EVAL_JOBS[$FOLD]}"
    done
    echo
    echo "Stage 5 final aggregate: $FINAL_AGG_JOB"
} | tee "$MANIFEST"

echo
echo "======================================================================"
echo "FULL FIVE-FOLD PIPELINE QUEUED"
echo "======================================================================"
echo
echo "Submission manifest:"
echo "  $ROOT/$MANIFEST"
echo
echo "Final summaries will be written to:"
echo "  $ROOT/cv_summaries/cutmixema1000_context1024/"
echo
echo "Inspect the queue with:"
echo "  squeue -u \$USER"
echo

squeue -u "$USER"
