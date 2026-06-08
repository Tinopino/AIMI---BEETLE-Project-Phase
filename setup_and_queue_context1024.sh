#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase

DATALOADER="$ROOT/nnUNet_pathology/nnunetv2/training/nnUNetTrainer/variants/pathology/nnUNetTrainer_WSD_undefined_dataloader.py"

TRAINER_FILE="$ROOT/nnUNet_pathology/nnunetv2/training/nnUNetTrainer/nnUnetTrainerBEETLE.py"

cd "$ROOT"
mkdir -p logs

###############################################################################
# 1. Patch the WSD loader once:
#    keep the existing batch size 8 for all existing trainers,
#    but allow an explicit override for the new 1024-context trainer.
###############################################################################
cp -n "$DATALOADER" "${DATALOADER}.before_context1024.bak" || true

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
# 2. Append two trainer classes once:
#    - five-epoch pilot for memory/runtime verification;
#    - full 100-epoch context experiment.
###############################################################################
cp -n "$TRAINER_FILE" "${TRAINER_FILE}.before_context1024.bak" || true

python - <<'PY'
from pathlib import Path

path = Path(
    "nnUNet_pathology/nnunetv2/training/nnUNetTrainer/"
    "nnUnetTrainerBEETLE.py"
)

text = path.read_text()

marker = "class nnUNetTrainerPathologyFocalClassMetricsAlphaContext1024FT100"

if marker in text:
    print("Context-1024 trainer classes already installed")
else:
    addition = r'''

# =============================================================================
# LARGER-CONTEXT ABLATION:
# WEIGHTED FOCAL + 1024x1024 PATCHES + LOW-LR FINE-TUNING INITIALIZATION
# =============================================================================

class nnUNetTrainerPathologyFocalClassMetricsAlphaContext1024FT100(
    nnUNetTrainerPathologyFocalClassMetricsAlpha
):
    """
    Larger-context ablation for BEETLE.

    Keeps unchanged:
    - network topology;
    - Dice + alpha-weighted focal loss;
    - class-specific validation metrics and checkpoints;
    - normal 0.25 / 0.25 / 0.25 / 0.25 label-anchor sampling;
    - pathology augmentations.

    Changes:
    - receives 1024x1024 training patches through configuration
      `2d_context1024`;
    - uses WSD batch size 2;
    - uses initial learning rate 0.001;
    - runs for 100 epochs.

    The SLURM launcher initializes all network parameters from the completed
    250-epoch weighted-focal checkpoint before training begins.
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
                "Context-1024 trainer requires configuration patch_size="
                f"{expected_patch_size}, but received {actual_patch_size}. "
                "Use configuration='2d_context1024'."
            )

        # The pathology WSD loader otherwise defaults to batch size 8.
        self.wsd_batch_size_override = 2

        # Low-LR fine-tuning schedule.
        self.initial_lr = 1e-3
        self.num_epochs = 100
        self.save_every = 1

        self.print_to_log_file(
            "Using larger-context weighted-focal fine-tuning: "
            "patch_size=[1024, 1024], WSD batch_size=2, "
            "initial_lr=0.001, num_epochs=100",
            also_print_to_console=True,
        )


class nnUNetTrainerPathologyFocalClassMetricsAlphaContext1024Pilot5(
    nnUNetTrainerPathologyFocalClassMetricsAlphaContext1024FT100
):
    """
    Five-epoch pilot to verify GPU memory usage and runtime before queueing the
    full 100-epoch context experiment.
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

        self.num_epochs = 5

        self.print_to_log_file(
            "Running five-epoch Context1024 pilot.",
            also_print_to_console=True,
        )
'''

    path.write_text(text + addition)
    print("Appended Context1024 trainer classes")
PY

python -m py_compile "$DATALOADER"
python -m py_compile "$TRAINER_FILE"

###############################################################################
# 3. Create one SLURM launcher for pilot and full training.
###############################################################################
cat > run_pathology_context1024_ft.slurm <<'SLURM'
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
#SBATCH --job-name=beetle_ctx1024
#SBATCH --output=logs/pathology-context1024-%x-%j.out
#SBATCH --error=logs/pathology-context1024-%x-%j.out

set -euo pipefail
umask 002

ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase

RUN_KIND="${1:-pilot}"
FOLD="${2:-0}"

case "$RUN_KIND" in
    pilot)
        TRAINER="nnUNetTrainerPathologyFocalClassMetricsAlphaContext1024Pilot5"
        ;;
    full)
        TRAINER="nnUNetTrainerPathologyFocalClassMetricsAlphaContext1024FT100"
        ;;
    *)
        echo "ERROR: first argument must be 'pilot' or 'full'" >&2
        exit 2
        ;;
esac

PLANNER="nnUNetWholeSlideDataPlans"
CONFIG="2d_context1024"

PRETRAINED=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results/Dataset301_BEETLE/nnUNetTrainerPathologyFocalClassMetricsAlpha__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d/fold_0/checkpoint_best.pth

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

MODEL_BASE="$nnUNet_results/Dataset301_BEETLE/${TRAINER}__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__${CONFIG}"

RESULT_DIR="$MODEL_BASE/fold_${FOLD}"

echo "======================================================================"
echo "BEETLE weighted-focal larger-context experiment"
echo "Started:              $(date)"
echo "Node:                 $(hostname)"
echo "SLURM_JOB_ID:         ${SLURM_JOB_ID:-none}"
echo "Run kind:             $RUN_KIND"
echo "Fold:                 $FOLD"
echo "Trainer:              $TRAINER"
echo "Configuration:        $CONFIG"
echo "Patch size:           1024 x 1024"
echo "WSD batch size:       2"
echo "Initial LR:           0.001"
echo "Pretrained checkpoint:$PRETRAINED"
echo "Result directory:     $RESULT_DIR"
echo "======================================================================"

if [[ ! -s "$PRETRAINED" ]]; then
    echo "ERROR: weighted-focal checkpoint not found:" >&2
    echo "  $PRETRAINED" >&2
    exit 3
fi

if [[ -e "$RESULT_DIR/checkpoint_latest.pth" || \
      -e "$RESULT_DIR/checkpoint_final.pth" || \
      -e "$RESULT_DIR/checkpoint_best.pth" ]]; then
    echo "ERROR: output folder already contains checkpoints:" >&2
    echo "  $RESULT_DIR" >&2
    echo "Remove the folder deliberately before rerunning this fresh experiment." >&2
    exit 4
fi

###############################################################################
# Add a separate plans configuration. The original 2d configuration remains
# untouched. WSD extracts patches online, so it can reuse the existing
# data_identifier while changing the network input crop size.
###############################################################################
python -u - <<'PY'
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

# Reuse the same data identifier: this pathology pipeline reads WSI patches
# online rather than creating a different preprocessed patch dataset.
context["data_identifier"] = base["data_identifier"]

plans["configurations"]["2d_context1024"] = context

with plans_path.open("w") as f:
    json.dump(plans, f, indent=4)

print("Configured 2d_context1024:")
print(json.dumps(plans["configurations"]["2d_context1024"], indent=4))
PY

###############################################################################
# Keep 10 WSD iterator CPUs, leaving two requested cores available.
###############################################################################
python -u - <<'PY'
import json
import os
from pathlib import Path

dataset_path = (
    Path(os.environ["nnUNet_preprocessed"])
    / "Dataset301_BEETLE"
    / "dataset.json"
)

with dataset_path.open() as f:
    dataset_json = json.load(f)

dataset_json["cpus"] = 10

with dataset_path.open("w") as f:
    json.dump(dataset_json, f, indent=4)

print("Set dataset.json cpus=10")
PY

###############################################################################
# Train a new model initialized from ALL network weights of the completed
# weighted-focal model. Patch-size changes do not alter parameter shapes.
###############################################################################
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
        "Expected key 'network_weights' in pretrained checkpoint. "
        f"Available keys: {sorted(checkpoint.keys())}"
    )

network = trainer.network

if hasattr(network, "_orig_mod"):
    network = network._orig_mod

load_result = network.load_state_dict(
    checkpoint["network_weights"],
    strict=True,
)

print("Loaded all weighted-focal network parameters successfully")
print("Missing keys:", load_result.missing_keys)
print("Unexpected keys:", load_result.unexpected_keys)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

trainer.run_training()
PY

###############################################################################
# Prevent dependent jobs from running after a swallowed Python error.
###############################################################################
if [[ ! -s "$MODEL_BASE/dataset.json" ]]; then
    echo "ERROR: training returned without creating dataset.json:" >&2
    echo "  $MODEL_BASE/dataset.json" >&2
    exit 20
fi

if [[ ! -s "$RESULT_DIR/checkpoint_final.pth" ]]; then
    echo "ERROR: training returned without creating checkpoint_final.pth:" >&2
    echo "  $RESULT_DIR/checkpoint_final.pth" >&2
    exit 21
fi

echo "Verified completed training outputs:"
echo "  $MODEL_BASE/dataset.json"
echo "  $RESULT_DIR/checkpoint_final.pth"

echo "Finished: $(date)"
SLURM

chmod +x run_pathology_context1024_ft.slurm

###############################################################################
# 4. Queue:
#    5-epoch pilot -> 100-epoch full run -> mirrored validation analysis.
###############################################################################
PILOT_JOB=$(sbatch --parsable \
  --job-name=beetle_ctx1024_pilot \
  run_pathology_context1024_ft.slurm \
  pilot 0)

echo "Queued five-epoch context pilot: $PILOT_JOB"

FULL_JOB=$(sbatch --parsable \
  --dependency=afterok:${PILOT_JOB} \
  --job-name=beetle_ctx1024_ft100 \
  run_pathology_context1024_ft.slurm \
  full 0)

echo "Queued 100-epoch context experiment: $FULL_JOB"

EVAL_JOB=$(sbatch --parsable \
  --dependency=afterok:${FULL_JOB} \
  --nodelist=cn48 \
  --time=12:00:00 \
  --job-name=eval_ctx1024_mirror \
  --output=logs/eval-context1024-mirror-%j.out \
  --error=logs/eval-context1024-mirror-%j.err \
  --export=ALL,MODEL_BASE_PATH=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results/Dataset301_BEETLE/nnUNetTrainerPathologyFocalClassMetricsAlphaContext1024FT100__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d_context1024,CHECKPOINT_NAME=checkpoint_best.pth,CHECKPOINT_TAG=context1024_ft100_best_mirror_visual,SAVE_VISUALS=1,USE_MIRRORING=1 \
  run_original_beetle_fast_eval.slurm)

echo "Queued mirrored context-model evaluation: $EVAL_JOB"

echo
echo "======================================================================"
echo "CONTEXT-1024 PIPELINE QUEUED"
echo "======================================================================"
echo "Pilot:      $PILOT_JOB"
echo "Full run:   $FULL_JOB"
echo "Evaluation: $EVAL_JOB"
echo
echo "Dependency chain:"
echo "  5-epoch pilot -> 100-epoch full run -> mirrored full-fold evaluation"
echo

squeue -u "$USER"
