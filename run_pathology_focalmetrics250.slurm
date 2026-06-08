#!/bin/bash
#SBATCH --account=cseduimc037
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --nodelist=cn48
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=18G
#SBATCH --time=12:00:00
#SBATCH --job-name=beetle_focalmetrics250
#SBATCH --output=logs/pathology-focalmetrics250-fold-%A_%a.out
#SBATCH --error=logs/pathology-focalmetrics250-fold-%A_%a.out
set -euo pipefail
umask 002

###############################################################################
# Usage:
#   sbatch run_pathology_alpha1000_chunk.slurm 0 fresh
#   sbatch --dependency=afterany:<JOB_ID> run_pathology_alpha1000_chunk.slurm 0 resume
#
# The first argument is the fold. The second is either "fresh" or "resume".
###############################################################################

FOLD="${1:-0}"
MODE="${2:-fresh}"

if [[ "$MODE" != "fresh" && "$MODE" != "resume" ]]; then
    echo "ERROR: MODE must be 'fresh' or 'resume', got: $MODE" >&2
    exit 2
fi

TRAINER="nnUNetTrainerPathologyFocalClassMetrics"
PLANNER="nnUNetWholeSlideDataPlans"

cd /home/tijnveldwijk/AIMI---BEETLE-Project-Phase
mkdir -p logs

source /scratch/$USER/virtual_environments/AIMI-BEETLE/bin/activate
export PYTHONNOUSERSITE=1

export nnUNet_raw=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_raw
export nnUNet_preprocessed=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_preprocessed
export nnUNet_results=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export WANDB_MODE=disabled
export WANDB_DISABLED=true

RESULT_DIR="$nnUNet_results/Dataset301_BEETLE/${TRAINER}__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d/fold_${FOLD}"

echo "======================================================================"
echo "BEETLE focal-loss per-class-metrics 250-epoch run"
echo "Started:              $(date)"
echo "Node:                 $(hostname)"
echo "SLURM_JOB_ID:         ${SLURM_JOB_ID:-none}"
echo "Fold:                 $FOLD"
echo "Mode:                 $MODE"
echo "Trainer:              $TRAINER"
echo "Result directory:     $RESULT_DIR"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "======================================================================"

python -u - <<'PY'
import torch, nnunetv2
print("torch:", torch.__version__, flush=True)
print("cuda:", torch.cuda.is_available(), flush=True)
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", flush=True)
print("nnunetv2:", nnunetv2.__file__, flush=True)
PY

# Use ten CPUs for the WSI iterators, while leaving two reserved cores available.
python -u - <<'PY'
import json, os
path = os.path.join(
    os.environ["nnUNet_preprocessed"],
    "Dataset301_BEETLE",
    "dataset.json",
)
with open(path) as f:
    dataset_json = json.load(f)
dataset_json["cpus"] = 10
with open(path, "w") as f:
    json.dump(dataset_json, f, indent=4)
print("Set dataset.json cpus=10", flush=True)
PY

# Verify that the long-run trainer class can be discovered before requesting GPU work.
python -u - <<'PY'
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
import nnunetv2
from os.path import join

trainer_name = "nnUNetTrainerPathologyFocalClassMetrics"
trainer = recursive_find_python_class(
    join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
    trainer_name,
    "nnunetv2.training.nnUNetTrainer",
)
if trainer is None:
    raise RuntimeError(f"Could not locate trainer class: {trainer_name}")
print("Located trainer:", trainer.__name__, flush=True)
PY

if [[ "$MODE" == "fresh" ]]; then
    if [[ -e "$RESULT_DIR/checkpoint_latest.pth" || -e "$RESULT_DIR/checkpoint_final.pth" || -e "$RESULT_DIR/checkpoint_best.pth" ]]; then
        echo "ERROR: A checkpoint already exists in the new long-run output folder:" >&2
        echo "  $RESULT_DIR" >&2
        echo "Use MODE=resume to continue it, or remove the folder deliberately before a new run." >&2
        exit 3
    fi

    echo "Running pathology planner if needed..."
    python -u nnUNet_pathology/nnunetv2/experiment_planning/experiment_planners/pathology_experiment_planner.py \
      301 \
      --gpu_gb=11
else
    if [[ -e "$RESULT_DIR/checkpoint_final.pth" ]]; then
        echo "Training already finished: checkpoint_final.pth exists."
        echo "No continuation is required."
        exit 0
    fi

    if [[ ! -e "$RESULT_DIR/checkpoint_latest.pth" && ! -e "$RESULT_DIR/checkpoint_best.pth" ]]; then
        echo "ERROR: Resume requested, but no checkpoint_latest.pth or checkpoint_best.pth exists:" >&2
        echo "  $RESULT_DIR" >&2
        exit 4
    fi

    echo "Resume checkpoint candidates:"
    ls -lh "$RESULT_DIR"/checkpoint_latest.pth "$RESULT_DIR"/checkpoint_best.pth 2>/dev/null || true
fi

echo "GPU status:"
nvidia-smi

echo "Starting training chunk in MODE=$MODE..."

python -u - "$FOLD" "$TRAINER" "$PLANNER" "$MODE" <<'PY'
import sys

from nnunetv2.run.run_training_pathology import run_training

fold = int(sys.argv[1])
trainer = sys.argv[2]
planner = sys.argv[3]
mode = sys.argv[4]

run_training(
    dataset_name_or_id="301",
    configuration="2d",
    fold=fold,
    trainer_class_name=trainer,
    plans_identifier=planner,
    continue_training=(mode == "resume"),
)
PY

echo "Finished chunk: $(date)"
