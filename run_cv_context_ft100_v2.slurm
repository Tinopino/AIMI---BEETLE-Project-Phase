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
