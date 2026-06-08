#!/usr/bin/env bash
set -euo pipefail
umask 002

###############################################################################
# PATHS
###############################################################################
ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase

NNUNET_PREPROCESSED=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_preprocessed

NNUNET_RESULTS=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results

DATASET=Dataset301_BEETLE
PLANNER=nnUNetWholeSlideDataPlans

BASE_TRAINER=nnUNetTrainer_CutMixStainEMA
CONTEXT_TRAINER=nnUNetTrainer_CutMixStainEMA_Context1024FT100

BASE_MODEL_DIR="$NNUNET_RESULTS/$DATASET/${BASE_TRAINER}__${PLANNER}__wsd_None_iterator_nnunet_aug__2d"

CONTEXT_MODEL_DIR="$NNUNET_RESULTS/$DATASET/${CONTEXT_TRAINER}__${PLANNER}__wsd_None_iterator_nnunet_aug__2d_context1024"

EVAL_MANIFEST="$ROOT/queued_basecv_evaluations.txt"

DATALOADER_FILE="$ROOT/nnUNet_pathology/nnunetv2/training/nnUNetTrainer/variants/pathology/nnUNetTrainer_WSD_undefined_dataloader.py"

BEETLE_TRAINER_FILE="$ROOT/nnUNet_pathology/nnunetv2/training/nnUNetTrainer/nnUnetTrainerBEETLE.py"

CONTEXT_TRAINER_FILE="$ROOT/nnUNet_pathology/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_CutMixStainEMA_Context1024FT100.py"

cd "$ROOT"
mkdir -p logs

###############################################################################
# 1. PRE-FLIGHT CHECKS
###############################################################################
echo "======================================================================"
echo "CONTEXT-FINETUNING PRE-FLIGHT CHECKS"
echo "======================================================================"

test -s "$EVAL_MANIFEST" || {
    echo "ERROR: base-evaluation manifest is missing:" >&2
    echo "  $EVAL_MANIFEST" >&2
    echo >&2
    echo "Queue the five base evaluations first." >&2
    exit 10
}

test -s "$DATALOADER_FILE" || {
    echo "ERROR: missing WSD dataloader file:" >&2
    echo "  $DATALOADER_FILE" >&2
    exit 11
}

test -s "$BEETLE_TRAINER_FILE" || {
    echo "ERROR: missing BEETLE trainer file:" >&2
    echo "  $BEETLE_TRAINER_FILE" >&2
    exit 12
}

###############################################################################
# Refuse duplicate submissions.
###############################################################################
if squeue -h -u "$USER" -o "%j" |
    grep -Eq '^ctx1024ft-f[0-4]-c[12]$'; then

    echo "ERROR: context-finetuning jobs already appear to be queued." >&2
    echo "Inspect with:" >&2
    echo "  squeue -u \$USER" >&2
    exit 13
fi

###############################################################################
# Extract evaluation job IDs from the existing evaluation manifest.
###############################################################################
get_eval_job_id() {
    local fold="$1"

    local id

    id="$(
        grep -E "fold ${fold}( evaluation)?: [0-9]+$" "$EVAL_MANIFEST" |
        awk '{print $NF}' |
        head -n 1
    )"

    if [[ ! "$id" =~ ^[0-9]+$ ]]; then
        echo "ERROR: could not extract fold-$fold evaluation job ID from:" >&2
        echo "  $EVAL_MANIFEST" >&2
        exit 20
    fi

    echo "$id"
}

E0=$(get_eval_job_id 0)
E1=$(get_eval_job_id 1)
E2=$(get_eval_job_id 2)
E3=$(get_eval_job_id 3)
E4=$(get_eval_job_id 4)

echo
echo "Base-evaluation jobs:"
echo "  fold 0: $E0"
echo "  fold 1: $E1"
echo "  fold 2: $E2"
echo "  fold 3: $E3"
echo "  fold 4: $E4"

ALL_BASE_EVALS="$E0:$E1:$E2:$E3:$E4"

###############################################################################
# 2. VERIFY THAT THE INVALID EARLIER CONTEXT CLASS IS NOT APPENDED TO THE
#    GENERAL BEETLE MODULE
###############################################################################
python - <<'PY'
from pathlib import Path

path = Path(
    "nnUNet_pathology/nnunetv2/training/"
    "nnUNetTrainer/nnUnetTrainerBEETLE.py"
)

text = path.read_text()

marker = "class nnUNetTrainer_CutMixStainEMA_Context1024FT100"

if marker in text:
    raise RuntimeError(
        "The invalid earlier context class is still appended to "
        "nnUnetTrainerBEETLE.py. Do not queue fine-tuning until it is removed."
    )

print("Verified: no invalid context class remains in nnUnetTrainerBEETLE.py")
PY

###############################################################################
# 3. ENSURE THE WSD LOADER SUPPORTS A PER-TRAINER BATCH-SIZE OVERRIDE
#
# Existing 512x512 trainers continue using batch size 8.
# The new 1024x1024 trainer sets batch size 2.
###############################################################################
cp -n "$DATALOADER_FILE" \
      "${DATALOADER_FILE}.before_context1024.bak" \
      || true

python - <<'PY'
from pathlib import Path

path = Path(
    "nnUNet_pathology/nnunetv2/training/"
    "nnUNetTrainer/variants/pathology/"
    "nnUNetTrainer_WSD_undefined_dataloader.py"
)

text = path.read_text()

override = """        batch_size = int(getattr(self, "wsd_batch_size_override", 8))
        print(f'\\n\\n\\nWSD BATCH SIZE {batch_size}\\n\\n\\n')
"""

old = """        print('\\n\\n\\nTEMP BATCH SIZE 8\\n\\n\\n')
        # batch_size = self.configuration_manager.batch_size
        batch_size = 8
"""

if override in text:
    print("WSD batch-size override already installed")
elif old in text:
    path.write_text(text.replace(old, override, 1))
    print("Installed WSD batch-size override")
else:
    raise RuntimeError(
        "Could not locate the expected WSD batch-size block. "
        "Inspect the dataloader manually before queueing."
    )
PY

###############################################################################
# 4. CREATE THE CONTEXT TRAINER IN ITS OWN MODULE
#
# This correctly imports the existing CutMix+EMA trainer.
###############################################################################
cat > "$CONTEXT_TRAINER_FILE" <<'PY'
import torch

from .nnUNetTrainer_CutMixStainEMA import nnUNetTrainer_CutMixStainEMA


class nnUNetTrainer_CutMixStainEMA_Context1024FT100(
    nnUNetTrainer_CutMixStainEMA
):
    """
    Fine-tunes a completed CutMix + stain-jitter + EMA model using a larger
    1024x1024 WholeSlideData context window.

    Inherited:
    - alpha-weighted Dice + focal loss;
    - CutMix;
    - stain jitter;
    - EMA;
    - general-best and class-specific-best checkpoint saving;
    - checkpoint_latest.pth and checkpoint_final.pth;
    - per-class metric logging.

    Changed:
    - patch size: 1024x1024;
    - online WSD batch size: 2;
    - initial learning rate: 0.0005;
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
PY

###############################################################################
# 5. ADD OR VERIFY THE 1024x1024 PLANS CONFIGURATION
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

# WholeSlideData extracts patches online.
context["data_identifier"] = base["data_identifier"]

plans["configurations"]["2d_context1024"] = context

with plans_path.open("w") as f:
    json.dump(plans, f, indent=4)

reloaded = json.loads(plans_path.read_text())

configured = reloaded["configurations"]["2d_context1024"]

assert configured["patch_size"] == [1024, 1024]
assert configured["batch_size"] == 2

print("Verified 2d_context1024 configuration:")
print(json.dumps(configured, indent=4))
PY

###############################################################################
# 6. CREATE THE CONTEXT-FINETUNING PYTHON RUNNER
#
# First execution:
# - loads the base fold's checkpoint_best.pth weights;
# - starts a new 100-epoch refinement run;
# - initializes EMA from the imported best weights.
#
# Continuation execution:
# - resumes from the context model's checkpoint_latest.pth.
###############################################################################
cat > run_context1024_ft100.py <<'PY'
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Mapping

import torch

from batchgenerators.utilities.file_and_folder_operations import load_json

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_CutMixStainEMA_Context1024FT100 import (
    nnUNetTrainer_CutMixStainEMA_Context1024FT100,
)


def unwrap_model(model):
    return getattr(model, "_orig_mod", model)


def normalized_state_dict(
    model,
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Normalize common DataParallel / compile prefixes before strict loading.
    """
    target = unwrap_model(model)
    target_keys = set(target.state_dict().keys())

    result: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        candidate = key

        if candidate.startswith("module.") and candidate[7:] in target_keys:
            candidate = candidate[7:]

        if (
            candidate.startswith("_orig_mod.")
            and candidate[10:] in target_keys
        ):
            candidate = candidate[10:]

        result[candidate] = value

    return result


def strict_load(model, state_dict, label: str) -> None:
    target = unwrap_model(model)

    normalized = normalized_state_dict(
        target,
        state_dict,
    )

    load_result = target.load_state_dict(
        normalized,
        strict=True,
    )

    print(
        f"Loaded {label} weights strictly. "
        f"Missing={load_result.missing_keys}, "
        f"Unexpected={load_result.unexpected_keys}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        required=True,
        type=int,
        choices=range(5),
    )

    args = parser.parse_args()

    fold = args.fold

    nnunet_preprocessed = Path(os.environ["nnUNet_preprocessed"])
    nnunet_results = Path(os.environ["nnUNet_results"])

    dataset_folder = (
        nnunet_preprocessed
        / "Dataset301_BEETLE"
    )

    plans = load_json(
        dataset_folder
        / "nnUNetWholeSlideDataPlans.json"
    )

    dataset_json = load_json(
        dataset_folder
        / "dataset.json"
    )

    base_model_dir = (
        nnunet_results
        / "Dataset301_BEETLE"
        / (
            "nnUNetTrainer_CutMixStainEMA__"
            "nnUNetWholeSlideDataPlans__"
            "wsd_None_iterator_nnunet_aug__2d"
        )
    )

    context_model_dir = (
        nnunet_results
        / "Dataset301_BEETLE"
        / (
            "nnUNetTrainer_CutMixStainEMA_Context1024FT100__"
            "nnUNetWholeSlideDataPlans__"
            "wsd_None_iterator_nnunet_aug__2d_context1024"
        )
    )

    source_checkpoint = (
        base_model_dir
        / f"fold_{fold}"
        / "checkpoint_best.pth"
    )

    output_folder = (
        context_model_dir
        / f"fold_{fold}"
    )

    latest_checkpoint = (
        output_folder
        / "checkpoint_latest.pth"
    )

    final_checkpoint = (
        output_folder
        / "checkpoint_final.pth"
    )

    if final_checkpoint.is_file():
        print(
            f"Context fold {fold} already completed: "
            f"{final_checkpoint}",
            flush=True,
        )
        return

    trainer = nnUNetTrainer_CutMixStainEMA_Context1024FT100(
        plans=plans,
        configuration="2d_context1024",
        fold=fold,
        dataset_json=dataset_json,
        unpack_dataset=True,
        device=torch.device("cuda"),
    )

    if latest_checkpoint.is_file():
        print(
            f"Resuming context fold {fold} from: "
            f"{latest_checkpoint}",
            flush=True,
        )

        trainer.load_checkpoint(
            str(latest_checkpoint)
        )

        print(
            f"Resumed context fold {fold} at epoch "
            f"{trainer.current_epoch}",
            flush=True,
        )

    else:
        if not source_checkpoint.is_file():
            raise FileNotFoundError(
                "Missing base-model best checkpoint: "
                f"{source_checkpoint}"
            )

        print(
            f"Starting context fold {fold} from base checkpoint: "
            f"{source_checkpoint}",
            flush=True,
        )

        trainer.initialize()

        checkpoint = torch.load(
            source_checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        if "network_weights" not in checkpoint:
            raise KeyError(
                "Base checkpoint is missing network_weights. "
                f"Available keys: {sorted(checkpoint.keys())}"
            )

        source_weights = checkpoint["network_weights"]

        strict_load(
            trainer.network,
            source_weights,
            "network",
        )

        if not trainer._ema_initialized:
            trainer._build_ema_model()

        strict_load(
            trainer.ema_model,
            source_weights,
            "EMA",
        )

        trainer.current_epoch = 0
        trainer._global_step = 0

        print(
            "Initialized fresh 100-epoch context refinement "
            "from the base general-best checkpoint",
            flush=True,
        )

    trainer.run_training()


if __name__ == "__main__":
    main()
PY

chmod +x run_context1024_ft100.py

###############################################################################
# 7. COMPILE AND IMPORT TEST BEFORE SUBMISSION
###############################################################################
source /scratch/$USER/virtual_environments/AIMI-BEETLE/bin/activate

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

export nnUNet_raw=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_raw

export nnUNet_preprocessed="$NNUNET_PREPROCESSED"

export nnUNet_results="$NNUNET_RESULTS"

python -m py_compile \
    "$DATALOADER_FILE" \
    "$CONTEXT_TRAINER_FILE" \
    run_context1024_ft100.py

python - <<'PY'
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_CutMixStainEMA import (
    nnUNetTrainer_CutMixStainEMA,
)

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_CutMixStainEMA_Context1024FT100 import (
    nnUNetTrainer_CutMixStainEMA_Context1024FT100,
)

assert issubclass(
    nnUNetTrainer_CutMixStainEMA_Context1024FT100,
    nnUNetTrainer_CutMixStainEMA,
)

print("Verified context trainer import")
print("Verified CutMix+EMA inheritance")
PY

###############################################################################
# 8. CREATE A ROBUST 12-HOUR CONTEXT-FINETUNING CHUNK LAUNCHER
###############################################################################
cat > run_context1024_ft100_chunk.slurm <<'SLURM'
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
#SBATCH --job-name=ctx1024ft
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
umask 002

ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase

FOLD="${1:?Usage: sbatch run_context1024_ft100_chunk.slurm <fold> <intermediate|final>}"

ROLE="${2:?Usage: sbatch run_context1024_ft100_chunk.slurm <fold> <intermediate|final>}"

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

MODEL_BASE="$nnUNet_results/Dataset301_BEETLE/nnUNetTrainer_CutMixStainEMA_Context1024FT100__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d_context1024"

RESULT_DIR="$MODEL_BASE/fold_${FOLD}"

LATEST="$RESULT_DIR/checkpoint_latest.pth"

FINAL="$RESULT_DIR/checkpoint_final.pth"

echo "======================================================================"
echo "BEETLE 1024x1024 CONTEXT FINE-TUNING CHUNK"
echo "======================================================================"
echo "Started:      $(date)"
echo "Node:         $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "Fold:         $FOLD"
echo "Role:         $ROLE"
echo "Result dir:   $RESULT_DIR"
echo "======================================================================"

if [[ -s "$FINAL" ]]; then
    echo "Context fold $FOLD already completed 100 epochs."
    exit 0
fi

set +e

timeout \
    --signal=TERM \
    --kill-after=60s \
    42600s \
    python -u run_context1024_ft100.py \
        --fold "$FOLD"

RC=$?

set -e

echo
echo "Fine-tuning command exit code: $RC"

###############################################################################
# Expected outcomes:
#   0   natural completion
#   124 expected timeout after 11h50m
###############################################################################
if [[ "$RC" -ne 0 && "$RC" -ne 124 ]]; then
    echo "ERROR: unexpected fine-tuning failure with exit code $RC." >&2
    exit 30
fi

if [[ -s "$FINAL" ]]; then
    echo "SUCCESS: context fold $FOLD completed 100 epochs."
    exit 0
fi

if [[ ! -s "$LATEST" ]]; then
    echo "ERROR: context run ended without checkpoint_latest.pth:" >&2
    echo "  $LATEST" >&2
    exit 31
fi

if [[ "$ROLE" == "intermediate" ]]; then
    echo "SUCCESS: context fold $FOLD remains resumable."
    echo "The next queued chunk will resume from checkpoint_latest.pth."
    exit 0
fi

if [[ "$ROLE" == "final" ]]; then
    echo "ERROR: context fold $FOLD remains incomplete after its allocated chunks." >&2
    echo "Queue one additional resume chunk before evaluation." >&2
    exit 32
fi

echo "ERROR: role must be intermediate or final." >&2
exit 33
SLURM

chmod +x run_context1024_ft100_chunk.slurm

###############################################################################
# 9. QUEUE ONLY THE CONTEXT-FINETUNING JOBS
#
# Wait until ALL five base evaluations complete.
#
# Two GPU lanes:
#
# Lane A:
#   fold 0 chunk 1 -> fold 0 chunk 2
#   -> fold 2 chunk 1 -> fold 2 chunk 2
#   -> fold 4 chunk 1 -> fold 4 chunk 2
#
# Lane B:
#   fold 1 chunk 1 -> fold 1 chunk 2
#   -> fold 3 chunk 1 -> fold 3 chunk 2
###############################################################################
T0_C1="$(
    sbatch \
        --parsable \
        --dependency=afterok:${ALL_BASE_EVALS} \
        --job-name=ctx1024ft-f0-c1 \
        run_context1024_ft100_chunk.slurm \
        0 intermediate
)"

T0_C2="$(
    sbatch \
        --parsable \
        --dependency=afterok:${T0_C1} \
        --job-name=ctx1024ft-f0-c2 \
        run_context1024_ft100_chunk.slurm \
        0 final
)"

T1_C1="$(
    sbatch \
        --parsable \
        --dependency=afterok:${ALL_BASE_EVALS} \
        --job-name=ctx1024ft-f1-c1 \
        run_context1024_ft100_chunk.slurm \
        1 intermediate
)"

T1_C2="$(
    sbatch \
        --parsable \
        --dependency=afterok:${T1_C1} \
        --job-name=ctx1024ft-f1-c2 \
        run_context1024_ft100_chunk.slurm \
        1 final
)"

T2_C1="$(
    sbatch \
        --parsable \
        --dependency=afterok:${T0_C2} \
        --job-name=ctx1024ft-f2-c1 \
        run_context1024_ft100_chunk.slurm \
        2 intermediate
)"

T2_C2="$(
    sbatch \
        --parsable \
        --dependency=afterok:${T2_C1} \
        --job-name=ctx1024ft-f2-c2 \
        run_context1024_ft100_chunk.slurm \
        2 final
)"

T3_C1="$(
    sbatch \
        --parsable \
        --dependency=afterok:${T1_C2} \
        --job-name=ctx1024ft-f3-c1 \
        run_context1024_ft100_chunk.slurm \
        3 intermediate
)"

T3_C2="$(
    sbatch \
        --parsable \
        --dependency=afterok:${T3_C1} \
        --job-name=ctx1024ft-f3-c2 \
        run_context1024_ft100_chunk.slurm \
        3 final
)"

T4_C1="$(
    sbatch \
        --parsable \
        --dependency=afterok:${T2_C2} \
        --job-name=ctx1024ft-f4-c1 \
        run_context1024_ft100_chunk.slurm \
        4 intermediate
)"

T4_C2="$(
    sbatch \
        --parsable \
        --dependency=afterok:${T4_C1} \
        --job-name=ctx1024ft-f4-c2 \
        run_context1024_ft100_chunk.slurm \
        4 final
)"

###############################################################################
# 10. SAVE READABLE MANIFEST
###############################################################################
MANIFEST="$ROOT/queued_context1024_finetuning.txt"

cat > "$MANIFEST" <<EOF
Submitted: $(date --iso-8601=seconds)

Prerequisite base evaluations:
  fold 0: $E0
  fold 1: $E1
  fold 2: $E2
  fold 3: $E3
  fold 4: $E4

Maximum simultaneous context-finetuning GPU jobs: 2

Lane A:
  fold 0 chunk 1: $T0_C1
  fold 0 chunk 2: $T0_C2
  fold 2 chunk 1: $T2_C1
  fold 2 chunk 2: $T2_C2
  fold 4 chunk 1: $T4_C1
  fold 4 chunk 2: $T4_C2

Lane B:
  fold 1 chunk 1: $T1_C1
  fold 1 chunk 2: $T1_C2
  fold 3 chunk 1: $T3_C1
  fold 3 chunk 2: $T3_C2
EOF

echo
echo "======================================================================"
echo "QUEUED ONLY THE FIVE CONTEXT-FINETUNING FOLDS"
echo "======================================================================"
echo
echo "Fine-tuning starts only after all five mirrored base evaluations finish."
echo
echo "Maximum simultaneous GPU jobs during fine-tuning: 2"
echo
echo "Lane A:"
echo "  fold 0 -> fold 2 -> fold 4"
echo
echo "Lane B:"
echo "  fold 1 -> fold 3"
echo
echo "No context evaluations were queued yet."
echo
echo "Manifest:"
echo "  $MANIFEST"
echo

squeue -u "$USER" \
    -o "%.18i %.32j %.2t %.12M %.45R"
