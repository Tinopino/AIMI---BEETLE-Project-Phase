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
