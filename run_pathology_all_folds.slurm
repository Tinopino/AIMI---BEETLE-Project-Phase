#!/bin/bash
#SBATCH --job-name=beetle_path_cv
#SBATCH --account=cseduimc037
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --nodelist=cn48
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=18G
#SBATCH --time=6:00:00
#SBATCH --output=logs/pathology-fold-%A_%a.out
#SBATCH --error=logs/pathology-fold-%A_%a.out

set -euo pipefail
umask 002

FOLD="${SLURM_ARRAY_TASK_ID:-${1:-0}}"
TRAINER="nnUNetTrainerPathologyFocalClassMetricsAlpha"

cd ~/AIMI---BEETLE-Project-Phase
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

echo "============================================================"
echo "BEETLE nnU-Net-for-pathology fold $FOLD"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-none}"
echo "VENV=$VIRTUAL_ENV"
echo "Trainer=$TRAINER"
echo "============================================================"

python -u - <<'PY'
import torch, nnunetv2
print("torch:", torch.__version__, flush=True)
print("cuda:", torch.cuda.is_available(), flush=True)
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", flush=True)
print("nnunetv2:", nnunetv2.__file__, flush=True)
PY

# Use more iterator CPUs now that the job reserves 10 cores
python -u - <<'PY'
import json, os
p = os.path.join(os.environ["nnUNet_preprocessed"], "Dataset301_BEETLE", "dataset.json")
with open(p) as f:
    d = json.load(f)
d["cpus"] = 10
with open(p, "w") as f:
    json.dump(d, f, indent=4)
print("Set dataset.json cpus=8", flush=True)
PY

echo "Running planner if needed..."
python -u nnUNet_pathology/nnunetv2/experiment_planning/experiment_planners/pathology_experiment_planner.py \
  301 \
  --gpu_gb=11
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
echo "Training fold $FOLD..."
python -u nnUNet_pathology/nnunetv2/run/run_training_pathology.py \
  301 \
  "$FOLD" \
  "$TRAINER"

echo "Finished fold $FOLD: $(date)"
