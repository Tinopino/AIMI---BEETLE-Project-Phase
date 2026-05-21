#!/bin/bash
#SBATCH --job-name=beetle_path_fold0
#SBATCH --account=cseduimc037
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --nodelist=cn48
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=14G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pathology-fold0-cn48-%j.out
#SBATCH --error=logs/pathology-fold0-cn48-%j.out

set -euo pipefail
umask 002

cd ~/AIMI---BEETLE-Project-Phase

# Use the synced scratch venv on cn48
source /scratch/$USER/virtual_environments/AIMI-BEETLE/bin/activate
export PYTHONNOUSERSITE=1

# nnU-Net-for-pathology paths
export nnUNet_raw=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_raw
export nnUNet_preprocessed=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_preprocessed
export nnUNet_results=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results

# Keep cluster imports / dataloading stable
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export WANDB_MODE=disabled
export WANDB_DISABLED=true
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

echo "============================================================"
echo "BEETLE nnU-Net-for-pathology fold 0"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "User: $(whoami)"
echo "Working directory: $(pwd)"
echo "VENV=$VIRTUAL_ENV"
echo "Python: $(which python)"
echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-none}"
echo "============================================================"

echo "Checking Python packages..."
python -u - <<'PY'
import torch
import nnunetv2

print("torch:", torch.__version__, flush=True)
print("cuda available:", torch.cuda.is_available(), flush=True)
print("cuda device count:", torch.cuda.device_count(), flush=True)
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0), flush=True)

print("nnunetv2:", nnunetv2.__file__, flush=True)
PY

echo "Checking GPU..."
nvidia-smi

echo "Checking pathology dataset JSONs..."
ls -lh "$nnUNet_preprocessed/Dataset301_BEETLE/files.json"
ls -lh "$nnUNet_preprocessed/Dataset301_BEETLE/splits.json"
ls -lh "$nnUNet_preprocessed/Dataset301_BEETLE/dataset.json"
ls -lh "$nnUNet_preprocessed/Dataset301_BEETLE/used_cases.csv" || true

echo "Previewing fold setup..."
python -u - <<'PY'
import json
import os

base = os.path.join(os.environ["nnUNet_preprocessed"], "Dataset301_BEETLE")

with open(os.path.join(base, "splits.json")) as f:
    splits = json.load(f)

for fold, split in sorted(splits.items(), key=lambda x: int(x[0])):
    print(
        f"fold {fold}: "
        f"train WSIs={len(split['training'])}, "
        f"val WSIs={len(split['validation'])}",
        flush=True
    )
PY

echo "============================================================"
echo "Running pathology experiment planner..."
echo "============================================================"

python -u nnUNet_pathology/nnunetv2/experiment_planning/experiment_planners/pathology_experiment_planner.py \
  301 \
  --gpu_gb=11

echo "Files after planning:"
ls -lh "$nnUNet_preprocessed/Dataset301_BEETLE"

echo "============================================================"
echo "Starting GPU monitor..."
echo "============================================================"

(
  while true; do
    echo "===== GPU STATUS $(date) ====="
    nvidia-smi
    sleep 60
  done
) &
GPU_MONITOR_PID=$!

cleanup() {
  echo "Stopping GPU monitor..."
  kill "$GPU_MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "============================================================"
echo "Training fold 0 with standard pathology trainer..."
echo "Command:"
echo "python nnUNet_pathology/nnunetv2/run/run_training_pathology.py 301 0 nnUNetTrainer_WSD_wei_i0_nnunet_aug_json"
echo "============================================================"

python -u nnUNet_pathology/nnunetv2/run/run_training_pathology.py \
  301 \
  0 \
  nnUNetTrainer_WSD_wei_i0_nnunet_aug_json

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
