#!/bin/bash
#SBATCH --job-name=beetle_dicefocal_cn48
#SBATCH --account=cseduimc037
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --nodelist=cn48
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=14G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dicefocal-cn48-%j.out
#SBATCH --error=logs/dicefocal-cn48-%j.out

set -euo pipefail

cd ~/AIMI---BEETLE-Project-Phase
mkdir -p logs

source /scratch/$USER/virtual_environments/AIMI-BEETLE/bin/activate

export nnUNet_raw=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/dicefocal/nnUNet_raw
export nnUNet_preprocessed=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/dicefocal/nnUNet_preprocessed
export nnUNet_results=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/dicefocal/nnUNet_results

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Node: $(hostname)"
echo "Started: $(date)"
echo "VENV=$VIRTUAL_ENV"
echo "Python: $(which python)"
echo "nnUNetv2_train: $(which nnUNetv2_train)"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"

nvidia-smi

echo "Checking torch/CUDA..."
python -c 'import torch; print("torch", torch.__version__, flush=True); print("cuda", torch.cuda.is_available(), flush=True); print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", flush=True)'

echo "Checking trainer..."
python -c 'from nnunetv2.training.nnUNetTrainer.nnUNetTrainerDiceFocal import nnUNetTrainerDiceFocal; print("trainer", nnUNetTrainerDiceFocal.__name__, flush=True)'

echo "Checking preprocessed dataset..."
ls -lh "$nnUNet_preprocessed/Dataset301_BEETLE"
test -f "$nnUNet_preprocessed/Dataset301_BEETLE/nnUNetPlans.json"

echo "Starting GPU monitor..."
(
  while true; do
    echo "===== GPU STATUS $(date) ====="
    nvidia-smi
    sleep 60
  done
) &
GPU_MONITOR_PID=$!

trap "kill $GPU_MONITOR_PID 2>/dev/null || true" EXIT

echo "Starting nnU-Net DiceFocal training..."
echo "Command: nnUNetv2_train 301 2d 0 -tr nnUNetTrainerDiceFocal -device cuda"

nnUNetv2_train 301 2d 0 -tr nnUNetTrainerDiceFocal -device cuda

echo "Finished: $(date)"