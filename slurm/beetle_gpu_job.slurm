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
#SBATCH --job-name=beetle
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
umask 002

ROOT="${AIMI_REPO_ROOT:-/home/tijnveldwijk/AIMI---BEETLE-Project-Phase}"
cd "${ROOT}"
mkdir -p logs

source "${BEETLE_VENV:-/scratch/${USER}/virtual_environments/AIMI-BEETLE}/bin/activate"
source "${BEETLE_ENV_FILE:-reproducibility/configs/cluster_paths.env}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

if [[ "$#" -lt 1 ]]; then
    echo "Usage:"
    echo "  sbatch slurm/beetle_gpu_job.slurm train.py --experiment cutmix_stain_ema --fold 0"
    echo "  sbatch slurm/beetle_gpu_job.slurm validate.py wsi --experiment context1024_ft100 --fold 0 --save-visuals"
    echo "  sbatch slurm/beetle_gpu_job.slurm validate.py external --experiment context1024_ft100"
    exit 2
fi

echo "Executing: python -u $*"
python -u "$@"
