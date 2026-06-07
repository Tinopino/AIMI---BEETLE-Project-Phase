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
