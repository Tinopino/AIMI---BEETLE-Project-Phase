#!/usr/bin/env bash
#SBATCH --account=cseduimc037
#SBATCH --partition=csedu
#SBATCH --nodelist=cn48
#SBATCH --qos=csedu-normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --job-name=filter_wfhard
#SBATCH --output=logs/filter-wfhard-%j.out
#SBATCH --error=logs/filter-wfhard-%j.err

set -euo pipefail

cd /home/tijnveldwijk/AIMI---BEETLE-Project-Phase

source /scratch/$USER/virtual_environments/AIMI-BEETLE/bin/activate

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

python -u filter_wfhard_manifest.py
