#!/bin/bash
#SBATCH --job-name=beetle_inspect
#SBATCH --account=cseduimc037
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

cd /home/tijnveldwijk/AIMI---BEETLE-Project-Phase
mkdir -p logs outputs

source /vol/csedu-nobackup/course/IMC037_aimi/group14/venvs/tijn/beetle/bin/activate

python preprocessing/dataset_inspection.py \
  --image-dir /vol/csedu-nobackup/course/IMC037_aimi/group14/aalina/images/development/wsis \
  --mask-dir /vol/csedu-nobackup/course/IMC037_aimi/group14/aalina/annotations/masks \
  --out outputs/dataset_audit_part_${SLURM_ARRAY_TASK_ID}.csv \
  --chunk-size 8192 \
  --flush-every 10 \
  --skip-rgb-channel-check \
  --num-shards 4 \
  --shard-index ${SLURM_ARRAY_TASK_ID}
