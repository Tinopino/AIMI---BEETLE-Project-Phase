#!/usr/bin/env bash
set -euo pipefail

ACCOUNT=cseduimc037
NODE=cn48

VENV_NAME=AIMI-BEETLE
VENV_PARENT=/scratch/$USER/virtual_environments
VENV_PATH=$VENV_PARENT/$VENV_NAME

if [[ "$HOSTNAME" != "cn84"* ]]; then
  echo "Run this script from cn84"
  exit 1
fi

echo "Checking source venv on cn84..."
if [[ ! -d "$VENV_PATH" ]]; then
  echo "ERROR: source venv does not exist: $VENV_PATH"
  exit 1
fi

echo "Source venv size:"
du -sh "$VENV_PATH"

echo "Checking scratch space on $NODE..."
srun -p csedu -A "$ACCOUNT" --qos=csedu-normal -w "$NODE" df -h /scratch

echo "Creating target folder on $NODE..."
srun -p csedu -A "$ACCOUNT" --qos=csedu-normal -w "$NODE" mkdir -p "$VENV_PARENT"

echo "Syncing venv to $NODE..."
srun -p csedu -A "$ACCOUNT" --qos=csedu-normal -w "$NODE" \
  rsync -ah --delete cn84:"$VENV_PATH/" "$VENV_PATH/"

echo "Testing synced venv on $NODE..."
srun -p csedu -A "$ACCOUNT" --qos=csedu-normal -w "$NODE" \
  "$VENV_PATH/bin/python" -c 'import torch; print("torch", torch.__version__); import nnunetv2; print("nnunetv2", nnunetv2.__file__)'

echo "Done syncing to $NODE"
