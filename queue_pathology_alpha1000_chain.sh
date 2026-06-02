#!/bin/bash
set -euo pipefail

###############################################################################
# Queue sequential 12-hour chunks for the alpha-weighted focal-loss run.
#
# Usage:
#   bash queue_pathology_alpha1000_chain.sh
#   bash queue_pathology_alpha1000_chain.sh 2
#   bash queue_pathology_alpha1000_chain.sh 3 0
#
# Arguments:
#   $1 = number of 12-hour chunks to queue; default: 2
#   $2 = fold; default: 0
#
# The first chunk starts a fresh run. Each later chunk starts only after the
# previous job ends, including a TIMEOUT. Continuation chunks resume from
# checkpoint_latest.pth. If training already reached 1000 epochs, a later queued
# chunk exits immediately because checkpoint_final.pth exists.
###############################################################################

N_CHUNKS="${1:-2}"
FOLD="${2:-0}"
SCRIPT="run_pathology_all_folds.slurm"

if ! [[ "$N_CHUNKS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: number of chunks must be a positive integer, got: $N_CHUNKS" >&2
    exit 2
fi

cd /home/tijnveldwijk/AIMI---BEETLE-Project-Phase
mkdir -p logs

FIRST_JOB_ID="$(sbatch --parsable "$SCRIPT" "$FOLD" fresh)"
echo "Queued chunk 1/$N_CHUNKS as job $FIRST_JOB_ID (fresh run)"

PREVIOUS_JOB_ID="$FIRST_JOB_ID"

for (( chunk=2; chunk<=N_CHUNKS; chunk++ )); do
    JOB_ID="$(sbatch \
        --parsable \
        --dependency="afterany:${PREVIOUS_JOB_ID}" \
        "$SCRIPT" \
        "$FOLD" \
        resume
    )"

    echo "Queued chunk $chunk/$N_CHUNKS as job $JOB_ID (resume after job $PREVIOUS_JOB_ID)"
    PREVIOUS_JOB_ID="$JOB_ID"
done

echo
echo "Dependency chain submitted."
echo "Inspect it with:"
echo "  squeue -u \$USER -o '%.18i %.28j %.10T %.20E %.10M %.10l %R'"
