#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Locate exactly one queued job by name.
###############################################################################
job_id() {
    local name="$1"
    mapfile -t ids < <(
        squeue -h -u "$USER" -n "$name" -o "%A"
    )

    if [[ "${#ids[@]}" -ne 1 ]]; then
        echo "ERROR: expected exactly one queued job named '$name'," >&2
        echo "but found ${#ids[@]}." >&2
        echo >&2
        echo "Inspect the queue with:" >&2
        echo "  squeue -u \$USER -o '%.18i %.32j %.2t %.12M %.40R'" >&2
        exit 1
    fi

    echo "${ids[0]}"
}

###############################################################################
# Resolve all jobs created by setup_and_queue_cutmixema_context_cv.sh.
###############################################################################

# Stage 1: 1000-epoch CutMix + EMA training
F0_C2=$(job_id cutmixema-f0-c2)

F1_C1=$(job_id cutmixema-f1-c1)
F1_C2=$(job_id cutmixema-f1-c2)

F2_C1=$(job_id cutmixema-f2-c1)
F2_C2=$(job_id cutmixema-f2-c2)

F3_C1=$(job_id cutmixema-f3-c1)
F3_C2=$(job_id cutmixema-f3-c2)

F4_C1=$(job_id cutmixema-f4-c1)
F4_C2=$(job_id cutmixema-f4-c2)

# Stage 2: baseline full-fold mirrored evaluations
BASE_E0=$(job_id eval-cema1000-f0)
BASE_E1=$(job_id eval-cema1000-f1)
BASE_E2=$(job_id eval-cema1000-f2)
BASE_E3=$(job_id eval-cema1000-f3)
BASE_E4=$(job_id eval-cema1000-f4)

BASE_AGG=$(job_id aggregate-cema1000)

# Stage 3: 1024x1024 context fine-tuning
CTX_T0=$(job_id ctx1024-ft-f0)
CTX_T1=$(job_id ctx1024-ft-f1)
CTX_T2=$(job_id ctx1024-ft-f2)
CTX_T3=$(job_id ctx1024-ft-f3)
CTX_T4=$(job_id ctx1024-ft-f4)

# Stage 4: context-model full-fold mirrored evaluations
CTX_E0=$(job_id eval-ctx1024-f0)
CTX_E1=$(job_id eval-ctx1024-f1)
CTX_E2=$(job_id eval-ctx1024-f2)
CTX_E3=$(job_id eval-ctx1024-f3)
CTX_E4=$(job_id eval-ctx1024-f4)

###############################################################################
# Make sure jobs that need new dependencies have not started already.
###############################################################################
TO_PATCH=(
    "$F2_C1" "$F3_C1" "$F4_C1"
    "$BASE_E0" "$BASE_E1" "$BASE_E2" "$BASE_E3" "$BASE_E4"
    "$CTX_T0" "$CTX_T1" "$CTX_T2" "$CTX_T3" "$CTX_T4"
    "$CTX_E0" "$CTX_E1" "$CTX_E2" "$CTX_E3" "$CTX_E4"
)

for id in "${TO_PATCH[@]}"; do
    state=$(squeue -h -j "$id" -o "%T")

    if [[ "$state" != "PENDING" ]]; then
        echo "ERROR: job $id is already in state '$state'." >&2
        echo "No dependencies were changed." >&2
        echo "Paste the output of squeue -u \$USER before modifying anything." >&2
        exit 2
    fi
done

###############################################################################
# Hold jobs briefly while updating dependencies.
###############################################################################
for id in "${TO_PATCH[@]}"; do
    scontrol hold "$id"
done

release_jobs() {
    for id in "${TO_PATCH[@]}"; do
        scontrol release "$id" 2>/dev/null || true
    done
}

trap release_jobs EXIT

###############################################################################
# STAGE 1: two GPU lanes
#
# Lane A:
#   existing fold-0 chunk 1
#   -> fold-0 resume
#   -> fold-2 chunk 1 -> fold-2 resume
#   -> fold-4 chunk 1 -> fold-4 resume
#
# Lane B:
#   fold-1 chunk 1 -> fold-1 resume
#   -> fold-3 chunk 1 -> fold-3 resume
###############################################################################
scontrol update JobId="$F2_C1" Dependency="afterany:$F0_C2"
scontrol update JobId="$F3_C1" Dependency="afterany:$F1_C2"
scontrol update JobId="$F4_C1" Dependency="afterany:$F2_C2"

###############################################################################
# STAGE 2: at most two mirrored evaluations simultaneously
###############################################################################
BASE_FINALS="$F0_C2:$F1_C2:$F2_C2:$F3_C2:$F4_C2"

scontrol update JobId="$BASE_E0" Dependency="afterok:$BASE_FINALS"
scontrol update JobId="$BASE_E1" Dependency="afterok:$BASE_FINALS"
scontrol update JobId="$BASE_E2" Dependency="afterok:$BASE_FINALS:$BASE_E0"
scontrol update JobId="$BASE_E3" Dependency="afterok:$BASE_FINALS:$BASE_E1"
scontrol update JobId="$BASE_E4" Dependency="afterok:$BASE_FINALS:$BASE_E2"

###############################################################################
# STAGE 3: at most two 1024x1024 fine-tunes simultaneously
###############################################################################
scontrol update JobId="$CTX_T0" Dependency="afterok:$BASE_AGG"
scontrol update JobId="$CTX_T1" Dependency="afterok:$BASE_AGG"
scontrol update JobId="$CTX_T2" Dependency="afterok:$BASE_AGG:$CTX_T0"
scontrol update JobId="$CTX_T3" Dependency="afterok:$BASE_AGG:$CTX_T1"
scontrol update JobId="$CTX_T4" Dependency="afterok:$BASE_AGG:$CTX_T2"

###############################################################################
# STAGE 4: at most two context-model evaluations simultaneously
###############################################################################
CTX_TRAINS="$CTX_T0:$CTX_T1:$CTX_T2:$CTX_T3:$CTX_T4"

scontrol update JobId="$CTX_E0" Dependency="afterok:$CTX_TRAINS"
scontrol update JobId="$CTX_E1" Dependency="afterok:$CTX_TRAINS"
scontrol update JobId="$CTX_E2" Dependency="afterok:$CTX_TRAINS:$CTX_E0"
scontrol update JobId="$CTX_E3" Dependency="afterok:$CTX_TRAINS:$CTX_E1"
scontrol update JobId="$CTX_E4" Dependency="afterok:$CTX_TRAINS:$CTX_E2"

###############################################################################
# Release updated jobs.
###############################################################################
release_jobs
trap - EXIT

echo
echo "============================================================"
echo "PIPELINE CAPPED AT TWO SIMULTANEOUS GPU JOBS"
echo "============================================================"
echo
echo "Stage-1 training lanes:"
echo "  lane A: current fold 0 -> fold 0 resume -> fold 2 -> fold 4"
echo "  lane B: fold 1 -> fold 3"
echo
echo "Later evaluation and context-fine-tuning stages are also capped at two GPUs."
echo

squeue -u "$USER" \
  -o "%.18i %.32j %.2t %.12M %.45R"
