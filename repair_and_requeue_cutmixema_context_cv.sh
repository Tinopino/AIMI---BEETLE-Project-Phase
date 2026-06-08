#!/usr/bin/env bash
set -euo pipefail
umask 002

###############################################################################
# CURRENT RUNNING FOLD-0 JOB: DO NOT CANCEL
###############################################################################
FOLD0_ACTIVE_JOB=10406217

ROOT=/home/tijnveldwijk/AIMI---BEETLE-Project-Phase

BASE_MODEL_DIR=/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results/Dataset301_BEETLE/nnUNetTrainer_CutMixStainEMA__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d

cd "$ROOT"
mkdir -p logs

###############################################################################
# 1. CANCEL ONLY STALE PENDING JOBS FROM THE BROKEN SUBMISSION
#
# Use "|" as a separator so only clean numeric job IDs are passed to scancel.
###############################################################################
mapfile -t STALE_IDS < <(
    squeue -h -u "$USER" -o "%A|%j" |
    awk -F'|' '
        $2 ~ /^cutmixema-f[0-4]-c[12]$/ ||
        $2 ~ /^eval-cema1000-f[0-4]$/ ||
        $2 == "aggregate-cema1000" ||
        $2 ~ /^ctx1024-ft-f[0-4]$/ ||
        $2 ~ /^eval-ctx1024-f[0-4]$/ ||
        $2 == "aggregate-final-cv"
        { print $1 }
    ' |
    sort -u
)

if [[ "${#STALE_IDS[@]}" -gt 0 ]]; then
    echo "Cancelling stale pending jobs:"
    printf '  %s\n' "${STALE_IDS[@]}"
    scancel "${STALE_IDS[@]}"
else
    echo "No stale pending jobs found."
fi

###############################################################################
# 2. FIX TIMEOUT SYNTAX
#
# GNU timeout on this cluster accepts 42600s, not 11h50m.
# 42600 seconds = 11 hours 50 minutes.
###############################################################################
sed -i 's/11h50m/42600s/g' \
    run_cv_cutmixema1000_chunk.slurm \
    run_cv_cutmixema_context1024_ft100.slurm

echo
echo "Timeout configuration:"
grep -n "42600s" \
    run_cv_cutmixema1000_chunk.slurm \
    run_cv_cutmixema_context1024_ft100.slurm

###############################################################################
# 3. CLEAN OUTPUT FILENAMES
#
# %a is only meaningful for SLURM arrays. These jobs are not arrays.
###############################################################################
sed -i \
    's|logs/cutmixema1000-fold-%a-%j.out|logs/cutmixema1000-%x-%j.out|g' \
    run_cv_cutmixema1000_chunk.slurm

sed -i \
    's|logs/cutmixema1000-fold-%a-%j.err|logs/cutmixema1000-%x-%j.err|g' \
    run_cv_cutmixema1000_chunk.slurm

sed -i \
    's|logs/context1024-ft100-fold-%a-%j.out|logs/context1024-ft100-%x-%j.out|g' \
    run_cv_cutmixema_context1024_ft100.slurm

sed -i \
    's|logs/context1024-ft100-fold-%a-%j.err|logs/context1024-ft100-%x-%j.err|g' \
    run_cv_cutmixema_context1024_ft100.slurm

###############################################################################
# 4. REMOVE EMPTY FOLD-1 TO FOLD-4 DIRECTORIES CREATED BY THE FAILED JOBS
#
# Abort rather than delete anything if a real checkpoint unexpectedly exists.
###############################################################################
for FOLD in 1 2 3 4; do
    DIR="$BASE_MODEL_DIR/fold_${FOLD}"

    if [[ -d "$DIR" ]]; then
        if find "$DIR" -maxdepth 1 -type f -name 'checkpoint*.pth' \
            -print -quit | grep -q .; then
            echo "ERROR: fold $FOLD unexpectedly contains a checkpoint:" >&2
            find "$DIR" -maxdepth 1 -type f -name 'checkpoint*.pth' -print >&2
            exit 20
        fi

        echo "Removing empty failed-run folder:"
        echo "  $DIR"
        rm -rf "$DIR"
    fi
done

join_colon() {
    local IFS=:
    echo "$*"
}

###############################################################################
# 5. STAGE 1: 1000-EPOCH CUTMIX + STAIN JITTER + EMA
#
# Strict two-GPU lane structure:
#
# Lane A:
#   existing fold 0
#   -> fold 0 resume
#   -> fold 2 fresh -> fold 2 resume
#   -> fold 4 fresh -> fold 4 resume
#
# Lane B:
#   fold 1 fresh -> fold 1 resume
#   -> fold 3 fresh -> fold 3 resume
###############################################################################
F0_C2=$(
    sbatch --parsable \
        --dependency=afterany:${FOLD0_ACTIVE_JOB} \
        --job-name=cutmixema-f0-c2 \
        run_cv_cutmixema1000_chunk.slurm \
        0 resume chunk2
)

F1_C1=$(
    sbatch --parsable \
        --job-name=cutmixema-f1-c1 \
        run_cv_cutmixema1000_chunk.slurm \
        1 fresh chunk1
)

F1_C2=$(
    sbatch --parsable \
        --dependency=afterany:${F1_C1} \
        --job-name=cutmixema-f1-c2 \
        run_cv_cutmixema1000_chunk.slurm \
        1 resume chunk2
)

F2_C1=$(
    sbatch --parsable \
        --dependency=afterany:${F0_C2} \
        --job-name=cutmixema-f2-c1 \
        run_cv_cutmixema1000_chunk.slurm \
        2 fresh chunk1
)

F2_C2=$(
    sbatch --parsable \
        --dependency=afterany:${F2_C1} \
        --job-name=cutmixema-f2-c2 \
        run_cv_cutmixema1000_chunk.slurm \
        2 resume chunk2
)

F3_C1=$(
    sbatch --parsable \
        --dependency=afterany:${F1_C2} \
        --job-name=cutmixema-f3-c1 \
        run_cv_cutmixema1000_chunk.slurm \
        3 fresh chunk1
)

F3_C2=$(
    sbatch --parsable \
        --dependency=afterany:${F3_C1} \
        --job-name=cutmixema-f3-c2 \
        run_cv_cutmixema1000_chunk.slurm \
        3 resume chunk2
)

F4_C1=$(
    sbatch --parsable \
        --dependency=afterany:${F2_C2} \
        --job-name=cutmixema-f4-c1 \
        run_cv_cutmixema1000_chunk.slurm \
        4 fresh chunk1
)

F4_C2=$(
    sbatch --parsable \
        --dependency=afterany:${F4_C1} \
        --job-name=cutmixema-f4-c2 \
        run_cv_cutmixema1000_chunk.slurm \
        4 resume chunk2
)

BASE_FINAL_DEP=$(join_colon "$F0_C2" "$F1_C2" "$F2_C2" "$F3_C2" "$F4_C2")

###############################################################################
# 6. STAGE 2: MIRRORED FULL VALIDATION OF THE FIVE BASE MODELS
#
# Evaluations are also limited to two concurrent GPU jobs.
###############################################################################
E0=$(
    sbatch --parsable \
        --dependency=afterok:${BASE_FINAL_DEP} \
        --job-name=eval-cema1000-f0 \
        run_cv_full_eval.slurm \
        base 0
)

E1=$(
    sbatch --parsable \
        --dependency=afterok:${BASE_FINAL_DEP} \
        --job-name=eval-cema1000-f1 \
        run_cv_full_eval.slurm \
        base 1
)

E2=$(
    sbatch --parsable \
        --dependency=afterok:${E0} \
        --job-name=eval-cema1000-f2 \
        run_cv_full_eval.slurm \
        base 2
)

E3=$(
    sbatch --parsable \
        --dependency=afterok:${E1} \
        --job-name=eval-cema1000-f3 \
        run_cv_full_eval.slurm \
        base 3
)

E4=$(
    sbatch --parsable \
        --dependency=afterok:${E2} \
        --job-name=eval-cema1000-f4 \
        run_cv_full_eval.slurm \
        base 4
)

BASE_EVAL_DEP=$(join_colon "$E0" "$E1" "$E2" "$E3" "$E4")

BASE_AGG=$(
    sbatch --parsable \
        --dependency=afterok:${BASE_EVAL_DEP} \
        --job-name=aggregate-cema1000 \
        run_cv_aggregate.slurm \
        base
)

###############################################################################
# 7. STAGE 3: 100-EPOCH 1024x1024 CONTEXT FINE-TUNING
#
# Again limited to two concurrent GPU jobs.
###############################################################################
T0=$(
    sbatch --parsable \
        --dependency=afterok:${BASE_AGG} \
        --job-name=ctx1024-ft-f0 \
        run_cv_cutmixema_context1024_ft100.slurm \
        0
)

T1=$(
    sbatch --parsable \
        --dependency=afterok:${BASE_AGG} \
        --job-name=ctx1024-ft-f1 \
        run_cv_cutmixema_context1024_ft100.slurm \
        1
)

T2=$(
    sbatch --parsable \
        --dependency=afterok:${T0} \
        --job-name=ctx1024-ft-f2 \
        run_cv_cutmixema_context1024_ft100.slurm \
        2
)

T3=$(
    sbatch --parsable \
        --dependency=afterok:${T1} \
        --job-name=ctx1024-ft-f3 \
        run_cv_cutmixema_context1024_ft100.slurm \
        3
)

T4=$(
    sbatch --parsable \
        --dependency=afterok:${T2} \
        --job-name=ctx1024-ft-f4 \
        run_cv_cutmixema_context1024_ft100.slurm \
        4
)

###############################################################################
# 8. STAGE 4: MIRRORED FULL VALIDATION OF THE FIVE CONTEXT MODELS
###############################################################################
ALL_CONTEXT_TRAINS=$(join_colon "$T0" "$T1" "$T2" "$T3" "$T4")

CE0=$(
    sbatch --parsable \
        --dependency=afterok:${ALL_CONTEXT_TRAINS} \
        --job-name=eval-ctx1024-f0 \
        run_cv_full_eval.slurm \
        context 0
)

CE1=$(
    sbatch --parsable \
        --dependency=afterok:${ALL_CONTEXT_TRAINS} \
        --job-name=eval-ctx1024-f1 \
        run_cv_full_eval.slurm \
        context 1
)

CE2=$(
    sbatch --parsable \
        --dependency=afterok:${CE0} \
        --job-name=eval-ctx1024-f2 \
        run_cv_full_eval.slurm \
        context 2
)

CE3=$(
    sbatch --parsable \
        --dependency=afterok:${CE1} \
        --job-name=eval-ctx1024-f3 \
        run_cv_full_eval.slurm \
        context 3
)

CE4=$(
    sbatch --parsable \
        --dependency=afterok:${CE2} \
        --job-name=eval-ctx1024-f4 \
        run_cv_full_eval.slurm \
        context 4
)

CONTEXT_EVAL_DEP=$(join_colon "$CE0" "$CE1" "$CE2" "$CE3" "$CE4")

FINAL_AGG=$(
    sbatch --parsable \
        --dependency=afterok:${CONTEXT_EVAL_DEP} \
        --job-name=aggregate-final-cv \
        run_cv_aggregate.slurm \
        both
)

###############################################################################
# 9. WRITE JOB MANIFEST
###############################################################################
MANIFEST=queued_cutmixema_context_cv_jobs_repaired.txt

cat > "$MANIFEST" <<EOF
Submitted: $(date --iso-8601=seconds)

Existing fold-0 chunk 1:
  $FOLD0_ACTIVE_JOB

Stage 1: 1000-epoch CutMix + EMA
  fold 0 chunk 2: $F0_C2
  fold 1 chunk 1: $F1_C1
  fold 1 chunk 2: $F1_C2
  fold 2 chunk 1: $F2_C1
  fold 2 chunk 2: $F2_C2
  fold 3 chunk 1: $F3_C1
  fold 3 chunk 2: $F3_C2
  fold 4 chunk 1: $F4_C1
  fold 4 chunk 2: $F4_C2

Stage 2: mirrored base evaluations
  fold 0: $E0
  fold 1: $E1
  fold 2: $E2
  fold 3: $E3
  fold 4: $E4
  aggregate: $BASE_AGG

Stage 3: 1024x1024 context fine-tunes
  fold 0: $T0
  fold 1: $T1
  fold 2: $T2
  fold 3: $T3
  fold 4: $T4

Stage 4: mirrored context evaluations
  fold 0: $CE0
  fold 1: $CE1
  fold 2: $CE2
  fold 3: $CE3
  fold 4: $CE4

Stage 5: final aggregate
  $FINAL_AGG
EOF

echo
echo "============================================================"
echo "REPAIRED TWO-GPU PIPELINE QUEUED"
echo "============================================================"
echo
echo "GPU lane A:"
echo "  existing fold 0 -> fold 0 resume -> fold 2 -> fold 4"
echo
echo "GPU lane B:"
echo "  fold 1 -> fold 3"
echo
echo "Maximum simultaneous GPU jobs from this pipeline: 2"
echo
echo "Manifest:"
echo "  $ROOT/$MANIFEST"
echo

squeue -u "$USER" \
    -o "%.18i %.32j %.2t %.12M %.45R"
