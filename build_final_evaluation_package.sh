#!/bin/bash

set -euo pipefail
umask 002

###############################################################################
# BEETLE final evaluation package
#
# Policy:
# - Copy every non-heavy file from every experiment.
# - Preserve the original experiment and fold structure.
# - Exclude model weights and large non-visual files.
# - Copy all qualitative visual files, including large ones.
# - Record copied and skipped files in manifests.
###############################################################################

DRY_RUN="${DRY_RUN:-0}"
MAX_NONVISUAL_MB="${MAX_NONVISUAL_MB:-100}"

GROUP_ROOT="/vol/csedu-nobackup/course/IMC037_aimi/group14"
PATHOLOGY_ROOT="${GROUP_ROOT}/nnunet/tijn/pathology"

RESULTS_ROOT="${PATHOLOGY_ROOT}/nnUNet_results/Dataset301_BEETLE"
VISUALS_ROOT="${PATHOLOGY_ROOT}/validation_visuals"

REPO_ROOT="/home/tijnveldwijk/AIMI---BEETLE-Project-Phase"
HOME_ROOT="/home/tijnveldwijk"

PACKAGE_PARENT="${GROUP_ROOT}/final_evaluation_packages"
PACKAGE_ROOT="${PACKAGE_PARENT}/BEETLE_final_evaluation_package"

CORE="${PACKAGE_ROOT}/paper_core"
VISUAL_ARCHIVE="${PACKAGE_ROOT}/visual_archive"

export DRY_RUN
export MAX_NONVISUAL_MB
export RESULTS_ROOT
export VISUALS_ROOT
export REPO_ROOT
export HOME_ROOT
export PACKAGE_ROOT
export CORE
export VISUAL_ARCHIVE

echo "======================================================================"
echo "BEETLE FINAL EVALUATION PACKAGE BUILDER"
echo "======================================================================"
echo "DRY_RUN:            ${DRY_RUN}"
echo "MAX_NONVISUAL_MB:   ${MAX_NONVISUAL_MB}"
echo "RESULTS_ROOT:       ${RESULTS_ROOT}"
echo "VISUALS_ROOT:       ${VISUALS_ROOT}"
echo "PACKAGE_ROOT:       ${PACKAGE_ROOT}"
echo "Started:            $(date)"
echo "======================================================================"

if [[ ! -d "${RESULTS_ROOT}" ]]; then
    echo "ERROR: missing results directory: ${RESULTS_ROOT}"
    exit 1
fi

if [[ ! -d "${VISUALS_ROOT}" ]]; then
    echo "ERROR: missing visual-analysis directory: ${VISUALS_ROOT}"
    exit 1
fi

if [[ "${DRY_RUN}" == "0" ]]; then
    case "${PACKAGE_ROOT}" in
        "${PACKAGE_PARENT}/BEETLE_final_evaluation_package")
            rm -rf "${PACKAGE_ROOT}"
            ;;
        *)
            echo "ERROR: refusing to remove unexpected directory: ${PACKAGE_ROOT}"
            exit 1
            ;;
    esac

    mkdir -p \
        "${CORE}/experiments_raw" \
        "${CORE}/experiments_by_alias" \
        "${CORE}/external_metrics" \
        "${CORE}/report_tables" \
        "${CORE}/inference_outputs" \
        "${CORE}/repo_snapshot" \
        "${CORE}/logs" \
        "${CORE}/evaluation_inputs" \
        "${CORE}/manifests" \
        "${VISUAL_ARCHIVE}/validation_visuals" \
        "${VISUAL_ARCHIVE}/selected_for_paper"
fi

###############################################################################
# 1. Copy all non-heavy files while preserving the raw directory structure.
###############################################################################

echo
echo "[1/8] Copying all non-heavy experiment files, repository files, logs,"
echo "      and external-inference outputs..."

python - <<'PY'
import os
import shutil
from pathlib import Path

dry_run = os.environ["DRY_RUN"] == "1"
max_nonvisual_bytes = int(os.environ["MAX_NONVISUAL_MB"]) * 1024 * 1024

results_root = Path(os.environ["RESULTS_ROOT"])
repo_root = Path(os.environ["REPO_ROOT"])
home_root = Path(os.environ["HOME_ROOT"])
core = Path(os.environ["CORE"])

visual_extensions = {
    ".png", ".jpg", ".jpeg", ".svg", ".eps", ".pdf"
}

weight_extensions = {
    ".pth", ".pt", ".ckpt", ".safetensors"
}

copied = []
skipped = []


def should_skip_file(path: Path, allow_large: bool = False):
    suffix = path.suffix.lower()
    name_lower = path.name.lower()

    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return True, "missing_during_copy", 0

    if suffix in weight_extensions:
        return True, "model_weight_or_checkpoint", size

    if (
        "checkpoint" in name_lower
        and suffix in {".pth", ".pt", ".ckpt", ".safetensors"}
    ):
        return True, "model_weight_or_checkpoint", size

    if (
        not allow_large
        and size > max_nonvisual_bytes
        and suffix not in visual_extensions
    ):
        return True, "large_nonvisual_file", size

    return False, "", size


def copy_tree(
    source: Path,
    destination: Path,
    category: str,
    *,
    allow_large: bool = False,
    excluded_directories=None,
):
    if excluded_directories is None:
        excluded_directories = set()

    if not source.exists():
        print(f"WARNING: source does not exist: {source}")
        return

    for root, directories, files in os.walk(source, followlinks=False):
        root_path = Path(root)

        directories[:] = [
            directory
            for directory in directories
            if directory not in excluded_directories
        ]

        for filename in files:
            source_path = root_path / filename

            if source_path.is_symlink():
                skipped.append(
                    (
                        str(source_path),
                        0,
                        category,
                        "symbolic_link_not_followed",
                    )
                )
                continue

            relative_path = source_path.relative_to(source)
            destination_path = destination / relative_path

            skip, reason, size = should_skip_file(
                source_path,
                allow_large=allow_large,
            )

            if skip:
                skipped.append(
                    (
                        str(source_path),
                        size,
                        category,
                        reason,
                    )
                )
                continue

            copied.append(
                (
                    str(source_path),
                    str(destination_path),
                    size,
                    category,
                )
            )

            if not dry_run:
                destination_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                shutil.copy2(source_path, destination_path)


copy_tree(
    results_root,
    core / "experiments_raw",
    "nnunet_results",
)

copy_tree(
    repo_root,
    core / "repo_snapshot",
    "repository_snapshot",
    excluded_directories={
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "inference_outputs",
        "logs",
    },
)

copy_tree(
    repo_root / "logs",
    core / "logs",
    "repository_logs",
    allow_large=True,
)

copy_tree(
    repo_root / "inference_outputs",
    core / "inference_outputs",
    "external_inference_outputs",
    allow_large=True,
)

# Copy fold-specific validation CSV files stored in the user's home directory.
for csv_path in sorted(home_root.glob("fold*_validation_inference_inputs.csv")):
    destination_path = core / "evaluation_inputs" / csv_path.name
    size = csv_path.stat().st_size

    copied.append(
        (
            str(csv_path),
            str(destination_path),
            size,
            "validation_input_csv",
        )
    )

    if not dry_run:
        destination_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        shutil.copy2(csv_path, destination_path)

print()
print("Selective copy summary")
print("----------------------")
print(f"Copied files:  {len(copied)}")
print(f"Skipped files: {len(skipped)}")

copied_bytes = sum(row[2] for row in copied)
skipped_bytes = sum(row[1] for row in skipped)

print(f"Copied bytes:  {copied_bytes:,}")
print(f"Skipped bytes: {skipped_bytes:,}")

if not dry_run:
    manifest_directory = core / "manifests"
    manifest_directory.mkdir(parents=True, exist_ok=True)

    copied_manifest = manifest_directory / "copied_files.tsv"
    skipped_manifest = manifest_directory / "skipped_files.tsv"

    with copied_manifest.open("w", encoding="utf-8") as handle:
        handle.write("source_path\tdestination_path\tsize_bytes\tcategory\n")

        for source, destination, size, category in copied:
            handle.write(
                f"{source}\t{destination}\t{size}\t{category}\n"
            )

    with skipped_manifest.open("w", encoding="utf-8") as handle:
        handle.write("source_path\tsize_bytes\tcategory\treason\n")

        for source, size, category, reason in skipped:
            handle.write(
                f"{source}\t{size}\t{category}\t{reason}\n"
            )
else:
    print()
    print("Largest skipped files:")
    print("----------------------")

    for source, size, category, reason in sorted(
        skipped,
        key=lambda row: row[1],
        reverse=True,
    )[:40]:
        print(
            f"{size / (1024 ** 3):8.2f} GiB | "
            f"{reason:28s} | "
            f"{source}"
        )
PY

###############################################################################
# 2. Copy the complete validation-visual archive without size filtering.
###############################################################################

echo
echo "[2/8] Copying the complete validation-visual archive..."

if [[ "${DRY_RUN}" == "1" ]]; then
    rsync -an \
        --stats \
        "${VISUALS_ROOT}/" \
        "${VISUAL_ARCHIVE}/validation_visuals/"
else
    rsync -a \
        --info=progress2 \
        "${VISUALS_ROOT}/" \
        "${VISUAL_ARCHIVE}/validation_visuals/"
fi

###############################################################################
# Stop here for dry-run mode.
###############################################################################

if [[ "${DRY_RUN}" == "1" ]]; then
    echo
    echo "======================================================================"
    echo "DRY RUN COMPLETED"
    echo "No files were copied."
    echo "Run the builder again with DRY_RUN=0 to create the package."
    echo "======================================================================"
    exit 0
fi

###############################################################################
# 3. Add readable aliases for the main paper experiments.
###############################################################################

echo
echo "[3/8] Creating readable experiment aliases..."

ALIASES_TSV="${CORE}/manifests/experiment_aliases.tsv"

printf "alias\traw_experiment_directory\n" > "${ALIASES_TSV}"

make_alias() {
    alias_name="$1"
    raw_name="$2"

    raw_path="${CORE}/experiments_raw/${raw_name}"
    alias_path="${CORE}/experiments_by_alias/${alias_name}"

    printf "%s\t%s\n" "${alias_name}" "${raw_name}" >> "${ALIASES_TSV}"

    if [[ -d "${raw_path}" ]]; then
        ln -sfn "../experiments_raw/${raw_name}" "${alias_path}"
        echo "OK: ${alias_name}"
    else
        echo "WARNING: alias source not found: ${raw_name}"
    fi
}

make_alias \
    "00_released_beetle_baseline" \
    "nnUNetTrainer_WSD_wei_i0_nnunet_aug_json__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"

make_alias \
    "01_focal_loss" \
    "nnUNetTrainerPathologyFocalClassMetrics__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"

make_alias \
    "02_weighted_focal_loss" \
    "nnUNetTrainerPathologyFocalClassMetricsAlpha__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"

make_alias \
    "03_confusion_aware_sampling" \
    "nnUNetTrainerPathologyWFCMAWS250__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"

make_alias \
    "04_targeted_hard_example_mining" \
    "nnUNetTrainerPathologyWFCHardMining250__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"

make_alias \
    "05_weighted_focal_1000_epochs" \
    "nnUNetTrainerPathologyFocalClassMetricsAlpha1000Milestones__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"

make_alias \
    "06_cutmix_stain_jitter_ema_1000_epochs" \
    "nnUNetTrainer_CutMixStainEMA__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"

make_alias \
    "07_weighted_focal_context1024_ft100" \
    "nnUNetTrainerPathologyFocalClassMetricsAlphaContext1024FT100__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d_context1024"

make_alias \
    "08_cutmix_stain_jitter_ema_context1024_ft100" \
    "nnUNetTrainer_CutMixStainEMA_Context1024FT100__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d_context1024"

###############################################################################
# 4. Record checkpoint locations without copying checkpoint files.
###############################################################################

echo
echo "[4/8] Recording excluded checkpoint locations..."

{
    printf "checkpoint_path\tsize_bytes\n"

    find "${RESULTS_ROOT}" \
        -type f \
        \( \
            -name '*.pth' \
            -o -name '*.pt' \
            -o -name '*.ckpt' \
            -o -name '*.safetensors' \
        \) \
        -printf '%p\t%s\n' \
        | sort
} > "${CORE}/manifests/checkpoint_locations_not_copied.tsv"

###############################################################################
# 5. Write report-ready metric files.
###############################################################################

echo
echo "[5/8] Writing report-ready metric files..."

cat > "${CORE}/report_tables/external_metrics_comparison.csv" <<'CSV'
model,center,overall_dice,other,non_invasive_epithelium,invasive_epithelium,necrosis
released_BEETLE_baseline,Overall,0.8660,0.9370,0.6523,0.7755,0.5110
released_BEETLE_baseline,SCDC,0.8487,0.9247,0.5553,0.7559,0.0849
released_BEETLE_baseline,Biopticka,0.8791,0.9420,0.6973,0.8161,0.6896
released_BEETLE_baseline,UW_Medicine,0.8708,0.9446,0.6734,0.7340,0.4420
cutmix_stain_ema,Overall,0.8441,0.9298,0.6000,0.7249,0.6141
cutmix_stain_ema,SCDC,0.8229,0.9137,0.5041,0.6856,0.2066
cutmix_stain_ema,Biopticka,0.8613,0.9298,0.6480,0.8032,0.7624
cutmix_stain_ema,UW_Medicine,0.8462,0.9466,0.6187,0.6291,0.4053
context1024_ft100,Overall,0.8541,0.9355,0.5981,0.7343,0.6477
context1024_ft100,SCDC,0.8502,0.9283,0.5549,0.7231,0.0914
context1024_ft100,Biopticka,0.8707,0.9350,0.6584,0.8118,0.7750
context1024_ft100,UW_Medicine,0.8385,0.9437,0.5702,0.5887,0.4969
CSV

cat > "${CORE}/report_tables/external_subtype_comparison.csv" <<'CSV'
model,subtype,invasive_epithelium_dice
released_BEETLE_baseline,ILC,0.7791
released_BEETLE_baseline,NST,0.7742
cutmix_stain_ema,ILC,0.8053
cutmix_stain_ema,NST,0.6961
context1024_ft100,ILC,0.8040
context1024_ft100,NST,0.7097
CSV

cat > "${CORE}/report_tables/fold0_ablation_metrics.csv" <<'CSV'
model,epochs,tta,other,non_invasive_epithelium,invasive_epithelium,necrosis,macro_dice,overall_dice
BEETLE_paper_fold0,934,No,0.976,0.787,0.755,0.743,0.815,0.911
dice_plus_focal,250,No,0.972,0.731,0.684,0.708,0.774,0.890
class_weighted_focal,250,No,0.970,0.728,0.737,0.709,0.786,0.895
weighted_focal_plus_confusion_aware_sampling,250,No,0.973,0.716,0.678,0.740,0.777,0.886
weighted_focal_plus_targeted_hard_example_mining,250,Yes,0.972,0.671,0.694,0.754,0.773,0.882
weighted_focal_1000_epochs,1000,No,0.977,0.759,0.707,0.763,0.802,0.902
weighted_focal_plus_cutmix_stain_jitter_ema,1000,Yes,0.979,0.770,0.743,0.771,0.816,0.909
weighted_focal_plus_context1024_finetuning,1000_plus_100_ft,Yes,0.971,0.796,0.769,0.659,0.799,0.908
weighted_focal_plus_cutmix_stain_jitter_ema_plus_context1024_finetuning,1000_plus_100_ft,Yes,0.977,0.780,0.713,0.695,0.791,0.903
CSV

cat > "${CORE}/report_tables/known_internal_cv_metrics.csv" <<'CSV'
experiment,fold,other,non_invasive_epithelium,invasive_epithelium,necrosis,macro_dice,overall_dice,status
cutmix_stain_ema,0,0.979,0.770,0.743,0.771,0.816,0.909,completed
cutmix_stain_ema,1,0.969,0.736,0.737,0.665,0.776,0.894,completed
cutmix_stain_ema,2,0.979,0.850,0.789,0.778,0.849,0.932,completed
cutmix_stain_ema,3,0.963,0.901,0.814,0.582,0.815,0.924,completed
context1024_ft100,3,0.9692,0.9198,0.8583,0.6215,0.8422,0.9400,completed
CSV

###############################################################################
# 6. Write package README files.
###############################################################################

echo
echo "[6/8] Writing package documentation..."

cat > "${PACKAGE_ROOT}/README.md" <<'TXT'
# BEETLE final evaluation package

This package contains the complete compact evaluation evidence for the AIMI
BEETLE paper.

## Inclusion policy

Included:
- all non-heavy files from every nnU-Net experiment directory;
- all JSON, CSV, TSV, TXT, log, script, report, and configuration files;
- all generated qualitative visual-analysis panels;
- all external-inference outputs and challenge submission ZIP files;
- readable aliases for the central paper experiments;
- manifests listing every copied and skipped file;
- report-ready summary CSV files.

Excluded:
- large model checkpoints and weight files;
- other large non-visual artifacts above the configured size threshold.

All excluded paths are recorded in:
`paper_core/manifests/skipped_files.tsv`

Checkpoint locations are recorded separately in:
`paper_core/manifests/checkpoint_locations_not_copied.tsv`

## Folder guide

- `paper_core/experiments_raw/`
  Complete compact result files, preserving the original trainer and fold layout.

- `paper_core/experiments_by_alias/`
  Readable symbolic links to the central paper experiments.

- `paper_core/report_tables/`
  Report-ready CSV tables.

- `paper_core/inference_outputs/`
  External predictions and challenge-ready ZIP archives.

- `paper_core/repo_snapshot/`
  Source code, report sources, Python scripts, and SLURM wrappers.

- `paper_core/logs/`
  Copied repository logs.

- `paper_core/evaluation_inputs/`
  Fold-specific WSI evaluation CSV files.

- `visual_archive/validation_visuals/`
  Complete qualitative visual-analysis archive.

- `visual_archive/selected_for_paper/`
  Folder for the final selected qualitative examples.

## Main external result

Released BEETLE baseline:
overall Dice = 0.8660

CutMix + stain jitter + EMA:
overall Dice = 0.8441

CutMix + stain jitter + EMA + context-1024 FT100:
overall Dice = 0.8541

The context-1024 fine-tuned ensemble is the strongest local model on the
external evaluation set, but it remains below the released BEETLE baseline
overall. It improves necrosis substantially while epithelial classes remain
the main limitation under center and scanner domain shift.
TXT

cat > "${VISUAL_ARCHIVE}/selected_for_paper/README.md" <<'TXT'
# Selected qualitative cases

Copy the final selected figure panels into this folder.

Recommended selection:
1. One representative case showing improved invasive versus non-invasive
   epithelial segmentation after context-1024 fine-tuning.
2. One representative remaining limitation, preferably an epithelial error
   on an externally difficult morphology or a necrosis-related trade-off.
3. Optionally, one supplementary ILC-success or NST-failure example.
TXT

###############################################################################
# 7. Record environment information and inventories.
###############################################################################

echo
echo "[7/8] Recording environment information and file inventories..."

{
    echo "Created: $(date)"
    echo "Host: $(hostname)"
    echo "User: ${USER}"
    echo "MAX_NONVISUAL_MB: ${MAX_NONVISUAL_MB}"
    echo
    echo "Python:"
    command -v python || true
    python --version || true
    echo
    echo "PyTorch:"
    python - <<'PY' || true
try:
    import torch
    print("torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
except Exception as error:
    print("Unable to inspect PyTorch:", error)
PY
} > "${CORE}/manifests/environment_summary.txt"

if git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "${REPO_ROOT}" rev-parse HEAD \
        > "${CORE}/manifests/git_commit.txt"

    git -C "${REPO_ROOT}" status --short \
        > "${CORE}/manifests/git_status.txt"
fi

python -m pip freeze \
    > "${CORE}/manifests/pip_freeze.txt" \
    || true

{
    printf "path\tsize_bytes\n"

    find "${PACKAGE_ROOT}" \
        -type f \
        -printf '%P\t%s\n' \
        | sort
} > "${CORE}/manifests/package_inventory.tsv"

{
    printf "path\tsize_bytes\n"

    find "${VISUAL_ARCHIVE}/validation_visuals" \
        -type f \
        -printf '%P\t%s\n' \
        | sort
} > "${CORE}/manifests/validation_visual_inventory.tsv"

###############################################################################
# 8. Final summary.
###############################################################################

echo
echo "[8/8] Final package summary..."

echo
echo "======================================================================"
echo "PACKAGE COMPLETED SUCCESSFULLY"
echo "======================================================================"
echo "Finished:     $(date)"
echo "Package root: ${PACKAGE_ROOT}"
echo
echo "Total package size:"
du -sh "${PACKAGE_ROOT}"
echo
echo "Paper-core size:"
du -sh "${CORE}"
echo
echo "Visual-archive size:"
du -sh "${VISUAL_ARCHIVE}"
echo
echo "Experiment directories copied:"
find "${CORE}/experiments_raw" \
    -mindepth 1 \
    -maxdepth 1 \
    -type d \
    | wc -l
echo
echo "Visual files copied:"
find "${VISUAL_ARCHIVE}/validation_visuals" \
    -type f \
    | wc -l
echo
echo "Submission ZIP files:"
find "${CORE}/inference_outputs" \
    -type f \
    -name '*.zip' \
    -printf '%P\n' \
    | sort
echo
echo "Skipped-file manifest:"
echo "${CORE}/manifests/skipped_files.tsv"
echo "======================================================================"
