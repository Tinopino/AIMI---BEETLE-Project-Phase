# AIMI BEETLE Project

Minimal reproduction repository for multiclass breast-cancer histopathology
segmentation on the BEETLE benchmark.

## Task

The model predicts four classes:

| Label | Class |
|---:|---|
| 1 | Other tissue |
| 2 | Non-invasive epithelium |
| 3 | Invasive epithelium |
| 4 | Necrosis |

The experiments focus on the dominant error pattern:
**invasive epithelium ↔ non-invasive epithelium confusion**.

## Final external results

| Model | Overall Dice |
|---|---:|
| Released BEETLE baseline | 0.8660 |
| CutMix + stain jitter + EMA | 0.8441 |
| CutMix + stain jitter + EMA + context-1024 FT100 | 0.8541 |

The context-1024 model is the strongest local model, but the released baseline
remains strongest overall on the multicentric external benchmark.

## Repository structure

```text
.
├── configure_data_paths.py
├── configure_validation_inputs.py
├── experiments.py
├── train.py
├── validate.py
├── paths.env.example
├── paths.local.ps1.example
├── splits.json
├── splits_full587.json
├── splits_project396.json
├── REPRODUCING_FINAL_MODEL.md
├── pipeline/
├── nnUNet_pathology/
└── outputs/
```

The public interface consists of five files:

| File | Purpose |
|---|---|
| `configure_data_paths.py` | Rewrite the committed pathology split paths for another filesystem while preserving fold membership. |
| `configure_validation_inputs.py` | Generate held-out WSI validation CSV manifests from `splits.json` and raster masks. |
| `experiments.py` | Experiment phases, trainer classes, checkpoint tags, and native inference geometry. |
| `train.py` | Train or resume one registered experiment fold. |
| `validate.py` | Held-out WSI validation, fold aggregation, external five-fold inference, submission checks, and ZIP creation. |

`pipeline/` contains the implementation modules called by the public entry
points. `nnUNet_pathology/` is the modified pathology-specific nnU-Net fork.

## Scope of reproduction

Raw WSIs, annotations, preprocessed tensors, and checkpoints are not included
because they are external or large generated artifacts. This repository
reproduces the project from a **prepared BEETLE nnU-Net dataset onward**:

```text
configure paths
→ preserve the original five-fold WSI split
→ generate held-out validation manifests
→ train or resume registered experiment folds
→ run context-1024 fine-tuning
→ run held-out WSI validation
→ aggregate fold metrics
→ run external five-fold ensemble inference
→ validate 170 PNG masks and create a submission ZIP
```

The default WSI-level fold assignment used for the actual final model training
is the complete 587-slide BEETLE development split and is stored in:

```text
splits.json
```

For clarity, the split profiles are also stored explicitly as:

| File | Purpose |
|---|---|
| `splits.json` | Default actual final-model split; identical to `splits_full587.json`. |
| `splits_full587.json` | Complete 587-slide BEETLE development split used for the actual final model. |
| `splits_project396.json` | 396-slide project split used for earlier experiments and ablations. |

Use `--reference-splits` with `configure_data_paths.py` when reproducing a
non-default profile. Do not mix full587 final-model runs and project396
experiment/ablation runs in the same `nnUNet_preprocessed` or
`nnUNet_results` directories.

For the shortest end-to-end instructions for reproducing the final model on the
course cluster, see:

```text
REPRODUCING_FINAL_MODEL.md
```

## Required external data

The code expects a prepared BEETLE dataset with the following layout:

```text
<BEETLE_DATA_ROOT>/
├── images/
│   ├── development/
│   │   └── wsis/
│   │       ├── patient1_wsi1.tif
│   │       └── ...
│   └── evaluation/
│       └── rois/
│           ├── patient320_wsi1_roi1.png
│           └── ...
└── annotations/
    ├── jsons/
    │   ├── patient1_wsi1.json
    │   └── ...
    └── masks/
        ├── patient1_wsi1.tif
        └── ...
```

The raster masks under `annotations/masks/` are required only for held-out WSI
evaluation. Training uses the WSI and annotation JSON paths stored in
`splits.json`. External challenge inference uses the 170 ROI PNG files under
`images/evaluation/rois/`.

The pathology nnU-Net pipeline also expects:

```text
<NNUNET_ROOT>/
├── nnUNet_raw/
├── nnUNet_preprocessed/
│   └── Dataset301_BEETLE/
│       ├── dataset.json
│       ├── nnUNetWholeSlideDataPlans.json
│       └── splits.json
└── nnUNet_results/
    └── Dataset301_BEETLE/
        └── <experiment-directory>/
            ├── dataset.json
            ├── plans.json
            ├── fold_0/
            │   └── checkpoint_best.pth
            ├── ...
            └── fold_4/
                └── checkpoint_best.pth
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ./nnUNet_pathology
pip install -r requirements-cluster.txt
```

`requirements-cluster.txt` lists project-level packages that are used by the
BEETLE training/validation scripts but are not part of the upstream nnU-Net
fork metadata.

Configure local paths:

```bash
cp paths.env.example paths.env
nano paths.env
source paths.env
```

The defaults in `paths.env.example` match the original course-cluster layout.
On another machine, adapt `BEETLE_DATA_ROOT`, `nnUNet_raw`,
`nnUNet_preprocessed`, `nnUNet_results`, and `BEETLE_VENV`.

On Windows/PowerShell, use `paths.local.ps1.example` as a template instead of
the Bash-oriented `paths.env.example`.

## Rewrite the committed split paths for another machine

The committed `splits.json` preserves the exact five-fold WSI assignments used
in this project. Its WSI and annotation records contain the original cluster
paths. Rewrite those path prefixes for another filesystem with:

```bash
python configure_data_paths.py \
    --data-root /path/to/beetle-data \
    --nnunet-preprocessed /path/to/nnUNet_preprocessed \
    --reference-splits splits_full587.json
```

This writes:

```text
/path/to/nnUNet_preprocessed/Dataset301_BEETLE/splits.json
```

The helper preserves fold membership, rewrites WSI and annotation paths from
the referenced filenames, checks that the files exist, and refuses to
silently overwrite a different split file. Preview the operation without
writing with:

```bash
python configure_data_paths.py \
    --data-root /path/to/beetle-data \
    --nnunet-preprocessed /path/to/nnUNet_preprocessed \
    --reference-splits splits_full587.json \
    --check-only
```

To reproduce the earlier 396-slide experiments/ablations, replace
`splits_full587.json` with `splits_project396.json` and use separate
`nnUNet_preprocessed` and `nnUNet_results` roots.

Use `--force` only after reviewing an existing destination file. Use
`--skip-file-check` only when preparing paths before the external files are
mounted.

## Generate held-out WSI validation manifests

The fold-specific validation CSV files are generated artifacts and are not
committed to Git. Generate them from the adapted pathology split and the
raster-mask directory:

```bash
python configure_validation_inputs.py
```

With the default environment template this writes:

```text
outputs/validation_inputs/
├── fold0_validation_inference_inputs.csv
├── fold1_validation_inference_inputs.csv
├── fold2_validation_inference_inputs.csv
├── fold3_validation_inference_inputs.csv
└── fold4_validation_inference_inputs.csv
```

Each generated CSV contains:

```text
file_key,slide,wsi_path,mask_path
```

Preview generation without writing with:

```bash
python configure_validation_inputs.py --check-only
```

To use another mask directory or output location:

```bash
python configure_validation_inputs.py \
    --mask-root /path/to/beetle-data/annotations/masks \
    --output-dir /path/to/generated-validation-inputs
```

`validate.py wsi` automatically generates the requested fold manifest when it
is missing, using the configured split file and mask root.

## List experiment phases

```bash
python experiments.py --show
```

## Train or resume one fold

```bash
python train.py \
    --experiment cutmix_stain_ema \
    --fold 0
```

Fine-tune the corresponding fold with larger context:

```bash
python train.py \
    --experiment context1024_ft100 \
    --fold 0
```

## Run held-out WSI validation

```bash
python validate.py wsi \
    --experiment context1024_ft100 \
    --fold 0 \
    --save-visuals
```

Inference-time mirroring is enabled for final comparisons.

## Aggregate completed folds

```bash
python validate.py aggregate \
    --stage both
```

## Run external five-fold ensemble inference

```bash
python validate.py external \
    --experiment context1024_ft100
```

This validates all 170 ROI predictions and creates a challenge-ready ZIP under
`outputs/submissions/`.

## Checkpoints

Trained checkpoints are intentionally not committed to Git. To run external
ensemble inference without retraining, place `checkpoint_best.pth` under the
experiment-specific result directories expected by `experiments.py`:

```text
${nnUNet_results}/Dataset301_BEETLE/<experiment-directory>/fold_0/checkpoint_best.pth
...
${nnUNet_results}/Dataset301_BEETLE/<experiment-directory>/fold_4/checkpoint_best.pth
```

## Native WSI inference geometry

| Model | Model patch | Sampler tile | Written output tile |
|---|---:|---:|---:|
| Standard models | 512 | 2048 | 1536 |
| Context models | 1024 | 2048 | 1024 |

The written output tile equals `sampler tile − model patch size`, discarding
lower-overlap border predictions before stitching.

## Reproduction audit

The cleaned public repository was tested from a fresh clone on the course
cluster with an existing prepared BEETLE dataset and trained checkpoints. The
audit verified trainer discovery, split consistency, dry-run dispatch for all
registered experiments, checkpoint discovery, metric aggregation, and one
real context-1024 GPU ROI prediction.

## Files intentionally excluded from Git

Generated checkpoints, predictions, ZIP submissions, logs, visual archives,
validation CSV manifests, raw WSIs, annotations, and preprocessed tensors are
intentionally excluded.
