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
├── experiments.py
├── train.py
├── validate.py
├── paths.env.example
├── splits_final.json
├── pipeline/
├── nnUNet_pathology/
└── outputs/
```

The public interface consists of three files:

| File | Purpose |
|---|---|
| `experiments.py` | Experiment phases, trainer classes, checkpoint tags, and native inference geometry. |
| `train.py` | Train or resume one registered experiment fold. |
| `validate.py` | Held-out WSI validation, fold aggregation, external five-fold inference, submission checks, and ZIP creation. |

`pipeline/` contains the implementation modules called by the public entry
points. `nnUNet_pathology/` is the modified pathology-specific nnU-Net fork.

## Scope of reproduction

Raw WSIs, annotations, preprocessed tensors, and checkpoints are not included
because they are external or large generated artifacts. This repository
reproduces the project from a prepared BEETLE nnU-Net dataset onward.

The exact patient-level split used in the project is stored in:

```text
splits_final.json
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ./nnUNet_pathology
```

Configure local paths:

```bash
cp paths.env.example paths.env
nano paths.env
source paths.env
```

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

## Native WSI inference geometry

| Model | Model patch | Sampler tile | Written output tile |
|---|---:|---:|---:|
| Standard models | 512 | 2048 | 1536 |
| Context models | 1024 | 2048 | 1024 |

The written output tile equals `sampler tile − model patch size`, discarding
lower-overlap border predictions before stitching.

## Files intentionally excluded from Git

Generated checkpoints, predictions, ZIP submissions, logs, visual archives,
raw WSIs, annotations, and preprocessed tensors are intentionally excluded.
