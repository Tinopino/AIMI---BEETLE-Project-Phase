# AIMI BEETLE Project Phase

Clean reproduction interface for the AIMI BEETLE breast-cancer
histopathology-segmentation project.

## Task

Segment H&E pathology images into:

| Label | Class |
|---:|---|
| 1 | Other tissue |
| 2 | Non-invasive epithelium |
| 3 | Invasive epithelium |
| 4 | Necrosis |

The project focused on the dominant clinically relevant error pattern:
**invasive epithelium ↔ non-invasive epithelium confusion**.

## Final project result

| Model | External overall Dice |
|---|---:|
| Released BEETLE baseline | 0.8660 |
| CutMix + stain jitter + EMA | 0.8441 |
| CutMix + stain jitter + EMA + context-1024 FT100 | 0.8541 |

The final context model improves necrosis substantially and partially recovers
the external-performance gap, but epithelial generalization remains weaker than
the released baseline under multicentric domain shift.

## Clean interface

The user-facing workflow is intentionally reduced to three Python files:

| File | Responsibility |
|---|---|
| `experiments.py` | Single source of truth for experiment variants, trainers, paths, checkpoint tags, and native inference geometry. |
| `train.py` | Train or resume any registered experiment. It also dispatches context-1024 fine-tuning. |
| `validate.py` | Run held-out WSI evaluation, aggregate folds, run five-fold external ROI inference, validate PNGs, and create the submission ZIP. |

Internal implementation modules live in `internal/`. The pathology-specific
nnU-Net source remains in `nnUNet_pathology/`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ./nnUNet_pathology
```

Create the cluster configuration:

```bash
cp reproducibility/configs/cluster_paths.env.example \
   reproducibility/configs/cluster_paths.env

nano reproducibility/configs/cluster_paths.env
source reproducibility/configs/cluster_paths.env
```

## List registered experiments

```bash
python experiments.py --show
```

## Train or resume one fold

```bash
python train.py \
    --experiment cutmix_stain_ema \
    --fold 0
```

Standard nnU-Net pathology training automatically resumes when
`checkpoint_latest.pth` exists.

## Fine-tune one fold with larger spatial context

```bash
python train.py \
    --experiment context1024_ft100 \
    --fold 0
```

This imports the completed CutMix-stain-EMA fold's best weights and performs
100 additional epochs using 1024 × 1024 model patches.

## Run held-out WSI evaluation

```bash
python validate.py wsi \
    --experiment context1024_ft100 \
    --fold 0 \
    --save-visuals
```

Final comparable evaluation uses inference-time mirroring by default.

## Aggregate five completed folds

```bash
python validate.py aggregate \
    --stage both
```

## Run external five-fold inference and create the challenge ZIP

```bash
python validate.py external \
    --experiment context1024_ft100
```

The command validates:

- exactly 170 ROI predictions;
- identical input and output filenames;
- identical image dimensions;
- single-channel PNG format;
- permitted class labels `{1, 2, 3, 4}`;
- ZIP files with PNGs stored directly at the archive root.

## Submit through SLURM

```bash
sbatch slurm/beetle_gpu_job.slurm \
    train.py \
    --experiment cutmix_stain_ema \
    --fold 0

sbatch slurm/beetle_gpu_job.slurm \
    validate.py wsi \
    --experiment context1024_ft100 \
    --fold 0 \
    --save-visuals

sbatch slurm/beetle_gpu_job.slurm \
    validate.py external \
    --experiment context1024_ft100
```

## Native WSI inference geometry

| Model | Model patch | Sampler tile | Written output tile |
|---|---:|---:|---:|
| Standard models | 512 | 2048 | 1536 |
| Context models | 1024 | 2048 | 1024 |

The written output tile equals `sampler tile − model patch size`. This discards
lower-overlap border predictions before stitching.

## Repository layout

```text
.
├── experiments.py
├── train.py
├── validate.py
├── slurm/
│   └── beetle_gpu_job.slurm
├── internal/
│   ├── aggregate_cv_results.py
│   ├── context_finetune.py
│   ├── wsi_validation_engine.py
│   └── tools/
├── nnUNet_pathology/
├── preprocessing/
├── reproducibility/
├── outputs/
└── archive/
    ├── README.md
    └── historical_hard_mining_package/
```

## Historical scripts

The earlier root directory contained many one-off setup, repair, queue, and
experiment-specific SLURM wrappers. They are removed from the clean tree but
remain available through Git history at commit:

```text
fbcb9477914e3d0db0b424cbf8f8a87f4bec2f49
```

## Files intentionally excluded from Git

Raw WSIs, annotation masks, nnU-Net preprocessed data, model checkpoints,
generated external predictions, submission ZIP files, logs, and the full visual
archive remain outside Git.

## Methodological lesson

Ordinary held-out-fold validation did not fully predict external multicentric
generalization. Future model-selection workflows should incorporate a
source- or scanner-stratified validation proxy before committing compute to
full five-fold training.
