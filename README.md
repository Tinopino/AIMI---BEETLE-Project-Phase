# AIMI BEETLE Project Phase

Reproducible training, evaluation, and external-inference pipeline for multiclass
breast-cancer histopathology segmentation on the BEETLE benchmark.

## Task

Each H&E image is segmented into four classes:

| Label | Class |
|---:|---|
| 1 | Other tissue |
| 2 | Non-invasive epithelium |
| 3 | Invasive epithelium |
| 4 | Necrosis |

The project focuses on the dominant epithelial error pattern:
**invasive epithelium ↔ non-invasive epithelium confusion**.

## Main experiment sequence

1. Released nnU-Net-for-pathology baseline.
2. Dice + focal loss.
3. Class-weighted focal loss.
4. Confusion-aware sampling.
5. Targeted hard-example mining.
6. Weighted focal loss + CutMix + stain jitter + EMA.
7. Context-1024 fine-tuning.

The final local model is the five-fold CutMix + stain-jitter + EMA ensemble
fine-tuned with 1024 × 1024 model patches.

## Main external results

| Model | Overall Dice | Other | Non-invasive | Invasive | Necrosis |
|---|---:|---:|---:|---:|---:|
| Released BEETLE baseline | 0.8660 | 0.9370 | 0.6523 | 0.7755 | 0.5110 |
| CutMix + stain jitter + EMA | 0.8441 | 0.9298 | 0.6000 | 0.7249 | 0.6141 |
| CutMix + stain jitter + EMA + context-1024 FT100 | 0.8541 | 0.9355 | 0.5981 | 0.7343 | 0.6477 |

The context-1024 fine-tuned ensemble is the strongest local external model.
It improves necrosis substantially and partially recovers the performance gap,
but epithelial segmentation remains less robust than the released baseline
under multicentric domain shift.

## Repository layout

- `nnUNet_pathology/`  
  Pathology-specific nnU-Net implementation and custom trainer variants.

- `preprocessing/`  
  Dataset inspection and patient-level split utilities.

- `reproducibility/`  
  Compact metrics, configuration snapshots, split information, and checkpoint
  location manifests. Large weight files are intentionally excluded.

- `outputs/splits/`  
  Generated split summaries.

- `beetle_wf_hard_mining_package/`  
  Hard-example-mining utilities.

- Root-level `*.slurm` and `*.sh` files  
  Training, evaluation, aggregation, inference, and queue-orchestration
  wrappers used during the project.

## Installation

Create and activate an environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ./nnUNet_pathology
```

Set the nnU-Net paths:

```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

## Inference geometry

Standard models:

```text
model patch:  512 × 512
sampler tile: 2048 × 2048
output tile:  1536 × 1536
```

Context models:

```text
model patch:  1024 × 1024
sampler tile: 2048 × 2048
output tile:  1024 × 1024
```

The output crop removes lower-overlap borders before writing WSI predictions.
Inference-time mirroring should remain enabled for final comparisons unless a
speed ablation explicitly disables it.

## Reproduction workflow

```text
prepare dataset and patient-level splits
→ train fold-specific models
→ run held-out WSI evaluation per fold
→ aggregate per-class Dice and confusion matrices
→ run five-fold probability-averaged external ROI inference
→ validate the 170 single-channel PNG predictions
→ create the submission ZIP
```

Important wrappers include:

```text
run_external_ensemble_inference.slurm
run_external_ensemble_inference_ctx1024ft100.slurm
queue_only_five_cutmixema_folds.sh
queue_context1024_finetuning_only.sh
queue_context1024_evaluations.sh
aggregate_cv_results.py
aggregate_cv_results_v2.py
```

## Files intentionally excluded from Git

The following are not versioned:

- raw WSIs and annotation masks;
- nnU-Net preprocessed data;
- model checkpoints (`*.pth`, `*.pt`, `*.ckpt`, `*.safetensors`);
- generated external ROI predictions;
- challenge ZIP submissions;
- large visual archives;
- cluster logs and temporary queue records.

Checkpoint paths are recorded in:

```text
reproducibility/checkpoints/checkpoint_locations_not_versioned.tsv
```

## Methodological conclusion

Ordinary held-out-fold validation did not fully predict external multicentric
generalization. Future model-selection workflows should incorporate an
internal domain-shift proxy, such as source- or scanner-stratified validation,
before committing compute to full five-fold training.

## References

- Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H.
  (2021). *nnU-Net: a self-configuring method for deep learning-based
  biomedical image segmentation*. Nature Methods, 18(2), 203–211.
- BEETLE breast-cancer histopathology segmentation benchmark.
