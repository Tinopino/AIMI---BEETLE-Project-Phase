# Reproducing the Final BEETLE Model

This guide describes the intended reproduction path for the final submitted
model on the course cluster.

The actual final model training configuration uses the complete 587-slide
BEETLE development split:

```text
splits_full587.json
```

The repository default `splits.json` is identical to `splits_full587.json`.
The 396-slide project split used for earlier experiments/ablations is retained
separately as `splits_project396.json`. Do not mix project396 and full587
outputs in the same `nnUNet_preprocessed` or `nnUNet_results` roots.

## 1. Clone And Install

```bash
git clone https://github.com/Tinopino/AIMI---BEETLE-Project-Phase.git
cd AIMI---BEETLE-Project-Phase

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ./nnUNet_pathology
python -m pip install -r requirements-cluster.txt
```

If installing `wholeslidedata` fails on a local Windows machine, reproduce the
final training on the Linux course cluster environment instead. The pathology
training dataloader depends on `wholeslidedata`.

Equivalent explicit package list:

```bash
python -m pip install opencv-python tiffslide openslide-python wandb wholeslidedata
```

## 2. Configure Cluster Paths

Copy the environment template and adapt paths if needed:

```bash
cp paths.env.example paths.env
nano paths.env
source paths.env
```

On the original course cluster setup, the important paths were:

```bash
export AIMI_GROUP_ROOT="/vol/csedu-nobackup/course/IMC037_aimi/group14"
export BEETLE_DATA_ROOT="${AIMI_GROUP_ROOT}/aalina"
export nnUNet_raw="${AIMI_GROUP_ROOT}/nnunet/tijn/pathology/nnUNet_raw"
export nnUNet_preprocessed="${AIMI_GROUP_ROOT}/nnunet/tijn/pathology/nnUNet_preprocessed"
export nnUNet_results="${AIMI_GROUP_ROOT}/nnunet/tijn/pathology/nnUNet_results"
```

The BEETLE data root must contain:

```text
images/development/wsis/
images/evaluation/rois/
annotations/jsons/
annotations/masks/
```

All files referenced by `splits_full587.json` are required to reproduce the
actual final model. TIGER WSIs are an external BEETLE dependency and must be
present for this full587 run.

## 3. Verify Prepared nnU-Net Dataset Files

This repository reproduces from a prepared pathology nnU-Net dataset onward.
Before training, verify that the prepared Dataset301 folder exists:

```bash
ls "${nnUNet_preprocessed}/Dataset301_BEETLE/dataset.json"
ls "${nnUNet_preprocessed}/Dataset301_BEETLE/nnUNetWholeSlideDataPlans.json"
```

If these files are missing, the prepared Dataset301_BEETLE artifacts must be
created or restored before training. Do not start training without them.

## 4. Write The Full587 Split Into nnUNet_preprocessed

Preview first:

```bash
python configure_data_paths.py \
  --data-root "${BEETLE_DATA_ROOT}" \
  --nnunet-preprocessed "${nnUNet_preprocessed}" \
  --reference-splits splits_full587.json \
  --check-only
```

If the check passes, write the split:

```bash
python configure_data_paths.py \
  --data-root "${BEETLE_DATA_ROOT}" \
  --nnunet-preprocessed "${nnUNet_preprocessed}" \
  --reference-splits splits_full587.json
```

This writes:

```text
${nnUNet_preprocessed}/Dataset301_BEETLE/splits.json
```

## 5. Generate Held-Out Validation Manifests

Preview first:

```bash
python configure_validation_inputs.py --check-only
```

Then generate:

```bash
python configure_validation_inputs.py
```

## 6. Train The Final Base Model

The final base model is:

```text
cutmix_stain_ema
```

Run all five folds. A dry run can be used to confirm dispatch first:

```bash
for fold in 0 1 2 3 4; do
  python train.py --experiment cutmix_stain_ema --fold "${fold}" --dry-run
done
```

Then launch actual training using the cluster's normal GPU job mechanism:

```bash
for fold in 0 1 2 3 4; do
  python train.py --experiment cutmix_stain_ema --fold "${fold}"
done
```

On SLURM, wrap each command in the course-provided `srun`/`sbatch` GPU job
template rather than running all folds directly on a login node.

## 7. Fine-Tune The Context-1024 Model

The final context model is:

```text
context1024_ft100
```

This stage depends on the corresponding completed `cutmix_stain_ema` fold
checkpoint:

```text
${nnUNet_results}/Dataset301_BEETLE/nnUNetTrainer_CutMixStainEMA__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d/fold_<N>/checkpoint_best.pth
```

Dry run first:

```bash
for fold in 0 1 2 3 4; do
  python train.py --experiment context1024_ft100 --fold "${fold}" --dry-run
done
```

Then launch the fine-tuning jobs:

```bash
for fold in 0 1 2 3 4; do
  python train.py --experiment context1024_ft100 --fold "${fold}"
done
```

Again, use the cluster GPU job scheduler rather than running long jobs on a
login node.

## 8. Validate And Run External Inference

Held-out WSI validation for one fold:

```bash
python validate.py wsi --experiment context1024_ft100 --fold 0 --save-visuals
```

Aggregate folds after all validations finish:

```bash
python validate.py aggregate --stage both
```

External five-fold inference and submission ZIP creation:

```bash
python validate.py external --experiment context1024_ft100
```

## Notes

- `splits_full587.json` is the actual final-model training split.
- `splits_project396.json` is the earlier experiment/ablation split; those
  runs reproduce the 396-split experimental metrics, not the final 587 model.
- Use separate `nnUNet_preprocessed` and `nnUNet_results` roots if comparing
  full587 and project396 runs.
- Checkpoints are not committed to Git. They are generated by training or must
  be restored into the documented `nnUNet_results` layout.
