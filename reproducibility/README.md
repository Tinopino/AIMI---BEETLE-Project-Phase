# AIMI BEETLE experiment reproducibility snapshot

This branch preserves the code, launch scripts, compact configuration files,
hard-mining manifests, environment information, and small reference outputs
needed to reproduce and audit the completed AIMI BEETLE experiments.

## Experiment sequence

1. Released nnU-Net-for-pathology checkpoint evaluation.
2. Dice + focal loss, trained for 250 epochs.
3. Class-weighted focal loss, trained for 250 epochs.
4. Weighted focal loss with confusion-aware sampling, trained for 250 epochs.
5. Weighted focal loss with targeted hard-example mining, trained for 250 epochs.
6. Weighted focal loss extended to 1000 epochs.
7. Weighted focal loss with CutMix, stain jitter, and EMA, trained for 1000 epochs.
8. Exploratory context-1024 fine-tuning for 100 additional epochs.
9. Five-fold WSI validation for the selected CutMix + stain-jitter + EMA model.
10. Five-model external ensemble inference.

## Updating compact outputs after folds 3 and 4 finish

Run:

    ./reproducibility/update_reference_results.sh
    git add reproducibility/reference_results reproducibility/reference_logs
    git commit -m "Add completed fold 3 and fold 4 validation outputs"
    git push

## Data and checkpoint policy

Raw WSIs, annotations, preprocessed tensors, and model checkpoints are not
versioned in Git. They remain external course assets or generated artefacts.
