#!/usr/bin/env python3

from pathlib import Path

from nnunetv2.run.run_training_pathology import get_trainer_from_args


checkpoint = Path(
    "/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/"
    "nnUNet_results/Dataset301_BEETLE/"
    "nnUNetTrainer_CutMixStainEMA__"
    "nnUNetWholeSlideDataPlans__"
    "wsd_None_iterator_nnunet_aug__2d/"
    "fold_0/checkpoint_latest.pth"
)

if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)

trainer = get_trainer_from_args(
    "301",
    "2d",
    0,
    "nnUNetTrainer_CutMixStainEMA",
    "nnUNetWholeSlideDataPlans",
)

print("Loading fold-0 checkpoint:", checkpoint, flush=True)

trainer.load_checkpoint(str(checkpoint))

print("Resume smoke test passed", flush=True)
print("Loaded current_epoch:", trainer.current_epoch, flush=True)
print("Loaded _best_ema:", trainer._best_ema, flush=True)
print("Loaded _global_step:", trainer._global_step, flush=True)
print("EMA initialized:", trainer._ema_initialized, flush=True)
