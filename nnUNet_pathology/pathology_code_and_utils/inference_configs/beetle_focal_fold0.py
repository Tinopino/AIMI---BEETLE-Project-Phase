from pathlib import Path

def norm_01(x_batch):
    x_batch = x_batch / 255.0
    x_batch = x_batch.transpose(3, 0, 1, 2)
    return x_batch

model_base_path = (
    "/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results/"
    "Dataset301_BEETLE/"
    "nnUNetTrainer_WSD_wei_i0_nnunet_aug_json_DiceFocal__"
    "nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"
)

folds_to_use = (0,)
checkpoint_name = "checkpoint_best.pth"

norm = norm_01
output_minus_1 = False

output_folder = Path(
    "/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/"
    "validation_inference/DiceFocal_fold0"
)

matches_to_run = []

rerun_unfinished = False
overwrite = True

spacing = 0.5
model_patch_size = 512
sampler_patch_size = 2048
cpus = 4

use_wandb = False
