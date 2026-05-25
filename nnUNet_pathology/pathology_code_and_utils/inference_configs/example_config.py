import os
from pathlib import Path
import pandas as pd

#################################################################
### Functions and utilities
#################################################################
def norm_01(x_batch): # Use this for models trained on 0-1 scaled data
    x_batch = x_batch / 255
    x_batch = x_batch.transpose(3, 0, 1, 2)
    return x_batch

def z_norm(x_batch): # use this for default nnunet models, using z-score normalized data
    mean = x_batch.mean(axis=(-2,-1), keepdims=True)
    std = x_batch.std(axis=(-2,-1), keepdims=True)
    x_batch = ((x_batch - mean) / (std + 1e-8))
    x_batch = x_batch.transpose(3, 0, 1, 2)
    return x_batch

def csv_to_matches(path_df):
    df = pd.read_csv(path_df)
    if len(df.columns) > 2:
        raise ValueError("The DataFrame should have only 2 columns: first contains image path and second contains mask path.")
    matches = list(df.itertuples(index=False, name=None))
    return matches

def return_matches_to_run(matches, output_folder):
    """
    The whole pipeline is costructed in such a way that a <name>_runtime.txt file is created once a python script started to work on this WSI + mask match. This means that this file should not be processed anymore by another python script that is run in parralel (also doesnt need to be copied to the local machine (chis is the case on our computing cluster))
    """
    runtime_stems = [file[:-12] for file in os.listdir(output_folder) if file.endswith('_runtime.txt')]
    imgs, _ = zip(*matches)
    img_stems = [Path(file).stem for file in imgs]
    matches_to_run_idx = [i for i in range(len(img_stems)) if img_stems[i] not in runtime_stems]
    matches_to_run = [matches[i] for i in matches_to_run_idx]

    if len(matches_to_run) == 0:
        print(f"\nAll files have been processed already, see {output_folder}")
    else:
        print(f"\nReturning {len(matches_to_run)} matches that are not finished yet")
    return matches_to_run


#################################################################
### SET CONFIG
#################################################################

### TASK AND MODEL
model_base_path = '/data/temporary/joey/data/nnUNet_v2_data_base/nnUNet_results'
model_base_path = os.path.join(model_base_path, 'Dataset024_PDL1_new_data_train_only', 'nnUNetTrainer_WSD_points_bal_i0_nnunet_aug_json__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d')

norm = norm_01 # z_norm or norm_01, norm_01 for models trained on 0-1 scaled data, z_norm for default nnunet models, using z-score normalized data  
output_minus_1 = True # Set to True if you want to subtract 1 from the argmax (for example when label 0 is ignore and label 1 is background, which you want to be 0 for visualisation purposes)

### OUTPUT FOLDER
# output_folder = None # set via in and output paths directly provided to the script
output_folder = Path('/data/temporary/joey/data/nnunet_v2_test')
# These folder will be created if they do not exist 

### MATCHES YOU WANT TO RUN
# This is a list of tuples, where each tuple contains the path to the wsi and the path to the tissue mask
# Example 1: [('path/to/wsi1', 'path/to/tissue_mask1'), ('path/to/wsi2', 'path/to/tissue_mask2')]
# Example 2: all_matches = csv_to_matches('path/to/csv_file.csv')
#            matches_to_run = return_matches_to_run(all_matches, output_folder)
# Example 3: matches_to_run = None # set via in and output paths directly provided to the script
# img_in = convert_path(r"T:\archives\lung\dedication\images\HE\DED-ELK-018_HE_E002.tif")
# mask_in = convert_path(r"T:\archives\lung\dedication\tissue_masks\HE\DED-ELK-018_HE_E002_tissue_mask.tif")
# matches_to_run = None # set via in and output paths directly provided to the script [(img_in, mask_in),]
matches_to_run = [('/data/temporary/archives/lung/nsclc_annotations/pdl1/images/pdl1_verona/PD-L1 AK2_1022280_core2.tif',
                  '/data/temporary/archives/lung/nsclc_annotations/pdl1/roi_masks/pdl1_verona/PD-L1 AK2_1022280_core2_roi_mask.tif')]

rerun_unfinished = True # [READ THIS -->] this will check if the '<stem>_runtime.txt file is present in the outputfolder. If it is not it means that this file is not processed fully. IMPORTANT: if this is set to True multiple parallel jobs cannot track which slides are being processed, set this to false if you want to run multiple jobs processing the same 'matches_to_run' in a parralel. Alternatively, remove all unfinished files in the output folder (the files that do not have a .runtime.txt file). Default is False 
overwrite = True # [READ THIS -->] this will always process and overwrite files, even if they are already processed. May break logic if you have multiple jobs processing the same 'matches_to_run' running in parallel. Default is False

### SAMPLING STUFF
spacing = 0.5 # spacing for batch_shape spacing and annotation_parser output_spacing (leave its processing spacing on 4 or higher)
model_patch_size = 512 # input size of model (should be square)
sampler_patch_size = 4 * model_patch_size # use this as batch shape (= 8 * model_patch_size)
cpus = 1 # number of cpus to use for sampling (note that buffer states are printed during inference time now, which may give you information about the number of cpus you should use, I noticed 1 is already enough to have a saturated buffer state when you sample: 4 * model_patch_size)

### WANDB
use_wandb = False # If True put "export WANDB_API_KEY=<your key>"" in your .bashrc or .bash_profile

#################################################################
### USAGE
#################################################################

"""
1) Provide only the stem of the config file
    python3 -u /data/temporary/joey/github/nnUNet-for-pathology_v1/nnunet_for_pathology_example_usage/nnUNet_run_WSI_inference_REWORK_using_config.py Task032_config
OR
2) Provide the stem of the config file and the input and output paths
    python3 -u /data/temporary/joey/github/nnUNet-for-pathology_v1/nnunet_for_pathology_example_usage/nnUNet_run_WSI_inference_REWORK_using_config.py Task032_config /data/temporary/archives/lung/dedication/images/HE/DED-ELK-018_HE_E002.tif /data/temporary/archives/lung/dedication/tissue_masks/HE/DED-ELK-018_HE_E002_tissue_mask.tif /data/temporary/joey/data/nnunet_v1_test/DED-ELK-018_HE_E002_nnunet.tif /data/temporary/joey/data/nnunet_v1_test/DED-ELK-018_HE_E002_nnunet.tif/data/temporary/joey/data/nnunet_v1_test/DED-ELK-018_HE_E002_uncertainty.tif
"""










# #################################################################
# ### CHANGE THINGS HERE
# #################################################################

# ### MODEL SETTINGS
# model_base_path = '/data/pathology/projects/pathology-lung-TIL/nnUNet_v2/data/nnUNet_results/Dataset008_PDL1_simplified_and_union_annotations/nnUNetTrainer_WSD_wei_i0_nnunet_aug__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d'
# norm = norm_01 # z_norm or norm_01, norm_01 for models trained on 0-1 scaled data, z_norm for default nnunet models, using z-score normalized data 
# output_minus_1 = True # Set to True if you want to subtract 1 from the argmax (for example when label 0 is ignore and label 1 is background, which you want to be 0 for visualisation purposes)

# ### OUTPUT FOLDER AND DATASET NAME
# dataset_name = 'V2_PDL1_test' # name that gets added to folder names and file names
# output_folder = Path('/data/pathology/projects/pathology-lung-TIL/nnUNet_raw_data_base/inference_results/v2_Task008_PDL1_simplified_and_union_annotations_wei_i0_nnunet_aug_TEST_SET')
# local_output_folder = Path('/home/user/workdir')
# os.makedirs(output_folder, exist_ok=True)
# os.makedirs(local_output_folder, exist_ok=True)

# ### MATCHES YOU WANT TO RUN
# # This is a list of tuples, where each tuple contains the path to the wsi and the path to the tissue mask
# # Example: [('path/to/wsi1', 'path/to/tissue_mask1'), ('path/to/wsi2', 'path/to/tissue_mask2')]
# all_matches = csv_to_matches(convert_path(r"Z:\archives\lung\ignite_retrospective_radboudumc\csv\TEMP_he_mask_paths.csv"))
# matches_to_run = return_matches_to_run(all_matches, output_folder)

# # image_path = convert_path(r"Z:\archives\lung\pembro_rt\images\HE\IG_S02_P000001_C0001_B101_V01_T01_L01_A15_E02.tif")
# # mask_path = convert_path(r"Z:\archives\lung\pembro_rt\tissue_masks\HE\IG_S02_P000001_C0001_B101_V01_T01_L01_A15_E02_tissue.tif")
# # matches_to_run = [(image_path, mask_path),]

# rerun_unfinished = False # [READ THIS -->] this will check if the '<stem>_runtime.txt file is present in the outputfolder. If it is not it means that this file is not processed fully. IMPORTANT: if this is set to True multiple parallel jobs cannot track which slides are being processed, set this to false if you want to run multiple jobs in a parralel way. Alternatively, remove all unfinished files in the output folder (the files that do nt have a .runtime.txt file). Default is False 

# ### SAMPLING STUFF
# spacing = 0.5 # spacing for batch_shape spacing and annotation_parser output_spacing (leave its processing spacing on 4 or higher)
# model_patch_size = 512 # input size of model (should be square)
# sampler_patch_size = 4 * model_patch_size # size of the sampled patch

# ### WANDB
# wandb = False
# wandb_api_key = ''