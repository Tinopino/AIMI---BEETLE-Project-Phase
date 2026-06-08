import os
import torch
from nnunetv2.training.nnUNetTrainer.variants.pathology.nnUNetTrainer_WSD_undefined_dataloader import nnUNetTrainer_WSD_undefined_dataloader

class nnUNetTrainer_WSD_points_wei_i0_nnunet_aug_json(nnUNetTrainer_WSD_undefined_dataloader):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
 
        # SET THESE
        self.ignore0 = True
        self.time = False
        self.albumentations_aug = False
        self.label_sampling_strategy = 'weighted' #'roi' # 'balanced' # 'weighted' 
        self.sample_double = False # this means we for example sample 1024x1024, augment, and return 512x512 center crop to remove artifacts induced by zooming and rotating, not needed if using albumentations_aug

        # AUTO
        self.wandb = True if 'WANDB_API_KEY' in os.environ else False
        self.aug = 'alb' if self.albumentations_aug else 'nnunet'
        self.iterator_template = f'wsd_{self.label_sampling_strategy}_point_iterator_{self.aug}_aug_json'

    def modify_fill_template(self, fill_template):
        point_to_seg_callback_name = "PointsToSegBatchCallback"
        callback_idx = [fill_template['batch_callbacks'][i]['*object'].split('.')[-1] for i in range(len(fill_template['batch_callbacks']))].index(point_to_seg_callback_name)
            
        fill_template['batch_callbacks'][callback_idx]['point_sizes_dict'] = self.dataset_json["point_sizes_dict"]
        fill_template['batch_callbacks'][callback_idx]['spacing'] = self.dataset_json["spacing"] if "spacing" in self.dataset_json else 0.5