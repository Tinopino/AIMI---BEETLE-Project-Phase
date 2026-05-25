import torch
import torch.nn as nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.compound_losses import MultipleOutputLoss2

class FocalLoss(nn.Module):
    """
    Standard PyTorch implementation of Focal Loss for multi-class segmentation.
    """
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logit, target):
        if len(target.shape) == len(logit.shape):
            target = target.squeeze(1)
        ce_loss = self.ce(logit, target.long())
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class DC_and_Focal_loss(nn.Module):
    """
    Custom wrapper to combine Dice Loss and Focal Loss.
    """
    def __init__(self, soft_dice_kwargs, focal_kwargs, weight_ce=1.0, weight_dice=1.0, ignore_label=None):
        super(DC_and_Focal_loss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label
        
        self.dc = MemoryEfficientSoftDiceLoss(**soft_dice_kwargs)
        self.focal = FocalLoss(**focal_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        focal_loss = self.focal(net_output, target) if self.weight_ce != 0 else 0
        return self.weight_dice * dc_loss + self.weight_ce * focal_loss

# NOTE: In nnUNet-for-pathology, WSI handling, stain augmentations, and batch 
# norm are handled via the default nnUNetTrainer or its specific plans files.
# By inheriting from nnUNetTrainer here, we get all the pathology features 
# automatically when using the pathology fork, while only overriding the loss.

class nnUNetTrainerPathologyFocal(nnUNetTrainer):
    """
    Custom Trainer for the BEETLE Challenge WSI data.
    Uses Dice + Focal Loss to handle severe class imbalance for Necrosis
    while retaining the pathology fork's native WSI loading and augmentations.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # IMPORTANT: Set to 250 or 1000 for your actual server training run!
        self.num_epochs = 250 

    def _build_loss(self):
        dice_kwargs = {
            'batch_dice': self.configuration_manager.batch_dice,
            'smooth': 1e-5,
            'do_bg': False,
            'ddp': self.is_ddp
        }
        
        # gamma=2.0 heavily penalizes the network for missing rare classes (necrosis)
        focal_kwargs = {'gamma': 2.0}

        loss = DC_and_Focal_loss(
            soft_dice_kwargs=dice_kwargs,
            focal_kwargs=focal_kwargs,
            weight_ce=1.0,
            weight_dice=1.0,
            ignore_label=self.label_manager.ignore_label
        )

        if self.enable_deep_supervision:
            from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
            loss = DeepSupervisionWrapper(loss, self.configuration_manager.pool_op_kernel_sizes)
            
        return loss