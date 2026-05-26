import torch
import torch.nn as nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

class FocalLoss(nn.Module):
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

class nnUNetTrainerPathologyFocal(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250

    def _build_loss(self):
        dice_kwargs = {
            'batch_dice': self.configuration_manager.batch_dice,
            'smooth': 1e-5,
            'do_bg': False,
            'ddp': self.is_ddp
        }
        
        focal_kwargs = {'gamma': 2.0}

        loss = DC_and_Focal_loss(
            soft_dice_kwargs=dice_kwargs,
            focal_kwargs=focal_kwargs,
            weight_ce=1.0,
            weight_dice=1.0,
            ignore_label=self.label_manager.ignore_label
        )

        # Check if deep_supervision is enabled using getattr to be completely safe
        # across both old and new nnU-Net versions
        do_ds = getattr(self, 'enable_deep_supervision', getattr(self, 'deep_supervision', True))
        
        if do_ds:
            from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
            # In the pathology fork, deep supervision weights are precomputed and stored
            # in self.ds_loss_weights, so we must pass those instead of kernel sizes!
            loss = DeepSupervisionWrapper(loss, getattr(self, 'ds_loss_weights', None))
            
        return loss