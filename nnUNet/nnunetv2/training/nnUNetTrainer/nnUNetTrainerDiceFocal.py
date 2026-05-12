import torch
from torch import nn
import numpy as np
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        # nnU-Net passes target as (B, 1, X, Y, Z). CrossEntropy expects (B, X, Y, Z)
        target = target[:, 0].long()
        
        ce_loss = nn.functional.cross_entropy(
            input, target, weight=self.alpha, reduction='none', ignore_index=self.ignore_index
        )
        
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)

        if self.ignore_index != -100:
            mask = target != self.ignore_index
            focal_loss = focal_loss[mask]

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DC_and_Focal_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, focal_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        super(DC_and_Focal_loss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.dc = MemoryEfficientSoftDiceLoss(**soft_dice_kwargs)
        self.focal = FocalLoss(**focal_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        focal_loss = self.focal(net_output, target)
        
        return self.weight_dice * dc_loss + self.weight_ce * focal_loss


class nnUNetTrainerDiceFocal(nnUNetTrainer):
    def _build_loss(self):
        # HACK: Bypass nnU-Net's strict __init__ by setting epochs here for local testing on MAC with limited resources. Remove this line for actual training.
        #self.num_epochs = 1
        
        loss_opts = {
            'batch_dice': self.configuration_manager.batch_dice,
            'smooth': 1e-5, 
            'do_bg': False, 
            'ddp': self.is_ddp
        }
        
        ignore_label = self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100

        loss = DC_and_Focal_loss(
            soft_dice_kwargs=loss_opts,
            focal_kwargs={'gamma': 2.0, 'reduction': 'mean', 'ignore_index': ignore_label},
            weight_ce=1,
            weight_dice=1,
            ignore_label=ignore_label
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0  
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss