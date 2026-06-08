import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

from nnunetv2.training.nnUNetTrainer.variants.pathology.nnUNetTrainer_WSD_wei_i0_nnunet_aug_json import (
    nnUNetTrainer_WSD_wei_i0_nnunet_aug_json,
)


def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=1)


class FocalLoss(nn.Module):
    """
    Focal loss for multiclass semantic segmentation.

    Expects:
        input:  (B, C, H, W) or (B, C, D, H, W)
        target: (B, 1, H, W) or (B, 1, D, H, W)
                or already squeezed as (B, H, W) / (B, D, H, W)
    """

    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean", ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

        if alpha is not None:
            self.alpha = torch.as_tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None

    @staticmethod
    def _squeeze_target(target: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # nnU-Net usually gives target as (B, 1, H, W) or (B, 1, D, H, W)
        if target.ndim == input_tensor.ndim and target.shape[1] == 1:
            target = target[:, 0]

        return target.long()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = self._squeeze_target(target, input)

        weight = self.alpha.to(input.device) if self.alpha is not None else None

        ce_loss = F.cross_entropy(
            input,
            target,
            weight=weight,
            reduction="none",
            ignore_index=self.ignore_index,
        )

        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.ignore_index != -100:
            valid_mask = target != self.ignore_index
            focal_loss = focal_loss[valid_mask]

            # Avoid NaN if a patch contains only ignored pixels
            if focal_loss.numel() == 0:
                return input.sum() * 0.0

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        elif self.reduction == "none":
            return focal_loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class DC_and_Focal_loss(nn.Module):
    """
    Dice + Focal loss.

    This replaces nnU-Net's usual Dice + CrossEntropy loss with
    Dice + Focal loss, while keeping deep supervision handled outside
    by DeepSupervisionWrapper.
    """

    def __init__(
        self,
        soft_dice_kwargs: dict,
        focal_kwargs: dict,
        weight_dice: float = 1.0,
        weight_focal: float = 1.0,
        ignore_label=None,
    ):
        super().__init__()

        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.ignore_label = ignore_label

        self.dc = MemoryEfficientSoftDiceLoss(**soft_dice_kwargs)
        self.focal = FocalLoss(**focal_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.ignore_label is not None:
            target_for_dice = target.clone()
            loss_mask = target_for_dice != self.ignore_label
            target_for_dice[target_for_dice == self.ignore_label] = 0

            try:
                dc_loss = self.dc(net_output, target_for_dice, loss_mask=loss_mask)
            except TypeError:
                # Fallback for older nnU-Net versions whose Dice loss does not accept loss_mask
                dc_loss = self.dc(net_output, target_for_dice)
        else:
            dc_loss = self.dc(net_output, target)

        focal_loss = self.focal(net_output, target)

        return self.weight_dice * dc_loss + self.weight_focal * focal_loss


class nnUNetTrainer_WSD_wei_i0_nnunet_aug_json_DiceFocal(
    nnUNetTrainer_WSD_wei_i0_nnunet_aug_json
):
    """
    Pathology nnU-Net trainer using Dice + Focal loss.

    Important:
    - This subclasses the working pathology trainer:
      nnUNetTrainer_WSD_wei_i0_nnunet_aug_json
    - Therefore it keeps the pathology/WSI dataloader and sampling behavior.
    - Only _build_loss() is changed.
    """

    def _build_loss(self):
        # Focal loss below is for multiclass softmax training.
        # If someone later uses this trainer with region-based training,
        # fall back to the parent loss instead of silently doing the wrong thing.
        if getattr(self.label_manager, "has_regions", False):
            self.print_to_log_file(
                "WARNING: DiceFocal trainer does not support region-based training. "
                "Falling back to parent _build_loss()."
            )
            return super()._build_loss()

        ignore_label = (
            self.label_manager.ignore_label
            if getattr(self.label_manager, "has_ignore_label", False)
            else None
        )

        focal_ignore_index = ignore_label if ignore_label is not None else -100

        loss_opts = {
            "apply_nonlin": softmax_helper_dim1,
            "batch_dice": self.configuration_manager.batch_dice,
            "smooth": 1e-5,
            "do_bg": False,
            "ddp": self.is_ddp,
        }

        loss = DC_and_Focal_loss(
            soft_dice_kwargs=loss_opts,
            focal_kwargs={
                "gamma": 2.0,
                "alpha": None,
                "reduction": "mean",
                "ignore_index": focal_ignore_index,
            },
            weight_dice=1.0,
            weight_focal=1.0,
            ignore_label=ignore_label,
        )

        # Same deep supervision weighting logic as nnU-Net/pathology trainer.
        deep_supervision_scales = self._get_deep_supervision_scales()

        weights = np.array(
            [1 / (2 ** i) for i in range(len(deep_supervision_scales))],
            dtype=np.float32,
        )

        # nnU-Net usually ignores the lowest-resolution output.
        # In DDP without compile, a true zero can sometimes trigger unused-parameter issues.
        if self.is_ddp and not self._do_i_compile():
            weights[-1] = 1e-6
        else:
            weights[-1] = 0.0

        weights = weights / weights.sum()
        loss = DeepSupervisionWrapper(loss, weights)

        return loss