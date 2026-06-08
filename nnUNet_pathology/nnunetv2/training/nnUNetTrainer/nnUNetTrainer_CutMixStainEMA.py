from copy import deepcopy
from typing import Union

import numpy as np
import torch
from torch import autocast
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import dummy_context

from .nnUnetTrainerBEETLE import (
    nnUNetTrainerPathologyFocalClassMetricsAlpha1000Milestones as ParentTrainer,
)
from .stain_transforms import StainJitterTransform


class nnUNetTrainer_CutMixStainEMA(ParentTrainer):
    """
    BEETLE trainer extending the weighted-focal 1000-epoch milestone trainer
    with:
      - CutMix augmentation
      - H&E stain jitter augmentation
      - Exponential moving average (EMA) weights
      - EMA-based validation and inference checkpoints
    """

    CUTMIX_PROB = 0.35
    CUTMIX_ALPHA = 1.0
    CUTMIX_DISABLE_LAST_FRAC = 0.2

    EMA_DECAY = 0.999
    EMA_WARMUP_STEPS = 1000
    VALIDATE_WITH_EMA = True

    STAIN_JITTER_SIGMA = 0.05
    STAIN_JITTER_BIAS = 0.05
    STAIN_JITTER_PROB = 0.8

    def __init__(
        self,
        plans,
        configuration,
        fold,
        dataset_json,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans,
            configuration,
            fold,
            dataset_json,
            unpack_dataset,
            device,
        )

        self.ema_model: Union[torch.nn.Module, None] = None
        self._ema_initialized = False
        self._global_step = 0

    @staticmethod
    def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
        """
        Return the original model when torch.compile or DDP wrapping is used.
        """
        if isinstance(model, DDP):
            model = model.module

        if isinstance(model, OptimizedModule):
            model = model._orig_mod

        return model

    def _build_ema_model(self) -> None:
        """
        Create a frozen EMA copy of the network.
        """
        if self._ema_initialized:
            return

        base_model = self._unwrap(self.network)
        ema_model = deepcopy(base_model).to(self.device)

        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)

        ema_model.eval()

        self.ema_model = ema_model
        self._ema_initialized = True

        self.print_to_log_file(
            "EMA model built. "
            f"decay={self.EMA_DECAY}, "
            f"warmup={self.EMA_WARMUP_STEPS} steps."
        )

    @torch.no_grad()
    def _update_ema(self) -> None:
        """
        Update the EMA model after each optimizer step.

        During warm-up, decay is set to zero so that the EMA model directly
        follows the current network. Afterwards, the configured EMA decay is
        applied.
        """
        if not self._ema_initialized:
            return

        decay = (
            0.0
            if self._global_step < self.EMA_WARMUP_STEPS
            else self.EMA_DECAY
        )

        model_state = self._unwrap(self.network).state_dict()
        ema_state = self.ema_model.state_dict()

        for key, ema_value in ema_state.items():
            model_value = model_state[key]

            if ema_value.dtype.is_floating_point:
                ema_value.mul_(decay).add_(
                    model_value.detach(),
                    alpha=1.0 - decay,
                )
            else:
                ema_value.copy_(model_value)

    def _cutmix_active(self) -> bool:
        """
        Disable CutMix during the final fraction of training so that the model
        can converge on unmodified examples.
        """
        cutoff_epoch = int(
            self.num_epochs * (1.0 - self.CUTMIX_DISABLE_LAST_FRAC)
        )

        return self.current_epoch < cutoff_epoch

    @staticmethod
    def _rand_bbox(spatial_shape, lam):
        """
        Sample a random CutMix bounding box.
        """
        ndim = len(spatial_shape)
        cut_ratio = (1.0 - lam) ** (1.0 / ndim)

        output = []

        for dim_size in spatial_shape:
            cut_size = max(1, int(round(dim_size * cut_ratio)))
            center = np.random.randint(0, dim_size)

            lower = max(0, center - cut_size // 2)
            upper = min(dim_size, lower + cut_size)
            lower = max(0, upper - cut_size)

            output.append((lower, upper))

        return output

    @staticmethod
    def _scale_box(box_lohis, src_shape, dst_shape):
        """
        Scale a CutMix box to the resolution of a deep-supervision target.
        """
        output = []

        for (lower, upper), src_size, dst_size in zip(
            box_lohis,
            src_shape,
            dst_shape,
        ):
            scale = dst_size / max(src_size, 1)

            lower_scaled = int(round(lower * scale))
            upper_scaled = int(round(upper * scale))

            lower_scaled = max(0, min(lower_scaled, dst_size))
            upper_scaled = max(
                lower_scaled + 1,
                min(upper_scaled, dst_size),
            )

            output.append((lower_scaled, upper_scaled))

        return output

    @staticmethod
    def _make_slices(box_lohis):
        """
        Convert a bounding box to tensor slices.
        """
        return (
            slice(None),
            slice(None),
            *[slice(lower, upper) for lower, upper in box_lohis],
        )

    def _apply_cutmix(self, data: torch.Tensor, target):
        """
        Apply CutMix to the image batch and the corresponding segmentation
        targets. Deep-supervision target resolutions are handled separately.
        """
        if not self._cutmix_active():
            return data, target

        if np.random.rand() > self.CUTMIX_PROB or data.size(0) < 2:
            return data, target

        lam = float(
            np.random.beta(
                self.CUTMIX_ALPHA,
                self.CUTMIX_ALPHA,
            )
        )

        permutation = torch.randperm(
            data.size(0),
            device=data.device,
        )

        data_spatial_shape = tuple(data.shape[2:])
        box = self._rand_bbox(data_spatial_shape, lam)
        data_slices = self._make_slices(box)

        data = data.clone()
        data[data_slices] = data[permutation][data_slices]

        if isinstance(target, list):
            new_target = []

            for target_tensor in target:
                target_tensor = target_tensor.clone()

                target_box = self._scale_box(
                    box,
                    data_spatial_shape,
                    tuple(target_tensor.shape[2:]),
                )

                target_slices = self._make_slices(target_box)

                target_tensor[target_slices] = target_tensor[
                    permutation
                ][target_slices]

                new_target.append(target_tensor)

            target = new_target

        else:
            target = target.clone()
            target[data_slices] = target[permutation][data_slices]

        return data, target

    @staticmethod
    def get_training_transforms(*args, **kwargs):
        """
        Extend the parent trainer's augmentation pipeline with stain jitter.
        """
        composed = ParentTrainer.get_training_transforms(*args, **kwargs)

        stain_jitter = StainJitterTransform(
            sigma=nnUNetTrainer_CutMixStainEMA.STAIN_JITTER_SIGMA,
            bias=nnUNetTrainer_CutMixStainEMA.STAIN_JITTER_BIAS,
            p_per_sample=nnUNetTrainer_CutMixStainEMA.STAIN_JITTER_PROB,
            data_key="data",
        )

        composed.transforms.insert(1, stain_jitter)

        return composed

    def initialize(self) -> None:
        """
        Initialize the regular training components and then create the EMA
        network.
        """
        super().initialize()
        self._build_ema_model()

    def set_deep_supervision_enabled(self, enabled: bool) -> None:
        """
        Keep the EMA decoder's deep-supervision setting synchronized with the
        training network.
        """
        super().set_deep_supervision_enabled(enabled)

        if self._ema_initialized:
            self.ema_model.decoder.deep_supervision = enabled

    def train_step(self, batch: dict) -> dict:
        """
        Run one optimizer step using CutMix where applicable and update EMA
        afterwards.
        """
        data = batch["data"]
        target = batch["target"]

        data = data.to(
            self.device,
            non_blocking=True,
        )

        if isinstance(target, list):
            target = [
                target_tensor.to(
                    self.device,
                    non_blocking=True,
                )
                for target_tensor in target
            ]
        else:
            target = target.to(
                self.device,
                non_blocking=True,
            )

        data, target = self._apply_cutmix(
            data,
            target,
        )

        self.optimizer.zero_grad(
            set_to_none=True,
        )

        autocast_context = (
            autocast(
                self.device.type,
                enabled=True,
            )
            if self.device.type == "cuda"
            else dummy_context()
        )

        with autocast_context:
            output = self.network(data)
            loss = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()

            self.grad_scaler.unscale_(
                self.optimizer,
            )

            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(),
                12,
            )

            self.grad_scaler.step(
                self.optimizer,
            )

            self.grad_scaler.update()

        else:
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(),
                12,
            )

            self.optimizer.step()

        self._global_step += 1
        self._update_ema()

        return {
            "loss": loss.detach().cpu().numpy(),
        }

    def validation_step(self, batch: dict) -> dict:
        """
        Evaluate using EMA weights when enabled.
        """
        if not (
            self.VALIDATE_WITH_EMA
            and self._ema_initialized
        ):
            return super().validation_step(batch)

        data = batch["data"]
        target = batch["target"]

        data = data.to(
            self.device,
            non_blocking=True,
        )

        if isinstance(target, list):
            target = [
                target_tensor.to(
                    self.device,
                    non_blocking=True,
                )
                for target_tensor in target
            ]
        else:
            target = target.to(
                self.device,
                non_blocking=True,
            )

        autocast_context = (
            autocast(
                self.device.type,
                enabled=True,
            )
            if self.device.type == "cuda"
            else dummy_context()
        )

        with autocast_context:
            output = self.ema_model(data)
            del data

            loss = self.loss(
                output,
                target,
            )

        output = output[0]
        target = target[0]

        axes = [
            0,
            *list(range(2, output.ndim)),
        ]

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (
                torch.sigmoid(output) > 0.5
            ).long()

        else:
            output_segmentation = output.argmax(1)[:, None]

            predicted_segmentation_onehot = torch.zeros(
                output.shape,
                device=output.device,
                dtype=torch.float32,
            )

            predicted_segmentation_onehot.scatter_(
                1,
                output_segmentation,
                1,
            )

            del output_segmentation

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (
                    target != self.label_manager.ignore_label
                ).float()

                target[
                    target == self.label_manager.ignore_label
                ] = 0

            else:
                mask = 1 - target[:, -1:]
                target = target[:, :-1]

        else:
            mask = None

        true_positive, false_positive, false_negative, _ = (
            get_tp_fp_fn_tn(
                predicted_segmentation_onehot,
                target,
                axes=axes,
                mask=mask,
            )
        )

        true_positive_hard = (
            true_positive.detach().cpu().numpy()
        )

        false_positive_hard = (
            false_positive.detach().cpu().numpy()
        )

        false_negative_hard = (
            false_negative.detach().cpu().numpy()
        )

        if not self.label_manager.has_regions:
            true_positive_hard = true_positive_hard[1:]
            false_positive_hard = false_positive_hard[1:]
            false_negative_hard = false_negative_hard[1:]

        return {
            "loss": loss.detach().cpu().numpy(),
            "tp_hard": true_positive_hard,
            "fp_hard": false_positive_hard,
            "fn_hard": false_negative_hard,
        }

    def on_train_end(self) -> None:
        """
        Copy EMA weights into the main network before the final checkpoint is
        written.
        """
        if self._ema_initialized:
            self.print_to_log_file(
                "Copying EMA weights into network for final checkpoint."
            )

            self._unwrap(self.network).load_state_dict(
                self.ema_model.state_dict()
            )

        super().on_train_end()

    def save_checkpoint(self, filename: str) -> None:
        """
        Save all fields created by the parent trainer.

        For compatibility with standard nnU-Net inference:
          - network_weights stores EMA weights when EMA validation is enabled.
          - raw_network_weights stores the ordinary training-network weights.
        """
        if self.local_rank != 0 or self.disable_checkpointing:
            return

        super().save_checkpoint(filename)

        checkpoint = torch.load(
            filename,
            map_location="cpu",
            weights_only=False,
        )

        checkpoint["raw_network_weights"] = deepcopy(
            checkpoint["network_weights"]
        )

        ema_state = None

        if self._ema_initialized:
            ema_state = {
                key: value.detach().cpu()
                for key, value in self.ema_model.state_dict().items()
            }

        checkpoint["ema_state_dict"] = ema_state
        checkpoint["_global_step"] = self._global_step

        if (
            self.VALIDATE_WITH_EMA
            and ema_state is not None
        ):
            checkpoint["network_weights"] = ema_state

        torch.save(
            checkpoint,
            filename,
        )

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        """
        Resume training using ordinary network weights while restoring the EMA
        state separately.
        """
        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(
                filename_or_checkpoint,
                map_location=self.device,
                weights_only=False,
            )
        else:
            checkpoint = filename_or_checkpoint

        resume_checkpoint = dict(checkpoint)

        if checkpoint.get("raw_network_weights") is not None:
            resume_checkpoint["network_weights"] = checkpoint[
                "raw_network_weights"
            ]

        super().load_checkpoint(
            resume_checkpoint
        )

        if not self._ema_initialized:
            self._build_ema_model()

        if checkpoint.get("ema_state_dict") is not None:
            self.ema_model.load_state_dict(
                checkpoint["ema_state_dict"]
            )

        self._global_step = checkpoint.get(
            "_global_step",
            0,
        )