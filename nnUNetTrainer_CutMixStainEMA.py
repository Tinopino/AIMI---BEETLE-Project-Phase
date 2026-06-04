from copy import deepcopy
from typing import Union

import numpy as np
import torch
from torch import autocast
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import dummy_context
from nnUNetTrainerBEETLE import nnUNetTrainerPathologyFocal
from stain_transforms import StainJitterTransform


class nnUNetTrainer_CutMixStainEMA(nnUNetTrainerPathologyFocal):
    CUTMIX_PROB = 0.35
    CUTMIX_ALPHA = 1.0
    CUTMIX_DISABLE_LAST_FRAC = 0.2
    EMA_DECAY = 0.999
    EMA_WARMUP_STEPS = 1000
    VALIDATE_WITH_EMA = True
    STAIN_JITTER_SIGMA = 0.05
    STAIN_JITTER_BIAS = 0.05
    STAIN_JITTER_PROB = 0.8

    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.ema_model: Union[torch.nn.Module, None] = None
        self._ema_initialized = False
        self._global_step = 0

    @staticmethod
    def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
        if isinstance(model, DDP):
            model = model.module
        if isinstance(model, OptimizedModule):
            model = model._orig_mod
        return model

    def _build_ema_model(self):
        if self._ema_initialized:
            return
        base = self._unwrap(self.network)
        ema = deepcopy(base).to(self.device)
        for p in ema.parameters():
            p.requires_grad_(False)
        ema.eval()
        self.ema_model = ema
        self._ema_initialized = True
        self.print_to_log_file(
            f"EMA model built. decay={self.EMA_DECAY}, warmup={self.EMA_WARMUP_STEPS} steps."
        )

    @torch.no_grad()
    def _update_ema(self):
        if not self._ema_initialized:
            return
        decay = 0.0 if self._global_step < self.EMA_WARMUP_STEPS else self.EMA_DECAY
        msd = self._unwrap(self.network).state_dict()
        esd = self.ema_model.state_dict()
        for k, ev in esd.items():
            mv = msd[k]
            if ev.dtype.is_floating_point:
                ev.mul_(decay).add_(mv.detach(), alpha=1.0 - decay)
            else:
                ev.copy_(mv)

    def _cutmix_active(self) -> bool:
        cutoff = int(self.num_epochs * (1.0 - self.CUTMIX_DISABLE_LAST_FRAC))
        return self.current_epoch < cutoff

    @staticmethod
    def _rand_bbox(spatial_shape, lam):
        ndim = len(spatial_shape)
        cut_ratio = (1.0 - lam) ** (1.0 / ndim)
        out = []
        for dim_size in spatial_shape:
            cut = max(1, int(round(dim_size * cut_ratio)))
            center = np.random.randint(0, dim_size)
            lo = max(0, center - cut // 2)
            hi = min(dim_size, lo + cut)
            lo = max(0, hi - cut)
            out.append((lo, hi))
        return out

    @staticmethod
    def _scale_box(box_lohis, src_shape, dst_shape):
        out = []
        for (lo, hi), s, d in zip(box_lohis, src_shape, dst_shape):
            scale = d / max(s, 1)
            lo2 = int(round(lo * scale))
            hi2 = int(round(hi * scale))
            lo2 = max(0, min(lo2, d))
            hi2 = max(lo2 + 1, min(hi2, d))
            out.append((lo2, hi2))
        return out

    @staticmethod
    def _make_slices(box_lohis):
        return (slice(None), slice(None), *[slice(lo, hi) for lo, hi in box_lohis])

    def _apply_cutmix(self, data: torch.Tensor, target):
        if not self._cutmix_active():
            return data, target
        if np.random.rand() > self.CUTMIX_PROB or data.size(0) < 2:
            return data, target

        lam = float(np.random.beta(self.CUTMIX_ALPHA, self.CUTMIX_ALPHA))
        perm = torch.randperm(data.size(0), device=data.device)

        data_spatial = tuple(data.shape[2:])
        box = self._rand_bbox(data_spatial, lam)
        sl = self._make_slices(box)

        data = data.clone()
        data[sl] = data[perm][sl]

        if isinstance(target, list):
            new_target = []
            for t in target:
                t = t.clone()
                t_box = self._scale_box(box, data_spatial, tuple(t.shape[2:]))
                t_sl = self._make_slices(t_box)
                t[t_sl] = t[perm][t_sl]
                new_target.append(t)
            target = new_target
        else:
            target = target.clone()
            target[sl] = target[perm][sl]
        return data, target

    @staticmethod
    def get_training_transforms(*args, **kwargs):
        # Parent's static method builds the full Compose
        composed = nnUNetTrainerPathologyFocal.get_training_transforms(*args, **kwargs)
        # Insert stain jitter near the start of the pipeline, right after
        # spatial transforms (which produce contiguous numpy data).
        stain = StainJitterTransform(
            sigma=nnUNetTrainer_CutMixStainEMA.STAIN_JITTER_SIGMA,
            bias=nnUNetTrainer_CutMixStainEMA.STAIN_JITTER_BIAS,
            p_per_sample=nnUNetTrainer_CutMixStainEMA.STAIN_JITTER_PROB,
            data_key='data',
        )
        composed.transforms.insert(1, stain)
        return composed

    def initialize(self):
        super().initialize()
        self._build_ema_model()

    def set_deep_supervision_enabled(self, enabled: bool):
        super().set_deep_supervision_enabled(enabled)
        if self._ema_initialized:
            self.ema_model.decoder.deep_supervision = enabled

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        data, target = self._apply_cutmix(data, target)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        self._global_step += 1
        self._update_ema()
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        if not (self.VALIDATE_WITH_EMA and self._ema_initialized):
            return super().validation_step(batch)

        data = batch['data']
        target = batch['target']
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.ema_model(data)
            del data
            l = self.loss(output, target)

        output = output[0]
        target = target[0]
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(),
                'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def on_train_end(self):
        if self._ema_initialized:
            self.print_to_log_file("Copying EMA weights into network for final checkpoint.")
            self._unwrap(self.network).load_state_dict(self.ema_model.state_dict())
        super().on_train_end()

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank != 0 or self.disable_checkpointing:
            return
        mod = self._unwrap(self.network)
        checkpoint = {'network_weights': mod.state_dict(),
            'ema_state_dict': self.ema_model.state_dict() if self._ema_initialized else None,
            '_global_step': self._global_step,
            'optimizer_state': self.optimizer.state_dict(),
            'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
            'logging': self.logger.get_checkpoint(),
            '_best_ema': self._best_ema,
            'current_epoch': self.current_epoch + 1,
            'init_args': self.my_init_kwargs,
            'trainer_name': self.__class__.__name__,
            'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,}
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        super().load_checkpoint(filename_or_checkpoint)
        if isinstance(filename_or_checkpoint, str):
            ckpt = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
        else:
            ckpt = filename_or_checkpoint
        if not self._ema_initialized:
            self._build_ema_model()
        if ckpt.get('ema_state_dict') is not None:
            self.ema_model.load_state_dict(ckpt['ema_state_dict'])
        self._global_step = ckpt.get('_global_step', 0)
