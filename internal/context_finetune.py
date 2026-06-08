#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Mapping

import torch

from batchgenerators.utilities.file_and_folder_operations import load_json

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_CutMixStainEMA_Context1024FT100 import (
    nnUNetTrainer_CutMixStainEMA_Context1024FT100,
)


def unwrap_model(model):
    return getattr(model, "_orig_mod", model)


def normalized_state_dict(
    model,
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Normalize common DataParallel / compile prefixes before strict loading.
    """
    target = unwrap_model(model)
    target_keys = set(target.state_dict().keys())

    result: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        candidate = key

        if candidate.startswith("module.") and candidate[7:] in target_keys:
            candidate = candidate[7:]

        if (
            candidate.startswith("_orig_mod.")
            and candidate[10:] in target_keys
        ):
            candidate = candidate[10:]

        result[candidate] = value

    return result


def strict_load(model, state_dict, label: str) -> None:
    target = unwrap_model(model)

    normalized = normalized_state_dict(
        target,
        state_dict,
    )

    load_result = target.load_state_dict(
        normalized,
        strict=True,
    )

    print(
        f"Loaded {label} weights strictly. "
        f"Missing={load_result.missing_keys}, "
        f"Unexpected={load_result.unexpected_keys}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        required=True,
        type=int,
        choices=range(5),
    )

    args = parser.parse_args()

    fold = args.fold

    nnunet_preprocessed = Path(os.environ["nnUNet_preprocessed"])
    nnunet_results = Path(os.environ["nnUNet_results"])

    dataset_folder = (
        nnunet_preprocessed
        / "Dataset301_BEETLE"
    )

    plans = load_json(
        dataset_folder
        / "nnUNetWholeSlideDataPlans.json"
    )

    dataset_json = load_json(
        dataset_folder
        / "dataset.json"
    )

    base_model_dir = (
        nnunet_results
        / "Dataset301_BEETLE"
        / (
            "nnUNetTrainer_CutMixStainEMA__"
            "nnUNetWholeSlideDataPlans__"
            "wsd_None_iterator_nnunet_aug__2d"
        )
    )

    context_model_dir = (
        nnunet_results
        / "Dataset301_BEETLE"
        / (
            "nnUNetTrainer_CutMixStainEMA_Context1024FT100__"
            "nnUNetWholeSlideDataPlans__"
            "wsd_None_iterator_nnunet_aug__2d_context1024"
        )
    )

    source_checkpoint = (
        base_model_dir
        / f"fold_{fold}"
        / "checkpoint_best.pth"
    )

    output_folder = (
        context_model_dir
        / f"fold_{fold}"
    )

    latest_checkpoint = (
        output_folder
        / "checkpoint_latest.pth"
    )

    final_checkpoint = (
        output_folder
        / "checkpoint_final.pth"
    )

    if final_checkpoint.is_file():
        print(
            f"Context fold {fold} already completed: "
            f"{final_checkpoint}",
            flush=True,
        )
        return

    trainer = nnUNetTrainer_CutMixStainEMA_Context1024FT100(
        plans=plans,
        configuration="2d_context1024",
        fold=fold,
        dataset_json=dataset_json,
        unpack_dataset=True,
        device=torch.device("cuda"),
    )

    if latest_checkpoint.is_file():
        print(
            f"Resuming context fold {fold} from: "
            f"{latest_checkpoint}",
            flush=True,
        )

        trainer.load_checkpoint(
            str(latest_checkpoint)
        )

        print(
            f"Resumed context fold {fold} at epoch "
            f"{trainer.current_epoch}",
            flush=True,
        )

    else:
        if not source_checkpoint.is_file():
            raise FileNotFoundError(
                "Missing base-model best checkpoint: "
                f"{source_checkpoint}"
            )

        print(
            f"Starting context fold {fold} from base checkpoint: "
            f"{source_checkpoint}",
            flush=True,
        )

        trainer.initialize()

        checkpoint = torch.load(
            source_checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        if "network_weights" not in checkpoint:
            raise KeyError(
                "Base checkpoint is missing network_weights. "
                f"Available keys: {sorted(checkpoint.keys())}"
            )

        source_weights = checkpoint["network_weights"]

        strict_load(
            trainer.network,
            source_weights,
            "network",
        )

        if not trainer._ema_initialized:
            trainer._build_ema_model()

        strict_load(
            trainer.ema_model,
            source_weights,
            "EMA",
        )

        trainer.current_epoch = 0
        trainer._global_step = 0

        print(
            "Initialized fresh 100-epoch context refinement "
            "from the base general-best checkpoint",
            flush=True,
        )

    trainer.run_training()


if __name__ == "__main__":
    main()
