#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from nnunetv2.run.run_training_pathology import get_trainer_from_args


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume pathology nnU-Net training from checkpoint_latest.pth."
    )

    parser.add_argument("dataset_name_or_id")
    parser.add_argument("fold", type=int)
    parser.add_argument("trainer")

    parser.add_argument(
        "--planner",
        default="nnUNetWholeSlideDataPlans",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Resume checkpoint does not exist: {checkpoint}"
        )

    trainer = get_trainer_from_args(
        args.dataset_name_or_id,
        "2d",
        args.fold,
        args.trainer,
        args.planner,
    )

    print(f"Loading checkpoint: {checkpoint}", flush=True)

    trainer.load_checkpoint(str(checkpoint))

    print("Checkpoint loaded successfully", flush=True)
    print(f"Resuming from epoch: {trainer.current_epoch}", flush=True)
    print(f"Best EMA: {trainer._best_ema}", flush=True)

    trainer.run_training()


if __name__ == "__main__":
    main()
