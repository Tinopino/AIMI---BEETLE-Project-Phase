#!/usr/bin/env python3
"""Single user-facing training entry point for all BEETLE experiments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

from experiments import DATASET_ID, PLANNER, get_experiment, repo_root


def require_environment() -> None:
    required = ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Source reproducibility/configs/cluster_paths.env.example "
              "after adapting it to your system."
        )


def regular_training_command(experiment_name: str, fold: int) -> list[str]:
    exp = get_experiment(experiment_name)
    return [
        sys.executable,
        "-u",
        str(repo_root() / "nnUNet_pathology" / "nnunetv2" / "run" / "run_training_pathology.py"),
        DATASET_ID,
        str(fold),
        exp.trainer,
        "--planner",
        PLANNER,
    ]


def context_finetuning_command(fold: int) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(repo_root() / "internal" / "context_finetune.py"),
        "--fold",
        str(fold),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train or resume a BEETLE experiment. Standard trainers automatically "
            "resume when checkpoint_latest.pth exists. The context experiment "
            "imports the completed CutMix-stain-EMA best checkpoint and then "
            "fine-tunes for 100 epochs."
        )
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--fold", required=True, type=int, choices=range(5))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    require_environment()
    exp = get_experiment(args.experiment)

    if exp.mode == "finetune":
        command = context_finetuning_command(args.fold)
    else:
        command = regular_training_command(args.experiment, args.fold)

    print("Experiment:", exp.name)
    print("Trainer:   ", exp.trainer)
    print("Fold:      ", args.fold)
    print("Command:   ", shlex.join(command), flush=True)

    if args.dry_run:
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
