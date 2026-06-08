#!/usr/bin/env python3
"""Central experiment registry for the BEETLE project.

This file is the single source of truth for:
- trainer classes;
- nnU-Net configurations;
- result-directory names;
- native WSI inference geometry;
- checkpoint names and tags;
- final fine-tuning dependencies.

All user-facing runners import this registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import os


DATASET_ID = "301"
DATASET_NAME = "Dataset301_BEETLE"
PLANNER = "nnUNetWholeSlideDataPlans"


@dataclass(frozen=True)
class Experiment:
    name: str
    trainer: str
    configuration: str
    model_directory: str
    patch_size: int
    sampler_tile: int
    output_tile: int
    checkpoint_name: str = "checkpoint_best.pth"
    checkpoint_tag: str = "checkpoint_best"
    use_mirroring: bool = True
    mode: str = "train"
    source_experiment: str | None = None
    description: str = ""

    def validate(self) -> None:
        expected_output = self.sampler_tile - self.patch_size
        if self.output_tile != expected_output:
            raise ValueError(
                f"{self.name}: expected output_tile={expected_output} from "
                f"sampler_tile - patch_size, received {self.output_tile}."
            )
        if self.mode not in {"train", "finetune"}:
            raise ValueError(f"{self.name}: unsupported mode {self.mode!r}.")
        if self.mode == "finetune" and not self.source_experiment:
            raise ValueError(f"{self.name}: fine-tuning requires source_experiment.")


EXPERIMENTS: dict[str, Experiment] = {
    "released_baseline": Experiment(
        name="released_baseline",
        trainer="nnUNetTrainer_WSD_wei_i0_nnunet_aug_json",
        configuration="2d",
        model_directory=(
            "nnUNetTrainer_WSD_wei_i0_nnunet_aug_json__"
            "nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"
        ),
        patch_size=512,
        sampler_tile=2048,
        output_tile=1536,
        checkpoint_tag="released_baseline_best_mirror_visual",
        description="Released nnU-Net-for-pathology reference.",
    ),
    "focal_250": Experiment(
        name="focal_250",
        trainer="nnUNetTrainerPathologyFocalClassMetrics",
        configuration="2d",
        model_directory=(
            "nnUNetTrainerPathologyFocalClassMetrics__"
            "nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"
        ),
        patch_size=512,
        sampler_tile=2048,
        output_tile=1536,
        checkpoint_tag="focalmetrics250_best_mirror_visual",
        description="Dice + focal-loss fold-0 ablation.",
    ),
    "weighted_focal_250": Experiment(
        name="weighted_focal_250",
        trainer="nnUNetTrainerPathologyFocalClassMetricsAlpha",
        configuration="2d",
        model_directory=(
            "nnUNetTrainerPathologyFocalClassMetricsAlpha__"
            "nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"
        ),
        patch_size=512,
        sampler_tile=2048,
        output_tile=1536,
        checkpoint_tag="alpha_ckpt_best_mirror_visual",
        description="Class-weighted focal-loss fold-0 ablation.",
    ),
    "confusion_aware_sampling_250": Experiment(
        name="confusion_aware_sampling_250",
        trainer="nnUNetTrainerPathologyWFCMAWS250",
        configuration="2d",
        model_directory=(
            "nnUNetTrainerPathologyWFCMAWS250__"
            "nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"
        ),
        patch_size=512,
        sampler_tile=2048,
        output_tile=1536,
        checkpoint_tag="wfcmaws250_best_mirror_visual",
        description="Confusion-aware sampling fold-0 ablation.",
    ),
    "hard_mining_250": Experiment(
        name="hard_mining_250",
        trainer="nnUNetTrainerPathologyWFCHardMining250",
        configuration="2d",
        model_directory=(
            "nnUNetTrainerPathologyWFCHardMining250__"
            "nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"
        ),
        patch_size=512,
        sampler_tile=2048,
        output_tile=1536,
        checkpoint_tag="wfhardmine250_best_mirror_visual",
        description="Targeted epithelial hard-example-mining fold-0 ablation.",
    ),
    "weighted_focal_1000": Experiment(
        name="weighted_focal_1000",
        trainer="nnUNetTrainerPathologyFocalClassMetricsAlpha1000Milestones",
        configuration="2d",
        model_directory=(
            "nnUNetTrainerPathologyFocalClassMetricsAlpha1000Milestones__"
            "nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"
        ),
        patch_size=512,
        sampler_tile=2048,
        output_tile=1536,
        checkpoint_tag="alpha1000_general_best_mirror_visual",
        description="Long weighted-focal run with milestone checkpoints.",
    ),
    "cutmix_stain_ema": Experiment(
        name="cutmix_stain_ema",
        trainer="nnUNetTrainer_CutMixStainEMA",
        configuration="2d",
        model_directory=(
            "nnUNetTrainer_CutMixStainEMA__"
            "nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"
        ),
        patch_size=512,
        sampler_tile=2048,
        output_tile=1536,
        checkpoint_tag="cutmixema1000_best_mirror_visual",
        description="Five-fold weighted-focal + CutMix + stain-jitter + EMA model.",
    ),
    "context1024_ft100": Experiment(
        name="context1024_ft100",
        trainer="nnUNetTrainer_CutMixStainEMA_Context1024FT100",
        configuration="2d_context1024",
        model_directory=(
            "nnUNetTrainer_CutMixStainEMA_Context1024FT100__"
            "nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d_context1024"
        ),
        patch_size=1024,
        sampler_tile=2048,
        output_tile=1024,
        checkpoint_tag="cutmixema1000_context1024_ft100_best_mirror_visual",
        mode="finetune",
        source_experiment="cutmix_stain_ema",
        description="Final five-fold context-1024 FT100 ensemble.",
    ),
}

for _experiment in EXPERIMENTS.values():
    _experiment.validate()


def get_experiment(name: str) -> Experiment:
    try:
        return EXPERIMENTS[name]
    except KeyError as exc:
        available = ", ".join(sorted(EXPERIMENTS))
        raise KeyError(f"Unknown experiment {name!r}. Available: {available}") from exc


def repo_root() -> Path:
    return Path(os.environ.get("AIMI_REPO_ROOT", Path(__file__).resolve().parent))


def results_root() -> Path:
    return Path(os.environ["nnUNet_results"]) / DATASET_NAME


def model_dir(experiment: Experiment) -> Path:
    return results_root() / experiment.model_directory


def validation_csv(fold: int) -> Path:
    folder = Path(os.environ.get("BEETLE_VALIDATION_CSV_DIR", str(Path.home())))
    return folder / f"fold{fold}_validation_inference_inputs.csv"


def output_root() -> Path:
    return Path(os.environ.get("BEETLE_OUTPUT_ROOT", str(repo_root() / "outputs")))


def visual_root() -> Path:
    fallback = output_root() / "visuals"
    return Path(os.environ.get("BEETLE_VISUAL_ROOT", str(fallback)))


def visual_dir(experiment: Experiment, fold: int) -> Path:
    return visual_root() / experiment.name / f"fold_{fold}" / experiment.checkpoint_tag


def external_roi_folder() -> Path:
    return Path(os.environ["BEETLE_EXTERNAL_ROIS"])


def external_output_folder(experiment: Experiment) -> Path:
    base = Path(os.environ.get(
        "BEETLE_INFERENCE_OUTPUT_ROOT",
        str(output_root() / "external_predictions"),
    ))
    return base / experiment.name


def external_zip_path(experiment: Experiment) -> Path:
    base = Path(os.environ.get(
        "BEETLE_INFERENCE_OUTPUT_ROOT",
        str(output_root() / "submissions"),
    ))
    return base / f"{experiment.name}.zip"


def print_registry() -> None:
    header = (
        "name",
        "trainer",
        "configuration",
        "patch",
        "sampler",
        "output",
        "mode",
        "description",
    )
    rows = [header]
    for name in sorted(EXPERIMENTS):
        exp = EXPERIMENTS[name]
        rows.append((
            exp.name,
            exp.trainer,
            exp.configuration,
            str(exp.patch_size),
            str(exp.sampler_tile),
            str(exp.output_tile),
            exp.mode,
            exp.description,
        ))
    widths = [max(len(row[index]) for row in rows) for index in range(len(header))]
    for row_index, row in enumerate(rows):
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
        if row_index == 0:
            print("  ".join("-" * width for width in widths))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect registered BEETLE experiments.")
    parser.add_argument("--show", action="store_true", help="Print the experiment registry.")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS))
    args = parser.parse_args()

    if args.show or args.experiment is None:
        print_registry()
    else:
        print(get_experiment(args.experiment))
