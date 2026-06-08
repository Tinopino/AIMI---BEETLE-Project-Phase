#!/usr/bin/env python3
"""Single user-facing validation and external-inference entry point.

Subcommands:
- wsi: held-out WSI evaluation with optional qualitative panels;
- aggregate: aggregate completed fold-level WSI evaluations;
- external: five-fold external ROI inference, submission validation, and ZIP;
- check-submission: validate an existing prediction folder and create a ZIP.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import torch
from PIL import Image

from experiments import (
    external_output_folder,
    external_roi_folder,
    external_zip_path,
    get_experiment,
    model_dir,
    repo_root,
    validation_csv,
    visual_dir,
)

EXPECTED_LABELS = {1, 2, 3, 4}
EXPECTED_EXTERNAL_ROIS = 170


def require_environment() -> None:
    required = ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + "."
        )


def norm_01(x_batch: np.ndarray) -> np.ndarray:
    x_batch = x_batch.astype(np.float32) / 255.0
    return x_batch.transpose(3, 0, 1, 2)


def validate_submission(
    roi_folder: Path,
    output_folder: Path,
    zip_path: Path | None,
    *,
    expected_count: int = EXPECTED_EXTERNAL_ROIS,
) -> None:
    roi_paths = sorted(roi_folder.glob("*.png"))
    prediction_paths = sorted(output_folder.glob("*.png"))

    roi_names = {path.name for path in roi_paths}
    prediction_names = {path.name for path in prediction_paths}

    print(f"Input ROI count:      {len(roi_paths)}")
    print(f"Prediction PNG count: {len(prediction_paths)}")

    if len(roi_paths) != expected_count:
        raise RuntimeError(
            f"Expected exactly {expected_count} ROI inputs, found {len(roi_paths)}."
        )
    if len(prediction_paths) != expected_count:
        raise RuntimeError(
            f"Expected exactly {expected_count} predictions, found {len(prediction_paths)}."
        )

    missing = sorted(roi_names - prediction_names)
    unexpected = sorted(prediction_names - roi_names)
    if missing:
        raise RuntimeError(f"Missing predictions: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected predictions: {unexpected}")

    observed_labels: set[int] = set()

    for roi_path in roi_paths:
        prediction_path = output_folder / roi_path.name

        with Image.open(roi_path) as roi_image:
            roi_size = roi_image.size

        with Image.open(prediction_path) as prediction_image:
            prediction_size = prediction_image.size
            prediction_array = np.asarray(prediction_image)

        if prediction_size != roi_size:
            raise RuntimeError(
                f"Dimension mismatch for {roi_path.name}: "
                f"ROI={roi_size}, prediction={prediction_size}"
            )
        if prediction_array.ndim != 2:
            raise RuntimeError(
                f"{roi_path.name} is not single-channel: shape={prediction_array.shape}"
            )

        labels = set(int(value) for value in np.unique(prediction_array))
        invalid = labels - EXPECTED_LABELS
        if invalid:
            raise RuntimeError(
                f"{roi_path.name} contains invalid labels: {sorted(invalid)}"
            )
        observed_labels.update(labels)

    print("Observed labels:      ", sorted(observed_labels))

    if zip_path is not None:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            for prediction_path in prediction_paths:
                handle.write(prediction_path, arcname=prediction_path.name)
        print("Validated ZIP created:", zip_path)


def run_external_inference(
    experiment_name: str,
    folds: tuple[int, ...],
    roi_folder: Path,
    output_folder: Path,
    zip_path: Path,
    *,
    clean_output: bool,
) -> None:
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    exp = get_experiment(experiment_name)
    base_path = model_dir(exp)

    for metadata_name in ("plans.json", "dataset.json"):
        metadata_path = base_path / metadata_name
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing model metadata: {metadata_path}")

    for fold in folds:
        checkpoint = base_path / f"fold_{fold}" / exp.checkpoint_name
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    if clean_output and output_folder.exists():
        shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    predictor = nnUNetPredictor()
    predictor.initialize_from_trained_model_folder(
        base_path,
        use_folds=folds,
        checkpoint_name=exp.checkpoint_name,
    )

    roi_paths = sorted(roi_folder.glob("*.png"))
    print(f"Running ensemble inference for {len(roi_paths)} ROIs.")

    for index, roi_path in enumerate(roi_paths, start=1):
        with Image.open(roi_path) as image:
            patch = np.expand_dims(np.asarray(image), axis=0)
        patch = norm_01(patch)

        with torch.no_grad():
            logits_list = predictor.get_logits_list_from_preprocessed_data(
                torch.tensor(patch, dtype=torch.float32)
            )
            probabilities = [
                predictor.label_manager.apply_inference_nonlin(logits).cpu().numpy()
                for logits in logits_list
            ]

        prediction = np.squeeze(np.argmax(np.mean(probabilities, axis=0), axis=0))
        Image.fromarray(prediction.astype(np.uint8)).save(output_folder / roi_path.name)

        print(f"[{index:03d}/{len(roi_paths):03d}] {roi_path.name}")

    validate_submission(roi_folder, output_folder, zip_path)


def ensure_validation_csv(fold: int) -> Path:
    """Generate the fold manifest from splits.json when it is not present yet."""
    csv_path = validation_csv(fold)
    if csv_path.is_file():
        return csv_path

    generator = repo_root() / "configure_validation_inputs.py"
    if not generator.is_file():
        raise FileNotFoundError(
            f"Missing validation CSV: {csv_path}. "
            f"Generator not found: {generator}"
        )

    print(f"Validation CSV is missing; generating fold {fold}: {csv_path}")
    subprocess.run(
        [
            sys.executable,
            str(generator),
            "--fold",
            str(fold),
        ],
        check=True,
    )

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Validation manifest generator completed, but CSV is still missing: {csv_path}"
        )
    return csv_path


def run_wsi_evaluation(experiment_name: str, fold: int, save_visuals: bool) -> None:
    exp = get_experiment(experiment_name)
    base_path = model_dir(exp)
    checkpoint = base_path / f"fold_{fold}" / exp.checkpoint_name

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    csv_path = ensure_validation_csv(fold)

    updates = {
        "MODEL_BASE_PATH": str(base_path),
        "CSV_PATH": str(csv_path),
        "EVAL_FOLD": str(fold),
        "CHECKPOINT_NAME": exp.checkpoint_name,
        "CHECKPOINT_TAG": exp.checkpoint_tag,
        "SAVE_VISUALS": "1" if save_visuals else "0",
        "USE_MIRRORING": "1" if exp.use_mirroring else "0",
        "MODEL_PATCH_SIZE": str(exp.patch_size),
        "VIS_OUT_DIR": str(visual_dir(exp, fold)),
    }
    os.environ.update(updates)

    print(json.dumps(updates, indent=2))
    runpy.run_path(
        str(repo_root() / "pipeline" / "wsi_validation_engine.py"),
        run_name="__main__",
    )


def run_aggregation(stage: str) -> None:
    subprocess.run(
        [
            sys.executable,
            str(repo_root() / "pipeline" / "aggregate_cv_results.py"),
            stage,
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    wsi = subparsers.add_parser("wsi", help="Run held-out WSI evaluation.")
    wsi.add_argument("--experiment", required=True)
    wsi.add_argument("--fold", required=True, type=int, choices=range(5))
    wsi.add_argument("--save-visuals", action="store_true")

    aggregate = subparsers.add_parser("aggregate", help="Aggregate completed WSI folds.")
    aggregate.add_argument("--stage", choices=("base", "context", "both"), default="both")

    external = subparsers.add_parser(
        "external",
        help="Run external five-fold ROI inference and create a validated ZIP.",
    )
    external.add_argument("--experiment", required=True)
    external.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    external.add_argument("--roi-folder", type=Path)
    external.add_argument("--output-folder", type=Path)
    external.add_argument("--zip-path", type=Path)
    external.add_argument("--keep-existing-output", action="store_true")

    check = subparsers.add_parser(
        "check-submission",
        help="Validate existing ROI predictions and create a ZIP.",
    )
    check.add_argument("--roi-folder", type=Path, required=True)
    check.add_argument("--output-folder", type=Path, required=True)
    check.add_argument("--zip-path", type=Path, required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Only model-running commands need nnU-Net environment variables.
    # This keeps --help and check-submission usable on any machine.
    if args.command in {"wsi", "external", "aggregate"}:
        require_environment()

    if args.command == "wsi":
        run_wsi_evaluation(args.experiment, args.fold, args.save_visuals)
        return

    if args.command == "aggregate":
        run_aggregation(args.stage)
        return

    if args.command == "external":
        exp = get_experiment(args.experiment)
        run_external_inference(
            args.experiment,
            tuple(args.folds),
            args.roi_folder or external_roi_folder(),
            args.output_folder or external_output_folder(exp),
            args.zip_path or external_zip_path(exp),
            clean_output=not args.keep_existing_output,
        )
        return

    if args.command == "check-submission":
        validate_submission(args.roi_folder, args.output_folder, args.zip_path)
        return

    raise RuntimeError(f"Unexpected command: {args.command}")


if __name__ == "__main__":
    main()
