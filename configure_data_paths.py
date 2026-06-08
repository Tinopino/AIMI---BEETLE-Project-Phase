#!/usr/bin/env python3
"""Rewrite the committed BEETLE WSI split file for a local dataset layout.

The repository stores the exact five-fold assignments used in the project. The
reference split file necessarily contains paths from the original cluster. This
helper preserves fold membership while rewriting each WSI and annotation path
for another machine.
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

DATASET_NAME = "Dataset301_BEETLE"
EXPECTED_FOLDS = ["0", "1", "2", "3", "4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite the committed BEETLE pathology splits.json for a local "
            "dataset root while preserving the original five-fold assignments."
        )
    )
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help=(
            "BEETLE data root containing images/development/wsis and "
            "annotations/jsons."
        ),
    )
    parser.add_argument(
        "--nnunet-preprocessed",
        type=Path,
        default=(
            Path(os.environ["nnUNet_preprocessed"])
            if "nnUNet_preprocessed" in os.environ
            else None
        ),
        help=(
            "nnUNet_preprocessed root. Defaults to the nnUNet_preprocessed "
            "environment variable."
        ),
    )
    parser.add_argument(
        "--reference-splits",
        type=Path,
        default=Path(__file__).resolve().with_name("splits.json"),
        help="Reference split file committed with this repository.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Explicit output path. By default, writes "
            "<nnunet-preprocessed>/Dataset301_BEETLE/splits.json."
        ),
    )
    parser.add_argument(
        "--skip-file-check",
        action="store_true",
        help="Write remapped paths without requiring every referenced file to exist.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate and print the remapped layout without writing a file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file when its contents differ.",
    )
    return parser.parse_args()


def output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output
    if args.nnunet_preprocessed is None:
        raise SystemExit(
            "Provide --nnunet-preprocessed or set the nnUNet_preprocessed "
            "environment variable."
        )
    return args.nnunet_preprocessed / DATASET_NAME / "splits.json"


def validate_reference_splits(splits: Any) -> dict[str, Any]:
    if not isinstance(splits, dict):
        raise ValueError("Expected splits.json to contain a dictionary keyed by fold.")
    if sorted(splits, key=int) != EXPECTED_FOLDS:
        raise ValueError(
            f"Expected fold keys {EXPECTED_FOLDS}, found {sorted(splits, key=int)}."
        )

    for fold in EXPECTED_FOLDS:
        entry = splits[fold]
        if not isinstance(entry, dict):
            raise ValueError(f"Fold {fold} must contain a dictionary.")
        for subset in ("training", "validation"):
            records = entry.get(subset)
            if not isinstance(records, list) or not records:
                raise ValueError(f"Fold {fold}: {subset} must be a non-empty list.")
    return splits


def remap_record(record: Any, data_root: Path) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("Each split record must be a dictionary.")

    item = deepcopy(record)
    try:
        original_wsi = Path(item["wsi"]["path"])
        original_wsa = Path(item["wsa"]["path"])
    except (KeyError, TypeError) as error:
        raise ValueError(f"Malformed split record: {record!r}") from error

    item["wsi"]["path"] = str(
        data_root / "images" / "development" / "wsis" / original_wsi.name
    )
    item["wsa"]["path"] = str(
        data_root / "annotations" / "jsons" / original_wsa.name
    )
    return item


def remap_splits(splits: dict[str, Any], data_root: Path) -> dict[str, Any]:
    remapped: dict[str, Any] = {}
    for fold in EXPECTED_FOLDS:
        remapped[fold] = {}
        for subset in ("training", "validation"):
            remapped[fold][subset] = [
                remap_record(record, data_root)
                for record in splits[fold][subset]
            ]
    return remapped


def referenced_paths(splits: dict[str, Any]) -> set[Path]:
    paths: set[Path] = set()
    for fold in EXPECTED_FOLDS:
        for subset in ("training", "validation"):
            for record in splits[fold][subset]:
                paths.add(Path(record["wsi"]["path"]))
                paths.add(Path(record["wsa"]["path"]))
    return paths


def validate_referenced_files(splits: dict[str, Any]) -> None:
    missing = sorted(path for path in referenced_paths(splits) if not path.is_file())
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:20])
        extra = "" if len(missing) <= 20 else f"\n  ... and {len(missing) - 20} more"
        raise FileNotFoundError(
            f"Missing {len(missing)} referenced WSI or annotation files:\n"
            f"{preview}{extra}"
        )


def summary(splits: dict[str, Any]) -> None:
    print("Configured pathology split summary")
    for fold in EXPECTED_FOLDS:
        print(
            f"  fold {fold}: "
            f"train={len(splits[fold]['training'])}, "
            f"val={len(splits[fold]['validation'])}"
        )
    print(f"  unique referenced files: {len(referenced_paths(splits))}")


def write_json_atomic(path: Path, contents: str, *, force: bool) -> None:
    if path.exists():
        existing = path.read_text()
        if existing == contents:
            print(f"No changes required: {path}")
            return
        if not force:
            raise FileExistsError(
                f"Refusing to overwrite existing file: {path}\n"
                "Re-run with --force after reviewing the destination."
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(contents)
    temporary.replace(path)
    print(f"Wrote remapped split file: {path}")


def main() -> None:
    args = parse_args()
    reference_path = args.reference_splits.resolve()
    destination = output_path(args).resolve()
    data_root = args.data_root.resolve()

    if not reference_path.is_file():
        raise SystemExit(f"Reference split file not found: {reference_path}")

    splits = validate_reference_splits(json.loads(reference_path.read_text()))
    remapped = remap_splits(splits, data_root)

    if not args.skip_file_check:
        validate_referenced_files(remapped)

    summary(remapped)
    print(f"  reference: {reference_path}")
    print(f"  destination: {destination}")

    if args.check_only:
        print("Check-only mode: no file written.")
        return

    rendered = json.dumps(remapped, indent=4) + "\n"
    write_json_atomic(destination, rendered, force=args.force)


if __name__ == "__main__":
    main()
