#!/usr/bin/env python3
"""Generate fold-specific WSI validation CSV manifests from pathology splits.

The held-out WSI evaluation engine consumes CSV manifests with aligned WSI and
raster-mask paths. These files are generated artifacts: fold membership comes
from the pathology splits.json file, while raster masks are resolved from a
configured mask directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

DATASET_NAME = "Dataset301_BEETLE"
EXPECTED_FOLDS = [0, 1, 2, 3, 4]
FIELDNAMES = ["file_key", "slide", "wsi_path", "mask_path"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate fold<N>_validation_inference_inputs.csv files from the "
            "pathology splits.json validation entries."
        )
    )
    parser.add_argument(
        "--splits",
        type=Path,
        help=(
            "Pathology splits.json. Defaults to "
            "$nnUNet_preprocessed/Dataset301_BEETLE/splits.json when available, "
            "otherwise the repository splits.json."
        ),
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        help=(
            "Directory containing rasterized validation masks named after the "
            "WSI stem, for example patient3_wsi1.tif. Defaults to "
            "$BEETLE_VALIDATION_MASK_ROOT or $BEETLE_DATA_ROOT/annotations/masks."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory for generated CSV files. Defaults to "
            "$BEETLE_VALIDATION_CSV_DIR or outputs/validation_inputs."
        ),
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=EXPECTED_FOLDS,
        action="append",
        help="Generate only one fold. Repeat the option for multiple folds.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate inputs and print the planned outputs without writing files.",
    )
    parser.add_argument(
        "--skip-file-check",
        action="store_true",
        help="Generate manifests without requiring every WSI and mask to exist.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing CSV when its contents differ.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def default_splits_path() -> Path:
    preprocessed = os.environ.get("nnUNet_preprocessed")
    if preprocessed:
        candidate = Path(preprocessed) / DATASET_NAME / "splits.json"
        if candidate.is_file():
            return candidate
    return repo_root() / "splits.json"


def resolve_splits_path(args: argparse.Namespace) -> Path:
    return (args.splits if args.splits is not None else default_splits_path()).resolve()


def resolve_mask_root(args: argparse.Namespace) -> Path:
    if args.mask_root is not None:
        return args.mask_root.resolve()

    configured = os.environ.get("BEETLE_VALIDATION_MASK_ROOT")
    if configured:
        return Path(configured).resolve()

    data_root = os.environ.get("BEETLE_DATA_ROOT")
    if data_root:
        return (Path(data_root) / "annotations" / "masks").resolve()

    raise SystemExit(
        "Provide --mask-root, set BEETLE_VALIDATION_MASK_ROOT, or set "
        "BEETLE_DATA_ROOT."
    )


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    configured = os.environ.get("BEETLE_VALIDATION_CSV_DIR")
    if configured:
        return Path(configured).resolve()
    return (repo_root() / "outputs" / "validation_inputs").resolve()


def validate_splits(splits: Any) -> dict[str, Any]:
    if not isinstance(splits, dict):
        raise ValueError("Expected splits.json to contain a dictionary keyed by fold.")
    expected = [str(fold) for fold in EXPECTED_FOLDS]
    if sorted(splits, key=int) != expected:
        raise ValueError(f"Expected fold keys {expected}, found {sorted(splits, key=int)}.")

    for fold in expected:
        entry = splits[fold]
        if not isinstance(entry, dict):
            raise ValueError(f"Fold {fold} must contain a dictionary.")
        validation = entry.get("validation")
        if not isinstance(validation, list) or not validation:
            raise ValueError(f"Fold {fold}: validation must be a non-empty list.")
    return splits


def record_to_row(record: Any, mask_root: Path) -> dict[str, str]:
    if not isinstance(record, dict):
        raise ValueError(f"Malformed validation record: {record!r}")
    try:
        wsi_path = Path(record["wsi"]["path"]).resolve()
    except (KeyError, TypeError) as error:
        raise ValueError(f"Malformed validation record: {record!r}") from error

    slide = wsi_path.stem
    mask_path = (mask_root / f"{slide}.tif").resolve()
    return {
        "file_key": slide,
        "slide": slide,
        "wsi_path": str(wsi_path),
        "mask_path": str(mask_path),
    }


def rows_for_fold(splits: dict[str, Any], fold: int, mask_root: Path) -> list[dict[str, str]]:
    rows = [record_to_row(record, mask_root) for record in splits[str(fold)]["validation"]]
    rows.sort(key=lambda row: row["slide"])
    return rows


def validate_files(rows: list[dict[str, str]], fold: int) -> None:
    missing_wsis = sorted(Path(row["wsi_path"]) for row in rows if not Path(row["wsi_path"]).is_file())
    missing_masks = sorted(Path(row["mask_path"]) for row in rows if not Path(row["mask_path"]).is_file())

    if not missing_wsis and not missing_masks:
        return

    lines = [f"Fold {fold}: missing required validation files."]
    if missing_wsis:
        lines.append(f"Missing WSIs: {len(missing_wsis)}")
        lines.extend(f"  - {path}" for path in missing_wsis[:10])
        if len(missing_wsis) > 10:
            lines.append(f"  ... and {len(missing_wsis) - 10} more")
    if missing_masks:
        lines.append(f"Missing masks: {len(missing_masks)}")
        lines.extend(f"  - {path}" for path in missing_masks[:10])
        if len(missing_masks) > 10:
            lines.append(f"  ... and {len(missing_masks) - 10} more")
    raise FileNotFoundError("\n".join(lines))


def render_csv(rows: list[dict[str, str]]) -> str:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_text_atomic(path: Path, contents: str, *, force: bool) -> None:
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
    print(f"Wrote validation manifest: {path}")


def main() -> None:
    args = parse_args()
    splits_path = resolve_splits_path(args)
    mask_root = resolve_mask_root(args)
    output_dir = resolve_output_dir(args)
    folds = args.fold if args.fold is not None else EXPECTED_FOLDS

    if not splits_path.is_file():
        raise SystemExit(f"Split file not found: {splits_path}")

    splits = validate_splits(json.loads(splits_path.read_text()))

    print("Generating held-out WSI validation manifests")
    print(f"  splits: {splits_path}")
    print(f"  mask root: {mask_root}")
    print(f"  output dir: {output_dir}")

    for fold in folds:
        rows = rows_for_fold(splits, fold, mask_root)
        if not args.skip_file_check:
            validate_files(rows, fold)

        output = output_dir / f"fold{fold}_validation_inference_inputs.csv"
        print(f"  fold {fold}: validation slides={len(rows)} -> {output}")

        if not args.check_only:
            write_text_atomic(output, render_csv(rows), force=args.force)

    if args.check_only:
        print("Check-only mode: no files written.")


if __name__ == "__main__":
    main()
