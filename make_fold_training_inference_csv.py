#!/usr/bin/env python3
"""
Create a CSV for whole-slide inference from an nnU-Net pathology splits.json.

The output columns match mine_hard_confusions.py:
    file_key, slide, wsi_path, mask_path

This parser is deliberately defensive because split entry structures can vary.
After running it, inspect the first rows before queuing the mining job.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Optional


def unwrap_path(value: Any) -> Optional[str]:
    if value is None:
        return None

    if isinstance(value, str):
        return value

    if isinstance(value, (list, tuple)):
        for item in value:
            path = unwrap_path(item)
            if path:
                return path
        return None

    if isinstance(value, dict):
        for key in ("path", "filepath", "file", "filename"):
            if key in value:
                path = unwrap_path(value[key])
                if path:
                    return path

        for item in value.values():
            path = unwrap_path(item)
            if path:
                return path

    return None


def first_named_path(entry: dict, keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        if key in entry:
            path = unwrap_path(entry[key])
            if path:
                return path

    lowered = {str(key).lower(): key for key in entry.keys()}
    for key in keys:
        actual = lowered.get(str(key).lower())
        if actual is not None:
            path = unwrap_path(entry[actual])
            if path:
                return path

    return None


def convert_entry(entry: Any, index: int) -> dict:
    if not isinstance(entry, dict):
        raise ValueError(
            f"Entry {index} is not a dictionary. Upload splits.json or its first "
            f"few entries so the parser can be aligned. Entry: {entry!r}"
        )

    wsi_path = first_named_path(
        entry,
        ("wsi", "image", "images", "wsi_path", "image_path"),
    )
    mask_path = first_named_path(
        entry,
        ("mask", "masks", "mask_path", "wsa", "annotation", "annotations"),
    )

    if not wsi_path or not mask_path:
        raise ValueError(
            f"Could not extract WSI/mask paths from split entry {index}: {entry!r}"
        )

    file_key = (
        entry.get("file_key")
        or entry.get("key")
        or entry.get("id")
        or Path(wsi_path).stem
    )

    return {
        "file_key": str(file_key),
        "slide": Path(wsi_path).stem,
        "wsi_path": str(wsi_path),
        "mask_path": str(mask_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-json", required=True, type=Path)
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument(
        "--subset",
        default="training",
        choices=("training", "validation"),
    )
    parser.add_argument("--out-csv", required=True, type=Path)
    args = parser.parse_args()

    with args.splits_json.open() as f:
        splits = json.load(f)

    fold_data = splits.get(str(args.fold), splits.get(args.fold))
    if fold_data is None:
        raise KeyError(f"Fold {args.fold} not found in {args.splits_json}")

    entries = fold_data[args.subset]
    rows = [convert_entry(entry, idx) for idx, entry in enumerate(entries)]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=("file_key", "slide", "wsi_path", "mask_path"),
        )
        writer.writeheader()
        writer.writerows(rows)

    unusual_masks = [
        row["mask_path"]
        for row in rows
        if Path(row["mask_path"]).suffix.lower() not in (".tif", ".tiff")
    ]

    print(f"Wrote {len(rows)} rows: {args.out_csv}")
    print("First rows:")
    for row in rows[:3]:
        print(row)

    if unusual_masks:
        print(
            "\nWARNING: Some mask paths are not .tif/.tiff files. "
            "mine_hard_confusions.py reads masks through TiffSlide. "
            "Do not queue mining until these paths are checked."
        )
        for path in unusual_masks[:10]:
            print("  ", path)


if __name__ == "__main__":
    main()
