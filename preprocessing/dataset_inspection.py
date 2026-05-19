#!/usr/bin/env python3
"""
Audit BEETLE image/mask files before nnU-Net preprocessing.

This version is designed for whole-slide-sized files:
- image dimensions are read from metadata
- TIFF masks are counted block by block through tifffile/zarr
- PNG/JPG masks are cropped in blocks instead of converted all at once
- preview generation is optional and uses downsampled reads

Outputs:
- outputs/dataset_audit.csv
- outputs/dataset_audit_summary.txt
- optional preview images with image/mask/overlay
"""

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from tqdm import tqdm


IMAGE_EXTS = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]

IMAGE_DIR = Path(r"D:\AIMI---BEETLE-Project-Phase\data\images\images\development\wsis")
MASK_DIR = Path(r"D:\AIMI---BEETLE-Project-Phase\data\annotations\masks")


LABEL_NAMES = {
    0: "zero_background_or_unannotated",
    1: "other",
    2: "non_invasive_epithelium",
    3: "invasive_epithelium",
    4: "necrosis",
}


@dataclass(frozen=True)
class ArrayLayout:
    y_axis: int
    x_axis: int
    channel_axis: int | None = None


def find_files(folder: Path):
    files = []
    for ext in IMAGE_EXTS:
        files.extend(folder.glob(f"*{ext}"))
    return sorted(files)


def is_tiff(path: Path):
    return path.suffix.lower() in [".tif", ".tiff"]


def get_shape_and_axes(path: Path):
    """Read array shape from metadata without loading the full image."""
    if is_tiff(path):
        with tifffile.TiffFile(str(path)) as tif:
            if not tif.series:
                raise RuntimeError("TIFF has no series")
            series = tif.series[0]
            return tuple(series.shape), getattr(series, "axes", "")

    with Image.open(path) as img:
        shape = (img.height, img.width, len(img.getbands()))
        if shape[-1] == 1:
            shape = shape[:2]
        return shape, "YXS" if len(shape) == 3 else "YX"


def infer_layout(shape, axes=""):
    axes = (axes or "").upper()
    if axes and len(axes) == len(shape) and "Y" in axes and "X" in axes:
        y_axis = axes.index("Y")
        x_axis = axes.index("X")
        channel_axis = None
        for candidate in ("S", "C"):
            if candidate in axes and shape[axes.index(candidate)] <= 4:
                channel_axis = axes.index(candidate)
                break
        return ArrayLayout(y_axis=y_axis, x_axis=x_axis, channel_axis=channel_axis)

    if len(shape) == 2:
        return ArrayLayout(y_axis=0, x_axis=1)

    if len(shape) == 3:
        if shape[-1] <= 4:
            return ArrayLayout(y_axis=0, x_axis=1, channel_axis=2)
        if shape[0] <= 4:
            return ArrayLayout(y_axis=1, x_axis=2, channel_axis=0)
        return ArrayLayout(y_axis=1, x_axis=2)

    squeezed = [i for i, size in enumerate(shape) if size > 1]
    if len(squeezed) >= 2:
        return ArrayLayout(y_axis=squeezed[-2], x_axis=squeezed[-1])

    raise RuntimeError(f"Unsupported image shape: {shape}")


def get_hw_channels(shape, axes=""):
    layout = infer_layout(shape, axes)
    h = int(shape[layout.y_axis])
    w = int(shape[layout.x_axis])
    channels = int(shape[layout.channel_axis]) if layout.channel_axis is not None else 1
    return h, w, channels


def guess_patient_id(stem: str):
    """
    Tries to extract patient id from names like:
    patient320_wsi1_roi1
    patient_320_xxx
    """
    m = re.search(r"(patient[_-]?\d+)", stem, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()

    stem2 = re.sub(r"(_roi\d+.*)$", "", stem, flags=re.IGNORECASE)
    stem2 = re.sub(r"(_x\d+_y\d+.*)$", "", stem2, flags=re.IGNORECASE)
    stem2 = re.sub(r"(_\d+_\d+)$", "", stem2)
    return stem2


def spatial_slice(shape, layout, y0, y1, x0, x1, channel=None, y_step=None, x_step=None):
    selection = []
    for axis, size in enumerate(shape):
        if axis == layout.y_axis:
            selection.append(slice(y0, y1, y_step))
        elif axis == layout.x_axis:
            selection.append(slice(x0, x1, x_step))
        elif axis == layout.channel_axis:
            selection.append(channel if channel is not None else slice(None))
        else:
            selection.append(0 if size > 1 else slice(None))
    return tuple(selection)


def open_tiff_series_array(path: Path, series, series_index=0, level_index=None):
    """
    Open a TIFF series as a lazy array.

    zarr handles tiled/compressed TIFFs well. If zarr is unavailable, memmap is
    used for simple memory-mappable TIFFs.
    """
    try:
        import zarr

        store = series.aszarr()
        root = zarr.open(store, mode="r")
        if hasattr(root, "shape"):
            return root, store

        key = str(level_index or 0)
        if key not in root:
            key = sorted(root.keys(), key=lambda value: int(value) if value.isdigit() else value)[0]
        return root[key], store
    except ModuleNotFoundError as zarr_error:
        try:
            arr = tifffile.memmap(
                str(path),
                series=series_index,
                level=level_index,
                mode="r",
            )
            return arr, None
        except Exception as memmap_error:
            raise RuntimeError(
                "This TIFF cannot be chunk-read without zarr. Install the "
                "preprocessing requirements with: pip install -r "
                "preprocessing/requirements-preprocessing.txt"
            ) from memmap_error
    except Exception as zarr_error:
        try:
            arr = tifffile.memmap(
                str(path),
                series=series_index,
                level=level_index,
                mode="r",
            )
            return arr, None
        except Exception as memmap_error:
            raise RuntimeError(
                f"Could not open TIFF lazily through zarr or memmap: {zarr_error}"
            ) from memmap_error


def close_tiff_store(store):
    close = getattr(store, "close", None)
    if close is not None:
        close()


def add_label_counts(values, counts, unexpected):
    values = np.asarray(values)
    if values.size == 0:
        return

    if np.issubdtype(values.dtype, np.integer):
        min_value = int(values.min())
        max_value = int(values.max())
        if min_value >= 0 and max_value <= 65535:
            bincounts = np.bincount(values.ravel(), minlength=max_value + 1)
            for label in counts:
                if label < len(bincounts):
                    counts[label] += int(bincounts[label])
            for label in np.flatnonzero(bincounts):
                label = int(label)
                if label not in counts:
                    unexpected.add(label)
            return

    unique, unique_counts = np.unique(values, return_counts=True)
    for value, n in zip(unique, unique_counts):
        label = int(value)
        if label in counts:
            counts[label] += int(n)
        else:
            unexpected.add(label)


def first_channel_from_chunk(chunk, layout):
    chunk = np.asarray(chunk)
    if chunk.ndim <= 2:
        return chunk

    channel_axis = layout.channel_axis
    if channel_axis is None:
        channel_axis = chunk.ndim - 1 if chunk.shape[-1] <= 4 else 0

    return np.take(chunk, 0, axis=channel_axis)


def count_mask_labels_tiff_chunked(mask_path: Path, chunk_size: int, check_rgb_channels: bool):
    counts = {label: 0 for label in LABEL_NAMES}
    unexpected = set()
    rgb_channels_equal = True

    with tifffile.TiffFile(str(mask_path)) as tif:
        if not tif.series:
            raise RuntimeError("TIFF has no series")

        series = tif.series[0]
        mask_shape = tuple(series.shape)
        axes = getattr(series, "axes", "")
        layout = infer_layout(mask_shape, axes)
        h, w, channels = get_hw_channels(mask_shape, axes)
        mask_was_rgb = channels > 1

        arr, store = open_tiff_series_array(mask_path, series, series_index=0)
        try:
            for y in range(0, h, chunk_size):
                y2 = min(y + chunk_size, h)
                for x in range(0, w, chunk_size):
                    x2 = min(x + chunk_size, w)

                    selection = spatial_slice(
                        mask_shape,
                        layout,
                        y,
                        y2,
                        x,
                        x2,
                        channel=0 if mask_was_rgb else None,
                    )
                    chunk = np.asarray(arr[selection])
                    add_label_counts(chunk, counts, unexpected)

                    if check_rgb_channels and channels >= 3 and rgb_channels_equal:
                        ch1 = np.asarray(
                            arr[spatial_slice(mask_shape, layout, y, y2, x, x2, channel=1)]
                        )
                        ch2 = np.asarray(
                            arr[spatial_slice(mask_shape, layout, y, y2, x, x2, channel=2)]
                        )
                        rgb_channels_equal = bool(
                            np.array_equal(chunk, ch1) and np.array_equal(chunk, ch2)
                        )
        finally:
            close_tiff_store(store)

    return {
        "shape": mask_shape,
        "axes": axes,
        "height": h,
        "width": w,
        "channels": channels,
        "counts": counts,
        "unexpected": unexpected,
        "mask_was_rgb": mask_was_rgb,
        "rgb_channels_equal": rgb_channels_equal,
    }


def count_mask_labels_pil_chunked(mask_path: Path, chunk_size: int, check_rgb_channels: bool):
    counts = {label: 0 for label in LABEL_NAMES}
    unexpected = set()
    rgb_channels_equal = True

    with Image.open(mask_path) as img:
        bands = img.getbands()
        h, w = img.height, img.width
        channels = len(bands)
        mask_was_rgb = channels > 1
        shape = (h, w, channels) if mask_was_rgb else (h, w)

        for y in range(0, h, chunk_size):
            y2 = min(y + chunk_size, h)
            for x in range(0, w, chunk_size):
                x2 = min(x + chunk_size, w)
                crop = np.asarray(img.crop((x, y, x2, y2)))

                if crop.ndim == 3:
                    first = crop[:, :, 0]
                    if check_rgb_channels and crop.shape[2] >= 3 and rgb_channels_equal:
                        rgb_channels_equal = bool(
                            np.array_equal(first, crop[:, :, 1])
                            and np.array_equal(first, crop[:, :, 2])
                        )
                    crop = first

                add_label_counts(crop, counts, unexpected)

    return {
        "shape": shape,
        "axes": "YXS" if mask_was_rgb else "YX",
        "height": h,
        "width": w,
        "channels": channels,
        "counts": counts,
        "unexpected": unexpected,
        "mask_was_rgb": mask_was_rgb,
        "rgb_channels_equal": rgb_channels_equal,
    }


def audit_mask(mask_path: Path, chunk_size: int, check_rgb_channels: bool):
    if is_tiff(mask_path):
        return count_mask_labels_tiff_chunked(mask_path, chunk_size, check_rgb_channels)
    return count_mask_labels_pil_chunked(mask_path, chunk_size, check_rgb_channels)


def read_tiff_preview(path: Path, max_size: int, is_mask: bool):
    with tifffile.TiffFile(str(path)) as tif:
        if not tif.series:
            raise RuntimeError("TIFF has no series")

        levels = getattr(tif.series[0], "levels", None) or [tif.series[0]]
        best = levels[0]
        best_shape = tuple(best.shape)
        best_axes = getattr(best, "axes", "")
        best_level_index = 0

        for level_index, level in enumerate(levels):
            shape = tuple(level.shape)
            axes = getattr(level, "axes", "")
            h, w, _ = get_hw_channels(shape, axes)
            best = level
            best_shape = shape
            best_axes = axes
            best_level_index = level_index
            if max(h, w) <= max_size * 2:
                break

        layout = infer_layout(best_shape, best_axes)
        h, w, channels = get_hw_channels(best_shape, best_axes)
        step = max(1, math.ceil(max(h, w) / max_size))

        arr, store = open_tiff_series_array(
            path,
            best,
            series_index=0,
            level_index=best_level_index,
        )
        try:
            channel = 0 if is_mask and channels > 1 else None
            selection = spatial_slice(
                best_shape,
                layout,
                0,
                h,
                0,
                w,
                channel=channel,
                y_step=step,
                x_step=step,
            )
            preview = np.asarray(arr[selection])
        finally:
            close_tiff_store(store)

    if is_mask:
        preview = first_channel_from_chunk(preview, infer_layout(preview.shape))
    elif preview.ndim == 2:
        preview = np.stack([preview] * 3, axis=-1)
    elif preview.ndim == 3 and preview.shape[0] <= 4 and preview.shape[-1] > 4:
        preview = np.moveaxis(preview, 0, -1)

    return preview


def read_pil_preview(path: Path, max_size: int, is_mask: bool):
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    with Image.open(path) as img:
        img.thumbnail((max_size, max_size), resample=resample)
        arr = np.asarray(img)

    if is_mask and arr.ndim == 3:
        return arr[:, :, 0]
    if not is_mask and arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1)
    return arr


def read_preview(path: Path, max_size: int, is_mask: bool):
    if is_tiff(path):
        return read_tiff_preview(path, max_size, is_mask)
    return read_pil_preview(path, max_size, is_mask)


def make_preview(image_path: Path, mask_path: Path, out_path: Path, max_size: int):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    image_rgb = read_preview(image_path, max_size, is_mask=False)
    mask = read_preview(mask_path, max_size, is_mask=True)

    if image_rgb.ndim == 3 and image_rgb.shape[2] > 3:
        image_rgb = image_rgb[:, :, :3]
    if image_rgb.ndim != 3 or mask.ndim != 2:
        raise RuntimeError("Could not create 2D mask/RGB image preview")

    if image_rgb.shape[:2] != mask.shape[:2]:
        mask_img = Image.fromarray(mask.astype(np.uint8))
        mask = np.asarray(
            mask_img.resize((image_rgb.shape[1], image_rgb.shape[0]), Image.Resampling.NEAREST)
        )

    cmap = ListedColormap(["black", "cyan", "yellow", "magenta", "blue"])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    axes[1].set_title("Mask labels")
    axes[1].axis("off")

    axes[2].imshow(image_rgb)
    axes[2].imshow(mask, cmap=cmap, vmin=0, vmax=4, alpha=0.35, interpolation="nearest")
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def audit_one_case(image_path: Path, mask_path: Path | None, args):
    stem = image_path.stem
    row = {
        "case_id": stem,
        "patient_id_guess": guess_patient_id(stem),
        "image_path": str(image_path),
        "mask_path": str(mask_path) if mask_path else "",
        "has_mask": mask_path is not None,
        "error": "",
    }

    image_shape, image_axes = get_shape_and_axes(image_path)
    image_h, image_w, image_channels = get_hw_channels(image_shape, image_axes)

    row["image_shape"] = str(image_shape)
    row["image_axes"] = image_axes
    row["image_height"] = image_h
    row["image_width"] = image_w
    row["image_channels"] = image_channels

    if mask_path is None:
        row["error"] = "missing_mask"
        return row

    mask_info = audit_mask(
        mask_path,
        chunk_size=args.chunk_size,
        check_rgb_channels=not args.skip_rgb_channel_check,
    )

    counts = mask_info["counts"]
    unexpected = mask_info["unexpected"]
    total = int(mask_info["height"] * mask_info["width"])
    present_expected = [label for label, n in counts.items() if n > 0]
    unique_labels = sorted(present_expected + list(unexpected))

    row["mask_shape_original"] = str(mask_info["shape"])
    row["mask_axes"] = mask_info["axes"]
    row["mask_shape_2d"] = str((mask_info["height"], mask_info["width"]))
    row["mask_was_rgb"] = mask_info["mask_was_rgb"]
    row["mask_rgb_channels_equal"] = mask_info["rgb_channels_equal"]
    row["shape_match"] = image_h == mask_info["height"] and image_w == mask_info["width"]
    row["unique_labels"] = ",".join(map(str, unique_labels))
    row["n_unique_labels"] = len(unique_labels)
    row["total_pixels"] = total

    for label, name in LABEL_NAMES.items():
        n = int(counts[label])
        row[f"n_label_{label}_{name}"] = n
        row[f"frac_label_{label}_{name}"] = n / total if total else 0.0

    row["annotated_frac_mask_gt_0"] = (total - counts[0]) / total if total else 0.0
    row["contains_other"] = counts[1] > 0
    row["contains_non_invasive"] = counts[2] > 0
    row["contains_invasive"] = counts[3] > 0
    row["contains_necrosis"] = counts[4] > 0
    row["unexpected_labels"] = ",".join(map(str, sorted(unexpected)))

    flags = []
    if not row["shape_match"]:
        flags.append("SHAPE_MISMATCH")
    if unexpected:
        flags.append("UNEXPECTED_LABELS")
    if row["annotated_frac_mask_gt_0"] < 0.01:
        flags.append("MOSTLY_ZERO_MASK")
    if row["mask_was_rgb"] and not row["mask_rgb_channels_equal"]:
        flags.append("RGB_MASK_CHANNELS_DIFFER")
    if row["image_channels"] < 3:
        flags.append("IMAGE_NOT_RGB")
    row["flags"] = ";".join(flags)
    row["preview_path"] = ""

    return row


def write_summary(df: pd.DataFrame, summary_path: Path, image_dir: Path, mask_dir: Path):
    with open(summary_path, "w") as f:
        f.write("BEETLE dataset audit summary\n")
        f.write("===========================\n\n")
        f.write(f"Image dir: {image_dir}\n")
        f.write(f"Mask dir: {mask_dir}\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Has mask: {int(df['has_mask'].sum())} / {len(df)}\n\n")

        if "error" in df.columns:
            errors = df["error"].fillna("")
            errors = errors[errors != ""]
            if not errors.empty:
                f.write("Errors:\n")
                f.write(str(errors.value_counts()))
                f.write("\n\n")

        if "flags" in df.columns:
            f.write("Flags:\n")
            all_flags = df["flags"].fillna("").str.split(";").explode()
            all_flags = all_flags[all_flags != ""]
            f.write(str(all_flags.value_counts()))
            f.write("\n\n")

        count_cols = [c for c in df.columns if c.startswith("n_label_")]
        if count_cols:
            f.write("Total pixel counts per label:\n")
            for c in count_cols:
                f.write(f"{c}: {int(df[c].fillna(0).sum())}\n")
            f.write("\n")

        presence_cols = [
            "contains_other",
            "contains_non_invasive",
            "contains_invasive",
            "contains_necrosis",
        ]
        existing_presence = [c for c in presence_cols if c in df.columns]
        if existing_presence:
            f.write("Number of files containing each class:\n")
            for c in existing_presence:
                f.write(f"{c}: {int(df[c].fillna(False).sum())}\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--mask-dir", type=Path, default=MASK_DIR)
    parser.add_argument("--out", type=Path, default=Path("outputs/dataset_audit.csv"))
    parser.add_argument("--preview-dir", type=Path, default=Path("outputs/audit_previews"))
    parser.add_argument("--make-previews", action="store_true")
    parser.add_argument("--max-previews", type=int, default=20)
    parser.add_argument("--preview-max-size", type=int, default=1024)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Mask block size in pixels. Lower this to 1024 if RAM is still tight.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=5,
        help="Write partial CSV every N cases so progress is preserved.",
    )
    parser.add_argument(
        "--skip-rgb-channel-check",
        action="store_true",
        help="Avoid reading RGB mask channels 2/3. Saves IO, not much RAM.",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()

    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {args.image_dir}")
    if not args.mask_dir.exists():
        raise FileNotFoundError(f"Mask directory does not exist: {args.mask_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.make_previews:
        args.preview_dir.mkdir(parents=True, exist_ok=True)

    image_files = find_files(args.image_dir)
    mask_files = find_files(args.mask_dir)
    if args.num_shards > 1:
        image_files = image_files[args.shard_index :: args.num_shards]
    if args.max_cases is not None:
        image_files = image_files[: args.max_cases]
    mask_by_stem = {p.stem: p for p in mask_files}

    rows = []
    previews_written = 0

    print(f"Found {len(image_files)} image files")
    print(f"Found {len(mask_files)} mask files")
    print(f"Mask chunk size: {args.chunk_size} x {args.chunk_size}")

    for i, image_path in enumerate(tqdm(image_files, desc="Auditing")):
        mask_path = mask_by_stem.get(image_path.stem)

        try:
            row = audit_one_case(image_path, mask_path, args)

            if (
                args.make_previews
                and mask_path is not None
                and previews_written < args.max_previews
                and row.get("shape_match")
            ):
                preview_path = args.preview_dir / f"{image_path.stem}.png"
                make_preview(image_path, mask_path, preview_path, args.preview_max_size)
                row["preview_path"] = str(preview_path)
                previews_written += 1

        except Exception as e:
            row = {
                "case_id": image_path.stem,
                "patient_id_guess": guess_patient_id(image_path.stem),
                "image_path": str(image_path),
                "mask_path": str(mask_path) if mask_path else "",
                "has_mask": mask_path is not None,
                "error": f"audit_error: {e}",
                "preview_path": "",
            }

        rows.append(row)

        if args.flush_every and (i + 1) % args.flush_every == 0:
            pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    summary_path = args.out.with_name(args.out.stem + "_summary.txt")
    write_summary(df, summary_path, args.image_dir, args.mask_dir)

    print(f"\nSaved audit CSV to: {args.out}")
    print(f"Saved summary to: {summary_path}")
    if args.make_previews:
        print(f"Saved previews to: {args.preview_dir}")


if __name__ == "__main__":
    main()
