#!/usr/bin/env python3
"""
Mine class-2 <-> class-3 confusion coordinates from TRAINING slides.

This reuses the same tiled whole-slide inference strategy as visual_analysis.py,
but writes a compact CSV manifest for targeted hard-example oversampling.

Use only a fold's TRAINING slides. Never mine from validation slides.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import openslide
import torch
from tiffslide import TiffSlide

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


MODEL_BASE_PATH = Path(os.environ.get(
    "MODEL_BASE_PATH",
    "/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/"
    "nnUNet_results/Dataset301_BEETLE/"
    "nnUNetTrainerPathologyFocalClassMetricsAlpha__"
    "nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"
))
CSV_PATH = Path(os.environ.get(
    "CSV_PATH",
    "/home/tijnveldwijk/fold0_training_inference_inputs.csv"
))
OUT_CSV = Path(os.environ.get(
    "OUT_CSV",
    "/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/"
    "hard_mining/wf250_fold0_train_hard_confusions.csv"
))
CHECKPOINT_NAME = os.environ.get("CHECKPOINT_NAME", "checkpoint_best.pth")
USE_MIRRORING = os.environ.get("USE_MIRRORING", "1") not in ("0", "false", "False")
MAX_SLIDES_RAW = os.environ.get("MAX_SLIDES")
MAX_SLIDES = None if MAX_SLIDES_RAW in (None, "", "None") else int(MAX_SLIDES_RAW)

MODEL_PATCH_SIZE = 512
CONTEXT = MODEL_PATCH_SIZE // 2
INPUT_TILE = 2048
OUTPUT_TILE = INPUT_TILE - 2 * CONTEXT  # 1536

# Hard coordinates are extracted in 512x512 cells so large erroneous regions can
# contribute multiple spatially distributed examples without flooding the list.
GRID_CELL = int(os.environ.get("HARD_GRID_CELL", "512"))
MIN_CONFUSION_PIXELS_PER_CELL = int(
    os.environ.get("MIN_CONFUSION_PIXELS_PER_CELL", "500")
)
MAX_POINTS_PER_SLIDE_DIRECTION = int(
    os.environ.get("MAX_POINTS_PER_SLIDE_DIRECTION", "200")
)

TARGET_PAIRS = [
    (2, 3),  # non-invasive epithelium predicted as invasive
    (3, 2),  # invasive epithelium predicted as non-invasive
]


def norm_01(x_batch: np.ndarray) -> np.ndarray:
    x_batch = x_batch.astype(np.float32) / 255.0
    return x_batch.transpose(3, 0, 1, 2)


def read_rgb_with_padding(wsi, x: int, y: int, size: int) -> np.ndarray:
    out = np.ones((size, size, 3), dtype=np.uint8) * 255
    wsi_w, wsi_h = wsi.dimensions

    src_x0 = max(x, 0)
    src_y0 = max(y, 0)
    src_x1 = min(x + size, wsi_w)
    src_y1 = min(y + size, wsi_h)

    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return out

    read_w = src_x1 - src_x0
    read_h = src_y1 - src_y0

    patch = wsi.read_region((src_x0, src_y0), 0, (read_w, read_h)).convert("RGB")
    patch = np.asarray(patch, dtype=np.uint8)

    dst_x0 = src_x0 - x
    dst_y0 = src_y0 - y
    out[dst_y0:dst_y0 + read_h, dst_x0:dst_x0 + read_w] = patch
    return out


def read_mask_region(mask_slide, x: int, y: int, w: int, h: int) -> np.ndarray:
    patch = mask_slide.read_region((x, y), 0, (w, h))
    arr = np.asarray(patch)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8)


def predict_tile(predictor, rgb_tile: np.ndarray) -> np.ndarray:
    batch = np.expand_dims(rgb_tile, axis=0)
    prep = norm_01(batch)

    with torch.no_grad():
        logits_list = predictor.get_logits_list_from_preprocessed_data(
            torch.tensor(prep, dtype=torch.float32)
        )
        softmax_list = [
            predictor.label_manager.apply_inference_nonlin(logits).cpu().numpy()
            for logits in logits_list
        ]

    softmax_mean = np.mean(softmax_list, axis=0)
    return np.squeeze(np.argmax(softmax_mean, axis=0)).astype(np.uint8)


def representative_confusion_point(
    confusion_mask: np.ndarray,
    global_x0: int,
    global_y0: int,
) -> tuple[int, int]:
    """
    Select an actual confusion pixel nearest the component cell's mean location.
    """
    coords = np.argwhere(confusion_mask)
    mean_yx = coords.mean(axis=0)
    distances = ((coords - mean_yx) ** 2).sum(axis=1)
    y, x = coords[int(np.argmin(distances))]
    return int(global_x0 + x), int(global_y0 + y)


def main() -> None:
    print("MODEL_BASE_PATH:", MODEL_BASE_PATH, flush=True)
    print("CSV_PATH:", CSV_PATH, flush=True)
    print("OUT_CSV:", OUT_CSV, flush=True)
    print("CHECKPOINT_NAME:", CHECKPOINT_NAME, flush=True)
    print("USE_MIRRORING:", USE_MIRRORING, flush=True)
    print("GRID_CELL:", GRID_CELL, flush=True)
    print("MIN_CONFUSION_PIXELS_PER_CELL:", MIN_CONFUSION_PIXELS_PER_CELL, flush=True)
    print("MAX_POINTS_PER_SLIDE_DIRECTION:", MAX_POINTS_PER_SLIDE_DIRECTION, flush=True)

    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Training CSV not found: {CSV_PATH}")

    with CSV_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if MAX_SLIDES is not None:
        rows = rows[:MAX_SLIDES]

    if not rows:
        raise RuntimeError(f"No rows found in training CSV: {CSV_PATH}")

    required_columns = {"wsi_path", "mask_path"}
    missing = required_columns - set(rows[0].keys())
    if missing:
        raise RuntimeError(
            f"Training CSV is missing columns {sorted(missing)}: {CSV_PATH}"
        )

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=USE_MIRRORING,
        perform_everything_on_gpu=True,
        device=torch.device("cuda", 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(MODEL_BASE_PATH),
        use_folds=(0,),
        checkpoint_name=CHECKPOINT_NAME,
    )

    print("ALLOWED_MIRRORING_AXES:", predictor.allowed_mirroring_axes, flush=True)
    print(
        "EFFECTIVE_MIRRORING:",
        bool(USE_MIRRORING and predictor.allowed_mirroring_axes),
        flush=True,
    )

    all_candidates = []
    processed_tiles = 0
    skipped_empty_tiles = 0

    for slide_idx, row in enumerate(rows):
        wsi_path = Path(row["wsi_path"])
        mask_path = Path(row["mask_path"])
        file_key = row.get("file_key") or wsi_path.stem
        slide = row.get("slide") or wsi_path.stem

        print(f"\n[{slide_idx + 1}/{len(rows)}] {slide}", flush=True)
        print("WSI :", wsi_path, flush=True)
        print("MASK:", mask_path, flush=True)

        wsi = openslide.OpenSlide(str(wsi_path))
        mask = TiffSlide(str(mask_path))

        if wsi.dimensions != mask.dimensions:
            raise RuntimeError(
                f"Dimension mismatch for {slide}: "
                f"WSI {wsi.dimensions} vs MASK {mask.dimensions}"
            )

        slide_candidates = defaultdict(list)
        mask_w, mask_h = mask.dimensions

        for y_out in range(0, mask_h, OUTPUT_TILE):
            h_out = min(OUTPUT_TILE, mask_h - y_out)

            for x_out in range(0, mask_w, OUTPUT_TILE):
                w_out = min(OUTPUT_TILE, mask_w - x_out)
                gt = read_mask_region(mask, x_out, y_out, w_out, h_out)
                valid = gt != 0

                if not valid.any():
                    skipped_empty_tiles += 1
                    continue

                x_in = x_out - CONTEXT
                y_in = y_out - CONTEXT
                rgb_in = read_rgb_with_padding(wsi, x_in, y_in, INPUT_TILE)
                pred_in = predict_tile(predictor, rgb_in)
                pred_out = pred_in[
                    CONTEXT:CONTEXT + h_out,
                    CONTEXT:CONTEXT + w_out,
                ].astype(np.uint8)

                if pred_out.shape != gt.shape:
                    raise RuntimeError(
                        f"Prediction/GT shape mismatch at {slide}: "
                        f"pred {pred_out.shape}, gt {gt.shape}"
                    )

                for gt_label, pred_label in TARGET_PAIRS:
                    pair_mask = (gt == gt_label) & (pred_out == pred_label) & valid

                    for cell_y0 in range(0, h_out, GRID_CELL):
                        cell_h = min(GRID_CELL, h_out - cell_y0)

                        for cell_x0 in range(0, w_out, GRID_CELL):
                            cell_w = min(GRID_CELL, w_out - cell_x0)
                            cell_pair_mask = pair_mask[
                                cell_y0:cell_y0 + cell_h,
                                cell_x0:cell_x0 + cell_w,
                            ]
                            confusion_pixels = int(cell_pair_mask.sum())

                            if confusion_pixels < MIN_CONFUSION_PIXELS_PER_CELL:
                                continue

                            cell_valid = valid[
                                cell_y0:cell_y0 + cell_h,
                                cell_x0:cell_x0 + cell_w,
                            ]
                            center_x, center_y = representative_confusion_point(
                                cell_pair_mask,
                                global_x0=x_out + cell_x0,
                                global_y0=y_out + cell_y0,
                            )

                            direction = f"gt{gt_label}_pred{pred_label}"
                            slide_candidates[direction].append({
                                "file_key": file_key,
                                "slide": slide,
                                "wsi_path": str(wsi_path),
                                "mask_path": str(mask_path),
                                "direction": direction,
                                "gt_label": gt_label,
                                "pred_label": pred_label,
                                "center_x": center_x,
                                "center_y": center_y,
                                "confusion_pixels_in_cell": confusion_pixels,
                                "valid_pixels_in_cell": int(cell_valid.sum()),
                                "tile_x_out": x_out,
                                "tile_y_out": y_out,
                                "cell_x_out": x_out + cell_x0,
                                "cell_y_out": y_out + cell_y0,
                            })

                processed_tiles += 1
                if processed_tiles % 25 == 0:
                    print(
                        f"  processed annotated tiles total: {processed_tiles}",
                        flush=True,
                    )

        wsi.close()
        mask.close()

        # Prevent a few slides with extensive errors from dominating the
        # manifest. Keep the strongest cells per direction on each slide.
        for direction, candidates in slide_candidates.items():
            candidates.sort(
                key=lambda row: int(row["confusion_pixels_in_cell"]),
                reverse=True,
            )
            kept = candidates[:MAX_POINTS_PER_SLIDE_DIRECTION]
            all_candidates.extend(kept)
            print(
                f"  {direction}: kept {len(kept)} / {len(candidates)} hard cells",
                flush=True,
            )

    # Deduplicate defensively.
    unique = {}
    for row in all_candidates:
        key = (
            row["file_key"],
            row["direction"],
            int(row["center_x"]),
            int(row["center_y"]),
        )
        previous = unique.get(key)
        if previous is None or int(row["confusion_pixels_in_cell"]) > int(
            previous["confusion_pixels_in_cell"]
        ):
            unique[key] = row

    all_candidates = sorted(
        unique.values(),
        key=lambda row: (
            str(row["direction"]),
            str(row["file_key"]),
            -int(row["confusion_pixels_in_cell"]),
        ),
    )

    if not all_candidates:
        raise RuntimeError(
            "No hard examples were mined. Lower MIN_CONFUSION_PIXELS_PER_CELL "
            "or verify that the CSV points to fold-0 TRAINING masks."
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file_key",
        "slide",
        "wsi_path",
        "mask_path",
        "direction",
        "gt_label",
        "pred_label",
        "center_x",
        "center_y",
        "confusion_pixels_in_cell",
        "valid_pixels_in_cell",
        "tile_x_out",
        "tile_y_out",
        "cell_x_out",
        "cell_y_out",
    ]

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_candidates)

    summary = {
        "model_base_path": str(MODEL_BASE_PATH),
        "checkpoint_name": CHECKPOINT_NAME,
        "csv_path": str(CSV_PATH),
        "out_csv": str(OUT_CSV),
        "use_mirroring": USE_MIRRORING,
        "allowed_mirroring_axes": predictor.allowed_mirroring_axes,
        "effective_mirroring": bool(USE_MIRRORING and predictor.allowed_mirroring_axes),
        "processed_annotated_tiles": processed_tiles,
        "skipped_empty_tiles": skipped_empty_tiles,
        "number_of_hard_examples": len(all_candidates),
        "counts_by_direction": dict(Counter(row["direction"] for row in all_candidates)),
        "counts_by_slide": dict(Counter(row["slide"] for row in all_candidates)),
        "grid_cell": GRID_CELL,
        "min_confusion_pixels_per_cell": MIN_CONFUSION_PIXELS_PER_CELL,
        "max_points_per_slide_direction": MAX_POINTS_PER_SLIDE_DIRECTION,
    }

    summary_path = OUT_CSV.with_suffix(".summary.json")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=4)

    print("\n=== HARD-MINING MANIFEST COMPLETE ===", flush=True)
    print("Hard examples:", len(all_candidates), flush=True)
    print("By direction:", summary["counts_by_direction"], flush=True)
    print("Saved CSV:", OUT_CSV, flush=True)
    print("Saved summary:", summary_path, flush=True)


if __name__ == "__main__":
    main()
