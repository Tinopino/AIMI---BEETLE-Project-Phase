import csv
import json
import os
import hashlib
from pathlib import Path

import numpy as np
import openslide
import torch
from PIL import Image, ImageDraw
from tiffslide import TiffSlide

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


# =============================================================================
# CONFIG
# =============================================================================

MODEL_BASE_PATH = Path(os.environ.get(
    "MODEL_BASE_PATH",
    "/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/nnUNet_results/"
    "Dataset301_BEETLE/"
    "nnUNetTrainerPathologyFocal__nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d"
))

CSV_PATH = Path(os.environ.get(
    "CSV_PATH",
    "/home/tijnveldwijk/fold0_validation_inference_inputs.csv"
))

EVAL_FOLD = int(os.environ.get("EVAL_FOLD", "0"))
FOLDS_TO_USE = (EVAL_FOLD,)
CHECKPOINT_NAME = os.environ.get("CHECKPOINT_NAME", "checkpoint_best.pth")

# Used only for naming output files/directories so multiple checkpoint validations
# do not overwrite each other.
CHECKPOINT_TAG = os.environ.get("CHECKPOINT_TAG", Path(CHECKPOINT_NAME).stem)
CHECKPOINT_TAG = (
    CHECKPOINT_TAG
    .replace("checkpoint_", "ckpt_")
    .replace("-", "_")
    .replace(" ", "_")
    .replace("/", "_")
    .replace("\\", "_")
)

# For quick screening you can set USE_MIRRORING=0. For final comparable
# validation, keep USE_MIRRORING=1 if earlier validation used mirroring.
USE_MIRRORING = os.environ.get("USE_MIRRORING", "1") not in ("0", "false", "False")

MODEL_PATCH_SIZE = int(os.environ.get("MODEL_PATCH_SIZE", "512"))
CONTEXT = MODEL_PATCH_SIZE // 2
INPUT_TILE = 2048
OUTPUT_TILE = INPUT_TILE - 2 * CONTEXT  # 1536

NUM_LABELS = 5
EVAL_LABELS = [1, 2, 3, 4]

CLASSES = {
    1: "other",
    2: "non-invasive epithelium",
    3: "invasive epithelium",
    4: "necrosis",
}

CLASS_SHORT = {
    1: "other",
    2: "non_invasive",
    3: "invasive",
    4: "necrosis",
}

LABEL_NAMES = {
    0: "unannotated_or_ignored",
    1: "other",
    2: "non-invasive epithelium",
    3: "invasive epithelium",
    4: "necrosis",
}

MAX_SLIDES = os.environ.get("MAX_SLIDES")
MAX_SLIDES = None if MAX_SLIDES in (None, "", "None") else int(MAX_SLIDES)

SAVE_VISUALS = os.environ.get("SAVE_VISUALS", "1") not in ("0", "false", "False")

VIS_OUT_DIR_BASE = Path(os.environ.get(
    "VIS_OUT_DIR_BASE",
    "/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/"
    "validation_visuals/PathologyFocal_fold0_hybrid_analysis"
))
VIS_OUT_DIR = Path(os.environ.get(
    "VIS_OUT_DIR",
    str(VIS_OUT_DIR_BASE / CHECKPOINT_TAG)
))

# Per-slide coverage examples
VIS_TOP_K_PER_SLIDE = int(os.environ.get("VIS_TOP_K_PER_SLIDE", "2"))
VIS_LOW_K_PER_SLIDE = int(os.environ.get("VIS_LOW_K_PER_SLIDE", "1"))

# Per-class examples
VIS_TOP_K_PER_CLASS_FN = int(os.environ.get("VIS_TOP_K_PER_CLASS_FN", "10"))
VIS_TOP_K_PER_CLASS_FP = int(os.environ.get("VIS_TOP_K_PER_CLASS_FP", "10"))

# Key confusion examples
VIS_TOP_K_PER_CONFUSION = int(os.environ.get("VIS_TOP_K_PER_CONFUSION", "10"))

# Thresholds for class/confusion example selection
MIN_GT_PIXELS_FOR_FN = int(os.environ.get("MIN_GT_PIXELS_FOR_FN", "500"))
MIN_FP_PIXELS = int(os.environ.get("MIN_FP_PIXELS", "500"))
MIN_CONFUSION_PIXELS = int(os.environ.get("MIN_CONFUSION_PIXELS", "500"))

# Extra qualitative-analysis selections.
# These prevent the visual analysis from being dominated by tiny annotated regions.
VIS_TOP_K_LARGE_AREA = int(os.environ.get("VIS_TOP_K_LARGE_AREA", "30"))
VIS_TOP_K_ABSOLUTE_ERROR = int(os.environ.get("VIS_TOP_K_ABSOLUTE_ERROR", "30"))

# Only consider tiles with at least this many annotated pixels for the large-area category.
MIN_VALID_PIXELS_LARGE_AREA = int(os.environ.get("MIN_VALID_PIXELS_LARGE_AREA", "100000"))

VIS_PANEL_SIZE = int(os.environ.get("VIS_PANEL_SIZE", "512"))

# Confusion pairs are ordered as:
#     (ground_truth_label, predicted_label)
#
# Example:
#     (3, 2) = invasive epithelium predicted as non-invasive epithelium.
CONFUSION_PAIRS = [
    (2, 3),  # non-invasive predicted as invasive
    (3, 2),  # invasive predicted as non-invasive
    (1, 2),  # other predicted as non-invasive
    (4, 2),  # necrosis predicted as non-invasive
    (3, 1),  # invasive predicted as other
    (4, 1),  # necrosis predicted as other
    (1, 4),  # other predicted as necrosis
]

# BEETLE paper-style visualisation palette:
# 0 = black       = unannotated / ignored
# 1 = cyan        = other
# 2 = yellow      = non-invasive epithelium
# 3 = magenta     = invasive epithelium
# 4 = dark blue   = necrosis
#
# Important:
# These colours are only for visualisation. They do not change the label IDs used
# for evaluation, Dice, or the confusion matrix.
COLORS = {
    0: (0, 0, 0),          # unannotated / ignored
    1: (0, 190, 210),      # other
    2: (255, 230, 0),      # non-invasive epithelium
    3: (240, 0, 130),      # invasive epithelium
    4: (35, 55, 140),      # necrosis
}


# =============================================================================
# IO / MODEL HELPERS
# =============================================================================

def norm_01(x_batch: np.ndarray) -> np.ndarray:
    """
    Input:  x_batch with shape (N, H, W, C), values 0-255
    Output: shape (C, N, H, W), values 0-1

    This matches RGBTo01Normalization.
    Do not replace with z-score normalization.
    """
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
    pred = np.squeeze(np.argmax(softmax_mean, axis=0)).astype(np.uint8)
    return pred


# =============================================================================
# METRIC HELPERS
# =============================================================================

def update_confusion_matrix(cm: np.ndarray, gt: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> None:
    """
    Rows = GT labels, columns = predicted labels.
    Only GT != 0 pixels are counted.
    Predictions may still be 0.
    """
    gt_valid = gt[valid].astype(np.int64)
    pred_valid = pred[valid].astype(np.int64)

    gt_valid = np.clip(gt_valid, 0, NUM_LABELS - 1)
    pred_valid = np.clip(pred_valid, 0, NUM_LABELS - 1)

    cm_tile = np.bincount(
        NUM_LABELS * gt_valid + pred_valid,
        minlength=NUM_LABELS * NUM_LABELS,
    ).reshape(NUM_LABELS, NUM_LABELS)

    cm += cm_tile


def dice_from_confusion_matrix(cm: np.ndarray) -> dict:
    """
    Compute class Dice from full confusion matrix.
    Rows are GT, columns are predicted.
    """
    out = {}

    for c in EVAL_LABELS:
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum() - cm[c, c])
        fp = int(cm[:, c].sum() - cm[c, c])
        denom = 2 * tp + fp + fn
        dice = 2 * tp / denom if denom > 0 else float("nan")
        out[CLASSES[c]] = dice

    return out


def save_confusion_matrix_csv(cm: np.ndarray, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "GT \\ Pred",
        "0_unannotated_or_ignored",
        "1_other",
        "2_non_invasive_epithelium",
        "3_invasive_epithelium",
        "4_necrosis",
    ]

    with open(out_csv, "w", newline="") as f:
        f.write("sep=,\n")
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(NUM_LABELS):
            writer.writerow([f"{i}_{LABEL_NAMES[i]}"] + cm[i, :].astype(int).tolist())


def save_normalized_confusion_matrix_csv(cm: np.ndarray, out_csv: Path) -> None:
    """
    Row-normalized confusion matrix.
    Each row sums to 1, unless the GT class has zero pixels.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cm_float = cm.astype(np.float64)
    row_sums = cm_float.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_float, row_sums, out=np.zeros_like(cm_float), where=row_sums > 0)

    header = [
        "GT \\ Pred",
        "0_unannotated_or_ignored",
        "1_other",
        "2_non_invasive_epithelium",
        "3_invasive_epithelium",
        "4_necrosis",
    ]

    with open(out_csv, "w", newline="") as f:
        f.write("sep=,\n")
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(NUM_LABELS):
            writer.writerow([f"{i}_{LABEL_NAMES[i]}"] + [f"{x:.6f}" for x in cm_norm[i, :]])


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def colorize(mask: np.ndarray) -> np.ndarray:
    """
    Convert a label mask to an RGB colour mask using the BEETLE paper palette.
    """
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label, color in COLORS.items():
        out[mask == label] = color
    return out


def overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
    show_unannotated_black: bool = False,
) -> np.ndarray:
    """
    Overlay class colours on top of the RGB image.

    If show_unannotated_black=True, label 0 is displayed as black. This is useful
    for matching the BEETLE paper figures, where black means unannotated.

    If show_unannotated_black=False, label 0 remains as the original RGB tissue.
    """
    rgb = rgb.astype(np.uint8)
    color = colorize(mask)

    out = rgb.copy()

    if show_unannotated_black:
        out[mask == 0] = COLORS[0]

    valid = mask > 0
    out[valid] = ((1 - alpha) * out[valid] + alpha * color[valid]).astype(np.uint8)

    return out


def make_error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Error map:
        black   = unannotated / ignored
        green   = correct prediction
        magenta = wrong prediction
    """
    out = np.zeros((*gt.shape, 3), dtype=np.uint8)

    valid = gt > 0
    correct = valid & (gt == pred)
    wrong = valid & (gt != pred)

    out[correct] = (0, 180, 0)
    out[wrong] = (255, 0, 255)
    out[~valid] = COLORS[0]

    return out


def resize_for_panel(arr: np.ndarray, size: int = VIS_PANEL_SIZE) -> Image.Image:
    img = Image.fromarray(arr.astype(np.uint8))
    img.thumbnail((size, size), Image.Resampling.NEAREST)
    return img


def add_title(img: Image.Image, title: str) -> Image.Image:
    canvas = Image.new("RGB", (img.width, img.height + 28), (255, 255, 255))
    canvas.paste(img, (0, 28))

    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(0, 0, 0))

    return canvas


def save_visual_panel(
    rgb: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """
    Save a 4-panel qualitative figure:
        1. RGB tissue
        2. Ground truth overlay
        3. Prediction overlay
        4. Error map

    Predictions outside annotated GT pixels are set to 0 for display. This avoids
    colouring ignored/unannotated areas as if they contributed to evaluation.
    """
    valid = gt > 0

    pred_display = pred.copy()
    pred_display[~valid] = 0

    gt_overlay = overlay(
        rgb,
        gt,
        alpha=0.45,
        show_unannotated_black=True,
    )

    pred_overlay = overlay(
        rgb,
        pred_display,
        alpha=0.45,
        show_unannotated_black=True,
    )

    err = make_error_map(gt, pred)

    panels = [
        add_title(resize_for_panel(rgb), "RGB"),
        add_title(resize_for_panel(gt_overlay), "Ground truth"),
        add_title(resize_for_panel(pred_overlay), "Prediction"),
        add_title(resize_for_panel(err), "Error: green correct, magenta wrong"),
    ]

    width = sum(p.width for p in panels)
    height = max(p.height for p in panels) + 34

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill=(0, 0, 0))

    x = 0
    for p in panels:
        canvas.paste(p, (x, 34))
        x += p.width

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


# =============================================================================
# CANDIDATE SELECTION HELPERS
# =============================================================================

def class_summary(mask: np.ndarray) -> str:
    labels = sorted(int(x) for x in np.unique(mask) if int(x) != 0)
    if not labels:
        return "none"
    return "-".join(str(x) for x in labels)


def dominant_class(mask: np.ndarray, valid: np.ndarray) -> int:
    values = mask[valid]
    values = values[values != 0]
    if values.size == 0:
        return 0

    labels, counts = np.unique(values, return_counts=True)
    return int(labels[np.argmax(counts)])


def make_candidate(
    score: float,
    score_name: str,
    err_rate: float,
    global_tile_idx: int,
    slide_stem: str,
    x_out: int,
    y_out: int,
    rgb: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    valid: np.ndarray,
    extra: dict | None = None,
) -> dict:
    if extra is None:
        extra = {}

    return {
        "score": float(score),
        "score_name": score_name,
        "err_rate": float(err_rate),
        "global_tile_idx": int(global_tile_idx),
        "slide": slide_stem,
        "x_out": int(x_out),
        "y_out": int(y_out),
        "gt_classes": class_summary(gt),
        "pred_classes": class_summary(pred[valid]),
        "dominant_gt": dominant_class(gt, valid),
        "dominant_pred": dominant_class(pred, valid),
        "extra": extra,
        "rgb": rgb.copy(),
        "gt": gt.copy(),
        "pred": pred.copy(),
    }


def update_top_desc(candidates: list, candidate: dict, k: int) -> None:
    if k <= 0:
        return
    candidates.append(candidate)
    candidates.sort(key=lambda d: d["score"], reverse=True)
    del candidates[k:]


def update_top_asc(candidates: list, candidate: dict, k: int) -> None:
    if k <= 0:
        return
    candidates.append(candidate)
    candidates.sort(key=lambda d: d["score"])
    del candidates[k:]


def sanitize_name(s: str) -> str:
    return (
        s.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(";", "_")
    )


def short_hash(s: str, n: int = 8) -> str:
    return hashlib.sha1(str(s).encode("utf-8")).hexdigest()[:n]


def save_candidate(candidate: dict, out_dir: Path, prefix: str, rank: int) -> dict:
    """
    Save one visual panel with a short filename.

    Full metadata is returned and later saved to metadata files by
    save_candidate_group(). This avoids Linux filename length errors caused by
    long TCGA slide names plus many metadata fields.
    """
    safe_prefix = sanitize_name(prefix)[:70]
    slide_hash = short_hash(candidate["slide"])

    out_path = out_dir / (
        f"{safe_prefix}_"
        f"r{rank:02d}_"
        f"t{candidate['global_tile_idx']:06d}_"
        f"s{slide_hash}_"
        f"x{candidate['x_out']}_"
        f"y{candidate['y_out']}.png"
    )

    title = (
        f"{safe_prefix} r{rank:02d} | "
        f"tile={candidate['global_tile_idx']} | "
        f"x={candidate['x_out']} y={candidate['y_out']} | "
        f"err={candidate['err_rate']:.3f} | "
        f"{candidate['score_name']}={candidate['score']:.3f} | "
        f"GT={candidate['dominant_gt']} pred={candidate['dominant_pred']}"
    )

    save_visual_panel(
        rgb=candidate["rgb"],
        gt=candidate["gt"],
        pred=candidate["pred"],
        title=title,
        out_path=out_path,
    )

    return {
        "path": str(out_path),
        "filename": out_path.name,
        "rank": rank,
        "score": candidate["score"],
        "score_name": candidate["score_name"],
        "err_rate": candidate["err_rate"],
        "global_tile_idx": candidate["global_tile_idx"],
        "slide": candidate["slide"],
        "slide_hash": slide_hash,
        "x_out": candidate["x_out"],
        "y_out": candidate["y_out"],
        "gt_classes": candidate["gt_classes"],
        "pred_classes": candidate["pred_classes"],
        "dominant_gt": candidate["dominant_gt"],
        "dominant_pred": candidate["dominant_pred"],
        "extra": candidate["extra"],
    }


def save_metadata_files(saved: list, out_dir: Path, prefix: str) -> None:
    if not saved:
        return

    safe_prefix = sanitize_name(prefix)[:70]

    json_path = out_dir / f"{safe_prefix}_metadata.json"
    csv_path = out_dir / f"{safe_prefix}_metadata.csv"
    txt_path = out_dir / f"{safe_prefix}_metadata.txt"

    with open(json_path, "w") as f:
        json.dump(saved, f, indent=4)

    fieldnames = [
        "rank",
        "filename",
        "path",
        "score",
        "score_name",
        "err_rate",
        "global_tile_idx",
        "slide",
        "slide_hash",
        "x_out",
        "y_out",
        "gt_classes",
        "pred_classes",
        "dominant_gt",
        "dominant_pred",
        "extra_json",
    ]

    with open(csv_path, "w", newline="") as f:
        f.write("sep=,\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in saved:
            row = {k: item.get(k, "") for k in fieldnames if k != "extra_json"}
            row["extra_json"] = json.dumps(item.get("extra", {}), sort_keys=True)
            writer.writerow(row)

    with open(txt_path, "w") as f:
        f.write(f"Visual candidate group: {prefix}\n")
        f.write(f"Number of saved images: {len(saved)}\n\n")
        f.write("Label mapping / colours:\n")
        for label in range(NUM_LABELS):
            f.write(f"  {label}: {LABEL_NAMES[label]} | RGB={COLORS[label]}\n")

        f.write("\nSaved candidates:\n")
        for item in saved:
            f.write(
                f"\nRank {item['rank']:02d}: {item['filename']}\n"
                f"  slide: {item['slide']}\n"
                f"  tile: {item['global_tile_idx']} | x={item['x_out']} y={item['y_out']}\n"
                f"  err_rate: {item['err_rate']:.6f}\n"
                f"  score: {item['score']:.6f} ({item['score_name']})\n"
                f"  gt_classes: {item['gt_classes']}\n"
                f"  pred_classes: {item['pred_classes']}\n"
                f"  dominant_gt: {item['dominant_gt']}\n"
                f"  dominant_pred: {item['dominant_pred']}\n"
                f"  extra: {json.dumps(item.get('extra', {}), sort_keys=True)}\n"
            )


def save_candidate_group(candidates: list, out_dir: Path, prefix: str) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    used = set()
    unique_candidates = []
    for c in candidates:
        key = (c["slide"], c["x_out"], c["y_out"], c["global_tile_idx"])
        if key in used:
            continue
        used.add(key)
        unique_candidates.append(c)

    for rank, c in enumerate(unique_candidates):
        saved.append(save_candidate(c, out_dir, prefix, rank))

    save_metadata_files(saved, out_dir, prefix)

    return saved


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("MODEL_BASE_PATH:", MODEL_BASE_PATH)
    print("CSV_PATH:", CSV_PATH)
    print("FOLDS_TO_USE:", FOLDS_TO_USE)
    print("CHECKPOINT_NAME:", CHECKPOINT_NAME)
    print("CHECKPOINT_TAG:", CHECKPOINT_TAG)
    print("USE_MIRRORING:", USE_MIRRORING)
    print("INPUT_TILE:", INPUT_TILE)
    print("OUTPUT_TILE:", OUTPUT_TILE)
    print("SAVE_VISUALS:", SAVE_VISUALS)

    if SAVE_VISUALS:
        VIS_OUT_DIR.mkdir(parents=True, exist_ok=True)
        print("VIS_OUT_DIR:", VIS_OUT_DIR)
        print("VIS_TOP_K_PER_SLIDE:", VIS_TOP_K_PER_SLIDE)
        print("VIS_LOW_K_PER_SLIDE:", VIS_LOW_K_PER_SLIDE)
        print("VIS_TOP_K_PER_CLASS_FN:", VIS_TOP_K_PER_CLASS_FN)
        print("VIS_TOP_K_PER_CLASS_FP:", VIS_TOP_K_PER_CLASS_FP)
        print("VIS_TOP_K_PER_CONFUSION:", VIS_TOP_K_PER_CONFUSION)
        print("MIN_GT_PIXELS_FOR_FN:", MIN_GT_PIXELS_FOR_FN)
        print("MIN_FP_PIXELS:", MIN_FP_PIXELS)
        print("MIN_CONFUSION_PIXELS:", MIN_CONFUSION_PIXELS)
        print("VIS_TOP_K_LARGE_AREA:", VIS_TOP_K_LARGE_AREA)
        print("VIS_TOP_K_ABSOLUTE_ERROR:", VIS_TOP_K_ABSOLUTE_ERROR)
        print("MIN_VALID_PIXELS_LARGE_AREA:", MIN_VALID_PIXELS_LARGE_AREA)
        print("VISUAL COLORS:")
        for label, color in COLORS.items():
            print(f"  {label}_{LABEL_NAMES[label]}: RGB{color}")

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
        use_folds=FOLDS_TO_USE,
        checkpoint_name=CHECKPOINT_NAME,
    )

    # Explicitly enable both spatial axes for 2D test-time augmentation.
    if USE_MIRRORING and predictor.allowed_mirroring_axes is None:
        predictor.allowed_mirroring_axes = (0, 1)
        print("FORCED_MIRRORING_AXES:", predictor.allowed_mirroring_axes)

    print("ALLOWED_MIRRORING_AXES:", predictor.allowed_mirroring_axes)
    print(
        "EFFECTIVE_MIRRORING:",
        bool(USE_MIRRORING and predictor.allowed_mirroring_axes),
    )

    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    if MAX_SLIDES is not None:
        rows = rows[:MAX_SLIDES]

    print(f"Processing {len(rows)} validation slides")

    tp = {c: 0 for c in EVAL_LABELS}
    fp = {c: 0 for c in EVAL_LABELS}
    fn = {c: 0 for c in EVAL_LABELS}

    confusion_matrix = np.zeros((NUM_LABELS, NUM_LABELS), dtype=np.int64)

    processed_tiles = 0
    skipped_empty_tiles = 0

    per_slide_results = []

    global_fn_candidates = {c: [] for c in EVAL_LABELS}
    global_fp_candidates = {c: [] for c in EVAL_LABELS}
    global_confusion_candidates = {pair: [] for pair in CONFUSION_PAIRS}

    # Extra global visual selections for qualitative analysis.
    global_large_area_candidates = []
    global_absolute_error_candidates = []

    for slide_idx, row in enumerate(rows):
        wsi_path = Path(row["wsi_path"])
        mask_path = Path(row["mask_path"])
        slide_stem = wsi_path.stem

        print(f"\n[{slide_idx + 1}/{len(rows)}] {slide_stem}", flush=True)
        print("WSI :", wsi_path, flush=True)
        print("MASK:", mask_path, flush=True)

        wsi = openslide.OpenSlide(str(wsi_path))
        mask = TiffSlide(str(mask_path))

        wsi_w, wsi_h = wsi.dimensions
        mask_w, mask_h = mask.dimensions

        print("WSI dimensions :", (wsi_w, wsi_h), flush=True)
        print("MASK dimensions:", (mask_w, mask_h), flush=True)

        if (wsi_w, wsi_h) != (mask_w, mask_h):
            raise RuntimeError(
                f"Dimension mismatch for {slide_stem}: "
                f"WSI {(wsi_w, wsi_h)} vs MASK {(mask_w, mask_h)}"
            )

        slide_valid_tiles = 0
        slide_skipped_tiles = 0

        slide_tp = {c: 0 for c in EVAL_LABELS}
        slide_fp = {c: 0 for c in EVAL_LABELS}
        slide_fn = {c: 0 for c in EVAL_LABELS}
        slide_confusion_matrix = np.zeros((NUM_LABELS, NUM_LABELS), dtype=np.int64)

        slide_high_error_candidates = []
        slide_low_error_candidates = []

        for y_out in range(0, mask_h, OUTPUT_TILE):
            h_out = min(OUTPUT_TILE, mask_h - y_out)

            for x_out in range(0, mask_w, OUTPUT_TILE):
                w_out = min(OUTPUT_TILE, mask_w - x_out)

                gt = read_mask_region(mask, x_out, y_out, w_out, h_out)

                valid = gt != 0
                if not valid.any():
                    skipped_empty_tiles += 1
                    slide_skipped_tiles += 1
                    continue

                x_in = x_out - CONTEXT
                y_in = y_out - CONTEXT

                rgb_in = read_rgb_with_padding(wsi, x_in, y_in, INPUT_TILE)
                pred_in = predict_tile(predictor, rgb_in)

                pred_out = pred_in[
                    CONTEXT:CONTEXT + h_out,
                    CONTEXT:CONTEXT + w_out,
                ].astype(np.uint8)

                rgb_out = rgb_in[
                    CONTEXT:CONTEXT + h_out,
                    CONTEXT:CONTEXT + w_out,
                ].astype(np.uint8)

                if pred_out.shape != gt.shape:
                    raise RuntimeError(
                        f"Prediction/GT shape mismatch at {slide_stem}: "
                        f"pred {pred_out.shape}, gt {gt.shape}"
                    )

                wrong_pixels = int(((pred_out != gt) & valid).sum())
                valid_pixels = int(valid.sum())
                err_rate = float(wrong_pixels / valid_pixels)

                update_confusion_matrix(confusion_matrix, gt, pred_out, valid)
                update_confusion_matrix(slide_confusion_matrix, gt, pred_out, valid)

                if SAVE_VISUALS:
                    slide_candidate = make_candidate(
                        score=err_rate,
                        score_name="tile_error_rate",
                        err_rate=err_rate,
                        global_tile_idx=processed_tiles,
                        slide_stem=slide_stem,
                        x_out=x_out,
                        y_out=y_out,
                        rgb=rgb_out,
                        gt=gt,
                        pred=pred_out,
                        valid=valid,
                        extra={
                            "valid_pixels": valid_pixels,
                            "wrong_pixels": wrong_pixels,
                        },
                    )
                    update_top_desc(slide_high_error_candidates, slide_candidate, VIS_TOP_K_PER_SLIDE)
                    update_top_asc(slide_low_error_candidates, slide_candidate, VIS_LOW_K_PER_SLIDE)

                    # Global examples ranked by absolute number of wrong annotated pixels.
                    # These often contribute most to Dice loss and are useful for qualitative analysis.
                    absolute_error_candidate = make_candidate(
                        score=float(wrong_pixels),
                        score_name="wrong_pixels",
                        err_rate=err_rate,
                        global_tile_idx=processed_tiles,
                        slide_stem=slide_stem,
                        x_out=x_out,
                        y_out=y_out,
                        rgb=rgb_out,
                        gt=gt,
                        pred=pred_out,
                        valid=valid,
                        extra={
                            "valid_pixels": valid_pixels,
                            "wrong_pixels": wrong_pixels,
                        },
                    )
                    update_top_desc(
                        global_absolute_error_candidates,
                        absolute_error_candidate,
                        VIS_TOP_K_ABSOLUTE_ERROR,
                    )

                    # Global examples with substantial annotated area, ranked by error rate.
                    # These are better for report figures than tiny annotated regions.
                    if valid_pixels >= MIN_VALID_PIXELS_LARGE_AREA:
                        large_area_candidate = make_candidate(
                            score=err_rate,
                            score_name="large_area_error_rate",
                            err_rate=err_rate,
                            global_tile_idx=processed_tiles,
                            slide_stem=slide_stem,
                            x_out=x_out,
                            y_out=y_out,
                            rgb=rgb_out,
                            gt=gt,
                            pred=pred_out,
                            valid=valid,
                            extra={
                                "valid_pixels": valid_pixels,
                                "wrong_pixels": wrong_pixels,
                            },
                        )
                        update_top_desc(
                            global_large_area_candidates,
                            large_area_candidate,
                            VIS_TOP_K_LARGE_AREA,
                        )

                for c in EVAL_LABELS:
                    gt_c = (gt == c) & valid
                    pred_c = (pred_out == c) & valid

                    tile_tp = int((gt_c & pred_c).sum())
                    tile_fp = int((~gt_c & pred_c).sum())
                    tile_fn = int((gt_c & ~pred_c).sum())

                    gt_area = int(gt_c.sum())
                    pred_area = int(pred_c.sum())

                    tp[c] += tile_tp
                    fp[c] += tile_fp
                    fn[c] += tile_fn

                    slide_tp[c] += tile_tp
                    slide_fp[c] += tile_fp
                    slide_fn[c] += tile_fn

                    if SAVE_VISUALS:
                        if gt_area >= MIN_GT_PIXELS_FOR_FN and tile_fn > 0:
                            fn_rate = tile_fn / gt_area

                            fn_candidate = make_candidate(
                                score=fn_rate,
                                score_name=f"FN_rate_class_{c}",
                                err_rate=err_rate,
                                global_tile_idx=processed_tiles,
                                slide_stem=slide_stem,
                                x_out=x_out,
                                y_out=y_out,
                                rgb=rgb_out,
                                gt=gt,
                                pred=pred_out,
                                valid=valid,
                                extra={
                                    "class": c,
                                    "fn_pixels": tile_fn,
                                    "gt_area": gt_area,
                                    "valid_pixels": valid_pixels,
                                    "wrong_pixels": wrong_pixels,
                                },
                            )
                            update_top_desc(global_fn_candidates[c], fn_candidate, VIS_TOP_K_PER_CLASS_FN)

                        if tile_fp >= MIN_FP_PIXELS:
                            fp_rate = tile_fp / max(pred_area, 1)

                            fp_candidate = make_candidate(
                                score=float(tile_fp),
                                score_name=f"FP_pixels_class_{c}",
                                err_rate=err_rate,
                                global_tile_idx=processed_tiles,
                                slide_stem=slide_stem,
                                x_out=x_out,
                                y_out=y_out,
                                rgb=rgb_out,
                                gt=gt,
                                pred=pred_out,
                                valid=valid,
                                extra={
                                    "class": c,
                                    "fp_pixels": tile_fp,
                                    "fp_rate": fp_rate,
                                    "pred_area": pred_area,
                                    "valid_pixels": valid_pixels,
                                    "wrong_pixels": wrong_pixels,
                                },
                            )
                            update_top_desc(global_fp_candidates[c], fp_candidate, VIS_TOP_K_PER_CLASS_FP)

                if SAVE_VISUALS:
                    for gt_label, pred_label in CONFUSION_PAIRS:
                        confusion_mask = (gt == gt_label) & (pred_out == pred_label) & valid
                        confusion_pixels = int(confusion_mask.sum())

                        if confusion_pixels >= MIN_CONFUSION_PIXELS:
                            gt_area_for_pair = int(((gt == gt_label) & valid).sum())
                            confusion_rate = confusion_pixels / max(gt_area_for_pair, 1)

                            confusion_candidate = make_candidate(
                                score=float(confusion_pixels),
                                score_name=f"confusion_gt{gt_label}_pred{pred_label}_pixels",
                                err_rate=err_rate,
                                global_tile_idx=processed_tiles,
                                slide_stem=slide_stem,
                                x_out=x_out,
                                y_out=y_out,
                                rgb=rgb_out,
                                gt=gt,
                                pred=pred_out,
                                valid=valid,
                                extra={
                                    "gt": gt_label,
                                    "pred": pred_label,
                                    "confusion_pixels": confusion_pixels,
                                    "confusion_rate": confusion_rate,
                                    "gt_area": gt_area_for_pair,
                                    "valid_pixels": valid_pixels,
                                    "wrong_pixels": wrong_pixels,
                                },
                            )
                            update_top_desc(
                                global_confusion_candidates[(gt_label, pred_label)],
                                confusion_candidate,
                                VIS_TOP_K_PER_CONFUSION,
                            )

                processed_tiles += 1
                slide_valid_tiles += 1

                if processed_tiles % 25 == 0:
                    print(
                        f"  processed annotated tiles total: {processed_tiles}",
                        flush=True,
                    )

        wsi.close()
        mask.close()

        saved_slide_visuals = []
        if SAVE_VISUALS and slide_valid_tiles > 0:
            slide_out_dir = VIS_OUT_DIR / "per_slide" / slide_stem

            saved_slide_visuals.extend(
                save_candidate_group(slide_high_error_candidates, slide_out_dir, "high_error")
            )
            saved_slide_visuals.extend(
                save_candidate_group(slide_low_error_candidates, slide_out_dir, "low_error")
            )

        slide_class_dices = {}
        for c, name in CLASSES.items():
            denom = 2 * slide_tp[c] + slide_fp[c] + slide_fn[c]
            dice = 2 * slide_tp[c] / denom if denom > 0 else float("nan")
            slide_class_dices[name] = dice

        slide_result = {
            "slide": slide_stem,
            "wsi_path": str(wsi_path),
            "mask_path": str(mask_path),
            "slide_valid_tiles": slide_valid_tiles,
            "slide_skipped_empty_tiles": slide_skipped_tiles,
            "slide_tp": slide_tp,
            "slide_fp": slide_fp,
            "slide_fn": slide_fn,
            "slide_class_dices": slide_class_dices,
            "slide_confusion_matrix_rows_gt_cols_pred": slide_confusion_matrix.tolist(),
            "saved_slide_visuals": saved_slide_visuals,
        }
        per_slide_results.append(slide_result)

        print(f"slide annotated tiles processed: {slide_valid_tiles}", flush=True)
        print(f"slide coverage visuals saved: {len(saved_slide_visuals)}", flush=True)

    saved_class_visuals = {}
    saved_confusion_visuals = {}
    saved_large_area_visuals = []
    saved_absolute_error_visuals = []

    if SAVE_VISUALS:
        for c, name in CLASSES.items():
            class_dir = VIS_OUT_DIR / "per_class" / f"class_{c}_{CLASS_SHORT[c]}"

            saved_class_visuals[f"class_{c}_{CLASS_SHORT[c]}_FN"] = save_candidate_group(
                global_fn_candidates[c],
                class_dir / "false_negatives",
                f"class{c}_false_negative",
            )

            saved_class_visuals[f"class_{c}_{CLASS_SHORT[c]}_FP"] = save_candidate_group(
                global_fp_candidates[c],
                class_dir / "false_positives",
                f"class{c}_false_positive",
            )

        for gt_label, pred_label in CONFUSION_PAIRS:
            confusion_name = (
                f"gt{gt_label}_{CLASS_SHORT[gt_label]}__"
                f"pred{pred_label}_{CLASS_SHORT[pred_label]}"
            )
            saved_confusion_visuals[confusion_name] = save_candidate_group(
                global_confusion_candidates[(gt_label, pred_label)],
                VIS_OUT_DIR / "confusions" / confusion_name,
                confusion_name,
            )

        saved_large_area_visuals = save_candidate_group(
            global_large_area_candidates,
            VIS_OUT_DIR / "large_area_high_error",
            "large_area_high_error",
        )

        saved_absolute_error_visuals = save_candidate_group(
            global_absolute_error_candidates,
            VIS_OUT_DIR / "large_absolute_error",
            "large_absolute_error",
        )

    print(f"\n=== Fold {EVAL_FOLD} full validation Dice over annotated pixels ===")
    print("processed annotated tiles:", processed_tiles)
    print("skipped empty tiles:", skipped_empty_tiles)

    class_dices = {}

    for c, name in CLASSES.items():
        denom = 2 * tp[c] + fp[c] + fn[c]
        dice = 2 * tp[c] / denom if denom > 0 else float("nan")
        class_dices[name] = dice
        print(f"{name}: {dice:.4f}  TP={tp[c]} FP={fp[c]} FN={fn[c]}")

    class_dices_from_cm = dice_from_confusion_matrix(confusion_matrix)

    macro = float(np.nanmean(list(class_dices.values())))

    micro_tp = sum(tp.values())
    micro_fp = sum(fp.values())
    micro_fn = sum(fn.values())
    micro = 2 * micro_tp / (2 * micro_tp + micro_fp + micro_fn)

    print(f"\nmacro mean Dice: {macro:.4f}")
    print(f"micro/overall Dice: {micro:.4f}")

    print("\n=== Confusion matrix, rows=GT, cols=Prediction ===")
    print(confusion_matrix)

    cm_csv = MODEL_BASE_PATH / f"fold_{EVAL_FOLD}" / f"fold{EVAL_FOLD}_{CHECKPOINT_TAG}_confusion_matrix_rows_gt_cols_pred.csv"
    cm_norm_csv = MODEL_BASE_PATH / f"fold_{EVAL_FOLD}" / f"fold{EVAL_FOLD}_{CHECKPOINT_TAG}_confusion_matrix_row_normalized.csv"
    save_confusion_matrix_csv(confusion_matrix, cm_csv)
    save_normalized_confusion_matrix_csv(confusion_matrix, cm_norm_csv)

    print("Saved confusion matrix CSV:", cm_csv)
    print("Saved row-normalized confusion matrix CSV:", cm_norm_csv)

    total_saved_slide = sum(len(x["saved_slide_visuals"]) for x in per_slide_results)
    total_saved_class = sum(len(v) for v in saved_class_visuals.values())
    total_saved_confusion = sum(len(v) for v in saved_confusion_visuals.values())
    total_saved_large_area = len(saved_large_area_visuals)
    total_saved_absolute_error = len(saved_absolute_error_visuals)

    print("\n=== Visual panels saved ===")
    print("per-slide coverage:", total_saved_slide)
    print("per-class FP/FN:", total_saved_class)
    print("confusions:", total_saved_confusion)
    print("large-area high-error:", total_saved_large_area)
    print("large absolute-error:", total_saved_absolute_error)
    print(
        "total:",
        total_saved_slide
        + total_saved_class
        + total_saved_confusion
        + total_saved_large_area
        + total_saved_absolute_error,
    )

    out_json = MODEL_BASE_PATH / f"fold_{EVAL_FOLD}" / f"fold{EVAL_FOLD}_{CHECKPOINT_TAG}_full_validation_dice_tiffslide_hybrid_visual_cm.json"

    result = {
        "model_base_path": str(MODEL_BASE_PATH),
        "csv_path": str(CSV_PATH),
        "folds_to_use": list(FOLDS_TO_USE),
        "checkpoint_name": CHECKPOINT_NAME,
        "labels": LABEL_NAMES,
        "classes": CLASSES,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "class_dices": class_dices,
        "class_dices_from_confusion_matrix": class_dices_from_cm,
        "macro_mean_dice": macro,
        "micro_overall_dice": micro,
        "confusion_matrix_rows_gt_cols_pred": confusion_matrix.tolist(),
        "confusion_matrix_csv": str(cm_csv),
        "confusion_matrix_row_normalized_csv": str(cm_norm_csv),
        "processed_annotated_tiles": processed_tiles,
        "skipped_empty_tiles": skipped_empty_tiles,
        "save_visuals": SAVE_VISUALS,
        "visual_output_dir": str(VIS_OUT_DIR),
        "vis_top_k_per_slide": VIS_TOP_K_PER_SLIDE,
        "vis_low_k_per_slide": VIS_LOW_K_PER_SLIDE,
        "vis_top_k_per_class_fn": VIS_TOP_K_PER_CLASS_FN,
        "vis_top_k_per_class_fp": VIS_TOP_K_PER_CLASS_FP,
        "vis_top_k_per_confusion": VIS_TOP_K_PER_CONFUSION,
        "min_gt_pixels_for_fn": MIN_GT_PIXELS_FOR_FN,
        "min_fp_pixels": MIN_FP_PIXELS,
        "min_confusion_pixels": MIN_CONFUSION_PIXELS,
        "vis_top_k_large_area": VIS_TOP_K_LARGE_AREA,
        "vis_top_k_absolute_error": VIS_TOP_K_ABSOLUTE_ERROR,
        "min_valid_pixels_large_area": MIN_VALID_PIXELS_LARGE_AREA,
        "confusion_pairs_gt_pred": CONFUSION_PAIRS,
        "visualization_colors_rgb": COLORS,
        "per_slide_results": per_slide_results,
        "saved_class_visuals": saved_class_visuals,
        "saved_confusion_visuals": saved_confusion_visuals,
        "saved_large_area_visuals": saved_large_area_visuals,
        "saved_absolute_error_visuals": saved_absolute_error_visuals,
    }

    with open(out_json, "w") as f:
        json.dump(result, f, indent=4)

    print("\nSaved Dice JSON:", out_json)

    if SAVE_VISUALS:
        print("Saved visual panels in:", VIS_OUT_DIR)


if __name__ == "__main__":
    main()
