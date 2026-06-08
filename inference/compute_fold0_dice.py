import csv
from pathlib import Path
import numpy as np
import multiresolutionimageinterface as mir

CSV = Path("/home/tijnveldwijk/fold0_validation_inference_inputs.csv")
PRED_DIR = Path("/vol/csedu-nobackup/course/IMC037_aimi/group14/nnunet/tijn/pathology/validation_inference/DiceFocal_fold0")

CLASSES = {
    1: "other",
    2: "non-invasive epithelium",
    3: "invasive epithelium",
    4: "necrosis",
}

TILE = 2048

reader = mir.MultiResolutionImageReader()

tp = {c: 0 for c in CLASSES}
fp = {c: 0 for c in CLASSES}
fn = {c: 0 for c in CLASSES}

def open_img(path):
    img = reader.open(str(path))
    if img is None or not img.valid():
        raise RuntimeError(f"Could not open: {path}")
    return img

def patch_to_2d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.int64)

with open(CSV) as f:
    rows = list(csv.DictReader(f))

for i, row in enumerate(rows):
    wsi = Path(row["wsi_path"])
    gt_path = Path(row["mask_path"])
    pred_path = PRED_DIR / f"{wsi.stem}_nnunet.tif"

    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction: {pred_path}")

    print(f"[{i+1}/{len(rows)}] {wsi.stem}", flush=True)

    gt = open_img(gt_path)
    pred = open_img(pred_path)

    gt_w, gt_h = gt.getLevelDimensions(0)
    pr_w, pr_h = pred.getLevelDimensions(0)

    if (gt_w, gt_h) != (pr_w, pr_h):
        raise RuntimeError(f"Shape mismatch for {wsi.stem}: GT {(gt_w, gt_h)} vs pred {(pr_w, pr_h)}")

    for y in range(0, gt_h, TILE):
        h = min(TILE, gt_h - y)
        for x in range(0, gt_w, TILE):
            w = min(TILE, gt_w - x)

            gt_patch = patch_to_2d(gt.getUCharPatch(x, y, w, h, 0))
            pred_patch = patch_to_2d(pred.getUCharPatch(x, y, w, h, 0))

            valid = gt_patch != 0

            if not valid.any():
                continue

            for c in CLASSES:
                gt_c = (gt_patch == c) & valid
                pred_c = (pred_patch == c) & valid

                tp[c] += int((gt_c & pred_c).sum())
                fp[c] += int((~gt_c & pred_c).sum())
                fn[c] += int((gt_c & ~pred_c).sum())

print("\n=== Fold 0 Dice, aggregated over all annotated pixels ===")
micro_tp = sum(tp.values())
micro_fp = sum(fp.values())
micro_fn = sum(fn.values())

for c, name in CLASSES.items():
    denom = 2 * tp[c] + fp[c] + fn[c]
    dice = 2 * tp[c] / denom if denom > 0 else float("nan")
    print(f"{name}: {dice:.4f}")

micro_dice = 2 * micro_tp / (2 * micro_tp + micro_fp + micro_fn)
macro_dice = np.nanmean([
    2 * tp[c] / (2 * tp[c] + fp[c] + fn[c])
    for c in CLASSES
    if (2 * tp[c] + fp[c] + fn[c]) > 0
])

print(f"\nmacro mean Dice: {macro_dice:.4f}")
print(f"micro/overall Dice: {micro_dice:.4f}")
