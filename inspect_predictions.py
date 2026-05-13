from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# BEETLE label mapping
# 1 = Other
# 2 = Non-invasive epithelium
# 3 = Invasive epithelium
# 4 = Necrosis
COLORS = {
    0: (0, 0, 0),          # optional background / unknown
    1: (0, 220, 220),      # Other = cyan
    2: (255, 230, 0),      # Non-invasive epithelium = yellow
    3: (255, 0, 150),      # Invasive epithelium = magenta/pink
    4: (0, 0, 180),        # Necrosis = dark blue
}

CLASS_NAMES = {
    0: "Background/unknown",
    1: "Other",
    2: "Non-invasive epithelium",
    3: "Invasive epithelium",
    4: "Necrosis",
}


def read_image(path: Path) -> np.ndarray:
    """Read RGB image or label mask."""
    img = Image.open(path)
    arr = np.array(img)

    # If prediction was saved as RGB but actually contains same value in each channel,
    # reduce it to one channel.
    if arr.ndim == 3:
        if arr.shape[2] >= 3 and np.all(arr[..., 0] == arr[..., 1]) and np.all(arr[..., 1] == arr[..., 2]):
            arr = arr[..., 0]

    return arr


def normalize_rgb_for_display(img: np.ndarray) -> np.ndarray:
    """Convert image to uint8 RGB for display."""
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    if img.shape[-1] > 3:
        img = img[..., :3]

    img = img.astype(np.float32)

    # Robust percentile scaling, useful for TIFFs that are not uint8.
    lo, hi = np.percentile(img, [1, 99])
    if hi > lo:
        img = (img - lo) / (hi - lo)
    else:
        img = img / max(img.max(), 1)

    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert label mask with values 1-4 to RGB color image."""
    if mask.ndim == 3:
        mask = mask[..., 0]

    mask = mask.astype(np.uint8)
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)

    for label, rgb in COLORS.items():
        color[mask == label] = rgb

    return color


def make_overlay(image_rgb: np.ndarray, mask_color: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay colorized mask on original image, ignoring label 0/black pixels."""
    image_rgb = image_rgb.astype(np.float32)
    mask_color = mask_color.astype(np.float32)

    mask_nonzero = np.any(mask_color != 0, axis=-1, keepdims=True)

    overlay = image_rgb.copy()
    overlay[mask_nonzero[..., 0]] = (
        (1 - alpha) * image_rgb[mask_nonzero[..., 0]]
        + alpha * mask_color[mask_nonzero[..., 0]]
    )

    return np.clip(overlay, 0, 255).astype(np.uint8)


def inspect_pair(image_path: Path, pred_path: Path, out_dir: Path, alpha: float = 0.45):
    image = read_image(image_path)
    pred = read_image(pred_path)

    image_rgb = normalize_rgb_for_display(image)
    pred_color = colorize_mask(pred)
    overlay = make_overlay(image_rgb, pred_color, alpha=alpha)

    unique, counts = np.unique(pred, return_counts=True)

    print(f"\nPrediction: {pred_path.name}")
    print(f"Image shape: {image.shape}")
    print(f"Prediction shape: {pred.shape}")
    print("Unique prediction values:")
    for value, count in zip(unique, counts):
        name = CLASS_NAMES.get(int(value), "Unknown label")
        percentage = 100 * count / pred.size
        print(f"  {int(value):>3} | {name:<25} | {count:>10} pixels | {percentage:6.2f}%")

    if image_rgb.shape[:2] != pred.shape[:2]:
        print("WARNING: image and prediction sizes do not match!")
        print(f"  image: {image_rgb.shape[:2]}")
        print(f"  pred:  {pred.shape[:2]}")

    out_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem
    Image.fromarray(pred_color).save(out_dir / f"{stem}_pred_color.png")
    Image.fromarray(overlay).save(out_dir / f"{stem}_overlay.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(pred_color)
    axes[1].set_title("Colorized prediction")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(out_dir / f"{stem}_inspection.png", dpi=150)
    plt.close(fig)


def main():
    # Hardcoded paths
    image_dir = Path(r"D:\AIMI---BEETLE-Project-Phase\data\images\images\evaluation\rois")
    pred_dir = Path(r"C:\Uni\Jaar 4 - Master jaar 1\Semester 2\AI in Medical Imaging\Project phase\AIMI---BEETLE-Project-Phase\inference")
    out_dir = Path("inspection_outputs")
    alpha = 0.45

    image_paths = []
    for ext in ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]:
        image_paths.extend(image_dir.glob(ext))

    image_paths = sorted(image_paths)

    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    for image_path in image_paths:
        pred_path = pred_dir / image_path.name

        # Sometimes original is .tif but prediction is .png
        if not pred_path.exists():
            pred_path = pred_dir / f"{image_path.stem}.png"

        if not pred_path.exists():
            print(f"Skipping {image_path.name}: no matching prediction found")
            continue

        inspect_pair(image_path, pred_path, out_dir, alpha=alpha)


if __name__ == "__main__":
    main()