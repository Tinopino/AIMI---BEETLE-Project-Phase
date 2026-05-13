from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Disable PIL decompression bomb check for large WSI images
Image.MAX_IMAGE_PIXELS = None


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


def inspect_wsi(image_path: Path, out_dir: Path):
    """Inspect a single WSI image and save downsampled preview."""
    print(f"\nWSI: {image_path.name}")
    
    # Open image to get size info
    with Image.open(image_path) as img:
        original_shape = img.size
        print(f"Original size: {original_shape[0]} × {original_shape[1]} pixels")
        
        # Create a downsampled preview (max 2048 pixels on longest side)
        max_dim = 2048
        scale = min(1.0, max_dim / max(original_shape))
        preview_size = (int(original_shape[0] * scale), int(original_shape[1] * scale))
        
        print(f"Creating preview at {preview_size[0]} × {preview_size[1]} pixels...")
        preview = img.resize(preview_size, Image.Resampling.LANCZOS)
    
    preview_array = np.array(preview)
    preview_rgb = normalize_rgb_for_display(preview_array)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    
    # Save the downsampled preview
    Image.fromarray(preview_rgb).save(out_dir / f"{stem}_preview.png")
    
    # Create and save a visualization figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(preview_rgb)
    ax.set_title(f"WSI Preview: {image_path.name}\n(Original: {original_shape[0]}×{original_shape[1]} → Preview: {preview_size[0]}×{preview_size[1]})")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_dir / f"{stem}_view.png", dpi=100, bbox_inches='tight')
    plt.close(fig)


def main():
    # Hardcoded paths
    image_dir = Path(r"D:\AIMI---BEETLE-Project-Phase\data\images\images\development\wsis")
    out_dir = Path("inspection_outputs_wsis")

    image_paths = []
    for ext in ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]:
        image_paths.extend(image_dir.glob(ext))

    image_paths = sorted(image_paths)

    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    print(f"Found {len(image_paths)} WSI images")

    for image_path in image_paths:
        inspect_wsi(image_path, out_dir)

    print(f"\n✓ Inspection complete! Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
