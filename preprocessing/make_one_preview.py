from pathlib import Path
import argparse
import numpy as np
import tifffile
from PIL import Image

COLORS = {
    0: (0, 0, 0),
    1: (0, 255, 255),     # other
    2: (255, 255, 0),     # non-invasive
    3: (255, 0, 255),     # invasive
    4: (0, 0, 255),       # necrosis
}

def read_lowres(path, is_mask=False, max_size=2048):
    with tifffile.TiffFile(str(path)) as tif:
        s = tif.series[0]
        levels = getattr(s, "levels", None) or [s]
        arr = levels[-1].asarray()

    if arr.ndim == 3 and is_mask:
        arr = arr[:, :, 0]
    if arr.ndim == 2 and not is_mask:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and not is_mask and arr.shape[-1] > 3:
        arr = arr[:, :, :3]

    arr = arr.astype(np.uint8)
    h, w = arr.shape[:2]
    scale = max(h / max_size, w / max_size, 1)
    if scale > 1:
        size = (int(w / scale), int(h / scale))
        resample = Image.NEAREST if is_mask else Image.BILINEAR
        arr = np.asarray(Image.fromarray(arr).resize(size, resample))

    return arr

def colorize_mask(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label, color in COLORS.items():
        rgb[mask == label] = color
    return rgb

def overlay(image, mask_rgb, alpha=0.35):
    return ((1 - alpha) * image + alpha * mask_rgb).astype(np.uint8)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-size", type=int, default=2048)
    args = p.parse_args()

    image = read_lowres(args.image, is_mask=False, max_size=args.max_size)
    mask = read_lowres(args.mask, is_mask=True, max_size=args.max_size)

    if image.shape[:2] != mask.shape[:2]:
        mask = np.asarray(
            Image.fromarray(mask.astype(np.uint8)).resize(
                (image.shape[1], image.shape[0]),
                Image.NEAREST,
            )
        )

    mask_rgb = colorize_mask(mask)
    overlay_rgb = overlay(image, mask_rgb)

    h, w = image.shape[:2]
    canvas = Image.new("RGB", (w * 3, h), (255, 255, 255))
    canvas.paste(Image.fromarray(image), (0, 0))
    canvas.paste(Image.fromarray(mask_rgb), (w, 0))
    canvas.paste(Image.fromarray(overlay_rgb), (w * 2, 0))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print("saved", args.out)

if __name__ == "__main__":
    main()
