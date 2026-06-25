"""
Synthetic test-image generator so the detector is runnable with zero external data.

- make_scene(): a smooth, low-frequency "natural-ish" background (gradients + soft blobs).
- add_patch(): paste a high-frequency, high-saturation noise block = a stand-in for a
  printed adversarial patch.

Real evaluation would use datasets like APRICOT or real stop-sign patch photos; this lets
you demo and unit-test the pipeline today.
"""
import numpy as np
from PIL import Image, ImageFilter


def make_scene(size=256, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    # smooth color gradient background
    r = 120 + 80 * (xx / size)
    g = 130 + 70 * (yy / size)
    b = 160 - 60 * (xx / size)
    img = np.stack([r, g, b], axis=2)
    # a few soft low-frequency blobs (still smooth = low gradient energy)
    for _ in range(3):
        cx, cy = rng.integers(0, size, 2)
        rad = rng.integers(size // 6, size // 3)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        img += np.stack([np.clip(40 - d * 40 / rad, 0, 40)] * 3, axis=2)
    pic = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2))
    return np.asarray(pic, dtype=np.float32)


def add_patch(img, size=44, pos=None, seed=1):
    rng = np.random.default_rng(seed)
    H, W, _ = img.shape
    out = img.copy()
    if pos is None:
        px = int(rng.integers(0, W - size))
        py = int(rng.integers(0, H - size))
    else:
        px, py = pos
    # high-frequency, high-saturation noise = adversarial-patch stand-in
    patch = rng.integers(0, 256, size=(size, size, 3)).astype(np.float32)
    out[py:py + size, px:px + size] = patch
    return out, {"x": px, "y": py, "w": size, "h": size}
