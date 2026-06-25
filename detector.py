"""
Adversarial patch detector — a model-free baseline.

Adversarial patches (the printed stickers that fool a self-driving car's vision model
into misreading a stop sign) are physically unusual: a small, localized region of very
HIGH-FREQUENCY, often HIGH-SATURATION pixels that doesn't match the smooth statistics of
the surrounding scene. This detector slides a window over the image and flags regions
whose local gradient energy is far above the image's own median — no trained network
needed, fully explainable, runs anywhere.

Ties directly to AV-perception security research (local/physical patch attacks on VLMs).
"""
import numpy as np
from PIL import Image


def load_image(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def _gradient_energy(gray):
    """High-frequency content per pixel (|dx| + |dy|)."""
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, :-1] = np.abs(np.diff(gray, axis=1))
    gy[:-1, :] = np.abs(np.diff(gray, axis=0))
    return gx + gy


def _saturation(img):
    mx = img.max(axis=2)
    mn = img.min(axis=2)
    return (mx - mn) / (mx + 1e-6)


def _merge_bbox(regions):
    if not regions:
        return None
    x0 = min(r["x"] for r in regions)
    y0 = min(r["y"] for r in regions)
    x1 = max(r["x"] + r["w"] for r in regions)
    y1 = max(r["y"] + r["h"] for r in regions)
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def detect(img, win=32, stride=16, ratio_thresh=4.0, sat_weight=0.5):
    """
    Returns (result_dict, heatmap). A region is flagged when its local gradient energy
    exceeds `ratio_thresh` x the image-wide median energy.
    """
    gray = img.mean(axis=2)
    energy = _gradient_energy(gray)
    sat = _saturation(img)
    H, W = gray.shape
    med = float(np.median(energy)) + 1e-6

    regions = []
    heat = np.zeros((H, W), dtype=np.float32)
    for y in range(0, max(1, H - win + 1), stride):
        for x in range(0, max(1, W - win + 1), stride):
            e = float(energy[y:y + win, x:x + win].mean())
            s = float(sat[y:y + win, x:x + win].mean())
            ratio = e / med
            score = ratio * (1 + sat_weight * s)
            heat[y:y + win, x:x + win] = np.maximum(heat[y:y + win, x:x + win], score)
            if ratio >= ratio_thresh:
                regions.append({"x": x, "y": y, "w": win, "h": win,
                                "energy_ratio": round(ratio, 2),
                                "saturation": round(s, 2), "score": round(score, 2)})

    result = {
        "verdict": "ADVERSARIAL PATCH LIKELY" if regions else "no patch detected",
        "num_flagged_windows": len(regions),
        "global_median_energy": round(med, 3),
        "bounding_box": _merge_bbox(regions),
        "top_regions": sorted(regions, key=lambda r: -r["score"])[:5],
    }
    return result, heat


def save_overlay(img, result, out_path):
    """Save the image with a red box around the suspected patch region."""
    from PIL import ImageDraw
    pic = Image.fromarray(img.astype(np.uint8))
    bb = result.get("bounding_box")
    if bb:
        d = ImageDraw.Draw(pic)
        d.rectangle([bb["x"], bb["y"], bb["x"] + bb["w"], bb["y"] + bb["h"]],
                    outline=(255, 0, 0), width=4)
    pic.save(out_path)
    return out_path
