#!/usr/bin/env python3
"""
adversarial-patch-detector — find printed-patch attacks in images.

Usage:
    python cli.py demo                 # generate a clean + patched image, detect, save overlays
    python cli.py scan path/to/img.png # run on a real image; saves an overlay next to it
"""
import argparse
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from PIL import Image

import detector
import synth

OUT = Path(__file__).parent / "outputs"


def _report(name, result):
    print(f"\n[{name}] -> {result['verdict']}")
    print(f"  flagged windows : {result['num_flagged_windows']}")
    print(f"  median energy   : {result['global_median_energy']}")
    if result["bounding_box"]:
        bb = result["bounding_box"]
        print(f"  suspected region: x={bb['x']} y={bb['y']} {bb['w']}x{bb['h']}")
    for r in result["top_regions"][:3]:
        print(f"    · ({r['x']},{r['y']}) energy={r['energy_ratio']}x median, sat={r['saturation']}")


def cmd_demo(args):
    OUT.mkdir(exist_ok=True)
    clean = synth.make_scene(seed=0)
    patched, truth = synth.add_patch(clean, seed=3)

    res_clean, _ = detector.detect(clean)
    res_patch, _ = detector.detect(patched)

    _report("clean image", res_clean)
    _report("patched image", res_patch)
    print(f"\n  ground-truth patch was at x={truth['x']} y={truth['y']} "
          f"{truth['w']}x{truth['h']}")

    Image.fromarray(clean.astype(np.uint8)).save(OUT / "clean.png")
    Image.fromarray(patched.astype(np.uint8)).save(OUT / "patched.png")
    detector.save_overlay(patched, res_patch, OUT / "patched_detected.png")
    print(f"\n  saved images to {OUT}\\  (clean.png, patched.png, patched_detected.png)")


def cmd_scan(args):
    img = detector.load_image(args.path)
    result, _ = detector.detect(img)
    _report(Path(args.path).name, result)
    out = Path(args.path).with_name(Path(args.path).stem + "_detected.png")
    detector.save_overlay(img, result, out)
    print(f"  overlay saved -> {out}")


def _ensure_sample():
    """Grab a real photo (ImageNet-classifiable) for the defense demo."""
    samp = Path(__file__).parent / "samples" / "dog.jpg"
    samp.parent.mkdir(exist_ok=True)
    if not samp.exists():
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg", samp)
    return str(samp)


def cmd_defend(args):
    """Real pipeline: classify -> adversarial patch fools the model -> detect -> mask -> recover."""
    import vision  # heavy (torch) — imported only when needed
    OUT.mkdir(exist_ok=True)
    path = args.path or _ensure_sample()
    print(f"Image: {path}\n")

    x = vision.load_image_01(path)
    clean = vision.classify(x)
    print(f"  1. Clean image      -> {clean['label']} ({clean['conf']*100:.1f}%)")

    print("  2. Optimizing an adversarial patch to fool the model... (a few seconds)")
    x_adv, true_box = vision.make_adversarial_patch(x)
    attacked = vision.classify(x_adv)
    fooled = attacked["idx"] != clean["idx"]
    print(f"     After patch      -> {attacked['label']} ({attacked['conf']*100:.1f}%)  "
          f"[{'FOOLED' if fooled else 'not fooled'}]")

    adv_np = vision.to_numpy_image(x_adv)
    result, _ = detector.detect(adv_np, ratio_thresh=3.0)
    det_box = result["bounding_box"]
    overlap = vision.iou(det_box, true_box)
    print(f"  3. Detector found patch at {det_box}  (IoU {overlap:.2f} vs true patch)")

    if det_box:
        recovered = vision.classify(vision.mask_region(x_adv, det_box))
        ok = recovered["idx"] == clean["idx"]
        print(f"  4. Mask + reclassify -> {recovered['label']} ({recovered['conf']*100:.1f}%)  "
              f"[{'RECOVERED' if ok else 'still wrong'}]")
    else:
        print("  4. No region detected — nothing to mask.")

    Image.fromarray(adv_np.astype(np.uint8)).save(OUT / "adversarial.png")
    detector.save_overlay(adv_np, result, OUT / "adversarial_detected.png")
    print(f"\n  saved: {OUT}\\adversarial.png, adversarial_detected.png")


def main():
    p = argparse.ArgumentParser(description="Adversarial patch detector")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("demo")
    s = sub.add_parser("scan")
    s.add_argument("path")
    d = sub.add_parser("defend", help="real model + real adversarial patch + detect/mask/recover")
    d.add_argument("path", nargs="?", default=None, help="image (defaults to a sample photo)")
    args = p.parse_args()
    if args.cmd == "scan":
        cmd_scan(args)
    elif args.cmd == "defend":
        cmd_defend(args)
    else:
        cmd_demo(args)


if __name__ == "__main__":
    main()
