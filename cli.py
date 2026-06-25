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


def main():
    p = argparse.ArgumentParser(description="Adversarial patch detector")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("demo")
    s = sub.add_parser("scan")
    s.add_argument("path")
    args = p.parse_args()
    if args.cmd == "scan":
        cmd_scan(args)
    else:
        cmd_demo(args)


if __name__ == "__main__":
    main()
