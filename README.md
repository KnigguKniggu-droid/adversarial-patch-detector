# Adversarial Patch Detector

Detects **adversarial patch attacks** — the printed stickers/perturbations that fool a
vision model (e.g. an autonomous vehicle misreading a stop sign). A model-free, fully
explainable baseline: it flags small image regions whose local high-frequency energy is
far above the rest of the scene.

Companion to the CAN-bus IDS in this portfolio — together they cover **two layers of the
AV attack surface**: perception (this) and in-vehicle network (CAN IDS).

## How it works
1. Compute per-pixel **gradient energy** (high-frequency content) and **saturation**.
2. Slide a window across the image; score each window by how far its energy exceeds the
   image-wide **median**.
3. Flag windows above a ratio threshold and merge them into a suspected patch box.

Adversarial patches are, by construction, dense high-frequency noise — they light up
against the smooth statistics of a real scene.

## Run it (numpy + Pillow, no dataset needed)
```bash
pip install numpy pillow
python cli.py demo                 # makes a clean + patched image, detects, saves overlays
python cli.py scan path/to/img.png # run on your own image; writes *_detected.png
```
`demo` writes `outputs/clean.png`, `outputs/patched.png`, and `outputs/patched_detected.png`
(the last with a red box around the detected patch).

## Honest scope & roadmap
- This is a **heuristic frequency/saturation** detector — a strong, explainable baseline.
  It will flag any dense high-frequency region, so a busy natural texture *could* trip it;
  it is not a trained classifier. Stated plainly, not hidden.
- **ML path:** train a small CNN patch-vs-clean classifier and compare precision/recall to
  this baseline (this baseline becomes your honest control).
- **Eval path:** evaluate on a real dataset (e.g. APRICOT) and report ROC/AP.
- **Robustness tie-in:** feed detections to a downstream model as a mask to measure
  recovered accuracy under attack — the metric AV-perception papers care about.
