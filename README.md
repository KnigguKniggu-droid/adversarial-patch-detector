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

## Run it

**Detector demo (numpy + Pillow, no dataset):**
```bash
pip install numpy pillow
python cli.py demo                 # synthetic clean + patched image, detect, save overlays
python cli.py scan path/to/img.png # run on your own image; writes *_detected.png
```

**Real end-to-end defense against a REAL model** (`pip install torch torchvision`):
```bash
python cli.py defend               # uses a sample photo (or: defend path/to/img.png)
```
This is the real experiment (`vision.py`):
1. classify a real photo with **pretrained MobileNetV2** (ImageNet),
2. **gradient-optimize an adversarial patch** that genuinely flips the prediction,
3. run the detector to **localize** the patch,
4. **mask** the detected region and re-classify → measure **recovered accuracy**.

Example run (sample dog photo):
```
1. Clean image       -> Samoyed (16.6%)        [correct]
2. Adversarial patch -> jackfruit (97.6%)      [FOOLED — model confidently wrong]
3. Detector localizes the patch (IoU 0.44)
4. Mask + reclassify -> Samoyed (25.0%)        [RECOVERED]
```
That step-4 number — recovered accuracy after masking the detected patch — is the metric
AV-perception robustness papers actually report.

## Honest scope & roadmap
- The detector itself is a **heuristic frequency/saturation** baseline — explainable, no
  training. It can flag any dense high-frequency region, so busy natural texture *could*
  trip it. Stated plainly, not hidden.
- **Real model + real attack (done):** `vision.py` runs a pretrained classifier, optimizes
  a true adversarial patch that fools it, then measures recovery after masking the
  detected region. Not synthetic noise — a gradient attack on a real network.
- **Next:** train a small CNN patch-vs-clean classifier and compare precision/recall to this
  heuristic baseline; evaluate on a real dataset (e.g. APRICOT) and report ROC/AP.
