"""
Real adversarial-patch pipeline against a real pretrained image classifier.

This turns the detector from "flags high-frequency blobs in synthetic images" into a real
adversarial-robustness experiment:

  1. classify a real photo with a pretrained MobileNetV2 (ImageNet)
  2. OPTIMIZE an adversarial patch (gradient descent) that genuinely flips the prediction
  3. run the detector to localize the patch
  4. MASK the detected region and re-classify -> measure recovered accuracy

Step 4 ("recovered accuracy under attack") is the metric AV-perception robustness papers
actually report. Needs: torch, torchvision (CPU is fine).
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

_WEIGHTS = MobileNet_V2_Weights.DEFAULT
_MODEL = mobilenet_v2(weights=_WEIGHTS).eval()
_CATS = _WEIGHTS.meta["categories"]
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_image_01(path):
    """Load any image -> (1,3,224,224) tensor in [0,1]."""
    img = Image.open(path).convert("RGB").resize((224, 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def to_numpy_image(x01):
    """(1,3,224,224) [0,1] tensor -> (224,224,3) float[0,255] for the detector."""
    a = x01.detach().clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255.0
    return a.astype(np.float32)


def classify(x01):
    with torch.no_grad():
        logits = _MODEL((x01 - _MEAN) / _STD)
        probs = logits.softmax(1)
        conf, idx = probs.max(1)
    return {"label": _CATS[idx.item()], "conf": conf.item(), "idx": idx.item()}


def make_adversarial_patch(x01, size=64, pos=(80, 80), steps=200, lr=0.04):
    """Gradient-optimize a patch in the given region until it fools the classifier."""
    true_idx = classify(x01)["idx"]
    px, py = pos
    patch = torch.rand(1, 3, size, size, requires_grad=True)
    opt = torch.optim.Adam([patch], lr=lr)
    target = torch.tensor([true_idx])
    for _ in range(steps):
        x = x01.clone()
        x[:, :, py:py + size, px:px + size] = patch.clamp(0, 1)
        logits = _MODEL((x - _MEAN) / _STD)
        loss = -F.cross_entropy(logits, target)   # maximize loss on the true class
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            patch.clamp_(0, 1)
    x_adv = x01.clone()
    x_adv[:, :, py:py + size, px:px + size] = patch.detach().clamp(0, 1)
    return x_adv, {"x": px, "y": py, "w": size, "h": size}


def mask_region(x01, box, fill=0.5):
    """Neutralize a detected region (gray fill) before re-classifying."""
    x = x01.clone()
    x[:, :, box["y"]:box["y"] + box["h"], box["x"]:box["x"] + box["w"]] = fill
    return x


def iou(a, b):
    if not a or not b:
        return 0.0
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    ix = max(0, min(ax2, bx2) - max(a["x"], b["x"]))
    iy = max(0, min(ay2, by2) - max(a["y"], b["y"]))
    inter = ix * iy
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union else 0.0
