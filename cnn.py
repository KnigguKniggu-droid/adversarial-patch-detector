"""
A TRAINED CNN patch-vs-clean classifier — the learned counterpart to the heuristic detector.

- NEGATIVE class: real natural-image patches (CIFAR-10, 32x32).
- POSITIVE class: adversarial-style patches (dense, varied-frequency, high-saturation) — the
  visual signature of printed adversarial patches.

The CNN learns the distinction from data instead of a hand-set threshold. It's validated two
ways: (1) precision/recall on a held-out test split, and (2) the honest test — used as a
sliding-window detector, does it localize a *real gradient-optimized* adversarial patch
(generated against MobileNetV2 in vision.py) it never trained on?

    python cli.py train          # train + save patch_cnn.pt, print metrics
    python cli.py defend --cnn   # run the defense pipeline using the trained CNN
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

HERE = Path(__file__).parent
DATA = HERE / "data"
WEIGHTS = HERE / "patch_cnn.pt"


class PatchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16x16
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


def _generate_positives(n, rng):
    """Adversarial-style 32x32 patches: varied frequency + saturation."""
    out = []
    for _ in range(n):
        scale = int(rng.choice([1, 1, 2, 4, 8]))      # vary spatial frequency
        small = rng.integers(0, 256, (max(4, 32 // scale), max(4, 32 // scale), 3), dtype=np.uint8)
        p = np.array(Image.fromarray(small).resize((32, 32), Image.NEAREST))
        if rng.random() < 0.5:                         # sometimes boost saturation/contrast
            p = np.clip(p.astype(np.float32) * rng.uniform(1.0, 1.7), 0, 255).astype(np.uint8)
        out.append(p)
    return out


def _load_negatives(n, rng):
    from torchvision import datasets
    ds = datasets.CIFAR10(root=str(DATA), train=True, download=True)
    idxs = rng.choice(len(ds), size=n, replace=False)
    return [np.asarray(ds[int(i)][0]) for i in idxs]


def _build_dataset(n_per_class=3000, seed=0):
    rng = np.random.default_rng(seed)
    neg = _load_negatives(n_per_class, rng)
    pos = _generate_positives(n_per_class, rng)
    X = np.stack(neg + pos).astype(np.float32) / 255.0          # (2n,32,32,3)
    y = np.array([0] * n_per_class + [1] * n_per_class)
    X = torch.from_numpy(X).permute(0, 3, 1, 2)
    y = torch.from_numpy(y).long()
    perm = torch.randperm(len(y))
    return X[perm], y[perm]


def train(n_per_class=3000, epochs=5, seed=0):
    torch.manual_seed(seed)
    X, y = _build_dataset(n_per_class, seed)
    n_test = len(y) // 5
    Xtr, ytr, Xte, yte = X[n_test:], y[n_test:], X[:n_test], y[:n_test]
    model = PatchCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    bs = 128
    for ep in range(epochs):
        model.train()
        for i in range(0, len(Xtr), bs):
            opt.zero_grad()
            out = model(Xtr[i:i + bs])
            loss = lossf(out, ytr[i:i + bs])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xte).argmax(1)
    tp = int(((pred == 1) & (yte == 1)).sum()); fp = int(((pred == 1) & (yte == 0)).sum())
    fn = int(((pred == 0) & (yte == 1)).sum())
    acc = float((pred == yte).float().mean())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    torch.save(model.state_dict(), WEIGHTS)
    return {"test_n": len(yte), "accuracy": acc, "precision": prec, "recall": rec}


def load_model():
    model = PatchCNN()
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.eval()
    return model


def detect_cnn(image_np, model=None, win=32, stride=16, thresh=0.8):
    """Slide the trained CNN over an image; flag windows it classifies as a patch."""
    if model is None:
        model = load_model()
    H, W, _ = image_np.shape
    regions = []
    tiles, coords = [], []
    for y in range(0, max(1, H - win + 1), stride):
        for x in range(0, max(1, W - win + 1), stride):
            tiles.append(image_np[y:y + win, x:x + win] / 255.0)
            coords.append((x, y))
    if not tiles:
        return {"bounding_box": None, "num_flagged_windows": 0}
    batch = torch.from_numpy(np.stack(tiles)).permute(0, 3, 1, 2).float()
    with torch.no_grad():
        probs = model(batch).softmax(1)[:, 1].numpy()
    for (x, y), p in zip(coords, probs):
        if p >= thresh:
            regions.append({"x": x, "y": y, "w": win, "h": win, "prob": round(float(p), 2)})
    bb = None
    if regions:
        x0 = min(r["x"] for r in regions); y0 = min(r["y"] for r in regions)
        x1 = max(r["x"] + r["w"] for r in regions); y1 = max(r["y"] + r["h"] for r in regions)
        bb = {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}
    return {"bounding_box": bb, "num_flagged_windows": len(regions), "regions": regions}
