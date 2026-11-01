"""Smoke test: synthetic logits so the analysis pipeline can run without data or a GPU."""
import numpy as np, os
from .common import OUT, MODELS
rng = np.random.default_rng(0); os.makedirs(f"{OUT}/logits", exist_ok=True)
for tag in ["std", "robust"]:
    for seed in [0, 1]:
        ys = {s: rng.integers(0, 5, n) for s, n in [("val", 600), ("testA", 800), ("testB", 800)]}  # shared labels
        for m in MODELS:
            d = {}
            for s, noise in [("val", 1.2), ("testA", 1.2), ("testB", 2.2)]:
                y = ys[s]; n = len(y); z = rng.normal(0, noise, (n, 5)); z[np.arange(n), y] += 2.5
                d[f"{s}_logits"], d[f"{s}_y"] = z * 2, y
            np.savez(f"{OUT}/logits/{m}_{tag}_s{seed}.npz", **d)
