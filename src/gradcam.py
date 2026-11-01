"""Grad-CAM per base model + quantitative nucleus-overlap check.
Usage: python -m src.gradcam --model resnet50 --tag std_s0 --n 200
Nucleus mask is a heuristic (Otsu on the green channel: nuclei are dark in G). Granule-rich cells
(basophils/eosinophils) also darken, so treat the metric as a sanity check, not ground truth.
"""
import argparse, os
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from .common import CLASSES, OUT, get_device
from .data import WBCDataset, build_tf, MEAN, STD
from .models import build_model, cam_layer


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.a, self.g = model, None, None
        self.handle = layer.register_forward_hook(self._fwd)

    def _fwd(self, m, i, o):
        self.a = o
        if torch.is_grad_enabled() and not torch.is_inference_mode_enabled() and o.requires_grad:
            o.register_hook(lambda g: setattr(self, "g", g))

    def close(self):
        self.handle.remove()

    def __call__(self, x):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x); pred = logits.argmax(1)
        logits.gather(1, pred[:, None]).sum().backward()
        w = self.g.mean((2, 3), keepdim=True)
        cam = F.relu((w * self.a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        mn = cam.amin((1, 2), keepdim=True); mx = cam.amax((1, 2), keepdim=True)
        return ((cam - mn) / (mx - mn + 1e-8)).detach().cpu().numpy(), logits.detach().cpu()


def denorm(x):
    m = torch.tensor(MEAN)[:, None, None]; s = torch.tensor(STD)[:, None, None]
    return (x.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def nucleus_mask(img):
    g = img[..., 1]; return g < threshold_otsu(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tag", default="std_s0")
    ap.add_argument("--manifest", default="manifest.csv"); ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args(); dev = get_device()
    model = build_model(a.model, pretrained=False)
    model.load_state_dict(torch.load(f"{OUT}/ckpt/{a.model}_{a.tag}.pt", map_location="cpu")); model.to(dev).eval()
    cam = GradCAM(model, cam_layer(model, a.model))
    man = pd.read_csv(a.manifest); os.makedirs(f"{OUT}/gradcam", exist_ok=True)
    rows, examples = [], {}
    for dom in ["testA", "testB"]:
        sub = man[man.split == dom]; d = sub.sample(min(a.n, len(sub)), random_state=0)
        ds = WBCDataset(d, build_tf(False))
        for i in range(0, len(ds), 32):
            idx = range(i, min(i + 32, len(ds)))
            xb = torch.stack([ds[j][0] for j in idx]).to(dev); yb = [ds[j][1] for j in idx]
            cams, lg = cam(xb); pr = lg.argmax(1).numpy()
            for j in range(len(yb)):
                img = denorm(xb[j]); mk = nucleus_mask(img); c = cams[j]
                inside = float(c[mk].sum() / (c.sum() + 1e-8)); area = float(mk.mean())
                ok = bool(pr[j] == yb[j])
                rows.append({"domain": dom, "correct": ok, "true": CLASSES[yb[j]], "pred": CLASSES[pr[j]],
                             "cam_in_nucleus": inside, "nucleus_area": area, "focus_ratio": inside / (area + 1e-8)})
                ex = examples.setdefault((dom, ok), [])
                if len(ex) < 4:
                    ex.append((img, c, yb[j], pr[j]))
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/gradcam/{a.model}_{a.tag}_overlap.csv", index=False)
    print(df.groupby(["domain", "correct"])[["cam_in_nucleus", "focus_ratio"]].mean().round(3))

    keys = sorted(examples)
    fig, ax = plt.subplots(len(keys), 8, figsize=(16, 2.2 * len(keys) + 1), squeeze=False)
    for r, key in enumerate(keys):
        for k, (img, c, t, p) in enumerate(examples[key]):
            ax[r, 2 * k].imshow(img); ax[r, 2 * k + 1].imshow(img); ax[r, 2 * k + 1].imshow(c, cmap="jet", alpha=.45)
            ax[r, 2 * k].set_title(f"{key[0]} {'OK' if key[1] else 'ERR'}\nT:{CLASSES[t][:4]} P:{CLASSES[p][:4]}", fontsize=8)
    for x in ax.ravel(): x.axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/gradcam/{a.model}_{a.tag}_examples.png", dpi=130)


if __name__ == "__main__":
    main()
