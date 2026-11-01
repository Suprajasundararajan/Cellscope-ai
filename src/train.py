"""Train one model (transfer learning) and dump logits for val/testA/testB.
Usage: python -m src.train --model mobilenetv2 --seed 0 --aug std|robust
"""
import argparse, os, json
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
from .common import CLASSES, OUT, seed_everything, get_device
from .data import WBCDataset, build_tf
from .models import build_model, count_params
from .infer import collect_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--aug", default="std", choices=["std", "robust"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--subset", type=int, default=0, help="debug: use N images per split")
    ap.add_argument("--resume", action="store_true", help="resume from the latest epoch checkpoint")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--num-threads", type=int, default=0, help="CPU threads; 0 keeps the PyTorch default")
    a = ap.parse_args()

    seed_everything(a.seed); dev = get_device(a.device)
    if a.num_threads > 0 and dev.type == "cpu":
        torch.set_num_threads(a.num_threads)
    man = pd.read_csv(a.manifest)
    if a.subset:
        man = pd.concat([d.sample(min(len(d), a.subset), random_state=0) for _, d in man.groupby("split")])
    tr, va = man[man.split == "train"], man[man.split == "val"]
    tag = f"{a.model}_{a.aug}_s{a.seed}"
    os.makedirs(f"{OUT}/ckpt", exist_ok=True); os.makedirs(f"{OUT}/logits", exist_ok=True)

    dl = DataLoader(
        WBCDataset(tr, build_tf(True, a.aug == "robust")), batch_size=a.bs,
        shuffle=True, num_workers=a.workers, drop_last=True,
        pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0,
    )
    cnt = np.bincount(tr.label, minlength=len(CLASSES)).astype(float)
    w = torch.tensor(cnt.sum() / (len(CLASSES) * np.maximum(cnt, 1)), dtype=torch.float32).to(dev)
    crit = nn.CrossEntropyLoss(weight=w)  # class-weighted loss for rare classes (e.g. basophils)

    model = build_model(a.model).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(dl))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    state_path = f"{OUT}/ckpt/{tag}_resume.pt"
    best, hist, start_epoch = -1, [], 0
    if a.resume and os.path.exists(state_path):
        state = torch.load(state_path, map_location=dev)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        best = state["best"]
        hist = state["history"]
        start_epoch = state["epoch"] + 1
        print(f"Resuming {tag} from epoch {start_epoch}/{a.epochs}")

    for ep in range(start_epoch, a.epochs):
        model.train(); tot = 0
        for x, y in tqdm(dl, desc=f"{tag} ep{ep}"):
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        lg, _ = collect_logits(model, va, dev, workers=a.workers)
        f1 = f1_score(va.label, lg.argmax(1), average="macro")  # model selection on in-domain val only
        hist.append({"epoch": ep, "loss": tot / len(dl), "val_macro_f1": f1}); print(hist[-1])
        if f1 > best:
            best = f1; torch.save(model.state_dict(), f"{OUT}/ckpt/{tag}.pt")
        torch.save(
            {"epoch": ep, "model": model.state_dict(), "optimizer": opt.state_dict(),
             "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
             "best": best, "history": hist},
            state_path,
        )

    model.load_state_dict(torch.load(f"{OUT}/ckpt/{tag}.pt", map_location=dev))
    save, ms = {}, {}
    for s in ["val", "testA", "testB"]:
        d = man[man.split == s]
        save[f"{s}_logits"], ms[s] = collect_logits(model, d, dev, workers=a.workers)
        save[f"{s}_y"] = d.label.values
    np.savez(f"{OUT}/logits/{tag}.npz", **save)
    json.dump({"hist": hist, "params_M": count_params(model), "ms_per_img": ms,
               "device": str(dev), "args": vars(a), "classes": CLASSES},
              open(f"{OUT}/logits/{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
