import time
import torch
from torch.utils.data import DataLoader
from .data import WBCDataset, build_tf


@torch.no_grad()
def collect_logits(model, df, device, bs=64, workers=2):
    model.eval().to(device)
    dl = DataLoader(
        WBCDataset(df, build_tf(False)), batch_size=bs, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
    )
    out, t0 = [], time.time()
    for x, _ in dl:
        out.append(model(x.to(device)).float().cpu())
    return torch.cat(out).numpy(), (time.time() - t0) / max(len(df), 1) * 1000  # logits, ms/img (incl. loading)
