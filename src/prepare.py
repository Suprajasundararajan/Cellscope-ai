"""Build manifest.csv: scan Raabin-WBC, hash images, remove near-duplicates, make a leakage-aware val split.

Splits written: train / val (both from Raabin 'Train') and testA / testB (official test sets).
Raabin file names carry no patient/smear IDs, so we approximate group-level separation by
clustering near-duplicate images (perceptual hash) and splitting by cluster.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, imagehash
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
from .data import find_split_dir, scan_split

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def phash64(path):
    return np.uint64(int(str(imagehash.phash(Image.open(path).convert("RGB"))), 16))


def hamming(h, H):
    x = np.ascontiguousarray(np.bitwise_xor(H, h).astype(np.uint64))
    return _POP[x.view(np.uint8).reshape(-1, 8)].sum(1)


def cluster(hashes, thr):
    parent = list(range(len(hashes)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i in tqdm(range(len(hashes) - 1), desc="clustering"):
        js = np.where(hamming(hashes[i], hashes[i + 1:]) <= thr)[0] + i + 1
        for j in js:
            parent[find(j)] = find(i)
    return np.array([find(i) for i in range(len(hashes))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Raabin-WBC dataset root")
    ap.add_argument("--out", default="manifest.csv")
    ap.add_argument("--thr", type=int, default=3, help="pHash Hamming threshold for near-duplicates")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dfs = {"train": scan_split(find_split_dir(a.root, ["Train"])),
           "testA": scan_split(find_split_dir(a.root, ["Test-A", "TestA"])),
           "testB": scan_split(find_split_dir(a.root, ["Test-B", "TestB"]))}
    for k, d in dfs.items():
        print(k, len(d), d.label.value_counts().sort_index().to_dict())

    for k, d in dfs.items():
        with ThreadPoolExecutor(8) as ex:
            d["hash"] = list(tqdm(ex.map(phash64, d.path), total=len(d), desc=f"hash {k}"))

    # 1) remove train images that are near-duplicates of any test image (cross-split leakage)
    test_h = np.array(dfs["testA"].hash.tolist() + dfs["testB"].hash.tolist(), dtype=np.uint64)
    tr = dfs["train"]
    leak = np.array([hamming(h, test_h).min() <= a.thr for h in tr.hash])
    print(f"Removing {leak.sum()} train images near-duplicate to test images")
    tr = tr[~leak].reset_index(drop=True)

    # 2) group-aware train/val split on near-duplicate clusters
    tr["group"] = cluster(np.array(tr.hash.tolist(), dtype=np.uint64), a.thr)
    print("clusters:", tr.group.nunique(), "largest:", tr.group.value_counts().max())
    gss = GroupShuffleSplit(1, test_size=a.val_frac, random_state=a.seed)
    ti, vi = next(gss.split(tr, tr.label, tr.group))
    tr["split"] = "train"; tr.loc[vi, "split"] = "val"

    out = [tr[["path", "label", "split", "group"]]]
    for k in ["testA", "testB"]:
        d = dfs[k].copy(); d["split"] = k; d["group"] = -1
        out.append(d[["path", "label", "split", "group"]])
    man = pd.concat(out, ignore_index=True)
    man.to_csv(a.out, index=False)
    print(man.groupby(["split", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
