import os
from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from .common import CLASSES

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
_NORM = lambda s: s.lower().replace("_", "-").replace(" ", "-")


def find_split_dir(root, names):
    """Locate a split folder (e.g. Train, Test-A) anywhere under root that contains class sub-folders."""
    names = {_NORM(n) for n in names}
    lower = {c.lower() for c in CLASSES}
    for dp, dns, _ in os.walk(root):
        for d in dns:
            if _NORM(d) in names:
                sub = {x.name.lower() for x in (Path(dp) / d).iterdir() if x.is_dir()}
                if sub & lower:
                    return Path(dp) / d
    raise FileNotFoundError(f"Could not find split {names} under {root}")


def scan_split(split_dir):
    """Collect (path, label) for the 5 target classes; other folders (e.g. Smudge, Artifact) are ignored."""
    cmap = {c.lower(): i for i, c in enumerate(CLASSES)}
    rows = []
    for sub in sorted(Path(split_dir).iterdir()):
        if sub.is_dir() and sub.name.lower() in cmap:
            for f in sorted(sub.rglob("*")):
                if f.suffix.lower() in IMG_EXT:
                    rows.append((str(f), cmap[sub.name.lower()]))
    return pd.DataFrame(rows, columns=["path", "label"])


def build_tf(train: bool, robust: bool = False, size: int = 224):
    if not train:
        return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])
    aug = [T.Resize((size, size)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
           T.RandomRotation(180, fill=0)]
    if robust:  # domain-robustness augmentation (stain / camera / illumination variation)
        aug += [T.ColorJitter(0.4, 0.4, 0.4, 0.1), T.RandomGrayscale(0.1),
                T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=0.3)]
    else:
        aug += [T.ColorJitter(0.1, 0.1, 0.1, 0.02)]
    return T.Compose(aug + [T.ToTensor(), T.Normalize(MEAN, STD)])


class WBCDataset(Dataset):
    def __init__(self, df, tf):
        self.paths, self.labels, self.tf = df["path"].tolist(), df["label"].tolist(), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as image:
            return self.tf(image.convert("RGB")), self.labels[i]
