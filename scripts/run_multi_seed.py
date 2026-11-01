"""Run the reproducible training matrix from Windows, macOS, or Linux.

Example:
    python scripts/run_multi_seed.py --seeds 0 1 2 --epochs 8 --workers 0
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--augmentations", nargs="+", choices=["std", "robust"],
                        default=["std", "robust"])
    parser.add_argument("--models", nargs="+",
                        choices=["mobilenetv2", "efficientnetb0", "resnet50"],
                        default=["mobilenetv2", "efficientnetb0", "resnet50"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--manifest", default="manifest.csv")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    for augmentation in args.augmentations:
        for seed in args.seeds:
            for model in args.models:
                command = [
                    sys.executable, "-m", "src.train",
                    "--model", model, "--aug", augmentation,
                    "--seed", str(seed), "--epochs", str(args.epochs),
                    "--workers", str(args.workers), "--device", args.device,
                    "--manifest", args.manifest,
                ]
                print("Running:", " ".join(command), flush=True)
                subprocess.run(command, cwd=root, check=True)

    command = [
        sys.executable, "-m", "src.analyze",
        "--tags", *args.augmentations, "--seeds", *(str(s) for s in args.seeds),
    ]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
