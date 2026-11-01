# split_data.py
"""Utility to invoke the dataset preparation pipeline.

Running this script will:
1. Compute perceptual hashes, remove near‑duplicate train images that appear in Test‑A/Test‑B.
2. Perform a group‑aware train/validation split (default 15% validation).
3. Produce a `manifest.csv` file listing `path`, `label`, `split`, and `group` columns.

The heavy lifting is done by ``src/prepare.py`` – this wrapper simply forwards the required arguments.
"""

import subprocess
import pathlib
import sys

def main():
    root = pathlib.Path(r"C:\Users\S.Supraja\OneDrive\Desktop\dl\cellscope-ai\RaabinWBC microscopic blood cell Dataset")
    if not root.exists():
        print(f"Dataset root not found: {root}")
        sys.exit(1)
    # Path to the prepare script
    prepare_script = pathlib.Path(__file__).parent.parent / "src" / "prepare.py"
    if not prepare_script.is_file():
        print(f"prepare.py not found at {prepare_script}")
        sys.exit(1)
    # Build command line
    cmd = [
        sys.executable,
        str(prepare_script),
        "--root", str(root),
        "--out", "manifest.csv",
        "--thr", "3",
        "--val_frac", "0.15",
        "--seed", "0",
    ]
    print("Running dataset preparation ...")
    subprocess.run(cmd, check=True)
    print("✅ manifest.csv created in the current directory.")

if __name__ == "__main__":
    main()
