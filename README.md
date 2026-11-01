# CellScope-AI
Explainable, uncertainty-aware ensemble (MobileNetV2 + EfficientNetB0 + ResNet50) for white-blood-cell
classification on Raabin-WBC, focused on **cross-domain robustness** (Test-A in-domain vs Test-B external).

## Pipeline
| Step | File | What it does |
|---|---|---|
| 1 | `src/prepare.py` | Scans Train / Test-A / Test-B, pHash near-duplicate removal (train vs test leakage), cluster-aware train/val split -> `manifest.csv` |
| 2 | `src/train.py` | Fine-tunes one backbone (class-weighted CE, AdamW + OneCycle). Model selection on in-domain val macro-F1 only. Saves logits for val/Test-A/Test-B |
| 3 | `src/analyze.py` | Singles vs equal vote vs validation-tuned weighted vote vs stacking; temperature scaling; ECE/NLL; per-class recall; bootstrap CIs; McNemar; uncertainty (entropy, ensemble disagreement) and accuracy-coverage curves; reliability diagrams; domain-gap table; std vs robust-augmentation ablation |
| 4 | `src/gradcam.py` | Grad-CAM per model, correct vs wrong, Test-A vs Test-B, plus a nucleus-overlap metric |

Design rules that protect the validity of the results:
* Ensemble weights, stacking meta-learner and temperature are fit **only on the in-domain validation split**. Test-B is never used for tuning.
* Base models never train on val, so the stacker sees honest (held-out) predictions. Val-side temperature for stacking uses 5-fold out-of-fold predictions.
* "Best single model" for McNemar is chosen on validation, not test.

## Setup
```bash
pip install -r requirements.txt
kaggle datasets download -d raabindata/raabin-wbc -p data --unzip     # or the oulabspring mirror
bash scripts/run_all.sh data/                                           # full run (GPU: Colab/Kaggle)
```
Quick debug on CPU: `python -m src.train --model mobilenetv2 --subset 200 --epochs 1 --workers 0`
Pipeline smoke test without data: `python -m src.make_fake_logits && python -m src.analyze --tags std robust --seeds 0 1`
(synthetic numbers, for checking the code only). Set `CELLSCOPE_OUT=/some/dir` to change the output folder.

Resume an interrupted training run from its latest epoch checkpoint:
`python -m src.train --model resnet50 --aug std --epochs 8 --workers 0 --resume`

The Streamlit application runs from the repository root:
`python -m streamlit run app.py`
It reports top-three predictions and flags low-confidence or high-disagreement cases
for manual review. It also screens strongly green/non-microscopy images as
`Not a WBC` before showing a cell class; this is a conservative input-domain
heuristic, not a replacement for a dedicated open-set detector. The ensemble weights are tuned from validation logits when they are
available; Test-A and Test-B are not used for tuning.

Run the complete multi-seed matrix on any platform:
`python scripts/run_multi_seed.py --seeds 0 1 2 --epochs 8 --workers 0`
Use `--device cuda` when a CUDA-enabled PyTorch installation is available.

Outputs: `outputs/results/*.csv|png`, `outputs/gradcam/*`, and `outputs/logits/*.json` (parameters,
timing, and training history). Generated checkpoints, logits, and local datasets are intentionally
ignored by Git.

## Results

The following results are from the checked-in `outputs/results/summary.csv`. They are shown here so
the reported numbers are visible without downloading the dataset or rerunning training. Values are
accuracy, macro-F1, ECE, and NLL.

| Method | Test-A accuracy | Test-A macro-F1 | Test-B accuracy | Test-B macro-F1 |
|---|---:|---:|---:|---:|
| EfficientNet-B0 | 0.9765 | 0.9631 | 0.2879 | 0.1755 |
| MobileNetV2 | 0.9703 | 0.9552 | 0.3653 | 0.2041 |
| ResNet50 | 0.9710 | 0.9560 | 0.3723 | 0.2312 |
| Equal vote | 0.9753 | 0.9625 | 0.3742 | 0.2258 |
| Validation-weighted vote | 0.9749 | 0.9619 | 0.3747 | 0.2203 |
| Stacking | 0.9737 | 0.9590 | 0.2685 | 0.1565 |

These are single-run results, not a multi-seed mean and standard deviation. Test-B is an external
domain and shows a substantial domain gap; the results should not be interpreted as clinical
performance.

## Dataset and licensing

The Raabin-WBC images are not included in this repository. Obtain them directly from the
[Raabin-WBC dataset page](https://www.kaggle.com/datasets/raabindata/raabin-wbc) and follow its
license, attribution, and usage terms. The dataset has terms separate from this code repository.

## Verify before you publish (I could not run these on the real data)
1. **Dataset layout and split semantics.** `prepare.py` finds folders named Train / Test-A / Test-B and ignores non-target classes (Smudge, Artifact). Read the Raabin-WBC paper to confirm what Test-A and Test-B represent, and cite it. Check class counts printed by `prepare.py`.
2. **No patient IDs in file names** -> the val split is cluster-based (pHash), an approximation of smear-level separation. Say so in the limitations. Tune `--thr`: too high merges unrelated cells, too low leaves duplicates.
3. **Staining:** Raabin is all Giemsa. Claims about stain robustness need an external dataset with a different stain.
4. **Timing:** `ms_per_img` includes data loading; for a clean efficiency table, benchmark on a fixed tensor batch.
5. **Nucleus mask** in Grad-CAM is a heuristic (Otsu on green channel). Use it as a sanity check, or replace it with annotated masks (Raabin provides some segmentation masks for subsets) for a stronger claim.
6. Run >= 3 seeds (`run_all.sh` does) and report mean +/- std plus the bootstrap CI / McNemar outputs.
7. Do a literature check: cross-domain WBC classification and ensembles on Raabin-WBC already exist; position your contribution (shift-aware evaluation + calibration + selective prediction) accordingly.

## Suggested paper tables/figures
* Table 1: single vs ensembles on Test-A / Test-B (acc, macro-F1, per-class recall, ECE, NLL) from `summary.csv`
* Table 2: domain gap and std-vs-robust augmentation from `domain_gap.csv`
* Fig: reliability diagrams, accuracy-coverage curves, Grad-CAM grids, overlap statistics
* Table: params / ms-per-image per model

Research/educational use only; not a diagnostic device.

This repository's source code is released under the [MIT License](LICENSE).
