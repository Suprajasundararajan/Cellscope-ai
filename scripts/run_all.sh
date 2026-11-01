#!/usr/bin/env bash
# Full pipeline. Usage: bash scripts/run_all.sh /path/to/raabin-wbc
set -e
ROOT=${1:?dataset root}
python -m src.prepare --root "$ROOT" --out manifest.csv
for AUG in std robust; do
  for SEED in 0 1 2; do
    for M in mobilenetv2 efficientnetb0 resnet50; do
      python -m src.train --model $M --aug $AUG --seed $SEED --epochs 8
    done
  done
done
python -m src.analyze --tags std robust --seeds 0 1 2
for M in mobilenetv2 efficientnetb0 resnet50; do python -m src.gradcam --model $M --tag std_s0; done
