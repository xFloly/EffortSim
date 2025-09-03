#!/bin/bash

for f in sweeps/*.yaml; do
  echo "[run] training with $f"
  python train.py --config "$f"
done
