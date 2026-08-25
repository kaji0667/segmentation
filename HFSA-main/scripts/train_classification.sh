#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

exec "$PYTHON_BIN" -u train_classification.py \
  --data-dir "${DATA_DIR:-data/VRSBench_scene}" \
  --model "${MODEL:-ultralytics/cfg/models/v12/yolov12m-classification.yaml}" \
  --weights "${WEIGHTS:-pretrain_model/yolov12m.pt}" \
  --save-dir "${SAVE_DIR:-runs/classification/vrsbench_scene}" \
  --device "${DEVICE:-cuda:0}" \
  --batch "${BATCH:-32}" \
  --imgsz "${IMGSZ:-640}" \
  --epochs "${EPOCHS:-20}" \
  --workers "${WORKERS:-2}" \
  --seed "${SEED:-42}" \
  "$@"
