#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export BATCH="${BATCH:-4}"
export EPOCHS="${EPOCHS:-60}"
export PATIENCE="${PATIENCE:-8}"
export DEVICE="${DEVICE:-cuda:0}"
export IMGSZ="${IMGSZ:-512}"
export WORKERS="${WORKERS:-2}"
export TEST_AFTER_TRAIN="${TEST_AFTER_TRAIN:-1}"
export MAX_TEST_BATCHES="${MAX_TEST_BATCHES:-0}"
export SAVE_DIR="${SAVE_DIR:-runs/semseg/srp_yolov12m_axis}"

exec bash "$PROJECT_ROOT/run_semseg_preset.sh" baseline \
  --data "${DATA:-pre_datasets/RRSIS-D_refseg/data.yaml}" \
  --model "${MODEL:-ultralytics/cfg/models/v12/yolov12m-semseg.yaml}" \
  --weights "${WEIGHTS:-pretrain_model/yolov12m.pt}" \
  "$@"
