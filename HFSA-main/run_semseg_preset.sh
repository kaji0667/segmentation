#!/usr/bin/env bash
set -euo pipefail

preset="${1:-precision}"
shift || true

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp}"
if [[ -n "${GPU:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

common_args=(
  --data pre_datasets/RRSIS-D_refseg/data.yaml
  --model ultralytics/cfg/models/v12/yolov12-semseg.yaml
  --weights yolov12n.pt
  --device "${DEVICE:-cuda:0}"
  --imgsz "${IMGSZ:-512}"
  --batch "${BATCH:-8}"
  --epochs "${EPOCHS:-80}"
  --max-batches 0
  --max-val-batches 0
  --workers "${WORKERS:-2}"
  --print-interval 200
  --text-queries
  --text-encoder openclip
  --text-model-name ViT-L-14
  --text-pretrained openai
  --text-precision fp32
  --scheduler cosine
  --min-lr 1e-6
  --freeze-backbone
  --patience "${PATIENCE:-10}"
  --min-delta 0.001
  --augment
  --augment-hflip 0.5
  --augment-vflip 0.5
  --augment-color-jitter 0.15
  --seed "${SEED:-42}"
)

case "$preset" in
  baseline)
    preset_args=(
      --pos-weight-max 10
      --small-target-boost 1.5
      --small-target-area 0.0025
      --loss-small-target-weight 1.5
      --loss-small-target-area 0.0025
      --val-thresholds 0.6,0.7,0.8,0.85,0.9,0.95
      --save-dir "${SAVE_DIR:-runs/semseg/rrsisd_baseline_b${BATCH:-8}_seed${SEED:-42}}"
    )
    ;;
  precision)
    preset_args=(
      --pos-weight-max 8
      --small-target-boost 1.2
      --small-target-area 0.0025
      --loss-small-target-weight 1.2
      --loss-small-target-area 0.0025
      --loss-tversky-fp-weight 0.65
      --loss-fp-weight 0.05
      --val-thresholds 0.75,0.8,0.85,0.9,0.95
      --val-select-metric fbeta
      --val-fbeta 0.7
      --save-dir "${SAVE_DIR:-runs/semseg/rrsisd_precision_tversky_b${BATCH:-8}_seed${SEED:-42}}"
    )
    ;;
  strict)
    preset_args=(
      --pos-weight-max 6
      --small-target-boost 1.0
      --loss-small-target-weight 1.0
      --loss-tversky-fp-weight 0.75
      --loss-fp-weight 0.1
      --val-thresholds 0.8,0.85,0.9,0.95
      --val-select-metric fbeta
      --val-fbeta 0.5
      --save-dir "${SAVE_DIR:-runs/semseg/rrsisd_strict_fp_b${BATCH:-8}_seed${SEED:-42}}"
    )
    ;;
  smoke)
    preset_args=(
      --batch "${BATCH:-2}"
      --epochs 1
      --max-batches 2
      --max-val-batches 2
      --workers 0
      --val-thresholds 0.8,0.9
      --save-dir "${SAVE_DIR:-runs/semseg/smoke}"
    )
    ;;
  *)
    echo "Unknown preset: $preset" >&2
    echo "Available presets: baseline, precision, strict, smoke" >&2
    exit 2
    ;;
esac

python train_semseg.py "${common_args[@]}" "${preset_args[@]}" "$@"
