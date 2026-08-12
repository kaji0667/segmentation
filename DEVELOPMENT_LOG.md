# Development Log

All entries are append-only. `PROJECT_RULES.md` remains the single source of truth for rules.

## 2026-07-15

### Semantic Segmentation Training Adjustment

Context:
- User requested execution of recommendations 2, 3, and 4 from the latest RRSIS-D result analysis.
- The goal is to reduce overfitting pressure and keep the mainline experiment on the best observed non-P2 configuration.
- Backbone, neck, OpenCLIP encoder, and general Ultralytics internals were not modified.

Changes:
- Added train-time lightweight augmentation in `HFSA-main/dataset/rrsisd_refseg_dataset.py`.
- Augmentation includes horizontal flip, vertical flip, and brightness/contrast jitter.
- Geometric flips are automatically skipped when text contains directional/positional words such as left, right, top, bottom, upper, lower, or center.
- Augmentation is only passed for `split == "train"` in `HFSA-main/train_semseg.py`.
- Added CLI controls:
  - `--augment` / `--no-augment`
  - `--augment-hflip`
  - `--augment-vflip`
  - `--augment-color-jitter`
- Changed `--small-target-boost` default to `2.0`.
- Confirmed the default model remains non-P2: `ultralytics/cfg/models/v12/yolov12-semseg.yaml`.

Rationale:
- Recent runs showed non-P2 had the best target IoU among inspected experiments.
- P2 improved validation loss in one run but did not improve final target IoU.
- Small-target boost of `3.0` is likely too aggressive for the current dataset and may increase sampling bias.
- Rotation was not added because direction words in free-text referring expressions can become semantically invalid after rotation.
- After checking the current metadata, directional/positional words appear in about 68% of RRSIS-D referring expressions, so default geometric flip must be text-aware rather than unconditional.

Verification:
- Ran syntax validation:
  - `python -m py_compile .\dataset\rrsisd_refseg_dataset.py .\train_semseg.py`
- Result: passed.
- Ran directional-text predicate check:
  - `['a plane on the left', 'a ship near harbor', 'upper tennis court', 'center building'] -> [True, False, True, True]`
- Checked that augmentation is train-only in `build_semseg_dataset()`.
- Counted directional/positional text frequency in `test.jsonl`, `train.jsonl`, and `val.jsonl`: 11850 / 17402 samples, about 68.10%.
- Checked that the default model argument points to non-P2 config.

Not Verified:
- Full GPU smoke test was not run in this turn.
- Pure in-memory augmentation behavior test was attempted but the Windows-side Python environment is missing `numpy`.
- Root and `HFSA-main` `git status --short` both report `fatal: not a git repository`.

## 2026-07-16

### Validation Preview Threshold Alignment

Context:
- Latest run `runs/semseg/rrsisd_openclip_512_b4_e28_nonp2_aug_boost2` reached `target_iou=0.6725714206695557` and `binary_miou=0.8259929418563843`.
- The validation sweep selected `best_threshold=0.6`, but `val_batch0_pred_epoch*.jpg` was still rendered with a fixed `0.5` threshold.
- This made qualitative previews slightly inconsistent with the metrics reported in `results.csv`.

Changes:
- Updated `save_preview()` in `HFSA-main/train_semseg.py` to accept a `threshold` argument.
- Updated `validate()` to cache the first validation batch and render the preview after threshold sweep, using the epoch's selected `best_threshold`.
- Training preview keeps the default `0.5` threshold because no validation threshold has been selected at that point.

Analysis Notes:
- The latest preview's second sample is `val_3890`, class `dam`, text `The large dam`; it shows over-segmentation.
- The fourth sample is `val_14636`, class `vehicle`, text `The vehicle is above the chimney at the bottom`; it shows small-object miss/near-miss behavior.
- The weak classes in the latest epoch include `vehicle=0.1954`, `harbor=0.2607`, `tenniscourt=0.4371`, and `windmill=0.4755`.

Verification:
- Ran syntax validation:
  - `python -m py_compile .\train_semseg.py`
- Result: passed.

## 2026-07-17

### Mask-Area Sampling and Per-Sample Loss

Context:
- Latest experiments showed `small-target-boost=2.5` did not improve the mainline over `small-target-boost=2.0`.
- Analysis found that bbox area is a poor proxy for RRSIS-D foreground size. For example, many `vehicle` samples have large boxes but very small true masks.
- User requested three changes: use true mask area for small-target sampling, change batch-level Dice to per-sample Dice, and add area-aware loss weighting.

Changes:
- Updated `build_text_query_sampler()` in `HFSA-main/train_semseg.py` to compute sample area from RLE foreground pixels when segmentation metadata is available.
- Kept bbox area as fallback for rows without RLE metadata.
- Added CLI controls:
  - `--loss-small-target-weight`
  - `--loss-small-target-area`
- Passed the area-aware loss settings from `train_semseg.py` to `SemanticSegmentationLoss` through model attributes.
- Updated binary mask loss in `HFSA-main/ultralytics/utils/loss.py`:
  - BCE is now computed per sample.
  - Dice is now computed per sample.
  - Per-sample losses are averaged across the batch.
  - Samples whose foreground area ratio is below `--loss-small-target-area` can receive a multiplier from `--loss-small-target-weight`.

Rationale:
- Per-sample loss prevents large masks in the same batch from dominating the Dice term for small targets.
- Mask-area sampling is aligned with the actual binary target, unlike bbox area for thin or sparse objects.
- Area-aware loss weighting directly increases the training signal for small masks without changing the backbone or OpenCLIP encoder.

Verification:
- Ran syntax validation:
  - `python -m py_compile .\train_semseg.py .\ultralytics\utils\loss.py`
- Result: passed.
- Verified RLE foreground area extraction on a training `vehicle` sample; example `train_6000` has true mask area ratio about `0.002225`.

Not Verified:
- Runtime tensor loss test was not run in this Windows-side Python environment because `torch` is not installed.

Recommended Next Experiment:

```bash
cd /mnt/d/code/python/HFSA/HFSA-main
python train_semseg.py \
  --data pre_datasets/RRSIS-D_refseg/data.yaml \
  --model ultralytics/cfg/models/v12/yolov12-semseg.yaml \
  --weights yolov12n.pt \
  --device cuda:0 \
  --imgsz 512 \
  --batch 4 \
  --epochs 28 \
  --patience 4 \
  --min-delta 0.001 \
  --freeze-backbone \
  --pos-weight-max 10 \
  --small-target-boost 2.0 \
  --small-target-area 0.0025 \
  --val-thresholds 0.3,0.4,0.5,0.6 \
  --text-queries \
  --text-encoder openclip \
  --text-model-name ViT-L-14 \
  --text-pretrained openai \
  --text-precision fp32 \
  --scheduler cosine \
  --min-lr 1e-6 \
  --save-dir runs/semseg/rrsisd_nonp2_aug_b4_e28_boost2
```

Suggested Commit Message:

```text
feat(semseg): add lightweight train augmentation

Context:
- RRSIS-D runs show validation-loss growth and possible overfitting.
- Non-P2 remains the stronger mainline by target IoU.

Changes:
- Add train-only text-aware flip and brightness/contrast augmentation for RRSIS-D referring segmentation.
- Expose augmentation CLI controls.
- Change small-target boost default to 2.0.
- Keep non-P2 semantic segmentation config as the default model.

Tests:
- python -m py_compile .\dataset\rrsisd_refseg_dataset.py .\train_semseg.py

Docs:
- Update ARCHITECTURE.md and DEVELOPMENT_LOG.md.
```

## 2026-07-20

### Mask-Area Loss Experiments

Context:
- After the 2026-07-17 code changes, two full RRSIS-D validation runs were completed to test true mask-area sampling, per-sample BCE/Dice, and area-aware small-mask loss weighting.
- The comparison baseline is `runs/semseg/rrsisd_openclip_512_b4_e28_nonp2_aug_boost2`, which previously reached `target_iou=0.6725714206695557` and `binary_miou=0.8259929418563843`.
- Backbone, neck, OpenCLIP text encoder, and model YAML stayed unchanged.

Code Changes Under Test:
- `HFSA-main/train_semseg.py`
  - `build_text_query_sampler()` now uses true RLE foreground mask area when available.
  - bbox area is used only as a fallback.
  - Added `--loss-small-target-weight` and `--loss-small-target-area`.
  - Passes small-mask loss settings to the model before loss construction.
- `HFSA-main/ultralytics/utils/loss.py`
  - Binary BCE is computed per sample.
  - Dice is computed per sample.
  - Per-sample losses are averaged across the batch.
  - Samples with foreground ratio `<= --loss-small-target-area` can receive `--loss-small-target-weight`.
- `HFSA-main/train_semseg.py`
  - Validation preview images use the selected validation threshold, so qualitative previews match `results.csv` more closely.

Run 1 Command:

```bash
MPLCONFIGDIR=/tmp YOLO_CONFIG_DIR=/tmp python train_semseg.py \
    --model ultralytics/cfg/models/v12/yolov12-semseg.yaml \
    --device cuda:0 \
    --imgsz 512 \
    --batch 4 \
    --epochs 28 \
    --max-batches 0 \
    --max-val-batches 0 \
    --workers 2 \
    --print-interval 200 \
    --text-queries \
    --text-encoder openclip \
    --text-model-name ViT-L-14 \
    --text-pretrained openai \
    --text-precision fp32 \
    --scheduler cosine \
    --min-lr 1e-6 \
    --freeze-backbone \
    --patience 4 \
    --min-delta 0.001 \
    --pos-weight-max 10 \
    --small-target-boost 1.5 \
    --small-target-area 0.0025 \
    --loss-small-target-weight 1.5 \
    --loss-small-target-area 0.0025 \
    --augment \
    --augment-hflip 0.5 \
    --augment-vflip 0.5 \
    --augment-color-jitter 0.15 \
    --val-thresholds 0.4,0.5,0.6,0.7,0.8 \
    --save-dir runs/semseg/rrsisd_nonp2_maskarea_lossw15
```

Run 1 Result:
- Output: `runs/semseg/rrsisd_nonp2_maskarea_lossw15`
- Best epoch: `28`
- `target_iou=0.6824873685836792`
- `binary_miou=0.8316676616668701`
- `val_loss=0.7491593355419992`
- `best_threshold=0.8`
- `precision=0.8060332536697388`
- `recall=0.8166031837463379`
- `f1=0.8112837921971922`

Run 1 Key Class IoU:
- `vehicle=0.3006022572517395`
- `harbor=0.25323954224586487`
- `windmill=0.5432848930358887`
- `tenniscourt=0.4968189299106598`
- `dam=0.5703408718109131`
- `trainstation=0.5237587690353394`
- `baseballfield=0.7178502678871155`
- `bridge=0.6282254457473755`
- `ship=0.6513206362724304`

Run 2 Command:

```bash
MPLCONFIGDIR=/tmp YOLO_CONFIG_DIR=/tmp python train_semseg.py \
    --model ultralytics/cfg/models/v12/yolov12-semseg.yaml \
    --device cuda:0 \
    --imgsz 512 \
    --batch 4 \
    --epochs 28 \
    --max-batches 0 \
    --max-val-batches 0 \
    --workers 2 \
    --print-interval 200 \
    --text-queries \
    --text-encoder openclip \
    --text-model-name ViT-L-14 \
    --text-pretrained openai \
    --text-precision fp32 \
    --scheduler cosine \
    --min-lr 1e-6 \
    --freeze-backbone \
    --patience 4 \
    --min-delta 0.001 \
    --pos-weight-max 10 \
    --small-target-boost 1.5 \
    --small-target-area 0.0025 \
    --loss-small-target-weight 1.25 \
    --loss-small-target-area 0.0025 \
    --augment \
    --augment-hflip 0.5 \
    --augment-vflip 0.5 \
    --augment-color-jitter 0.15 \
    --val-thresholds 0.6,0.7,0.8,0.85,0.9,0.95 \
    --save-dir runs/semseg/rrsisd_nonp2_maskarea_lossw125
```

Run 2 Result:
- Output: `runs/semseg/rrsisd_nonp2_maskarea_lossw125`
- Best epoch: `25`
- `target_iou=0.6807118654251099`
- `binary_miou=0.8307759165763855`
- `val_loss=0.7143737212337297`
- `best_threshold=0.85`
- `precision=0.8095194697380066`
- `recall=0.8105373382568359`
- `f1=0.8100280842381645`
- Last epoch `28`: `target_iou=0.6798320412635803`, `binary_miou=0.8302220106124878`, `best_threshold=0.8`

Run 2 Key Class IoU:
- `vehicle=0.30675309896469116`
- `harbor=0.26998037099838257`
- `windmill=0.5550129413604736`
- `tenniscourt=0.5118797421455383`
- `dam=0.5563564300537109`
- `trainstation=0.5258992314338684`
- `baseballfield=0.7614495754241943`
- `bridge=0.6377878785133362`
- `ship=0.6960523724555969`

Comparison:
- Baseline `rrsisd_openclip_512_b4_e28_nonp2_aug_boost2`: `target_iou=0.6725714206695557`, `binary_miou=0.8259929418563843`, `best_threshold=0.6`.
- `lossw15` improved baseline by about `+0.0099 target_iou` and `+0.0057 binary_miou`.
- `lossw125` improved baseline by about `+0.0081 target_iou` and `+0.0048 binary_miou`.
- `lossw15` is currently the best overall validation run by `target_iou`.
- `lossw125` is slightly lower overall but improves several weak or important classes compared with `lossw15`, including `vehicle`, `harbor`, `windmill`, `tenniscourt`, `baseballfield`, `bridge`, and `ship`.

Qualitative Notes:
- `lossw125` improved the fixed preview batch's fourth sample (`val_14636`, `vehicle`) from near-complete miss to a prediction area close to the ground-truth red area.
- The fixed preview batch's second sample (`val_3890`, `dam`) remains over-segmented. This appears to be a boundary/shape problem rather than a small-mask sampling problem.
- `best_threshold` increased from `0.6` in the old baseline to `0.8` / `0.85` in the new runs. This suggests that the new loss and sampling policy make foreground logits stronger and require higher thresholds for best mask extraction.

Interpretation:
- The code modification is effective for the intended small-mask problem. `vehicle` increased from `0.1953996866941452` in the old `boost2` baseline to `0.3006022572517395` in `lossw15` and `0.30675309896469116` in `lossw125`.
- Lowering `--loss-small-target-weight` from `1.5` to `1.25` reduced the overall best `target_iou` slightly, but made several weak classes more balanced.
- The remaining `dam` and `harbor` issues are not primarily caused by small-target sampling. They likely need boundary-aware loss, mask refinement, or stronger spatial/text grounding.

Current Recommendation:
- Use `rrsisd_nonp2_maskarea_lossw15` as the current overall best result.
- Keep `rrsisd_nonp2_maskarea_lossw125` as a useful ablation showing better weak-class balance.
- Next development should prioritize standard evaluation metrics (`oIoU`, per-sample `mIoU`, `P@0.5` to `P@0.9`) before further model changes.
- For further model improvement, focus on boundary/over-segmentation handling rather than increasing small-target weights again.

## 2026-07-21

### Reproducibility Seed Control

Context:
- Recent runs showed meaningful gains, but some differences between `lossw15`, `lossw125`, and `lossw15_thrwide` may come from random initialization, weighted sampling order, dataloader workers, and train-time augmentation randomness.
- To make future parameter comparisons and lightweight ablation more defensible, training needs explicit seed control.

Changes:
- Added `--seed`, default `42`, to `HFSA-main/train_semseg.py`.
- Added `--deterministic` as an optional flag for stricter PyTorch/CUDA deterministic behavior when supported.
- Added `set_random_seed()` to seed Python `random`, NumPy, PyTorch CPU, and PyTorch CUDA.
- Added `seed_worker()` for DataLoader worker-level NumPy/Python randomness.
- Added a seeded `torch.Generator` for `WeightedRandomSampler` and DataLoader.
- Printed seed settings in the training configuration output.

Verification:
- Ran syntax validation:
  - `python -m py_compile .\train_semseg.py`
- Result: passed.

Usage Note:
- Future controlled comparisons should include `--seed 42`.
- Use `--deterministic` only when strict reproducibility is more important than speed; it may slow CUDA training or warn about unsupported deterministic kernels.
