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

## 2026-08-17

### RRSIS-D Official Metric Protocol

Context:
- Competition guidance requires official dataset metrics and reproducible threshold handling.
- RRSIS-D papers define oIoU as cumulative foreground intersection over union, mIoU as the mean of per-sample IoUs, and Pr@0.5 through Pr@0.9 as sample success rates.
- The previous CSV field `miou` contained foreground aggregate IoU, while the paper mIoU was stored as `sample_miou`. Per-category `class_iou` was category aggregate IoU rather than category sample-mean IoU.

Changes:
- Added explicit `oiou` and `official_miou` result fields while retaining legacy fields.
- Added official per-category mIoU and separate per-category oIoU output.
- Added `--val-select-metric oiou|miou`; legacy `iou` remains an oIoU alias.
- Added `--test-after-train` and `--max-test-batches`.
- Test evaluation loads `best.pt` and freezes the validation-selected threshold.
- Test reports include official metrics, per-category metrics, parameter counts, checkpoint size, evaluation duration, mean time per sample, and peak allocated GPU memory.
- Added `TEST_AFTER_TRAIN=1` and `MAX_TEST_BATCHES` support to `run_semseg_preset.sh`.

Verification:
- `python -m py_compile train_semseg.py`: passed.
- Synthetic metric check: oIoU `2/3`, sample mIoU `0.75`, and per-category values `1.0/0.5`: passed.
- GPU smoke training with `--val-select-metric miou`: passed.
- GPU smoke training plus fixed-threshold test evaluation: passed and produced `test_results.json`.
- Test text embedding cache for all 3,481 RRSIS-D test samples was generated successfully.
- Full 1,740-sample validation rerun of `rrsisd_learnable_gate_b4_e60_seed42/weights/best.pt` at its frozen threshold `0.85`: `oIoU=0.6997935`, `mIoU=0.5255537`, `class_macro_mIoU=0.5544193`, `Pr@0.5/0.7/0.9=0.5931034/0.4166667/0.1459770`.

Protocol Note:
- Historical `target_iou` is equivalent to official oIoU.
- Historical `sample_miou` is equivalent to official mIoU.
- Historical `binary_miou` is the mean of background and foreground IoUs and must not be compared with RRSIS-D paper mIoU.
- Historical `class_iou` is category aggregate IoU and must not be reported as category mIoU.

## 2026-08-18

### Fixed P3/P4 Text-Conditioned Deep Supervision Candidate

Context:
- The latest official run generalized normally but retained a large oIoU-to-sample-mIoU gap and weak vehicle/harbor/bridge results.
- The user requested one controlled fixed-weight deep-supervision experiment without changing the backbone, neck, OpenCLIP encoder, final decoder, or the two learnable spatial-gate weights.

Changes:
- Added text-conditioned P3 and P4 auxiliary mask heads to `TextPromptSegment`.
- Auxiliary branches execute only during training when their configured weights are positive.
- Added fixed loss options `--loss-aux-p3-weight` and `--loss-aux-p4-weight`, both disabled by default.
- Auxiliary targets use foreground-preserving adaptive max pooling.
- Added the `ds` script preset with weights `P3=0.20`, `P4=0.10` and output `runs/semseg/ds_p3p4`.
- Validation and test inference still return and evaluate only the final mask.

Verification:
- `python -m py_compile train_semseg.py ultralytics/nn/modules/head.py ultralytics/utils/loss.py`: passed.
- `bash -n run_semseg_preset.sh`: passed.
- CPU random-tensor regression test verified main/P3/P4 shapes, finite combined loss, gradients on both auxiliary heads, and final-only eval output: passed.
- Full dataset training was intentionally not started; the user will launch it with the preset script.

Controlled run command:

```bash
GPU=0 BATCH=4 EPOCHS=60 PATIENCE=8 TEST_AFTER_TRAIN=1 SAVE_DIR=runs/semseg/ds_p3p4 bash run_semseg_preset.sh ds
```

Interpretation constraint:
- This is an experimental candidate, not a confirmed improvement. Compare it against `rrsisd_learnable_gate_official_seed42` using oIoU, official mIoU, Pr@0.5-0.9, per-category mIoU, precision/recall, and predicted-positive versus target-positive rates.

### P3-Only Follow-up Preset

Result-driven decision:
- The completed P3/P4 run improved validation sample mIoU and several categories, but official test oIoU and high-IoU success rates did not improve.
- Bridge, tenniscourt, windmill, and overpass regressed, consistent with a possible coarse P4 auxiliary-target effect.
- The raw best validation epoch was not saved because its improvement over `best.pt` was about `0.00097`, just below the previous `min_delta=0.001`.

Changes:
- Added `ds_p3` preset with fixed `P3=0.20`, `P4=0.0`.
- Set the preset-specific `min_delta=0.0002`.
- Kept the original `ds` preset unchanged for reproduction.
- Default output is `runs/semseg/ds_p3`.

Run command:

```bash
GPU=0 BATCH=4 EPOCHS=60 PATIENCE=8 TEST_AFTER_TRAIN=1 SAVE_DIR=runs/semseg/ds_p3 bash run_semseg_preset.sh ds_p3
```

Status:
- Script-only follow-up configuration; full training has not been started by Codex.

## 2026-08-19

### Deep-Supervision Results and Baseline Restoration

Completed results:
- No-deep-supervision baseline test: `oIoU=0.691823`, `mIoU=0.509016`, `class_macro_mIoU=0.530612`.
- P3/P4 deep supervision test: `oIoU=0.685367`, `mIoU=0.508498`, `class_macro_mIoU=0.532307`.
- P3-only deep supervision test: `oIoU=0.671189`, `mIoU=0.493307`, `class_macro_mIoU=0.521660`.
- P3-only also reduced `Pr@0.8` from `0.294168` to `0.260557` and regressed most categories, despite improving harbor.

Interpretation:
- Removing P4 did not recover performance, so coarse P4 auxiliary targets were not the sole cause.
- Directly forcing shallow P3 features to solve the full referring-mask task likely conflicted with the final multi-scale, text-conditioned objective.
- The current bottleneck is not insufficient P3/P4 supervision; raw-scale auxiliary mask supervision is rejected as the active direction.

Rollback:
- Removed P3/P4 auxiliary modules from `TextPromptSegment`.
- Restored the original single-output BCE-Tversky loss path.
- Removed auxiliary-loss CLI parameters, `ds`/`ds_p3` presets, and the deep-supervision regression test.
- Retained the two learnable spatial-gate weights and official RRSIS-D evaluation implementation.
- Preserved `runs/semseg/ds_p3p4` and `runs/semseg/ds_p3` as experiment evidence.

### Official-mIoU Raw-Best Checkpoint Retention

Changes:
- The standard `baseline` preset now passes `--val-select-metric miou`.
- Added `best_raw.pt`, saved on every strict validation selection-score maximum without applying `min_delta`.
- Kept `best.pt` and early stopping under the existing `min_delta` rule.
- Test-after-train now prefers `best_raw.pt`, falls back to legacy `best.pt`, and records the checkpoint selection metric and threshold source.

Verification:
- `python -m py_compile train_semseg.py tests/test_semseg_checkpoint_selection.py`: passed.
- `python tests/test_semseg_checkpoint_selection.py -v`: passed (3 tests).
- `bash -n run_semseg_preset.sh`: passed.
- GPU smoke with 2 train, 2 validation, and 2 test batches: passed after correcting one stale report variable name.
- Smoke output created `last.pt`, `best.pt`, and `best_raw.pt`; `test_results.json` recorded `checkpoint=.../best_raw.pt`, `checkpoint_selection_metric=miou`, and `threshold_source=raw-best validation checkpoint`.

### Full Official-mIoU Selection Experiment

Command:
```bash
GPU=0 DEVICE=cuda:0 BATCH=4 EPOCHS=60 PATIENCE=8 TEST_AFTER_TRAIN=1 \
SAVE_DIR=runs/semseg/base_miou bash run_semseg_preset.sh baseline
```

Execution:
- Completed successfully on the RTX 4060 Laptop GPU.
- Early stopped after epoch 36; summed epoch time was about 3.54 hours.
- Validation selected epoch 28 by official mIoU: `mIoU=0.523579`, `oIoU=0.676562`, threshold `0.70`.
- `best_raw.pt` and `best.pt` both ended at epoch 28; epoch 25 independently demonstrated the new behavior by updating only `best_raw.pt` for a sub-`min_delta` mIoU gain.
- Full test evaluated all 3,481 samples with the frozen validation threshold and loaded `best_raw.pt`.

Test result:
- `oIoU=0.672197`
- `mIoU=0.509192`
- `class_macro_mIoU=0.536258`
- `precision=0.774748`, `recall=0.835480`, `F1=0.803969`
- `Pr@0.5/0.7/0.8/0.9=0.558460/0.381212/0.272336/0.126688`
- `pred_pos_rate=0.050308`, target positive rate `0.046652`

Comparison with `rrsisd_learnable_gate_official_seed42`:
- mIoU changed by only `+0.000176`, effectively a tie without repeated-seed evidence.
- class-macro mIoU improved by `+0.005646`, led by harbor `+0.077195`.
- oIoU regressed by `-0.019627`, F1 by `-0.013875`, precision by `-0.050674`, and Pr@0.9 by `-0.020396`.
- Recall increased by `+0.025077` and predicted-positive rate by `+0.004506`, indicating more foreground coverage and more overflow.
- Bridge, Expressway-Service-area, windmill, baseballfield, and airport were the largest category regressions.

Decision:
- The raw-best checkpoint mechanism is verified and retained.
- This run does not replace the previous learnable-gate checkpoint as the best balanced model.
- Official-mIoU-only selection improves category balance slightly but does not solve weak-instance consistency and sacrifices cumulative overlap and high-IoU success rates.

## 2026-08-20

### Learnable Text Token Pooling Candidate

Diagnosis:
- Cached OpenCLIP inputs have shape `[N,77,768]`, while the active head used unconditional `tokens.mean(1)`.
- Validation expressions contain 6.69 valid tokens on average.

Changes:
- Added a zero-initialized bias-free `Linear(768,1)` token scorer and one zero-initialized valid-token bias.
- Softmax-weighted pooling is mathematically equal to old mean pooling at initialization.
- Added 769 parameters; OpenCLIP, image backbone/neck, decoder, loss, spatial-gate weights, and evaluation remain unchanged.
- Rejected object/spatial role residuals after independent review found BPE alignment and distractor-object ambiguity.

Evidence:
- Literature basis: CLIP-Adapter (2110.04544), Global-Local Context Features (2303.17811), RMSIN (2312.12470), RSRefSeg (2501.06809).
- Python compilation and 3 focused unit tests passed.
- Independent agent-A review found no blocking issue or redundant fallback code.
- GPU smoke passed 2 train, 2 validation, and 2 test batches and saved all expected checkpoints/reports under `runs/semseg/tpool_smoke`.

Full command:
```bash
GPU=0 DEVICE=cuda:0 BATCH=4 EPOCHS=60 PATIENCE=8 TEST_AFTER_TRAIN=1 \
SAVE_DIR=runs/semseg/tpool bash run_semseg_preset.sh baseline
```


### Learnable Text Token Pooling Full Result

Execution:
- The seed-42 full run completed after 47 epochs and evaluated all 3,481 test samples with `best_raw.pt` from epoch 39 and the frozen validation threshold `0.70`.

Test result:
- `oIoU=0.683420`
- `mIoU=0.519818`
- `class_macro_mIoU=0.545010`
- `precision=0.788342`, `recall=0.836999`, `F1=0.811943`
- `Pr@0.5/0.7/0.8/0.9=0.573973/0.383510/0.275783/0.137604`
- `pred_pos_rate=0.049531`, target positive rate `0.046652`

Same-protocol comparison with `base_miou`:
- oIoU improved by `+0.011223`.
- official mIoU improved by `+0.010626`.
- class-macro mIoU improved by `+0.008753`.
- Precision, Recall, F1, and Pr@0.5-0.9 all improved; predicted-positive rate decreased slightly.

Decision:
- Retain learnable token pooling as the active text aggregation path.
- It does not fully solve foreground overflow relative to the older oIoU-selected checkpoint, so the next experiment targets the spatial attention heatmap rather than adding another text-role heuristic.

### Calibrated Spatial Attention Heatmap Candidate

Diagnosis:
- `key` and `query` were both L2-normalized, but their cosine logits were divided again by `sqrt(128)`.
- The logits were therefore bounded near `[-0.088, 0.088]`; a 4,096-position softmax could vary by at most about `1.19x` between its theoretical maximum and minimum.
- The resulting attention channel was nearly uniform and had mean magnitude `1/4096`, making it poorly scaled for a dense segmentation gate and decoder input.

Changes:
- Added one learnable `attention_logit_scale`, initialized so the positive temperature is `1.0`.
- Removed the additional `1/sqrt(embed_dim)` compression.
- Converted the spatial softmax to relative density `probability * num_positions - 1`.
- Bounded the heatmap with `tanh`, so uniform attention is exactly zero and the output stays in `[-1, 1]`.
- Kept token pooling, P3/P4/P5 fusion, FiLM, similarity path, decoder, loss, backbone, neck, OpenCLIP, and evaluation unchanged.

Verification:
- `python -m py_compile ultralytics/nn/modules/head.py tests/test_semseg_spatial_attention.py tests/test_semseg_text_token_pooling.py`: passed.
- `python tests/test_semseg_spatial_attention.py -v`: 4 tests passed.
- `python tests/test_semseg_text_token_pooling.py -v`: 3 tests passed.
- GPU smoke with 2 train, 2 validation, and 2 test batches: passed under `runs/semseg/attnmap_smoke`.
- Smoke saved `last.pt`, `best.pt`, `best_raw.pt`, and `test_results.json`; strict checkpoint loading during test evaluation passed.
- Trainable parameter count increased from `2,747,849` to `2,747,850`.

Planned controlled full command:
```bash
GPU=0 DEVICE=cuda:0 BATCH=4 EPOCHS=60 PATIENCE=8 TEST_AFTER_TRAIN=1 \
SAVE_DIR=runs/semseg/attnmap bash run_semseg_preset.sh baseline
```

### Attention Heatmap Full Result and No-Attention Ablation

Completed attention-map result:
- Run: `runs/semseg/attnmap`, 37 epochs with raw-best epoch 29 and frozen threshold `0.70`.
- Test: `oIoU=0.678791`, `mIoU=0.514869`, class-macro mIoU `0.538315`, precision `0.796597`, recall `0.821107`, F1 `0.808666`, and `Pr@0.9=0.128699`.
- Relative to `tpool`, precision improved by `0.008254`, but oIoU regressed by `0.004630`, mIoU by `0.004949`, recall by `0.015892`, and `Pr@0.9` by `0.008905`.
- Decision: reject global spatial-softmax attention as the active mask-grounding branch.

No-attention changes:
- Removed query/key spatial attention, its two scalar parameters, and the attention decoder channel.
- Retained token pooling, FiLM, similarity, visual spatial gate, value projection, decoder, backbone/neck, loss, and evaluation.
- Removed unused object/spatial/context token-mask generation, loading, forwarding, and head arguments.
- Legacy caches remain compatible; only token embeddings and `text_token_mask` are exposed to the active model.

Verification:
- Python compilation passed.
- `python -m unittest discover -s tests -p "test_semseg_*.py" -v`: 11 tests passed.
- CUDA smoke with 2 train, 2 validation, and 2 test batches passed under `runs/semseg/noattn_smoke2`.
- Smoke trainable parameter count: `2,631,752`.

Planned full command:
```bash
GPU=0 DEVICE=cuda:0 BATCH=4 EPOCHS=60 PATIENCE=8 TEST_AFTER_TRAIN=1 \
SAVE_DIR=runs/semseg/noattn bash run_semseg_preset.sh baseline
```

## 2026-08-21

### RRSIS-D Mask-Size Reconciliation and Axis-Aware Flip Ablation

Context:
- The first `runs/semseg/noattn` process stopped after epoch 11 and did not produce `test_results.json`; it is an incomplete run and is not accepted as an ablation result.
- Auditing `instances.json`, JPEG headers, annotations, and refs found 17 non-`800 x 800` source images. Three have matching RLE sizes; the remaining 14 have actual JPEG heights from 784 to 813 but `800 x 800` RLE masks.
- The 14 mismatches are distributed as `train=9`, `val=2`, and `test=3`. Width is 800 for every affected image, and the official image metadata matches the decoded JPEG dimensions.

Changes:
- Added explicit nearest-neighbor mask-to-image alignment before the common training resize, preserving all official samples and binary mask values.
- Split direction detection into horizontal and vertical axes.
- Added `above` and `below` as vertical direction words.
- Added `--augment-direction-policy legacy|axis-aware`; `axis-aware` is the new default, while `legacy` reproduces the former rule that any directional word blocks both flips.
- Both policies consume the same two per-sample flip draws before policy gating, keeping later color-jitter randomness aligned in the strict ablation.
- Added `AUGMENT_DIRECTION_POLICY` support to `run_semseg_preset.sh`.
- Added directed tests for axis classification, horizontal-only blocking, vertical-only blocking, legacy behavior, and mismatched mask alignment.

Verification completed before full runs:
- `python -m py_compile dataset/rrsisd_refseg_dataset.py train_semseg.py tests/test_rrsisd_axis_aware_augmentation.py`: passed.
- `python tests/test_rrsisd_axis_aware_augmentation.py -v`: 5 tests passed.
- `python -m unittest discover -s tests -p 'test_*.py' -v`: 16 tests passed in the WSL/PyTorch environment.
- Loaded all 14 mismatched official samples without final resizing and verified each repaired image/mask shape pair exactly matches.
- Axis-aware CUDA smoke completed 2 train, 2 validation, and 2 test batches under `runs/semseg/axis_aug_smoke`; strict checkpoint loading and `test_results.json` generation passed.
- `bash -n run_semseg_preset.sh`: passed.

Controlled experiment plan:
```bash
GPU=0 DEVICE=cuda:0 BATCH=4 EPOCHS=60 PATIENCE=8 TEST_AFTER_TRAIN=1 \
AUGMENT_DIRECTION_POLICY=legacy SAVE_DIR=runs/semseg/noattn_aug_legacy \
bash run_semseg_preset.sh baseline

GPU=0 DEVICE=cuda:0 BATCH=4 EPOCHS=60 PATIENCE=8 TEST_AFTER_TRAIN=1 \
AUGMENT_DIRECTION_POLICY=axis-aware SAVE_DIR=runs/semseg/noattn_aug_axis \
bash run_semseg_preset.sh baseline
```

Control constraints:
- Same no-attention architecture, seed 42, split, image size, batch size, optimizer, scheduler, loss, sampler, validation thresholds, checkpoint selection, and frozen-threshold test protocol.
- The only experiment variable is `augment_direction_policy`.

Controlled experiment outcome:
- The sequential script recorded publish commit `4ed9ed26e842f264516f003c889dd8763000b879` and identical source hashes before launching both runs.
- `legacy` completed 50 epochs with raw-best epoch 42 and validation-selected threshold `0.80`. Test: `oIoU=0.691092`, `mIoU=0.521327`, class-macro mIoU `0.543093`, Precision `0.814823`, Recall `0.819856`, F1 `0.817332`, Pr@0.8 `0.288997`, Pr@0.9 `0.151681`, and predicted-positive rate `0.046940`.
- `axis-aware` completed 54 epochs with raw-best epoch 53 and validation-selected threshold `0.70`. Test: `oIoU=0.698654`, `mIoU=0.530917`, class-macro mIoU `0.553549`, Precision `0.794239`, Recall `0.853056`, F1 `0.822597`, Pr@0.8 `0.295030`, Pr@0.9 `0.142488`, and predicted-positive rate `0.050106`.
- Axis-aware improved oIoU by `0.007562`, official mIoU by `0.009590`, class-macro mIoU by `0.010456`, Recall by `0.033200`, F1 by `0.005265`, and Pr@0.8 by `0.006033`. Precision declined by `0.020585` and Pr@0.9 by `0.009193`.
- Class mIoU improved for 14 of 20 classes. Largest gains: Expressway-Service-area `+0.048046`, harbor `+0.044988`, tenniscourt `+0.041026`, ship `+0.023354`, and vehicle `+0.016611`. Largest regression: stadium `-0.025539`.
- Decision: retain the no-attention head and `axis-aware` augmentation default. Keep `legacy` available for reproduction and record the Precision/Pr@0.9 tradeoff as a follow-up risk.
- Verified final artifacts: `best_raw.pt`, `test_results.json`, and `test_confusion_matrix.png` exist for both runs; the sequential training process exited normally.

## 2026-08-22

### Final Standardized Test Evaluation

Scope:
- Re-evaluated the accepted final checkpoint; no retraining and no model-source changes were performed.
- Checkpoint: `HFSA-main/runs/semseg/noattn_aug_axis/weights/best_raw.pt`, raw-best epoch 53.
- Protocol: official RRSIS-D test split, 3,481 expression-mask samples, image size 512, frozen validation-selected threshold `0.70`.
- Reproduction command: `PYTHONPATH=. python /mnt/d/code/python/HFSA/tmp/run_final_standardized_evaluation.py` in the WSL project environment.

Final official metrics:
- `oIoU/cIoU=0.6986542302`
- `mIoU/gIoU=0.5309167877`
- `Pr@0.5/0.6/0.7/0.8/0.9=0.6001149095/0.5168055157/0.4073542086/0.2950301637/0.1424877909`

Required diagnostics:
- `Precision=0.7942385365`, `Recall=0.8530562803`, `F1=0.8225973453`
- Predicted-positive rate `0.0501062349`; target-positive rate `0.0466514386`

Resource evidence on NVIDIA GeForce RTX 4060 Laptop GPU:
- Segmentation parameters: `3,971,624`; currently loaded full system including OpenCLIP: `431,588,137`.
- Segmentation checkpoint `35.57 MB`; OpenCLIP weights `889.56 MB`; combined weight files `925.12 MB`.
- Cached-text batch-1 mean/P95: `30.13/45.11 ms`, peak GPU memory `123.04 MB`.
- Raw-text end-to-end batch-1 mean/P95: `141.48/179.83 ms`, peak GPU memory `1,754.27 MB`.
- Peak process CPU RSS: `4,638.53 MB`.

Reporting decision and verification:
- Core metrics are fixed to oIoU, mIoU, and Pr@0.5-0.9.
- Final reports omit per-category IoU, class-macro-mIoU, pixel accuracy, background/foreground binary mIoU, class-oIoU, and legacy aliases; internal compatibility behavior remains unchanged.
- The re-evaluated official and diagnostic metrics match the previous `test_results.json` values, confirming deterministic metric reproduction for the fixed checkpoint and threshold.
- Outputs: `HFSA-main/runs/semseg/noattn_aug_axis/final_evaluation_20260822.json` and `HFSA-main/runs/semseg/noattn_aug_axis/final_evaluation_20260822.md`.
- Source/checkpoint SHA256 values are embedded in the JSON report.
- Git commit and push were not possible because `D:\code\python\HFSA` is not a Git work tree; no repository was initialized or overwritten.

## 2026-08-23

### Target-Background Twin-Stream Decoder Candidate

Context:
- The accepted comparison baseline is `runs/semseg/noattn_aug_axis`: test `oIoU=0.698654`, `mIoU=0.530917`, and `Pr@0.5-0.9=0.600115/0.516806/0.407354/0.295030/0.142488`.
- The experiment is constrained to the post-fusion binary-mask decoder. YOLOv12 backbone, neck, OpenCLIP, dataset, loss, augmentation, checkpoint selection, and evaluation protocol remain fixed.
- Experiment source was published as commit `c718f78931930adacff123e74fa4cb03b63607b6` before full training.

Changes:
- Replaced the single final decoder in `TextPromptSegment` with symmetric, parameter-independent `target_decoder` and `background_decoder` branches.
- Kept the decoder input unchanged: gated visual features, gated value features, and pixel-text similarity.
- Combined the branches as `target_logits - background_logits + similarity + bias`, preserving the external `[B,1,H,W]` logits interface.
- Added directed tests for input channels, parameter independence, output shape, gradients through both branches, and subtraction sign.

Controlled full command:
```bash
GPU=0 DEVICE=cuda:0 BATCH=4 EPOCHS=60 PATIENCE=8 TEST_AFTER_TRAIN=1 \
SAVE_DIR=runs/semseg/tbtd bash run_semseg_preset.sh baseline
```

Status:
- Python compilation passed.
- `python -m unittest discover -s tests -p "test_semseg_*.py" -v`: 14 tests passed.
- CUDA smoke completed 2 train, 2 validation, and 2 test batches under `runs/semseg/tbtd_smoke_20260823`; `best_raw.pt`, strict checkpoint loading, and `test_results.json` generation passed.
- Smoke trainable parameter count is `3,075,913`, an increase of `444,161` over the no-attention single-decoder candidate.
- Full seed-42 training stopped at epoch 57 by patience=8; raw-best was epoch 49 with frozen validation threshold `0.70`.
- Full test over 3,481 samples: `oIoU=0.693151`, `mIoU=0.532644`, `Pr@0.5-0.9=0.597817/0.517380/0.410227/0.299052/0.145361`, `Precision=0.802104`, `Recall=0.836144`, and `F1=0.818771`.
- Test evaluation took `95.12s`; `best_raw.pt` is `40.66 MB`. Artifacts are under `runs/semseg/tbtd` and are not committed.
- Relative to `noattn_aug_axis`: oIoU `-0.005503`, mIoU `+0.001727`, Pr@0.5 `-0.002298`, Pr@0.6/0.7/0.8/0.9 `+0.000575/+0.002873/+0.004022/+0.002873`, Precision `+0.007866`, Recall `-0.016912`, and F1 `-0.003827`.
- Decision: do not replace the active single-decoder baseline. Twin-stream decoding slightly improves sample mIoU and higher-IoU success rates while reducing over-segmentation, but the oIoU and recall regression makes it a mixed, insufficient gain.
- Mainline cleanup: restored the active single `mask_decoder` and its regression test after recording the experiment. The complete twin-stream implementation remains reproducible at commit `c718f78931930adacff123e74fa4cb03b63607b6`; experiment artifacts remain under `runs/semseg/tbtd`.

### Image-Conditioned Bidirectional Token Adapter Experiment

Context and implementation:
- The accepted comparison baseline remained `runs/semseg/noattn_aug_axis`: test `oIoU=0.698654`, `mIoU=0.530917`, and `Pr@0.5-0.9=0.600115/0.516806/0.407354/0.295030/0.142488`.
- Added a zero-initialized residual adapter only inside `TextPromptSegment`: pooled `8 x 8` visual regions condition text tokens, then adapted text conditions visual regions.
- Kept backbone, neck, OpenCLIP, P3/P4/P5 inputs, single decoder, loss, data, axis-aware augmentation, seed, checkpoint selection, and evaluation fixed.
- Candidate `head.py` SHA256 before cleanup: `6EBB30F2E11E2F75DBFD35743269D5AF8D18EE1DD2EF98235DDA4C624D40A61D`.

Verification and run:
- All 14 semantic-segmentation directed/regression tests passed.
- CUDA smoke passed 2 train, 2 validation, and 2 test batches, including strict checkpoint reload and report generation.
- Full run: `HFSA-main/runs/semseg/bta_axis`; early stopped at epoch 55; raw-best epoch 47; validation-selected threshold `0.80`; full test 3,481 samples.
- Full command used the existing baseline preset with `GPU=0 DEVICE=cuda:0 BATCH=4 EPOCHS=60 PATIENCE=8 TEST_AFTER_TRAIN=1 SAVE_DIR=runs/semseg/bta_axis`.

Test result:
- Candidate: `oIoU=0.694395`, `mIoU=0.539623`, `Pr@0.5-0.9=0.602126/0.521689/0.422867/0.304797/0.152255`, Precision `0.803445`, Recall `0.836496`, F1 `0.819638`.
- Relative to baseline: oIoU `-0.004259`, mIoU `+0.008707`, Pr@0.5/0.6/0.7/0.8/0.9 `+0.002011/+0.004884/+0.015513/+0.009767/+0.009767`, Precision `+0.009207`, Recall `-0.016560`, F1 `-0.002960`.
- Predicted-positive rate was `0.048571` against target `0.046651`, closer than the baseline's `0.050106`.
- Parameters `4,119,978` (`+148,354`), checkpoint `37.29 MB` (`+1.72 MB`), test latency `29.33 ms/sample` versus `23.17`, peak GPU `370.35 MB` versus `336.51`.
- Checkpoint gate audit: `tanh(text_gate)=-0.015298`, `tanh(visual_gate)=-0.064627`; the adapter learned a nonzero contribution.

Decision and cleanup:
- Reject promotion. The mIoU and all Pr metrics improve, but oIoU, Recall, F1, latency, memory, and size regress; this is not the clear comprehensive improvement required for replacing the current optimum.
- Restored active `head.py` to the published single-decoder no-attention baseline and removed the candidate-only test after preserving the experiment evidence.
- Kept `runs/semseg/bta_axis` locally. Per user instruction, no candidate code or documentation was pushed to GitHub.

## 2026-08-24

### Text-Persistent Progressive Decoder Experiment

Context and implementation:
- Compared against `runs/semseg/noattn_aug_axis`: test `oIoU=0.698654`, `mIoU=0.530917`, and `Pr@0.5-0.9=0.600115/0.516806/0.407354/0.295030/0.142488`.
- Replaced one-shot P3-aligned fusion with a shared-parameter P5→P4→P3 top-down decoder inside `TextPromptSegment`.
- Every scale reused text FiLM, pixel-text similarity, spatial gate, value projection, and the same mask decoder.
- P5/P4 internal mask residuals were controlled by zero-initialized learned gates; only the final combined mask received the existing loss.
- Backbone, neck, OpenCLIP, data, loss, seed, axis-aware augmentation, checkpoint selection, and evaluation protocol were unchanged.

Verification and run:
- Python compilation passed.
- `python -m unittest discover -s tests -p "test_*.py" -v`: 20 tests passed.
- CUDA smoke passed 2 train, 2 validation, and 2 test batches under `runs/semseg/tpd_smoke_20260823`, including strict checkpoint reload and report generation.
- Training and local publish copies had the same CRLF-normalized `head.py` SHA256: `a55a8799b056285717b9adac309cd91d2a321e0c3e2011b2a08e3fb03c0ca9e8`.
- Full command: `GPU=0 DEVICE=cuda:0 BATCH=4 EPOCHS=60 PATIENCE=8 TEST_AFTER_TRAIN=1 SAVE_DIR=runs/semseg/tpd_axis bash run_semseg_preset.sh baseline`.
- Full run early stopped at epoch 37; raw-best epoch 29; frozen validation threshold `0.60`; full test 3,481 samples.

Test result:
- Candidate: `oIoU=0.685062`, `mIoU=0.511833`, `Pr@0.5-0.9=0.557886/0.482045/0.377190/0.275783/0.130997`, Precision `0.794161`, Recall `0.832965`, F1 `0.813100`.
- Relative to baseline: oIoU `-0.013592`, mIoU `-0.019084`, every Pr metric regressed, Recall `-0.020092`, and F1 `-0.009497`.
- Predicted-positive rate improved from `0.050106` to `0.048931` against target `0.046651`, but the better area calibration did not translate into better masks.
- Parameters `4,267,050` (`+295,426`), checkpoint `38.96 MB` (`+3.39 MB`), mean test latency `28.64 ms/sample` versus `23.17`, peak GPU `338.38 MB` versus `336.51`.
- Learned coarse-gate tanh values were `0.325729/0.439594`, confirming that both progressive residuals were active.

Decision:
- Reject promotion. The candidate is worse on all core metrics and less efficient.
- Restore the published no-attention, axis-aware, single-decoder baseline before starting the learned target/relation/position token-pooling experiment.
- Keep `runs/semseg/tpd_axis` locally; do not push candidate code to GitHub.

### Markdown Source Migration for AI Reading

Scope:
- Registered the MinerU Markdown conversions as the preferred AI-readable sources for the competition plan, two organizer Q&A documents, and the local RRSIS literature corpus.
- Updated `PROJECT_RULES.md`, `AGENTS.md`, and `LITERATURE_READING_GUIDE.md`; no model, data, training, or evaluation code changed.

Verification:
- Confirmed 19 root-level `MinerU_markdown_*.md` files totaling about 1.07 MB.
- Confirmed the competition plan and both Q&A conversions contain searchable headings and text.
- Confirmed 16 converted papers contain external MinerU image references; rules now require cross-checking figures, tables, formulas, and OCR-sensitive claims.
- Kept the 19 converted source documents local and out of the public Git commit because they contain full paper text and organizer contact information; public redistribution requires explicit user approval.
## 2026-08-24: Learned Semantic-Role Token Pooling Full Experiment

Scope:
- Added learned target/relation/position token pooling only inside `TextPromptSegment`.
- Kept backbone, neck, OpenCLIP, data, axis-aware augmentation, BCE-Tversky loss, seed 42, batch 4, image size 512, mIoU checkpoint selection, `best_raw.pt`, and frozen-threshold test protocol unchanged.

Verification before the full run:
- Python compilation passed.
- All 19 repository tests passed; 14 semantic-segmentation directed tests covered initialization equivalence, role differentiation, gradients, no-attention structure, token pooling, checkpoint selection, and legacy-cache routing.
- A 2-train/2-val/2-test CUDA smoke passed under `runs/semseg/srp_smoke_20260824`.
- Normalized SHA-256 values for the training and publication copies of `head.py` and the directed test matched.

Full command:
- `GPU=0 DEVICE=cuda:0 BATCH=4 EPOCHS=60 PATIENCE=8 TEST_AFTER_TRAIN=1 SAVE_DIR=runs/semseg/srp_axis bash run_semseg_preset.sh baseline`

Outcome:
- Early stopped at epoch 52; raw-best epoch 44; frozen validation threshold `0.80`.
- Full test: `oIoU=0.700990`, `mIoU=0.539022`, `Pr@0.5-0.9=0.598391/0.516231/0.413100/0.305085/0.151106`.
- Precision/Recall/F1 `0.799135/0.850918/0.824214`; predicted/target positive rates `0.049674/0.046651`.
- Parameters `4,072,364`; checkpoint `36.72 MB`; mean test latency `28.95 ms/sample`; peak GPU `339.31 MB`.
- Relative to `noattn_aug_axis`: oIoU `+0.002336`, mIoU `+0.008105`, Pr@0.6-0.9 and F1 improved; Pr@0.5 and Recall decreased slightly.
- Role scorer rows and role biases differentiated; position gate tanh was `-0.426264`.

Decision:
- Accept as the current comprehensive best candidate and use it as the base for uncertainty-gated P2 residual.
- Do not commit or push yet; the user requested the P2 residual experiment after the first two experiments.

## 2026-08-24: Uncertainty-Gated P2 Boundary Residual Full Experiment and Final Promotion

Scope and verification:
- Added a dedicated P2 residual on top of the accepted semantic-role pooling coarse head; P2 never entered full-image P3/P4/P5 fusion.
- Used detached `4*p*(1-p)` uncertainty and a 3x3 probability morphological gradient; the hard residual mask required uncertainty `>=0.5` and boundary strength `>=0.05`.
- Residual decoder last layer was zero-initialized, and the existing single-mask BCE-Tversky loss and all data/training/evaluation settings remained fixed.
- All 23 repository tests and a 2-train/2-val/2-test CUDA smoke passed before the full run.

Full run and test:
- Run: `HFSA-main/runs/semseg/p2ubr_axis`.
- Early stopped at epoch 52; raw-best epoch 44; frozen threshold `0.80`; full test 3,481 samples.
- Test: `oIoU=0.694145`, `mIoU=0.535108`, `Pr@0.5-0.9=0.593795/0.510773/0.404194/0.300201/0.151681`.
- Precision/Recall/F1 `0.790502/0.850628/0.819463`; predicted/target positive rates `0.050200/0.046651`.
- Parameters `4,821,421`; checkpoint `45.32 MB`; mean evaluation time `86.04 ms/sample`; peak GPU `558.66 MB`.
- Relative to `srp_axis`: oIoU `-0.006845`, mIoU `-0.003914`, Pr@0.5-0.8 and F1 regressed; only Pr@0.9 improved by `0.000575`.

Audit:
- P2 residual last-layer weight norm `1.011367`, bias magnitude `0.100263`; P2 visual projection, FiLM, and text projection all learned nonzero parameters.
- Full-test uncertainty-boundary coverage averaged `1.9474%`; per-image median `0.6104%`, P90 `5.2979%`, max `34.3018%`; `187/3481` images had zero coverage.
- The negative result therefore reflects an active, correctly localized P2 branch rather than a dead residual or an unconstrained full-image path.

Decision:
- Reject the P2 residual because it is worse on both primary IoU metrics, most Pr metrics, F1, parameters, latency, and memory.
- Remove its active config/test/head branch, retain only the ADR and local run evidence, and restore learned target/relation/position token pooling as the comprehensive best.
- Promote and publish ADR-0015 after final regression and Git review; do not include runs, checkpoints, caches, or paper full text.

Final publication verification:
- Restored training and publication copies passed Python compilation and all 19 repository tests independently.
- A corrected 2-train/2-val/2-test CUDA smoke passed under `runs/semseg/srp_release_smoke2_20260824`, including raw-best save, strict reload, and test report generation.
- Normalized SHA-256 matched between training and publication copies: `head.py=3109e5ca32a50d09a0dcee3bf83a06bb5c88f98561e46ffc85d0073aba487230`, semantic-role test `61d12722cd315348e19eef0e4ea9258a42bbaac29219d63b3ee27f4b344ef60f`.
- Final staged scope contains only the active head, directed test, ADR-0013 through ADR-0016, architecture, development log, and thread log; `git diff --check` passed.

## 2026-08-24: Switch Semantic Segmentation Initialization to YOLOv12m

Scope:
- Kept the accepted ADR-0015 semantic-role pooling head, P3/P4/P5 wiring, OpenCLIP cache, loss, axis-aware augmentation, sampler, checkpoint selection, and evaluation unchanged.
- Changed the training entry and reusable preset from the implicit n-scale model plus `yolov12n.pt` to the matched m-scale alias `yolov12m-semseg.yaml` plus local `pretrain_model/yolov12m.pt`.
- Added ADR-0018 and a regression test covering both CLI defaults and m-scale alias resolution.

Verification before the full run:
- Static construction resolved `scale=m`; total parameters are `20,273,356`.
- Pretrained loading matched `678/762` tensors and skipped only the custom final head prefix `model.21.*`.
- With the backbone frozen, trainable/frozen parameters are `9,493,644/10,794,304`.
- Python compilation, `bash -n`, and all 21 repository tests passed.
- A 2-train/2-val/2-test CUDA smoke passed at batch 2 under `runs/semseg/srp_yolov12m_smoke_20260824`.
- A baseline-hyperparameter 2-train/2-val/2-test CUDA smoke passed at batch 4 under `runs/semseg/srp_yolov12m_b4_smoke_20260824`; checkpoint save, strict reload, and frozen-threshold test report generation all succeeded.

Planned controlled full command:
```bash
GPU=0 DEVICE=cuda:0 BATCH=4 EPOCHS=60 PATIENCE=8 TEST_AFTER_TRAIN=1 \
SAVE_DIR=runs/semseg/srp_yolov12m_axis bash run_semseg_preset.sh baseline
```

Status:
- Full seed-42 training completed at epoch 52 by patience; raw-best epoch 44 and frozen threshold `0.70`.
- Full test: `oIoU=0.701171`, `mIoU=0.552562`, `Pr@0.5-0.9=0.623959/0.540362/0.425452/0.319161/0.164321`.
- Relative to YOLOv12n `srp_axis`, mIoU improved `+0.013540` and all Pr metrics improved, while oIoU `+0.000181` and F1 `+0.000125` were effectively unchanged; Recall decreased `0.005875`.
- Parameters/checkpoint/peak test GPU changed from `4.07M/36.72 MB/339.31 MB` to `20.29M/150.33 MB/726.26 MB`.
- Decision: retain YOLOv12m as an accuracy-oriented candidate, but do not call it the comprehensive resource-efficient optimum without a deployment-priority decision and paired latency benchmark.

## 2026-08-25: Add Project-Relative Referring-Segmentation Scripts

Scope:
- Added a self-contained `scripts/train_refseg.sh` that directly calls `train_semseg.py` with the accepted YOLOv12m/RRSIS-D baseline arguments, plus `scripts/test_refseg.sh` for independent checkpoint evaluation. The task scripts do not require `run_semseg_preset.sh` in the deployment package.
- Added `train_semseg.py --eval-only --checkpoint` so testing skips train/val dataset construction and pretrained initialization, strictly reloads the requested full checkpoint, and preserves its validation-selected threshold.
- Kept backbone, neck, OpenCLIP, segmentation head, loss, data semantics, and training hyperparameters unchanged.

Verification:
- `bash -n` passed for both Shell scripts in the publication and working copies.
- Python compilation and all 25 repository tests passed; the added regression asserts that the training script contains no `run_semseg_preset.sh` reference.
- Dry-run command expansion passed with `TEST_AFTER_TRAIN=0` and `TEST_AFTER_TRAIN=1`; the latter correctly appended `--test-after-train --max-test-batches`.
- A 2-batch CUDA evaluation-only smoke loaded `srp_yolov12m_axis/weights/best_raw.pt`, reported `train samples: 0`, `val samples: 0`, `test samples: 3481`, reused threshold `0.70`, and wrote a separate `test_results.json` under `runs/semseg/srp_yolov12m_axis_eval_smoke_scripts`.
- Normalized SHA-256 hashes match between publication and working copies for `train_semseg.py`, both scripts, the script README, and the directed test.

## 2026-08-25: Audit RRSIS-D Cleaning and Augmentation

Audit:
- Scanned all 17,402 expressions: split counts `12181/1740/3481`, no duplicate IDs, no empty text, no invalid class index, and no missing segmentation field.
- Found three zero-foreground RLE annotations: train `train_22187`, `train_20203`; test `test_413`.
- Found 4,250 tiny targets at area ratio `<=0.0025`; nearest-neighbor resize to 512 did not erase any non-empty mask. Nearest area-ratio preservation had median `1.000015` and p1/p99 `0.943731/1.058417`.
- Audited 8,458 direction-bearing expressions; no candidate direction token escaped the current axis-aware blocker.

Changes:
- Added strict RLE count/size validation and direct encoded foreground-area calculation.
- Added `drop/error/keep` empty-mask policies; task scripts and training CLI default to `drop` and record the policy in checkpoint arguments.
- Stopped silently replacing missing text with a class name, prioritized standard `category_id` while retaining legacy `categories_id` compatibility, and validated flip/jitter ranges.
- Kept image/mask resize, axis-aware flips, color jitter, backbone, neck, OpenCLIP, segmentation head, loss, sampler settings, and evaluation protocol unchanged.

Verification:
- Python compilation passed; all 29 repository tests passed.
- Real dataset construction produced `12179/1740/3480` samples and reported exactly the three audited IDs.
- Batch-4 YOLOv12m CUDA smoke under `runs/semseg/srp_yolov12m_clean_empty_smoke` completed 2 train/2 val/2 test batches, loaded `678/762` pretrained tensors, saved/reloaded checkpoints, and generated `test_results.json`.

Planned controlled run:
```bash
GPU=0 DEVICE=cuda:0 BATCH=4 EPOCHS=60 PATIENCE=8 TEST_AFTER_TRAIN=1 \
SAVE_DIR=runs/semseg/srp_yolov12m_axis_clean_empty bash scripts/train_refseg.sh
```

For fair attribution, the old YOLOv12m raw-best checkpoint will also be evaluated on the cleaned 3,480-sample test split before comparing it with the newly trained model.

## 2026-08-25: Consolidate Multi-Task Training Integration Decision

Scope:
- Compared the existing text-guided detection `train.py` with the referring-segmentation `train_semseg.py` and confirmed they share YOLOv12m/OpenCLIP infrastructure but not Dataset, Head output, Loss, training loop, or evaluation protocol.
- Accepted task-specific Trainers and self-contained train/test scripts as the integration unit; a future common `train.py` is limited to thin task dispatch. Added ADR-0020 and refreshed `CURRENT_STATE.md`.
- Kept all model, data, loss, augmentation, checkpoint, and evaluation code unchanged in this documentation-only update.

Current verified result status:
- No new full model experiment was completed in this conversation.
- The latest completed YOLOv12m full result remains the historical 3,481-sample run `srp_yolov12m_axis`: `oIoU=0.701171`, `mIoU=0.552562`, frozen threshold `0.70`.
- The current cleaned protocol is `12179/1740/3480`; its batch-4 CUDA smoke passed, but the full cleaned seed-42 result remains pending and must not be invented or compared against the historical result until completed.

Next verification:
- Complete `runs/semseg/srp_yolov12m_axis_clean_empty` on A5000 and re-evaluate the old raw-best checkpoint on the same 3,480-sample test split.
- Audit each teammate task's model YAML, Head interface, Loss, dataset, checkpoint, metrics, and resource-measurement command before adding a common dispatcher.

## 2026-08-25: Integrate Object Counting as a Task-Specific Head and Class-Based Module

Scope:
- Audited the teammate repository and confirmed its counting algorithm is text-guided class-agnostic detection followed by NMS box counting, not a density-map or count-regression architecture.
- Added `CountingDetect(Detect)` as a behavior-preserving task Head and registered it in Ultralytics parsing. Added `yolov12-counting.yaml`; the standard alias resolves to m scale with `nc=1`.
- Added class-based counting configuration, VOC sample access, letterbox preprocessing, OpenCLIP prompt caching, inference, EM/MAE/RMSE evaluation, per-class reporting, and visualization under `HFSA-main/counting/`.
- Added independent `train_counting.py`, `test_counting.py`, `scripts/train_counting.sh`, and `scripts/test_counting.sh`. Removed the teammate's personal absolute paths, hardware label, and IDE assumptions while preserving the training and evaluation structure.
- Kept Backbone, Neck, OpenCLIP, detection Loss, segmentation Head, segmentation training, and the isolated teammate repository unchanged.

Protocol:
- Counting training continues to use `TextGuidedDetectionTrainer` and `TextGuidedDetectionValidator`.
- Counting configuration now actually propagates `visual_attr_include_geom=False`, `lambda_relation=0`, and `lambda_spatial_quadrant=0`, which the teammate entry assigned but its base config builder did not carry.
- Evaluation preserves the teammate positive-query VOC protocol and explicitly records `positive_voc_class_queries`; zero-count class queries remain outside the current protocol.

Verification:
- Python compilation passed for active and publication counting sources.
- `bash -n` passed for both counting scripts; `PYTHON_BIN=echo` dry runs confirmed project-relative model/data/weight/output arguments and trailing CLI override behavior.
- Six counting tests passed: Detect/CountingDetect forward equivalence, m-scale YAML/Head parsing, task configuration, metric formulas, relative CLI defaults, and script contract.
- Static model construction loaded `pretrain_model/yolov12m.pt` into `yolov12m-counting.yaml`, transferred `793/799` items, and confirmed `head=CountingDetect`, `scale=m`, `nc=1`.
- Publication repository full CPU regression passed all 39 tests with `CUDA_VISIBLE_DEVICES` empty.
- Active training-copy regression ran 36 tests; its only error was the pre-existing `test_semseg_checkpoint_recovery` import mismatch because active `train_semseg.py` has not yet received publication commit `e42b12e`. No counting test failed.
- Active and publication copies of all counting source, script, test, YAML, and Ultralytics registration files were CRLF-normalized and content-compared with no differences.

Limitations:
- No VRSBench dataset or teammate-trained counting checkpoint is bundled, so no real-image training or counting inference smoke was run.
- GPU tests were intentionally not started because the user had an active experiment; all regression tests forced CPU visibility.
# 2026-08-25 cleaned YOLOv12m 正式结果与场景分类整合

- 完成 `runs/semseg/srp_yolov12m_axis_clean_empty`：epoch 39 早停，raw-best epoch 31、阈值 0.80，3,480 条 test 得到 oIoU 0.693618、mIoU 0.539086、Pr@0.5-0.9 0.598563/0.520690/0.410920/0.299713/0.154885。
- 完成旧 epoch-44 checkpoint 的 cleaned-test 公平复评：oIoU 0.702938、mIoU 0.552721，五档 Pr 及 Precision/Recall/F1 均高于新训练。决定保留数据清洗，不发布新权重。
- 将 checkpoint 原子保存、重试、optimizer/scheduler/early-stop/RNG 完整恢复和协议一致性检查从临时 `train_semseg_resume.py` 同步回活动 `train_semseg.py`，与发布副本哈希一致。
- 审计活动副本计数任务：`CountingDetect`、`counting/`、训练/测试 Python 与 Shell 脚本、专项测试均存在；核心任务文件与发布副本一致。
- 导入 `changjingfenlei` commit `688c2a9`，阅读多尺度分类 Head、NWPU/VRSBench 数据、训练、推理、评测与 benchmark。新增 `SceneClassifyHead`、m-scale YAML、`classification/` 类化模块及训练/测试/数据准备入口。
- 验证：Python/Shell 语法通过；5 项分类专项测试通过；源 `scene_vrsbench_best.pth` Head strict load 无 missing/unexpected；6 项计数专项测试通过；CPU 全模型 forward 输出 `(2,3)`。
- 分类 CPU smoke：从现有 VRSBench 生成 21 类、每类 2 张硬链接样本；batch 2、imgsz 64、1 train/1 val batch 成功，随后 1 test batch 成功并生成 `runs/smoke/classification_eval/test_results.json`。smoke 指标不用于效果结论。
- GPU 当前被其他桌面/WSL 进程占用约 3.1 GB，未启动新的 GPU smoke 或长周期训练。

## 2026-08-26: Review and Harden the Integrated Scene-Classification Task

Scope:
- Independently reviewed the background integration commit against the teammate source Head, VRSBench/NWPU data flow, optimizer/scheduler, checkpoint contract, task YAML, and active HFSA copy.
- Kept `SceneClassifyHead`, Backbone/Neck freezing, preprocessing, loss, optimizer, scheduler, sampling, and checkpoint parameter names unchanged.
- Fixed three CLI/data boundaries: single-image inference now builds preprocessing without requiring `data_dir`; `split=all` merges explicit train/val/test folders instead of treating split names as classes; training rejects batch size 1 and drops only a final singleton batch required by the source `BatchNorm1d` Head.
- Synchronized the missing legacy cosine-resume regression test into the active copy; canonical `train_semseg.py` and all task implementations already matched the publication repository.

Verification:
- Scene-classification directed tests increased from 5 to 7 and all passed.
- A real single-image CPU CLI run succeeded while `--data-dir` intentionally pointed to a missing directory and returned Top-3 predictions from the saved smoke checkpoint.
- Fresh CPU smoke under `runs/smoke/classification_review_20260826`: 21 classes, batch 2, imgsz 64, 1 train/1 val batch; YOLOv12m pretrained loading matched 678 tensors and skipped 121, with 1,304,155 trainable Head parameters and 18,198,080 frozen parameters. The saved checkpoint was strictly reloaded for a 1-test-batch evaluation.
- Active HFSA full CPU regression passed 46 tests; publication repository full CPU regression passed 46 tests. GPU smoke was not started because another GPU workload remained active.

## 2026-08-26: Remove Teammate Source Snapshots and Integrate the Scene Loss

Scope:
- Added `SceneClassificationLoss(nn.Module)` to `HFSA-main/ultralytics/utils/loss.py`. It validates `[B, C]` logits and batch-aligned labels, then applies the same unsmoothed `nn.CrossEntropyLoss` used by the teammate implementation.
- Updated both `SceneClassificationTrainer` and `SceneClassificationEvaluator` to use the explicit task Loss class; no optimizer, scheduler, Head, checkpoint, preprocessing, sampling, or metric formula changed.
- Removed the complete active `HFSA-Object-Counting` and `scene_classification_reference` nested repositories after verifying that integrated counting/classification code had no runtime imports from them. The active directories were sent to the Windows Recycle Bin.
- Removed all 11 tracked scene-classification upstream source files from the publication repository. ADR-0021/0022 retain algorithm decisions and the upstream classification commit for provenance.
- Added the permanent delivery rule that teammate repositories are temporary audit inputs only and must not remain in the final activity tree or publication repository after integration.

Verification:
- Scene-classification directed tests increased to 8 and verified that `SceneClassificationLoss` is numerically identical to `torch.nn.functional.cross_entropy` and propagates gradients.
- Active and publication repositories independently passed all 47 CPU tests after the upstream repositories were removed.
- Fresh CPU smoke under `runs/smoke/classification_loss_review_20260826` completed 1 train and 1 validation batch through `SceneClassificationTrainer -> SceneClassificationLoss -> backward`, saved `best.pt`, strictly reloaded it, and completed a 1-test-batch evaluation.
- The smoke retained 678/799 matched/skipped pretrained tensors and 1,304,155 trainable versus 18,198,080 frozen parameters. GPU smoke and complete scene training remain pending because another GPU workload was active.
