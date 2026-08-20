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
