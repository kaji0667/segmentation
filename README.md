# HFSA Remote-Sensing Multi-Task Heads

This repository contains the HFSA task-specific referring-segmentation, object-counting, and scene-classification heads and entry points built on the shared YOLOv12m feature network.

## Included

- `HFSA-main/train_semseg.py`: training and validation entry point for text-guided binary masks.
- `HFSA-main/dataset/`: RRSIS-D referring segmentation dataset utilities.
- `HFSA-main/text_encoder/`: text embedding and text-related helper modules.
- `HFSA-main/ultralytics/`: local Ultralytics/YOLOv12 code with the semantic segmentation head.
- `HFSA-main/ultralytics/cfg/models/v12/yolov12-semseg.yaml`: main non-P2 segmentation model.
- `HFSA-main/ultralytics/cfg/models/v12/yolov12-semseg-p2.yaml`: experimental P2 model.
- `HFSA-main/run_semseg_preset.sh`: reusable training presets.
- `HFSA-main/scripts/train_refseg.sh`: project-relative YOLOv12m training entry for the referring-segmentation task.
- `HFSA-main/scripts/test_refseg.sh`: evaluation-only entry for an existing referring-segmentation checkpoint.
- `HFSA-main/counting/`, `train_counting.py`, and `test_counting.py`: text-guided detection-based object counting.
- `HFSA-main/classification/`, `train_classification.py`, and `test_classification.py`: multi-scale single-label scene classification.
- Project notes: `CURRENT_STATE.md`, `ARCHITECTURE.md`, `DEVELOPMENT_LOG.md`, `THREAD_CHANGE_LOG.md`, `PROJECT_RULES.md`, `LITERATURE_READING_GUIDE.md`.

## Excluded

Large or machine-local artifacts are intentionally not tracked:

- datasets under `HFSA-main/data/`
- generated JSONL splits and OpenCLIP embedding caches, except `data.yaml`
- training outputs under `HFSA-main/runs/`
- pretrained/checkpoint weights such as `*.pt`
- teammate/upstream repository snapshots and nested Git metadata; ADRs retain source repository and commit provenance
- virtual environments, PDF notes, images, and temporary files

## Key Files For Future Head Changes

- `HFSA-main/ultralytics/nn/modules/head.py`
  - `TextPromptSegment` is the current text-conditioned segmentation head.
  - `CountingDetect` preserves the detection tensor/Loss contract for counting.
  - `SceneClassifyHead` preserves the teammate P3/P4/P5 spatial-attention + GeM classification architecture.
- `HFSA-main/ultralytics/cfg/models/v12/yolov12-semseg.yaml`
  - non-P2 baseline model wiring.
- `HFSA-main/ultralytics/cfg/models/v12/yolov12-semseg-p2.yaml`
  - P2/P3/P4/P5 experimental wiring.
- `HFSA-main/ultralytics/utils/loss.py`
  - `SemanticSegmentationLoss` contains BCE, Tversky/Dice-style loss, and optional false-positive penalties.
  - `SceneClassificationLoss` preserves the teammate single-label cross-entropy objective behind an explicit HFSA task class.

## Typical Training

After preparing the RRSIS-D dataset locally, place `yolov12m.pt` at `HFSA-main/pretrain_model/yolov12m.pt`. The training entry and presets resolve the shared semantic-segmentation YAML with `scale=m` so the model channels match the pretrained weight. Run from `HFSA-main`:

```bash
bash run_semseg_preset.sh baseline
```

The task-specific wrappers locate `HFSA-main` automatically, so they may be called through an absolute or relative script path from any working directory. `train_refseg.sh` embeds the accepted baseline arguments and calls `train_semseg.py` directly; it does not require `run_semseg_preset.sh` in the deployment package:

```bash
bash HFSA-main/scripts/train_refseg.sh
bash HFSA-main/scripts/test_refseg.sh
```

The second command evaluates `runs/semseg/srp_yolov12m_axis/weights/best_raw.pt` by default and writes to the separate `runs/semseg/srp_yolov12m_axis_eval` directory without retraining. See `HFSA-main/scripts/README.md` for environment-variable overrides.

The empty-mask-cleaned retraining run did not outperform that checkpoint on the same 3,480-sample cleaned test (`0.693618/0.539086` versus `0.702938/0.552721` oIoU/mIoU), so the epoch-44 checkpoint remains the release choice while `empty-mask-policy=drop` remains the data default.

Task-specific counting and scene-classification wrappers are also available:

```bash
bash HFSA-main/scripts/train_counting.sh
bash HFSA-main/scripts/test_counting.sh
python HFSA-main/prepare_classification_data.py --voc-root HFSA-main/data/VRSBench --output-dir HFSA-main/data/VRSBench_scene
bash HFSA-main/scripts/train_classification.sh
bash HFSA-main/scripts/test_classification.sh
```

Scene classification uses the unchanged YOLOv12m Backbone/Neck with an independent P3/P4/P5 Head. The released integration has passed a minimal CPU chain smoke only; no complete scene-classification result is claimed.

To evaluate the best validation checkpoint on the official test split with the validation-selected threshold frozen:

```bash
TEST_AFTER_TRAIN=1 bash run_semseg_preset.sh baseline
```

Directional flip augmentation defaults to axis-aware control: horizontal words only block horizontal flips, while vertical words (including `above` and `below`) only block vertical flips. Reproduce the former all-axis blocking policy with:

```bash
AUGMENT_DIRECTION_POLICY=legacy bash run_semseg_preset.sh baseline
```

The RRSIS-D loader also preserves and explicitly reconciles the 14 official samples whose JPEG height differs from the stored RLE mask height by resizing the decoded mask to the actual image size with nearest-neighbor interpolation.

The main RRSIS-D report fields are:

- `oiou`: cumulative foreground intersection over union, matching paper oIoU/cIoU.
- `official_miou`: mean of per-sample foreground IoUs, matching paper mIoU/gIoU.
- `pr_0_5` through `pr_0_9`: fraction of samples reaching each IoU threshold.
- `class_miou`: official per-category mean sample IoU.
- `class_oiou`: per-category cumulative foreground IoU.

When test evaluation is enabled, `test_results.json` also records parameter counts, checkpoint size, evaluation duration, mean time per sample, and peak allocated GPU memory.

The best historical local branch was:

```text
non-P2 + frozen backbone + mask-area small-target loss weight 1.5 + wide validation threshold scan
```

Current unresolved issues include validation overfitting, local mask over-segmentation, incomplete thin-object masks, and weak small-object or complex-class samples.
