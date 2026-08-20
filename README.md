# HFSA Text-Guided Segmentation

This repository contains the code needed for the HFSA text-guided remote-sensing segmentation experiments.

## Included

- `HFSA-main/train_semseg.py`: training and validation entry point for text-guided binary masks.
- `HFSA-main/dataset/`: RRSIS-D referring segmentation dataset utilities.
- `HFSA-main/text_encoder/`: text embedding and text-related helper modules.
- `HFSA-main/ultralytics/`: local Ultralytics/YOLOv12 code with the semantic segmentation head.
- `HFSA-main/ultralytics/cfg/models/v12/yolov12-semseg.yaml`: main non-P2 segmentation model.
- `HFSA-main/ultralytics/cfg/models/v12/yolov12-semseg-p2.yaml`: experimental P2 model.
- `HFSA-main/run_semseg_preset.sh`: reusable training presets.
- Project notes: `ARCHITECTURE.md`, `DEVELOPMENT_LOG.md`, `THREAD_CHANGE_LOG.md`, `PROJECT_RULES.md`, `LITERATURE_READING_GUIDE.md`.

## Excluded

Large or machine-local artifacts are intentionally not tracked:

- datasets under `HFSA-main/data/`
- generated JSONL splits and OpenCLIP embedding caches, except `data.yaml`
- training outputs under `HFSA-main/runs/`
- pretrained/checkpoint weights such as `*.pt`
- virtual environments, PDF notes, images, and temporary files

## Key Files For Future Head Changes

- `HFSA-main/ultralytics/nn/modules/head.py`
  - `TextPromptSegment` is the current text-conditioned segmentation head.
- `HFSA-main/ultralytics/cfg/models/v12/yolov12-semseg.yaml`
  - non-P2 baseline model wiring.
- `HFSA-main/ultralytics/cfg/models/v12/yolov12-semseg-p2.yaml`
  - P2/P3/P4/P5 experimental wiring.
- `HFSA-main/ultralytics/utils/loss.py`
  - `SemanticSegmentationLoss` contains BCE, Tversky/Dice-style loss, and optional false-positive penalties.

## Typical Training

After preparing the RRSIS-D dataset locally and placing the pretrained YOLO weight file, run from `HFSA-main`:

```bash
bash run_semseg_preset.sh baseline
```

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
