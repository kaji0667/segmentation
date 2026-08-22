# HFSA Architecture

This document records the current architecture facts needed to continue the HFSA semantic segmentation branch. `PROJECT_RULES.md` remains the single source of truth for development rules.

## Project Boundary

HFSA targets multimodal remote-sensing interpretation. The current personal branch focuses on text-guided referring semantic segmentation on RRSIS-D:

- Input: one remote-sensing image and one free-text referring expression.
- Output: one binary mask for the referred target.
- Main training entry: `HFSA-main/train_semseg.py`.
- Dataset adapter: `HFSA-main/dataset/rrsisd_refseg_dataset.py`.
- Main model config: `HFSA-main/ultralytics/cfg/models/v12/yolov12-semseg.yaml`.

The project baseline and core YOLO/OpenCLIP code were mostly completed by the senior teammate. This branch should not modify backbone, neck, OpenCLIP encoder, or general Ultralytics internals unless explicitly reviewed and approved.

## Core Modules

- Data layer: parses RRSIS-D referring segmentation metadata, reads images, decodes binary masks, loads cached text embeddings, and applies train-only lightweight augmentation.
- Text embedding layer: uses cached OpenCLIP text vectors, currently expected to match `text_dim=768`.
- Model layer: uses YOLOv12 backbone/neck with a text-guided segmentation head through `TextPromptSegment`.
- Training layer: `train_semseg.py` builds datasets, samplers, model, loss, metrics, checkpoints, plots, and run artifacts.
- Experiment artifacts: `HFSA-main/runs/` stores training results and should not be treated as source code.

## Dependency Direction

Allowed:

- `train_semseg.py` depends on dataset adapters, model config, Ultralytics model construction, and cached text embeddings.
- Dataset adapters depend on standard libraries, NumPy, PyTorch, OpenCV/Pillow fallback, and metadata files.
- The segmentation branch may add training options, sampling policy, metrics, and segmentation-head integration logic.

Restricted:

- Dataset code must not depend on trainer state.
- Model code must not hard-code local dataset paths.
- Validation must not mutate training or dataset construction policy.
- Backbone/neck/OpenCLIP changes require explicit architecture review.

## Current Semantic Segmentation Training Policy

- Mainline model config is non-P2: `yolov12-semseg.yaml`.
- P2 config remains an experimental alternative, not the default mainline.
- Small-target sampling boost default is `2.0`, reduced from the previously recommended `3.0` to avoid over-amplifying small-object samples.
- Small-target sampling uses true foreground mask area from RLE metadata when available, with bbox area only as a fallback.
- Binary segmentation loss is computed per sample before averaging, with optional area-aware loss weighting for very small masks.
- Train-time augmentation is lightweight and only enabled for the training split:
  - `axis-aware` is the active policy: horizontal words block only horizontal flip, while vertical words block only vertical flip
  - `above` and `below` are vertical-axis constraints; diagonal compass words constrain both axes
  - `legacy` remains selectable for controlled ablation and blocks both flips whenever any directional/positional word is present
  - brightness/contrast jitter
- Rotation is intentionally not included because RRSIS-D text may include directional expressions such as left, right, top, bottom, above, or below.
- Fourteen official samples have JPEG heights that differ from their `800 x 800` RLE mask size (`train=9`, `val=2`, `test=3`). The dataset adapter preserves every sample and nearest-neighbor resizes the decoded binary mask to the actual JPEG coordinate size before the common training resize.

## Known Risks

- Validation loss and IoU can diverge because BCE/Dice-style losses and thresholded mask IoU optimize different surfaces.
- RRSIS-D weak classes observed in recent experiments include vehicle, harbor, windmill, and tenniscourt.
- Text-guided spatial relationships remain a hard case because global text vectors have limited token-level grounding.
- `train_semseg.py` still concentrates many responsibilities and should be split only after the current experimental baseline is stable.
- Git repository state is currently abnormal: `git status` reports that the current root is not recognized as a Git repository despite `.git` directories being present.

## Recent Architecture Decision Status

ADR-0004 standardizes the RRSIS-D evaluation protocol while retaining legacy result fields for historical compatibility.

## RRSIS-D Evaluation Protocol

The semantic segmentation entry now distinguishes the published RRSIS-D metrics from internal diagnostics:

- `oiou`: cumulative foreground intersection divided by cumulative foreground union across the split. Historical `target_iou` is equivalent.
- `official_miou`: mean of per-sample foreground IoUs. Historical `sample_miou` is equivalent.
- `Pr@0.5` through `Pr@0.9`: fraction of samples whose foreground IoU reaches the corresponding threshold.
- `class_miou`: per-category mean of sample IoUs, grouped by `class_idx`.
- `class_oiou`: per-category cumulative foreground IoU. Historical `class_iou` is equivalent and is not the paper per-category mIoU.

Validation may select a checkpoint by oIoU or official mIoU. Optional test evaluation loads the best validation checkpoint and reuses its frozen validation-selected mask threshold. It does not scan thresholds on the test split. See ADR-0004.

## Rejected Deep-Supervision Experiment

ADR-0005 evaluated fixed text-conditioned auxiliary mask losses on P3/P4 and P3-only. Neither configuration improved the official test result, and P3-only caused a clear regression. ADR-0006 therefore restores the active architecture to the no-deep-supervision learnable-gate baseline.

The current no-attention candidate has one output path: learnable token pooling, P3/P4/P5 fusion, FiLM, pixel-text similarity, one learnable similarity-gate weight, the retained value projection, and the final mask decoder. Only the final mask receives the BCE-Tversky training loss. The failed experiment directories remain under `runs/semseg/ds_p3p4` and `runs/semseg/ds_p3` for reproducibility.

## Checkpoint Selection and Retention

The standard semantic-segmentation baseline selects both the validation threshold and checkpoint score by official sample mIoU. `best.pt` retains the historical `min_delta` rule used by early stopping, while `best_raw.pt` records every strict raw maximum without applying `min_delta`. Test-after-train prefers `best_raw.pt`, falls back to legacy `best.pt`, and always reuses the checkpoint's validation-selected threshold. See ADR-0007.

## Learnable Text Token Pooling Candidate

ADR-0008 adds a lightweight token-pooling adapter inside `TextPromptSegment`. Cached OpenCLIP token features are scored by a zero-initialized `Linear(768, 1)` and one learnable valid-token bias, then reduced with softmax-weighted pooling. Zero initialization reproduces the former fixed `tokens.mean(1)` behavior, so training determines whether particular contextual tokens and valid positions should receive more weight.

This candidate adds 769 parameters and does not modify OpenCLIP, backbone, neck, P3/P4/P5 fusion, spatial-gate weights, decoder, loss, or evaluation. The heuristic object/spatial token masks are intentionally not used because they are not guaranteed to align with OpenCLIP BPE spans and may mark same-class reference objects.

The full seed-42 run improved the same-protocol mIoU-selection baseline on test from `oIoU=0.672197`, `mIoU=0.509192`, and class-macro mIoU `0.536258` to `oIoU=0.683420`, `mIoU=0.519818`, and class-macro mIoU `0.545010`. Token pooling is therefore retained as the active text aggregation path.

## Rejected Calibrated Spatial Attention Heatmap

ADR-0009 corrects the query/key attention map inside `TextPromptSegment`. The old implementation normalized key and query, divided their cosine logits by `sqrt(128)`, and then applied a 4,096-position softmax, producing an almost uniform map with `1/HW` magnitude.

The controlled seed-42 run converted the probability to bounded relative density but regressed against token pooling on test: `oIoU` fell from `0.683420` to `0.678791`, official mIoU from `0.519818` to `0.514869`, and class-macro mIoU from `0.545010` to `0.538315`. Precision increased while recall and high-IoU success rates decreased. The global spatial softmax is therefore rejected for dense mask grounding because pixels compete for fixed probability mass and the map is relative spatial rank rather than independent foreground evidence.

## No-Attention Token-Pooling Ablation

ADR-0010 removes the query/key spatial-softmax branch while retaining learnable token pooling, multi-scale fusion, FiLM, independent pixel-text similarity, the visual spatial gate, the value projection, and the decoder. The decoder input changes from `2 * embed_dim + 2` channels to `2 * embed_dim + 1` because only gated visual, gated value, and similarity remain.

The same cleanup removes the unused `text_object_mask`, `text_spatial_mask`, and `text_context_mask` interfaces and stops generating their heuristic cache fields. Existing embedding caches remain compatible because extra legacy fields are ignored. `text_token_mask` remains active and is used by learnable token pooling.

The candidate passed syntax checks, 11 directed tests, and a 2-train/2-val/2-test CUDA smoke under `runs/semseg/noattn_smoke2`. Under the legacy augmentation policy, the completed seed-42 run improved test oIoU from `0.683420` to `0.691092` and official mIoU from `0.519818` to `0.521327`, while class-macro mIoU changed from `0.545010` to `0.543093`. This supports removing the spatial-softmax branch without losing the primary aggregate metrics.

The follow-up strict augmentation ablation kept the no-attention model and every other training/evaluation setting fixed. Axis-aware flips improved test oIoU to `0.698654`, official mIoU to `0.530917`, and class-macro mIoU to `0.553549`. Recall, F1, and Pr@0.5-0.8 improved; Precision and Pr@0.9 declined. The no-attention head with axis-aware augmentation is therefore the active mainline, with the high-IoU precision tradeoff retained as a known risk.

## Target-Background Twin-Stream Decoder Candidate

ADR-0012 changes only the final decoder inside `TextPromptSegment`. The shared P3/P4/P5 projection, text-controlled scale weighting, fusion/context blocks, FiLM conditioning, pixel-text similarity, spatial gate, and value projection remain unchanged. Their concatenated tensor is decoded by two symmetric but parameter-independent branches:

```text
decoder_input
├─ target_decoder     -> target_logits
└─ background_decoder -> background_logits

mask_logits = target_logits - background_logits + similarity + bias
```

The public output remains one `[B, 1, H, W]` logit tensor, so the existing BCE-Tversky loss, checkpoint conventions, threshold selection, and oIoU/mIoU/Pr@ evaluation code require no interface changes. No auxiliary target/background loss is introduced; the controlled experiment changes only the decoder parameterization. The candidate must be trained from the same `yolov12n.pt` initialization and compared against `runs/semseg/noattn_aug_axis` under the same seed-42 axis-aware protocol before it can replace the active mainline.
