# HFSA Architecture

This document records the current architecture facts needed to continue the HFSA semantic segmentation branch. `PROJECT_RULES.md` remains the single source of truth for development rules.

## Project Boundary

HFSA targets multimodal remote-sensing interpretation. The current personal branch focuses on text-guided referring semantic segmentation on RRSIS-D:

- Input: one remote-sensing image and one free-text referring expression.
- Output: one binary mask for the referred target.
- Main entries: `HFSA-main/train_refseg.py` and `HFSA-main/test_refseg.py`; the obsolete `train_semseg.py` compatibility wrapper has been removed.
- Dataset adapter: `HFSA-main/dataset/rrsisd_refseg_dataset.py`.
- Main model config: the shared `HFSA-main/ultralytics/cfg/models/v12/yolov12-semseg.yaml`, invoked through the standard `yolov12m-semseg.yaml` scale alias.

The project baseline and core YOLO/OpenCLIP code were mostly completed by the senior teammate. This branch should not modify backbone, neck, OpenCLIP encoder, or general Ultralytics internals unless explicitly reviewed and approved.

## Core Modules

- Data layer: parses RRSIS-D referring segmentation metadata, reads images, decodes binary masks, loads cached text embeddings, and applies train-only lightweight augmentation.
- Text embedding layer: uses cached OpenCLIP text vectors, currently expected to match `text_dim=768`.
- Model layer: uses YOLOv12 backbone/neck with a text-guided segmentation head through `TextPromptSegment`.
- Task layer: `tasks/refseg/engine.py` builds datasets, samplers, model, loss, metrics, plots, and run artifacts; `tasks/refseg/checkpoint.py` owns atomic persistence of the sole deployable checkpoint.
- Single-image inference layer: `tasks/refseg/inference.py` strictly loads a full RefSeg checkpoint, encodes the supplied expression into OpenCLIP token features, restores the predicted probability/mask to the original image size, and emits mask/probability/overlay artifacts.
- Experiment artifacts: `HFSA-main/runs/` stores training results and should not be treated as source code.

## Dependency Direction

Allowed:

- `tasks/refseg/engine.py` depends on dataset adapters, model config, Ultralytics model construction, cached text embeddings, and `RefSegCheckpointManager`.
- Dataset adapters depend on standard libraries, NumPy, PyTorch, OpenCV/Pillow fallback, and metadata files.
- The segmentation branch may add training options, sampling policy, metrics, and segmentation-head integration logic.

Restricted:

- Dataset code must not depend on trainer state.
- Model code must not hard-code local dataset paths.
- Validation must not mutate training or dataset construction policy.
- Backbone/neck/OpenCLIP changes require explicit architecture review.

## Current Semantic Segmentation Training Policy

- Mainline model config is non-P2: `yolov12-semseg.yaml`.
- The active training default is now YOLOv12m: `yolov12m-semseg.yaml` resolves the shared config with `scale=m`, and backbone/neck initialization uses local `pretrain_model/yolov12m.pt`.
- The accepted ADR-0015 n-scale result remains the comparison baseline until the controlled YOLOv12m run completes; no n-scale checkpoint is loaded into the m-scale model.
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
- `tasks/refseg/engine.py` still concentrates data construction, metrics, visualization, and the training loop; checkpoint policy and public applications have been split out, while further extraction remains incremental technical debt.
- Git repository state is currently abnormal: `git status` reports that the current root is not recognized as a Git repository despite `.git` directories being present.

## Recent Architecture Decision Status

ADR-0004 standardizes the RRSIS-D evaluation protocol while retaining legacy result fields for historical compatibility.

ADR-0018 switches the requested mainline training initialization from matched YOLOv12n model/weights to matched YOLOv12m model/weights while retaining the ADR-0015 segmentation head and the established seed-42 protocol.

The completed YOLOv12m run improved test mIoU from `0.539022` to `0.552562` and every Pr@0.5-0.9 metric, while test oIoU changed only from `0.700990` to `0.701171`. Total parameters increased from `4.07M` to `20.29M`, checkpoint size from `36.72 MB` to `150.33 MB`, and reported peak test GPU memory from `339.31 MB` to `726.26 MB`. YOLOv12m is therefore retained as an accuracy-oriented candidate under the user-requested default, not yet established as the resource-efficient comprehensive optimum.

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

The semantic-segmentation baseline selects both the validation threshold and checkpoint score by official sample mIoU. The sole checkpoint artifact is `weights/best_raw.pt`, saved on every strict raw maximum without applying `min_delta`. Its `hfsa_refseg_deployment_v1` payload excludes optimizer, scheduler, RNG and restart state. RefSeg no longer exposes resume-state saving or legacy `best.pt` fallback; interrupted training restarts from the beginning. Test evaluation reuses the checkpoint's frozen validation threshold. See ADR-0007 and ADR-0029.

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

The controlled run stopped at epoch 57 and selected raw-best epoch 49 with threshold `0.70`. Test results were `oIoU=0.693151`, `mIoU=0.532644`, and `Pr@0.5-0.9=0.597817/0.517380/0.410227/0.299052/0.145361`. Compared with `noattn_aug_axis`, mIoU and Pr@0.6-0.9 improved slightly, while oIoU, Pr@0.5, Recall, and F1 regressed. The lower predicted-positive rate reduced over-segmentation, but the corresponding recall loss prevented an aggregate improvement. The twin-stream decoder is therefore rejected as the active mainline; the single-decoder no-attention axis-aware model remains current.

## Rejected Image-Conditioned Bidirectional Token Adapter

ADR-0013 evaluated a low-resolution bidirectional adapter inspired by recent RRSIS vision-language interaction designs. The adapter pooled the pre-weighted P3/P4/P5 projections into `8 x 8` visual region tokens, allowed OpenCLIP text tokens to query those regions, then allowed the regions to query the adapted text. Independent zero-initialized residual gates preserved the baseline behavior at initialization. Backbone, neck, OpenCLIP, loss, decoder, data, augmentation, and evaluation were unchanged.

The full seed-42 run stopped at epoch 55 and selected raw-best epoch 47 with threshold `0.80`. Test results were `oIoU=0.694395`, `mIoU=0.539623`, and `Pr@0.5-0.9=0.602126/0.521689/0.422867/0.304797/0.152255`. Compared with `noattn_aug_axis`, mIoU increased by `0.008707` and every Pr metric increased, but oIoU decreased by `0.004259`, Recall by `0.016560`, and F1 by `0.002960`. The learned text/visual gate tanh values were `-0.015298/-0.064627`, confirming that the adapter was active. It also added 148,354 parameters and increased test latency and peak GPU memory.

Because the result is a tradeoff rather than a clear aggregate improvement, the adapter is rejected as the active architecture. The current model remains the no-attention token-pooling head with axis-aware augmentation and a single mask decoder. Local experiment artifacts remain under `runs/semseg/bta_axis`; candidate source is not published.

## Rejected Text-Persistent Progressive Decoder

ADR-0014 evaluated a P5→P4→P3 top-down mask decoder inspired by CADFormer TCMD, LSCF CLA, and SRGFormer PMR. P3/P4/P5 retained the existing text-controlled scale weights. A shared text FiLM, pixel-text similarity, spatial gate, value branch, and mask decoder were executed at every scale. P5/P4 mask logits entered the final P3 result only through zero-initialized learned residual gates, and only the final output received the existing BCE-Tversky loss.

The seed-42 run stopped at epoch 37 and selected raw-best epoch 29 with threshold `0.60`. Test results were `oIoU=0.685062`, `mIoU=0.511833`, and `Pr@0.5-0.9=0.557886/0.482045/0.377190/0.275783/0.130997`. Every core metric regressed against `noattn_aug_axis`; Recall and F1 also decreased. Parameters increased by 295,426 and mean test latency increased from `23.17` to `28.64 ms/sample`. Both coarse residual gates learned nonzero values, so the negative result reflects the active progressive path. The candidate is rejected and the single-decoder mainline is restored before the semantic-role pooling experiment.

## Active Learned Semantic-Role Token Pooling

ADR-0015 extends the no-attention head with three end-to-end token-pooling residual scorers for target, relation, and position roles. Target conditions FiLM, relation controls P3/P4/P5 scale weights, the target/relation mean supplies the primary pixel-text similarity, and position supplies an independent cosine map through a zero-initialized scalar spatial-gate path. It does not use heuristic word masks or spatial softmax.

The seed-42 run `runs/semseg/srp_axis` stopped at epoch 52 and selected raw-best epoch 44 with frozen threshold `0.80`. Full test results were `oIoU=0.700990`, `mIoU=0.539022`, and `Pr@0.5-0.9=0.598391/0.516231/0.413100/0.305085/0.151106`. Relative to `noattn_aug_axis`, both primary IoU metrics, Pr@0.6-0.9, Precision, and F1 improved. The learned role scorers and position gate were materially nonzero and differentiated.

This is the active comprehensive best architecture. The follow-up uncertainty-gated P2 residual experiment did not exceed it, so this P3/P4/P5 semantic-role pooling head is restored for publication.

## Rejected Uncertainty-Gated P2 Boundary Residual

ADR-0016 evaluated a dedicated P2 residual on top of the active semantic-role pooling head. P2 was excluded from coarse P3/P4/P5 fusion and could modify logits only where detached coarse predictions were both uncertain and locally boundary-like. The residual decoder was zero-initialized, used P2 visual/text similarity plus coarse context, and retained the existing single-mask BCE-Tversky supervision.

The seed-42 run `runs/semseg/p2ubr_axis` stopped at epoch 52 and selected raw-best epoch 44 with threshold `0.80`. Full test results were `oIoU=0.694145`, `mIoU=0.535108`, and `Pr@0.5-0.9=0.593795/0.510773/0.404194/0.300201/0.151681`. Relative to `srp_axis`, both primary IoU metrics, Pr@0.5-0.8, Precision, Recall, and F1 regressed; only Pr@0.9 improved by `0.000575`.

The residual last-layer weight norm was `1.011367`, and the complete test mask covered `1.9474%` of pixels on average, confirming that the P2 path learned and obeyed the boundary-only constraint. It nevertheless added `749,057` parameters and increased mean evaluation time from `28.95` to `86.04 ms/sample` and peak GPU memory from `339.31` to `558.66 MB`. The candidate is rejected; its active source/config/test are removed, while the local run and ADR retain the negative evidence.

## Referring-Segmentation Task Execution

`HFSA-main/scripts/train_refseg.sh` is the self-contained task-level training wrapper. It locates `HFSA-main`, changes to that directory, and invokes the thin `train_refseg.py` entry with the accepted baseline parameters: YOLOv12m, RRSIS-D, batch 4, 60 epochs, and patience 8. It performs training plus per-epoch validation only; test evaluation is a separate `scripts/test_refseg.sh` step. The training wrapper has no runtime dependency on `run_semseg_preset.sh`; environment variables and trailing CLI arguments may override the defaults.

`HFSA-main/scripts/test_refseg.sh` is the independent evaluation wrapper. It invokes `test_refseg.py --checkpoint ...`; `RefSegEvaluationApplication` supplies evaluation-only mode, builds only the official test dataset/cache/loader, strictly loads the full checkpoint, reuses the checkpoint's stored validation threshold, and writes to a separate evaluation directory. It does not build the train or validation datasets and does not load YOLO pretraining weights. All wrapper paths are repository-relative; no machine-specific drive or `/mnt` path is embedded.

## RRSIS-D Validation and Empty-Mask Cleaning

`RRSISDRefSegDataset` now validates stable sample IDs, non-empty referring expressions, non-negative class indices, RLE dimensions/count sums, and augmentation ranges before training. Encoded foreground area is computed directly from RLE runs, without allocating a dense mask during dataset construction.

The default `empty_mask_policy=drop` explicitly removes zero-foreground annotations and reports their IDs. The audited split changes from `12181/1740/3481` to `12179/1740/3480`, removing `train_22187`, `train_20203`, and `test_413`. `error` supports strict audits and `keep` preserves the historical protocol. Empty masks are not repaired from bbox because that would introduce unsupported rectangular pseudo-labels.

The resize path remains bilinear for images and nearest-neighbor for masks: the full audit found no non-empty mask that became empty at 512, while area-preserving threshold resize could inflate some target areas by up to `1.528x`. Axis-aware flips and color jitter strength `0.15` remain unchanged, so the controlled YOLOv12m experiment changes only the confirmed empty-annotation policy.

## Multi-Task Training and Inference Routing

ADR-0020 and ADR-0023 fix the integration boundary at a shared YOLOv12m Backbone/Neck plus task-specific Heads, Trainers, datasets, losses, checkpoints, and evaluators. Referring segmentation, counting, and classification now live under `HFSA-main/tasks/` with thin root entries. Detection intentionally continues to use its existing `train.py`, `val.py`, and `text_encoder/` pipeline and was not reorganized in this change.

Each task owns self-contained `scripts/train_<task>.sh` and `scripts/test_<task>.sh` wrappers. A future common `train.py` may parse a task name or number and dispatch to the corresponding Trainer, but it must not become a combined task implementation. The final interactive task switch is a separate inference router that selects the task configuration, Head, checkpoint, preprocessing, and postprocessing. This decision does not authorize joint multi-dataset or simultaneous multi-Head training.

The inference router uses explicit manual task selection. After selection, each task requests only its own inputs and retains its own output type: RefSeg uses image plus text and returns a mask; counting uses image plus target text and returns count/boxes; classification uses only an image and returns class probabilities. The router must not impose a universal image-text or mask response contract.

## Local Web Interface and Routing Shell

`HFSA-main/web_app.py` is the current local browser entry. It uses the Python standard-library HTTP server and serves dependency-free assets from `HFSA-main/web/`; no model framework is imported merely to render the page. The browser reads `/api/tasks`, renders three manual task cards and builds the input form from the task-owned schema.

`HFSA-main/tasks/routing/` separates the stable control-plane boundary from pending model interfaces:

- `config.py` owns JSON-serializable task/input/output definitions.
- `registry.py` owns display order, project-relative checkpoint defaults, user-facing labels and the three distinct contracts.
- `router.py` owns task lookup, schema validation, lazy Adapter factory registration, instance reuse and release.
- `adapters/refseg.py` owns browser image decoding, bounded Chinese-to-English prompt translation, lazy `RefSegPredictor` construction, serialized prediction execution, PNG data-URL encoding and model resource release.

The default Web router registers only RefSeg. Registration itself imports no model code; the adapter and `RefSegPredictor` are constructed on the first prediction request, then cached for later requests. RefSeg accepts a browser image payload plus text and returns task-owned summary metadata together with overlay, mask and probability PNG data URLs. Counting and classification remain unregistered and return `interface_pending`; they must later receive separate thin Adapters around their own predictors and keep their own output schemas.

`web_app.py` limits request JSON to 64 MiB, converts uncaught inference failures into structured `inference_error` responses and closes cached adapters with the server. The native front end limits raw image files to 40 MiB, renders RefSeg-specific results and exposes direct PNG downloads. Public hosting, authentication, HTTPS and reverse proxy configuration remain outside this local interface module.

The RefSeg checkpoint remains in its English OpenCLIP prompt domain. The Web adapter performs deterministic offline translation only for supported RRSIS-D categories and basic position/color/size modifiers, records both the input and model prompts, and rejects unsupported multi-category or complex Chinese relations. It does not replace the text encoder or claim general machine translation. An empty binary mask is rendered as a no-target result rather than a successful semantic match. See ADR-0028.

## Text-Guided Object Counting Task

The teammate counting task is now integrated as a detection-compatible task boundary rather than a copied repository. Its algorithm remains:

```text
image + target-class prompt
-> OpenCLIP text embedding
-> TextGuidedDetectionModel with CountingDetect
-> class-agnostic detections
-> NMS
-> number of retained boxes
```

`CountingDetect` subclasses `Detect` without overriding `forward`. It exists to make the task Head explicit in YAML, checkpoints, architecture reports, and future routing while preserving the original detection tensor contract, pretrained parameter names, and loss behavior. `yolov12-counting.yaml` shares the YOLOv12 Backbone/Neck and uses a one-class Head; the standard script selects the m-scale alias and initializes from `yolov12m.pt`.

The task-specific class boundary is:

- `CountingTextConfig`: applies the teammate's counting preset, including disabled geometry/relation/quadrant terms and retained semantic enhancement options.
- `VRSCountingDataset`: reads VOC split files, images, XML annotations, and positive per-class counts.
- `CountingImagePreprocessor`: preserves the original 800-square letterbox and normalization path.
- `CountingPromptEncoder`: caches the original remote-sensing OpenCLIP class prompts.
- `ObjectCounter`: runs the text-guided model and counts NMS-filtered detections.
- `CountingEvaluator` and `CountingVisualizer`: compute EM/MAE/RMSE, per-class results, reports, and box visualizations.
- `CountingTrainingApplication` and `CountingEvaluationApplication`: own the independent CLI orchestration.

`scripts/train_counting.sh` and `scripts/test_counting.sh` follow the same project-relative, environment-overridable task-script contract as the referring-segmentation scripts. The teammate repository is not retained in the final activity tree or publication repository; provenance lives in ADR-0021 and the integrated task package is the only runtime implementation.

The current evaluator intentionally preserves the teammate's positive-query VOC protocol: only classes present in each XML are queried. Zero-count class queries are not included, so these reports must not be presented as a complete counting-QA protocol. No full VRSBench smoke or trained-checkpoint evaluation has been completed because neither the dataset nor teammate checkpoint is bundled. See ADR-0021.
# 2026-08-25 场景分类任务接入

- 场景分类继续复用统一 YOLOv12m Backbone/Neck 的 P3/P4/P5，不修改共享特征网络。
- 最终 Head 为 `SceneClassifyHead`：每个尺度独立 `1x1 Conv + BN + ReLU` 投影，同时执行可学习空间注意力池化和 GeM，拼接三尺度结果后进入三层 MLP，输出 `[B, num_classes]` logits。
- `ultralytics/cfg/models/v12/yolov12-classification.yaml` 定义共享 Backbone/Neck 与独立分类 Head；`parse_model()` 注入三尺度通道，`guess_model_task()` 识别为 `classify`。
- `tasks/classification/` 按职责拆分配置、ImageFolder 数据、VRSBench 单场景筛选、模型、指标、训练、评测和推理。分类 checkpoint 保存 Head 状态、类别顺序、模型 YAML、预训练匹配报告和配置；Backbone/Neck 继续从团队 `yolov12m.pt` 加载并冻结。

## 2026-08-26 Integrated Task-Package Layout

The three tasks currently in integration scope use the same outer dependency shape:

```text
scripts/train_<task>.sh or scripts/test_<task>.sh
-> thin root train_<task>.py or test_<task>.py
-> tasks/<task>/ application and engine classes
-> task-specific Head/Loss in the shared Ultralytics fork
```

`tasks/refseg/checkpoint.py` is intentionally minimal and persists only the inference/test `best_raw.pt`; early stopping remains in the training loop but restart state is not serialized. Classification stores a Head-only checkpoint in its trainer. Counting uses the existing Ultralytics detection checkpoint contract. File-name symmetry is not required when the task framework owns different responsibilities.
- `SceneClassificationLoss` 位于统一 `ultralytics/utils/loss.py`，封装原单标签 CrossEntropy；Trainer 和 Evaluator 均调用该任务 Loss 类，不依赖上游训练脚本。
- 推理预处理可通过 `SceneDataModule.build_transform()` 独立构建，因此单图 Top-K 不依赖数据集目录；`split=all` 在显式 train/val/test 布局下合并各 split 并校验类别顺序。训练保持原 Head 的 BatchNorm 结构，并避免产生末尾单样本 batch。
- 分类、计数和指代分割保持独立 Trainer、Loss、数据与评测协议；本次未修改 OpenCLIP、`TextPromptSegment` 或现有分割/计数训练链路。
- `scripts/train_classification.sh` 固化正式训练默认值，并在默认 `data/VRSBench_scene` 尚未生成时调用 `prepare_classification_data.py`，从 VOC 风格 `data/VRSBench` 一次性构建 ImageFolder 数据；正常训练入口不要求用户手工输入数据、模型、权重或训练超参数。

## 2026-08-28 RefSeg Test Preview Batches

The referring-segmentation evaluator keeps one metric pass and the frozen validation threshold, but may now retain CPU copies of the first configured test batches for qualitative output. `--test-preview-batches` defaults to `5`; the task scripts expose the same setting as `TEST_PREVIEW_BATCHES`. Files are named `test_batch0_pred.jpg`, `test_batch1_pred.jpg`, and so on in the evaluation `SAVE_DIR`.

This is a bounded visualization/reporting path. It does not alter dataloader order, logits, threshold selection, confusion matrices, oIoU/mIoU/Pr metrics, checkpoint selection, or model state. Validation previews remain one `val_batch0_pred_epoch<N>.jpg` per epoch.


## 2026-10-09 Four-Task Official-Protocol Adapter (supersedes RefSeg-only scope)

`participant_api_starter-main/hfsa_adapter.py` is the new primary adapter module. `RealModel` delegates construction and prediction. It parses a bounded set of question forms together with response constraints, calls existing task-owned Predictors, then formats scene choice identifiers/exact enum aliases, numeric strings, target-presence Yes/No, native detection boxes or RefSeg mask enclosing boxes. Unsupported semantics raise errors handled by the unchanged official server. Two-image requests fail before any model is called.

All four checkpoints load before the HTTP listener starts. Detection and counting share the existing global-text `TextPromptEncoder` through their supported factory injection hook; RefSeg keeps its independent token encoder. Checkpoints resolve under sibling `HFSA_models/`, overridable by `HFSA_MODELS_DIR`; device is `HFSA_API_DEVICE` (`auto` by default). Classification and RefSeg YAML paths are resolved from the delivery code directory, avoiding stale checkpoint path metadata.

This explicit official-protocol exception does not change the Web's manual task selection or lazy initialization. Backbone, Neck, task Heads, OpenCLIP implementation, training and evaluation remain untouched. Counting keeps the existing positive-query deployment policy and does not add spatial counting. Scene candidates must map unambiguously to the checkpoint class list. See ADR-0035 and the API README for supported question examples and local-only startup.


## 2026-10-09 Verified ngrok HTTPS Ingress

The user subsequently authorized public ingress and selected ngrok. A signature-verified Windows ngrok 3.39.11 agent forwards HTTPS traffic to `http://127.0.0.1:9001`, which Windows can reach through this computer's existing WSL localhost forwarding. This measured route allows the agent to run on Windows while the four-model API stays in its WSL environment; system proxy, TUN, DNS and firewall settings were not changed.

The agent runs with HTTP inspection disabled and info logging. Its account authtoken lives only in the native user configuration outside the repository. The official model API still checks its separate Bearer key on health and prediction; no model credential is added to the public URL. Real HTTPS health and scene/count/presence calls succeeded without a special ngrok bypass header. The current runtime origin and restart requirements are recorded in CURRENT_STATE, not assumed constant. The initial ingress checks did not submit an official evaluation. Later user-run smoke results are recorded in CURRENT_STATE.

## 2026-10-09 Adapter Failure Diagnostics

The HFSA adapter emits one bounded JSON diagnostic per prediction to local stderr. Expected rejections have fixed AdapterFailure codes while remaining ValueError subclasses; unexpected errors expose only their type. Diagnostics include request/item identifiers, constraint type, image count, selected task, status and duration. Question text, choices, answers, image paths, credentials and raw exception messages are excluded. The official server and public HTTP response behavior are unchanged. Logging adds no unsupported model capability or fallback answer. Runtime log location and user-run smoke evidence are in CURRENT_STATE.

## 2026-10-09 Optional Private Request Replay

HFSA_API_REPLAY_DIR optionally saves a whitelisted protocol request before routing/inference to a private directory outside the project. It is off by default. Credentials, unrecognized metadata and model/gold answers are excluded; diagnostic logs still omit question text. Content hashes deduplicate requests, and a capture error never changes inference. replay_requests.py sends the captured requests sequentially to loopback HTTP only, bypasses proxies, refuses public endpoints and does not follow redirects. It checks response identifiers and answer constraints, reports no accuracy and never creates an evaluator run. This supports debugging a received request repeatedly without spending further website quota. Requests from the original uncaptured smoke cannot be recovered from audit hashes.
