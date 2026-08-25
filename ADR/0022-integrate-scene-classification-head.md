# ADR-0022: Integrate the Teammate Scene-Classification Head

## 状态

Accepted；代码与最小 smoke 已验证，真实完整训练尚未执行。

## 背景

团队场景分类仓库 `zhuoletian-collab/changjingfenlei` 使用冻结 YOLOv12m 特征提取器，从 P3/P4/P5 提取多尺度特征，并通过 `ClassifyHeadV2` 的独立通道投影、空间注意力池化、GeM 池化和三层 MLP 完成单标签场景分类。原实现由多个脚本和全局配置组成，包含个人目录、设备假设和独立 checkpoint 目录，尚未接入 HFSA 的任务 Head 解析与交付结构。

## 决策

1. 保留原 `ClassifyHeadV2` 参数命名和计算流程，包装为 `SceneClassifyHead`，通过 `parse_model()` 接收 P3/P4/P5 通道并加入任务识别。
2. 新增 `yolov12-classification.yaml`；通过 `yolov12m-classification.yaml` alias 解析 m scale，Backbone/Neck 与现有 YOLOv12 完全一致，仅替换最终任务 Head。
3. 保留冻结特征提取器、ImageNet normalization、水平/垂直翻转、ColorJitter、类别平衡采样、CrossEntropy、AdamW、warmup+cosine、梯度裁剪和按 top-1 accuracy 选模的原训练流程。
4. 将配置、ImageFolder 数据、VRSBench 场景数据准备、模型、训练、推理、评测与指标包装为独立类；新增 `train_classification.py`、`test_classification.py`、`prepare_classification_data.py` 和自包含 Shell 脚本。
5. checkpoint 继续保存独立 Head 状态和类别顺序，同时记录模型 YAML、预训练权重匹配数和完整配置；旧 `scene_vrsbench_best.pth` 的 `classify_head` 可 strict load 到新 Head。
6. 外部源码只作为可审计参考提交核心 Python 源码和 provenance；不提交嵌套 `.git`、权重、数据、runs、缓存或个人配置。

## 备选方案

- 继续手工截断检测模型并在脚本中提取 P3/P4/P5：拒绝，因为 Head 不可由 HFSA YAML 审计，任务解析和发布入口也无法统一。
- 改用 Ultralytics 单尺度 `Classify` Head：拒绝，因为会改变队友已确认的多尺度空间注意力 + GeM 算法。
- 将分类训练塞入分割或计数 Trainer：拒绝，因为数据、Loss、checkpoint 和指标契约不同，违反任务专用 Trainer 边界。

## 影响

- 不修改 Backbone、Neck、OpenCLIP、`TextPromptSegment`、现有分割或计数训练逻辑。
- m-scale 21 类模型约 `19.50M` 参数，其中分类 Head 约 `1.30M` 参数可训练。
- 真实 VRSBench 数据准备仍按“恰好一个场景类”筛选；多场景图和纯物体图被排除并写入审计报告。
- CPU 最小 smoke 使用 21 类、每类 2 张硬链接样本、batch 2、imgsz 64、1 train/1 val batch，成功生成并重新加载 checkpoint；该 smoke 只验证链路，不代表模型效果。
- 单图 Top-K 推理只需图像、checkpoint 和团队预训练权重，不强制存在 ImageFolder；显式 train/val/test 的 `all` 评测合并 split 并验证类别顺序。
- 原 Head 含 `BatchNorm1d`，因此训练 batch 必须至少为 2；当训练集长度会产生末尾单样本 batch 时，DataLoader 丢弃该末尾 batch，避免运行期 BatchNorm 错误而不改变 Head 计算和 checkpoint 参数契约。

## 关联

- 上游提交：`688c2a9ec281febb5877dcd6c29bf9eeec8c02bb`
- `HFSA-main/scene_classification_reference/PROVENANCE.md`
- `HFSA-main/classification/`
- `HFSA-main/ultralytics/nn/modules/head.py`
- `HFSA-main/ultralytics/cfg/models/v12/yolov12-classification.yaml`
- `HFSA-main/train_classification.py`
- `HFSA-main/test_classification.py`
- `HFSA-main/tests/test_scene_classification_integration.py`
