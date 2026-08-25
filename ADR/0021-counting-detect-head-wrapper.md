# ADR-0021: 使用 Detect 等价包装 Head 接入目标计数任务

## 状态

Accepted

## 背景

队友目标计数实现不是密度图或直接计数回归，而是复用文本引导类无关目标检测模型，对目标类别 Prompt 推理后执行 NMS，并把保留检测框数量作为计数结果。团队要求每个任务具有可识别的 Head 类、类化任务模块以及与指代分割一致的独立训练/测试脚本，同时不得改变队友已有算法结构或覆盖当前分割 Head。

队友仓库携带一份完整 Ultralytics、`text_encoder` 和数据准备副本，但实际新增代码只有计数配置、推理、评测和测试入口。整体复制会删除发布仓库已有的 `TextPromptSegment` 等分割扩展，并造成第三方代码分叉。

## 决策

1. 新增 `CountingDetect(Detect)` 作为目标计数任务 Head。该类不重写 `forward`，保持原 `Detect` 参数名、输出张量、Loss 契约和预训练权重兼容性。
2. 新增 `yolov12-counting.yaml`，Backbone/Neck 与统一 YOLOv12 完全一致，最后一层使用 `CountingDetect`；正式脚本通过 `yolov12m-counting.yaml` alias 选择 m scale。
3. 计数继续使用现有 `TextGuidedDetectionTrainer`、`TextGuidedDetectionValidator` 和检测 Loss。计数发生在推理后处理：文本 Prompt -> 检测 -> NMS -> 框数。
4. 将队友的配置、图像 letterbox、Prompt 缓存、VOC XML 解析、指标和可视化分别包装为类，入口由 `CountingTrainingApplication` 与 `CountingEvaluationApplication` 编排。
5. 新建 `scripts/train_counting.sh` 与 `scripts/test_counting.sh`，路径、权重、数据和输出使用项目相对默认值并允许环境变量覆盖。
6. 清除队友代码中的个人绝对路径、显卡名称和 IDE 目录假设；不复制其 Ultralytics、公共检测代码和空的 `dataset_vrs.py` 实现。
7. 当前评测保持原 positive-query VOC 协议，只查询 XML 中出现的类别。报告中必须显式标记该协议，不能等同于包含零计数类别查询的完整计数问答评测。

## 备选方案

- 继续直接使用普通 `Detect`，只增加测试脚本：拒绝，因为团队要求每个任务具有明确 Head 类型和可审计任务接口。
- 在 Head 内执行 NMS 并直接输出整数：拒绝，因为会改变训练/推理张量契约，破坏检测 Loss、导出和预训练权重兼容性。
- 新增密度图或回归式 Counting Head：拒绝，因为这会改变队友原有算法结构，需要新的标签、Loss 和完整对照实验。
- 合并队友整份 Ultralytics 副本：拒绝，因为会覆盖当前分割扩展并产生不可维护的第三方代码冲突。

## 影响

- 正面：计数任务具备明确 Head、独立入口、类化模块和统一脚本，同时保持原检测式计数逻辑。
- 正面：`CountingDetect` 与 `Detect` state dict 和 forward 等价，可直接加载 YOLOv12m 检测预训练权重。
- 代价：新增一个只表达任务身份的薄 Head 类；实际计数仍依赖 NMS 阈值和检测质量。
- 风险：当前没有队友训练 checkpoint 和本地 VRSBench 真实数据验证；静态构建不能替代真实计数 smoke。
- 测试要求：Head 前向等价、m-scale YAML 解析、计数配置、EM/MAE/RMSE、脚本相对路径、预训练权重加载及全库回归。

## 关联

- `HFSA-main/ultralytics/nn/modules/head.py`
- `HFSA-main/ultralytics/cfg/models/v12/yolov12-counting.yaml`
- `HFSA-main/counting/`
- `HFSA-main/train_counting.py`
- `HFSA-main/test_counting.py`
- `HFSA-main/scripts/train_counting.sh`
- `HFSA-main/scripts/test_counting.sh`
- `ADR/0020-task-specific-trainers-and-thin-dispatch.md`
