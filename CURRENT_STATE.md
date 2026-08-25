# CURRENT_STATE.md

最后更新：2026-08-25

## 当前主线

- 当前个人负责的任务是 RRSIS-D 文本引导单目标二值分割：输入遥感图像与自由文本描述，输出对应目标的 `[B,1,H,W]` mask。
- 团队已确认多任务共用 YOLOv12m Backbone 和 Neck；当前分割分支使用匹配的 `yolov12m-semseg.yaml`、`yolov12m.pt`、P3/P4/P5 和 ADR-0015 `TextPromptSegment` 语义角色 token pooling Head。
- 未经用户确认，不修改 Backbone、Neck、OpenCLIP 或其他成员任务实现。
- 当前发布代码基线为 `e42b12e`，包含空 mask 清洗与 checkpoint 恢复；工作副本的计数新增文件已与发布副本同步，但工作副本 `train_semseg.py` 仍缺少发布提交中的恢复函数，待当前实验结束后再单独同步。

## 已完成的分割任务封装

- `HFSA-main/scripts/train_refseg.sh` 是自包含训练脚本，直接调用 `train_semseg.py`，内置当前正式 YOLOv12m baseline 参数，不依赖部署时不会携带的 `run_semseg_preset.sh`。
- 训练脚本通过自身位置定位 `HFSA-main`；数据、模型、权重和输出均使用项目相对路径，并允许通过环境变量或末尾 CLI 参数覆盖。
- `HFSA-main/scripts/test_refseg.sh` 支持已有 checkpoint 的独立测试，默认输出到单独目录，不重新训练，也不覆盖原训练目录。
- `train_semseg.py` 已支持 `--eval-only --checkpoint`：只构建 test split，严格加载完整 checkpoint，并复用 checkpoint 中冻结的 validation 阈值与选模指标。
- 该封装通过 Shell 语法检查、参数展开检查、Python 编译、全库测试和 2-batch CUDA evaluation-only smoke。

## 当前数据协议

- ADR-0019 已接受显式空 mask 清洗：默认 `empty_mask_policy=drop`，剔除 `train_22187`、`train_20203` 和 `test_413`。
- 当前活动 split 为 `train=12179`、`val=1740`、`test=3480`；旧 `12181/1740/3481` 协议只能通过 `--empty-mask-policy keep` 复现。
- 不允许用 bbox 为零前景标注补矩形伪 mask。最近邻 mask resize、axis-aware 翻转和 0.15 色彩扰动保持不变。

## 当前实验结果

- 历史 YOLOv12m 完整运行：`runs/semseg/srp_yolov12m_axis`，raw-best epoch 44，冻结阈值 `0.70`，旧 3481-sample test 上 `oIoU=0.701171`、`mIoU=0.552562`、`Pr@0.5-0.9=0.623959/0.540362/0.425452/0.319161/0.164321`。
- 该结果早于空 mask 清洗，只作为历史基线；不能直接与 cleaned 3480-sample test 的新结果比较。
- cleaned 协议已经通过 batch-4、2-train/2-val/2-test CUDA smoke，输出目录为 `runs/semseg/srp_yolov12m_clean_empty_smoke`。
- cleaned 协议的完整 seed-42 训练尚未在文档中记录完成结果；正式目录计划为 `runs/semseg/srp_yolov12m_axis_clean_empty`。

## 多任务训练整合结论

- 不把检测、指代分割、计数、分类等任务的数据加载、Loss、训练循环和评测逻辑强行合并到一个巨型 `train.py`。
- 每个任务保留自己的 Trainer/训练文件和任务脚本，例如检测使用现有 `train.py`，分割使用 `train_semseg.py`，其他任务按相同规范提供独立入口。
- 如果最终需要统一训练命令，公共 `train.py` 只能作为薄分发器：解析 `--task` 后调用对应任务 Trainer，不在分发器中实现具体数据、Loss 或指标逻辑。
- 最终“输入 1/2/3/4/5 或任务名称后切换任务”属于统一推理入口，与训练脚本分开设计；训练脚本不承担在线任务切换。
- 该决策记录在 ADR-0020。

## 其他成员代码状态

- 队友目标计数代码已完成第一阶段任务化接入：新增不改变 `Detect` 行为的 `CountingDetect` Head、`yolov12m-counting.yaml`、类化 counting 包、`train_counting.py`、`test_counting.py` 和自包含训练/测试脚本。
- 计数仍采用原有“文本引导类无关检测 -> NMS -> 检测框数”逻辑，不引入密度图、计数回归 Head 或新 Loss；共享 YOLOv12m Backbone/Neck、OpenCLIP 和文本引导检测 Trainer。
- 计数评测当前明确保持队友的 positive-query VOC 协议，只查询 XML 中实际存在的类别，报告 EM、MAE、RMSE 与逐类别统计；不得把它表述为包含零计数问答的完整协议。
- 队友仓库未提供训练 checkpoint，本地也未完成 VRSBench 真实数据 smoke；当前验证范围为语法、参数展开、Head 等价性、模型构建和预训练权重静态加载。

## 下一步

1. 在 A5000 上完成 cleaned 协议的 YOLOv12m 正式训练与完整 test，并记录参数量、checkpoint 大小、显存、训练耗时和推理速度。
2. 使用旧 YOLOv12m raw-best checkpoint 在相同的 3480-sample cleaned test 上复评，再与新训练结果比较，分离数据清洗与重新训练的影响。
3. 获取计数队友的 VRSBench 数据路径和训练 checkpoint，运行最小训练 smoke 与 1-2 张图的独立计数评测，确认 checkpoint 严格加载和真实 EM/MAE/RMSE 输出。
4. 继续收集分类等其他任务的 YAML、Head、Loss、数据和 checkpoint；每个任务先建立独立脚本并通过 smoke，再实现可选薄分发器。
5. 所有任务稳定后，再设计统一推理入口和 A5000 同条件资源测评；当前不要提前合并为联合多数据集训练。

## 部署注意

- 分割训练包不需要携带 `run_semseg_preset.sh`；`scripts/train_refseg.sh` 已包含所需正式参数。
- 运行前仍需提供项目源码、RRSIS-D 数据与缓存、`pretrain_model/yolov12m.pt`，并激活具备 PyTorch、OpenCLIP、OpenCV 等依赖的环境。
- 计数训练还需提供 VOC 风格 VRSBench 数据；独立测试需提供计数 `best.pt`，默认路径均可通过脚本环境变量覆盖。
- 默认训练输出目录已有历史结果时，应通过 `SAVE_DIR` 指定新目录，避免覆盖旧实验。
