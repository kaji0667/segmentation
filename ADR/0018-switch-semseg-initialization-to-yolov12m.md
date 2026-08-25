# ADR-0018: Switch Semantic Segmentation Initialization to YOLOv12m

## 状态

Accepted as an accuracy-oriented candidate; not a resource-efficient comprehensive replacement for the YOLOv12n ADR-0015 baseline.

## 背景

当前 ADR-0015 学习型 target/relation/position token pooling 主线一直使用 `yolov12n.pt` 初始化，并通过未带 scale 的 `yolov12-semseg.yaml` 默认构建 n-scale backbone/neck。用户明确要求将预训练权重改为 `yolov12m.pt`，并按已经确定的 seed-42 baseline 策略完整运行一次。

只把权重路径改成 `yolov12m.pt` 而继续构建 n-scale 模型会产生大量通道尺寸不匹配，不能视为有效的 m 权重迁移。模型 scale 与预训练权重必须成对切换。

## 决策

1. `train_semseg.py` 默认模型改为 `ultralytics/cfg/models/v12/yolov12m-semseg.yaml`，默认权重改为 `pretrain_model/yolov12m.pt`。
2. `run_semseg_preset.sh` 使用同一组 m-scale 模型与权重默认值。
3. 依赖项目内 Ultralytics 的标准 scale alias：`yolov12m-semseg.yaml` 会读取统一基础配置 `yolov12-semseg.yaml`，同时从文件名解析 `scale=m`；不复制或分叉模型 YAML。
4. 保留 ADR-0015 的 `TextPromptSegment`、P3/P4/P5 接线、OpenCLIP、loss、axis-aware 增强、采样、checkpoint 和评估协议不变。
5. 完整实验保持 seed 42、batch 4、imgsz 512、epochs 60、patience 8、官方 mIoU 选模、`best_raw.pt` 和 validation 阈值冻结后的完整 test。

## 备选方案

- 只替换权重文件、不改变模型 scale：拒绝，因为 m 权重与 n-scale 通道不匹配。
- 复制一份完整 `yolov12m-semseg.yaml`：拒绝，因为会重复维护同一结构；现有 Ultralytics 已提供标准 scale alias。
- 从 ADR-0015 checkpoint 继续微调：拒绝，因为该 checkpoint 是 n-scale，不能严格加载到 m-scale；本实验从官方 m 预训练 backbone/neck 初始化。

## 影响

- 静态审计确认 `scale=m`，模型总参数为 `20,273,356`。
- `yolov12m.pt` 成功匹配 `678/762` 个目标 state tensors，仅跳过自定义语义分割头 `model.21.*`。
- 冻结 backbone 后 trainable/frozen 参数分别为 `9,493,644/10,794,304`，训练与推理资源成本将显著高于 n-scale ADR-0015 基线。
- `pretrain_model/yolov12m.pt` 属于本地大体积权重，不纳入 Git。
- 需要先通过 batch-4、2-train/2-val/2-test CUDA smoke，再启动完整受控实验并记录资源与测试指标。

## 关联

- `HFSA-main/train_semseg.py`
- `HFSA-main/run_semseg_preset.sh`
- `HFSA-main/tests/test_semseg_yolov12m_defaults.py`
- `ADR/0015-learned-semantic-role-token-pooling.md`

## 实验结果与评估

完整运行位于 `HFSA-main/runs/semseg/srp_yolov12m_axis`。模型在第 52 轮按 `patience=8` 早停，`best_raw.pt` 与 `best.pt` 均来自第 44 轮，validation 冻结阈值为 `0.70`，test 覆盖全部 3,481 条表达。

- test `oIoU=0.701171`
- test `mIoU=0.552562`
- `Pr@0.5-0.9=0.623959/0.540362/0.425452/0.319161/0.164321`
- Precision/Recall/F1 `0.804625/0.845044/0.824339`
- predicted/target positive rate `0.048995/0.046651`
- 实际模型参数 `20,287,948`，trainable 参数 `9,493,644`
- checkpoint `150.33 MB`，test 峰值 GPU `726.26 MB`
- 52 轮平均 `361.34 s/epoch`，累计 epoch 时间约 `5.22 h`

相对相同 seed、epoch、batch、imgsz、head、loss 和评估协议的 YOLOv12n `srp_axis`：

- oIoU `+0.000181`，基本持平。
- mIoU `+0.013540`，Pr@0.5-0.9 全部提升 `+0.012353` 至 `+0.025567`。
- Precision `+0.005490`，Recall `-0.005875`，F1 `+0.000125`。
- predicted-positive rate 更接近目标比例，外溢差值从 `0.003023` 降至 `0.002343`。
- 总参数约为 n-scale 的 `4.98x`，trainable 参数 `3.47x`，checkpoint `4.09x`，峰值 GPU 显存 `2.14x`。
- 单次 test 报告显示 m-scale 更快，但该结果与模型规模不一致，受同机状态影响，不能作为加速结论；需要同进程配对 benchmark 才能用于部署报告。

## 最终决策

YOLOv12m 对逐样本 mask 质量和全部 Pr 指标有稳定收益，适合作为准确率优先候选；但 oIoU 与 F1 几乎不变，且模型大小、参数量和显存代价显著。考虑比赛资源效率与星载适配占较高权重，当前证据不足以把它称为综合最优或无条件替代 YOLOv12n。代码默认保持用户要求的 m-scale，是否作为最终提交主线需在准确率优先与部署效率优先之间明确选择。
