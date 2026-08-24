# ADR-0015: Learned Target, Relation, and Position Token Pooling

## 状态

Accepted as the active mainline and publication candidate.

## 背景

当前活动 `TextPromptSegment` 只学习一个全局 token 权重分布，然后将 OpenCLIP tokens 压缩为单一文本向量。该向量同时负责尺度选择、FiLM 和像素文本相似度，目标类别、实例关系和空间位置缺少独立的梯度路径。

SRGFormer 的 Semantic Role Decomposition 使用三个独立学习分支，在没有语义角色标注的情况下从 token 特征中聚合 target、relation、position 表征。项目此前拒绝的是启发式词/BPE 角色 mask，不是否定端到端学习的角色分解。

## 决策

只在 `TextPromptSegment` 内加入学习型三角色 pooling 和轻量角色路由：

1. 保留现有共享 `text_token_score` 与 `valid_token_bias`。
2. 新增零初始化的 `role_token_score: Linear(text_dim, 3)` 和三个 role valid-token bias。
3. 共享分数与角色 residual 分数相加，分别 softmax 聚合 target、relation、position 三个向量。
4. target 向量负责现有 FiLM；target 与 relation 的均值负责主要 pixel-text similarity；relation 向量负责 P3/P4/P5 scale gate。
5. position 向量经独立投影生成逐像素余弦相似度，只以一个零初始化标量进入现有 spatial gate，不增加 spatial softmax 或 decoder 输入通道。
6. class embedding 或二维文本向量输入时，将同一个向量复制为三个角色，保持兼容。

初始化时角色 residual scorer、role bias 和 position gate 均为零，因此三个角色均等于现有 token pooling，外部 forward 与活动基线一致。不同角色进入不同下游路径，训练后可在仅有 mask 监督的情况下产生不同梯度。

不修改 backbone、neck、OpenCLIP、缓存格式、P3/P4/P5 融合、单 mask decoder、loss、数据、增强、checkpoint 或评估协议。

## 备选方案

- 使用人工目标词/位置词 mask：拒绝，已有 BPE 对齐与同类参照物歧义证据。
- 为三个角色增加独立 OpenCLIP 编码缓存：推迟，属于短语解析与缓存格式变更，不适合与本实验合并。
- 同时加入 SRGFormer graph transformer 或 progressive decoder：拒绝，实验一 decoder 已单独回退，且会破坏单变量归因。
- 使用空间 softmax 定位 position：拒绝，ADR-0009 已证明像素竞争会降低召回。

## 影响

- 增加一个 `Linear(768,3)`、三个 bias、一个 position projection 和一个标量 gate，公开输出仍为 `[B,1,H,W]`。
- 推理继续使用原缓存 token，不增加 OpenCLIP 调用。
- 旧 segmentation-head checkpoint 不能严格加载；完整实验从相同 `yolov12n.pt` backbone/neck 初始化。
- 需要验证初始化等价性、三角色可分化、梯度、全部回归测试、CUDA smoke 和 seed-42 完整对照。

## 验收

与 `noattn_aug_axis` 固定相同 split、seed、batch、imgsz、增强、loss、mIoU 选模和冻结阈值 test。核心比较 oIoU、mIoU、Pr@0.5-0.9，并记录角色 scorer 差异、position gate、Precision/Recall/F1、正像素比例和资源成本。

只有形成明确综合最优时才提升为活动主线并推送 GitHub；否则记录结果后恢复发布基线。

## 结果

完整 seed-42 实验位于 `HFSA-main/runs/semseg/srp_axis`。训练在第 52 轮因连续 8 轮无显著 mIoU 提升而早停，`best_raw.pt` 选中第 44 轮，冻结 validation 阈值为 `0.80`，test 覆盖全部 3,481 个表达样本。

- test `oIoU=0.700990`
- test `mIoU=0.539022`
- `Pr@0.5-0.9=0.598391/0.516231/0.413100/0.305085/0.151106`
- Precision/Recall/F1 `0.799135/0.850918/0.824214`
- predicted-positive rate `0.049674`，目标正像素率 `0.046651`
- 参数量 `4,072,364`，checkpoint `36.72 MB`
- test 平均延迟 `28.95 ms/sample`，峰值 GPU `339.31 MB`

相对 `noattn_aug_axis`，oIoU `+0.002336`、mIoU `+0.008105`、Pr@0.6-0.9、Precision 和 F1 均提升；Pr@0.5 `-0.001724`、Recall `-0.002138`。参数增加 `100,740`，延迟增加 `5.78 ms/sample`。

角色分支确实分化：三个 `role_token_score` 行范数为 `4.9714/5.5953/2.3424`，两两差异范数为 `2.1371/4.5811/4.6852`；三个 role valid-token bias 为 `0.1712/0.2696/0.0462`。position gate 的 raw/tanh 为 `-0.4553/-0.4263`，说明位置路径实际参与推理。

该候选在两个主 IoU 指标上同时超过既有基线。后续 uncertainty-gated P2 residual 完整实验未超过本结果，因此该学习型语义角色 pooling 版本恢复为活动主线并进入发布提交。

## 关联

- `HFSA-main/ultralytics/nn/modules/head.py`
- `HFSA-main/tests/test_semseg_semantic_role_pooling.py`
- `ADR/0008-learnable-token-pooling.md`
- `ADR/0010-remove-spatial-softmax-attention.md`
- `ADR/0014-text-persistent-progressive-decoder.md`
