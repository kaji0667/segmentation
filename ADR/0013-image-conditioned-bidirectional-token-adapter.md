# ADR-0013: Image-Conditioned Bidirectional Token Adapter

## 状态

Rejected after controlled experiment

## 背景

当前主线 `TextPromptSegment` 在进入视觉交互前，先使用一个只依赖文本 token 本身的 scorer 将 OpenCLIP token 特征压缩成全局向量。后续语言信息只通过 P3/P4/P5 尺度权重、FiLM、pixel-text cosine similarity 和单路卷积 decoder 注入。

当前最好基线 `runs/semseg/noattn_aug_axis` 的 test 指标为：

- `oIoU=0.698654`
- `mIoU=0.530917`
- `Pr@0.5-0.9=0.600115/0.516806/0.407354/0.295030/0.142488`

RRSIS 近年方法（CADFormer、SBANet、CroBIM）普遍使用视觉到语言、语言到视觉的双向细粒度交互。当前实现的语言聚合与图像无关，难以根据具体图像区分目标实体、空间关系和同类参照物。

项目已经证明，对全部 P3 像素做空间 softmax 会导致目标像素竞争固定概率质量并降低召回。因此本实验不能恢复已拒绝的 64×64 query/key spatial-softmax 分支。

## 决策

只在 `TextPromptSegment` 内加入一个低分辨率、零初始化 residual 的双向 token adapter：

1. 先把已投影但尚未文本加权的 P3/P4/P5 特征统一到 P3 分辨率并取均值，构造视觉 seed。
2. 将视觉 seed 自适应池化为 `8×8` region tokens，并投影到 64 维 adapter 空间。
3. OpenCLIP text tokens 查询 region tokens，获得 image-conditioned text residual。
4. region tokens 再查询更新后的 text tokens，获得 language-conditioned visual residual。
5. 两个 residual 分别由独立的零初始化标量 gate 控制。初始化时 adapter 对输出严格无影响。
6. 更新后的 text tokens 继续走现有 learnable token pooling；visual residual 加到现有 FPN fuse 输出前。

不修改：

- YOLOv12 backbone 和 neck
- OpenCLIP encoder、维度和缓存格式
- P3/P4/P5 输入层
- FiLM、pixel-text similarity、spatial gate、value branch 和单路 mask decoder
- BCE-Tversky loss、采样、增强、seed、checkpoint 和评估协议

## 备选方案

- 在全部 P3 像素上恢复 cross-attention：拒绝，已有空间 softmax 回退证据，且计算量更大。
- 直接引入 P2：拒绝，已有直接 P2 融合失败证据；不属于本次文本交互实验。
- 同时加入 contrastive alignment loss：推迟，避免同时改变结构和训练目标。
- 同时拆分 object/spatial/reference phrase：推迟，现有启发式 BPE 角色掩码不可靠。

## 影响

- 预计增加约 15 万个训练参数，不改变公开 `[B,1,H,W]` 输出接口。
- 旧 segmentation-head checkpoint 不能严格加载新增参数；完整实验仍从相同 `yolov12n.pt` backbone/neck 初始化开始。
- 零 gate 使初始外部行为与当前主线一致，同时允许训练逐步启用双向交互。
- 需要定向测试、全部 semseg 回归测试、CUDA smoke 和 seed-42 完整对照。

## 验收

与 `noattn_aug_axis` 保持相同训练与测试协议，比较：

- 核心：oIoU、mIoU、Pr@0.5-0.9
- 诊断：Precision、Recall、F1、predicted-positive rate、target-positive rate
- 资源：参数量、checkpoint 大小、测试耗时、峰值显存

只有核心结果形成明确综合提升时才替换主线并推送 GitHub；若不如当前最优，则记录失败实验，不推送候选代码。

## 实验结果

seed-42 完整实验使用与 `noattn_aug_axis` 相同的数据划分、axis-aware 增强、loss、采样、阈值扫描、mIoU 选模与冻结阈值 test 协议。运行目录为 `HFSA-main/runs/semseg/bta_axis`，第 55 轮 early stop，`best_raw.pt` 来自第 47 轮，验证集冻结阈值为 `0.80`，完整 test 覆盖 3,481 个表达样本。

相对当前最好基线：

- oIoU：`0.698654 -> 0.694395`，变化 `-0.004259`
- mIoU：`0.530917 -> 0.539623`，变化 `+0.008707`
- Pr@0.5-0.9：`0.600115/0.516806/0.407354/0.295030/0.142488 -> 0.602126/0.521689/0.422867/0.304797/0.152255`
- Precision：`0.794239 -> 0.803445`
- Recall：`0.853056 -> 0.836496`
- F1：`0.822597 -> 0.819638`
- predicted-positive rate：`0.050106 -> 0.048571`，更接近 target-positive rate `0.046651`
- 参数量：`3,971,624 -> 4,119,978`
- checkpoint：`35.57 MB -> 37.29 MB`
- test 延迟：`23.17 -> 29.33 ms/sample`
- 峰值 GPU：`336.51 -> 370.35 MB`

raw-best checkpoint 中两个零初始化 gate 都已学习为非零：text gate 的 `tanh=-0.015298`，visual gate 的 `tanh=-0.064627`。因此该结果反映了实际启用的双向适配器，而不是初始化恒等路径。

## 结论

该结构提高了官方 mIoU 和全部 Pr@0.5-0.9，尤其改善了较高 IoU 阈值下的样本成功率，也使预测前景比例更接近目标比例；但它同时降低 oIoU、Recall 和 F1，并增加参数、checkpoint、延迟和显存。结果属于 Pareto 权衡，而不是对当前最好基线的明确综合超越，不满足本 ADR 的推广门槛。

活动源码已恢复到已发布的 no-attention、axis-aware、单解码主线，候选专用测试已从活动测试集移除。实验 checkpoint 和报告保留在本地 `runs/semseg/bta_axis`，候选代码不推送 GitHub。

## 关联

- `HFSA-main/ultralytics/nn/modules/head.py`
- `HFSA-main/tests/test_semseg_bidirectional_token_adapter.py`
- `ADR/0008-learnable-token-pooling.md`
- `ADR/0010-remove-spatial-softmax-attention.md`
- `research/RRSIS_LITERATURE_REVIEW_20260823.md`
