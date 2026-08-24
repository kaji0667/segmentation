# ADR-0014: Text-Persistent Progressive Mask Decoder

## 状态

Rejected after controlled experiment

## 背景

当前 `TextPromptSegment` 将 P3/P4/P5 全部对齐到 P3 后一次性融合，文本通过尺度权重、FiLM、像素文本相似度和空间门控参与一次解码。近期 RRSIS 方法中的 CADFormer TCMD、LSCF CLA 和 SRGFormer PMR 均表明，文本语义应在从粗到细的 mask 解码过程中持续参与，而不是在解码前一次性注入。

当前最好基线 `runs/semseg/noattn_aug_axis` 的 test 指标为：

- `oIoU=0.698654`
- `mIoU=0.530917`
- `Pr@0.5-0.9=0.600115/0.516806/0.407354/0.295030/0.142488`

项目已拒绝 P3/P4 辅助深监督，因此本实验不能给中间尺度增加独立 loss。项目也已拒绝直接 P2 融合、全局空间 softmax 和简单双流 decoder。

## 决策

只在 `TextPromptSegment` 内将一次性 P3 对齐融合改为共享参数的 P5→P4→P3 文本贯穿渐进解码：

1. P3/P4/P5 继续使用现有投影和文本尺度权重。
2. 从 P5 开始，通过 top-down 上采样和相邻尺度拼接依次生成 P4、P3 解码特征。
3. 每一级都使用同一个文本 FiLM、像素文本相似度、视觉空间门控、value 分支和 mask decoder，避免为三个尺度复制完整解码参数。
4. P5 和 P4 mask 仅作为内部 residual，经两个零初始化可学习 gate 对齐到 P3 后加到 P3 logits。
5. 只对最终相加后的单一 `[B,1,H,W]` 输出计算现有 BCE-Tversky loss，不增加辅助监督。

不修改 backbone、neck、OpenCLIP、数据、增强、loss、checkpoint 选择和评估协议。

## 备选方案

- 为 P3/P4/P5 分别增加辅助 loss：拒绝，ADR-0005 已有明确回退证据。
- 为每一级复制独立 decoder：拒绝，参数和星载部署成本过高，且增加额外变量。
- 同时加入语义角色 token pooling：推迟到本实验结束后，保持单变量归因。
- 同时加入 P2：推迟到前两项实验结束后，并且只能通过不确定边界门控使用。

## 影响

- 公开输出接口、训练 loss 和评估协议保持不变。
- top-down 融合会增加少量 head 参数和计算量，但 mask decoder、文本投影与空间门控在三个尺度间共享。
- 旧 segmentation-head checkpoint 不能严格加载；完整实验继续从相同 `yolov12n.pt` backbone/neck 初始化。
- 需要定向测试、全部 semseg 回归测试、CUDA smoke 和 seed-42 完整对照。

## 验收

与 `noattn_aug_axis` 保持相同 split、seed、batch、imgsz、增强、loss、mIoU 选模和冻结阈值 test 协议。比较核心 oIoU、mIoU、Pr@0.5-0.9，并记录 Precision、Recall、F1、正像素比例、参数、checkpoint、延迟和峰值显存。

只有形成明确综合最优时才提升为活动主线并推送 GitHub；否则记录结果后恢复已发布基线。

## 关联

- `HFSA-main/ultralytics/nn/modules/head.py`
- `HFSA-main/tests/test_semseg_text_persistent_decoder.py`
- `ADR/0005-fixed-p3-p4-deep-supervision.md`
- `ADR/0010-remove-spatial-softmax-attention.md`
- `ADR/0013-image-conditioned-bidirectional-token-adapter.md`

## 实验结果

seed-42 完整实验使用与 `noattn_aug_axis` 相同的数据划分、batch 4、512 输入、axis-aware 增强、loss、mIoU 选模与冻结阈值 test 协议。运行目录为 `HFSA-main/runs/semseg/tpd_axis`，第 37 轮 early stop，`best_raw.pt` 来自第 29 轮，验证集冻结阈值为 `0.60`，完整 test 覆盖 3,481 个表达样本。

相对活动基线：

- oIoU：`0.698654 -> 0.685062`，变化 `-0.013592`
- mIoU：`0.530917 -> 0.511833`，变化 `-0.019084`
- Pr@0.5-0.9：`0.600115/0.516806/0.407354/0.295030/0.142488 -> 0.557886/0.482045/0.377190/0.275783/0.130997`
- Precision：`0.794239 -> 0.794161`
- Recall：`0.853056 -> 0.832965`
- F1：`0.822597 -> 0.813100`
- predicted-positive rate：`0.050106 -> 0.048931`，更接近 target-positive rate `0.046651`
- 参数量：`3,971,624 -> 4,267,050`，增加 295,426
- checkpoint：`35.57 MB -> 38.96 MB`
- test 延迟：`23.17 -> 28.64 ms/sample`
- 峰值 GPU：`336.51 -> 338.38 MB`

raw-best checkpoint 中两个粗尺度 residual gate 均学习为明显非零：原始值为 `0.338043/0.471728`，经 tanh 后为 `0.325729/0.439594`。因此回退结果不是粗尺度 residual 没有启用造成的。

## 结论

共享参数的文本贯穿 decoder 能够稳定学习，并使预测前景比例更接近目标比例；但它同时显著降低 oIoU、mIoU、全部 Pr 指标、Recall 和 F1，并增加参数、checkpoint 和推理延迟。该结构不满足综合最优门槛，不替换活动主线，也不推送 GitHub。活动源码恢复到 no-attention、axis-aware、单 decoder 基线后，再单独评估学习型目标/关系/位置 token pooling。
