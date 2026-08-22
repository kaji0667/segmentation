# ADR-0012: Target-Background Twin-Stream Decoder

## 状态

Rejected as mainline after controlled experiment

## 背景

当前 RRSIS-D 主线使用固定 YOLOv12 backbone/neck、OpenCLIP 文本特征和 `TextPromptSegment`。最好基线 `noattn_aug_axis` 的 test oIoU 为 `0.698654`、mIoU 为 `0.530917`，但预测仍存在目标边界外溢和相似背景误激活。允许改动范围仅包含特征融合及后续二值掩膜解码，不能改 backbone、neck 或 OpenCLIP。

## 决策

只把 `TextPromptSegment` 的单路 `mask_decoder` 替换为结构对称、参数独立的 `target_decoder` 与 `background_decoder`。两路接收完全相同的现有解码输入：gated visual、gated value 和 pixel-text similarity。最终 logits 定义为：

```text
target_logits - background_logits + similarity + bias
```

模型仍只返回一个 `[B,1,H,W]` tensor，并继续使用现有 BCE-Tversky loss。目标流与背景流不增加单独标签或辅助损失，以保证本次实验只改变最终 decoder 的对比式参数化。

完整实验从相同 `yolov12n.pt` 初始化开始，固定 seed 42、RRSIS-D split、512 输入、batch 4、axis-aware augmentation、mIoU 选模、`best_raw.pt` 和验证阈值冻结后的 test 协议。核心对比指标为 oIoU、mIoU 和 Pr@0.5-0.9。

## 备选方案

- 在同一 decoder 内加入背景通道并做二类 softmax：会改变输出和损失接口，增加本次消融变量。
- 为背景流增加互补 mask 或边界辅助损失：会同时改变 decoder 和训练目标，无法归因。
- 修改 backbone、neck 或 OpenCLIP：超出已确认边界。

## 影响

- 正面：显式学习前景证据与背景证据的差值，且不改变训练和评测接口。
- 负面：最终 decoder 参数和计算量接近翻倍；两个无独立监督的分支可能发生冗余或抵消。
- 兼容性：旧 checkpoint 无法严格加载到新双流 head；新实验必须从共同 backbone 权重初始化，不能续训旧 segmentation head。
- 验证：需要语法检查、全部 semseg 回归测试、CUDA smoke，以及与 `noattn_aug_axis` 的完整单变量训练对照。

## 关联

- 代码：`HFSA-main/ultralytics/nn/modules/head.py`
- 测试：`HFSA-main/tests/test_semseg_target_background_decoder.py`
- 基线：`HFSA-main/runs/semseg/noattn_aug_axis`
- 候选实验：`HFSA-main/runs/semseg/tbtd`

## 实验结果

正式 seed-42 实验在 epoch 57 早停，`best_raw.pt` 来自 epoch 49，冻结 validation 阈值为 `0.70`。完整 test 3,481 样本结果：

- `oIoU=0.693151`
- `mIoU=0.532644`
- `Pr@0.5-0.9=0.597817/0.517380/0.410227/0.299052/0.145361`
- `Precision/Recall/F1=0.802104/0.836144/0.818771`

相对 `noattn_aug_axis`，mIoU 增加 `0.001727`，Pr@0.6-0.9 分别增加 `0.000575/0.002873/0.004022/0.002873`，Precision 增加 `0.007866`，预测正像素率从 `0.050106` 降至 `0.048631`，更接近目标正像素率 `0.046651`。但 oIoU 下降 `0.005503`，Pr@0.5 下降 `0.002298`，Recall 下降 `0.016912`，F1 下降 `0.003827`。

因此该结构显示出轻微的样本级/高阈值精度收益和外溢收敛，但牺牲整体累计交并比与召回，且 mIoU 增益很小。它不替代当前 no-attention axis-aware 主线；若后续重新研究，应作为需要独立监督或更强互补约束的候选，而不是直接重复本实验。
