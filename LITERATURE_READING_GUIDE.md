# HFSA Literature Reading Guide

本文档记录 HFSA 文本引导遥感语义分割 / 指代表达分割方向的文献阅读路线。当前重点服务于后续技术报告和实验方案设计，不作为代码实现记录。

## 针对目标外溢 / over-segmentation / boundary leakage 的推荐阅读文献

### 背景

HFSA 当前文本引导遥感分割主线已经在 RRSIS-D 上取得较好整体指标，但验证图仍存在明显目标外溢：预测 mask 把目标外部、视觉相似或相邻区域也吃进去。

之前较强的 Tversky / FP penalty 虽能提高 precision，但会导致 recall 降低和细长目标残缺。因此后续不应继续单纯压低 FP，而应优先查边界约束和文本-像素对齐方向。

当前判断：HFSA 的外溢主要不是 epoch 不够，也不是单纯阈值问题，而是区域型 loss、全局文本条件和遥感相似背景共同导致的边界泄漏。后续文献阅读和实验设计应优先围绕 boundary-aware loss 和 text-to-pixel alignment 展开。

### 推荐阅读顺序

1. **Boundary Loss for Highly Unbalanced Segmentation**
   - 链接：https://arxiv.org/abs/1812.07032
   - 用途：最直接针对区域型 Dice / CE loss 对边界约束弱的问题。
   - 重点阅读：boundary loss 如何与区域 loss 互补。
   - 可用于 HFSA：作为后续“旧 best 配置 + 轻量 boundary loss”的理论依据。

2. **Boundary Loss for Remote Sensing Imagery Semantic Segmentation**
   - 链接：https://arxiv.org/abs/1905.07852
   - 用途：遥感场景里的 boundary loss，更贴近当前遥感影像背景。
   - 重点阅读：它如何在遥感语义分割中约束边界。
   - 可用于 HFSA：评估是否适合迁移到二值 text-guided mask。

3. **Active Boundary Loss for Semantic Segmentation**
   - 链接：https://arxiv.org/abs/2102.02696
   - 用途：重点看预测边界向 GT 边界对齐的思想。
   - 重点阅读：主动边界对齐机制如何处理局部大块外溢。
   - 可用于 HFSA：作为边界修正方向，而不是简单压小整个 mask。

4. **RRSIS: Referring Remote Sensing Image Segmentation**
   - 链接：https://arxiv.org/abs/2306.08625
   - 用途：最贴近 HFSA 当前任务。
   - 重点阅读：遥感 referring segmentation 的难点、小目标、多尺度、语言引导跨尺度融合，以及 Pr@X / oIoU / mIoU 指标。
   - 可用于 HFSA：支撑 RRSIS-D 实验报告中的任务定义、难点分析和指标设计。

5. **CRIS: CLIP-Driven Referring Image Segmentation**
   - 链接：https://arxiv.org/abs/2111.15174
   - 用途：当前项目使用 OpenCLIP 文本向量，但像素级文本对齐弱。
   - 重点阅读：text-to-pixel alignment 和 CLIP 特征如何进入分割解码。
   - 可用于 HFSA：为后续引入像素-文本对齐辅助 loss 或轻量解码器提供依据。

6. **LAVT: Language-Aware Vision Transformer for Referring Image Segmentation**
   - 链接：https://arxiv.org/abs/2112.02244
   - 用途：解释为什么只用全局 text embedding 容易找错一大片。
   - 重点阅读：语言特征如何进入视觉编码和多层特征。
   - 可用于 HFSA：为轻量 language-aware feature fusion / gate 设计提供参考。

7. **ReSTR: Convolution-free Referring Image Segmentation Using Transformers**
   - 链接：https://arxiv.org/abs/2203.16768
   - 用途：辅助了解 Transformer 式跨模态交互。
   - 重点阅读：跨模态 token 交互和 decoder 结构。
   - 可用于 HFSA：作为报告中更复杂方案参考，不建议完整照搬。

8. **VLT: Vision-Language Transformer and Query Generation for Referring Segmentation**
   - 链接：https://arxiv.org/abs/2210.15871
   - 用途：重点看动态 query 和文本理解分支。
   - 重点阅读：query generation、语言理解分支和视觉-语言交互。
   - 可用于 HFSA：对复杂指代表达、相邻相似目标和空间关系建模有启发。

9. **Tversky Loss**
   - 链接：https://arxiv.org/abs/1706.05721
   - 用途：作为 precision / recall 权衡的理论依据。
   - 重点阅读：Tversky index 中 FP / FN 权重如何影响结果。
   - 可用于 HFSA：只能作为辅助 loss 参考。HFSA 已经试过较强 FP 权重，会产生欠分割，因此不建议作为主解法。

10. **Focal Tversky Loss**
    - 链接：https://arxiv.org/abs/1810.07842
    - 用途：适合类别不平衡和小目标。
    - 重点阅读：focal 参数如何强调难样本。
    - 可用于 HFSA：作为 loss 设计参考，但要警惕压过头导致 recall 下降和细长目标残缺。

### 当前检索关键词

- boundary-aware loss
- boundary loss semantic segmentation
- remote sensing boundary refinement
- text-to-pixel alignment
- CLIP referring image segmentation
- language-guided referring segmentation

### 后续实验路线提示

- 优先尝试轻量 boundary-aware loss，而不是继续提高 FP penalty。
- 保留当前 best-line 配置作为对照，只增加一个边界项或一个像素-文本对齐项。
- 所有 over-segmentation 实验都应同时记录 target IoU、precision、recall、F1、pred_pos_rate、weak-class IoU，以及典型验证图。
- 对细长目标和小目标需要单独观察，避免边界约束或高 precision 方案把目标压断。
