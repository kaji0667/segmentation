# ADR-0016: Uncertainty-Gated P2 Boundary Residual

## 状态

Rejected after full experiment; ADR-0015 restored as active mainline.

## 背景

旧 P2 实验把 P2 与 P3/P4/P5 一起进行全图融合，test 没有超过 non-P2 主线。实验二的学习型语义角色 pooling 已同时提高 oIoU 与 mIoU，但 P2 仍可能只在粗掩膜边缘难以确定的像素上提供局部高分辨率细节。

用户明确要求第三项实验只在不确定边界区域启用 P2 residual，不能再次做全图 P2 融合。

## 决策

以 ADR-0015 的完整 P3/P4/P5 head 为 coarse 基座，新增独立 P2 residual：

1. P2 不进入 scale gate、P3/P4/P5 fuse、coarse visual、coarse similarity 或 coarse decoder。
2. coarse logits 按原路径生成，保证基础结构和监督协议不变。
3. 从 detached coarse probability `p` 计算 uncertainty `4*p*(1-p)`。
4. 用 3×3 max/min pooling 计算 probability morphological boundary strength。
5. 仅当 `uncertainty >= 0.5` 且 `boundary_strength >= 0.05` 时启用 P2 residual；最终分辨率 mask 外 residual 严格为零。
6. P2 residual 使用 P2 visual、上采样 coarse visual、coarse probability 和 P2 pixel-text similarity预测一个 logit correction。
7. residual decoder 最后一层权重和 bias 零初始化，因此初始输出严格等于 ADR-0015 coarse 输出。
8. 仍只对最终单一 mask 使用现有 BCE-Tversky loss，不增加边界或辅助 loss。

不修改 backbone、neck、OpenCLIP、数据、增强、采样、训练超参、checkpoint 选择或 test 协议。

## 备选方案

- P2 全图融合：拒绝，旧实验已失败且不符合用户限定。
- 对全图使用软 uncertainty 权重：拒绝，无法保证 residual 只发生在边界区域。
- 增加边界监督或 auxiliary loss：拒绝，会同时改变结构和 loss，破坏单变量归因。
- 直接对二值 coarse mask 做固定形态学修补：拒绝，不利用 P2 视觉与文本信息。

## 影响

- 新增 P2 projection、轻量文本条件和 residual decoder；coarse head 参数与行为保持不变。
- 新模型配置接入 P2/P3/P4/P5，但 P2 只供 residual 分支使用。
- 需要测试初始化等价、mask 区域约束、mask 外严格零残差、P2 分支梯度、完整回归、CUDA smoke 和 seed-42 全量实验。

## 验收

与 `srp_axis` 固定相同 split、seed、batch、imgsz、axis-aware augmentation、loss、mIoU 选模和冻结阈值 test。比较 oIoU、mIoU、Pr@0.5-0.9、Precision/Recall/F1、正像素比例、参数、延迟、显存、uncertainty mask 覆盖率和 residual 参数是否实际学习。

只有形成新的综合最优时才保留并发布；否则记录结果后恢复 ADR-0015 候选。

## 结果

完整 seed-42 实验位于 `HFSA-main/runs/semseg/p2ubr_axis`。训练在第 52 轮因连续 8 轮 mIoU 未提升而早停，`best_raw.pt` 选中第 44 轮，冻结 validation 阈值为 `0.80`，test 覆盖全部 3,481 个表达样本。

- test `oIoU=0.694145`
- test `mIoU=0.535108`
- `Pr@0.5-0.9=0.593795/0.510773/0.404194/0.300201/0.151681`
- Precision/Recall/F1 `0.790502/0.850628/0.819463`
- predicted-positive rate `0.050200`，目标正像素率 `0.046651`
- 参数量 `4,821,421`，checkpoint `45.32 MB`
- test 平均评测耗时 `86.04 ms/sample`，峰值 GPU `558.66 MB`

相对 ADR-0015 的 `srp_axis`，oIoU `-0.006845`、mIoU `-0.003914`、Pr@0.5-0.8 全部回退；仅 Pr@0.9 小幅 `+0.000575`。Precision、Recall、F1 分别 `-0.008633/-0.000290/-0.004751`。参数增加 `749,057`，平均评测耗时增加约 `57.09 ms/sample`，峰值 GPU 增加约 `219.35 MB`。

P2 分支实际参与训练：residual decoder 最后一层权重范数为 `1.011367`，bias 绝对值为 `0.100263`，P2 FiLM、视觉投影和文本投影均为非零。完整 test 上 uncertainty-boundary mask 平均覆盖 `1.9474%` 像素，每图中位数 `0.6104%`、90 分位 `5.2979%`，仅 `187/3481` 个样本完全未触发。因此负结果不是分支未学习或约束未生效，而是局部高分辨率修正没有改善整体泛化。

## 最终决定

拒绝提升 P2 residual。活动代码删除专用 P2 配置、测试和 residual 分支，恢复 ADR-0015 的 P3/P4/P5 学习型 target/relation/position pooling 作为综合最优并发布。完整运行产物保留在本地，不提交 checkpoint、缓存或 `runs/`。

## 关联

- `ADR/0015-learned-semantic-role-token-pooling.md`
- `HFSA-main/ultralytics/nn/modules/head.py`
- `HFSA-main/ultralytics/cfg/models/v12/yolov12-semseg-p2-residual.yaml`
- `HFSA-main/tests/test_semseg_p2_uncertainty_residual.py`
