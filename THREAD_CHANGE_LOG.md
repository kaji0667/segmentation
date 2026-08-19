# HFSA 线程变更日志

来源：近期 Codex 线程摘要，以及本地 `runs/semseg/*/results.csv` 指标文件。

## 简短日志

### 1. LoveDA prompt 和 batch 实验
- 改动：加强了 LoveDA 各类别的 prompt，尤其是 `water`、`road`，以及其他更符合俯视遥感图像语境的描述。同步修改了数据集默认 prompt、`data.yaml` 和 `class_prompts.json`。
- 效果：本地 LoveDA 最好的一次 run 是 `loveda_text_openclip_512_b2_e30_pretrained_cosine`，`target_iou=0.5675`，`binary_miou=0.7343`。后续讨论过 prompt v2 和 batch 4 的实验目录 `loveda_text_openclip_512_b4_e30_prompt_v2`，但本地没有找到对应结果目录。
- 模型分析：LoveDA 这一路仍然更像“类别级二值 mask 学习”，还不是真正带实例、方位和关系理解的 referring segmentation。

### 2. 训练稳定性修改
- 改动：加入或规划了 early stopping、冻结 backbone/neck、可配置 `pos_weight` 上限、小目标加权采样等机制。主要目标是缓解 `val_loss` 后期上升，并避免把 YOLO 预训练特征带偏太多。
- 效果：`rrsisd_openclip_512_b4_e40_freeze_bb_pw10_small3` 达到 `target_iou=0.6591`，高于最早的 RRSIS-D baseline `0.6511`，但仍低于后续 non-P2 加增强的结果。
- 模型分析：`val_loss` 上升被判断为校准变差和过拟合压力，不是 mask 结果完全崩掉。因为 BCE+Dice loss 上升时，IoU 仍可能继续改善。

### 3. 切换到 RRSIS-D 数据集
- 改动：选择 RRSIS-D 作为更适合 `image + text -> mask` 的数据集，下载官方文件，并确认了 `refs(unc).p` 里的 train/val/test 划分结构。
- 效果：确认数据完整：`train=12181`，`val=1740`，`test=3481`。训练目标是每条文本表达对应一个二值 mask，而不是一次 forward 输出完整多类别 mask。
- 模型分析：这一步让项目方向更接近遥感指代表达分割，而不是 LoveDA 式类别语义分割。

### 4. 验证阈值扫描和 P2 实验
- 改动：加入 `--val-thresholds`，新增 `best_threshold`、precision、recall、F1 等验证指标；新增 `yolov12-semseg-p2.yaml`，让 `TextPromptSegment` 接入 P2/P3/P4/P5 多尺度特征。
- 效果：P2 实验 `rrsisd_openclip_512_b4_e40_p2_freeze_bb_pw10_small3` 得到 `target_iou=0.6502`，`binary_miou=0.8141`，`best_threshold=0.6`。它相比部分早期实验降低了 val loss，但没有提升 target IoU。
- 模型分析：P2 理论上有利于小目标，但本地实验没有证明它适合作为当前主线。

### 5. 文本感知增强和 non-P2 主线
- 改动：为 RRSIS-D 加入只在训练集启用的轻量增强。如果文本包含 `left`、`right`、`top`、`bottom`、`upper`、`lower`、`center` 等方向/位置词，就跳过水平或垂直翻转。默认模型保持 non-P2，并把默认 `small-target-boost` 降到 `2.0`。
- 效果：`rrsisd_openclip_512_b4_e28_nonp2_aug_boost2` 是当前本地确认的最好结果：`target_iou=0.6726`，`binary_miou=0.8260`，`val_loss=0.5975`，`best_threshold=0.6`。
- 模型分析：这是目前最强的已确认主线。

### 6. Boost 2.5 对比
- 改动：在 non-P2 加增强结果之后，对比了 `small-target-boost=2.5`。
- 效果：`rrsisd_openclip_512_b4_e28_nonp2_aug_boost25` 得到 `target_iou=0.6660`，`binary_miou=0.8223`，`val_loss=0.5997`，`best_threshold=0.6`。整体不如 boost 2。
- 模型分析：继续加大 boost 没有真正救起 `vehicle`。线程分析发现，当前 sampler 用的是 bbox 面积，而很多真实小 mask 在 bbox 面积上并不小。

### 7. 当前仍未解决的问题
- 改动状态：尚未完全解决。
- 效果：整体指标已经提升，但弱点仍然存在：极小 `vehicle`、复杂 `harbor`、`windmill`、`trainstation`、`tenniscourt`，以及 `dam` 过分割。
- 模型分析：当前 `TextPromptSegment` 主要使用全局文本向量，因此对 token 级空间关系较弱，例如 `"above the chimney at the bottom"`。在大改 head 之前，更高性价比的下一步是把小目标判断改成真实 mask 面积，并加入 per-sample 或 area-aware loss。

## 当前最佳确认结果

`HFSA-main/runs/semseg/rrsisd_openclip_512_b4_e28_nonp2_aug_boost2`

- `target_iou=0.6726`
- `binary_miou=0.8260`
- `val_loss=0.5975`
- `best_threshold=0.6`

## 推荐下一步

保留 `boost=2` 的 non-P2 结果作为 baseline，然后把小目标判断从 bbox 面积改为真实 RLE mask 面积。改完后再测试较温和的 boost，例如 `1.5` 或 `2.0`，重点比较弱类 IoU 和验证预览图。

### 8. RRSIS-D 官方评测口径标准化
- 新增明确的 `oiou` 与论文口径 `official_miou` 字段；旧 `target_iou`、`sample_miou` 继续保留兼容。
- 新增真正的每类别 sample-mIoU，并把原有类别累计结果明确命名为 class-oIoU。
- 新增按 oIoU 或 mIoU 选择验证 checkpoint 的能力。
- 新增训练完成后加载最佳 checkpoint、冻结验证阈值并评估 test split 的流程。
- test 报告同时记录 Pr@0.5 至 Pr@0.9、类别指标、参数量、checkpoint 大小、平均评测耗时和峰值 GPU 显存。

### 9. 固定权重 P3/P4 多尺度深监督候选
- 在现有 `TextPromptSegment` 上增加训练期文本条件 P3/P4 辅助 mask，固定损失权重分别为 `0.20/0.10`，不增加 P5 辅助监督。
- 辅助标签使用保留前景的最大池化下采样；主输出继续使用现有 BCE+Tversky、小目标加权和两个可学习 spatial-gate 权重。
- 验证和推理仍只返回最终 mask，不改变官方评测协议。
- 新增脚本预设 `ds`，默认实验目录为 `runs/semseg/ds_p3p4`；尚未运行完整训练，不能称为已验证提升。
- 完成 P3/P4 实验后新增 `ds_p3` 跟进预设：固定 `P3=0.20`、关闭 P4，并将该预设的 `min_delta` 降为 `0.0002`；默认目录为 `runs/semseg/ds_p3`。原 `ds` 预设保留不变。

### 10. 深监督实验结束并回退主线
- P3/P4测试为 `oIoU=0.685367`、`mIoU=0.508498`，未超过无深监督基线。
- P3-only测试进一步下降到 `oIoU=0.671189`、`mIoU=0.493307`，说明问题不只是P4粗尺度监督。
- 已从活动源码移除P3/P4辅助头、辅助损失参数及`ds`/`ds_p3`预设，恢复两个可学习spatial-gate权重的无深监督基线。
- 两个实验目录继续保留，作为失败尝试和复现实证。

### 11. 按官方 mIoU 保存 raw-best checkpoint
- 标准 `baseline` 预设改为按论文口径的逐样本 mIoU 选择验证阈值和 checkpoint。
- 新增 `best_raw.pt`：selection score 只要严格创新高就保存，不受 `min_delta` 影响。
- `best.pt` 继续保留原有 `min_delta` 与 early stopping 语义。
- 训练后 test 优先评估 `best_raw.pt`，旧实验缺失时回退 `best.pt`，并冻结该 checkpoint 保存的验证阈值。
