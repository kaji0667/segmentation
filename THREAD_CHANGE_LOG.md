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

### 12. 官方 mIoU 选模完整实验结果
- 完整实验 `runs/semseg/base_miou` 在第 36 轮早停，按验证 mIoU 选中第 28 轮，阈值为 `0.70`。
- test：`oIoU=0.672197`、`mIoU=0.509192`、`class_macro_mIoU=0.536258`、`F1=0.803969`。
- 相比原 learnable-gate 正式基线，mIoU 仅增加 `0.000176`，但 oIoU 下降 `0.019627`、F1 下降 `0.013875`、Pr@0.9 下降 `0.020396`。
- harbor 明显改善，类别宏平均略升；同时 Precision 下降、预测正像素率上升，说明外溢加重。
- 结论：保留 raw-best 保存机制，但该实验不替代原综合最好 checkpoint。

### 13. 可学习文本 token 池化候选
- 旧分割头对 `[B,77,768]` OpenCLIP token 固定平均，而验证表达平均只有约 6.69 个有效 token。
- 新增零初始化 `Linear(768,1)` token scorer 和 `valid_token_bias`，初始化时复现旧 mean pooling，仅增加 769 参数。
- 不使用存在 BPE 对齐和同类参照物歧义的启发式角色掩码。
- 单元测试、代理 A 复审和 GPU smoke 已通过；完整实验目录为 `runs/semseg/tpool`。

### 14. 可学习 token 池化完整结果
- `tpool` test 达到 `oIoU=0.683420`、`mIoU=0.519818`、`class_macro_mIoU=0.545010`。
- 相比相同 mIoU 选模协议的 `base_miou`，oIoU、mIoU、类别宏平均、Precision、Recall、F1 和 Pr@0.5-0.9 均提升，预测正像素比例略降。
- token pooling 被保留为活动文本聚合路径，但相对旧 oIoU 选模 checkpoint 仍有 Precision/高 IoU 成功率权衡。

### 15. 空间注意力热图标定候选
- 发现旧 attention 在 key/query 已归一化后又除以 `sqrt(128)`，并在 4096 个位置做 softmax，热图接近均匀且量级只有 `1/HW`。
- 新增一个可学习温度，使用 `softmax(temperature * cosine) * HW - 1` 构造相对密度，再经 `tanh` 限制到 `[-1,1]`。
- 均匀注意力现在严格对应 0；偏好位置为正、抑制位置为负，热图尺度不再随特征分辨率衰减。
- 仅增加 1 个参数；定向测试 4 项、原 token pooling 回归测试 3 项和 2-train/2-val/2-test GPU smoke 全部通过。
- 完整实验目录计划为 `runs/semseg/attnmap`，尚未完成全量训练，不能称为指标提升。

### 16. 注意力热图失败与 no-attention 消融
- `attnmap` 全量 test 为 `oIoU=0.678791`、`mIoU=0.514869`、类别宏平均 `0.538315`，低于 `tpool` 的 `0.683420/0.519818/0.545010`。
- 新热图提高 Precision、降低预测正像素率，但 Recall 和 Pr@0.9 明显下降，说明空间 softmax 的像素竞争使掩膜更保守、不完整。
- 活动候选完整移除 query/key attention、attention gate 权重、temperature 和 decoder attention 通道，保留 token pooling、similarity、visual gate 和 value 分支。
- 同时移除没有被活动 head 消费的 object/spatial/context token mask 生成、加载和转发；旧缓存中的额外字段会被忽略，无需重建缓存。
- 11 项测试与 2-train/2-val/2-test GPU smoke 已通过；全量目录为 `runs/semseg/noattn`，结果待运行。

### 17. 14 个尺寸异常样本与轴感知翻转消融
- 首次 `noattn` 进程只运行到第 11 轮且没有生成 test 报告，不能作为完整实验结果。
- 全库核对发现 17 张非 800×800 原图，其中 3 张 RLE 尺寸与原图一致；其余 14 张为 JPEG 实际高度 784–813、RLE 仍为 800×800，分布为 train 9、val 2、test 3。
- 数据加载器现在先用最近邻把解码 mask 对齐到 JPEG 实际尺寸，再执行统一训练缩放；不删除官方样本，不改变二值 mask 语义。
- 翻转策略拆为水平轴与垂直轴分别控制：水平词只禁水平翻转，垂直词只禁垂直翻转，新增 `above/below` 垂直词；`legacy` 策略保留用于严格消融。
- 正式消融固定 no-attention 模型、seed 42、split、loss、sampler、阈值与 checkpoint/test 协议，只比较 `legacy` 和 `axis-aware` 两种增强策略。

### 18. 轴感知翻转消融完成
- `legacy` 在第 50 轮早停，raw-best 为第 42 轮、阈值 `0.80`；test 为 `oIoU=0.691092`、`mIoU=0.521327`、类别宏平均 `0.543093`。
- `axis-aware` 在第 54 轮早停，raw-best 为第 53 轮、阈值 `0.70`；test 为 `oIoU=0.698654`、`mIoU=0.530917`、类别宏平均 `0.553549`。
- 轴感知策略提升了 oIoU、官方 mIoU、类别宏平均、Recall、F1 和 Pr@0.5-0.8，但 Precision 与 Pr@0.9 有所下降。
- 20 个语义类别中有 14 个 class-mIoU 提升，最大收益来自 Expressway-Service-area、harbor、tenniscourt、ship 和 vehicle；stadium 回退最大。
- 结论：保留 no-attention head 与 `axis-aware` 默认增强；`legacy` 继续作为可复现实验选项。
