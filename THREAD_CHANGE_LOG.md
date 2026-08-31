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

### 19. 最终指标精简与固定 checkpoint 复评
- 按最终确认口径，核心指标固定为 oIoU、mIoU 和 Pr@0.5-0.9；不再使用含义可能冲突的 cIoU/gIoU 别名。
- 最终报告删除每类别 IoU、class-macro-mIoU、pixel accuracy、背景/前景二分类 mIoU、class-oIoU 和历史 `target_iou`/`sample_miou` 别名；训练代码中的兼容字段不做破坏性删除。
- 使用 `noattn_aug_axis/weights/best_raw.pt`（epoch 53）和验证集冻结阈值 `0.70` 重新评估全部 3,481 个 test 表达样本，没有重新训练或修改模型结构。
- 复评结果：`oIoU=0.698654`、`mIoU=0.530917`、`Pr@0.5/0.6/0.7/0.8/0.9=0.600115/0.516806/0.407354/0.295030/0.142488`；与原测试结果一致。Precision、Recall 和 F1 仅作为非核心辅助诊断。
- 原始文本端到端 batch=1 平均/P95 为 `141.48/179.83 ms`，峰值 GPU `1,754.27 MB`，峰值进程 RSS `4,638.53 MB`；统计包含 JPEG 解码、OpenCLIP 文本编码、分割和阈值后处理。
- 新结果存于 `runs/semseg/noattn_aug_axis/final_evaluation_20260822.json` 与 `.md`，旧 `test_results.json` 未覆盖。

### 20. 目标-背景双流解码候选
- 仅修改 `TextPromptSegment` 最后的二值掩膜解码：共享融合输入分别送入参数独立且结构对称的目标流和背景流。
- 最终输出为 `target_logits - background_logits + similarity + bias`，保持 `[B,1,H,W]`、现有 BCE-Tversky loss、阈值选择和测试接口不变。
- backbone、neck、OpenCLIP、数据、增强和指标协议均未修改；不增加辅助 loss，确保相对 `noattn_aug_axis` 是单变量结构消融。
- 新增双流参数独立、双路梯度、输出尺寸和差值符号测试；正式运行目录为 `runs/semseg/tbtd`，完成前不能称为提升。

### 21. 目标-背景双流完整实验结果
- 正式实验在第 57 轮早停，raw-best 为第 49 轮，冻结阈值 `0.70`；完整 test 3,481 样本。
- test：`oIoU=0.693151`、`mIoU=0.532644`、`Pr@0.5-0.9=0.597817/0.517380/0.410227/0.299052/0.145361`。
- 相对 `noattn_aug_axis`：mIoU 与 Pr@0.6-0.9 小幅提升，Precision 提升且预测正像素率更接近目标；但 oIoU、Pr@0.5、Recall 和 F1 回退。
- 结论：双流不是全面提升，不替代当前 no-attention axis-aware 单流主线；实验代码、测试和产物保留供后续研究更强的目标/背景互补约束。
- 活动源码随后恢复为单流 `mask_decoder`；双流完整实现保留在 Git 提交 `c718f78`，结果文档与本地实验产物继续保留。

### 22. 图像条件双向 token adapter 完整实验与回退
- 在 `TextPromptSegment` 内加入低分辨率 `8 x 8` region-token 双向交互：text 查询 visual regions，regions 再查询 adapted text；两个 residual gate 均从零初始化。
- 14 项语义分割测试与 2-train/2-val/2-test CUDA smoke 通过；完整 `bta_axis` 在第 55 轮早停，raw-best 为第 47 轮，冻结阈值 `0.80`，test 覆盖 3,481 样本。
- test：`oIoU=0.694395`、`mIoU=0.539623`、`Pr@0.5-0.9=0.602126/0.521689/0.422867/0.304797/0.152255`。
- 相对 `noattn_aug_axis`：mIoU `+0.008707` 且全部 Pr 指标提升，但 oIoU `-0.004259`、Recall `-0.016560`、F1 `-0.002960`；参数、延迟和显存也增加。
- checkpoint 中 text/visual gate 的 `tanh` 为 `-0.015298/-0.064627`，说明交互分支实际参与训练；混合结果不是 gate 未开启所致。
- 结论：它改善逐样本与高 IoU 成功率，但不是对综合最优的明确超越。活动源码已恢复为 no-attention、axis-aware、单解码主线，候选仅保留本地实验产物，未推送 GitHub。

### 23. 文本贯穿渐进 decoder 完整实验与回退
- 在 `TextPromptSegment` 内实现共享参数的 P5→P4→P3 top-down decoder，每一级重复使用文本 FiLM、像素文本相似度、空间门控、value 分支和同一个 mask decoder。
- P5/P4 仅通过零初始化可学习 gate 作为内部 residual 加到 P3；没有增加 P3/P4/P5 辅助 loss，最终仍只输出一个二值 mask。
- 20 项测试与 2-train/2-val/2-test CUDA smoke 通过；完整 `tpd_axis` 在第 37 轮早停，raw-best 为第 29 轮，冻结阈值 `0.60`。
- test：`oIoU=0.685062`、`mIoU=0.511833`、`Pr@0.5-0.9=0.557886/0.482045/0.377190/0.275783/0.130997`。
- 相对 `noattn_aug_axis`，oIoU `-0.013592`、mIoU `-0.019084`，全部 Pr、Recall 和 F1 均回退；参数增加 295,426，平均 test 延迟从 `23.17` 增至 `28.64 ms/sample`。
- 两个粗尺度 gate 的 tanh 为 `0.325729/0.439594`，说明残差实际启用。候选被拒绝，活动源码恢复单 decoder 基线后再进入学习型语义角色 pooling 实验。

### 23. PDF 转 Markdown 后的资料入口更新
- 比赛方案首选入口改为根目录 MinerU Markdown；涉及比赛评测、报告或提交要求时，按需读取两次统一答疑 Markdown。
- 根目录其他 `MinerU_markdown_*.md` 登记为本地 RRSIS 论文语料库，按任务和标题选择相关论文，不为无关任务全量读取。
- Markdown 优先用于检索和章节定位；公式、表格、结构图、页码和 OCR 可疑内容必须通过原图、公开论文或源码交叉核对。
- 19 份转换稿保留在本地，不随本次规则更新公开提交；论文全文和比赛联系方式只有在用户明确确认后才能发布。
- 本次仅修改资料阅读与文档规则，没有改变模型、训练或评估实现。
### 24. 学习型目标/关系/位置 token pooling 完整实验
- 在共享 token scorer 上增加三个零初始化角色 residual scorer；target 用于 FiLM，relation 用于尺度选择，target/relation 均值用于主相似度，position 通过独立相似度和零初始化 gate 进入 spatial gate。
- 19 项全量测试与 2-train/2-val/2-test CUDA smoke 通过；正式 `srp_axis` 在第 52 轮早停，raw-best 为第 44 轮，冻结阈值 `0.80`。
- test：`oIoU=0.700990`、`mIoU=0.539022`、`Pr@0.5-0.9=0.598391/0.516231/0.413100/0.305085/0.151106`。
- 相对 `noattn_aug_axis`，oIoU 与 mIoU 同时提升，Pr@0.6-0.9、Precision 和 F1 提升；Pr@0.5 与 Recall 轻微回退，参数增加 100,740，平均 test 延迟增加 5.78 ms/sample。
- 三个角色 scorer、角色 bias 和 position gate 均明显分化。该候选作为第三项 uncertainty-gated P2 residual 的受控基座，待第三项结束后统一决定。

### 25. 不确定边界 P2 residual 完整实验、回退与最终发布选择
- P2 仅作为独立 residual 使用，不进入 P3/P4/P5 coarse fusion；硬 mask 同时要求 detached uncertainty `>=0.5` 和 3x3 boundary strength `>=0.05`。
- 23 项测试与 CUDA smoke 通过；正式 `p2ubr_axis` 在第 52 轮早停，raw-best 为第 44 轮，冻结阈值 `0.80`。
- test：`oIoU=0.694145`、`mIoU=0.535108`、`Pr@0.5-0.9=0.593795/0.510773/0.404194/0.300201/0.151681`。
- 相对 `srp_axis`，两个主 IoU、Pr@0.5-0.8、Precision、Recall 和 F1 均回退，仅 Pr@0.9 `+0.000575`；参数增至 `4,821,421`，平均评测耗时 `86.04 ms/sample`，峰值 GPU `558.66 MB`。
- residual 最后一层权重范数 `1.011367`；完整 test mask 平均覆盖 `1.9474%` 像素，说明 P2 分支确实学习且只局部启用。
- 结论：拒绝 P2 residual，移除活动配置、测试和 head 分支；恢复第 24 项学习型 target/relation/position pooling 作为综合最优并提交推送。

### 26. 语义分割主线切换为 YOLOv12m 初始化
- 保留 ADR-0015 学习型 target/relation/position token pooling、P3/P4/P5、OpenCLIP、loss、axis-aware 增强和评估协议不变。
- `train_semseg.py` 与 `run_semseg_preset.sh` 现在默认使用 `yolov12m-semseg.yaml` 和 `pretrain_model/yolov12m.pt`；m alias 复用统一 YAML，但明确解析 `scale=m`，避免只换权重导致通道不匹配。
- 静态审计确认总参数 `20,273,356`，预训练权重匹配 `678/762` tensors；冻结 backbone 后 trainable/frozen 为 `9,493,644/10,794,304`。
- 21 项测试、batch 2 smoke 和 baseline batch 4 的 2-train/2-val/2-test CUDA smoke 均通过；正式 seed-42 完整运行目录为 `runs/semseg/srp_yolov12m_axis`。
- 正式运行第 52 轮早停，raw-best 为第 44 轮、冻结阈值 `0.70`；test `oIoU=0.701171`、`mIoU=0.552562`，Pr@0.5-0.9 全部高于 n-scale `srp_axis`。
- mIoU 提升 `0.013540`，但 oIoU 与 F1 基本持平；模型参数、checkpoint 和峰值 test GPU 分别约为 n-scale 的 `4.98x/4.09x/2.14x`。
- 结论：YOLOv12m 是准确率优先候选，不是已证明的资源效率综合最优；默认 m-scale 按用户要求保留，最终提交需明确准确率与星载部署效率优先级。

### 27. 新增指代语义分割独立训练/测试脚本
- `HFSA-main/scripts/train_refseg.sh` 已将原 baseline 所需参数完整展开，直接调用 `train_semseg.py`；默认使用 YOLOv12m、RRSIS-D、batch 4、60 epochs、patience 8，并在训练完成后测试，部署时不需要携带 `run_semseg_preset.sh`。
- 新增 `HFSA-main/scripts/test_refseg.sh`，默认读取 `runs/semseg/srp_yolov12m_axis/weights/best_raw.pt`，输出到独立的 `runs/semseg/srp_yolov12m_axis_eval`，不会重新训练或覆盖原训练目录。
- `train_semseg.py` 新增 `--eval-only --checkpoint`：只构建 test split，严格加载完整 checkpoint，并复用其中保存的验证阈值。
- 两个脚本都通过脚本自身位置定位 `HFSA-main`，数据、模型、权重和输出全部使用项目相对路径，无本机盘符或 `/mnt` 硬编码。
- 两份代码均通过 Shell 语法检查、Python 编译、25 项全库测试；训练脚本的有/无训练后测试两种参数展开均通过，2-batch CUDA 只评测 smoke 成功。

### 28. RRSIS-D 空标注清洗与增强审计

- 全量审计 17,402 条表达，未发现重复 ID、空文本、非法类别或方向词漏拦截；发现 train 两条、test 一条 RLE 解码后前景面积为零。
- 默认 `--empty-mask-policy drop` 后 split 为 `12179/1740/3480`；`error` 用于严格审计，`keep` 用于复现旧协议。空 mask 不使用 bbox 补伪标签。
- 最近邻缩放到 512 未使任何非空 mask 消失，因此保留现有 mask resize；axis-aware 翻转和 `0.15` 色彩扰动不变，只新增参数范围校验。
- 数据准备不再把空文本静默替换为类别名，类别字段优先标准 `category_id` 并兼容旧 `categories_id`。
- 29 项测试和 YOLOv12m batch-4 的 2-train/2-val/2-test CUDA smoke 已通过；正式运行目录定为 `runs/semseg/srp_yolov12m_axis_clean_empty`。

### 29. 固化多任务训练入口整合方案

- 检测、指代分割、计数和分类保留各自 Trainer、数据加载、Head、Loss、checkpoint 与评测逻辑，不合并为一个巨型训练循环。
- 每个任务提供自包含的 `scripts/train_<task>.sh` 和 `scripts/test_<task>.sh`；分割脚本直接调用 `train_semseg.py`，不依赖 `run_semseg_preset.sh`。
- 若最终需要公共 `train.py`，它只解析任务编号/名称并分发到对应 Trainer；最终编号或任务名称切换由独立推理入口负责。
- 新增 ADR-0020 与 `CURRENT_STATE.md`，记录当前脚本、空 mask 清洗协议、历史 YOLOv12m 指标和 A5000 下一步。
- 本次仅更新架构与跨对话文档，没有产生新的完整模型实验结果；cleaned 3480-sample 正式训练仍待完成。

### 31. 将目标计数包装为独立 Head 与类化任务模块

- 新增 `CountingDetect(Detect)`，不重写前向过程；其参数、检测输出和 Loss 契约与原 `Detect` 等价。新增 `yolov12-counting.yaml`，正式入口通过 `yolov12m-counting.yaml` 使用统一 m-scale Backbone/Neck。
- 将队友原有计数配置、VOC XML 解析、letterbox、OpenCLIP Prompt、NMS 计数、EM/MAE/RMSE、可视化分别包装为类，保留“检测框数量即计数”的原逻辑。
- 新增 `train_counting.py`、`test_counting.py`、`scripts/train_counting.sh` 和 `scripts/test_counting.sh`；清除个人绝对路径、显卡名称和工程目录，默认均为项目相对路径。
- 计数评测继续采用 positive-query VOC 协议，只测试 XML 中存在的类别，并在 JSON 报告中明确记录，暂不扩展零计数查询。
- 6 项计数定向测试、脚本语法/参数展开、预训练静态加载通过；发布仓库全量 39 项 CPU 测试通过。由于用户正在运行实验，本次未启动 GPU smoke。
- 队友仓库缺少训练 checkpoint 和随仓库数据，真实 VRSBench 训练/测试 smoke 等拿到相应路径和权重后再执行。
# 2026-08-25 cleaned 实验收尾与场景分类任务接入

- `runs/semseg/srp_yolov12m_axis_clean_empty/test_results.json` 已完成；新训练不如 `srp_yolov12m_axis_clean_eval/test_results.json` 的 cleaned-test 公平基线，不替换发布 checkpoint。
- 活动 `HFSA-main/train_semseg.py` 已同步发布版可靠 resume 实现；临时恢复入口不再是唯一可恢复脚本。
- 活动副本已确认包含完整 CountingDetect 计数实现。
- 新增 `SceneClassifyHead`、`yolov12-classification.yaml`、`classification/`、`prepare_classification_data.py`、`train_classification.py`、`test_classification.py`、两个 Shell 脚本和分类专项测试。
- 场景分类参考来源固定为 `zhuoletian-collab/changjingfenlei@688c2a9`；发布时只带可审计源码和 provenance，不带嵌套 Git、checkpoint、数据、runs、缓存或个人配置。
- 分类 CPU 最小训练/测试 smoke 已通过；真实完整训练和 GPU smoke 未执行。

### 32. 统一三个非检测任务的目录与入口

- 新建 `HFSA-main/tasks/`，指代分割、场景分类和目标计数分别迁入 `tasks/refseg/`、`tasks/classification/`、`tasks/counting/`；原根目录 `classification/`、`counting/` 源码目录不再存在。
- 六个根目录训练/测试 Python 文件统一为薄入口；Application、数据、训练、评测、推理和指标类位于对应任务包。
- 指代分割新增真正独立的 `train_refseg.py` 和 `test_refseg.py`；`train_semseg.py` 只保留历史兼容转发。
- 原分割 checkpoint 保存与恢复函数包装为 `RefSegCheckpointManager`。它负责自定义训练循环的 optimizer、scheduler、early-stop、阈值、RNG 和 CSV 恢复；分类和计数继续使用各自原有 checkpoint 机制。
- 目标检测 `train.py`、`val.py`、`text_encoder/` 未修改，也未新增 `tasks/detection/`。
- 分类 `8/8`、计数 `6/6`、分割入口 `5/5`、checkpoint `7/7`、目录结构 `3/3` 均通过；活动与发布副本最终全量 CPU 回归均为 `51/51`。
- 新入口真实 CPU smoke：分割严格加载原 `best_raw.pt` 评测 1 batch；分类严格加载原 `best.pt` 评测 1 batch。计数仍因没有真实数据和训练 checkpoint 无法执行真实 smoke。

### 32. 场景分类整合复核与活动副本最终同步

- 独立复核 `SceneClassifyHead`、m-scale YAML、冻结权重加载、ImageFolder/VRSBench 数据、训练、评测和 checkpoint 契约，确认保持队友 P3/P4/P5 空间注意力 + GeM 主逻辑。
- 修复单图推理对数据集目录的无关依赖，修复显式 split 布局下 `--split all` 的类别误判，并避免 `BatchNorm1d` 接收末尾单样本训练 batch。
- 活动副本补齐发布仓库已有的 legacy cosine scheduler 恢复测试；`train_semseg.py`、计数、分类核心实现和测试集合均已同步。
- 分类 7 项定向测试、缺失数据目录条件下的单图 Top-K、全新 1-train/1-val/1-test CPU smoke 均通过；活动与发布副本完整 CPU 回归均为 46 项通过。
- GPU 仍有其他负载，本轮未启动 CUDA smoke 或完整场景分类训练；真实训练仍是后续事项。

### 33. 队友源码快照移除与分类 Loss 正式接入

- 新增统一 `SceneClassificationLoss`，位于 `ultralytics/utils/loss.py`；分类 Trainer/Evaluator 不再直接实例化裸 `nn.CrossEntropyLoss`，但数学公式保持不变。
- 活动副本中的 `HFSA-Object-Counting` 和 `scene_classification_reference` 完整嵌套仓库已移入回收站；发布仓库中的 11 个分类参考源码文件已删除。
- 融合后的 `CountingDetect`、`counting/`、`SceneClassifyHead`、`classification/`、模型 YAML、Loss、训练/测试脚本和测试均保留，且无队友目录运行时依赖。
- `PROJECT_RULES.md` 新增长期交付边界：队友仓库只能作为临时审计输入，融合后不得保留完整源码快照或嵌套 Git；来源由 ADR 的仓库地址和 commit 追溯。
- 分类专项测试 `8/8`、活动与发布完整 CPU 回归 `47/47`、新 Loss 的 1-train/1-val/1-test CPU smoke 均通过；GPU smoke 和完整场景分类训练仍待 GPU 空闲后执行。

### 34. 场景分类正式训练脚本改为直接启动

- `scripts/train_classification.sh` 已内置正式训练数据、模型、预训练权重、输出目录和超参数默认值，正常使用不再要求手工输入一串环境变量。
- 首次执行若缺少 `data/VRSBench_scene`，脚本会自动从 `data/VRSBench` 整理完整 21 类 ImageFolder 数据；后续执行直接复用。
- 数据整理保持原“恰好一个场景类别”筛选规则和硬链接默认方式，不修改分类算法、Head、Loss 或 Trainer。
- 检测到非空但无类别图片的残缺输出目录时显式停止，避免覆盖用户数据。

### 35. 指代分割训练/测试拆分与前 N 个测试 batch 预览

- `scripts/train_refseg.sh` 删除自动 test 部分，只执行训练与每轮 validation；测试改为训练完成后单独运行 `scripts/test_refseg.sh`。
- 独立测试新增 `TEST_PREVIEW_BATCHES`，默认 `5`，输出 `test_batch0_pred.jpg`、`test_batch1_pred.jpg` 等；实际测试 batch 少于请求数时只生成存在的部分。
- `test_results.json` 记录请求数量与实际预览文件列表；指标、验证阈值、最佳 checkpoint 选择和模型计算均未变化。
- 定向测试 `7/7`、Shell 语法/展开检查和全库 CPU 回归 `53/53` 通过；服务器真实 GPU smoke 与正式运行由用户上传增量包后执行。

### 36. 指代分割单图推理与手动总路由边界

- 新增 `tasks/refseg/inference.py`：`RefSegPredictor` 输入单张图像和非空指代表达，严格加载完整 checkpoint，在线生成训练同口径 OpenCLIP token features，并按 checkpoint validation 阈值输出原图尺寸 mask。
- `RefSegPrediction` 可保存 mask、概率图、叠加图和 JSON；模块支持 `python -m tasks.refseg.inference`，也可供未来最外层路由直接调用。
- 总路由已确认由用户手动选择任务；分类只要求图像，计数要求图像和目标文本，指代分割要求图像和指代表达，各任务保留自身输出。
- 按用户确认删除 `train_semseg.py`；测试改为直接引用 `tasks.refseg.engine` / `checkpoint`，历史 preset 改调 `train_refseg.py`。
- 验证：RefSeg 定向 `10/10`、发布仓库全库 CPU `56/56`、活动副本 `57/57`；真实 YOLOv12m epoch-44 checkpoint 在 CPU/512 输入上完成单图端到端 smoke 并生成可视化产物。

### 37. 本地 Web 界面与三任务路由骨架

- 新增零新增依赖的 `web_app.py` 与 `web/` 静态界面，默认在 `127.0.0.1:7860` 提供三任务卡片、图像预览、动态输入和结果工作区。
- 新增 `tasks/routing/`，统一管理可序列化任务配置、输入校验和延迟 Adapter 注册；不统一三个任务的业务输出。
- 当前 Adapter 明确为待接入状态，页面点击分析后由后台返回 `interface_pending`，没有加载模型或生成伪结果。
- 新增 ADR-0027、`WEB_APP.md` 和路由/Web 定向测试；公网访问配置留到最后部署阶段。

### 38. Web 首轮界面评审收敛

- 删除 Hero 下方操作步骤条，减少无实际功能的视觉元素。
- 任务选择改为紧凑的配置驱动自动流式布局，不再用三个超大卡片暗示系统只能扩展三项任务。
- 分析工作区默认隐藏，用户选定任务后才显示对应输入、结果和任务状态。
- 用户展示名称改为赛题用语“场景分类”“语义分割”，内部任务 ID 和实现模块不改；卡片说明支持两行完整展示。
- 用户界面品牌替换为项目正式名称“遥感图-文可解释轻量化多任务智能解译系统”；完整名称用于浏览器标题与 Hero，左上角采用紧凑简称，不修改仓库内部 HFSA 标识。
