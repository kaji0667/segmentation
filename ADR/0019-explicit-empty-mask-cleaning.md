# ADR-0019: Explicitly Clean Empty RRSIS-D Masks

## 状态

Accepted；完整 YOLOv12m 受控实验已完成，清洗策略保留，但新权重不替换当前发布 checkpoint。

## 背景

对 RRSIS-D 全部 17,402 条指代表达进行 RLE 解码审计后，发现 `train_22187`、`train_20203` 和 `test_413` 的 segmentation 前景面积为零。它们仍带有非空 bbox 和正常文本，因此旧加载器会把这些标注当成全背景监督，其中两条会进入训练，一条会进入官方 test。

同一审计确认：没有非空 mask 在最近邻缩放到 512 后消失，当前方向词规则也没有漏掉语料中的候选方位词。因此不能用这两个未经证实的问题替代确定的空标注清洗。

## 决策

1. 在数据准备和运行时加载阶段校验 sample ID、文本、class index、RLE 尺寸、count 总和与前景面积。
2. 默认使用 `empty_mask_policy=drop`，显式告警并记录被剔除的 sample ID；提供 `error` 严格模式和 `keep` 历史复现模式。
3. 不根据 bbox 为零前景标注补矩形 mask，避免引入无法由原始 segmentation 支持的伪标签。
4. 保持最近邻 mask resize、axis-aware 翻转、0.15 亮度/对比度扰动及全部模型/训练/评估设置不变。
5. 正式实验继续使用团队统一的 `yolov12m.pt` 与匹配的 m-scale 配置。先用旧 checkpoint 在清洗后 test split 复评，再训练新模型，以分离 test 样本移除和训练清洗的影响。

## 备选方案

- 保留空 mask：拒绝，因为会把标注错误解释为可靠背景监督，并让 test mIoU 中出现语义不明确的空目标。
- 用 bbox 生成矩形 mask：拒绝，因为 bbox 不能恢复真实轮廓，尤其 `test_413` 的大 bbox 会制造严重伪标签。
- 改用 `INTER_AREA > 0` 下采样：拒绝，因为最近邻没有造成空 mask，而该方案在审计中最高把目标面积放大到期望值的 `1.528x`。
- 新增旋转、裁剪或更强色彩增强：暂不采用，因为会同时改变多个变量，并可能破坏文本空间关系。

## 影响

- 活动 split 从 `12181/1740/3481` 变为 `12179/1740/3480`。
- 小目标采样器的 train 小目标计数从 2966 变为 2964；其余有效样本增强分布不变。
- 旧实验可通过 `--empty-mask-policy keep` 完整复现。
- 新旧完整指标必须在相同的 3,480-sample cleaned test 上比较，旧 3,481-sample 报告只作为历史参考。

## 完整受控实验结果（2026-08-25）

- 正式目录：`runs/semseg/srp_yolov12m_axis_clean_empty`。
- 协议：`yolov12m.pt`、m-scale、seed 42、batch 4、imgsz 512、60 epochs、patience 8、axis-aware、`empty-mask-policy=drop`、官方 mIoU 选模、完整 3,480-sample test。
- epoch 29 保存 `last.pt` 时遇到 Windows/WSL 文件锁；增加原子 checkpoint 保存、有限重试、完整 optimizer/scheduler/early-stop/RNG 恢复和协议一致性检查后从 `last.pt` 继续。最终 epoch 39 早停，raw-best 为 epoch 31，冻结阈值 `0.80`。
- 新模型 test：`oIoU=0.693618`、`mIoU=0.539086`、`Pr@0.5-0.9=0.598563/0.520690/0.410920/0.299713/0.154885`、Precision `0.796174`、Recall `0.843378`、F1 `0.819096`。
- 预测/目标正像素率为 `0.049432/0.046665`；平均延迟 `22.026 ms/sample`，峰值 GPU `812.316 MB`，参数 `20,287,948`，checkpoint `150.365 MB`。
- 使用旧 `srp_yolov12m_axis/weights/best_raw.pt` 在相同 cleaned test 公平复评：`oIoU=0.702938`、`mIoU=0.552721`、`Pr@0.5-0.9=0.624138/0.540517/0.425575/0.319253/0.164368`、Precision `0.806952`、Recall `0.845044`、F1 `0.825559`。
- 新训练相对公平基线：oIoU `-0.009320`、mIoU `-0.013635`，五档 Pr 全部回退，Precision `-0.010778`、Recall `-0.001666`、F1 `-0.006462`；预测正像素率略升 `+0.000564`。
- 结论：剔除确定为空的错误标注仍是正确的数据治理决策，但仅删除两条空训练 mask 没有带来泛化收益。本次新权重不替换发布 checkpoint，当前发布继续使用旧 epoch-44 checkpoint；cleaned 3,480-sample 复评作为今后的公平基线。

## 关联

- `HFSA-main/dataset/rrsisd_refseg_dataset.py`
- `HFSA-main/train_semseg.py`
- `HFSA-main/scripts/train_refseg.sh`
- `HFSA-main/scripts/test_refseg.sh`
- `HFSA-main/tests/test_rrsisd_axis_aware_augmentation.py`
- `runs/semseg/srp_yolov12m_clean_empty_smoke`
- `runs/semseg/srp_yolov12m_axis_clean_empty/test_results.json`
- `runs/semseg/srp_yolov12m_axis_clean_eval/test_results.json`
