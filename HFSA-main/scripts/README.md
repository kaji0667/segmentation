# HFSA 任务训练与测试脚本

所有脚本都会自动切换到 `HFSA-main`，使用项目相对路径，并允许通过环境变量或命令末尾参数覆盖默认值。

## 指代语义分割

训练脚本已经内置当前正式 baseline 的全部参数，直接调用 `train_semseg.py`，不依赖 `run_semseg_preset.sh`。

训练并在训练完成后测试：

```bash
bash scripts/train_refseg.sh
```

只测试已有 checkpoint，不会重新训练，也不会覆盖原训练目录：

```bash
bash scripts/test_refseg.sh
```

参数通过环境变量覆盖，例如：

```bash
DEVICE=cuda:0 BATCH=8 EPOCHS=80 bash scripts/train_refseg.sh
CHECKPOINT=runs/semseg/other/weights/best_raw.pt SAVE_DIR=runs/semseg/other_eval bash scripts/test_refseg.sh
```

还可在命令末尾追加 `train_semseg.py` 参数；末尾参数优先级最高。

## 目标计数

计数训练保持队友原有的文本引导检测训练逻辑，使用 `CountingDetect` Head 包装和 YOLOv12m 初始化：

```bash
bash scripts/train_counting.sh
```

独立评测已有计数 checkpoint，输出 EM、MAE、RMSE、逐类别统计和可视化：

```bash
bash scripts/test_counting.sh
```

常用覆盖示例：

```bash
VOC_ROOT=data/VRSBench DEVICE=cuda:0 BATCH=8 bash scripts/train_counting.sh
CHECKPOINT=runs/counting/other/weights/best.pt MAX_SAMPLES=20 bash scripts/test_counting.sh
```

计数评测沿用队友代码的 positive-query 协议：每张图只查询 XML 中实际存在的类别。该口径会在报告中显式记录，不与包含零计数查询的评测混用。
