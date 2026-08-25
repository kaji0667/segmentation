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

## 场景分类

先将 VOC 风格 VRSBench 中“恰好一个场景类”的图像整理为 ImageFolder。默认使用硬链接，避免重复占用图像空间；输出目录必须为空，工具不会删除旧数据：

```bash
python prepare_classification_data.py \
  --voc-root data/VRSBench \
  --output-dir data/VRSBench_scene
```

训练冻结共享 YOLOv12m Backbone/Neck，只优化 `SceneClassifyHead`：

```bash
bash scripts/train_classification.sh
```

在固定类别顺序上评测已有 checkpoint，输出 top-1/top-5、macro Precision/Recall/F1、混淆矩阵和逐类指标：

```bash
bash scripts/test_classification.sh
```

常用覆盖示例：

```bash
DATA_DIR=data/NWPU-RESISC45 EPOCHS=35 BATCH=32 bash scripts/train_classification.sh
CHECKPOINT=runs/classification/other/weights/best.pt SPLIT=test bash scripts/test_classification.sh
```

该任务保留队友代码的 P3/P4/P5 多尺度空间注意力 + GeM、类别平衡采样、CrossEntropy 和 warmup/cosine 训练流程。当前只完成最小 CPU smoke，不代表完整数据效果。
