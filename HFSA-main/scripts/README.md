# 指代语义分割脚本

两个脚本都会自动切换到 `HFSA-main`，其中的数据、模型、权重和输出目录均为项目相对路径，可以从任意目录调用。

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
