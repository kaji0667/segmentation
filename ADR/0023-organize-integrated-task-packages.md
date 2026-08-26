# ADR-0023: 将分类、计数和指代分割统一为任务包

## 状态

Accepted

## 背景

HFSA 中三个已整合任务形成于不同阶段。指代分割把训练、验证、测试、指标和 checkpoint 编排集中在 `train_semseg.py`；分类和计数虽然已经类化，却分别放在根目录 `classification/`、`counting/`，Application 类仍位于根目录入口。目标检测另有已验证的 `train.py`、`val.py` 和 `text_encoder/` 链路，用户明确要求本次不改目标检测。

目录差异容易让人误以为分类和计数仍是外部源码副本，也让训练、测试入口职责不一致。直接把所有实现摊回根目录会增加耦合；把四个任务强行放进一个 Trainer 又会违反 ADR-0020。

## 决策

1. 新建 `HFSA-main/tasks/`，将指代分割、场景分类和目标计数分别组织为 `tasks/refseg/`、`tasks/classification/`、`tasks/counting/`。
2. 根目录 `train_refseg.py`、`test_refseg.py`、`train_classification.py`、`test_classification.py`、`train_counting.py`、`test_counting.py` 只负责调用对应 Application。
3. 旧 `train_semseg.py` 暂时保留为兼容转发，继续导出历史公开 helper；正式脚本改用 `train_refseg.py` 和 `test_refseg.py`。
4. 指代分割的完整 checkpoint 保存和恢复逻辑包装为 `RefSegCheckpointManager`，放入 `tasks/refseg/checkpoint.py`。该文件是自定义 PyTorch 训练循环的任务专用组件，不要求其他任务建立空的 checkpoint 模块。
5. 分类继续由自身 Trainer 保存独立 Head checkpoint；计数继续委托 Ultralytics 检测 Trainer 管理 checkpoint，不改变原算法。
6. Head 和主任务 Loss 继续位于 `ultralytics/nn/modules/head.py` 与 `ultralytics/utils/loss.py`；本次不修改 Backbone、Neck、OpenCLIP、Head 计算、Loss 数值或数据协议。
7. 目标检测的 `train.py`、`val.py`、`text_encoder/` 和运行命令保持原状，不创建 `tasks/detection/`。

## 备选方案

- 删除任务包并把所有文件放到根目录：拒绝。根目录会继续膨胀，业务逻辑和入口职责混杂。
- 同时重构目标检测：拒绝。超出用户本次授权，也会扩大已验证检测链路的回归范围。
- 为三个任务强制完全相同的文件列表：拒绝。checkpoint、后处理和指标职责取决于任务训练框架；统一的是边界和入口，不是制造无用空模块。

## 影响

- 三项任务现在采用相同的“薄入口 -> tasks 任务包 -> Ultralytics Head/Loss”依赖方向。
- 原 `classification/` 和 `counting/` 源码目录被移除；队友源码快照仍不存在。
- 分割训练核心循环仍在 `tasks/refseg/engine.py` 中，后续可继续将数据构建、指标和可视化按单模块原则拆分，但本次不改变行为。
- 旧 `train_semseg.py` 命令仍可用；部署与新文档应使用 `train_refseg.py` / `test_refseg.py`。
- 验证要求包括三任务专项测试、全库 CPU 回归、脚本 dry-run，以及可用数据任务的真实 checkpoint smoke。

## 关联

- ADR-0020
- ADR-0021
- ADR-0022
- `HFSA-main/tasks/`
- `HFSA-main/scripts/`
