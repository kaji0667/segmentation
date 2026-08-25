# ADR-0020: 使用任务专用 Trainer 与薄统一分发入口

## 状态

Accepted

## 背景

团队最终需要在统一 YOLOv12m Backbone/Neck 基础上整合检测、指代分割、目标计数、场景分类等不同子任务。各任务虽然共享视觉主干，但数据格式、Head 输出、Loss、训练循环和评测协议不同。例如现有 `train.py` 使用 Ultralytics `YOLO.train()` 与文本引导检测 Trainer，输出检测框；`train_semseg.py` 使用 RRSIS-D JSONL/RLE mask、自定义 PyTorch 训练循环与二值分割指标，输出单通道 mask。

强行把这些训练细节塞进一个巨型 `train.py` 会制造大量任务条件分支，并让检测、分割、计数和分类的 Dataset、Loss、Validator 相互耦合。当前交付还要求每个任务能在独立数据集上单独训练和评测。

## 决策

1. 保留任务专用训练实现。检测、指代分割、计数、分类等任务分别拥有自己的 Trainer/训练文件、数据适配、Loss 和评测逻辑。
2. 每个任务提供自包含的 `scripts/train_<task>.sh` 与 `scripts/test_<task>.sh`。脚本必须从自身位置定位项目根目录、使用项目相对路径、显式传入数据/模型/权重/输出配置，并直接调用对应任务入口。
3. 如果团队最终要求一个统一训练命令，公共 `train.py` 只作为薄分发器：解析 `--task` 或任务编号后调用对应任务 Trainer；分发器不实现任务内部 Dataset、Loss、训练循环或指标。
4. 不采用联合多数据集、同时激活多个 Head 的联合训练作为当前整合方案。当前方案是按任务选择 Trainer、Head、数据集和 checkpoint，分别训练与评测。
5. 最终用户输入编号或“语义分割、场景分类”等任务名称后的自动切换属于统一推理入口。推理路由与训练脚本分离，负责选择对应模型配置、Head、checkpoint、预处理和后处理。
6. 所有任务继续遵守统一 YOLOv12m Backbone/Neck 接口；未经确认不得为适配单个任务修改公共 Backbone/Neck。

## 备选方案

- 将所有任务合并进一个巨型训练循环：拒绝。不同任务的数据和监督语义差异过大，条件分支难以维护、测试和复现。
- 只保留 Shell 脚本、不提供任何统一入口：当前阶段可用，但若最终演示或师兄要求统一命令，仍需要一个薄分发器提供稳定接口。
- 同一 batch 联合训练所有 Head：暂不采用。不同任务使用不同数据集，尚未确定采样、Loss 权重、缺失标签和资源调度策略，会显著扩大当前整合范围。

## 影响

- 正面：各任务可以独立开发、训练、测试和定位问题；不会为了统一入口破坏已验证的任务训练流程。
- 正面：Shell 脚本提供一致的运行方式，后续统一入口只需维护任务到 Trainer 的映射。
- 代价：不同 Trainer 之间可能存在少量参数解析、日志和 checkpoint 编排重复，后续只能在确认真正通用后逐步抽取。
- 代价：统一资源评测需要另行规定相同 A5000、batch、imgsz、计时区间和显存统计方法，不能直接比较各成员电脑上的旧日志。
- 测试要求：每个任务脚本必须独立通过 dry-run/smoke；薄分发器加入后必须验证它传递的参数与直接运行任务脚本一致。

## 关联

- `HFSA-main/train.py`
- `HFSA-main/train_semseg.py`
- `HFSA-main/scripts/train_refseg.sh`
- `HFSA-main/scripts/test_refseg.sh`
- `CURRENT_STATE.md`
- `ARCHITECTURE.md`
