# ADR-0026: 手动任务选择与任务专用推理接口

## 状态

Accepted

## 背景

最终系统需要让用户在指代分割、目标计数、场景分类等任务之间切换。不同任务的必要输入和输出语义并不相同：指代分割需要图像与指代表达并输出 mask；目标计数需要图像与目标类别文本并输出数量和检测框；场景分类只需要图像并输出类别概率。把所有任务强制成统一的“图像 + 文本 -> mask”接口会破坏任务边界。

## 决策

1. 最外层路由由用户手动选择任务编号或名称，不在第一版中自动猜测用户意图。
2. 路由选定任务后加载该任务 checkpoint，再由任务适配器声明并校验自身需要的输入。
3. 各任务保留自身输出类型，不强制转换为同一种结果：
   - 指代分割：图像 + 指代表达 -> 二值 mask、概率图和叠加图。
   - 目标计数：图像 + 目标类别文本 -> 计数、检测框和置信度。
   - 场景分类：图像 -> Top-K 场景类别和概率。
4. `tasks/refseg/inference.py` 作为指代分割单图推理边界，严格加载完整 checkpoint，在线生成与训练一致的 OpenCLIP token features，并复用 checkpoint 保存的 validation 阈值。
5. 本决策不合并各任务 Trainer、Dataset、Loss、checkpoint 格式或后处理，也不修改 Backbone、Neck、OpenCLIP 或任务 Head。

## 备选方案

- 根据用户自然语言自动识别任务：暂不采用。容易在“找出目标”“统计目标”“分割目标”等相近表达间误路由，且不利于比赛演示和故障定位。
- 所有任务统一为固定输入输出结构：拒绝。只能统一路由元数据，不能抹平任务语义。
- 路由直接调用数据集测试脚本：拒绝。测试脚本面向整 split 评测，不能替代任意单图推理接口。

## 影响

- 总路由可以通过稳定的任务类接口延迟加载所选模型。
- 每个任务必须独立补齐单样本 predictor；当前先完成指代分割。
- 测试必须覆盖 checkpoint 严格加载、任务输入校验、输出尺寸恢复和结果文件生成。

## 关联

- ADR-0020
- ADR-0023
- `HFSA-main/tasks/refseg/inference.py`
- `HFSA-main/tasks/classification/engine.py::SceneClassificationPredictor`
- `HFSA-main/tasks/counting/inference.py::ObjectCounter`
