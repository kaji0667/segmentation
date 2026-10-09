# CURRENT_STATE.md

## 2026-10-09 最新状态：官方 API 已接入四任务并通过真实本机检查

- 当前入口为 `提交/HFSA-main/participant_api_starter-main/`，正式权重位于同级 `提交/HFSA_models/`。`model_adapter.py::RealModel` 委托新 `hfsa_adapter.py`，复用已有场景分类、检测、计数和 RefSeg Predictor；主模块外只调整这一薄入口及文档/测试。
- 官方协议没有 task_id，新 Adapter 只按明确题干形式和答案约束选择任务。支持 21 类场景分类（短文本/单选/枚举）、具体类别的目标有无、独立计数权重输出整数字符串/数值选项，以及检测最高置信度单框或 RefSeg mask 外接框。网页仍手动选任务，模型结构/训练/Loss 未改。
- 已在 WSL `hfsa_env` / RTX 4060 Laptop GPU 完成官方及新增单元测试 `25/25`。真实 HTTP smoke 使用 800×800 RRSIS-D `03600.jpg` 和临时合成图片包：分类 `U` 对应 `windmill`，计数 `"1"`，有无 `"Yes"`，检测框 `[351,272,388,390]`，RefSeg 框 `[360,271,387,373]`；认证、ready、ID 回显及双图明确失败均通过。
- 本次完整模型 startup `100.64s`，单图请求约 `0.09–1.32s`，峰值 PyTorch allocated 显存 `3.86 GiB`。检测/计数共用文本编码器的实例身份已核验；这些只是部署 smoke 证据，不是正式效率结果或准确率。
- 双图变化、通用问答、泛化 anything 有无题、空间关系/多类别计数及未映射类别明确拒绝，官方外壳返回 `500/inference_failed`。不声称完成全部开发集；官方通用 checker 的 anything 题仍不适用。
- 本机测试服务及临时包均已关闭/清理，测试 Key 未打印/保存。用户暂缓公网，没有隧道、端点登记或网站评测提交。启动请使用 API README 第 8 节的 WSL 命令，等 `API listening` 出现。
- API 源码提交 `747723a` 已成功推送到 GitHub 分支 `codex/hfsa-official-api-20261009`，远端 main 保持原状。独立发布工作区 `api发布_20261009/` 补齐必要已有 Predictor，真实四任务输出与交付副本一致；活动根目录 Git 仍未重建。


最后更新：2026-10-09

## 2026-08-31 最新状态：RefSeg 测试命名收敛

- 三个仍验证当前 Head 的 `test_semseg_*` 文件合并为 `test_refseg_head.py`；文本输入与默认配置测试分别改名为 `test_refseg_text_inputs.py`、`test_refseg_defaults.py`。
- 原有 13 条断言全部保留；新文件定向 `13/13`、全库 CPU `65/65` 通过。
- 本次只调整测试文件组织与名称，不修改运行时代码或模型结构。
- 提交 `dbb9d4b` 已推送到 GitHub `main`。

## 2026-08-31 最新状态：RefSeg 严格单 checkpoint

- RefSeg 训练唯一 checkpoint 为 `weights/best_raw.pt`；payload 只包含模型、参数、数据配置、epoch、格式和指标。
- 已删除 `--resume`、optimizer/scheduler/RNG 恢复 checkpoint、旧 `best.pt` 自动回退，以及对应的两个历史测试文件。
- 必要的严格单-checkpoint校验并入 `test_refseg_scripts.py`；RefSeg 定向 `9/9`、全库 CPU `65/65` 与 1-batch CPU smoke 通过，smoke 仅生成 `weights/best_raw.pt`。
- 实现提交 `3c65c9c` 已推送到 GitHub `main`。
- 本次不修改检测、计数、分类的 checkpoint 机制，也不修改 Backbone、Neck、OpenCLIP、Head、Loss 或数据协议。

## 2026-08-31 最新状态：Web 语义分割真实模型接入

- 新增 `HFSA-main/web_app.py` 和 `HFSA-main/web/`，使用 Python 标准库与原生 HTML/CSS/JavaScript 提供无需新增依赖的本地浏览器界面。
- 页面包含三个任务卡片、图像拖放/预览、任务动态文本输入、结果占位区和响应式布局；默认访问地址为 `http://127.0.0.1:7860`。
- 根据页面评审删除顶部操作步骤条，将任务选择改为适合继续扩展的紧凑自动流式布局；分析工作区在选择任务前完全隐藏。
- 用户展示名称按赛题统一为“场景分类”和“语义分割”；内部 `classification`、`refseg` 标识与模型边界不变。任务说明允许两行显示，不再被单行省略号截断。
- 用户界面品牌已改为项目正式名称“遥感图-文可解释轻量化多任务智能解译系统”；浏览器标题与 Hero 展示全称，左上角使用紧凑简称，页面不再向用户展示 HFSA。
- 新增 `tasks/routing/`，三任务配置可序列化为 JSON，并严格保留分类仅图像、计数图像+目标类别、指代分割图像+描述的不同契约。
- 新增 `tasks/routing/adapters/refseg.py::RefSegAdapter`：浏览器 data URL 解码后调用现有 `RefSegPredictor`，返回叠加图、二值 Mask、概率图和 JSON 摘要；支持 `HFSA_REFSEG_CHECKPOINT`、`HFSA_REFSEG_DEVICE` 覆盖。
- 默认路由只注册 RefSeg。页面启动和任务列表请求不会导入 `tasks.refseg.inference` 或加载 PyTorch/OpenCLIP；第一次真实分割请求才加载 checkpoint，实例随后缓存并在服务关闭时释放。
- RefSeg 结果区展示阈值、前景占比、模型耗时、原图尺寸，并提供三张 PNG 下载。分类和计数继续返回 `interface_pending`，不伪装成真实推理结果。
- RefSeg Adapter 新增确定性中文提示转译：覆盖 RRSIS-D 常见类别和位置、颜色、大小描述，结果区显示实际英文模型提示；复杂多目标/关系描述显式要求改用英文。
- 空 Mask 现在显示“未找到符合描述的目标”，不再把请求成功误写成语义分割成功。
- HTTP JSON 上限为 64 MiB，浏览器与 Adapter 限制单张原始图像不超过 40 MiB；推理异常统一返回 `inference_error`。
- 公网 IP、域名、HTTPS、鉴权和反向代理仍留到部署阶段处理。
- 本阶段不修改任何 Backbone、Neck、OpenCLIP、任务 Head、Loss、训练流程或 checkpoint 格式。
- RefSeg Web Adapter 定向测试 `6/6`、Web 路由定向测试 `8/8`、活动副本全库 `71/71`、发布仓库全库 `70/70`，`node --check web/app.js` 通过；活动副本多一项既有本地回归测试。
- 真实 HTTP smoke 使用 `03600.jpg` 和 `The gray small windmill`：严格加载 epoch-44 `best_raw.pt`，阈值 `0.70`，返回 `800×800` 结果、前景像素 `1976`、前景占比 `0.0030875`，与直接 Predictor smoke 一致。
- 中文真实 HTTP smoke 使用飞机图和“最上方的飞机”：自动转为 `the topmost airplane`，返回 1357 个前景像素，`found_target=true`。

## 2026-08-30 最新状态：指代分割单图推理边界

- 新增 `tasks/refseg/inference.py::RefSegPredictor`：严格加载完整 RefSeg checkpoint，读取其中的模型 YAML、输入尺寸、OpenCLIP 配置和 validation 最佳阈值。
- 单图推理输入固定为“遥感图像 + 非空指代表达”，在线文本编码使用与训练缓存一致的 OpenCLIP token features；输出恢复到原图尺寸的 probability、二值 mask 和叠加图。
- 新增 `RefSegPrediction`，可保存 `*_mask.png`、`*_probability.png`、`*_overlay.png` 和 JSON 元数据；支持 `python -m tasks.refseg.inference` 独立运行，也可被未来总路由直接导入。
- 总路由边界已由用户和师兄确认：用户手动选择任务，路由加载对应 checkpoint，再由任务声明自身输入；分类只需要图像，计数需要图像和目标文本，指代分割需要图像和指代表达。各任务输出不强制统一。
- 已删除旧 `train_semseg.py` 兼容入口；训练和测试正式入口保持 `train_refseg.py`、`test_refseg.py`。
- 本次不修改 Backbone、Neck、OpenCLIP、`TextPromptSegment`、Loss、训练循环、数据协议或 checkpoint tensor 格式。

## 2026-08-28 最新状态：测试预览扩展为前 N 个 batch

- 指代分割训练、每轮 validation、raw-best checkpoint 选择和训练后 test 协议保持不变。
- 新增 `--test-preview-batches`，Python 默认值与两份 RefSeg Shell 脚本的 `TEST_PREVIEW_BATCHES` 默认值均为 `5`。
- 独立测试会输出 `test_batch0_pred.jpg` 至实际可用的前 N 个 batch；`0` 或 `--no-preview` 禁用测试预览。
- `scripts/train_refseg.sh` 已移除自动 test，只执行 train 与每轮 validation；训练完成后通过 `scripts/test_refseg.sh` 单独加载最佳 checkpoint 测试。
- `test_results.json` 新增请求数量和实际生成文件列表，不参与指标、阈值或 checkpoint 选择。
- 本次没有修改 Backbone、Neck、OpenCLIP、`TextPromptSegment`、Loss、指标、数据划分或 checkpoint tensor 格式。

## 2026-08-26 最新状态：三个非检测任务统一进入 `tasks/`

- 指代分割、场景分类和目标计数现统一位于 `HFSA-main/tasks/refseg/`、`tasks/classification/`、`tasks/counting/`；原根目录 `classification/`、`counting/` 源码目录已移除。
- 三项任务均采用独立薄入口：`train_refseg.py`/`test_refseg.py`、`train_classification.py`/`test_classification.py`、`train_counting.py`/`test_counting.py`。Shell 脚本已调用这些入口。
- 旧 `train_semseg.py` 兼容入口已删除；实际指代分割实现位于 `tasks/refseg/engine.py`。
- 分割 checkpoint 保存、raw-best 选择、resume 协议、RNG 和 CSV 恢复包装为 `tasks/refseg/checkpoint.py::RefSegCheckpointManager`。分类仍由自身 Trainer 保存 Head checkpoint；计数仍复用 Ultralytics 检测 checkpoint。
- 目标检测 `train.py`、`val.py` 和 `text_encoder/` 本次没有修改，也没有建立 `tasks/detection/`。
- 发布副本与活动副本变更文件哈希一致；两边最终全量 CPU 回归均为 `51/51`。新入口真实 CPU smoke 已完成：分割 1 个 test batch、分类 1 个 test batch；计数因缺少真实数据和训练 checkpoint 仍以静态构建与专项回归为验证边界。
- `scripts/train_classification.sh` 现在是可直接启动的自包含正式训练脚本：默认 `data/VRSBench_scene` 不存在时，会先从 `data/VRSBench` 自动生成完整 ImageFolder 场景数据，随后以 YOLOv12m、batch 32、imgsz 640、20 epochs 开始训练；后续运行检测到数据已存在便跳过整理。

## 2026-08-26 最新状态：场景分类接入复核完成

- `srp_yolov12m_axis_clean_empty` 已在 epoch 39 早停，raw-best epoch 31、阈值 `0.80`；完整 cleaned test 为 `oIoU=0.693618`、`mIoU=0.539086`。
- 旧 epoch-44 checkpoint 在相同 3,480 条 cleaned test 的公平复评为 `oIoU=0.702938`、`mIoU=0.552721`，且五档 Pr、Precision、Recall、F1 全部高于新训练。因此保留空 mask 清洗规则，但不替换当前发布 checkpoint。
- checkpoint 文件锁恢复能力位于 `tasks/refseg/checkpoint.py`；活动副本与发布副本的分割恢复逻辑一致。
- 目标计数的 `CountingDetect`、任务包、训练/测试入口和脚本已确认存在于活动副本，不再只存在于发布副本。
- 已基于 `zhuoletian-collab/changjingfenlei@688c2a9` 完成 `SceneClassifyHead`、`SceneClassificationLoss`、m-scale YAML、类化数据/配置/训练/推理/评测和独立脚本接入；上游源码快照已从活动目录和发布仓库移除，仅由 ADR 保留 commit 溯源。
- 场景分类已完成独立复核：单图推理不再依赖数据集目录，`--split all` 可正确合并显式 train/val/test，训练 DataLoader 不会向含 `BatchNorm1d` 的 Head 送入末尾单样本 batch。
- 场景分类 8 项专项测试、源 checkpoint strict 兼容检查、单图 Top-K CLI、`SceneClassificationLoss` 的 1-train/1-val/1-test CPU smoke 均通过；活动副本与发布仓库完整 CPU 回归均为 47 项通过。真实完整场景分类训练尚未执行，GPU smoke 因 GPU 仍有其他负载未启动。

## 当前主线

- 当前个人负责的任务是 RRSIS-D 文本引导单目标二值分割：输入遥感图像与自由文本描述，输出对应目标的 `[B,1,H,W]` mask。
- 团队已确认多任务共用 YOLOv12m Backbone 和 Neck；当前分割分支使用匹配的 `yolov12m-semseg.yaml`、`yolov12m.pt`、P3/P4/P5 和 ADR-0015 `TextPromptSegment` 语义角色 token pooling Head。
- 未经用户确认，不修改 Backbone、Neck、OpenCLIP 或其他成员任务实现。
- 当前发布代码已包含空 mask 清洗与 checkpoint 恢复、目标计数和场景分类接入；指代分割正式实现统一位于 `tasks/refseg/`。活动目录和发布仓库均不再携带队友完整源码仓库。

## 已完成的分割任务封装

- `HFSA-main/scripts/train_refseg.sh` 是自包含训练脚本，直接调用 `train_refseg.py`，内置当前正式 YOLOv12m baseline 参数，不依赖部署时不会携带的 `run_semseg_preset.sh`。
- 训练脚本通过自身位置定位 `HFSA-main`；数据、模型、权重和输出均使用项目相对路径，并允许通过环境变量或末尾 CLI 参数覆盖。
- `HFSA-main/scripts/test_refseg.sh` 支持已有 checkpoint 的独立测试，默认输出到单独目录，不重新训练，也不覆盖原训练目录。
- 正式独立测试入口为 `test_refseg.py`；它只构建 test split，严格加载完整 checkpoint，并复用 checkpoint 中冻结的 validation 阈值与选模指标。
- 该封装通过 Shell 语法检查、参数展开检查、Python 编译、全库测试和 2-batch CUDA evaluation-only smoke。

## 当前数据协议

- ADR-0019 已接受显式空 mask 清洗：默认 `empty_mask_policy=drop`，剔除 `train_22187`、`train_20203` 和 `test_413`。
- 当前活动 split 为 `train=12179`、`val=1740`、`test=3480`；旧 `12181/1740/3481` 协议只能通过 `--empty-mask-policy keep` 复现。
- 不允许用 bbox 为零前景标注补矩形伪 mask。最近邻 mask resize、axis-aware 翻转和 0.15 色彩扰动保持不变。

## 当前实验结果

- 历史 YOLOv12m 完整运行：`runs/semseg/srp_yolov12m_axis`，raw-best epoch 44，冻结阈值 `0.70`，旧 3481-sample test 上 `oIoU=0.701171`、`mIoU=0.552562`、`Pr@0.5-0.9=0.623959/0.540362/0.425452/0.319161/0.164321`。
- 该结果早于空 mask 清洗，只作为历史基线；不能直接与 cleaned 3480-sample test 的新结果比较。
- cleaned 协议完整 seed-42 训练位于 `runs/semseg/srp_yolov12m_axis_clean_empty`：epoch 39 早停，raw-best epoch 31、阈值 `0.80`，3,480 条 test 上 `oIoU=0.693618`、`mIoU=0.539086`。
- 旧 epoch-44 checkpoint 在相同 cleaned test 上为 `oIoU=0.702938`、`mIoU=0.552721`，因此数据清洗规则保留，但新训练 checkpoint 不替换当前发布候选。

## 多任务训练整合结论

- 不把检测、指代分割、计数、分类等任务的数据加载、Loss、训练循环和评测逻辑强行合并到一个巨型 `train.py`。
- 每个任务保留自己的 Trainer/训练文件和任务脚本，例如检测使用现有 `train.py`，分割使用 `train_refseg.py`，其他任务按相同规范提供独立入口。
- 如果最终需要统一训练命令，公共 `train.py` 只能作为薄分发器：解析 `--task` 后调用对应任务 Trainer，不在分发器中实现具体数据、Loss 或指标逻辑。
- 最终“输入 1/2/3/4/5 或任务名称后切换任务”属于统一推理入口，与训练脚本分开设计；训练脚本不承担在线任务切换。
- 该决策记录在 ADR-0020。

## 其他成员代码状态

- 队友目标计数代码已完成任务化接入：新增不改变 `Detect` 行为的 `CountingDetect` Head、`yolov12m-counting.yaml`、`tasks/counting/` 类化任务包、薄训练/测试入口和自包含脚本。
- 计数仍采用原有“文本引导类无关检测 -> NMS -> 检测框数”逻辑，不引入密度图、计数回归 Head 或新 Loss；共享 YOLOv12m Backbone/Neck、OpenCLIP 和文本引导检测 Trainer。
- 计数评测当前明确保持队友的 positive-query VOC 协议，只查询 XML 中实际存在的类别，报告 EM、MAE、RMSE 与逐类别统计；不得把它表述为包含零计数问答的完整协议。
- 队友仓库未提供训练 checkpoint，本地也未完成 VRSBench 真实数据 smoke；当前验证范围为语法、参数展开、Head 等价性、模型构建和预训练权重静态加载。
- 队友场景分类已接入 `SceneClassifyHead`、`SceneClassificationLoss`、m-scale YAML、类化配置/数据/训练/推理/评测、数据准备工具及独立 Python/Shell 入口；算法继续使用 P3/P4/P5 空间注意力 + GeM、单标签 CrossEntropy 和冻结 Backbone/Neck。
- 场景分类已完成 CPU 链路 smoke 与 checkpoint 严格重载，但尚未完成真实完整数据训练和空闲 GPU smoke；smoke 指标仅用于验证链路。

## 下一步

1. 获取计数队友的 VRSBench 数据路径和训练 checkpoint，运行最小训练 smoke 与 1-2 张图的独立计数评测，确认 checkpoint 严格加载和真实 EM/MAE/RMSE 输出。
2. 使用真实场景分类数据运行完整训练与独立 test；GPU 空闲时先执行最小 CUDA smoke，并记录参数、显存、耗时和 checkpoint 大小。
3. 场景分类正式 checkpoint 和单图 Predictor 验证完成后，按 RefSeg 相同边界接入独立 Classification Adapter；计数 checkpoint 可用后再接入 Counting Adapter。
4. 三个任务稳定后处理公网 IP、域名、HTTPS、鉴权和反向代理，并进行 A5000 同条件资源测评；当前不要提前合并为联合多数据集训练。

## 部署注意

- 分割训练包不需要携带 `run_semseg_preset.sh`；`scripts/train_refseg.sh` 已包含所需正式参数。
- 运行前仍需提供项目源码、RRSIS-D 数据与缓存、`pretrain_model/yolov12m.pt`，并激活具备 PyTorch、OpenCLIP、OpenCV 等依赖的环境。
- 计数训练还需提供 VOC 风格 VRSBench 数据；独立测试需提供计数 `best.pt`，默认路径均可通过脚本环境变量覆盖。
- 场景分类正常训练只需在已激活项目环境后执行 `bash scripts/train_classification.sh`；首次运行会自动从 VOC VRSBench 生成 ImageFolder 数据，在 `/mnt` 盘扫描 XML 可能耗时数分钟。独立测试需要分类 Head checkpoint；单图 `--image` 推理不要求数据集目录存在。
- 默认训练输出目录已有历史结果时，应通过 `SAVE_DIR` 指定新目录，避免覆盖旧实验。
