# PROJECT_RULES.md

本文件是 HFSA 项目的唯一权威开发规范（Single Source of Truth）。所有后续开发、重构、测试、文档、Git 提交和架构决策都必须以本文件为准。

规则维护原则：本文件长期维护。后续规则变更只能在“规则变更记录”中追加新的条目，并在相关章节补充“追加说明”；不得删除或覆盖既有历史原则。若本文件与其他文档冲突，以本文件为准，并通过 ADR 记录冲突原因和处理方式。

## 项目目标

### 项目定位

HFSA 面向“太空智算背景下的多模态遥感大模型应用探索”。项目目标是在资源受限或平台约束明确的环境中，围绕遥感图像理解任务构建、训练、验证和交付可复现的多模态模型能力。

### 核心功能

项目当前代码显示两条主要技术路线：

1. 文本引导目标检测：基于 Ultralytics/YOLOv12 训练框架，引入 OpenCLIP 文本嵌入、短语级特征、空间关系、依赖关系、类无关检测头和严格 grounding 验证。
2. 文本引导语义/指代表达分割：基于 YOLOv12 backbone/neck 和自定义 `TextPromptSegment` 语义分割头，在 RRSIS-D referring segmentation 数据上实现“图像 + 文本提示 -> 二值掩膜”训练与验证。
3. 数据集准备与缓存：支持 VOC 风格检测数据准备、RRSIS-D 指代表达分割元数据生成、OpenCLIP 文本嵌入预计算、YOLO data.yaml 生成与校验。
4. 实验输出：训练指标、验证指标、checkpoint、TensorBoard 日志、预览图、混淆矩阵和结果 CSV。

### 技术栈

当前项目技术栈以 Python 深度学习为主：

- Python
- PyTorch / torchvision
- Ultralytics YOLO
- YOLOv12 配置与自定义模型模块
- OpenCLIP / open-clip-torch
- NumPy / SciPy / pandas
- OpenCV / Pillow
- PyYAML
- spaCy / ftfy，用于文本和短语处理
- TensorBoard，用于训练日志

### 主要目录

- `HFSA-main/train.py`：文本引导目标检测训练入口。
- `HFSA-main/val.py`：文本引导目标检测验证和可视化入口。
- `HFSA-main/train_semseg.py`：文本引导语义/指代表达分割训练入口。
- `HFSA-main/dataset/`：VOC、RRSIS-D 数据准备、文本嵌入预计算、数据集校验。
- `HFSA-main/text_encoder/`：文本引导检测的训练器、验证器、模型、损失、匹配、短语分类、空间关系增强和嵌入读取。
- `HFSA-main/ultralytics/`：本项目随仓库携带的 Ultralytics 代码和自定义 YOLOv12/语义分割配置。
- `HFSA-main/pre_datasets/`：预处理数据和文本嵌入缓存。
- `HFSA-main/data/`：本地数据集目录，不应作为普通源码变更处理。
- `HFSA-main/runs/`：训练输出目录，不应作为普通源码变更处理。
- `hfsa_env/`：本地虚拟环境，不属于项目源码。

### 项目大纲来源

项目大纲当前来自根目录 PDF：`XH-202603_面向太空智算的多模态遥感大模型应用探索(2).pdf`。该文件说明项目关注多模态遥感大模型、自然语言指令驱动的遥感图像理解、开源数据集验证、模型/源码/说明文档提交，以及性能、资源效率和工程适配。

## 开发原则

1. 正确性优先：所有模型、数据处理、指标计算和评估逻辑必须先保证语义正确，再考虑速度和代码简洁。
2. 可维护性优先：新增功能必须有清晰边界，避免把训练、数据准备、模型定义、评估和可视化继续混在单个巨型脚本中。
3. 小步迭代：一次只完成一个明确目标，每一步都能独立验证。
4. 单模块开发：同一开发周期只能修改一个主模块，除非为了使该模块可运行必须触及少量相邻接口。
5. 禁止 Vibe Coding：不得凭感觉大面积生成代码。开始实现前必须先读现有实现、写清楚设计、确认影响范围。
6. 禁止一次生成整个项目：本项目已有代码和实验产物，后续只能增量演进。
7. 禁止跳步开发：不得绕过分析、设计、测试、文档更新和确认流程。
8. 实验可复现：任何训练/验证相关改动必须记录命令、数据路径、权重路径、随机种子、关键超参和环境差异。
9. 保护数据与权重：不得随意改动、删除或提交大体积数据集、模型权重、训练输出和虚拟环境。
10. 本地优先：优先使用仓库内已有 Ultralytics 分支和本地工具链，不轻易引入新的框架或远程依赖。

## 开发流程

所有开发必须严格执行以下阶段：

1. 分析：阅读本文件、项目大纲、架构文档、开发日志、ADR 和相关源码；明确需求、输入输出、影响范围、风险和验收标准。
2. 设计：写出模块边界、数据流、接口、异常处理、测试方案和文档更新点；必要时先提交 ADR 草案。
3. 实现：只实现已确认设计中的内容；禁止顺手重构无关代码。
4. 测试：执行与变更范围匹配的最小有效测试；训练类改动至少需要 smoke test 或 dry-run 级验证；指标类改动必须有可重复输入。
5. 文档更新：同步更新 README、ARCHITECTURE、DEVELOPMENT_LOG、ADR 或模块说明。
6. Git 提交：生成清晰 commit message；提交前检查 diff，排除数据、权重、缓存和环境文件。
7. 等待确认：完成后输出变更摘要、测试结果、风险和下一步建议，等待用户确认后再进入下一模块。

任何阶段缺失都视为未完成。若用户要求先评审或先设计，必须停止在对应阶段，不得提前写代码。

## 开发前置阅读规则

每次开始开发前，必须按顺序阅读：

1. 项目大纲：当前为根目录 `XH-202603_面向太空智算的多模态遥感大模型应用探索(2).pdf`，若后续有 Markdown 版大纲，以最新登记的大纲为准。
2. `PROJECT_RULES.md`。
3. `ARCHITECTURE.md`。
4. `DEVELOPMENT_LOG.md`。
5. `ADR/*`。

如果 `ARCHITECTURE.md`、`DEVELOPMENT_LOG.md` 或 `ADR/` 不存在，第一次相关开发前必须先创建最小版本，记录当前事实和缺失项。创建这些文档仍属于文档建设，不等同于业务开发。

阅读完成后，不得立即写业务代码。必须先进行 Architecture Review，并输出：

- 项目理解
- 核心模块
- 模块依赖关系
- 潜在风险
- 建议优化项
- 推荐开发顺序
- 是否发现架构问题

只有在用户确认后，才能进入实现阶段。

## 模块开发规则

1. 一次只能开发一个模块。
2. 模块完成后才能进入下一模块。
3. 禁止同时开发多个模块。
4. 模块边界必须在设计阶段写清楚。
5. 若一个需求跨模块，必须拆成多个顺序任务。
6. 公共接口变更必须优先评估调用方，而不是直接改实现。

### 模块完成标准

一个模块只有同时满足以下条件才算完成：

1. 功能完成。
2. 测试通过。
3. README 或对应模块文档更新。
4. `DEVELOPMENT_LOG.md` 更新。
5. `ARCHITECTURE.md` 更新。
6. Commit Message 已生成。
7. 若涉及架构取舍，ADR 已追加。
8. 若涉及训练结果，记录命令、数据、权重、指标和输出目录。

## 架构规范

### 当前核心模块

1. 数据准备层：`dataset/voc_object_dataset.py`、`dataset/rrsisd_refseg_dataset.py`、`dataset/precompute_text_embeddings.py`、`dataset/utils.py`。
2. 文本引导检测层：`text_encoder/trainer.py`、`text_encoder/validator.py`、`text_encoder/model.py`、`text_encoder/losses.py`、`text_encoder/embedding_store.py`、`text_encoder/matching.py`。
3. 文本与空间增强层：`text_encoder/phrase_classifier.py`、`text_encoder/spatial_embedding.py`、`text_encoder/fusion_blocks.py`。
4. 训练/验证入口层：`train.py`、`val.py`、`train_semseg.py`。
5. 模型配置层：`ultralytics/cfg/models/v12/*.yaml` 和 `ultralytics/nn/modules/*` 中的自定义模块。
6. 实验产物层：`pre_datasets/`、`runs/`、checkpoint、embedding cache、结果图表。

### 分层原则

- 入口脚本只负责编排参数、路径、训练/验证流程，不承载复杂业务逻辑。
- 数据集模块只负责数据解析、校验、转换和加载，不直接依赖训练器内部状态。
- 文本嵌入模块只负责文本编码、缓存、读取和 batch 对齐，不混入模型损失计算。
- 模型模块只负责 forward、head、融合结构和损失所需输出，不处理文件路径。
- 验证模块只负责推理、后处理、指标和可视化，不隐式修改训练配置。
- 通用工具只能放真正跨模块复用且无领域状态的函数。

### 依赖方向

允许依赖方向：

- 入口脚本 -> 数据准备层 / 文本引导层 / 模型配置层。
- 文本引导训练器/验证器 -> Ultralytics 基类。
- 文本引导模型 -> Ultralytics DetectionModel 和自定义融合/损失模块。
- 数据准备层 -> 标准库、PyTorch、OpenCLIP、OpenCV/Pillow、YAML。

禁止依赖方向：

- 数据准备层反向依赖训练入口。
- 模型层读取硬编码数据集路径。
- 验证逻辑修改数据准备逻辑。
- 工具模块依赖具体实验输出目录。
- 第三方 vendor 代码中散落项目业务逻辑而没有文档标记。

### 第三方代码修改规则

`HFSA-main/ultralytics/` 是随项目携带的第三方/本地修改代码。修改前必须：

1. 判断能否通过外部 adapter、子类或配置解决。
2. 若必须修改，记录修改点、原因、上游版本假设和回滚方式。
3. 在 `ARCHITECTURE.md` 中列出所有 fork patch。
4. 不得大面积格式化第三方文件。

## 代码规范

1. Python 代码遵循类型清晰、函数单一职责、异常显式、路径用 `pathlib.Path` 的风格。
2. 新增函数必须有明确输入输出；训练/评估关键函数应补充类型标注。
3. 避免复制粘贴解析逻辑。已存在的 `_parse_phrase_types`、`_parse_phrase_weight_string` 等重复函数，后续应逐步收敛到单一来源。
4. 禁止硬编码不可迁移的绝对路径。确有需要必须提供 CLI 参数或配置项。
5. 训练入口参数默认值必须适合 smoke test 或明确说明不是生产默认。
6. 数据加载必须检查文件存在、split 合法、embedding 维度匹配、sample_id 对齐。
7. 文本 embedding 的 shape、dtype、归一化策略和模型名必须记录在 payload metadata 中。
8. 指标计算必须说明阈值、NMS 参数、是否 class-agnostic、是否 strict grounding。
9. 日志输出必须足以复现实验，不得只输出“成功/失败”。
10. 新增依赖必须先评估是否已在 `requirements.txt` 中存在，并说明必要性。

## 测试规范

### 测试层级

1. 单元测试：文本解析、短语分类、RLE mask 解码、bbox 归一化、指标计算、路径解析。
2. 集成测试：数据准备 -> data.yaml -> dataset loader -> batch 样本；embedding cache -> trainer/validator batch 注入。
3. Smoke test：最小 batch、最小 epoch、CPU 或单 GPU 可运行命令。
4. 回归测试：固定小样本输入，验证指标、输出 shape 和错误处理不退化。

### 训练相关测试要求

- 训练循环改动必须至少跑 `--epochs 1`、小 `--batch`、小 `--imgsz`、有限 batch 的 smoke test。
- 分割训练改动优先使用 `train_semseg.py` 的 `--max-batches` 和 `--max-val-batches` 控制成本。
- 验证逻辑改动必须记录 `conf`、`iou`、`max_det`、`single_cls`、`agnostic_nms`。
- 不能运行完整训练时，必须说明原因，并提供可执行的最小验证命令。

### 数据测试要求

- 不得假设数据集完整。所有数据准备脚本都要对缺失图像、缺失标注、缺失 embedding 给出明确错误。
- RRSIS-D RLE 解码必须使用小型人工样本测试边界条件。
- VOC XML 解析必须覆盖缺失字段、越界 bbox、空对象和非标准命名。

## 文档规范

### 必备文档

项目根目录必须逐步维护：

- `PROJECT_RULES.md`：唯一权威规则。
- `ARCHITECTURE.md`：当前架构、模块边界、数据流、已知技术债、第三方 patch。
- `DEVELOPMENT_LOG.md`：按日期追加开发记录、测试结果、命令和风险。
- `ADR/`：架构决策记录目录。
- `README.md`：项目安装、数据准备、训练、验证和交付说明。

### 文档更新要求

- 每个模块完成必须更新对应文档。
- 文档只记录事实、命令、结果和决策，不写空泛总结。
- 实验结果必须包含日期、commit 或工作树状态、命令、数据路径、权重路径、关键超参、指标、输出目录。
- 若无法复现，必须标记为“未复现”并说明原因。

## Git 规范

1. 开发前必须检查工作树状态。
2. 不得回退用户已有改动。
3. 不得提交 `hfsa_env/`、`runs/`、大规模 `data/`、`*.pt`、缓存、`__pycache__/`。
4. 每次提交只包含一个逻辑变更。
5. Commit message 使用以下格式：

```text
<type>(<scope>): <summary>

Context:
- ...

Changes:
- ...

Tests:
- ...

Docs:
- ...
```

常用 type：

- `docs`
- `feat`
- `fix`
- `refactor`
- `test`
- `perf`
- `chore`
- `adr`

6. 如果仓库未正确初始化或 Git 状态不可用，必须在最终说明中明确记录，不得假装已提交。

## ADR 规范

### 何时必须写 ADR

以下情况必须新增 ADR：

- 选择或更换模型主干、文本编码器、融合结构、损失函数。
- 修改评估协议、核心指标或数据划分。
- 修改 `ultralytics/` 内第三方代码。
- 引入新数据集、新依赖、新训练范式。
- 改变项目目录结构或模块边界。
- 接受重要技术债或临时方案。

### ADR 命名

文件命名：

```text
ADR/0001-short-title.md
ADR/0002-short-title.md
```

### ADR 模板

```markdown
# ADR-0001: 标题

## 状态

Proposed / Accepted / Superseded

## 背景

说明问题、约束和上下文。

## 决策

说明最终选择。

## 备选方案

列出至少一个备选方案。

## 影响

说明正面影响、负面影响、迁移成本和测试要求。

## 关联

关联 issue、commit、实验、文档或代码路径。
```

## 重构规范

1. 重构必须有明确目标：降低重复、分离职责、改善可测试性或移除已确认技术债。
2. 禁止把功能开发和大规模重构混在同一个提交。
3. 重构前必须记录当前行为和最小回归测试。
4. 重构后必须证明外部行为不变，除非设计阶段明确说明行为变更。
5. 对训练脚本的重构必须保留旧命令兼容性，或提供迁移说明。
6. 对数据格式的重构必须提供转换脚本或兼容读取逻辑。

## 专项规范：多模态遥感与文本引导任务

### 数据集规范

- 所有数据集必须通过配置或 CLI 参数定位，不允许写死本机路径。
- 数据准备输出必须包含 `data.yaml`、split 文件、类别名、数据来源、忽略标签和样本计数。
- RRSIS-D 指代表达分割必须保留 `id`、`text`、`class_idx`、`segmentation`、`bbox`、`image`、`ann_id`、`ref_id`。
- VOC 风格检测必须保留 image/xml/label 的映射关系，sample_id 必须稳定。
- 数据转换脚本必须可重复运行，并支持检测已有缓存是否可复用。

### 文本嵌入规范

- 文本编码器当前默认 OpenCLIP `ViT-L-14` / `openai`，默认 embedding dim 为 768。
- embedding cache 文件必须包含 ids、texts 或短语文本、embeddings、model metadata。
- 训练和验证必须检查 sample_id 与 embedding ids 对齐。
- 如果缺少 embedding，不得静默退化为图像-only 模型；必须报错或明确 warning。
- 短语类型、短语权重、依赖关系、空间关系等配置必须可追踪。

### 模型与训练规范

- 检测任务当前采用 class-agnostic 文本引导 YOLO 方向；修改类别策略必须写 ADR。
- 分割任务当前采用 `TextPromptSegment` 头；修改 text_dim、head 输入层或类别数必须同步模型 YAML、训练脚本和验证逻辑。
- 任何新增 loss 必须说明权重、输入 tensor、目标 tensor、数值稳定性和关闭方式。
- 任何增强策略变更必须说明是否破坏文本和目标框/掩膜的对齐关系。
- 模型权重加载必须报告匹配 tensor 数、跳过层和 head 初始化状态。

### 评估规范

- 遥感多模态任务至少区分检测、分类、分割、变化检测、图像描述/问答等任务类型，不得混用指标。
- 检测指标必须记录 mAP、precision、recall、NMS 参数和 grounding 规则。
- 分割指标必须记录 pixel_acc、mIoU、target IoU、precision、recall、F1、threshold。
- 资源效率必须记录模型大小、参数量、显存、推理速度或训练吞吐，条件允许时纳入对比。
- 评估结果不得只报最好值，必须记录可复现命令和数据 split。

### 实验产物规范

- `runs/`、checkpoint、embedding cache、预览图和结果 CSV 默认为实验产物，不作为常规源码提交。
- 值得保留的实验结果必须整理成文档表格，并记录输出目录。
- 大体积模型和数据应通过外部存储或明确路径管理，不直接混入 Git。

## 安全与资源规范

1. 不得执行会删除数据、权重、环境或训练输出的命令，除非用户明确要求并确认。
2. 网络下载依赖、模型或数据前必须说明来源、大小和必要性。
3. GPU 训练命令必须明确设备、batch、imgsz、epoch、输出目录。
4. 长时间训练前必须先做短 smoke test。
5. 不得把私有路径、账号、令牌或邮箱等敏感信息写入代码。

## 当前已知问题

1. 根目录和 `HFSA-main` 内均显示 `.git` 目录项，但 `git status` 当前未能识别为 Git 仓库；后续 Git 操作前必须先诊断仓库状态。
2. 当前未发现根目录 `README.md`、`ARCHITECTURE.md`、`DEVELOPMENT_LOG.md`、`ADR/`，后续开发前需要补齐。
3. `hfsa_env/`、`data/`、`runs/`、`pretrain_model/`、`*.pt` 等大体积或环境内容混在项目目录中，存在误扫描、误提交和上下文污染风险。
4. 训练入口与部分工具函数存在重复解析逻辑，后续应收敛到单一工具模块。
5. `train_semseg.py` 文件承担参数、数据、训练、验证、绘图、checkpoint 等多种职责，后续应按模块逐步拆分。
6. PDF 项目大纲当前文本层存在编码提取问题，建议后续维护 Markdown 版项目大纲。

## 推荐开发顺序

1. 文档底座：补齐 `ARCHITECTURE.md`、`DEVELOPMENT_LOG.md`、`ADR/0001-project-baseline.md`、`README.md`。
2. 仓库卫生：确认 Git 仓库状态，建立 `.gitignore`，排除环境、数据、权重、缓存和训练输出。
3. 最小可复现路径：整理检测和分割各自的最小 smoke test 命令。
4. 数据集校验：完善 VOC/RRSIS-D 数据校验和小样本测试。
5. 文本嵌入缓存：统一 embedding metadata、sample_id 对齐检查和错误提示。
6. 训练脚本拆分：优先拆分 `train_semseg.py` 中的数据、指标、绘图、checkpoint 逻辑。
7. 模型与评估增强：在稳定测试和文档基础上再改 loss、fusion、head 或 strict grounding。

## 规则变更记录

### 2026-07-12

- 创建本文件，确立 `PROJECT_RULES.md` 为项目唯一权威开发规范。
- 记录当前项目目标、模块边界、开发流程、文档/Git/ADR/重构规则，以及多模态遥感专项规范。
- 明确开始开发前必须完成 Architecture Review，并等待用户确认。

## 追加说明：跨线程上下文补充

本节根据 2026-07-12 读取到的其他 HFSA 项目线程追加。后续开发必须继承这些上下文；若与旧章节有冲突，以本节更新的项目事实为准，并在 ADR 中说明原因。

### 已读取线程

- `019f17fe-5f3e-7092-999b-bde363f3bd63`：修改语义分割检测头。
- `019f2d61-3ac8-7930-ad71-a261fb68f3d0`：分析文本条件分割头方案。
- `019f3bed-e61b-77e2-a8cb-405c068d542f`：梳理语义分割逻辑。
- `019f3694-61d8-7813-a7a7-9588d4ebe524`：语义分割运行结果解释。
- `019f411d-8c44-7b73-8b0d-47e0abb46031`：验证 loss 不稳定讨论。
- `019ec116-9a17-7130-856a-1cc439d83fee`：项目说明命名。

### 当前主线修正

1. 当前个人负责方向是语义分割。比赛整体还包括目标计数、场景分类、遥感图像描述等任务，但本项目当前开发不要把其他成员任务混入语义分割主线。
2. 语义分割主线已经从早期 LoveDA 类别级语义分割，收敛为 RRSIS-D 指代表达分割：输入一张遥感图像和一条自由文本描述，输出该描述对应目标的单实例二值 mask。
3. 第一版目标是先跑通“图像 + 自由文本描述 -> 单目标二值 mask”闭环，不急着重写分割头。
4. 当前对外项目名称可使用“遥感影像-文本智能多任务解译系统”；若用于简历或材料，可根据语境压缩为“多模态遥感影像智能解译系统”。

### 不可随意修改的边界

1. 默认不要修改 backbone、neck、Ultralytics 主体代码或 YOLOv12 主干配置。
2. 除非用户明确确认，不要修改 `TextPromptSegment` 结构；当前先复用已有 head。
3. 必要改动应集中在数据集适配、训练入口路由、路径兼容、日志、文档和测试。
4. 如果后续必须改 head，应先给出设计评审，再考虑 TextConditionedMaskHead、CLIPSeg 或 LAVT 式 token-level cross-attention。

### RRSIS-D 数据事实

1. RRSIS-D 当前主标注来源是 `refs(unc).p + instances.json + JPEGImages`。
2. `ann_split/*.xml` 只作为人工核对来源，不作为训练主来源。
3. 当前元数据目标目录是 `HFSA-main/pre_datasets/RRSIS-D_refseg`。
4. 标准 split 数量为 `train=12181`、`val=1740`、`test=3481`。若数量不一致，必须先排查数据版本、路径和过滤逻辑。
5. RRSIS-D mask 是单表达对应的二值 mask，目标区域为 `1`，背景为 `0`；第一版不做多目标合并，也不做类别级语义分割。

### 文本嵌入事实

1. RRSIS-D 自由文本描述必须预计算或缓存 OpenCLIP 文本向量，避免每个 batch 重复编码。
2. 默认使用 OpenCLIP `ViT-L-14/openai`，因为当前 `TextPromptSegment` 假设 `text_dim=768`。
3. 默认缓存目录采用 `openclip_ViT-L-14_openai` 命名风格，缓存文件为 `{split}_text_embeddings.pt`。
4. 更换文本编码器、文本维度或缓存格式必须同步模型 head、缓存生成、训练入口和 ADR。

### 路径与环境事实

1. 该项目常在 WSL 中运行，路径可能同时出现 Windows 形式 `D:\...` 和 WSL 形式 `/mnt/d/...`。
2. 若 JSONL 或 YAML 中出现 Windows 绝对路径，WSL 运行时必须能转换到 `/mnt/d/...`，或退回到 `data.yaml` 所在目录解析 split 文件。
3. 当前 Windows 侧默认 Python 可能缺少 `torch`、`open_clip_torch`、`numpy`、`Pillow/OpenCV` 等依赖；不能因此声称训练 smoke test 已完成。
4. `hfsa_env` 是 Linux/WSL 风格虚拟环境，不能假设能在 Windows PowerShell 中直接运行。

### 训练与评估事实

1. 单目标文本引导分割的输出应是 `[B,1,H,W]`，使用二值 mask loss，如 BCE + Dice；不得与多类 `[B,C,H,W]` 语义分割指标混用。
2. `results.csv` 当前按 epoch 结束后追加，不是按 batch 实时写入。训练中途看不到新行不一定是错误，必须先确认完整 epoch 是否结束。
3. 历史 LoveDA 完整训练曾出现每 epoch 约 2000 秒的运行成本，后续长训练前必须先做 smoke test。
4. 分析 val loss 波动时，必须同时检查 batch size、学习率调度、是否跑完整验证集、`max_val_batches` 是否为 0、阈值/NMS/后处理是否一致。
5. 方向词和空间关系是当前已知难点。仅靠后处理很难判断“上方、左侧、靠近、远离”等带方位关系的地物；后续改进应优先考虑文本 token、空间关系建模和图文特征交互。
6. 早期线程中关于参数是否存在的结论可能已经过期。每次引用历史命令前，必须以当前 `parse_args()` 和源码为准复核。例如早期曾认为 `--patience` 未实现，但当前源码已有 `--patience` 和 `--min-delta`。

### 推荐 smoke test 命令

WSL：

```bash
cd /mnt/d/code/python/HFSA/HFSA-main
python train_semseg.py \
  --data pre_datasets/RRSIS-D_refseg/data.yaml \
  --text-queries \
  --text-encoder openclip \
  --epochs 1 \
  --batch 2 \
  --imgsz 128 \
  --max-batches 2 \
  --max-val-batches 2
```

Windows PowerShell：

```powershell
cd D:\code\python\HFSA\HFSA-main
python train_semseg.py --data pre_datasets/RRSIS-D_refseg/data.yaml --text-queries --text-encoder openclip --epochs 1 --batch 2 --imgsz 128 --max-batches 2 --max-val-batches 2
```

运行前必须确认当前环境具备项目依赖。若缺少 PyTorch 或 OpenCLIP，只能记录为“未完成训练 smoke test”。

### 后续文档要求

1. 后续新线程如果产生架构、数据、训练或评估结论，必须追加到本节或拆入 `ARCHITECTURE.md` / `DEVELOPMENT_LOG.md` / ADR。
2. 当前任务主线已经从 LoveDA 类别 prompt 迁移到 RRSIS-D 自由文本指代表达分割。若发现代码、文档或默认命令仍暗示 LoveDA 是主线，需要标记为历史遗留并评估是否清理。
3. 历史线程只能作为上下文来源，不能替代源码复核和当前实验验证。

### 2026-07-12 跨线程补充

- 追加其他 HFSA 线程中的有效上下文：RRSIS-D 文本引导单目标二值分割为当前主线，LoveDA 为历史探索。
- 追加边界：默认不改 backbone/neck/Ultralytics 主体，不改 `TextPromptSegment`，先完成数据接入和训练闭环。
- 追加数据、文本嵌入、WSL 路径、训练结果写入时机、方向词/空间关系风险、smoke test 命令等规则。

### 2026-07-12 职责边界补充

- 项目主要代码基础由师兄已经完成，后续队员主要负责各自分支功能的实现、接入、实验和说明，不应把整个主项目重新设计或重写。
- 当前个人负责的分支功能是语义分割。语义分割相关的设计、接入和新增实现属于当前负责范围，包括已有代码中为语义分割新增的类、函数、调用链、训练/验证入口适配，以及单独新增的语义分割代码文件。
- 师兄明确交代的技术边界是：可以围绕网络检测头/分割头做任务适配，但不要动骨干网络。后续任何涉及 backbone、neck、Ultralytics 主体或 YOLOv12 主干配置的修改，都必须先暂停并说明原因，等待用户确认。
- 判断代码归属时，以功能边界为准：语义分割 dataset、mask 解码、文本引导 mask head 接入、二值 mask loss/metric、语义分割训练与验证流程，均归入语义分割分支；通用检测主干、基础 YOLO 框架、非语义分割任务逻辑，不应主动改动。
- 对外描述项目贡献时，应区分“项目主框架/基础代码由师兄完成”和“本人负责语义分割分支设计与实现”，避免把团队已有主框架说成个人从零完成，也避免遗漏本人在语义分割新增类、函数、调用和独立文件上的贡献。
