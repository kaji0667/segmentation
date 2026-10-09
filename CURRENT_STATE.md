# CURRENT_STATE.md

## 2026-10-09 最新约束：当天smoke总额度8次，先本地验证；私有重放已就绪

- 用户明确提醒今天smoke总额度只有8次，要求先确定再使用；按用户当前约束安排，不以旧协议写的10次替代，也不推断账户剩余额度。撤回上一阶段“仅为拿原因码立即再跑smoke”的建议；本轮没有提交任何官网run或消耗smoke额度。
- 目前仍只有首次官方audit的7项500与1项有效但答错记录，没有那7项原始题干；哈希不能恢复题干。不能确认7题已修复，也不能把34项合成测试当作官方8题全部可答。四个已有单图模型仍未支持双图变化、通用VQA等能力。
- 在同一HFSA Adapter增加默认关闭的HFSA_API_REPLAY_DIR：保存收到的协议请求白名单到项目外本机私有目录，覆盖路由/推理失败请求、按内容哈希去重；不保存认证头、未知字段、模型答案或金标，不把题干放进诊断日志/Git，也不按题号查表猜答案。保存错误不影响推理。
- replay_requests.py只对loopback顺序调用/v1/predict，拒绝公网端点、关闭代理及重定向，核对回显及答案约束，不创建官网run或评估没有金标的准确率。34/34测试通过；真实当前API的独立合成分类请求200、未知问法500均已私有保存，二者本地重放及去重一致，文件未含当前Key。实际本机及ngrok健康检查均200/ready；这不是官网新成绩。
- API后台已启用私有捕获（本轮WSL PID1430），原Key/端口/图片包/四模型配置保留。日志 `%LOCALAPPDATA%/HFSA/PublicAPI/hfsa-api-replay-20261009-122139.log`；请求目录 `%LOCALAPPDATA%/HFSA/PublicAPI/requests-20261009/`（WSL为/mnt/c/Users/cyh/AppData/Local/HFSA/PublicAPI/requests-20261009）。目前只有2条TEST_REPLAY_*合成请求，不含原官方8题；路径外数据不上传Git。
- 下一步先保留今天额度：有真实请求后可无限次本地重现，不需要每改一次就跑官网。若仍必须使用一次官网提交来获取缺失的实际请求，应先向用户说明目的与现有能力缺口，由用户决定；当前不催促立即重跑，也不声称下一轮会全部成功。

## 2026-10-09 最新状态：官网8题smoke已完成，7题推理500；已补诊断

- 用户提供官网下载 audit 与 predictions：8题完成，7题为HTTP500，只有1题HTTP200并返回字符串 `"1"`；该有效答案评分为0，missing=7，malformed=0。结果是开发集连通性smoke，不能称正式客观成绩，也不能将7个缺失答案说成7个模型正常作答但错误。原下载文件留在用户下载目录，不复制到Git。
- 本次公网预检已通过，原密钥排错阶段已被用户实际评测覆盖。7项500来自模型适配/推理异常；官方server将异常统一转换为inference_failed，audit没有题干或原始异常，暂不能逐项断言是哪个分支拒绝。
- 已在HFSAAdapter加入固定原因码的本地JSON诊断：题号/请求编号、约束类型、图片数、所选任务、状态、耗时及异常类型。既有拒绝仍为ValueError子类并继续返回500，没有默认猜测答案，不记录题干、选项、答案、图片路径、认证头或原始异常文字。官方server及全部模型/训练结构未改。
- 原25项与新增3项诊断/隐私回归共28/28通过。交付与发布副本的Adapter、测试、README已同步。为生效已以同一启动参数、密钥和权重环境重启API至WSL后台（本轮PID914），保留ngrok；已实际核验本机与ngrok公网鉴权healthz均200/ready；真实模型的独立合成分类请求200/chimney，未知合成问法500/inference_failed，本地准确记录unsupported_question_form，没有ngrok跳过头。本机日志为 `%LOCALAPPDATA%/HFSA/PublicAPI/hfsa-api-diagnostics-20261009-112714.log`，不纳入Git。原用户WSL前台命令已退出，不应再重复启动9001。
- 当前实现仍只覆盖有明确支持形式的分类/检测/计数/RefSeg单图题；双图变化、通用问答等仍未实现。下一步由用户再运行一次8题smoke，将新audit与本机原因码对齐，再只修正有证据且现有模型能作答的适配问题；不盲跑full或承诺全部题型可覆盖。

## 2026-10-09 当前排错：官网预检失败与运行密钥重复粘贴

- 用户报告官网 `invalid_submission / participant preflight failed (http_error)`。ngrok 仍在线，日志有评测服务器到本机9001的连接；关闭了 HTTP inspection，尚未观察官网该次请求的实际响应码，不能直接断言官网收到401。
- 只在内存中核对正在运行的 WSL API：MODEL_MODE=real，模型密钥实际为86字符，两个43字符半段完全相同且无空白。使用实际86字符值访问本机及公网 healthz 均200/ready/2.0/rsu-dev100-v2；只用单个43字符半段访问公网明确返回401/unauthorized。密钥值没有打印或写入项目。
- 下一步让用户核对官网：Endpoint 为当前 ngrok origin，无路径及末尾斜杠；模型 API Key 必须与服务启动时的实际值完全一致。维持当前进程时需将此前43字符模型密钥连续输入两遍，无空格或 Bearer 前缀。若日后改回单遍，须重启模型服务并同步网站；这不是要求长期使用重复密钥。
- 本轮未修改业务代码、认证规则或隧道，也未发起网站评测。当前可控浏览器中的官网未登录，尚不能核对用户原填写值或确认预检已通过。Python API 客户端无 ngrok 专用头仍返回JSON；PowerShell默认请求另会遇到ngrok浏览器提示，不能把该提示当作官网此次失败证据。

## 2026-10-09 最新状态：ngrok 公网 HTTPS 与真实预测已验证

- 用户在自己的 WSL 终端以 real 模式启动了四任务 API，监听 `127.0.0.1:9001`，dataset_id 为 `rsu-dev100-v2`。已实际核验带鉴权 healthz 返回200/ready；Windows 也能经 localhost 转发访问该接口，未带 Key 返回401。用户的 API 进程保持运行。
- 用户同意继续公网接入，最终明确选择 ngrok。Windows 官方 agent 已安装至 `%LOCALAPPDATA%/HFSA/PublicAPI/ngrok.exe`，版本3.39.11，ngrok Inc. 的 Authenticode 签名有效；没有安装或修改 Python 模型依赖。
- 用户已提供 ngrok 账号令牌，仅保存到 ngrok 自身的 `%LOCALAPPDATA%/ngrok/ngrok.yml`，官方 config check 通过。agent 在 Windows 后台运行（本轮 PID18848），使用 `--inspect=false`、info日志，映射 `http://127.0.0.1:9001`。当前实际 origin 为 `https://clunky-obsessed-grumpily.ngrok-free.dev`，本机 agent API 已核验映射正确。
- Windows 客户端经上述真实 HTTPS origin 完成外层访问：无鉴权 healthz返回401/json；使用当前模型Key返回200/ready/2.0/rsu-dev100-v2。官方包内一张256×256图的合成题干返回分类 `chimney`、飞机计数 `"0"`、飞机有无 `"No"`，均200/json且协议与ID回显正确。检查未添加 ngrok 跳过提示头，没有以占位答案或官方泛 anything checker 代替真实预测。
- 前一次按用户提到的旧 Cloudflare 工具尝试 Quick Tunnel，虽生成临时地址，但到 Edge 的 TCP/TLS 连接失败，公网 health 返回530/1033；用户改选 ngrok后，已核验停止本轮 Cloudflare 进程。不能把旧临时地址当作可用端点。
- WSL 本轮出现外网 DNS 解析失败，但 Windows 能下载官方工具并访问本机9001，故本轮使用 Windows 隧道 agent。没有变更系统代理、TUN、WSL DNS 或防火墙。
- 本次只验证公网服务链路，没有官方准确率、跨第二网络独立检查或网站评测结果。用户的 WSL API及后台ngrok继续运行；停止API、关闭WSL/电脑或停止agent会影响访问，重启后须重核origin与health。
- 不把 MODEL_API_KEY 或 ngrok Authtoken 写入项目文档、源码或 Git。ngrok 的账户令牌与本机模型 API Key 各自用于对应接口。下一步官网登记上述HTTPS origin与原模型Key，再由用户发起8题smoke；尚未登记或消耗网站评测次数。

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
