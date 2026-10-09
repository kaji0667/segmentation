# ADR-0035：官方协议 2.0 的四任务题干适配

## 状态

Accepted — 用户在 2026-10-09 明确要求把其他任务接入现有 API，并暂缓公网。

## 背景

官方协议只有 question、choices、response_constraint 和本地图片引用，没有 Web 的 task_id。单任务 RefSeg 已验证，用户要求继续接入团队已有分类、检测和计数权重。作答约束仅决定格式，不能独自决定语义任务。

## 决策

1. 主模块为 `participant_api_starter-main/hfsa_adapter.py`；`model_adapter.py::RealModel` 只转交加载和预测。官方 `server.py` 继续处理鉴权、哈希、并发、协议和标识回显。
2. 只在官方 Adapter 按明确的中英文题干模板及作答约束选择任务。ADR-0026 的手动选择仍适用于 Web；不自动修改网页路由。
3. 复用四个任务已有 Predictor 和正式权重，不改 Backbone、Neck、Head、OpenCLIP、训练、Loss 或评测协议。
4. 分类单选/枚举按 checkpoint 全类别概率选择可映射的选项，返回原选项标识/原允许值；含未知或同义重复选项则拒绝。短文本返回 checkpoint 类名。
5. 计数使用独立 counting checkpoint，返回整数字符串或唯一匹配的数值选项；不截断/抬高预测以迎合作答约束。保持 conf=0.15、IoU=0.5、max_det=300；拒绝空间关系及多类别计数。
6. 明确目标类别的有无题使用 detection checkpoint；显式 Detect/检测且要求单框时选择最高置信度原图框。带指代表达的 Locate/框出题复用 RefSeg，保留限定词并取原图 mask 外接框。
7. 启动时预加载四任务，加载完成前不监听或报告 ready。检测和计数通过既有 prompt_encoder_factory 共享同配置的 TextPromptEncoder；RefSeg 仍使用独立 token encoder。
8. 拒绝双图变化、通用问答、泛化的 anything/object 有无题、未知类别、空目标和空定位。官方外壳返回 inference_failed，不编造合法格式的默认答案。

## 取舍

- 确定性模板只覆盖明确支持的问法，不声称通用语言理解或完整开发集覆盖。
- bbox 只能返回单框，显式类别检测的最高置信度选择与指代分割的 mask 外接框是不同策略；不把所有检测框合并成任意大框。
- 预加载延长启动并占用多任务显存，但避免在官方默认 120 秒单题超时内首次加载模型。本机完整 startup 100.64 秒、峰值 allocated 3.86 GiB，仅为 smoke 证据。
- 不硬编码 Yes 通过官方通用 checker；采用可重复的四任务真实本机 HTTP smoke。

## 验证

官方与新增测试 25/25 通过。真实权重用 RRSIS-D 800×800 风车图片和临时合成包完成四任务 HTTP、认证、ready、ID 回显、原图框以及双图明确失败。没有正式得分或全开发集准确率结论。

## 2026-10-09 诊断补充

用户首次官网8题smoke中7题返回500，audit不能提供本地异常。允许在同一HFSA Adapter的predict边界输出固定失败代码、题号、所选任务及耗时；已知拒绝保持ValueError子类，未知错误仅输出异常类型。诊断不包含题干、选项、答案、图片路径、认证头或原始异常文字，不修改server或公网错误内容，也不增加猜测答案。新增3项行为及隐私回归，加既有测试为28/28；真实官网逐题失败原因仍需下一轮请求后核验。
