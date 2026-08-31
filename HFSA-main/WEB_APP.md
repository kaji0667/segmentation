# 遥感图-文可解释轻量化多任务智能解译系统 Web 界面

当前 Web 界面已经完成三任务选择、动态输入、结果区域和后台路由骨架，并已接入语义分割真实模型。场景分类和目标计数仍显示“接口待接入”，不会生成伪结果。

## 启动

在 `HFSA-main` 目录执行：

```bash
python web_app.py
```

浏览器打开：

```text
http://127.0.0.1:7860
```

如果需要让同一局域网中的其他设备访问：

```bash
python web_app.py --host 0.0.0.0 --port 7860
```

此时使用运行机器的局域网 IP 访问。防火墙、公网 IP、域名、HTTPS 和鉴权不属于当前本地原型范围。

可选自动打开浏览器：

```bash
python web_app.py --open-browser
```

## 当前交互

- 首页根据 `/api/tasks` 自动生成紧凑任务选择项；当前按赛题名称展示场景分类、目标计数和语义分割，后续新增任务不需要重做页面布局。
- 分析工作区在用户选择任务前保持隐藏，选定任务后才显示对应输入和输出区域。
- 场景分类只要求图像。
- 目标计数要求图像和目标类别。
- 语义分割实现为文本引导分割，要求图像和目标描述；内部任务 ID 仍为 `refseg`。
- 浏览器在提交任务时把图像编码为 data URL。语义分割 Adapter 解码图像并调用 `RefSegPredictor`，返回叠加图、二值 Mask、概率图和推理摘要。
- 结果区展示分割阈值、前景占比、模型耗时和原图尺寸，并可分别下载三张 PNG。
- 场景分类和目标计数点击后仍返回 `interface_pending`。

首次语义分割请求需要加载 checkpoint 和 OpenCLIP，耗时会明显高于后续请求。模型实例在服务进程内缓存，关闭服务时释放。

## HTTP 端点

- `GET /api/health`：本地服务健康状态。
- `GET /api/tasks`：三任务可序列化配置。
- `POST /api/predict`：统一路由入口。RefSeg 成功时返回 HTTP 200；未注册任务返回 HTTP 503 和 `interface_pending`；模型异常返回 HTTP 500 和 `inference_error`。

请求 JSON 最大为 64 MiB，浏览器端限制单张原始图像不超过 40 MiB。

## RefSeg 运行配置

默认 checkpoint：

```text
runs/semseg/srp_yolov12m_axis/weights/best_raw.pt
```

可通过环境变量覆盖：

```bash
HFSA_REFSEG_CHECKPOINT=/path/to/best_raw.pt \
HFSA_REFSEG_DEVICE=cuda:0 \
python web_app.py
```

`HFSA_REFSEG_DEVICE` 默认为 `auto`。网页启动和 `/api/tasks` 不导入 RefSeg 模型；只有第一次实际分割请求才加载权重。

## 后续接入边界

每个任务通过 `TaskRouter.register_adapter(task_id, factory)` 注册自己的 Adapter。当前只注册 `RefSegAdapter`；分类和计数后续分别包装自身 Predictor。Adapter 负责 checkpoint 加载、任务预处理、真实推理和任务专属结果；页面与公共路由不修改 Backbone、Neck、OpenCLIP、Head、Loss 或 checkpoint 格式。
