# 遥感图-文可解释轻量化多任务智能解译系统 Web 界面

当前 Web 界面用于先完成三任务选择、动态输入、结果区域和后台路由骨架。页面可以立即访问，但三个真实模型 Adapter 仍按任务逐项接入；界面中的“接口待接入”是当前预期状态。

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
- 语义分割当前实现为文本引导分割，要求图像和目标描述；内部任务 ID 仍为 `refseg`。
- 上传图像后在浏览器本地生成预览，不会在当前接口待接入阶段执行模型推理。
- 点击任务按钮会经过 `/api/predict` 完成后台输入契约校验，并返回 `interface_pending`。

## HTTP 端点

- `GET /api/health`：本地服务健康状态。
- `GET /api/tasks`：三任务可序列化配置。
- `POST /api/predict`：统一路由入口；真实 Adapter 未注册前返回 HTTP 503 和 `interface_pending`。

## 后续接入边界

每个任务通过 `TaskRouter.register_adapter(task_id, factory)` 注册自己的 Adapter。Adapter 负责 checkpoint 加载、任务预处理、真实推理和任务专属结果；页面与公共路由不修改 Backbone、Neck、OpenCLIP、Head、Loss 或 checkpoint 格式。
