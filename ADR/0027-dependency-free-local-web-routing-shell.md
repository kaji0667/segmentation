# ADR-0027: 零新增依赖的本地 Web 路由界面骨架

## 状态

Accepted

## 背景

HFSA 最终需要让普通用户通过浏览器手动选择场景分类、目标计数或指代分割任务。当前三个任务的单图推理接口尚未全部统一，项目环境也未安装 Gradio、Streamlit 等前端框架。此阶段需要先形成可立即访问、可评审视觉效果、且不会提前耦合模型实现的界面与路由架构。

## 决策

1. 新增 `HFSA-main/web_app.py`，使用 Python 标准库 `ThreadingHTTPServer` 提供本地 HTTP 服务，不引入新的运行时依赖。
2. 原生 HTML、CSS 和 JavaScript 放在 `HFSA-main/web/`，页面通过 `/api/tasks` 读取任务配置，不在前端复制三任务输入输出定义。
3. 新增 `tasks/routing/`：
   - `config.py` 定义可序列化的任务、输入和输出结构；
   - `registry.py` 固化三任务展示信息、项目相对 checkpoint 和不同输入输出契约；
   - `router.py` 提供输入校验、延迟 Adapter 注册、实例缓存和释放边界。
4. 当前不导入或加载任何任务模型。未注册 Adapter 时，路由返回明确的 `interface_pending` 状态；后续逐项实现 Adapter，不改变页面任务配置协议。
5. 第一阶段默认只绑定 `127.0.0.1:7860`。公网域名、反向代理、HTTPS 和鉴权属于后续部署任务。

## 备选方案

- Gradio：暂不采用。当前环境未安装，且此阶段只需要界面与路由骨架，不应为原型额外引入依赖。
- Vue/React + FastAPI：暂不采用。视觉自由度高，但会在模型接口尚未齐备时增加构建链、前后端进程和部署复杂度。
- 单个巨型 `web_app.py` 内硬编码任务逻辑：拒绝。会导致界面、路由和模型加载无法独立测试与演进。

## 影响

- 在现有 Python 环境中执行 `python web_app.py` 即可访问页面。
- 前端可以先完成视觉与交互评审，模型接口按 RefSeg、Counting、Classification Adapter 逐项接入。
- 当前 `/api/predict` 只验证任务契约并报告接口待接入，不代表已完成真实模型推理。
- 若后续切换到 FastAPI 或独立前端，`tasks/routing/` 的配置与 Adapter 边界可以继续复用。

## 关联

- ADR-0026
- `HFSA-main/web_app.py`
- `HFSA-main/web/`
- `HFSA-main/tasks/routing/`
