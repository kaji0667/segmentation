# 参赛队伍最小模型 API 示例（协议 2.0）

完整字段、题型与错误处理见 [模型 API 协议 2.0](PROTOCOL_2_0.md)；评测网站的 `/help` 页面也提供独立可读的接入说明。

可选的真实 Qwen3.5-4B 参考实现与启动步骤见 [QWEN_ADAPTER.md](QWEN_ADAPTER.md)；`MODEL_MODE=demo` 仍无需 GPU 或模型依赖。

这个文件夹可以单独复制给队伍。它提供标准库协议外壳、可选的 Qwen3.5-4B 适配代码、测试和部署说明；**不含题目、金标、图片、模型权重或密钥**。队伍可以使用任意 VLM、推理框架和 GPU，只需让 API 遵守下面的接口。`DemoModel` 不看图、不懂题，只用于确认接入通路；不要拿它的分数衡量模型。

## 1. 整体工作方式

```text
评测网站 --HTTPS JSON(题干、约束、图片编号/哈希)--> 队伍 API --本地读取--> 预置图片
                                              |
                                              +--> 队伍自己的模型/GPU
评测网站 <--JSON(答案)-------------------------+
```

开发集 100 题的图片资源包由组委会另行提供，先下载到**运行模型 API 的机器**，解开后应是：

```text
/path/to/dev-image-package/
  manifest.json
  assets/
    <32 位小写十六进制 asset_id>
    ...
```

网站的单题请求**不传图片字节、图片 URL、原始文件路径或金标**；只发 `asset_id`、SHA-256、MIME 类型。服务从本地 `assets/` 取图并核验哈希。双时相题按请求 `images` 数组顺序送入模型。开发包的 `dataset_id` 应为 `rsu-dev100-v2`；正式包可能是另一编号，API 从本地 `manifest.json` 自动读取，不应硬编码开发编号。以组委会最终发布的包和校验值为准。

## 2. 文件说明

| 文件 | 用途 |
| --- | --- |
| `server.py` | `GET /healthz`、`POST /v1/predict`、Bearer 认证、本地图片核验 |
| `model_adapter.py` | 演示模型与需要队伍实现的 `RealModel` 接口 |
| `qwen_adapter.py`、`QWEN_ADAPTER.md` | Qwen3.5-4B 真实模型适配参考与逐步说明 |
| `requirements-qwen-reference.lock` | 本地 CUDA 13.0 环境参考版本，不是所有 GPU 的通用配置 |
| `check_api.py` | 对本机或公网地址做一次**不计分**的握手和预测检查 |
| `tests/test_server.py` | 用合成数据测试协议；不依赖真实赛题 |
| `Caddyfile.example` | 公网 GPU 服务器的 HTTPS 反向代理示意 |

协议外壳和演示模式只依赖 Python 3.10+ 标准库。`MODEL_MODE=qwen` 才需要 PyTorch、Transformers、Pillow 等；其他模型按队伍选择安装依赖。下文用 `python3` 表示选定环境中的 Python；如果机器上已有兼容的虚拟环境，可以直接用该环境的 `bin/python` 运行测试、服务和 `check_api.py`，无需重复安装依赖。运行真实 Qwen 时，先按 [Qwen 环境说明](QWEN_ADAPTER.md)核对 GPU、PyTorch 和模型目录。

## 3. 本地跑通协议

先在这个文件夹运行自带的合成测试，无需 GPU 或真实图片：

```bash
cd participant_api_starter
python3 -m unittest discover -s tests -v
```

然后把组委会提供的图片包解到推理机上的单独目录，确认 `manifest.json` 与 `assets/` 同级。设置一个**队伍自己生成**的模型 API token（不同于网站登录令牌、ngrok authtoken）：

```bash
# 从密码管理器取出至少 32 位随机值；此交互输入不会出现在 shell 命令历史。
read -rs -p 'MODEL_API_KEY: ' MODEL_API_KEY; echo
export MODEL_API_KEY
export MODEL_MODE=demo
python3 server.py --package /path/to/dev-image-package --host 127.0.0.1 --port 9001
```

上面的服务默认仅监听本机回环地址。另开一个终端，设置**同一个** `MODEL_API_KEY`，验证；两个终端的环境变量互不共享：

```bash
read -rs -p 'MODEL_API_KEY: ' MODEL_API_KEY; echo
export MODEL_API_KEY
python3 check_api.py --origin http://127.0.0.1:9001 --package /path/to/dev-image-package
```

`check_api.py` 从清单挑一张本地图片，发送合成问题，只检查协议、鉴权、图片查找和响应回显；它不会得到开发集分数。如果只是测健康接口，也可用：

```bash
curl -fsS -H "Authorization: Bearer $MODEL_API_KEY" http://127.0.0.1:9001/healthz
```

预期至少包含 `{"status":"ready","protocol_version":"2.0","dataset_id":"rsu-dev100-v2"}`。若 `dataset_id` 不符，检查是否指向了错误的图片包。如果 `/healthz` 或 `check_api.py` 返回 401，先确认检查终端使用的是**服务启动时**的密钥。服务在启动时读取 `MODEL_API_KEY`；之后只在某个终端修改或重新生成密钥，不会更新已运行的服务。更换密钥时，重启服务，并在检查终端和网站填写同一新密钥。

`check_api.py` 成功只证明一次合成题走通，演示模型也能通过。若要评测真实 Qwen，确认启动服务前设置了 `MODEL_MODE=qwen` 和 `QWEN_MODEL_DIR`，且启动日志没有 `demo mode` 警告。脚本在本地 HTTP 检查成功时也会打印 `HTTPS/API handshake`；这条固定提示不代表公网 HTTPS 已验证。

## 4. 接入自己的模型

想看真实模型接入，先读 [Qwen3.5-4B 适配示例](QWEN_ADAPTER.md)。准备本地权重和兼容环境后，设置 `MODEL_MODE=qwen`、`QWEN_MODEL_DIR=/absolute/path/to/Qwen3.5-4B`，继续使用同一个 `server.py`。其他 VLM 可借鉴其题干约束、双图顺序和 bbox 坐标转换。

编辑 `model_adapter.py` 中的 `RealModel`：

1. 在 `__init__` 里加载模型与权重，放到 GPU；服务启动时只做一次，**不要每题重载**。
2. 在 `predict(request, image_paths)` 中按 `image_paths` 顺序读取本地图像。不要把 `asset_id` 当成远程 URL，也不要向评测网站请求图片。
3. 将 `request["question"]`、可选的 `request["choices"]` 与 `request["response_constraint"]` 一起组织为模型输入。
4. 输出与题型相符的原始答案。短文本、单选、区域标签和整数题都返回**字符串**；`bbox` 返回原图像素坐标数组 `[xmin, ymin, xmax, ymax]`，必须满足 `xmin < xmax`、`ymin < ymax`。若模型产生 0–1000 归一化坐标，需要先乘原图宽高转换。
5. 不要在日志里打印 API token、完整题目、图片或模型私有信息。

改完再用 `export MODEL_MODE=real` 启动。未实现 `RealModel` 时服务会明确报错，这是为了防止把演示答案误当真实模型结果。这个示例一次只允许一个推理请求；评测站对同一队伍顺序发题，串行是安全的最小起点。若你的模型能并行推理，可按显存和实际吞吐自行调整 `server.py` 的并发限额。

### 接口样例

`GET /healthz` 和 `POST /v1/predict` 都要求 `Authorization: Bearer <MODEL_API_KEY>`。预测请求示意（编号和哈希取自本地图片包，此处不放真实题目）：

```json
{
  "protocol_version": "2.0",
  "request_id": "run-id:EXAMPLE:1",
  "item_id": "EXAMPLE",
  "question": "Is there anything in the image?",
  "images": [{
    "asset_id": "<manifest 中的 32 位编号>",
    "sha256": "<manifest 中的 64 位哈希>",
    "mime_type": "image/png"
  }],
  "response_constraint": {"type": "enum", "values": ["Yes", "No"]},
  "answer_type": "short_text"
}
```

成功时必须回显版本、请求 ID、题目 ID，并给出 `answer`：

```json
{"protocol_version":"2.0","request_id":"run-id:EXAMPLE:1","item_id":"EXAMPLE","answer":"Yes"}
```

`response_constraint.type` 可能为 `enum`、`single_choice`、`short_text`、`integer` 或 `bbox`。`single_choice` 另带 `choices`；`bbox` 另带 `image_width`、`image_height`。推理异常要返回非 2xx JSON，不能返回伪造的正确答案；失败题在网站上按缺失预测处理。

## 5. 最快公网接入：ngrok（适合调试）

如果 GPU 在本机、WSL 或不能开公网端口的机房，ngrok 是一个简单的起点：模型 API 继续监听 `127.0.0.1:9001`，ngrok 主动连出并提供公网 HTTPS 域名，不需要给 GPU 机开入站端口。**在运行 API 的同一系统/网络命名空间内**安装 [ngrok agent](https://ngrok.com/docs/getting-started/)，注册账号并按官网指引配置 authtoken；ngrok authtoken 与 `MODEL_API_KEY` 不能混用。

保持上一节的 API 终端运行，再开一个终端：

```bash
read -rs -p 'NGROK_AUTHTOKEN: ' NGROK_AUTHTOKEN; echo
ngrok config add-authtoken "$NGROK_AUTHTOKEN"
unset NGROK_AUTHTOKEN
ngrok http 9001
```

如果出现 [`ERR_NGROK_9009`](https://ngrok.com/docs/errors/err_ngrok_9009)，检查 ngrok 配置中的 `proxy_url`，以及当前终端的 `http_proxy`、`https_proxy`、`HTTP_PROXY`、`HTTPS_PROXY` 等变量。这个错误表示 ngrok agent 正尝试经 HTTP(S) 代理连接；若机器可以直连，可只对 ngrok 进程临时去掉代理变量后重试，不改变其他程序的代理设置：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u all_proxy -u ALL_PROXY ngrok http 9001
```

如果网络必须通过代理才能出站，去掉变量后可能无法连接，应改用符合账号能力的连接方式或下节的自有 HTTPS 服务。

不要把 authtoken 写进仓库、截图或发给组委会。以 ngrok 终端**实际显示**的 HTTPS 地址为准，域名后缀可能是 `.ngrok-free.dev` 等。**只把 origin 填到评测网站**，不要附加 `/healthz`、`/v1/predict`、问号参数或末尾路径。将 `MODEL_API_KEY` 的原始值填到网站的“模型 API key”字段，不加 `Bearer ` 前缀；网站登录令牌只用于登录网站。

先在当前机器用公网地址检查，再视需要从其他网络重复检查。运行 `check_api.py` 的机器必须能读取同一图片包的 `manifest.json`，并设置相同的 `MODEL_API_KEY`；检查脚本不读取该机的 `assets/`。把示例地址替换成 ngrok 当前显示的完整 HTTPS origin：

```bash
PUBLIC_ORIGIN=https://your-ngrok-domain.ngrok-free.dev
python3 check_api.py --origin "$PUBLIC_ORIGIN" --package /path/to/dev-image-package
```

如果检查失败，先用 `curl -i -H "Authorization: Bearer $MODEL_API_KEY" "$PUBLIC_ORIGIN/healthz"` 看是否返回 **200 + JSON**，而不是 HTML、重定向或 ngrok 错误页。ngrok 免费档目前有流量/请求额度与浏览器访问提示页；评测程序是 API 客户端，但是否受到提示页影响仍要以上述端到端检查为准。**不要依赖评测网站会替你加 ngrok 专用跳过提示头**。ngrok 定价和限额会变化，请以 [官方定价页](https://ngrok.com/pricing) 为准。重启 ngrok 后也要重新确认地址和健康检查，不要假定隧道始终在线。

评测期间必须保持 GPU 机器、模型进程、ngrok 进程和网络一直在线；电脑休眠、WSL 关闭或隧道退出都会导致失败。建议先在网站跑 8 题 `smoke`，等待运行完成并检查超时、缺失预测等异常，再跑 100 题 `full`。ngrok 适合开发和短时验证；若比赛要求较长稳定在线，优先用下一节的自有公网服务。

## 6. 长时间接入：自有 HTTPS 服务

### A. GPU 服务器本身有公网 IP

为 GPU 服务器准备自己的域名，配置 DNS A/AAAA 记录，并在云安全组和主机防火墙放行 TCP 80/443。模型 API 仍绑定 `127.0.0.1:9001`，不要直接开放 9001。用 [Caddy 官方反向代理配置](https://caddyserver.com/docs/quick-starts/reverse-proxy)中的方式把 HTTPS 流量转给它；可将 `Caddyfile.example` 的域名改为自己的域名。Caddy 在满足 DNS 和端口条件时自动办理 HTTPS 证书。启动后从外网运行上一节相同的 `check_api.py`，成功才登记到评测网站。

### B. GPU 机器在内网，但队伍有自己的轻量公网主机

在公网主机配置域名 + Caddy，反代到该主机的 `127.0.0.1:19001`。在 GPU 机器上启动模型 API 后，再手动建立 [OpenSSH 反向端口转发](https://man.openbsd.org/ssh.1)：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -R 127.0.0.1:19001:127.0.0.1:9001 relay-user@relay.team.example.com
```

这里 `relay.team.example.com`、SSH 用户和反代域名都由**队伍自己**提供；不要默认占用组委会开发站的轻量云服务器。公网主机 Caddy 反代目标改为 `127.0.0.1:19001`。应给 SSH 隧道使用独立低权限账号，并限制其可监听端口；同样需要防止 GPU 机器休眠和 SSH 连接断开。这个方案的公网主机仅转发小型 JSON 请求与答案，图片仍留在队伍 GPU 机器的本地资源包。

这两种方式都将公网 HTTPS 终止在队伍控制的服务或中继上；ngrok 则经过第三方服务。队伍应依据自身安全和可用性要求选择。

## 7. 到评测网站登记与排错

1. 用组委会给的队伍账号登录开发评测网站；队伍只能看到本队成绩。打开模型 API Endpoint 设置，填上**自己**的 HTTPS origin（例如 ngrok URL），保存。
2. 在提交评测时填入与模型 API 服务端一致的 `MODEL_API_KEY`。网站会先检查 `/healthz`，包括 `protocol_version=2.0` 和 `dataset_id`，再开始计次。
3. 先跑 `smoke`，确认图片包、本地推理、响应格式和公网连通性；再跑 `full`。示例里的 `DemoModel` 只用于检查；真实评测可使用已配置的 `MODEL_MODE=qwen`，或设置 `MODEL_MODE=real` 并实现自己的模型适配器。

| 现象 | 优先检查 |
| --- | --- |
| 本地 `check_api.py` 返回 401 | 检查终端和服务启动时的 `MODEL_API_KEY` 是否相同；改过密钥就重启服务 |
| 网站提交报 `invalid_submission`、`participant preflight failed (http_error)` | 网站预检访问 `/healthz` 得到了非 200 状态；在 ngrok 的本地请求检查界面或反向代理日志中查看该请求的实际状态码；界面地址以 ngrok 终端显示的为准，端口不一定是 4040。若是 401，即使本地或公网 `check_api.py` 成功，也应重填网站提交时的模型 API key；不要填 ngrok authtoken、网站登录令牌或 `Bearer ` 前缀 |
| `ERR_NGROK_9009` | 检查 ngrok 是否继承了 HTTP(S) 代理变量，或配置了 `proxy_url`；可按上文只对 ngrok 进程临时取消代理并验证直连 |
| `/healthz` 不是 `ready` 或数据集不符 | 是否用了正确的 `manifest.json`、`assets/` 包，以及正确的 `--package` 目录 |
| 422 图片错误 | 图片文件是否缺失/被改动；文件 SHA-256 是否与 `manifest.json` 相同 |
| 502/HTML/重定向 | ngrok、Caddy、浏览器提示页或反代配置；程序必须收到 200 + `application/json` |
| 连接超时 | GPU 机器/模型/隧道是否在线；外网 DNS、HTTPS 证书和防火墙是否正常 |
| 有结果但分数低 | 先确认不是 `DemoModel`；再检查答案类型、单选字母、bbox 原图像素坐标与题目约束 |

ngrok 的请求检查界面可能记录 `Authorization` 请求头；排错时只分享请求路径和状态码，不要公开原始请求头。不要把图片资源包、模型权重、`.env`、私钥或真实 API token 一并打包给他人。这个文件夹的 `.gitignore` 默认忽略常见本地机密和大文件，但仍应在发布前自行检查归档内容。

## 8. 本地 HFSA 四任务接入（2026-10-09）

`MODEL_MODE=real` 通过 `hfsa_adapter.py` 复用已有分类、检测、计数和 RefSeg Predictor。官方请求没有 `task_id`，因此只有这个 Adapter 按已支持的题干形式选择任务；原 Web 仍由用户手动选择。

当前电脑目录为 `提交/HFSA-main/`（代码），同级 `提交/HFSA_models/`（权重）。需要分类 `runs/classification/vrsbench_scene/weights/best.pt`、检测 `runs/detect/DIOR-RSVG-ViT-L-14/weights/best.pt`、计数 `runs/counting/counting-VRSBench-ViT-L-14/weights/best.pt`、分割 `runs/semseg/srp_yolov12m_axis/weights/best_raw.pt` 和 `pretrain_model/yolov12m.pt`。可用 `HFSA_MODELS_DIR` 指定其他权重根目录，`HFSA_API_DEVICE` 指定设备（默认 `auto`）。OpenCLIP ViT-L-14/openai 须已在运行账户缓存中。

在 **WSL 终端**执行以下本机命令，继续使用现有 `hfsa_env`，无需 Qwen 环境：

```bash
cd /mnt/d/code/python/HFSA/提交/HFSA-main
source /mnt/d/code/python/HFSA/hfsa_env/bin/activate
read -rs -p '输入你保存的 MODEL_API_KEY（至少32字符）: ' MODEL_API_KEY
printf '\n'
export MODEL_API_KEY
export MODEL_MODE=real
python participant_api_starter-main/server.py \
  --package /mnt/d/code/python/HFSA/rsu-dev100-v2-images \
  --host 127.0.0.1 --port 9001
```

四任务均在监听前加载一次，检测与计数共用一个既有 OpenCLIP 文本编码器；分割保留自己的 token encoder。启动会显示加载进度，等待 `API listening` 才表示可以接题。此前本机合成 smoke 启动约 101 秒，实际加载可能更久，以监听及 ready 为准，单次检查峰值 PyTorch 显存约 3.86 GiB；这不是正式效率评测。

| 任务 | 已支持题干示例 | 返回方式 |
| --- | --- | --- |
| 场景分类 | `Which scene category best describes the image?` / `图中属于什么场景？` | 单选返回字母，枚举返回允许值，短文本返回 checkpoint 类名 |
| 目标计数 | `How many ships are there in the image?` / `图中有多少架飞机？` | 整数字符串；数值选项返回匹配的标识 |
| 目标检测 | `Is there a ship in the image?` / `图中是否有飞机？` | 根据检测结果返回 Yes/No 或对应选项 |
| 目标检测 bbox | `Detect a windmill in the image.` / `检测图中的飞机` | 指定单一类别中置信度最高的原图像素框 |
| 指代分割转定位 | `Locate the airplane near the bridge.` / `框出图中灰色的小风车。` | 原图二值 Mask 的外接框；保留表达中的限定词 |

分类支持 checkpoint 中的 21 个场景类别，选项的英文别名或简单中文必须能映射到这些类别；含未知类或重复同义选项则报错。检测/计数要求明确的遥感目标类别。中文复杂关系仍由现有翻译器明确拒绝；英文 RefSeg 表达保持原样送入模型。计数保持原 `conf=0.15`、`IoU=0.5`、`max_det=300` 和独立计数权重，不增加空间关系计数策略。

未知题型、双图变化、通用问答、空间/多类别计数、空目标及空定位结果均失败，通过官方外壳返回 HTTP 500 `inference_failed`。本实现不是通用视觉语言模型，不保证覆盖整个开发集。官方 `check_api.py` 的 `Is there anything...` 合成题仍不支持，因为它没有指定目标类别；不要把该题失败误判为鉴权或连通故障。

本地 Adapter 现在会向 stderr 输出一行 `hfsa_api_prediction` JSON 诊断，包含题号、请求编号、答案约束、图片数量、所选任务、耗时及固定失败代码。它不记录题干、选项、答案、图片路径、认证头或原始异常文字；公网错误响应仍由官方 `server.py` 处理。代码更新后须重启 API 才能生效。

常见 `reason`：`unsupported_question_form` 为未覆盖的问法；`unsupported_image_count` 为当前不支持的双图任务；`unsupported_scene_options` 为场景选项超出分类权重类别；`unsupported_spatial_or_multi_category_count` 为未实现的计数条件；`empty_refseg_mask` / `empty_detections` 为模型未找到目标；`unexpected_exception` 为需要继续排查的异常，同时记录异常类型。诊断只解释失败类别，不会把未知题转为猜测答案，也不会自动提高成绩。

如果官网 `audit` 显示 HTTP 500，而 `predictions` 缺少相应行，应结合本机这些诊断定位；audit 本身只包含状态/摘要，不能还原题干或 Python 异常。先排查 8 题 smoke，不重复运行 full。后台运行时本机日志位置见项目 CURRENT_STATE，账号令牌及原始评测下载文件不纳入 Git。

可重复检查：

```bash
cd participant_api_starter-main
python -m unittest discover -s tests -v
python tests/smoke_hfsa_real.py \
  --image /path/to/your/windmill-image.jpg \
  --target windmill --refseg-question 'The gray small windmill'
```

真实 smoke 用用户提供的单张图创建临时合成图片包，只监听随机本机端口，并检查四任务、鉴权、健康检查、标识回显、像素框和双图拒绝；完成后关闭服务并删除临时包。它不改官方图片包、不打印测试 Key，也不提交网站评测。后续公网接入状态见第9节。

## 9. 当前电脑的 Windows ngrok 接入准备（2026-10-09）

用户随后决定继续公网接入，并选择 ngrok。已将官方 Windows agent 安装在 `%LOCALAPPDATA%\HFSA\PublicAPI\ngrok.exe`，签名有效，版本3.39.11。Windows 已核验能访问 WSL 的 `127.0.0.1:9001`，所以这台电脑可从 Windows 启动隧道。保持原 WSL 模型服务运行。

先在 [ngrok 令牌页面](https://dashboard.ngrok.com/get-started/your-authtoken) 登录或注册，复制 Authtoken。它用于 ngrok 账号连接；官网评测仍填写自己的 MODEL_API_KEY。在 **Windows PowerShell** 隐藏输入 ngrok 令牌并保存到 ngrok 自身的用户配置：

```powershell
$ngrokToken = Read-Host '粘贴 ngrok Authtoken' -AsSecureString
& "$env:LOCALAPPDATA\HFSA\PublicAPI\ngrok.exe" config add-authtoken ([System.Net.NetworkCredential]::new('', $ngrokToken).Password)
```

该命令只把账号令牌保存到 ngrok 的本机用户配置，不写项目文件。后续需要重新启动时执行（当前本轮已由助手在后台启动，无需再启动一个）：

```powershell
& "$env:LOCALAPPDATA\HFSA\PublicAPI\ngrok.exe" http http://127.0.0.1:9001 --inspect=false
```

使用 agent 实际显示的 HTTPS origin，并先验证认证 healthz 及本适配器支持的真实预测。本轮ngrok账号配置及公网验证已完成：无Key的healthz返回401，携带当前模型Key返回200/ready/正确dataset_id；分类、计数和目标有无合成题均由真实模型返回200/json及正确ID回显，未加ngrok专用跳过提示头。当前origin见项目 CURRENT_STATE；重启agent后应重新确认地址。先前 Cloudflare Quick Tunnel 因到 Edge 的TLS连接失败而未通，并已停止。

官网Endpoint填写HTTPS根地址，模型API Key填写此前用于启动模型服务的Key；ngrok Authtoken已用于agent账号配置。保持WSL模型终端、后台ngrok和电脑运行。先由用户在官网跑8题smoke，检查缺失预测/超时后再考虑完整开发集。本轮没有提交官网评测。
