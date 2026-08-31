const state = {
  tasks: [],
  selectedTask: null,
  imageFile: null,
  imageUrl: null,
};

const taskGrid = document.querySelector("#task-grid");
const taskCount = document.querySelector("#task-count");
const workspace = document.querySelector("#workspace");
const selectedBadge = document.querySelector("#selected-task-badge");
const inputSummary = document.querySelector("#input-summary");
const dynamicFields = document.querySelector("#dynamic-fields");
const runButton = document.querySelector("#run-button");
const runLabel = document.querySelector("#run-label");
const formMessage = document.querySelector("#form-message");
const resultHeading = document.querySelector("#result-heading");
const resultHelp = document.querySelector("#result-help");
const resultStage = document.querySelector("#result-stage");
const outputTags = document.querySelector("#output-tags");
const interfaceStatusChip = document.querySelector("#interface-status-chip");
const imageInput = document.querySelector("#image-input");
const uploadZone = document.querySelector("#upload-zone");
const uploadEmpty = document.querySelector("#upload-empty");
const previewWrap = document.querySelector("#image-preview-wrap");
const imagePreview = document.querySelector("#image-preview");
const imageName = document.querySelector("#image-name");
const imageSize = document.querySelector("#image-size");
const replaceImage = document.querySelector("#replace-image");

const iconCharacters = {
  scene: "▦",
  counting: "◎",
  refseg: "◫",
};

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "";
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function setMessage(message = "", kind = "") {
  formMessage.textContent = message;
  formMessage.className = `form-message${kind ? ` ${kind}` : ""}`;
}

function renderPlaceholder(message, detail) {
  resultStage.innerHTML = `
    <div class="result-placeholder">
      <div class="scan-frame" aria-hidden="true">
        <span class="corner top-left"></span>
        <span class="corner top-right"></span>
        <span class="corner bottom-left"></span>
        <span class="corner bottom-right"></span>
        <span class="scan-line"></span>
      </div>
      <strong></strong>
      <p></p>
    </div>
  `;
  resultStage.querySelector("strong").textContent = message;
  resultStage.querySelector("p").textContent = detail;
}

function setInterfaceStatus(task) {
  const available = ["ready", "loaded"].includes(task.interface_status);
  interfaceStatusChip.textContent = available ? "模型可用" : "接口待接入";
  interfaceStatusChip.classList.toggle("ready", available);
}

function renderTasks() {
  taskGrid.innerHTML = "";
  state.tasks.forEach((task) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "task-card";
    card.dataset.taskId = task.task_id;
    card.style.setProperty("--accent", task.accent_color);
    card.setAttribute("aria-pressed", "false");
    card.innerHTML = `
      <span class="task-icon" aria-hidden="true">${iconCharacters[task.icon] || "◇"}</span>
      <span class="task-copy">
        <strong>${task.title}</strong>
        <small>${task.description}</small>
      </span>
      <span class="task-chevron" aria-hidden="true">→</span>
    `;
    card.addEventListener("click", () => selectTask(task.task_id));
    taskGrid.appendChild(card);
  });
}

function renderDynamicFields(task) {
  dynamicFields.innerHTML = "";
  task.inputs.filter((field) => field.kind !== "image").forEach((field) => {
    const wrapper = document.createElement("div");
    wrapper.className = "dynamic-field";
    wrapper.innerHTML = `
      <label class="field-label" for="field-${field.key}">
        ${field.label}
        <span>${field.required ? "必填" : "选填"}</span>
      </label>
      <textarea id="field-${field.key}" data-field-key="${field.key}" placeholder="${field.placeholder}"></textarea>
      <p class="field-help">${field.help_text}</p>
    `;
    dynamicFields.appendChild(wrapper);
  });
}

function selectTask(taskId) {
  const task = state.tasks.find((item) => item.task_id === taskId);
  if (!task) return;
  state.selectedTask = task;
  document.documentElement.style.setProperty("--task-accent", task.accent_color);
  document.querySelectorAll(".task-card").forEach((card) => {
    const selected = card.dataset.taskId === taskId;
    card.classList.toggle("selected", selected);
    card.setAttribute("aria-pressed", String(selected));
  });
  workspace.hidden = false;
  selectedBadge.textContent = `${task.title} · ${task.technical_name}`;
  selectedBadge.classList.add("active");
  inputSummary.textContent = task.inputs.map((field) => field.label.replace("上传", "")).join(" + ");
  resultHeading.textContent = task.result_title;
  resultHelp.textContent = task.detail;
  runLabel.textContent = task.action_label;
  runButton.disabled = false;
  setMessage("");
  setInterfaceStatus(task);
  renderPlaceholder("结果将在这里呈现", task.detail);
  renderDynamicFields(task);
  outputTags.innerHTML = task.outputs.map((output) => `<span class="output-tag">${output.label}</span>`).join("");
  workspace.scrollIntoView({ behavior: "smooth", block: "start" });
}

function setImageFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/") && !/\.(tif|tiff)$/i.test(file.name)) {
    setMessage("请选择 JPG、PNG、WEBP 或 TIFF 图像。", "error");
    return;
  }
  if (file.size > 40 * 1024 * 1024) {
    setMessage("图像文件过大，请上传不超过 40 MiB 的图像。", "error");
    return;
  }
  if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
  state.imageFile = file;
  state.imageUrl = URL.createObjectURL(file);
  imagePreview.src = state.imageUrl;
  imageName.textContent = file.name;
  imageSize.textContent = formatBytes(file.size);
  uploadEmpty.hidden = true;
  previewWrap.hidden = false;
  setMessage("");
}

imageInput.addEventListener("change", (event) => setImageFile(event.target.files[0]));
replaceImage.addEventListener("click", (event) => {
  event.preventDefault();
  event.stopPropagation();
  imageInput.click();
});

["dragenter", "dragover"].forEach((name) => {
  uploadZone.addEventListener(name, (event) => {
    event.preventDefault();
    uploadZone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((name) => {
  uploadZone.addEventListener(name, (event) => {
    event.preventDefault();
    uploadZone.classList.remove("dragging");
  });
});

uploadZone.addEventListener("drop", (event) => setImageFile(event.dataTransfer.files[0]));

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error || new Error("FileReader failed"));
    reader.readAsDataURL(file);
  });
}

function formatPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(2)}%`;
}

function formatLatency(value) {
  const latency = Number(value);
  return Number.isFinite(latency) ? `${latency.toFixed(1)} ms` : "—";
}

function resultFilename(suffix) {
  const stem = (state.imageFile?.name || "refseg-result").replace(/\.[^.]+$/, "");
  return `${stem}_${suffix}.png`;
}

function createResultImage(label, source, className = "") {
  const figure = document.createElement("figure");
  figure.className = `result-image ${className}`.trim();
  const image = document.createElement("img");
  image.src = source;
  image.alt = label;
  const caption = document.createElement("figcaption");
  caption.textContent = label;
  figure.append(image, caption);
  return figure;
}

function renderRefsegResult(result) {
  const images = result?.images || {};
  const summary = result?.summary || {};
  if (!images.overlay || !images.mask || !images.probability) {
    throw new Error("语义分割结果缺少可视化图像。");
  }

  resultStage.innerHTML = "";
  const content = document.createElement("div");
  content.className = "refseg-result";

  if (summary.found_target === false || Number(summary.foreground_pixels) === 0) {
    const warning = document.createElement("div");
    warning.className = "result-warning";
    warning.textContent = "未找到符合描述的目标。请补充位置、颜色或大小信息后重试。";
    content.appendChild(warning);
  }

  if (summary.prompt_translated && summary.model_prompt) {
    const translation = document.createElement("div");
    translation.className = "translation-note";
    const label = document.createElement("span");
    const value = document.createElement("strong");
    label.textContent = "中文提示已转译为";
    value.textContent = summary.model_prompt;
    translation.append(label, value);
    content.appendChild(translation);
  }

  const gallery = document.createElement("div");
  gallery.className = "result-gallery";
  gallery.append(
    createResultImage("目标叠加效果", images.overlay, "result-image-main"),
    createResultImage("二值 Mask", images.mask),
    createResultImage("像素概率图", images.probability),
  );

  const metrics = [
    ["分割阈值", Number(summary.threshold).toFixed(2)],
    ["前景占比", formatPercent(summary.foreground_ratio)],
    ["模型耗时", formatLatency(summary.latency_ms)],
    ["原图尺寸", Array.isArray(summary.original_size) ? `${summary.original_size[1]} × ${summary.original_size[0]}` : "—"],
  ];
  const metricGrid = document.createElement("div");
  metricGrid.className = "result-metrics";
  metrics.forEach(([label, value]) => {
    const item = document.createElement("div");
    const name = document.createElement("span");
    const amount = document.createElement("strong");
    name.textContent = label;
    amount.textContent = value;
    item.append(name, amount);
    metricGrid.appendChild(item);
  });

  const downloads = document.createElement("div");
  downloads.className = "result-downloads";
  [
    ["下载叠加图", "overlay"],
    ["下载 Mask", "mask"],
    ["下载概率图", "probability"],
  ].forEach(([label, key]) => {
    const link = document.createElement("a");
    link.href = images[key];
    link.download = resultFilename(key);
    link.textContent = label;
    downloads.appendChild(link);
  });

  content.append(gallery, metricGrid, downloads);
  resultStage.appendChild(content);
}

function renderResult(taskId, result) {
  if (taskId === "refseg") {
    renderRefsegResult(result);
    return;
  }
  renderPlaceholder("任务执行完成", "该任务结果渲染器将在对应模型接口接入时补充。");
}

runButton.addEventListener("click", async () => {
  if (!state.selectedTask) {
    setMessage("请先选择一项分析任务。", "error");
    return;
  }
  if (!state.imageFile) {
    setMessage("请先上传一张遥感图像。", "error");
    return;
  }

  const inputs = {};
  for (const field of state.selectedTask.inputs.filter((item) => item.kind !== "image")) {
    const value = document.querySelector(`[data-field-key="${field.key}"]`)?.value.trim() || "";
    if (field.required && !value) {
      setMessage(`请填写“${field.label}”。`, "error");
      return;
    }
    inputs[field.key] = value;
  }

  runButton.disabled = true;
  setMessage("正在读取图像并准备推理…");
  renderPlaceholder("正在分析图像", "模型首次使用时需要加载 checkpoint 和文本编码器，请稍候。");
  try {
    inputs.image = {
      name: state.imageFile.name,
      data_url: await fileToDataUrl(state.imageFile),
    };
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: state.selectedTask.task_id, inputs }),
    });
    const payload = await response.json();
    if (response.ok) {
      renderResult(state.selectedTask.task_id, payload.result);
      const emptyRefseg = state.selectedTask.task_id === "refseg"
        && Number(payload.result?.summary?.foreground_pixels) === 0;
      setMessage(
        emptyRefseg ? "未找到符合描述的目标，请调整描述后重试。" : "任务执行完成。",
        emptyRefseg ? "error" : "success",
      );
      interfaceStatusChip.textContent = "模型已加载";
      interfaceStatusChip.classList.add("ready");
    } else if (payload.status === "interface_pending") {
      setMessage(payload.message, "error");
      resultHelp.textContent = `${state.selectedTask.title}的界面、输入校验和路由已经连通；下一步接入实际模型 Adapter。`;
      renderPlaceholder("该任务暂未接入模型", resultHelp.textContent);
    } else {
      const message = payload.message || payload.error || "任务请求失败。";
      setMessage(message, "error");
      renderPlaceholder("分析未完成", message);
    }
  } catch (error) {
    const message = error instanceof Error && error.message.includes("结果缺少")
      ? error.message
      : "无法完成请求，请确认本地服务正在运行且图像可以读取。";
    setMessage(message, "error");
    renderPlaceholder("分析未完成", message);
  } finally {
    runButton.disabled = false;
  }
});

async function initialize() {
  try {
    const response = await fetch("/api/tasks");
    if (!response.ok) throw new Error("Task API unavailable");
    const payload = await response.json();
    state.tasks = payload.tasks || [];
    taskCount.textContent = `当前已配置 ${state.tasks.length} 项能力 · 更多任务可继续接入`;
    renderTasks();
  } catch (error) {
    taskGrid.innerHTML = '<div class="loading-card">任务配置读取失败，请重新启动本地服务。</div>';
  }
}

initialize();
