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
const outputTags = document.querySelector("#output-tags");
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

runButton.addEventListener("click", async () => {
  if (!state.selectedTask) {
    setMessage("请先选择一项分析任务。", "error");
    return;
  }
  if (!state.imageFile) {
    setMessage("请先上传一张遥感图像。", "error");
    return;
  }

  const inputs = { image: state.imageFile.name };
  for (const field of state.selectedTask.inputs.filter((item) => item.kind !== "image")) {
    const value = document.querySelector(`[data-field-key="${field.key}"]`)?.value.trim() || "";
    if (field.required && !value) {
      setMessage(`请填写“${field.label}”。`, "error");
      return;
    }
    inputs[field.key] = value;
  }

  runButton.disabled = true;
  setMessage("正在检查任务路由…");
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: state.selectedTask.task_id, inputs }),
    });
    const payload = await response.json();
    if (response.ok) {
      setMessage("任务执行完成。", "success");
    } else if (payload.status === "interface_pending") {
      setMessage(payload.message, "success");
      resultHelp.textContent = `${state.selectedTask.title}的界面、输入校验和路由已经连通；下一步接入实际模型 Adapter。`;
    } else {
      setMessage(payload.error || "任务请求失败。", "error");
    }
  } catch (error) {
    setMessage("无法连接本地界面服务，请确认 web_app.py 正在运行。", "error");
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
