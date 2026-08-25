"""场景分类模型综合评测脚本.

一次性测量并输出完整报告, 包含:
- 硬件/软件环境
- 模型参数量与文件大小
- Accuracy (重新跑验证集, 记录测试数/失败数)
- 模型加载时间、单样本端到端时间
- 峰值 GPU 显存 / CPU 内存
- 测试设置 (预热/重复/批大小)

输出: eval_results/benchmark_report.md
"""
import os
import sys
import time
import platform
import subprocess
from datetime import datetime

import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path

from classify_head import ClassifyHeadV2
from train_caption_head import extract_multiscale_features
from train_vrsbench_scene import CONFIG, build_dataloaders
from train_scene_nwpu import load_backbone
from eval_scene_nwpu import get_cpu_name

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

RESULT_DIR = "eval_results"
WARMUP = 10          # 预热次数
REPEAT = 100         # 计时重复次数
BATCH_SIZE = 1       # 单样本批大小


def get_gpu_info():
    if not torch.cuda.is_available():
        return "无 GPU"
    name = torch.cuda.get_device_name(0)
    cuda_ver = torch.version.cuda
    # 驱动版本
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            timeout=5,
        ).decode("gbk", errors="ignore").strip()
        driver = out
    except Exception:
        driver = "未知"
    return f"{name} | 驱动 {driver} | CUDA {cuda_ver}"


def get_cpu_memory():
    if HAS_PSUTIL:
        return f"{psutil.virtual_memory().total / (1024**3):.1f} GB"
    return "未知 (未安装 psutil)"


def model_params_and_size(backbone, classify_head, ckpt_path):
    backbone_params = sum(p.numel() for p in backbone.parameters())
    head_params = sum(p.numel() for p in classify_head.parameters())
    total = backbone_params + head_params
    file_size = os.path.getsize(ckpt_path) / (1024**2)
    return backbone_params, head_params, total, file_size


def measure_load_time():
    """冷启动时间: 从加载 backbone 到分类头就绪."""
    device = torch.device(CONFIG["device"])
    t0 = time.perf_counter()
    backbone = load_backbone(device)
    ckpt = torch.load(os.path.join(CONFIG["save_dir"], CONFIG["save_name"]), map_location=device)
    classify_head = ClassifyHeadV2(
        num_classes=ckpt["num_classes"],
        proj_dim=ckpt.get("proj_dim", 256),
        hidden_dim=ckpt.get("hidden_dim", 512),
        dropout=0.3,
        use_gem=ckpt.get("use_gem", True),
    ).to(device)
    classify_head.load_state_dict(ckpt["classify_head"])
    classify_head.eval()
    load_time = time.perf_counter() - t0
    return backbone, classify_head, ckpt, load_time


def measure_end_to_end(backbone, classify_head, device, transform):
    """单样本端到端时间 (图像加载 → 预处理 → 推理 → 后处理)."""
    from torchvision import datasets
    raw_dataset = datasets.ImageFolder(CONFIG["data_dir"])
    img_path, _ = raw_dataset.samples[0]
    img = Image.open(img_path).convert("RGB")

    # 预热
    for _ in range(WARMUP):
        t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            p3, p4, p5 = extract_multiscale_features(backbone, t)
            classify_head((p3, p4, p5))

    # 计时 (每次完整走一遍: 图像加载 → 预处理 → 推理 → 后处理)
    torch.cuda.synchronize()
    times = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        # 图像加载 + 预处理
        im = Image.open(img_path).convert("RGB")
        im_t = transform(im).unsqueeze(0).to(device)
        # 推理
        with torch.no_grad():
            p3, p4, p5 = extract_multiscale_features(backbone, im_t)
            logits = classify_head((p3, p4, p5))
        # 后处理
        pred = logits.argmax(dim=1)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times.sort()
    avg = sum(times) / len(times)
    p95 = times[int(len(times) * 0.95)]
    return avg, p95


def measure_peak_memory(backbone, classify_head, device, transform):
    """峰值 GPU 显存 / CPU 内存."""
    from torchvision import datasets
    raw_dataset = datasets.ImageFolder(CONFIG["data_dir"])
    img_path, _ = raw_dataset.samples[0]
    img = Image.open(img_path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        p3, p4, p5 = extract_multiscale_features(backbone, t)
        classify_head((p3, p4, p5))

    # 推理后的总显存 = 模型权重 + 输入 + 激活
    peak_gpu = torch.cuda.memory_allocated() / (1024**2)
    if HAS_PSUTIL:
        peak_cpu = psutil.Process().memory_info().rss / (1024**2)
    else:
        peak_cpu = -1
    return peak_gpu, peak_cpu


def evaluate_accuracy(backbone, classify_head, val_loader, device):
    """重新跑验证集, 返回 (accuracy, 测试数, 失败数)."""
    classify_head.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            p3, p4, p5 = extract_multiscale_features(backbone, images)
            logits = classify_head((p3, p4, p5))
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
    return correct / total, total, total - correct


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    device = torch.device(CONFIG["device"])
    print(f"设备: {device}")

    # 1. 硬件环境
    env = {
        "os": platform.platform(),
        "cpu": get_cpu_name(),
        "memory": get_cpu_memory(),
        "gpu": get_gpu_info(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "precision": "FP32",
    }

    # 2. 模型加载 + 加载时间
    print("加载模型并计时...")
    backbone, classify_head, ckpt, load_time = measure_load_time()

    # 3. 参数量 + 文件大小
    backbone_p, head_p, total_p, file_size = model_params_and_size(
        backbone, classify_head, os.path.join(CONFIG["save_dir"], CONFIG["save_name"]))
    print(f"参数量: backbone {backbone_p:,} + head {head_p:,} = {total_p:,}")
    print(f"模型文件: {file_size:.1f} MB")

    # 4. 端到端时间
    transform = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print(f"测量端到端时间 (预热 {WARMUP} 次, 重复 {REPEAT} 次)...")
    avg_time, p95_time = measure_end_to_end(backbone, classify_head, device, transform)
    print(f"平均 {avg_time*1000:.1f} ms, P95 {p95_time*1000:.1f} ms")

    # 5. 峰值显存/内存
    peak_gpu, peak_cpu = measure_peak_memory(backbone, classify_head, device, transform)
    print(f"峰值 GPU 显存: {peak_gpu:.1f} MB")

    # 6. Accuracy (重新跑验证集)
    print("重新跑验证集评估准确率...")
    _, val_loader, num_classes, class_names = build_dataloaders()
    acc, test_count, fail_count = evaluate_accuracy(backbone, classify_head, val_loader, device)
    print(f"Accuracy: {acc:.2%} ({test_count - fail_count}/{test_count})")

    # 7. 写报告
    lines = []
    lines.append("# 场景分类模型综合评测报告\n")
    lines.append(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## 一、硬件与软件环境\n")
    lines.append(f"- 操作系统: {env['os']}")
    lines.append(f"- CPU: {env['cpu']}")
    lines.append(f"- 内存: {env['memory']}")
    lines.append(f"- GPU: {env['gpu']}")
    lines.append(f"- Python: {env['python']}")
    lines.append(f"- 推理框架: PyTorch {env['torch']}")
    lines.append(f"- 计算精度: {env['precision']}")
    lines.append("")
    lines.append("## 二、模型信息\n")
    lines.append(f"- 完整系统参数量: {total_p:,} (backbone {backbone_p:,} + 分类头 {head_p:,})")
    lines.append(f"- 其中可训练参数: {head_p:,} (backbone 冻结)")
    lines.append(f"- 模型文件实际存储占用: {file_size:.1f} MB")
    lines.append("")
    lines.append("## 三、准确率指标\n")
    lines.append(f"- Accuracy: **{acc:.2%}**")
    lines.append(f"- 测试任务数: {test_count}")
    lines.append(f"- 失败数: {fail_count}")
    lines.append("")
    lines.append("## 四、时间指标\n")
    lines.append(f"- 模型加载/冷启动时间: {load_time:.2f} s")
    lines.append(f"- 单样本端到端时间 (平均): {avg_time*1000:.1f} ms")
    lines.append(f"- 单样本端到端时间 (P95): {p95_time*1000:.1f} ms")
    lines.append("- 计时起止点: 从读取图像文件开始, 到输出预测类别结束")
    lines.append("- 端到端包含: 图像预处理(resize+归一化) + 多尺度特征提取 + 分类头推理 + 后处理(argmax)")
    lines.append("")
    lines.append("## 五、资源占用\n")
    lines.append(f"- 峰值 GPU 显存: {peak_gpu:.1f} MB")
    lines.append(f"- 峰值 CPU 内存: {peak_cpu:.1f} MB" if peak_cpu > 0 else "- 峰值 CPU 内存: 未测量 (需安装 psutil)")
    lines.append("")
    lines.append("## 六、测试设置\n")
    lines.append(f"- 预热次数: {WARMUP}")
    lines.append(f"- 计时重复次数: {REPEAT}")
    lines.append(f"- 批大小: {BATCH_SIZE} (单样本)")
    lines.append("- 任务路由: 无 (单任务场景分类, 无需路由)")
    lines.append("- 量化/模型卸载/缓存: 均未使用")
    lines.append("")

    out_path = Path(RESULT_DIR) / "benchmark_report.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n✓ 综合评测报告已保存: {out_path}")


if __name__ == "__main__":
    main()
