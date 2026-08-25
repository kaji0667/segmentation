"""NWPU 场景分类验证脚本.

在验证集上评估并生成:
1. 验证报告 (eval_report.md): 测试环境 + 总体指标 + 每类 F1 + 混淆类别对
2. 可视化成果图 (visualization/scene_predictions.png): 带标注的预测结果
"""
import os
import sys
import random
import subprocess
import platform
from datetime import datetime

import torch
from torchvision import datasets, transforms
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

from classify_head import ClassifyHeadV2
from train_caption_head import extract_multiscale_features
from train_scene_nwpu import CONFIG, load_backbone, build_dataloaders

RESULT_DIR = "eval_results"


# ================== 评估计算 ==================

def compute_confusion_and_metrics(logits, labels, num_classes):
    preds = logits.argmax(dim=1)
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for p, t in zip(preds.tolist(), labels.tolist()):
        conf_matrix[t, p] += 1

    per_class = {}
    for c in range(num_classes):
        tp = conf_matrix[c, c].item()
        fp = conf_matrix[:, c].sum().item() - tp
        fn = conf_matrix[c, :].sum().item() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1}
    return conf_matrix, per_class


def evaluate_full(backbone, classify_head, val_loader, device):
    classify_head.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            p3, p4, p5 = extract_multiscale_features(backbone, images)
            logits = classify_head((p3, p4, p5))
            all_logits.append(logits)
            all_labels.append(labels)
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


# ================== 可视化 ==================

def get_font(size):
    for fp in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"]:
        if Path(fp).exists():
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def annotate_image(img, gt_label, pred_labels):
    img = img.resize((384, 384))
    w, h = img.size
    canvas = Image.new("RGB", (w, h + 88), "white")
    canvas.paste(img, (0, 44))

    draw = ImageDraw.Draw(canvas)
    font_big = get_font(18)
    font_small = get_font(15)

    draw.rectangle([0, 0, w, 44], fill=(0, 128, 0))
    draw.text((10, 12), f"GT: {gt_label}", fill="white", font=font_big)

    draw.rectangle([0, h + 44, w, h + 88], fill=(30, 30, 30))
    draw.text((10, h + 55), f"Pred: {'  '.join(pred_labels)}", fill=(255, 220, 80), font=font_small)
    return canvas


def make_grid(images, cols=5):
    n = len(images)
    rows = (n + cols - 1) // cols
    img_w, img_h = images[0].size
    grid = Image.new("RGB", (img_w * cols, img_h * rows), "white")
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        grid.paste(img, (c * img_w, r * img_h))
    return grid


# ================== 环境信息 ==================

def get_cpu_name():
    """获取 CPU 型号 (Windows 用 PowerShell 查询, 失败则回退到 platform)."""
    try:
        out = subprocess.check_output(
            ["powershell", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            timeout=10,
        ).decode("gbk", errors="ignore").strip()
        return out if out else platform.processor()
    except Exception:
        return platform.processor()


def collect_env_info():
    info = {}
    info["test_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info["python_version"] = sys.version.split()[0]
    info["torch_version"] = torch.__version__
    info["platform"] = platform.platform()
    info["cpu"] = get_cpu_name()
    if torch.cuda.is_available():
        info["device"] = f"cuda ({torch.cuda.get_device_name(0)})"
    else:
        info["device"] = "cpu"
    info["data_dir"] = CONFIG["data_dir"]
    return info


# ================== 报告生成 ==================

def generate_report(env, overall_acc, top5_acc, f1_sorted, confusion_pairs,
                    class_names, ckpt, val_size, vis_results, out_path):
    cfg = ckpt.get("config", CONFIG)  # 优先用训练时保存的配置
    lines = []
    lines.append("# NWPU 场景分类验证报告\n")
    lines.append("## 测试环境\n")
    lines.append(f"- 测试时间: {env['test_time']}")
    lines.append(f"- CPU: {env['cpu']}")
    lines.append(f"- 运行设备: {env['device']}")
    lines.append(f"- 平台: {env['platform']}")
    lines.append(f"- PyTorch: {env['torch_version']}")
    lines.append(f"- Python: {env['python_version']}")
    lines.append(f"- 数据集: {env['data_dir']} (45 类, 31500 张)")
    lines.append(f"- 权重文件: checkpoints/scene_nwpu_best.pth (Epoch {ckpt.get('epoch', '?')})")
    lines.append(f"- 验证集规模: {val_size} 张 (8:2 分层划分)")
    lines.append("")
    lines.append("## 训练条件\n")
    lines.append(f"- 训练轮次: {cfg.get('epochs', '?')} epochs")
    lines.append(f"- 批大小: {cfg.get('batch_size', '?')}")
    lines.append(f"- 学习率: {cfg.get('lr', '?')} (warmup {cfg.get('warmup_epochs', '?')} epochs + cosine 衰减)")
    lines.append(f"- 优化器: AdamW (weight_decay={cfg.get('weight_decay', '?')})")
    lines.append("- 损失函数: CrossEntropyLoss (单标签)")
    lines.append(f"- 梯度裁剪: {cfg.get('grad_clip_norm', '?')}")
    lines.append("- 数据增强: 随机水平/垂直翻转 + 颜色抖动")
    lines.append("- 混淆类重采样: 15 个 F1 低的类加权采样")
    lines.append("- Backbone: 冻结 (YOLOv12m 预训练)")
    lines.append("- 分类头: ClassifyHeadV2 (可训练, ~131 万参数)")
    lines.append("")
    lines.append("## 总体结果\n")
    lines.append(f"- 总体准确率: **{overall_acc:.2%}**")
    lines.append(f"- Top-5 准确率: **{top5_acc:.2%}**")
    lines.append("")
    lines.append("## F1 最低的 10 个类 (容易混淆)\n")
    lines.append("| 类别 | F1 | Precision | Recall |")
    lines.append("|------|-----|-----------|--------|")
    for c, m in f1_sorted[:10]:
        lines.append(f"| {class_names[c]} | {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} |")
    lines.append("")
    lines.append("## F1 最高的 10 个类\n")
    lines.append("| 类别 | F1 | Precision | Recall |")
    lines.append("|------|-----|-----------|--------|")
    for c, m in f1_sorted[-10:]:
        lines.append(f"| {class_names[c]} | {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} |")
    lines.append("")
    lines.append("## 最常混淆的类别对 (前 10)\n")
    lines.append("| 真实类别 | 误判为 | 次数 |")
    lines.append("|---------|--------|------|")
    for src, dst, cnt in confusion_pairs[:10]:
        lines.append(f"| {src} | {dst} | {cnt} |")
    lines.append("")
    lines.append("## 可视化成果图\n")
    lines.append("图片: `scene_predictions.png`\n")
    lines.append("| # | 真实标签 | 预测 Top-3 | 是否正确 |")
    lines.append("|---|---------|-----------|---------|")
    for i, (gt, top3, correct) in enumerate(vis_results, 1):
        mark = "✓" if correct else "✗"
        lines.append(f"| {i} | {gt} | {', '.join(top3)} | {mark} |")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n✓ 验证报告已保存: {out_path}")


# ================== 主流程 ==================

def main():
    device = torch.device(CONFIG["device"])
    print(f"设备: {device}")

    env = collect_env_info()
    backbone = load_backbone(device)
    _, val_loader, num_classes, class_names = build_dataloaders()

    ckpt_path = Path(CONFIG["save_dir"]) / "scene_nwpu_best.pth"
    if not ckpt_path.exists():
        print(f"错误: 找不到权重文件 {ckpt_path}")
        return
    ckpt = torch.load(ckpt_path, map_location=device)
    classify_head = ClassifyHeadV2(
        num_classes=num_classes,
        proj_dim=ckpt.get("proj_dim", 256),
        hidden_dim=ckpt.get("hidden_dim", 512),
        dropout=0.3,
        use_gem=ckpt.get("use_gem", True),
    ).to(device)
    classify_head.load_state_dict(ckpt["classify_head"])
    classify_head.eval()
    print(f"✓ 已加载分类头 (Epoch {ckpt.get('epoch', '?')}, Val Acc {ckpt.get('val_acc', 0):.2%})")

    print("\n正在验证集上评估...")
    all_logits, all_labels = evaluate_full(backbone, classify_head, val_loader, device)

    preds = all_logits.argmax(dim=1)
    overall_acc = (preds == all_labels).float().mean().item()
    top5 = all_logits.topk(5, dim=1).indices
    top5_correct = sum(l in top5[i] for i, l in enumerate(all_labels.tolist()))
    top5_acc = top5_correct / len(all_labels)

    print(f"\n=== 总体 Accuracy: {overall_acc:.2%} ===")
    print(f"=== Top-5 Accuracy: {top5_acc:.2%} ===")

    conf_matrix, per_class = compute_confusion_and_metrics(all_logits, all_labels, num_classes)
    f1_sorted = sorted(per_class.items(), key=lambda x: x[1]["f1"])

    print("\n=== F1 最低的 10 个类 (容易混淆) ===")
    for c, m in f1_sorted[:10]:
        print(f"  {class_names[c]:25s} F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}")

    confusion_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and conf_matrix[i, j].item() > 0:
                confusion_pairs.append((class_names[i], class_names[j], conf_matrix[i, j].item()))
    confusion_pairs.sort(key=lambda x: -x[2])
    print("\n=== 最常混淆的类别对 (前 10) ===")
    for src, dst, cnt in confusion_pairs[:10]:
        print(f"  {src} → {dst}: {cnt} 次")

    # 可视化: 从验证集抽 10 张, 生成带标注的图片
    print("\n=== 生成可视化成果图 ===")
    os.makedirs(RESULT_DIR, exist_ok=True)
    raw_dataset = datasets.ImageFolder(CONFIG["data_dir"])
    val_dataset = val_loader.dataset
    random.seed(1234)
    sample_indices = random.sample(range(len(val_dataset)), 10)
    transform = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    images = []
    vis_results = []
    for idx in sample_indices:
        orig_idx = val_dataset.indices[idx]
        img_path, true_label = raw_dataset.samples[orig_idx]
        img = raw_dataset.loader(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            p3, p4, p5 = extract_multiscale_features(backbone, img_tensor)
            logits = classify_head((p3, p4, p5))
        top3 = logits.topk(3, dim=1).indices[0].tolist()
        pred_labels = [class_names[t] for t in top3]
        gt_name = class_names[true_label]
        correct = top3[0] == true_label
        images.append(annotate_image(img, gt_name, pred_labels))
        vis_results.append((gt_name, pred_labels, correct))

    grid = make_grid(images, cols=5)
    vis_path = Path(RESULT_DIR) / "scene_predictions.png"
    grid.save(vis_path)
    print(f"✓ 可视化成果图已保存: {vis_path}")

    # 生成报告
    generate_report(env, overall_acc, top5_acc, f1_sorted, confusion_pairs,
                    class_names, ckpt, len(val_dataset), vis_results,
                    str(Path(RESULT_DIR) / "eval_report.md"))


if __name__ == "__main__":
    main()
