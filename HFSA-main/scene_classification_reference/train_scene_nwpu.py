"""NWPU-RESISC45 单标签场景分类训练脚本.

架构: 冻结 YOLOv12m backbone → 多尺度特征(P3/P4/P5) → ClassifyHeadV2 → CrossEntropyLoss
数据: NWPU-RESISC45 (45 类 × 700 张, 256×256), 按 8:2 分层划分 train/val
"""
import os
import math
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm
from pathlib import Path

from text_encoder.model import TextGuidedDetectionModel
from classify_head import ClassifyHeadV2
from train_caption_head import extract_multiscale_features

CONFIG = {
    "data_dir": "NWPU-RESISC45",
    "backbone_weights": "pretrain_model/yolov12m.pt",
    "save_dir": "./checkpoints",
    "batch_size": 32,
    "epochs": 35,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "warmup_epochs": 2,
    "grad_clip_norm": 1.0,
    "img_size": 640,
    "val_ratio": 0.2,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def load_backbone(device):
    """加载并冻结预训练 backbone."""
    print("正在加载 TextGuidedDetectionModel (YOLOv12m)...")
    backbone = TextGuidedDetectionModel(cfg="yolov12m.yaml")
    if Path(CONFIG["backbone_weights"]).exists():
        ckpt = torch.load(CONFIG["backbone_weights"], map_location="cpu", weights_only=False)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        if hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()
        model_state = backbone.state_dict()
        filtered = {k: v for k, v in state_dict.items()
                    if k in model_state and model_state[k].shape == v.shape}
        skipped = len(state_dict) - len(filtered)
        backbone.load_state_dict(filtered, strict=False)
        print(f"✓ 已加载骨干权重 ({len(filtered)} 参数, 跳过 {skipped} 个 shape 不匹配)")
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.to(device)
    backbone.eval()
    return backbone


def build_dataloaders():
    """加载 NWPU 数据并按类分层划分 train/val.

    训练集带数据增强 + 混淆类重采样; 验证集无增强.
    """
    # 训练 transform: 镜像 + 颜色抖动 (缓解过拟合)
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # 验证 transform: 无增强
    val_transform = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_full = datasets.ImageFolder(CONFIG["data_dir"], transform=train_transform)
    val_full = datasets.ImageFolder(CONFIG["data_dir"], transform=val_transform)
    num_classes = len(train_full.classes)
    print(f"类别数: {num_classes}")
    print(f"总样本数: {len(train_full)}")

    # 按类别分层划分 (两个 ImageFolder 的样本顺序一致, 可共享索引)
    random.seed(CONFIG["seed"])
    train_idx, val_idx = [], []
    for cls_id in range(num_classes):
        cls_indices = [i for i, t in enumerate(train_full.targets) if t == cls_id]
        random.shuffle(cls_indices)
        n_val = int(len(cls_indices) * CONFIG["val_ratio"])
        val_idx.extend(cls_indices[:n_val])
        train_idx.extend(cls_indices[n_val:])

    random.shuffle(train_idx)
    random.shuffle(val_idx)

    train_set = Subset(train_full, train_idx)
    val_set = Subset(val_full, val_idx)
    print(f"训练集: {len(train_set)}, 验证集: {len(val_set)}")

    # 混淆类重采样: 给 F1 低的类更高采样权重 (基于 eval 混淆矩阵)
    hard_class_weight = {
        'palace': 2.0, 'church': 2.0, 'railway_station': 1.8,
        'freeway': 1.8, 'airport': 1.5, 'river': 1.5,
        'medium_residential': 1.5, 'commercial_area': 1.3,
        'industrial_area': 1.3, 'railway': 1.5,
        'basketball_court': 1.3, 'tennis_court': 1.3,
        'rectangular_farmland': 1.3, 'terrace': 1.3,
        'dense_residential': 1.3,
    }
    sample_weights = []
    for idx in train_idx:
        cls_name = train_full.classes[train_full.targets[idx]]
        sample_weights.append(hard_class_weight.get(cls_name, 1.0))
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"],
                              sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=CONFIG["batch_size"],
                            shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, num_classes, train_full.classes


def evaluate(backbone, classify_head, loader, criterion, device):
    """验证集评估, 返回 loss 和 accuracy."""
    classify_head.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            p3, p4, p5 = extract_multiscale_features(backbone, images)
            logits = classify_head((p3, p4, p5))
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
    classify_head.train()
    return total_loss / total, correct / total


def train():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    device = torch.device(CONFIG["device"])

    backbone = load_backbone(device)
    train_loader, val_loader, num_classes, class_names = build_dataloaders()

    classify_head = ClassifyHeadV2(
        num_classes=num_classes,
        proj_dim=256,
        hidden_dim=512,
        dropout=0.3,
        use_gem=True,
    ).to(device)
    trainable = sum(p.numel() for p in classify_head.parameters())
    print(f"可训练参数: {trainable:,} (ClassifyHeadV2)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classify_head.parameters(), lr=CONFIG["lr"],
                                  weight_decay=CONFIG["weight_decay"])

    total_steps = len(train_loader) * CONFIG["epochs"]
    warmup_steps = len(train_loader) * CONFIG["warmup_epochs"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_acc = 0.0
    for epoch in range(CONFIG["epochs"]):
        classify_head.train()
        total_loss, step = 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                p3, p4, p5 = extract_multiscale_features(backbone, images)

            logits = classify_head((p3, p4, p5))
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classify_head.parameters(),
                                           CONFIG["grad_clip_norm"])
            optimizer.step()
            scheduler.step()
            step += 1

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        val_loss, val_acc = evaluate(backbone, classify_head, val_loader, criterion, device)
        print(f"Epoch {epoch+1} 完成 | Train Loss: {avg_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "classify_head": classify_head.state_dict(),
                "classes": class_names,
                "num_classes": num_classes,
                "epoch": epoch,
                "val_acc": val_acc,
                "proj_dim": 256,
                "hidden_dim": 512,
                "use_gem": True,
                "config": CONFIG,
            }, os.path.join(CONFIG["save_dir"], "scene_nwpu_best.pth"))
            print(f"  ✓ 已保存最佳权重 (Val Acc={val_acc:.2%})")

    print(f"训练结束! 最佳验证准确率: {best_acc:.2%}")


if __name__ == "__main__":
    train()
