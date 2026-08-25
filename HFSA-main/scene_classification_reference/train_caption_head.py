import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from text_encoder.model import TextGuidedDetectionModel
from classify_head import ClassifyHeadV2, AsymmetricLoss

# ================== 配置 ==================
CONFIG = {
    "json_path": "train_caption_annotation.json",
    "backbone_weights": "pretrain_model/yolov12m.pt",
    "save_dir": "./checkpoints",
    "batch_size": 32,
    "epochs": 8,
    "lr": 2e-3,
    "img_size": 640,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # ClassifyHeadV2
    "proj_dim": 256,
    "hidden_dim": 512,
    "head_dropout": 0.3,
    "use_gem": True,
    # ASL Loss
    "asl_gamma_pos": 1.0,
    "asl_gamma_neg": 4.0,
    "asl_clip": 0.05,
    # Training
    "weight_decay": 1e-4,
    "warmup_epochs": 2,
    "grad_clip_norm": 1.0,
}


# ================== 类别解析 ==================

def extract_categories(caption):
    prefix = "An aerial remote sensing view containing "
    suffix = ", and the surrounding"
    if prefix in caption and suffix in caption:
        start = len(prefix)
        end = caption.index(suffix)
        middle = caption[start:end]
        return [c.strip() for c in middle.split(",") if c.strip()]
    return []


def build_category_vocab(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_cats = set()
    for item in data:
        cats = extract_categories(item["caption"])
        all_cats.update(cats)
    vocab = sorted(all_cats)
    cat_to_idx = {cat: i for i, cat in enumerate(vocab)}
    return vocab, cat_to_idx


# ================== 自定义分类头 ==================

class ClassifyHead(nn.Module):
    """3 层 MLP 分类头: 512 -> hidden -> hidden//2 -> 26"""
    def __init__(self, in_dim=512, hidden_dim=1024, num_classes=26, dropout=0.3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.pool(x)          # [B, 512, 1, 1]
        x = x.flatten(1)          # [B, 512]
        return self.fc(x)         # [B, num_classes]


def extract_multiscale_features(backbone, images):
    """提取 P3, P4, P5 多尺度特征图, 不经过 Detect 头.

    Returns:
        (p3, p4, p5): [B,256,80,80], [B,512,40,40], [B,512,20,20]
    """
    if images.dim() == 3:
        images = images.unsqueeze(0)

    y = []
    x = images
    for m in backbone.model[:-1]:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y.append(x if m.i in backbone.save else None)

    head = backbone.model[-1]
    return y[head.f[0]], y[head.f[1]], y[head.f[2]]


# ================== 数据集 ==================

class SceneClassificationDataset(Dataset):
    def __init__(self, json_path, transform, cat_to_idx):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.transform = transform
        self.cat_to_idx = cat_to_idx
        self.num_classes = len(cat_to_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            image = Image.open(item["image_path"]).convert("RGB")
            image = self.transform(image)
        except Exception:
            image = torch.zeros((3, CONFIG["img_size"], CONFIG["img_size"]))

        cats = extract_categories(item["caption"])
        multi_hot = torch.zeros(self.num_classes)
        for c in cats:
            if c in self.cat_to_idx:
                multi_hot[self.cat_to_idx[c]] = 1.0

        return image, multi_hot


# ================== Loss ==================

def build_balanced_sampler(dataset, cat_to_idx, num_classes, json_path, beta=0.9999):
    """基于 effective number weighting 的类别平衡采样器."""
    class_counts = torch.zeros(num_classes)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        cats = extract_categories(item["caption"])
        for c in cats:
            if c in cat_to_idx:
                class_counts[cat_to_idx[c]] += 1

    effective_num = 1.0 - beta ** class_counts
    class_weights = (1.0 - beta) / effective_num.clamp_min(1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes

    sample_weights = []
    for item in data:
        cats = extract_categories(item["caption"])
        multi_hot = torch.zeros(num_classes)
        for c in cats:
            if c in cat_to_idx:
                multi_hot[cat_to_idx[c]] = 1.0
        pos_mask = multi_hot > 0
        sample_weights.append(class_weights[pos_mask].mean().item() if pos_mask.any() else 0.0)

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return sampler


def compute_metrics(logits, labels):
    """多标签评测指标."""
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        tp = (preds * labels).sum(dim=0)
        fp = (preds * (1 - labels)).sum(dim=0)
        fn = ((1 - preds) * labels).sum(dim=0)
        prec = tp / (tp + fp).clamp_min(1e-8)
        rec = tp / (tp + fn).clamp_min(1e-8)
        f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-8)
        return f1.mean().item()


# ================== 训练 ==================

def train():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    device = torch.device(CONFIG["device"])

    # 1. 类别词表
    vocab, cat_to_idx = build_category_vocab(CONFIG["json_path"])
    num_classes = len(vocab)
    print(f"类别数: {num_classes}")
    print(f"类别: {vocab}")

    # 2. 加载 backbone
    print("正在加载 TextGuidedDetectionModel...")
    backbone = TextGuidedDetectionModel(cfg="yolov12m.yaml")
    if Path(CONFIG["backbone_weights"]).exists():
        ckpt = torch.load(CONFIG["backbone_weights"], map_location="cpu", weights_only=False)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        if hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()
        # 只加载 shape 匹配的参数, 跳过 Detect 头 (nc=80 vs nc=1) 等 shape 不匹配的 key
        model_state = backbone.state_dict()
        filtered = {k: v for k, v in state_dict.items()
                    if k in model_state and model_state[k].shape == v.shape}
        skipped = len(state_dict) - len(filtered)
        backbone.load_state_dict(filtered, strict=False)
        print(f"✓ 已加载骨干权重 ({len(filtered)} 个参数, 跳过 {skipped} 个 shape 不匹配)")

    # 冻结 backbone
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.to(device)
    backbone.eval()

    # 3. 分类头 V2 (多尺度 + 注意力池化 + GeM)
    classify_head = ClassifyHeadV2(
        num_classes=num_classes,
        proj_dim=CONFIG["proj_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        dropout=CONFIG["head_dropout"],
        use_gem=CONFIG["use_gem"],
    ).to(device)
    trainable = sum(p.numel() for p in classify_head.parameters())
    print(f"可训练参数: {trainable:,} (ClassifyHeadV2)")

    # 4. 损失函数
    criterion = AsymmetricLoss(
        gamma_neg=CONFIG["asl_gamma_neg"],
        gamma_pos=CONFIG["asl_gamma_pos"],
        clip=CONFIG["asl_clip"],
    ).to(device)

    # 5. 数据 + 平衡采样
    transform = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = SceneClassificationDataset(CONFIG["json_path"], transform, cat_to_idx)
    sampler = build_balanced_sampler(dataset, cat_to_idx, num_classes, CONFIG["json_path"])
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"],
                            sampler=sampler, num_workers=2)

    # 6. 优化器 + warmup + cosine decay
    optimizer = torch.optim.AdamW(
        classify_head.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    total_steps = len(dataloader) * CONFIG["epochs"]
    warmup_steps = len(dataloader) * CONFIG["warmup_epochs"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step = 0
    best_f1 = 0.
    for epoch in range(CONFIG["epochs"]):
        classify_head.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

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
            f1 = compute_metrics(logits, labels)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "f1": f"{f1:.3f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} 完成, Loss: {avg_loss:.4f}")

        # 每 epoch 结束做一次简单的验证评估
        classify_head.eval()
        val_loss = 0
        val_f1 = 0
        val_count = 0
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                p3, p4, p5 = extract_multiscale_features(backbone, images)
                logits = classify_head((p3, p4, p5))
                val_loss += criterion(logits, labels).item()
                val_f1 += compute_metrics(logits, labels)
                val_count += 1
                if val_count >= 20:  # 只评估前 20 个 batch 节省时间
                    break
        val_loss /= val_count
        val_f1 /= val_count
        print(f"  验证 Loss: {val_loss:.4f}, F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "classify_head": classify_head.state_dict(),
                "cat_to_idx": cat_to_idx,
                "epoch": epoch,
                "loss": avg_loss,
                "f1": val_f1,
                "proj_dim": CONFIG["proj_dim"],
                "hidden_dim": CONFIG["hidden_dim"],
                "use_gem": CONFIG["use_gem"],
            }, os.path.join(CONFIG["save_dir"], "scene_classify_best.pth"))
            print(f"  ✓ 已保存最佳权重 (F1={val_f1:.4f})")

    print(f"训练结束! 最佳 F1: {best_f1:.4f}")


if __name__ == "__main__":
    train()
