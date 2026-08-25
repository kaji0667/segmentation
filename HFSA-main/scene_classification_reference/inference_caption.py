import json
import torch
import torch.nn.functional as F
import random
from PIL import Image
from torchvision import transforms
from pathlib import Path
from text_encoder.model import TextGuidedDetectionModel
from train_caption_head import build_category_vocab, extract_multiscale_features
from classify_head import ClassifyHeadV2

INFERENCE_CONFIG = {
    "json_path": "train_caption_annotation.json",
    "backbone_weights": "pretrain_model/yolov12m.pt",
    "classify_head_weights": "./checkpoints/scene_classify_best.pth",
    "img_size": 640,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def load_model():
    device = torch.device(INFERENCE_CONFIG["device"])
    print(f"正在加载模型至设备: {device}...")

    backbone = TextGuidedDetectionModel(cfg="yolov12m.yaml")
    if Path(INFERENCE_CONFIG["backbone_weights"]).exists():
        ckpt = torch.load(INFERENCE_CONFIG["backbone_weights"], map_location="cpu", weights_only=False)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        if hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()
        model_state = backbone.state_dict()
        filtered = {k: v for k, v in state_dict.items()
                    if k in model_state and model_state[k].shape == v.shape}
        skipped = len(state_dict) - len(filtered)
        backbone.load_state_dict(filtered, strict=False)
        print(f"✓ 已加载骨干权重 ({len(filtered)} 个参数, 跳过 {skipped} 个 shape 不匹配)")
    backbone.to(device)
    backbone.eval()

    _, cat_to_idx = build_category_vocab(INFERENCE_CONFIG["json_path"])
    idx_to_cat = {i: c for c, i in cat_to_idx.items()}
    num_classes = len(cat_to_idx)
    print(f"类别数: {num_classes}")

    classify_head_ckpt = torch.load(INFERENCE_CONFIG["classify_head_weights"], map_location=device)
    classify_head = ClassifyHeadV2(
        num_classes=num_classes,
        proj_dim=classify_head_ckpt.get("proj_dim", 256),
        hidden_dim=classify_head_ckpt.get("hidden_dim", 512),
        dropout=0.3,
        use_gem=classify_head_ckpt.get("use_gem", True),
    ).to(device)
    classify_head.load_state_dict(classify_head_ckpt["classify_head"])
    classify_head.eval()
    print(f"✓ 已加载分类头权重 (Epoch {classify_head_ckpt.get('epoch', '?')}, F1 {classify_head_ckpt.get('f1', 0):.4f})")

    return backbone, classify_head, idx_to_cat, device


@torch.no_grad()
def classify_image(backbone, classify_head, image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    p3, p4, p5 = extract_multiscale_features(backbone, img_tensor)
    logits = classify_head((p3, p4, p5))
    probs = torch.sigmoid(logits).squeeze(0)
    return probs


def run_random_validation():
    if not Path(INFERENCE_CONFIG["json_path"]).exists():
        print("错误: 找不到数据集文件")
        return

    with open(INFERENCE_CONFIG["json_path"], "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = random.sample(data, min(8, len(data)))
    backbone, classify_head, idx_to_cat, device = load_model()

    transform = transforms.Compose([
        transforms.Resize((INFERENCE_CONFIG["img_size"], INFERENCE_CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 诊断: 检查 P5 特征是否可区分
    print("\n" + "=" * 50)
    print("P5 特征诊断")
    p5_list = []
    for item in samples[:5]:
        img_tensor = transform(Image.open(item["image_path"]).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            _, _, p5 = extract_multiscale_features(backbone, img_tensor)
        p5_pooled = F.adaptive_avg_pool2d(p5, (1, 1)).flatten(1)
        p5_list.append(p5_pooled)
    p5_tensor = torch.cat(p5_list, dim=0)  # [5, 512]
    p5_normed = F.normalize(p5_tensor, dim=1)
    cos_mat = torch.mm(p5_normed, p5_normed.T)
    print("P5 余弦相似度矩阵:")
    print(torch.round(cos_mat, decimals=4).detach().cpu().numpy())
    off_diag = cos_mat[~torch.eye(5, dtype=torch.bool, device=cos_mat.device)]
    print(f"平均相似度: {off_diag.mean().item():.4f}" +
          (" ✓ 特征可区分" if off_diag.mean().item() < 0.9 else " ⚠ 特征退化"))

    print("\n" + "=" * 50)
    print("场景分类推理验证")
    print("=" * 50)

    for i, item in enumerate(samples):
        img_path = item["image_path"]
        gt_caption = item["caption"]
        print(f"\n样本 {i+1}:")
        print(f"真实标签 (GT): {gt_caption}")

        try:
            probs = classify_image(backbone, classify_head, img_path, transform, device)
            top3 = probs.topk(min(3, len(probs)))
            top3_str = ", ".join(
                f"{idx_to_cat[j.item()]} ({top3.values[i].item():.2%})"
                for i, j in enumerate(top3.indices)
            )
            print(f"预测 Top-3: {top3_str}")
        except Exception as e:
            print(f"预测失败: {e}")


if __name__ == "__main__":
    run_random_validation()
