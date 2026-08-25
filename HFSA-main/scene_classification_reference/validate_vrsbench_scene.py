"""VRSBench 场景类验证: 只测标签为"场景"的图片, 看 NWPU 模型场景分类准确度.

排除物体类 (vehicle/ship/airplane/helicopter/container-crane),
只测能映射到 NWPU 的 13 个场景类.
"""
import os
import glob
import random
import xml.etree.ElementTree as ET

import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
from collections import defaultdict

from classify_head import ClassifyHeadV2
from train_caption_head import extract_multiscale_features
from train_scene_nwpu import CONFIG, load_backbone

VRSBENCH_JPEG = "dataset/VRSBench/JPEGImages"
VRSBENCH_XML = "dataset/VRSBench/Annotations"

# VRSBench 场景类 → NWPU 类映射 (13 个可映射的场景类)
SCENE_MAP = {
    'airport': 'airport',
    'baseball-diamond': 'baseball_diamond',
    'basketball-court': 'basketball_court',
    'bridge': 'bridge',
    'ground-track-field': 'ground_track_field',
    'harbor': 'harbor',
    'overpass': 'overpass',
    'roundabout': 'roundabout',
    'stadium': 'stadium',
    'storage-tank': 'storage_tank',
    'tennis-court': 'tennis_court',
    'trainstation': 'railway_station',
    'golffield': 'golf_course',
}

SAMPLES_PER_CLASS = 10


def read_xml_labels(xml_path):
    tree = ET.parse(xml_path)
    return [obj.find('name').text.strip() for obj in tree.getroot().iter('object')]


def collect_scene_images():
    """收集每个场景类的单标签图片路径."""
    scene_images = defaultdict(list)
    for xml in glob.glob(os.path.join(VRSBENCH_XML, "*.xml")):
        labels = read_xml_labels(xml)
        if len(set(labels)) == 1:  # 单标签
            lab = labels[0]
            if lab in SCENE_MAP:
                jpg = os.path.join(VRSBENCH_JPEG, Path(xml).stem + ".jpg")
                scene_images[lab].append(jpg)
    return scene_images


def main():
    device = torch.device(CONFIG["device"])
    print(f"设备: {device}")

    backbone = load_backbone(device)
    ckpt = torch.load("checkpoints/scene_nwpu_best.pth", map_location=device)
    class_names = ckpt["classes"]
    classify_head = ClassifyHeadV2(
        num_classes=ckpt["num_classes"],
        proj_dim=ckpt.get("proj_dim", 256),
        hidden_dim=ckpt.get("hidden_dim", 512),
        dropout=0.3,
        use_gem=ckpt.get("use_gem", True),
    ).to(device)
    classify_head.load_state_dict(ckpt["classify_head"])
    classify_head.eval()
    print(f"✓ 已加载 NWPU 分类头 (Val Acc {ckpt.get('val_acc', 0):.2%})")

    transform = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    scene_images = collect_scene_images()
    print(f"\n=== VRSBench 场景类验证 (每类抽 {SAMPLES_PER_CLASS} 张, 共 {len(SCENE_MAP)} 类) ===\n")

    total, top1_hit, top3_hit = 0, 0, 0
    per_class = {}
    random.seed(7)

    for vrsbench_cls, nwpu_cls in sorted(SCENE_MAP.items()):
        images = scene_images.get(vrsbench_cls, [])
        if len(images) == 0:
            print(f"  {vrsbench_cls}: 无单标签图片, 跳过")
            continue
        sample = random.sample(images, min(SAMPLES_PER_CLASS, len(images)))

        cls_top1 = 0
        cls_top3 = 0
        preds_detail = []
        for jpg in sample:
            img = Image.open(jpg).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                p3, p4, p5 = extract_multiscale_features(backbone, img_tensor)
                logits = classify_head((p3, p4, p5))
            top3 = logits.topk(3, dim=1).indices[0].tolist()
            pred_names = [class_names[t] for t in top3]
            preds_detail.append(pred_names[0])

            if pred_names[0] == nwpu_cls:
                cls_top1 += 1
            if nwpu_cls in pred_names:
                cls_top3 += 1

        n = len(sample)
        total += n
        top1_hit += cls_top1
        top3_hit += cls_top3
        per_class[vrsbench_cls] = (cls_top1 / n, cls_top3 / n)
        print(f"  {vrsbench_cls:20s} (→{nwpu_cls:20s}): Top1 {cls_top1}/{n} = {cls_top1/n:.0%}, "
              f"Top3 {cls_top3}/{n} = {cls_top3/n:.0%}")

    print(f"\n=== 总体 (13 个场景类, {total} 张) ===")
    print(f"Top-1 命中率: {top1_hit}/{total} = {top1_hit/total:.2%}")
    print(f"Top-3 命中率: {top3_hit}/{total} = {top3_hit/total:.2%}")


if __name__ == "__main__":
    main()
