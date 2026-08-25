"""从 VRSBench 筛选含场景标签的图像, 汇总到 VRSBench_scene/ 按主场景类归档.

筛选规则: 一张图的标签去重后, 若包含至少一个场景类, 则选中.
主场景类 = 该图去重后场景类列表排序取第一个 (确定性).
排除纯物体图 (vehicle/ship/airplane/helicopter/container-crane).
"""
import os
import glob
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

VRSBENCH_JPEG = "dataset/VRSBench/JPEGImages"
VRSBENCH_XML = "dataset/VRSBench/Annotations"
OUT_DIR = "VRSBench_scene"

OBJECT_CLASSES = {'vehicle', 'ship', 'airplane', 'helicopter', 'container-crane'}
SCENE_CLASSES = {
    'airport', 'baseball-diamond', 'basketball-court', 'bridge', 'chimney', 'dam',
    'expressway-service-area', 'expressway-toll-station', 'golffield', 'ground-track-field',
    'harbor', 'helipad', 'overpass', 'roundabout', 'soccer-ball-field', 'stadium',
    'storage-tank', 'swimming-pool', 'tennis-court', 'trainstation', 'windmill',
}


def read_xml_labels(xml_path):
    tree = ET.parse(xml_path)
    return [obj.find('name').text.strip() for obj in tree.getroot().iter('object')]


def main():
    # 清空旧的输出目录 (避免残留上一次的噪声图)
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    # 收集每个主场景类的图片路径 (只保留恰好 1 个场景类的图)
    class_images = defaultdict(list)
    dropped_multi_scene = 0
    dropped_pure_object = 0
    for xml in glob.glob(os.path.join(VRSBENCH_XML, "*.xml")):
        labels = read_xml_labels(xml)
        scene_in = sorted(set(labels) & SCENE_CLASSES)
        if len(scene_in) == 1:
            main_cls = scene_in[0]
            jpg = os.path.join(VRSBENCH_JPEG, Path(xml).stem + ".jpg")
            if os.path.exists(jpg):
                class_images[main_cls].append(jpg)
        elif len(scene_in) >= 2:
            dropped_multi_scene += 1
        else:
            dropped_pure_object += 1

    print(f"丢弃 2+ 场景类图 (标签噪声): {dropped_multi_scene} 张")
    print(f"丢弃纯物体图: {dropped_pure_object} 张")

    # 复制到新文件夹
    total = 0
    print(f"开始汇总到 {OUT_DIR}/ ...")
    for cls in sorted(class_images.keys()):
        out_dir = os.path.join(OUT_DIR, cls)
        os.makedirs(out_dir, exist_ok=True)
        for jpg in class_images[cls]:
            shutil.copy(jpg, os.path.join(out_dir, Path(jpg).name))
        total += len(class_images[cls])
        print(f"  {cls:28s}: {len(class_images[cls])} 张")

    print(f"\n✓ 汇总完成: 共 {total} 张, {len(class_images)} 个场景类, 输出到 {OUT_DIR}/")


if __name__ == "__main__":
    main()
