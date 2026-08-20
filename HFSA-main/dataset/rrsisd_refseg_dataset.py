from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_DIRECTIONAL_TEXT_RE = re.compile(
    r"\b("
    r"left|right|top|bottom|upper|lower|middle|center|centre|"
    r"north|south|east|west|northeast|northwest|southeast|southwest|"
    r"leftmost|rightmost|topmost|bottommost"
    r")\b",
    re.IGNORECASE,
)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _yaml_scalar(value: Any) -> str:
    text = str(value)
    if not text or any(ch in text for ch in ":#[]{}&,*!|>'\"%@`"):
        return json.dumps(text, ensure_ascii=False)
    return text


def _write_data_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"dataset_type: {data['dataset_type']}",
        f"path: {_yaml_scalar(data['path'])}",
        f"train: {data['train']}",
        f"val: {data['val']}",
        f"test: {data['test']}",
        f"nc: {data['nc']}",
        "names:",
    ]
    lines.extend(f"  {i}: {_yaml_scalar(name)}" for i, name in enumerate(data["names"]))
    lines.extend(
        [
            f"ignore_index: {data['ignore_index']}",
            "mask_value_mapping: rrsisd_refseg_binary_0_1",
            f"source_root: {_yaml_scalar(data['source_root'])}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_jsonl_rows(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _has_directional_text(text: str) -> bool:
    return bool(_DIRECTIONAL_TEXT_RE.search(str(text)))


def _load_refs(path: Path) -> List[Dict[str, Any]]:
    with path.open("rb") as f:
        refs = pickle.load(f)
    if not isinstance(refs, list):
        raise TypeError(f"RRSIS-D refs must be a list, got {type(refs)}")
    return refs


def _first_sentence(ref: Dict[str, Any]) -> str:
    sentences = ref.get("sentences", [])
    if not sentences:
        return ""
    first = sentences[0]
    if isinstance(first, dict):
        return str(first.get("sent") or first.get("raw") or "").strip()
    return str(first).strip()


def _category_names(categories: Sequence[Dict[str, Any]]) -> List[str]:
    if not categories:
        raise ValueError("RRSIS-D instances.json has no categories.")
    max_id = max(int(cat["id"]) for cat in categories)
    names = [str(i) for i in range(max_id + 1)]
    for cat in categories:
        names[int(cat["id"])] = str(cat["name"])
    return names


def collect_rrsisd_rows(root: str | Path, split: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    root = Path(root).expanduser().resolve()
    split = str(split).strip().lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported RRSIS-D split: {split}")

    refs_path = root / "rrsisd" / "refs(unc).p"
    instances_path = root / "rrsisd" / "instances.json"
    image_dir = root / "images" / "rrsisd" / "JPEGImages"
    if not refs_path.exists():
        raise FileNotFoundError(f"Missing RRSIS-D refs file: {refs_path}")
    if not instances_path.exists():
        raise FileNotFoundError(f"Missing RRSIS-D instances file: {instances_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing RRSIS-D image directory: {image_dir}")

    instances = json.loads(instances_path.read_text(encoding="utf-8"))
    names = _category_names(instances.get("categories", []))
    annotations = instances.get("annotations", [])
    ann_by_id = {int(ann["id"]): ann for ann in annotations}
    images = instances.get("images", [])
    image_by_id = {int(img["id"]): img for img in images}

    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for ref in _load_refs(refs_path):
        if str(ref.get("split", "")).lower() != split:
            continue
        ann_id = int(ref["ann_id"])
        image_id = int(ref["image_id"])
        ann = ann_by_id.get(ann_id)
        image_info = image_by_id.get(image_id, {})
        if ann is None:
            missing.append(f"ann_id={ann_id}")
            continue

        file_name = str(ref.get("file_name") or image_info.get("file_name") or f"{image_id:05d}.jpg")
        image_path = image_dir / file_name
        if not image_path.exists():
            missing.append(str(image_path))
            continue

        class_idx = int(ref.get("category_id", ann.get("categories_id", 0)))
        text = _first_sentence(ref)
        if not text:
            text = names[class_idx] if 0 <= class_idx < len(names) else str(class_idx)

        rows.append(
            {
                "id": f"{split}_{int(ref.get('ref_id', ann_id))}",
                "split": split,
                "image": str(image_path),
                "file_name": file_name,
                "image_id": image_id,
                "ann_id": ann_id,
                "ref_id": int(ref.get("ref_id", ann_id)),
                "text": text,
                "category_id": class_idx,
                "class_idx": class_idx,
                "category_name": names[class_idx] if 0 <= class_idx < len(names) else str(class_idx),
                "class_name": names[class_idx] if 0 <= class_idx < len(names) else str(class_idx),
                "height": int(image_info.get("height", 0) or ann.get("height", 0) or 0),
                "width": int(image_info.get("width", 0) or ann.get("width", 0) or 0),
                "bbox": ann.get("bbox", []),
                "segmentation": ann.get("segmentation", []),
                "mask_value_mapping": "RLE foreground -> 1; background -> 0",
            }
        )

    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} RRSIS-D item(s) for split {split}. First entries: {preview}")
    return rows, names


def prepare_rrsisd_refseg_dataset(
    root: str | Path = "data/RRSIS-D",
    output_dir: str | Path = "pre_datasets/RRSIS-D_refseg",
) -> Dict[str, Any]:
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    rows_by_split: Dict[str, List[Dict[str, Any]]] = {}
    names: List[str] = []
    for split in ("train", "val", "test"):
        rows, split_names = collect_rrsisd_rows(root, split)
        rows_by_split[split] = rows
        names = split_names
        _write_jsonl(output_dir / f"{split}.jsonl", rows)

    data = {
        "dataset_type": "rrsisd_refseg",
        "path": str(output_dir),
        "train": "train.jsonl",
        "val": "val.jsonl",
        "test": "test.jsonl",
        "nc": len(names),
        "names": names,
        "ignore_index": 255,
        "source_root": str(root),
    }
    _write_data_yaml(output_dir / "data.yaml", data)

    return {
        "root": str(root),
        "output_dir": str(output_dir),
        "data_yaml": str(output_dir / "data.yaml"),
        "train": len(rows_by_split["train"]),
        "val": len(rows_by_split["val"]),
        "test": len(rows_by_split["test"]),
        "classes": names,
    }


def decode_compressed_rle_counts(counts: str | Sequence[int]) -> List[int]:
    if isinstance(counts, (list, tuple)):
        return [int(x) for x in counts]
    if not isinstance(counts, str):
        raise TypeError(f"Unsupported RLE counts type: {type(counts)}")

    decoded: List[int] = []
    p = 0
    while p < len(counts):
        value = 0
        shift = 0
        while True:
            c = ord(counts[p]) - 48
            p += 1
            value |= (c & 0x1F) << shift
            shift += 5
            if not (c & 0x20):
                if c & 0x10:
                    value |= -1 << shift
                break
        if len(decoded) > 2:
            value += decoded[-2]
        decoded.append(int(value))
    return decoded


def decode_rle_mask(rle: Dict[str, Any]):
    import numpy as np

    size = rle.get("size", None)
    counts = rle.get("counts", None)
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError(f"RLE size must be [height, width], got {size}")
    height, width = int(size[0]), int(size[1])
    decoded_counts = decode_compressed_rle_counts(counts)

    expected = height * width
    flat = np.zeros(expected, dtype=np.uint8)
    index = 0
    value = 0
    for count in decoded_counts:
        count = int(count)
        if count < 0:
            raise ValueError(f"RLE count must be non-negative, got {count}")
        end = min(index + count, expected)
        if value == 1 and end > index:
            flat[index:end] = 1
        index += count
        value = 1 - value
    if index != expected:
        raise ValueError(f"RLE count sum {index} does not match mask size {expected}.")
    return flat.reshape((height, width), order="F")


def decode_segmentation_mask(segmentation: Any):
    import numpy as np

    if isinstance(segmentation, dict):
        return decode_rle_mask(segmentation)
    if isinstance(segmentation, list) and segmentation:
        masks = [decode_rle_mask(item) for item in segmentation if isinstance(item, dict)]
        if not masks:
            raise ValueError("RRSIS-D polygon segmentations are not supported in this loader.")
        merged = np.zeros_like(masks[0], dtype=np.uint8)
        for mask in masks:
            merged |= mask.astype(np.uint8)
        return merged
    raise ValueError("Missing or unsupported RRSIS-D segmentation.")


def _resolve_existing_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate

    raw = str(path)
    if len(raw) >= 3 and raw[1] == ":" and raw[2] in {"\\", "/"}:
        drive = raw[0].lower()
        rest = raw[3:].replace("\\", "/")
        wsl_candidate = Path(f"/mnt/{drive}/{rest}")
        if wsl_candidate.exists():
            return wsl_candidate

    raise FileNotFoundError(f"Path does not exist: {path}")


def _load_text_embeddings(path: str | Path) -> Dict[str, Any]:
    import torch

    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Text embedding payload must be a dict, got {type(payload)}")
    ids = payload.get("ids", [])
    embeddings = payload.get("embeddings", None)
    if not isinstance(ids, (list, tuple)) or embeddings is None:
        raise ValueError(f"Text embedding payload must contain ids and embeddings: {path}")
    if int(len(ids)) != int(embeddings.shape[0]):
        raise ValueError(f"Text embedding count mismatch in {path}: {len(ids)} ids vs {embeddings.shape[0]} vectors")
    optional_masks = {
        "text_token_mask": payload.get("token_masks"),
    }
    for name, value in optional_masks.items():
        if value is not None and int(value.shape[0]) != int(len(ids)):
            raise ValueError(f"{name} count mismatch in {path}: {len(ids)} ids vs {value.shape[0]} masks")

    result: Dict[str, Any] = {}
    for i, sample_id in enumerate(ids):
        item = {"text_embedding": embeddings[i].detach().cpu().float()}
        for name, value in optional_masks.items():
            if value is not None:
                item[name] = value[i].detach().cpu().bool()
        result[str(sample_id)] = item
    return result


class RRSISDRefSegDataset:
    """RRSIS-D referring segmentation dataset: image + expression -> binary mask."""

    def __init__(
        self,
        rows: Sequence[Dict[str, Any]] | str | Path,
        image_size: Optional[int] = None,
        normalize: bool = True,
        text_embedding_file: Optional[str | Path] = None,
        augment: bool = False,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        color_jitter: float = 0.15,
    ) -> None:
        self.rows = _load_jsonl_rows(rows) if isinstance(rows, (str, Path)) else list(rows)
        self.image_size = int(image_size) if image_size else None
        self.normalize = bool(normalize)
        self.text_embeddings = _load_text_embeddings(text_embedding_file) if text_embedding_file else {}
        self.require_text_embedding = bool(text_embedding_file)
        self.augment = bool(augment)
        self.hflip_prob = float(hflip_prob)
        self.vflip_prob = float(vflip_prob)
        self.color_jitter = max(float(color_jitter), 0.0)

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _read_image(path: str):
        try:
            import cv2

            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Unable to read image: {path}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except ImportError:
            import numpy as np
            from PIL import Image

            with Image.open(path) as im:
                return np.asarray(im.convert("RGB"))

    @staticmethod
    def _resize(image, mask, image_size: int):
        try:
            import cv2

            size = (image_size, image_size)
            return (
                cv2.resize(image, size, interpolation=cv2.INTER_LINEAR),
                cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST),
            )
        except ImportError:
            import numpy as np
            from PIL import Image

            size = (image_size, image_size)
            image_resized = Image.fromarray(image).resize(size, Image.BILINEAR)
            mask_resized = Image.fromarray(mask).resize(size, Image.NEAREST)
            return np.asarray(image_resized), np.asarray(mask_resized)

    def _augment(self, image, mask, text: str = ""):
        import numpy as np

        # Flips can invalidate referring expressions such as "left" or "upper".
        allow_geometric_flip = not _has_directional_text(text)
        if allow_geometric_flip:
            if self.hflip_prob > 0 and np.random.random() < self.hflip_prob:
                image = np.flip(image, axis=1)
                mask = np.flip(mask, axis=1)
            if self.vflip_prob > 0 and np.random.random() < self.vflip_prob:
                image = np.flip(image, axis=0)
                mask = np.flip(mask, axis=0)

        if self.color_jitter > 0:
            image_f = image.astype(np.float32)
            brightness = 1.0 + np.random.uniform(-self.color_jitter, self.color_jitter)
            contrast = 1.0 + np.random.uniform(-self.color_jitter, self.color_jitter)
            mean = image_f.mean(axis=(0, 1), keepdims=True)
            image = np.clip((image_f - mean) * contrast + mean, 0.0, 255.0)
            image = np.clip(image * brightness, 0.0, 255.0).astype(np.uint8)

        return np.ascontiguousarray(image), np.ascontiguousarray(mask)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        import torch

        row = self.rows[index]
        image_path = _resolve_existing_path(row["image"])
        image = self._read_image(str(image_path))
        mask = decode_segmentation_mask(row["segmentation"])

        if self.image_size:
            image, mask = self._resize(image, mask, self.image_size)

        text = str(row["text"])
        if self.augment:
            image, mask = self._augment(image, mask, text)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        if self.normalize:
            image_tensor = image_tensor / 255.0
        mask_tensor = torch.from_numpy(mask.astype("uint8")).contiguous().long()

        sample_id = str(row["id"])
        sample = {
            "img": image_tensor,
            "mask": mask_tensor,
            "im_file": str(image_path),
            "sample_id": sample_id,
            "text": text,
            "prompt": text,
            "class_idx": int(row["class_idx"]),
            "class_name": str(row["class_name"]),
            "category_name": str(row["category_name"]),
            "query_id": sample_id,
            "ann_id": int(row["ann_id"]),
            "ref_id": int(row["ref_id"]),
        }
        if self.text_embeddings:
            embedding_item = self.text_embeddings.get(sample_id)
            if embedding_item is None:
                raise KeyError(f"Missing text embedding for RRSIS-D sample_id={sample_id}")
            if isinstance(embedding_item, dict):
                sample.update(embedding_item)
            else:
                sample["text_embedding"] = embedding_item
        elif self.require_text_embedding:
            raise KeyError(f"Missing text embedding file entry for RRSIS-D sample_id={sample_id}")
        return sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare RRSIS-D referring segmentation metadata.")
    parser.add_argument("--root", type=str, default="data/RRSIS-D", help="RRSIS-D root directory.")
    parser.add_argument("--output-dir", type=str, default="pre_datasets/RRSIS-D_refseg", help="Output metadata dir.")
    return parser.parse_args()


def main() -> None:
    result = prepare_rrsisd_refseg_dataset(**vars(parse_args()))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
