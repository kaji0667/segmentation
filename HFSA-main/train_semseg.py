import argparse
import csv
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset.rrsisd_refseg_dataset import RRSISDRefSegDataset, decode_compressed_rle_counts
from ultralytics.nn import SemanticSegmentationModel
from ultralytics.utils import yaml_load, yaml_save


SAMPLE_IOU_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
SPATIAL_WORDS = {
    "above",
    "adjacent",
    "around",
    "atop",
    "below",
    "beside",
    "between",
    "bottom",
    "center",
    "central",
    "centre",
    "east",
    "eastern",
    "inside",
    "left",
    "lower",
    "middle",
    "near",
    "north",
    "northern",
    "outside",
    "right",
    "south",
    "southern",
    "top",
    "upper",
    "west",
    "western",
}
OBJECT_ALIASES = {
    "airplane": ("airplane", "plane", "aircraft"),
    "airport": ("airport",),
    "baseballfield": ("baseball", "field"),
    "basketballcourt": ("basketball", "court"),
    "bridge": ("bridge",),
    "chimney": ("chimney",),
    "dam": ("dam",),
    "expressway-service-area": ("expressway", "service", "area"),
    "expressway-toll-station": ("expressway", "toll", "station"),
    "golffield": ("golf", "field"),
    "groundtrackfield": ("ground", "track", "field"),
    "harbor": ("harbor", "harbour"),
    "overpass": ("overpass",),
    "ship": ("ship",),
    "stadium": ("stadium",),
    "storagetank": ("storage", "tank"),
    "tenniscourt": ("tennis", "court"),
    "trainstation": ("train", "station"),
    "vehicle": ("vehicle", "car", "truck"),
    "windmill": ("windmill",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the YOLOv12 text-guided semantic segmentation head on RRSIS-D.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=str, default="pre_datasets/RRSIS-D_refseg/data.yaml", help="Semantic dataset yaml.")
    parser.add_argument("--model", type=str, default="ultralytics/cfg/models/v12/yolov12-semseg.yaml", help="Model yaml.")
    parser.add_argument("--weights", type=str, default="yolov12n.pt", help="Optional YOLO pretrained weights for backbone/neck initialization; empty disables loading.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--batch", type=int, default=2, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=128, help="Square training image size.")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--device", type=str, default="cpu", help="Training device, e.g. cpu, cuda, cuda:0.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Python, NumPy, PyTorch, samplers, and dataloader workers.")
    parser.add_argument("--deterministic", action="store_true", help="Enable stricter deterministic PyTorch/CUDA behavior when supported.")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--scheduler", type=str, default="none", choices=("none", "cosine"), help="Learning-rate scheduler.")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate for cosine scheduler.")
    parser.add_argument("--patience", type=int, default=0, help="Early-stop after N epochs without validation selection-score improvement; 0 disables.")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Minimum validation selection-score improvement required to reset early-stop patience.")
    parser.add_argument("--train-head-only", action="store_true", help="Freeze all layers except the final text segmentation head.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze YOLO backbone layers.")
    parser.add_argument("--freeze-neck", action="store_true", help="Freeze YOLO neck layers before the final segmentation head.")
    parser.add_argument("--pos-weight-max", type=float, default=20.0, help="Maximum positive BCE weight for binary mask loss.")
    parser.add_argument("--loss-small-target-weight", type=float, default=1.0, help="Per-sample loss multiplier for masks with foreground area <= --loss-small-target-area; 1 disables.")
    parser.add_argument("--loss-small-target-area", type=float, default=0.0025, help="Foreground mask area ratio threshold for area-aware loss weighting.")
    parser.add_argument("--loss-tversky-fp-weight", type=float, default=0.5, help="Tversky false-positive weight for binary mask loss; >0.5 penalizes over-segmentation more.")
    parser.add_argument("--loss-fp-weight", type=float, default=0.0, help="Extra mean foreground-probability penalty on GT background pixels; 0 disables.")
    parser.add_argument("--loss-aux-p3-weight", type=float, default=0.0, help="Fixed P3 auxiliary mask loss weight; 0 disables P3 deep supervision.")
    parser.add_argument("--loss-aux-p4-weight", type=float, default=0.0, help="Fixed P4 auxiliary mask loss weight; 0 disables P4 deep supervision.")
    parser.add_argument("--small-target-boost", type=float, default=2.0, help="Sampler multiplier for small-mask/bbox samples; 1 disables.")
    parser.add_argument("--small-target-area", type=float, default=0.0025, help="Area ratio threshold for small-target sampler boosting.")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True, help="Enable lightweight train-time RRSIS-D image/mask augmentation.")
    parser.add_argument("--augment-hflip", type=float, default=0.5, help="Train-time horizontal flip probability for samples without directional text.")
    parser.add_argument("--augment-vflip", type=float, default=0.5, help="Train-time vertical flip probability for samples without directional text.")
    parser.add_argument("--augment-color-jitter", type=float, default=0.15, help="Train-time brightness/contrast jitter strength.")
    parser.add_argument("--val-thresholds", type=str, default="0.5", help="Comma-separated sigmoid thresholds to scan during binary validation.")
    parser.add_argument(
        "--val-select-metric",
        type=str,
        default="iou",
        choices=("iou", "oiou", "miou", "f1", "fbeta", "precision"),
        help="Metric used to choose validation threshold and best checkpoint; iou is the legacy alias for oIoU.",
    )
    parser.add_argument("--val-fbeta", type=float, default=0.7, help="Beta for precision-biased F-beta validation score when --val-select-metric=fbeta; beta < 1 favors precision.")
    parser.add_argument("--max-batches", type=int, default=2, help="Stop each epoch after this many train batches; 0 means full epoch.")
    parser.add_argument("--max-val-batches", type=int, default=2, help="Stop validation after this many batches; 0 means full validation.")
    parser.add_argument("--test-after-train", action="store_true", help="Evaluate the best checkpoint on the test split using the validation-selected threshold.")
    parser.add_argument("--max-test-batches", type=int, default=0, help="Stop test evaluation after this many batches; 0 means full test split.")
    parser.add_argument("--print-interval", type=int, default=10, help="Print train progress every N batches.")
    parser.add_argument("--save-dir", type=str, default="runs/semseg/train", help="Directory for run artifacts.")
    parser.add_argument("--nosave", action="store_true", help="Disable checkpoint and artifact saving.")
    parser.add_argument("--no-val", action="store_true", help="Disable validation.")
    parser.add_argument("--no-plots", action="store_true", help="Disable results and confusion-matrix plots.")
    parser.add_argument("--no-preview", action="store_true", help="Disable train/val image preview saving.")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard event logging.")
    parser.add_argument("--text-queries", action="store_true", help="Use image + text prompt -> binary mask samples.")
    parser.add_argument("--text-encoder", type=str, default="openclip", choices=("learned", "openclip"), help="Prompt embedding source for text-query training.")
    parser.add_argument("--text-model-name", type=str, default="ViT-L-14", help="OpenCLIP text model name.")
    parser.add_argument("--text-pretrained", type=str, default="openai", help="OpenCLIP pretrained weights tag.")
    parser.add_argument("--text-precision", type=str, default="auto", choices=("auto", "fp32", "fp16", "bf16"), help="OpenCLIP encoding precision.")
    parser.add_argument("--text-embed-batch-size", type=int, default=256, help="Batch size for OpenCLIP split text embedding cache generation.")
    parser.add_argument("--text-overwrite", action="store_true", help="Overwrite cached split text embeddings.")
    return parser.parse_args()


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_existing_path(path: str | Path) -> Optional[Path]:
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate

    raw = str(path)
    if len(raw) >= 3 and raw[1] == ":" and raw[2] in {"\\", "/"}:
        drive = raw[0].lower()
        rest = raw[3:].replace("\\", "/")
        wsl_candidate = Path(f"/mnt/{drive}/{rest}")
        if wsl_candidate.exists():
            return wsl_candidate
    return None


def resolve_data_root(data: Dict[str, Any]) -> Path:
    configured = resolve_existing_path(data["path"])
    if configured is not None:
        return configured

    yaml_dir = data.get("_yaml_dir")
    if yaml_dir:
        yaml_root = Path(yaml_dir).expanduser()
        if yaml_root.exists():
            return yaml_root

    return Path(data["path"]).expanduser()


def resolve_split_path(data: Dict[str, Any], split: str) -> Path:
    root = resolve_data_root(data)
    split_path = Path(data[split]).expanduser()
    return split_path if split_path.is_absolute() else root / split_path


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return moved


def load_pretrained_backbone(model: SemanticSegmentationModel, weights: str, device: torch.device) -> None:
    """Load matching pretrained YOLO weights while leaving the semantic segmentation head random."""
    if not weights:
        print("pretrained weights: disabled")
        return

    weights_path = Path(weights).expanduser()
    if not weights_path.exists() and not weights_path.is_absolute():
        weights_path = Path(__file__).resolve().parent / weights_path
    if not weights_path.exists():
        raise FileNotFoundError(f"Pretrained weights not found: {weights_path}")

    try:
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(weights_path, map_location=device)
    source = ckpt.get("model", ckpt.get("ema", ckpt)) if isinstance(ckpt, dict) else ckpt
    if isinstance(source, torch.nn.Module):
        source_state = source.float().state_dict()
    elif isinstance(source, dict):
        source_state = source
    else:
        raise TypeError(f"Unsupported pretrained weights format: {type(source)}")

    source_state = {k.removeprefix("module."): v for k, v in source_state.items() if isinstance(v, torch.Tensor)}
    target_state = model.state_dict()
    head_prefix = f"model.{len(model.model) - 1}."
    matched = {
        k: v
        for k, v in source_state.items()
        if k in target_state and target_state[k].shape == v.shape and not k.startswith(head_prefix)
    }
    model.load_state_dict(matched, strict=False)
    print(f"pretrained weights: loaded {len(matched)}/{len(target_state)} tensors from {weights_path}; skipped head prefix {head_prefix}")


def configure_trainable_layers(model: SemanticSegmentationModel, args: argparse.Namespace) -> int:
    layers = list(getattr(model, "model", []))
    if not layers:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.train_head_only:
        freeze_until = len(layers) - 1
    elif args.freeze_backbone and args.freeze_neck:
        freeze_until = len(layers) - 1
    elif args.freeze_backbone:
        freeze_until = min(9, len(layers) - 1)
    elif args.freeze_neck:
        freeze_until = 0
    else:
        freeze_until = 0

    if args.freeze_neck and not args.train_head_only and not args.freeze_backbone:
        freeze_indices = set(range(9, max(len(layers) - 1, 9)))
    else:
        freeze_indices = set(range(freeze_until))

    for i, layer in enumerate(layers):
        freeze = i in freeze_indices
        for param in layer.parameters():
            param.requires_grad = not freeze

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"trainable params: {trainable:,}; frozen params: {frozen:,}")
    return trainable


def predict_with_optional_text(
    model: SemanticSegmentationModel,
    batch: Dict[str, Any],
    text_queries: bool,
    prompt_embeddings: Optional[torch.Tensor] = None,
):
    """Run the model with class-conditioned prompt embeddings when text queries are enabled."""
    if text_queries:
        class_idx = batch.get("class_idx")
        text_embedding = batch.get("text_embedding", None)
        if text_embedding is None and prompt_embeddings is not None and class_idx is not None:
            text_embedding = prompt_embeddings[class_idx]
        return model(
            batch["img"],
            class_idx=class_idx,
            text_embedding=text_embedding,
            text_token_mask=batch.get("text_token_mask"),
            text_object_mask=batch.get("text_object_mask"),
            text_spatial_mask=batch.get("text_spatial_mask"),
            text_context_mask=batch.get("text_context_mask"),
        )
    return model(batch["img"])


def class_names(data: Dict[str, Any], nc: int) -> List[str]:
    names = data.get("names", None)
    if isinstance(names, dict):
        return [str(names.get(i, names.get(str(i), i))) for i in range(nc)]
    if isinstance(names, (list, tuple)):
        return [str(x) for x in names]
    return [str(i) for i in range(nc)]


def class_prompts(data: Dict[str, Any], names: List[str]) -> List[str]:
    prompts = data.get("prompts", {})
    if not isinstance(prompts, dict):
        prompts = {}
    return [str(prompts.get(name, name)) for name in names]


def dataset_type(data: Dict[str, Any]) -> str:
    return str(data.get("dataset_type", "rrsisd_refseg") or "rrsisd_refseg").strip().lower()


def _words_for_token_masks(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+", str(text).lower())


def _category_aliases(row: Dict[str, Any]) -> set[str]:
    names = {
        str(row.get("category_name", "")).strip().lower(),
        str(row.get("class_name", "")).strip().lower(),
    }
    aliases: set[str] = set()
    for name in names:
        compact = re.sub(r"[^a-z]+", "", name)
        hyphen = re.sub(r"[^a-z]+", "-", name).strip("-")
        if compact:
            aliases.add(compact)
        if hyphen:
            aliases.add(hyphen)
        aliases.update(OBJECT_ALIASES.get(compact, ()))
        aliases.update(OBJECT_ALIASES.get(hyphen, ()))
    return {alias for alias in aliases if alias}


def build_text_role_masks(rows: List[Dict[str, Any]], token_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build coarse object/spatial/context token masks for short RRSIS-D English expressions."""
    object_masks = torch.zeros_like(token_masks, dtype=torch.bool)
    spatial_masks = torch.zeros_like(token_masks, dtype=torch.bool)
    context_masks = token_masks.clone().to(dtype=torch.bool)
    for i, row in enumerate(rows):
        valid = torch.nonzero(token_masks[i].to(dtype=torch.bool), as_tuple=False).view(-1)
        if valid.numel() == 0:
            continue
        words = _words_for_token_masks(str(row.get("text", "")))
        aliases = _category_aliases(row)
        max_words = min(len(words), int(valid.numel()))
        for word_i in range(max_words):
            token_i = int(valid[word_i])
            word = words[word_i]
            if word in SPATIAL_WORDS:
                spatial_masks[i, token_i] = True
            if word in aliases:
                object_masks[i, token_i] = True
        if not object_masks[i].any():
            for word_i in range(max_words):
                token_i = int(valid[word_i])
                if words[word_i] not in SPATIAL_WORDS:
                    object_masks[i, token_i] = True
                    break
        context_masks[i] = token_masks[i].to(dtype=torch.bool)
    return object_masks, spatial_masks, context_masks


def parse_val_thresholds(raw: str) -> List[float]:
    values = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if not 0.0 < value < 1.0:
            raise ValueError(f"--val-thresholds values must be between 0 and 1, got {value}")
        values.append(value)
    return values or [0.5]


def validation_selection_score(metrics: Dict[str, Any], select_metric: str) -> float:
    if select_metric in {"iou", "oiou"}:
        return float(metrics["oiou"])
    if select_metric == "miou":
        return float(metrics["official_miou"])
    if select_metric == "f1":
        return float(metrics["f1"])
    if select_metric == "fbeta":
        return float(metrics["fbeta"])
    if select_metric == "precision":
        return float(metrics["precision"])
    raise ValueError(f"Unsupported validation selection metric: {select_metric}")


def _safe_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in str(value)).strip("-") or "default"


def _load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def build_rrsisd_text_embedding_cache(
    args: argparse.Namespace,
    data: Dict[str, Any],
    split_paths: Dict[str, Path],
    device: torch.device,
) -> Dict[str, Path]:
    """Build or reuse per-sample OpenCLIP text embeddings for RRSIS-D splits."""
    if dataset_type(data) != "rrsisd_refseg":
        return {}
    if args.text_encoder != "openclip":
        raise ValueError("RRSIS-D free-text segmentation requires --text-encoder openclip.")
    if int(args.text_embed_batch_size) <= 0:
        raise ValueError("--text-embed-batch-size must be > 0")

    embedding_root = resolve_data_root(data) / f"openclip_{_safe_tag(args.text_model_name)}_{_safe_tag(args.text_pretrained)}_tokens"
    paths = {split: embedding_root / f"{split}_text_embeddings.pt" for split in split_paths}
    missing = [split for split, path in paths.items() if args.text_overwrite or not path.exists()]
    if not missing:
        print(f"RRSIS-D text embeddings: using cache {embedding_root}")
        return paths

    from ultralytics.nn.modules.text_backbone import OpenCLIPTextEncoder

    encoder = OpenCLIPTextEncoder(
        model_name=args.text_model_name,
        pretrained=args.text_pretrained,
        device=str(device),
        precision=args.text_precision,
        normalize=True,
    )
    if int(encoder.embedding_dim) != 768:
        raise ValueError(
            f"OpenCLIP text dim is {int(encoder.embedding_dim)}, but yolov12-semseg.yaml expects 768. "
            "Use ViT-L-14 or update TextPromptSegment text_dim."
        )

    embedding_root.mkdir(parents=True, exist_ok=True)
    for split in missing:
        rows = _load_jsonl_rows(split_paths[split])
        ids = [str(row["id"]) for row in rows]
        texts = [str(row.get("text", "")).strip() for row in rows]
        print(f"Building RRSIS-D {split} text embeddings: {len(texts)} samples -> {paths[split]}")
        embeddings, token_masks = encoder.encode(texts, batch_size=int(args.text_embed_batch_size), return_tokens=True)
        object_masks, spatial_masks, context_masks = build_text_role_masks(rows, token_masks)
        payload = {
            "ids": ids,
            "texts": texts,
            "embeddings": embeddings.float().cpu(),
            "token_masks": token_masks.bool().cpu(),
            "object_token_masks": object_masks.bool().cpu(),
            "spatial_token_masks": spatial_masks.bool().cpu(),
            "context_token_masks": context_masks.bool().cpu(),
            "metadata": {
                "dataset_type": "rrsisd_refseg",
                "split": split,
                "model_name": args.text_model_name,
                "pretrained": args.text_pretrained,
                "embedding_dim": int(embeddings.shape[-1]) if embeddings.ndim >= 2 else 0,
                "embedding_kind": "token_features",
                "token_roles": "context/object/spatial masks from RRSIS-D text and category words",
            },
        }
        torch.save(payload, paths[split])
    return paths


def build_prompt_embeddings(args: argparse.Namespace, data: Dict[str, Any], names: List[str], device: torch.device) -> Optional[torch.Tensor]:
    if dataset_type(data) == "rrsisd_refseg":
        return None
    if not args.text_queries or args.text_encoder != "openclip":
        return None

    from ultralytics.nn.modules.text_backbone import OpenCLIPTextEncoder

    prompts = class_prompts(data, names)
    encoder = OpenCLIPTextEncoder(
        model_name=args.text_model_name,
        pretrained=args.text_pretrained,
        device=str(device),
        precision=args.text_precision,
        normalize=True,
    )
    embeddings = encoder.encode(prompts, batch_size=len(prompts), return_tokens=False)
    if embeddings.ndim != 2:
        raise RuntimeError(f"OpenCLIP prompt embeddings must be [N, D], got {tuple(embeddings.shape)}")
    if int(embeddings.shape[1]) != 768:
        raise ValueError(
            f"OpenCLIP text dim is {int(embeddings.shape[1])}, but yolov12-semseg.yaml currently expects 768. "
            "Use ViT-L-14 or update the TextPromptSegment text_dim argument in the model yaml."
        )
    print(
        f"OpenCLIP prompt embeddings: model={args.text_model_name}, "
        f"pretrained={args.text_pretrained}, shape={tuple(embeddings.shape)}"
    )
    return embeddings.to(device=device, dtype=torch.float32)


def build_semseg_dataset(
    data: Dict[str, Any],
    split_path: Path,
    args: argparse.Namespace,
    split: str,
    text_embedding_file: Optional[Path] = None,
):
    dtype = dataset_type(data)
    if dtype == "rrsisd_refseg":
        return RRSISDRefSegDataset(
            split_path,
            image_size=args.imgsz,
            normalize=True,
            text_embedding_file=text_embedding_file,
            augment=bool(args.augment and split == "train"),
            hflip_prob=args.augment_hflip,
            vflip_prob=args.augment_vflip,
            color_jitter=args.augment_color_jitter,
        )
    raise ValueError(f"Unsupported semantic dataset_type: {dtype}")


def _rle_foreground_area(rle: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    size = rle.get("size", None)
    counts = rle.get("counts", None)
    if not isinstance(size, (list, tuple)) or len(size) != 2 or counts is None:
        return None
    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        return None
    decoded_counts = decode_compressed_rle_counts(counts) if isinstance(counts, str) else list(counts)
    foreground = sum(float(count) for index, count in enumerate(decoded_counts) if index % 2 == 1)
    return foreground, float(height * width)


def _segmentation_area_ratio(segmentation: Any) -> Optional[float]:
    items = segmentation if isinstance(segmentation, list) else [segmentation]
    foreground = 0.0
    total = None
    for item in items:
        if not isinstance(item, dict):
            continue
        area = _rle_foreground_area(item)
        if area is None:
            continue
        item_foreground, item_total = area
        foreground += item_foreground
        total = item_total if total is None else total
    if total is None or total <= 0:
        return None
    return max(foreground, 0.0) / total


def _sample_area_ratio(row: Dict[str, Any]) -> Optional[float]:
    mask_area = _segmentation_area_ratio(row.get("segmentation", None))
    if mask_area is not None:
        return mask_area

    bbox = row.get("bbox", None)
    width = float(row.get("width") or 0)
    height = float(row.get("height") or 0)
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4 or width <= 0 or height <= 0:
        return None
    return max(float(bbox[2]), 0.0) * max(float(bbox[3]), 0.0) / max(width * height, 1.0)


def build_text_query_sampler(
    dataset,
    small_target_boost: float = 1.0,
    small_target_area: float = 0.0025,
    generator: Optional[torch.Generator] = None,
) -> Optional[WeightedRandomSampler]:
    query_items = getattr(dataset, "query_items", None)
    rows = getattr(dataset, "rows", None)

    if query_items:
        class_indices = [int(class_idx) for _, class_idx in query_items]
        area_ratios = [None] * len(class_indices)
    elif rows:
        class_indices = [int(row.get("class_idx", 0)) for row in rows]
        area_ratios = [_sample_area_ratio(row) for row in rows]
    else:
        return None

    counts: Dict[int, int] = {}
    for class_idx in class_indices:
        counts[class_idx] = counts.get(class_idx, 0) + 1

    boost = max(float(small_target_boost), 1.0)
    threshold = max(float(small_target_area), 0.0)
    weights_list = []
    boosted = 0
    for class_idx, area_ratio in zip(class_indices, area_ratios):
        weight = 1.0 / counts[class_idx]
        if boost > 1.0 and area_ratio is not None and area_ratio <= threshold:
            weight *= boost
            boosted += 1
        weights_list.append(weight)

    weights = torch.as_tensor(weights_list, dtype=torch.double)
    print(f"text query class counts: {dict(sorted(counts.items()))}")
    if boosted:
        print(f"small target sampler boost: {boosted}/{len(weights_list)} samples x{boost:g} (mask area <= {threshold:g})")
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True, generator=generator)


def make_palette(nc: int) -> np.ndarray:
    base = np.array(
        [
            [0, 0, 0],
            [220, 40, 40],
            [235, 170, 30],
            [40, 130, 220],
            [160, 110, 70],
            [45, 170, 90],
            [150, 205, 55],
            [180, 80, 200],
            [80, 200, 200],
            [240, 120, 160],
        ],
        dtype=np.uint8,
    )
    if nc <= len(base):
        return base[:nc]
    extra = np.random.default_rng(0).integers(0, 255, size=(nc - len(base), 3), dtype=np.uint8)
    return np.concatenate([base, extra], axis=0)


def mask_to_rgb(mask: torch.Tensor, palette: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    mask_np = mask.detach().cpu().numpy().astype(np.int64)
    rgb = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    valid = (mask_np >= 0) & (mask_np < len(palette))
    rgb[valid] = palette[mask_np[valid]]
    rgb[mask_np == ignore_index] = np.array([45, 45, 45], dtype=np.uint8)
    return rgb


def image_to_uint8(image: torch.Tensor) -> np.ndarray:
    img = image.detach().cpu().float().clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def save_preview(
    path: Path,
    batch: Dict[str, Any],
    preds: Optional[torch.Tensor],
    palette: np.ndarray,
    max_images: int = 4,
    threshold: float = 0.5,
) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    images = batch["img"].detach().cpu()
    masks = batch["mask"].detach().cpu()
    pred_masks = None
    if preds is not None:
        if preds.shape[1] == 1:
            pred_masks = (preds.detach().cpu().sigmoid().squeeze(1) > float(threshold)).long()
        else:
            pred_masks = preds.detach().cpu().argmax(1)

    rows = []
    n = min(max_images, images.shape[0])
    for i in range(n):
        panels = [image_to_uint8(images[i]), mask_to_rgb(masks[i], palette)]
        if pred_masks is not None:
            panels.append(mask_to_rgb(pred_masks[i], palette))
        row = np.concatenate(panels, axis=1)
        rows.append(row)

    grid = np.concatenate(rows, axis=0)
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def update_confusion_matrix(
    confusion: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    nc: int,
    ignore_index: int,
    threshold: float = 0.5,
) -> None:
    if preds.shape[1] == 1:
        pred_labels = (preds.squeeze(1).sigmoid() > float(threshold)).long().detach()
    else:
        pred_labels = preds.argmax(1).detach()
    target_labels = targets.detach()
    valid = (target_labels != ignore_index) & (target_labels >= 0) & (target_labels < nc)
    if valid.any():
        indices = target_labels[valid] * nc + pred_labels[valid]
        confusion += torch.bincount(indices, minlength=nc * nc).reshape(nc, nc).cpu()


def update_text_class_confusion(
    class_confusion: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    class_idx: torch.Tensor,
    ignore_index: int,
    threshold: float = 0.5,
) -> None:
    pred_labels = (preds.squeeze(1).sigmoid() > float(threshold)).long().detach()
    target_labels = targets.detach()
    valid = (target_labels != ignore_index) & (target_labels >= 0) & (target_labels <= 1)
    for i, cls in enumerate(class_idx.detach().cpu().view(-1).tolist()):
        sample_valid = valid[i]
        if sample_valid.any():
            indices = target_labels[i][sample_valid] * 2 + pred_labels[i][sample_valid]
            class_confusion[int(cls)] += torch.bincount(indices, minlength=4).reshape(2, 2).cpu()


def metrics_from_confusion(confusion: torch.Tensor, fbeta_beta: float = 1.0) -> Dict[str, Any]:
    matrix = confusion.float()
    diag = matrix.diag()
    total = matrix.sum().clamp_min(1)
    pixel_acc = float(diag.sum() / total)
    denom = matrix.sum(1) + matrix.sum(0) - diag
    iou = torch.where(denom > 0, diag / denom.clamp_min(1), torch.full_like(diag, float("nan")))
    valid_iou = iou[~torch.isnan(iou)]
    miou = float(valid_iou.mean()) if valid_iou.numel() else 0.0
    per_class_iou = [float(x) if not torch.isnan(x) else float("nan") for x in iou]
    target_iou = per_class_iou[1] if len(per_class_iou) > 1 else miou
    pred_pos_rate = float(matrix[:, 1].sum() / total) if matrix.shape[0] > 1 else float("nan")
    target_pos_rate = float(matrix[1, :].sum() / total) if matrix.shape[0] > 1 else float("nan")
    if matrix.shape[0] > 1:
        tp = matrix[1, 1]
        fp = matrix[0, 1]
        fn = matrix[1, 0]
        precision = float(tp / (tp + fp).clamp_min(1.0))
        recall = float(tp / (tp + fn).clamp_min(1.0))
        denom = precision + recall
        f1 = 2.0 * precision * recall / denom if denom > 0 else 0.0
        beta2 = float(fbeta_beta) ** 2
        fbeta_denom = beta2 * precision + recall
        fbeta = (1.0 + beta2) * precision * recall / fbeta_denom if fbeta_denom > 0 else 0.0
    else:
        precision = recall = f1 = fbeta = float("nan")
    return {
        "pixel_acc": pixel_acc,
        "miou": miou,
        "per_class_iou": per_class_iou,
        "target_iou": target_iou,
        "pred_pos_rate": pred_pos_rate,
        "target_pos_rate": target_pos_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fbeta": fbeta,
    }


def sample_iou_values(preds: torch.Tensor, targets: torch.Tensor, ignore_index: int, threshold: float) -> torch.Tensor:
    pred_labels = (preds.squeeze(1).sigmoid() > float(threshold)).bool().detach()
    target_labels = targets.detach()
    values = []
    for pred, target in zip(pred_labels, target_labels):
        valid = target != ignore_index
        if not valid.any():
            values.append(torch.full((), float("nan"), device=preds.device))
            continue
        pred = pred[valid]
        target = target[valid].clamp(0, 1).bool()
        intersection = (pred & target).sum().float()
        union = (pred | target).sum().float()
        if union <= 0:
            values.append(torch.ones((), device=preds.device))
        else:
            values.append(intersection / union)
    if not values:
        return torch.empty(0, device=preds.device)
    return torch.stack(values)


def text_class_target_ious(class_confusion: torch.Tensor, names: List[str]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for i, matrix in enumerate(class_confusion.float()):
        total = matrix.sum()
        if total <= 0:
            continue
        tp = matrix[1, 1]
        fp = matrix[0, 1]
        fn = matrix[1, 0]
        denom = tp + fp + fn
        result[names[i] if i < len(names) else str(i)] = float(tp / denom.clamp_min(1.0)) if denom > 0 else float("nan")
    return result


def text_class_mean_ious(iou_sums: torch.Tensor, iou_counts: torch.Tensor, names: List[str]) -> Dict[str, float]:
    """Return official per-category mIoU by averaging sample IoUs within each semantic category."""
    result: Dict[str, float] = {}
    for i, (iou_sum, count) in enumerate(zip(iou_sums.double(), iou_counts.long())):
        if int(count) <= 0:
            continue
        result[names[i] if i < len(names) else str(i)] = float(iou_sum / count)
    return result


def save_checkpoint(
    path: Path,
    model: SemanticSegmentationModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    data: Dict[str, Any],
    metrics: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "data": data,
            "metrics": metrics,
        },
        path,
    )


def append_results(path: Path, row: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "pixel_acc",
        "oiou",
        "official_miou",
        "miou",
        "target_iou",
        "binary_miou",
        "best_threshold",
        "precision",
        "recall",
        "f1",
        "fbeta",
        "selection_score",
        "sample_miou",
        "pr_0_5",
        "pr_0_6",
        "pr_0_7",
        "pr_0_8",
        "pr_0_9",
        "pred_pos_rate",
        "target_pos_rate",
        "class_macro_miou",
        "class_miou",
        "class_oiou",
        "class_iou",
        "threshold_metrics",
        "lr",
        "epoch_time",
    ]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def plot_results(results_csv: Path, save_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    with results_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return

    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    val_loss = [float(r["val_loss"]) if r["val_loss"] else np.nan for r in rows]
    official_miou = [
        float(r["official_miou"])
        if r.get("official_miou")
        else float(r["sample_miou"])
        if r.get("sample_miou")
        else np.nan
        for r in rows
    ]
    oiou = [
        float(r["oiou"])
        if r.get("oiou")
        else float(r["target_iou"])
        if r.get("target_iou")
        else np.nan
        for r in rows
    ]
    pixel_acc = [float(r["pixel_acc"]) if r["pixel_acc"] else np.nan for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), tight_layout=True)
    axes[0, 0].plot(epochs, train_loss, marker="o")
    axes[0, 0].set_title("train loss")
    axes[0, 1].plot(epochs, val_loss, marker="o")
    axes[0, 1].set_title("val loss")
    axes[1, 0].plot(epochs, oiou, marker="o", label="oIoU")
    axes[1, 0].plot(epochs, official_miou, marker="o", label="mIoU")
    axes[1, 0].set_title("official IoU metrics")
    axes[1, 0].legend()
    axes[1, 1].plot(epochs, pixel_acc, marker="o")
    axes[1, 1].set_title("pixel accuracy")
    for ax in axes.ravel():
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(confusion: torch.Tensor, names: List[str], save_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = confusion.float().numpy()
    row_sum = matrix.sum(1, keepdims=True)
    normalized = np.divide(matrix, np.maximum(row_sum, 1), where=row_sum > 0)

    fig, ax = plt.subplots(figsize=(8, 7), tight_layout=True)
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("normalized confusion matrix")
    ax.set_xlabel("predicted")
    ax.set_ylabel("target")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def validate(
    model: SemanticSegmentationModel,
    loader: DataLoader,
    device: torch.device,
    nc: int,
    ignore_index: int,
    max_batches: int,
    preview_path: Optional[Path],
    palette: np.ndarray,
    text_queries: bool = False,
    prompt_embeddings: Optional[torch.Tensor] = None,
    class_names_for_text: Optional[List[str]] = None,
    val_thresholds: Optional[List[float]] = None,
    val_select_metric: str = "iou",
    val_fbeta: float = 0.7,
) -> Dict[str, Any]:
    model.eval()
    thresholds = val_thresholds or [0.5]
    confusions = {float(threshold): torch.zeros((nc, nc), dtype=torch.int64) for threshold in thresholds}
    sample_iou_sums = {float(threshold): 0.0 for threshold in thresholds}
    sample_iou_counts = {float(threshold): 0 for threshold in thresholds}
    sample_iou_hits = {
        float(threshold): {iou_threshold: 0 for iou_threshold in SAMPLE_IOU_THRESHOLDS}
        for threshold in thresholds
    }
    class_confusions = (
        {float(threshold): torch.zeros((len(class_names_for_text or []), 2, 2), dtype=torch.int64) for threshold in thresholds}
        if text_queries
        else None
    )
    class_sample_iou_sums = (
        {float(threshold): torch.zeros(len(class_names_for_text or []), dtype=torch.float64) for threshold in thresholds}
        if text_queries
        else None
    )
    class_sample_iou_counts = (
        {float(threshold): torch.zeros(len(class_names_for_text or []), dtype=torch.int64) for threshold in thresholds}
        if text_queries
        else None
    )
    running = 0.0
    seen = 0
    preview_batch = None
    preview_logits = None

    for batch_i, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        preds = predict_with_optional_text(model, batch, text_queries, prompt_embeddings)
        loss, _ = model.loss(batch, preds)
        logits = preds[0] if isinstance(preds, (list, tuple)) else preds
        for threshold, confusion in confusions.items():
            update_confusion_matrix(confusion, logits, batch["mask"], nc, ignore_index, threshold=threshold)
            sample_ious = sample_iou_values(logits, batch["mask"], ignore_index, threshold)
            if sample_ious.numel():
                sample_ious_cpu = sample_ious.detach().cpu()
                finite = torch.isfinite(sample_ious_cpu)
                sample_iou_sums[threshold] += float(sample_ious_cpu[finite].sum())
                sample_iou_counts[threshold] += int(finite.sum())
                for iou_threshold in SAMPLE_IOU_THRESHOLDS:
                    sample_iou_hits[threshold][iou_threshold] += int((sample_ious_cpu[finite] >= iou_threshold).sum())
                if (
                    class_sample_iou_sums is not None
                    and class_sample_iou_counts is not None
                    and "class_idx" in batch
                    and sample_ious_cpu.numel() == batch["class_idx"].numel()
                ):
                    class_indices = batch["class_idx"].detach().cpu().view(-1).long()
                    for class_index, sample_iou, is_finite in zip(class_indices, sample_ious_cpu, finite):
                        index = int(class_index)
                        if bool(is_finite) and 0 <= index < len(class_sample_iou_sums[threshold]):
                            class_sample_iou_sums[threshold][index] += float(sample_iou)
                            class_sample_iou_counts[threshold][index] += 1
            if class_confusions is not None and "class_idx" in batch:
                update_text_class_confusion(
                    class_confusions[threshold],
                    logits,
                    batch["mask"],
                    batch["class_idx"],
                    ignore_index,
                    threshold=threshold,
                )
        running += float(loss.detach())
        seen = batch_i

        if preview_path is not None and batch_i == 1:
            preview_batch = batch
            preview_logits = logits
        if max_batches and batch_i >= max_batches:
            break

    threshold_metrics = []
    for threshold, confusion in confusions.items():
        metrics = metrics_from_confusion(confusion, fbeta_beta=val_fbeta)
        metrics["threshold"] = threshold
        metrics["binary_miou"] = metrics["miou"]
        if text_queries:
            metrics["miou"] = metrics["target_iou"]
        sample_count = max(sample_iou_counts[threshold], 1)
        metrics["sample_miou"] = sample_iou_sums[threshold] / sample_count
        metrics["oiou"] = metrics["target_iou"]
        metrics["official_miou"] = metrics["sample_miou"]
        for iou_threshold in SAMPLE_IOU_THRESHOLDS:
            metrics[f"pr_{str(iou_threshold).replace('.', '_')}"] = sample_iou_hits[threshold][iou_threshold] / sample_count
        metrics["selection_score"] = validation_selection_score(metrics, val_select_metric)
        threshold_metrics.append(metrics)

    best_metrics = max(
        threshold_metrics,
        key=lambda item: (float(item["selection_score"]), float(item["miou"]), float(item["f1"])),
    )
    best_threshold = float(best_metrics["threshold"])
    best_confusion = confusions[best_threshold]
    if preview_path is not None and preview_batch is not None and preview_logits is not None:
        save_preview(preview_path, preview_batch, preview_logits, palette, threshold=best_threshold)
    text_class_iou = (
        text_class_target_ious(class_confusions[best_threshold], class_names_for_text or [])
        if class_confusions is not None
        else {}
    )
    text_class_miou = (
        text_class_mean_ious(
            class_sample_iou_sums[best_threshold],
            class_sample_iou_counts[best_threshold],
            class_names_for_text or [],
        )
        if class_sample_iou_sums is not None and class_sample_iou_counts is not None
        else {}
    )
    class_macro_miou = float(np.mean(list(text_class_miou.values()))) if text_class_miou else float("nan")
    model.train()
    return {
        "loss": running / max(seen, 1),
        "pixel_acc": best_metrics["pixel_acc"],
        "miou": best_metrics["miou"],
        "oiou": best_metrics["oiou"],
        "official_miou": best_metrics["official_miou"],
        "target_iou": best_metrics["target_iou"],
        "binary_miou": best_metrics["binary_miou"],
        "best_threshold": best_threshold,
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "f1": best_metrics["f1"],
        "fbeta": best_metrics["fbeta"],
        "selection_score": best_metrics["selection_score"],
        "sample_miou": best_metrics["sample_miou"],
        **{f"pr_{str(iou_threshold).replace('.', '_')}": best_metrics[f"pr_{str(iou_threshold).replace('.', '_')}"] for iou_threshold in SAMPLE_IOU_THRESHOLDS},
        "pred_pos_rate": best_metrics["pred_pos_rate"],
        "target_pos_rate": best_metrics["target_pos_rate"],
        "text_class_iou": text_class_iou,
        "text_class_miou": text_class_miou,
        "class_macro_miou": class_macro_miou,
        "per_class_iou": best_metrics["per_class_iou"],
        "threshold_metrics": threshold_metrics,
        "confusion": best_confusion,
        "evaluated_samples": int(sample_iou_counts[best_threshold]),
    }


def build_summary_writer(save_dir: Path, disabled: bool):
    if disabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(str(save_dir))
    except Exception as exc:
        print(f"TensorBoard unavailable: {exc}")
        return None


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed, args.deterministic)
    val_thresholds = parse_val_thresholds(args.val_thresholds)
    data = yaml_load(args.data)
    data["_yaml_dir"] = str(Path(args.data).expanduser().resolve().parent)
    dtype = dataset_type(data)
    if dtype == "rrsisd_refseg":
        if not args.text_queries:
            print("RRSIS-D refseg data detected; enabling --text-queries.")
            args.text_queries = True
        if args.text_encoder != "openclip":
            raise ValueError("RRSIS-D refseg training requires --text-encoder openclip to use each free-text expression.")
    nc = int(data["nc"])
    ignore_index = int(data.get("ignore_index", 255))
    names = class_names(data, nc)
    metric_nc = 2 if args.text_queries else nc
    metric_names = ["other", "target"] if args.text_queries else names
    palette = make_palette(metric_nc)
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    weights_dir = save_dir / "weights"
    results_csv = save_dir / "results.csv"

    if args.test_after_train and args.nosave:
        raise ValueError("--test-after-train requires checkpoint saving; remove --nosave.")

    train_jsonl = resolve_split_path(data, "train")
    val_jsonl = resolve_split_path(data, "val") if "val" in data and not args.no_val else None
    test_jsonl = resolve_split_path(data, "test") if args.test_after_train and "test" in data else None
    split_paths = {"train": train_jsonl}
    if val_jsonl:
        split_paths["val"] = val_jsonl
    if test_jsonl:
        split_paths["test"] = test_jsonl
    text_embedding_paths = build_rrsisd_text_embedding_cache(args, data, split_paths, device)
    train_set = build_semseg_dataset(
        data,
        train_jsonl,
        args,
        "train",
        text_embedding_file=text_embedding_paths.get("train"),
    )
    val_set = (
        build_semseg_dataset(
            data,
            val_jsonl,
            args,
            "val",
            text_embedding_file=text_embedding_paths.get("val"),
        )
        if val_jsonl
        else None
    )
    test_set = (
        build_semseg_dataset(
            data,
            test_jsonl,
            args,
            "test",
            text_embedding_file=text_embedding_paths.get("test"),
        )
        if test_jsonl
        else None
    )
    data_generator = torch.Generator()
    data_generator.manual_seed(int(args.seed))
    train_sampler = (
        build_text_query_sampler(train_set, args.small_target_boost, args.small_target_area, generator=data_generator)
        if args.text_queries
        else None
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=data_generator,
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            worker_init_fn=seed_worker,
            generator=data_generator,
        )
        if val_set is not None
        else None
    )
    test_loader = (
        DataLoader(
            test_set,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            worker_init_fn=seed_worker,
            generator=data_generator,
        )
        if test_set is not None
        else None
    )

    if not args.nosave:
        save_dir.mkdir(parents=True, exist_ok=True)
        weights_dir.mkdir(parents=True, exist_ok=True)
        yaml_save(save_dir / "args.yaml", vars(args))

    writer = build_summary_writer(save_dir, args.no_tensorboard or args.nosave)
    model = SemanticSegmentationModel(args.model, ch=3, nc=nc, verbose=False).to(device)
    load_pretrained_backbone(model, args.weights, device)
    model.loss_pos_weight_max = float(args.pos_weight_max)
    model.loss_small_target_weight = float(args.loss_small_target_weight)
    model.loss_small_target_area = float(args.loss_small_target_area)
    model.loss_tversky_fp_weight = float(args.loss_tversky_fp_weight)
    model.loss_fp_weight = float(args.loss_fp_weight)
    model.loss_aux_p3_weight = max(float(args.loss_aux_p3_weight), 0.0)
    model.loss_aux_p4_weight = max(float(args.loss_aux_p4_weight), 0.0)
    deep_supervision_enabled = model.loss_aux_p3_weight > 0.0 or model.loss_aux_p4_weight > 0.0
    for module in model.modules():
        if hasattr(module, "deep_supervision_enabled"):
            module.deep_supervision_enabled = deep_supervision_enabled
    trainable_params = configure_trainable_layers(model, args)
    if trainable_params <= 0:
        raise RuntimeError("No trainable parameters remain after applying freeze options.")
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(args.epochs), 1),
            eta_min=float(args.min_lr),
        )
    prompt_embeddings = build_prompt_embeddings(args, data, names, device)

    print(f"data: {args.data}")
    print(f"dataset_type: {dtype}")
    print(f"train samples: {len(train_set)}")
    print(f"val samples: {len(val_set) if val_set is not None else 0}")
    print(f"test samples: {len(test_set) if test_set is not None else 0}")
    print(f"classes: {nc}")
    print(f"weights: {args.weights or '<none>'}")
    print(f"text queries: {args.text_queries}")
    print(f"text encoder: {args.text_encoder}")
    print(f"device: {device}")
    print(
        f"imgsz: {args.imgsz}, batch: {args.batch}, "
        f"max_batches: {args.max_batches}, max_val_batches: {args.max_val_batches}"
    )
    print(f"scheduler: {args.scheduler}, min_lr: {args.min_lr}")
    print(f"seed: {args.seed}, deterministic: {args.deterministic}")
    print(f"early stopping: patience={args.patience}, min_delta={args.min_delta}")
    print(f"loss pos_weight max: {args.pos_weight_max}")
    print(
        "loss small target: "
        f"weight={args.loss_small_target_weight}, area<={args.loss_small_target_area}"
    )
    print(
        "loss over-segmentation: "
        f"tversky_fp_weight={args.loss_tversky_fp_weight}, fp_weight={args.loss_fp_weight}"
    )
    print(
        "loss deep supervision: "
        f"enabled={deep_supervision_enabled}, p3_weight={model.loss_aux_p3_weight}, "
        f"p4_weight={model.loss_aux_p4_weight}"
    )
    print(
        "train augmentation: "
        f"enabled={args.augment}, hflip={args.augment_hflip}, "
        f"vflip={args.augment_vflip}, color_jitter={args.augment_color_jitter}"
    )
    print(f"validation thresholds: {val_thresholds}")
    print(
        "validation selection: "
        f"metric={args.val_select_metric}, fbeta={args.val_fbeta}"
    )
    if not args.nosave:
        print(f"save dir: {save_dir}")

    best_fitness = -float("inf")
    epochs_without_improvement = 0
    last_confusion = None

    for epoch in range(args.epochs):
        epoch_start = time.time()
        running = 0.0
        seen = 0
        for batch_i, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            preds = predict_with_optional_text(model, batch, args.text_queries, prompt_embeddings)
            loss, loss_items = model.loss(batch, preds)
            loss.backward()
            optimizer.step()

            running += float(loss.detach())
            seen = batch_i
            should_print = batch_i == 1 or batch_i % max(args.print_interval, 1) == 0
            if should_print:
                print(
                    f"epoch {epoch + 1}/{args.epochs} "
                    f"batch {batch_i}/{len(train_loader)} "
                    f"loss {float(loss.detach()):.4f} "
                    f"items {[round(float(x), 4) for x in loss_items.detach().cpu().view(-1)]}"
                )

            if not args.nosave and not args.no_preview and epoch == 0 and batch_i == 1:
                logits = preds[0] if isinstance(preds, (list, tuple)) else preds
                save_preview(save_dir / "train_batch0.jpg", batch, logits, palette)

            if args.max_batches and batch_i >= args.max_batches:
                break

        train_loss = running / max(seen, 1)
        val_metrics = None
        if val_loader is not None:
            preview_path = None
            if not args.nosave and not args.no_preview:
                preview_path = save_dir / f"val_batch0_pred_epoch{epoch + 1}.jpg"
            val_metrics = validate(
                model,
                val_loader,
                device,
                metric_nc,
                ignore_index,
                args.max_val_batches,
                preview_path,
                palette,
                args.text_queries,
                prompt_embeddings,
                names,
                val_thresholds,
                args.val_select_metric,
                args.val_fbeta,
            )
            last_confusion = val_metrics["confusion"]

        epoch_time = time.time() - epoch_start
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"] if val_metrics else "",
            "pixel_acc": val_metrics["pixel_acc"] if val_metrics else "",
            "oiou": val_metrics["oiou"] if val_metrics else "",
            "official_miou": val_metrics["official_miou"] if val_metrics else "",
            "miou": val_metrics["miou"] if val_metrics else "",
            "target_iou": val_metrics["target_iou"] if val_metrics else "",
            "binary_miou": val_metrics["binary_miou"] if val_metrics else "",
            "best_threshold": val_metrics["best_threshold"] if val_metrics else "",
            "precision": val_metrics["precision"] if val_metrics else "",
            "recall": val_metrics["recall"] if val_metrics else "",
            "f1": val_metrics["f1"] if val_metrics else "",
            "fbeta": val_metrics["fbeta"] if val_metrics else "",
            "selection_score": val_metrics["selection_score"] if val_metrics else "",
            "sample_miou": val_metrics["sample_miou"] if val_metrics else "",
            "pr_0_5": val_metrics["pr_0_5"] if val_metrics else "",
            "pr_0_6": val_metrics["pr_0_6"] if val_metrics else "",
            "pr_0_7": val_metrics["pr_0_7"] if val_metrics else "",
            "pr_0_8": val_metrics["pr_0_8"] if val_metrics else "",
            "pr_0_9": val_metrics["pr_0_9"] if val_metrics else "",
            "pred_pos_rate": val_metrics["pred_pos_rate"] if val_metrics else "",
            "target_pos_rate": val_metrics["target_pos_rate"] if val_metrics else "",
            "class_macro_miou": val_metrics["class_macro_miou"] if val_metrics else "",
            "class_miou": json.dumps(val_metrics["text_class_miou"], sort_keys=True) if val_metrics else "",
            "class_oiou": json.dumps(val_metrics["text_class_iou"], sort_keys=True) if val_metrics else "",
            "class_iou": json.dumps(val_metrics["text_class_iou"], sort_keys=True) if val_metrics else "",
            "threshold_metrics": json.dumps(val_metrics["threshold_metrics"], sort_keys=True) if val_metrics else "",
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": epoch_time,
        }
        print(
            f"epoch {epoch + 1} train_loss {train_loss:.4f} "
            + (
                f"val_loss {val_metrics['loss']:.4f} "
                f"pixel_acc {val_metrics['pixel_acc']:.4f} "
                f"oIoU {val_metrics['oiou']:.4f} "
                f"mIoU {val_metrics['official_miou']:.4f} "
                f"thr {val_metrics['best_threshold']:.2f} "
                f"P/R/F1/Fb {val_metrics['precision']:.4f}/{val_metrics['recall']:.4f}/{val_metrics['f1']:.4f}/{val_metrics['fbeta']:.4f} "
                f"select {val_metrics['selection_score']:.4f} "
                f"Pr@0.5/0.7/0.9 {val_metrics['pr_0_5']:.4f}/{val_metrics['pr_0_7']:.4f}/{val_metrics['pr_0_9']:.4f} "
                f"pred_pos {val_metrics['pred_pos_rate']:.4f} "
                if val_metrics
                else ""
            )
            + f"time {epoch_time:.1f}s"
        )
        if val_metrics and val_metrics.get("text_class_miou"):
            class_miou_text = ", ".join(
                f"{name}:{value:.3f}"
                for name, value in val_metrics["text_class_miou"].items()
                if name != "background"
            )
            print(f"per-class mIoU: {class_miou_text}")

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch + 1)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch + 1)
            if val_metrics:
                writer.add_scalar("loss/val", val_metrics["loss"], epoch + 1)
                writer.add_scalar("metrics/pixel_acc", val_metrics["pixel_acc"], epoch + 1)
                writer.add_scalar("metrics/oIoU", val_metrics["oiou"], epoch + 1)
                writer.add_scalar("metrics/mIoU", val_metrics["official_miou"], epoch + 1)
                writer.add_scalar("metrics/best_threshold", val_metrics["best_threshold"], epoch + 1)
                writer.add_scalar("metrics/precision", val_metrics["precision"], epoch + 1)
                writer.add_scalar("metrics/recall", val_metrics["recall"], epoch + 1)
                writer.add_scalar("metrics/f1", val_metrics["f1"], epoch + 1)
                writer.add_scalar("metrics/fbeta", val_metrics["fbeta"], epoch + 1)
                writer.add_scalar("metrics/selection_score", val_metrics["selection_score"], epoch + 1)
                for iou_threshold in SAMPLE_IOU_THRESHOLDS:
                    key = f"pr_{str(iou_threshold).replace('.', '_')}"
                    writer.add_scalar(f"metrics/Pr@{iou_threshold}", val_metrics[key], epoch + 1)
                writer.add_scalar("metrics/pred_pos_rate", val_metrics["pred_pos_rate"], epoch + 1)
                writer.add_scalar("metrics/target_pos_rate", val_metrics["target_pos_rate"], epoch + 1)
                writer.add_scalar("metrics/class_macro_mIoU", val_metrics["class_macro_miou"], epoch + 1)
                for name, value in val_metrics.get("text_class_miou", {}).items():
                    writer.add_scalar(f"metrics/class_mIoU/{name}", value, epoch + 1)
                for name, value in val_metrics.get("text_class_iou", {}).items():
                    writer.add_scalar(f"metrics/class_oIoU/{name}", value, epoch + 1)

        if not args.nosave:
            append_results(results_csv, metrics)
            checkpoint_metrics = {
                "train_loss": float(train_loss),
                "val_loss": float(val_metrics["loss"]) if val_metrics else float("nan"),
                "pixel_acc": float(val_metrics["pixel_acc"]) if val_metrics else float("nan"),
                "oiou": float(val_metrics["oiou"]) if val_metrics else float("nan"),
                "official_miou": float(val_metrics["official_miou"]) if val_metrics else float("nan"),
                "miou": float(val_metrics["miou"]) if val_metrics else float("nan"),
                "target_iou": float(val_metrics["target_iou"]) if val_metrics else float("nan"),
                "binary_miou": float(val_metrics["binary_miou"]) if val_metrics else float("nan"),
                "best_threshold": float(val_metrics["best_threshold"]) if val_metrics else float("nan"),
                "precision": float(val_metrics["precision"]) if val_metrics else float("nan"),
                "recall": float(val_metrics["recall"]) if val_metrics else float("nan"),
                "f1": float(val_metrics["f1"]) if val_metrics else float("nan"),
                "fbeta": float(val_metrics["fbeta"]) if val_metrics else float("nan"),
                "selection_score": float(val_metrics["selection_score"]) if val_metrics else float("nan"),
                "sample_miou": float(val_metrics["sample_miou"]) if val_metrics else float("nan"),
                "pr_0_5": float(val_metrics["pr_0_5"]) if val_metrics else float("nan"),
                "pr_0_6": float(val_metrics["pr_0_6"]) if val_metrics else float("nan"),
                "pr_0_7": float(val_metrics["pr_0_7"]) if val_metrics else float("nan"),
                "pr_0_8": float(val_metrics["pr_0_8"]) if val_metrics else float("nan"),
                "pr_0_9": float(val_metrics["pr_0_9"]) if val_metrics else float("nan"),
                "pred_pos_rate": float(val_metrics["pred_pos_rate"]) if val_metrics else float("nan"),
                "target_pos_rate": float(val_metrics["target_pos_rate"]) if val_metrics else float("nan"),
                "class_macro_miou": float(val_metrics["class_macro_miou"]) if val_metrics else float("nan"),
            }
            last_path = weights_dir / "last.pt"
            save_checkpoint(last_path, model, optimizer, epoch + 1, args, data, checkpoint_metrics)
            fitness = float(val_metrics["selection_score"]) if val_metrics else -train_loss
            improved = fitness > best_fitness + float(args.min_delta)
            if improved:
                best_fitness = fitness
                epochs_without_improvement = 0
                save_checkpoint(weights_dir / "best.pt", model, optimizer, epoch + 1, args, data, checkpoint_metrics)
                print(f"saved best checkpoint: {weights_dir / 'best.pt'}")
            elif val_metrics:
                epochs_without_improvement += 1
            print(f"saved last checkpoint: {last_path}")

            if not args.no_plots:
                plot_results(results_csv, save_dir / "results.png")
                if last_confusion is not None:
                    plot_confusion_matrix(last_confusion, metric_names, save_dir / "confusion_matrix.png")

        if scheduler is not None:
            scheduler.step()

        if val_metrics and args.patience > 0 and epochs_without_improvement >= args.patience:
            print(
                f"early stopping: no {args.val_select_metric} improvement for {epochs_without_improvement} "
                f"epoch(s); best score {best_fitness:.4f}"
            )
            break

    if test_loader is not None:
        best_path = weights_dir / "best.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"Cannot run test evaluation without best checkpoint: {best_path}")
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        checkpoint_metrics = checkpoint.get("metrics", {})
        test_threshold = float(checkpoint_metrics.get("best_threshold", val_thresholds[0]))
        if not np.isfinite(test_threshold):
            test_threshold = float(val_thresholds[0])
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        test_start = time.perf_counter()
        test_metrics = validate(
            model,
            test_loader,
            device,
            metric_nc,
            ignore_index,
            args.max_test_batches,
            None if args.no_preview else save_dir / "test_batch0_pred.jpg",
            palette,
            args.text_queries,
            prompt_embeddings,
            names,
            [test_threshold],
            "oiou",
            args.val_fbeta,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        test_seconds = time.perf_counter() - test_start
        evaluated_samples = max(int(test_metrics["evaluated_samples"]), 1)
        test_report = {
            "split": "test",
            "checkpoint": str(best_path),
            "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
            "threshold_source": "best validation checkpoint",
            "threshold": test_threshold,
            "evaluated_samples": int(test_metrics["evaluated_samples"]),
            "oIoU": test_metrics["oiou"],
            "mIoU": test_metrics["official_miou"],
            "Pr@0.5": test_metrics["pr_0_5"],
            "Pr@0.6": test_metrics["pr_0_6"],
            "Pr@0.7": test_metrics["pr_0_7"],
            "Pr@0.8": test_metrics["pr_0_8"],
            "Pr@0.9": test_metrics["pr_0_9"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1"],
            "pixel_acc": test_metrics["pixel_acc"],
            "pred_pos_rate": test_metrics["pred_pos_rate"],
            "target_pos_rate": test_metrics["target_pos_rate"],
            "class_macro_mIoU": test_metrics["class_macro_miou"],
            "class_mIoU": test_metrics["text_class_miou"],
            "class_oIoU": test_metrics["text_class_iou"],
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "checkpoint_size_mb": best_path.stat().st_size / (1024**2),
            "evaluation_seconds": test_seconds,
            "mean_ms_per_sample": test_seconds * 1000.0 / evaluated_samples,
            "peak_gpu_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
            ),
            "device": str(device),
            "imgsz": int(args.imgsz),
            "batch": int(args.batch),
            "text_encoder": args.text_encoder,
            "text_model_name": args.text_model_name,
            "text_pretrained": args.text_pretrained,
        }
        with (save_dir / "test_results.json").open("w", encoding="utf-8") as f:
            json.dump(test_report, f, ensure_ascii=False, indent=2, sort_keys=True)
        if not args.no_plots:
            plot_confusion_matrix(test_metrics["confusion"], metric_names, save_dir / "test_confusion_matrix.png")
        print(
            "test metrics: "
            f"oIoU={test_metrics['oiou']:.4f}, mIoU={test_metrics['official_miou']:.4f}, "
            f"Pr@0.5/0.7/0.9={test_metrics['pr_0_5']:.4f}/{test_metrics['pr_0_7']:.4f}/{test_metrics['pr_0_9']:.4f}, "
            f"threshold={test_threshold:.2f}"
        )
        print(f"saved test report: {save_dir / 'test_results.json'}")

    if writer is not None:
        writer.close()

    print("semantic segmentation train ok")


if __name__ == "__main__":
    main()
