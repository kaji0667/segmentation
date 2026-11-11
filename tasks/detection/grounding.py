"""Single-description Web grounding using the original detection-test matching.

Counting deliberately continues to use detection.inference and all NMS boxes.
This module has no dependency on the official API, annotations or target labels.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from text_encoder.matching import assign_by_score_matrix, score_matrix_from_phrase_maps
from text_encoder.phrase_classifier import classify_phrase, get_phrase_type_weight
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.modules.text_backbone import OpenCLIPTextEncoder, sanitize_text
from ultralytics.utils.ops import non_max_suppression

from .inference import TextGuidedDetectionPredictor, resolve_device


class DescriptionEncoder:
    """Encode one bare object description, as in the offline embedding cache."""

    def __init__(self, device: torch.device, pretrained: str | Path) -> None:
        self.encoder = OpenCLIPTextEncoder(
            model_name="ViT-L-14", pretrained=str(pretrained), device=str(device),
            precision="fp32", normalize=True,
        )
        self.tokenizer = self.encoder.tokenizer
        if (getattr(self.tokenizer, "context_length", None) != 77
                or not callable(getattr(self.tokenizer, "encode", None))):
            raise RuntimeError("Detection requires a verifiable 77-token OpenCLIP tokenizer.")
        self.cache: dict[str, torch.Tensor] = {}

    def encode(self, text: str) -> torch.Tensor:
        description = sanitize_text(text, empty_placeholder="")
        if not description:
            raise ValueError("目标检测需要填写目标描述。")
        if len(self.tokenizer.encode(description)) > 75:
            raise ValueError("目标描述超过文本编码器长度限制，请缩短描述；不会静默截断。")
        if description not in self.cache:
            features = self.encoder.encode([description], return_tokens=False)
            if (not isinstance(features, torch.Tensor) or features.shape != (1, 768)
                    or not torch.isfinite(features).all() or bool((features.norm(dim=-1) <= 0).any())):
                raise RuntimeError("Invalid detection description embedding.")
            # The training encoder has already normalized once. Preserve those bytes.
            self.cache[description] = features.detach().float().cpu().clone()
        return self.cache[description].clone()


@dataclass(frozen=True)
class GroundingPrediction:
    text: str
    boxes: np.ndarray
    scores: np.ndarray
    visualization: np.ndarray
    latency_ms: float
    original_size: tuple[int, int]
    candidate_count: int
    matching_score: float | None


class DescriptionGroundingPredictor:
    """One description -> zero or one real, phrase-map-matched candidate box."""

    def __init__(self, checkpoint: str | Path, device: str = "auto", imgsz: int = 800,
                 *, backend_factory=AutoBackend, encoder_factory=DescriptionEncoder) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Detection checkpoint not found: {self.checkpoint}")
        if isinstance(imgsz, bool) or not isinstance(imgsz, int) or imgsz <= 0 or imgsz % 32:
            raise ValueError("Detection imgsz must be a positive multiple of 32.")
        self.imgsz = imgsz
        self.device = resolve_device(device)
        self.backend = backend_factory(weights=str(self.checkpoint), device=self.device, dnn=False, fp16=False)
        if not (self.backend.pt or self.backend.nn_module):
            raise ValueError("Description grounding requires a PyTorch checkpoint.")
        self.model = self.backend.model.eval()
        # Use the delivered local asset. Never download a model at a browser click.
        # Checkpoints may be configured outside the project; the text asset belongs
        # to this deployment, not to a guessed directory next to external weights.
        pretrained = Path(__file__).resolve().parents[2] / "openclip/ViT-L-14.pt"
        if not pretrained.is_file():
            raise FileNotFoundError(f"Local OpenCLIP checkpoint not found: {pretrained}")
        self.encoder = encoder_factory(self.device, pretrained)

    def _preprocess(self, image: np.ndarray) -> tuple[torch.Tensor, tuple[float, float, int, int]]:
        height, width = image.shape[:2]
        stride = int(self.backend.stride)
        ratio = self.imgsz / max(height, width)
        resized_h = min(math.ceil(height * ratio), self.imgsz)
        resized_w = min(math.ceil(width * ratio), self.imgsz)
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        rect_shape = np.ceil(
            np.array([height, width], dtype=np.float64) / max(height, width) * self.imgsz / stride + .5
        ).astype(int) * stride
        left = int(round((int(rect_shape[1]) - resized_w) / 2 - .1))
        top = int(round((int(rect_shape[0]) - resized_h) / 2 - .1))
        padded = LetterBox(new_shape=tuple(rect_shape), scaleup=False, auto=False, stride=stride)(image=resized)
        tensor = torch.from_numpy(np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)))
        return tensor.float().unsqueeze(0) / 255, (resized_h / height, resized_w / width, left, top)

    def _phrase_maps(self) -> list[torch.Tensor]:
        outputs = getattr(self.model, "_last_text_outputs", None)
        maps = outputs.get("phrase_logits") if isinstance(outputs, dict) else None
        if not isinstance(maps, list) or not maps:
            raise RuntimeError("缺少文本匹配特征，无法定位目标；不会改用检测置信度选框。")
        if any(not isinstance(level, torch.Tensor) or level.ndim != 4
               or level.shape[:2] != (1, 1) or min(level.shape[2:]) <= 0
               or not torch.isfinite(level).all() for level in maps):
            raise RuntimeError("Invalid description-to-image phrase maps.")
        return maps

    def predict(self, image: Image.Image | str | Path | np.ndarray, text: str,
                conf_thres: float = .25, iou_thres: float = .7, max_det: int = 300) -> GroundingPrediction:
        description = sanitize_text(text, empty_placeholder="")
        if not description:
            raise ValueError("目标检测需要填写目标描述。")
        if (not math.isfinite(float(conf_thres)) or not 0 < conf_thres < 1
                or not math.isfinite(float(iou_thres)) or not 0 < iou_thres < 1
                or isinstance(max_det, bool) or not isinstance(max_det, int) or max_det <= 0):
            raise ValueError("Invalid detection NMS configuration.")
        original = TextGuidedDetectionPredictor._to_bgr(image)
        height, width = original.shape[:2]
        start = time.perf_counter()
        vectors = self.encoder.encode(description).to(self.device).unsqueeze(0)
        if vectors.shape != (1, 1, 768) or not torch.isfinite(vectors).all():
            raise RuntimeError("Invalid detection description embedding.")
        tensor, geometry = self._preprocess(original)
        tensor = tensor.to(self.device)
        mask = torch.ones((1, 1), device=self.device, dtype=torch.bool)
        phrase_weight = torch.tensor([[get_phrase_type_weight(classify_phrase(description))]],
                                     device=self.device, dtype=tensor.dtype)
        dependency = torch.ones((1, 1, 1), device=self.device, dtype=tensor.dtype)
        matching_score = None
        with torch.inference_mode():
            self.model._last_text_outputs = None
            raw = self.model(tensor, augment=False, txt_vec=vectors, txt_token_mask=mask,
                             txt_phrase_weight=phrase_weight, txt_dependency_adj=dependency)
            candidates = non_max_suppression(raw, conf_thres=conf_thres, iou_thres=iou_thres,
                                            multi_label=False, agnostic=True, max_det=max_det)[0]
            if (not isinstance(candidates, torch.Tensor) or candidates.ndim != 2
                    or candidates.shape[1] < 5 or not torch.isfinite(candidates).all()):
                raise RuntimeError("Invalid detection candidates.")
            candidate_count = len(candidates)
            selected = candidates[:0]
            if candidate_count:
                score = score_matrix_from_phrase_maps(
                    phrase_logits_levels=self._phrase_maps(), image_index=0,
                    token_indices=torch.zeros((1,), device=self.device, dtype=torch.long),
                    candidate_boxes_xyxy=candidates[:, :4], input_hw=tuple(tensor.shape[2:]),
                )
                if score.shape != (1, candidate_count) or not torch.isfinite(score).all():
                    raise RuntimeError("Invalid description matching scores.")
                rows, columns = assign_by_score_matrix(score, temperature=.7)
                if rows.numel() != 1 or columns.numel() != 1 or int(rows[0]) != 0:
                    raise RuntimeError("No unique description-to-box assignment.")
                index = int(columns[0])
                if not 0 <= index < candidate_count:
                    raise RuntimeError("Invalid description matching index.")
                selected = candidates[index:index + 1]
                matching_score = float(score[0, index])
            boxes = selected[:, :4].clone()
            gain_h, gain_w, left, top = geometry
            boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - left) / gain_w).clamp(0, width)
            boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - top) / gain_h).clamp(0, height)
            if len(boxes) and bool(((boxes[:, 2:] - boxes[:, :2]) <= 0).any()):
                raise RuntimeError("Matched target has no area inside the original image.")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = (time.perf_counter() - start) * 1000
        boxes_np = boxes.detach().cpu().numpy().astype(np.float32, copy=True)
        scores_np = selected[:, 4].detach().cpu().numpy().astype(np.float32, copy=True)
        return GroundingPrediction(
            text=description, boxes=boxes_np, scores=scores_np,
            visualization=TextGuidedDetectionPredictor._render(original, boxes_np, scores_np, description),
            latency_ms=elapsed, original_size=(height, width), candidate_count=candidate_count,
            matching_score=matching_score,
        )
