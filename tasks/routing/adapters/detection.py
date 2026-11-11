"""Web adapter for the senior teammate's text-guided target detector."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from ..config import TaskConfig
from .common import decode_data_url, png_data_url, release_model_resources, resolve_checkpoint
from .refseg import translate_refseg_prompt


PredictorFactory = Callable[..., Any]


def _default_predictor_factory(**kwargs: Any) -> Any:
    from tasks.detection.grounding import DescriptionGroundingPredictor

    return DescriptionGroundingPredictor(**kwargs)


class DetectionAdapter:
    """Translate browser input into prompted detections and an annotated PNG."""

    def __init__(self, config: TaskConfig, predictor_factory: PredictorFactory | None = None) -> None:
        self.config = config
        self.checkpoint = resolve_checkpoint(config, "HFSA_DETECTION_CHECKPOINT")
        self.device = os.environ.get("HFSA_DETECTION_DEVICE", "auto").strip() or "auto"
        defaults = config.runtime_defaults
        self.imgsz = int(defaults.get("imgsz", 800))
        self.conf_thres = float(defaults.get("conf_thres", 0.25))
        self.iou_thres = float(defaults.get("iou_thres", 0.7))
        self.max_det = int(defaults.get("max_det", 300))
        self._predictor_factory = predictor_factory or _default_predictor_factory
        self._predictor: Any | None = None
        self._lock = threading.Lock()

    def _get_predictor(self) -> Any:
        if self._predictor is None:
            if not self.checkpoint.is_file():
                raise FileNotFoundError(f"Detection checkpoint not found: {self.checkpoint}")
            self._predictor = self._predictor_factory(
                checkpoint=self.checkpoint,
                device=self.device,
                imgsz=self.imgsz,
            )
        return self._predictor

    def predict(self, image: Any, text: str) -> dict[str, Any]:
        prompt = str(text).strip()
        if not prompt:
            raise ValueError("目标检测需要填写目标描述。")
        model_prompt, translated = translate_refseg_prompt(prompt)
        decoded_image = decode_data_url(image)
        with self._lock:
            prediction = self._get_predictor().predict(
                decoded_image,
                model_prompt,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                max_det=self.max_det,
            )
        boxes = np.asarray(prediction.boxes, dtype=np.float32)
        scores = np.asarray(prediction.scores, dtype=np.float32)
        visualization = np.asarray(prediction.visualization, dtype=np.uint8)
        if boxes.ndim != 2 or boxes.shape[1:] != (4,) or scores.shape != (len(boxes),):
            raise RuntimeError("Detection boxes and scores have incompatible shapes.")
        if len(boxes) > 1 or not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise RuntimeError("指代检测必须返回零个或一个文本匹配框。")
        if visualization.ndim != 3 or visualization.shape[2] != 3:
            raise RuntimeError("Detection visualization must be an RGB image.")
        return {
            "summary": {
                "prompt": prompt,
                "model_prompt": model_prompt,
                "prompt_translated": translated,
                "candidate_count": int(prediction.candidate_count),
                "matching_score": prediction.matching_score,
                "matching_method": "phrase_map_assignment",
                "detection_count": int(len(boxes)),
                "latency_ms": float(prediction.latency_ms),
                "original_size": list(prediction.original_size),
                "conf_thres": self.conf_thres,
                "iou_thres": self.iou_thres,
            },
            "boxes": boxes.tolist(),
            "scores": scores.tolist(),
            "images": {"visualization": png_data_url(visualization, "RGB")},
        }

    def close(self) -> None:
        predictor, self._predictor = self._predictor, None
        release_model_resources(predictor)
