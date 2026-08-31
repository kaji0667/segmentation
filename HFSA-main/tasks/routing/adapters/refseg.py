"""Web adapter for single-image text-guided semantic segmentation."""

from __future__ import annotations

import base64
import binascii
import gc
import io
import os
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from ..config import TaskConfig


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAX_DECODED_IMAGE_BYTES = 40 * 1024 * 1024
PredictorFactory = Callable[..., Any]


def _resolve_checkpoint(config: TaskConfig) -> Path:
    configured = os.environ.get("HFSA_REFSEG_CHECKPOINT", config.checkpoint).strip()
    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _decode_data_url(value: Any) -> Image.Image:
    if not isinstance(value, Mapping):
        raise ValueError("图像输入格式无效，请重新选择本地图像。")
    data_url = value.get("data_url")
    if not isinstance(data_url, str) or not data_url.strip():
        raise ValueError("图像数据为空，请重新选择本地图像。")

    header, separator, encoded = data_url.partition(",")
    if not separator or not header.lower().startswith("data:image/") or ";base64" not in header.lower():
        raise ValueError("仅支持 base64 编码的图像数据。")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("图像数据解码失败，请重新上传。") from exc
    if not raw:
        raise ValueError("图像数据为空，请重新上传。")
    if len(raw) > MAX_DECODED_IMAGE_BYTES:
        raise ValueError("图像文件过大，请上传不超过 40 MiB 的图像。")

    try:
        with Image.open(io.BytesIO(raw)) as source:
            source.load()
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("无法识别上传的图像，请使用 JPG、PNG、WEBP 或 TIFF。") from exc
    if image.width <= 0 or image.height <= 0:
        raise ValueError("上传的图像尺寸无效。")
    return image


def _png_data_url(array: np.ndarray, mode: str) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array, mode=mode).save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _default_predictor_factory(**kwargs: Any) -> Any:
    from tasks.refseg.inference import RefSegPredictor

    return RefSegPredictor(**kwargs)


class RefSegAdapter:
    """Translate the browser JSON contract into the task-owned predictor API."""

    def __init__(
        self,
        config: TaskConfig,
        predictor_factory: PredictorFactory | None = None,
    ) -> None:
        self.config = config
        self.checkpoint = _resolve_checkpoint(config)
        self.device = os.environ.get("HFSA_REFSEG_DEVICE", "auto").strip() or "auto"
        self._predictor_factory = predictor_factory or _default_predictor_factory
        self._predictor: Any | None = None
        self._lock = threading.Lock()

    def _get_predictor(self) -> Any:
        if self._predictor is None:
            if not self.checkpoint.is_file():
                raise FileNotFoundError(f"RefSeg checkpoint not found: {self.checkpoint}")
            self._predictor = self._predictor_factory(
                checkpoint=self.checkpoint,
                device=self.device,
            )
        return self._predictor

    def predict(self, image: Any, text: str) -> dict[str, Any]:
        prompt = str(text).strip()
        if not prompt:
            raise ValueError("语义分割需要填写目标描述。")
        decoded_image = _decode_data_url(image)

        with self._lock:
            prediction = self._get_predictor().predict(decoded_image, prompt)

        mask = np.asarray(prediction.mask, dtype=bool)
        probability = np.asarray(prediction.probability, dtype=np.float32)
        overlay = np.asarray(prediction.overlay, dtype=np.uint8)
        if mask.ndim != 2 or probability.shape != mask.shape:
            raise RuntimeError("RefSeg prediction mask and probability shapes do not match.")
        if overlay.shape != (*mask.shape, 3):
            raise RuntimeError("RefSeg overlay must match the mask size and contain three RGB channels.")

        return {
            "summary": prediction.summary(),
            "images": {
                "overlay": _png_data_url(overlay, "RGB"),
                "mask": _png_data_url(mask.astype(np.uint8) * 255, "L"),
                "probability": _png_data_url(
                    np.clip(probability * 255.0, 0, 255).astype(np.uint8),
                    "L",
                ),
            },
        }

    def close(self) -> None:
        predictor = self._predictor
        self._predictor = None
        if predictor is None:
            return
        del predictor
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
