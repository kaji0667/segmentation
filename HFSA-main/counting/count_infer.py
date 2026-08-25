"""Class-based preprocessing, prompting, and inference for object counting."""

from typing import Any, Dict, Mapping, Tuple

import cv2
import numpy as np
import open_clip
import torch

from ultralytics.utils.ops import non_max_suppression


class CountingImagePreprocessor:
    """Apply the original letterbox and normalization path."""

    def __init__(self, image_size: int = 800) -> None:
        if int(image_size) <= 0:
            raise ValueError("image_size must be positive")
        self.image_size = int(image_size)

    def __call__(self, image: np.ndarray) -> Tuple[torch.Tensor, float, Tuple[float, float]]:
        if image is None or image.ndim != 3:
            raise ValueError("Expected a BGR image shaped [H, W, C]")
        shape = image.shape[:2]
        ratio = min(self.image_size / shape[0], self.image_size / shape[1])
        new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
        dw = (self.image_size - new_unpad[0]) / 2
        dh = (self.image_size - new_unpad[1]) / 2

        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        tensor = image[:, :, ::-1].transpose(2, 0, 1)
        tensor = torch.from_numpy(np.ascontiguousarray(tensor)).float() / 255.0
        return tensor.unsqueeze(0), ratio, (dw, dh)


class CountingPromptEncoder:
    """Cache OpenCLIP class prompts used by the teammate's evaluator."""

    CLASS_ALIASES: Mapping[str, str] = {
        "airplane": "airplane",
        "airport": "airport runway",
        "baseball diamond": "baseball field",
        "basketball court": "basketball court",
        "ground track field": "running track sports field",
        "storage tank": "oil storage tank",
        "tennis court": "tennis court",
        "expressway service area": "highway service area",
        "expressway toll station": "toll gate station",
        "vehicle": "small vehicle car",
        "ship": "ship vessel boat",
        "windmill": "wind turbine windmill",
    }

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str | torch.device = "cpu",
        prompt_template: str = "a satellite remote sensing photo of {class_name}",
    ) -> None:
        self.device = torch.device(device)
        self.prompt_template = str(prompt_template)
        if "{class_name}" not in self.prompt_template:
            raise ValueError("prompt_template must contain '{class_name}'")
        self.clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.clip_model = self.clip_model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.cache: Dict[str, torch.Tensor] = {}

    @classmethod
    def clean_class_name(cls, raw_name: str) -> str:
        name = str(raw_name).replace("-", " ").replace("_", " ").strip().lower()
        return cls.CLASS_ALIASES.get(name, name)

    def encode(self, raw_class_name: str) -> torch.Tensor:
        cache_key = str(raw_class_name).strip().lower()
        if cache_key in self.cache:
            return self.cache[cache_key]
        natural_name = self.clean_class_name(cache_key)
        prompt = self.prompt_template.format(class_name=natural_name)
        tokens = self.tokenizer([prompt]).to(self.device)
        with torch.no_grad():
            feature = self.clip_model.encode_text(tokens)
            feature = feature / feature.norm(dim=-1, keepdim=True)
        text_vector = feature.unsqueeze(1)
        self.cache[cache_key] = text_vector
        return text_vector


class ObjectCounter:
    """Count prompted objects by applying NMS to text-guided detections."""

    def __init__(
        self,
        model_instance: Any,
        prompt_encoder: CountingPromptEncoder,
        device: str | torch.device,
    ) -> None:
        self.device = torch.device(device)
        self.model = model_instance.to(self.device).eval()
        self.prompt_encoder = prompt_encoder

    def count(
        self,
        image_tensor: torch.Tensor,
        target_class: str,
        conf_thres: float = 0.15,
        iou_thres: float = 0.50,
        max_det: int = 300,
    ) -> Dict[str, Any]:
        text_vector = self.prompt_encoder.encode(target_class)
        token_mask = torch.ones((1, 1), device=self.device, dtype=torch.bool)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            predictions = self.model(
                image_tensor,
                txt_vec=text_vector,
                txt_token_mask=token_mask,
            )

        raw_prediction = predictions[0] if isinstance(predictions, tuple) else predictions
        detections = non_max_suppression(
            raw_prediction,
            conf_thres=float(conf_thres),
            iou_thres=float(iou_thres),
            agnostic=True,
            max_det=int(max_det),
        )[0]

        if detections is not None and len(detections) > 0:
            boxes = detections[:, :4].detach().cpu()
            scores = detections[:, 4].detach().cpu()
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            scores = torch.empty((0,), dtype=torch.float32)

        return {
            "target_class": str(target_class),
            "pred_count": int(len(boxes)),
            "pred_boxes": boxes,
            "pred_scores": scores,
        }
