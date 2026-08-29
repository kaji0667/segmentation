import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from tasks.refseg.inference import RefSegPredictor


class _DummySegmentationModel:
    last_instance = None

    def __init__(self, model_yaml, ch, nc, verbose):
        self.model_yaml = model_yaml
        self.nc = nc
        self.loaded_strict = None
        self.device = torch.device("cpu")
        _DummySegmentationModel.last_instance = self

    def load_state_dict(self, state, strict):
        self.loaded_strict = strict
        self.state = state

    def to(self, device):
        self.device = torch.device(device)
        return self

    def eval(self):
        return self

    def __call__(self, images, text_embedding=None, text_token_mask=None):
        if text_embedding is None or text_token_mask is None:
            raise AssertionError("Text inputs were not forwarded to the segmentation model.")
        logits = torch.full(
            (images.shape[0], 1, images.shape[2], images.shape[3]),
            -10.0,
            device=images.device,
        )
        logits[:, :, : images.shape[2] // 2] = 10.0
        return logits


class _DummyTextEncoder:
    embedding_dim = 768

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def encode(self, texts, batch_size, return_tokens):
        if texts != ["the airplane on the left"] or batch_size != 1 or not return_tokens:
            raise AssertionError("Unexpected online text-encoding request.")
        return torch.ones((1, 4, 768)), torch.tensor([[True, True, True, False]])


class RefSegInferenceTest(unittest.TestCase):
    def _checkpoint(self, directory: str) -> Path:
        path = Path(directory) / "best_raw.pt"
        torch.save(
            {
                "epoch": 44,
                "model": {"dummy": torch.tensor(1.0)},
                "args": {
                    "model": "ultralytics/cfg/models/v12/yolov12m-semseg.yaml",
                    "imgsz": 8,
                    "text_model_name": "ViT-L-14",
                    "text_pretrained": "openai",
                    "text_precision": "fp32",
                },
                "data": {"nc": 17},
                "metrics": {"best_threshold": 0.8},
            },
            path,
        )
        return path

    def test_predict_restores_original_size_and_uses_checkpoint_threshold(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = self._checkpoint(temp_dir)
            image = np.zeros((6, 10, 3), dtype=np.uint8)
            image[:, :, 1] = 100
            with patch("tasks.refseg.inference.SemanticSegmentationModel", _DummySegmentationModel), patch(
                "tasks.refseg.inference.OpenCLIPTextEncoder", _DummyTextEncoder
            ):
                predictor = RefSegPredictor(checkpoint, device="cpu")
                prediction = predictor.predict(image, "the airplane on the left")

            self.assertEqual(prediction.mask.shape, (6, 10))
            self.assertTrue(prediction.mask[:3].all())
            self.assertFalse(prediction.mask[3:].any())
            self.assertEqual(prediction.threshold, 0.8)
            self.assertEqual(predictor.checkpoint_epoch, 44)
            self.assertTrue(_DummySegmentationModel.last_instance.loaded_strict)

    def test_prediction_artifacts_are_saved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = self._checkpoint(temp_dir)
            image_path = Path(temp_dir) / "scene.png"
            Image.fromarray(np.zeros((5, 7, 3), dtype=np.uint8)).save(image_path)
            with patch("tasks.refseg.inference.SemanticSegmentationModel", _DummySegmentationModel), patch(
                "tasks.refseg.inference.OpenCLIPTextEncoder", _DummyTextEncoder
            ):
                prediction = RefSegPredictor(checkpoint, device="cpu").predict(
                    image_path, "the airplane on the left"
                )
            artifacts = prediction.save(Path(temp_dir) / "outputs")

            for path in artifacts.values():
                self.assertTrue(Path(path).is_file())
            report = json.loads(Path(artifacts["result"]).read_text(encoding="utf-8"))
            self.assertEqual(report["task"], "referring_segmentation")
            self.assertEqual(report["original_size"], [5, 7])

    def test_empty_text_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = self._checkpoint(temp_dir)
            with patch("tasks.refseg.inference.SemanticSegmentationModel", _DummySegmentationModel), patch(
                "tasks.refseg.inference.OpenCLIPTextEncoder", _DummyTextEncoder
            ):
                predictor = RefSegPredictor(checkpoint, device="cpu")
            with self.assertRaisesRegex(ValueError, "non-empty text"):
                predictor.predict(np.zeros((4, 4, 3), dtype=np.uint8), "   ")


if __name__ == "__main__":
    unittest.main()
