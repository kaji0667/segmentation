from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from tasks.counting.inference import CountingPredictor
from tasks.detection.inference import TextGuidedDetectionPredictor, TextPromptEncoder


class _DummyDetector:
    def detect(self, image_tensor, text, conf_thres, iou_thres, max_det):
        self.call = (tuple(image_tensor.shape), text, conf_thres, iou_thres, max_det)
        return {
            "boxes": torch.tensor([[80.0, 240.0, 240.0, 400.0]]),
            "scores": torch.tensor([0.85]),
        }


def _detector_factory(model, prompt_encoder, device):
    return _DummyDetector()


def _prompt_factory(**kwargs):
    return object()


class DetectionInferenceTest(unittest.TestCase):
    def test_prompt_cleaning_preserves_counting_alias_behavior(self):
        self.assertEqual(TextPromptEncoder.clean_class_name("storage-tank"), "oil storage tank")
        self.assertEqual(TextPromptEncoder.clean_prompt("ground_track_field"), "running track sports field")

    def test_detection_predictor_restores_boxes_to_original_image(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pt"
            checkpoint.write_bytes(b"placeholder")
            predictor = TextGuidedDetectionPredictor(
                checkpoint,
                device="cpu",
                imgsz=800,
                model_factory=lambda path: object(),
                prompt_encoder_factory=_prompt_factory,
                detector_factory=_detector_factory,
            )
            result = predictor.predict(Image.new("RGB", (100, 50)), "ship")
        np.testing.assert_allclose(result.boxes, [[10.0, 5.0, 30.0, 25.0]], atol=0.2)
        np.testing.assert_allclose(result.scores, [0.85])
        self.assertEqual(result.original_size, (50, 100))
        self.assertEqual(result.visualization.shape, (50, 100, 3))

    def test_counting_predictor_counts_the_shared_detector_boxes(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pt"
            checkpoint.write_bytes(b"placeholder")
            predictor = CountingPredictor(
                checkpoint,
                device="cpu",
                imgsz=800,
                model_factory=lambda path: object(),
                prompt_encoder_factory=_prompt_factory,
                detector_factory=_detector_factory,
            )
            result = predictor.predict(Image.new("RGB", (100, 50)), "ship")
        self.assertEqual(result.count, 1)
        self.assertEqual(len(result.boxes), 1)


if __name__ == "__main__":
    unittest.main()
