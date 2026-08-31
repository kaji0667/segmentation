import base64
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tasks.routing import get_task_config
from tasks.routing.adapters.refseg import RefSegAdapter


def _image_payload() -> dict[str, str]:
    buffer = io.BytesIO()
    Image.new("RGB", (3, 2), (20, 40, 60)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"name": "scene.png", "data_url": f"data:image/png;base64,{encoded}"}


class _DummyPrediction:
    mask = np.array([[True, False, False], [False, True, False]], dtype=bool)
    probability = np.array([[0.9, 0.1, 0.2], [0.3, 0.8, 0.4]], dtype=np.float32)
    overlay = np.array(
        [
            [[255, 0, 0], [20, 40, 60], [20, 40, 60]],
            [[20, 40, 60], [255, 0, 0], [20, 40, 60]],
        ],
        dtype=np.uint8,
    )

    @staticmethod
    def summary():
        return {
            "threshold": 0.7,
            "foreground_ratio": 1 / 3,
            "latency_ms": 12.5,
            "original_size": [2, 3],
        }


class _DummyPredictor:
    def __init__(self):
        self.calls = []

    def predict(self, image, text):
        self.calls.append((image.size, text))
        return _DummyPrediction()


class RefSegWebAdapterTest(unittest.TestCase):
    def test_adapter_lazily_builds_predictor_and_encodes_png_results(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best_raw.pt"
            checkpoint.write_bytes(b"placeholder")
            predictor = _DummyPredictor()
            factory_calls = []

            def factory(**kwargs):
                factory_calls.append(kwargs)
                return predictor

            with patch.dict(os.environ, {"HFSA_REFSEG_CHECKPOINT": str(checkpoint)}, clear=False):
                adapter = RefSegAdapter(get_task_config("refseg"), predictor_factory=factory)
                self.assertEqual(factory_calls, [])
                result = adapter.predict(_image_payload(), "the windmill")

            self.assertEqual(len(factory_calls), 1)
            self.assertEqual(factory_calls[0]["checkpoint"], checkpoint.resolve())
            self.assertEqual(predictor.calls, [((3, 2), "the windmill")])
            self.assertEqual(result["summary"]["threshold"], 0.7)
            for key in ("overlay", "mask", "probability"):
                self.assertTrue(result["images"][key].startswith("data:image/png;base64,"))
                raw = base64.b64decode(result["images"][key].split(",", 1)[1])
                with Image.open(io.BytesIO(raw)) as image:
                    self.assertEqual(image.size, (3, 2))

    def test_adapter_rejects_invalid_browser_image_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best_raw.pt"
            checkpoint.write_bytes(b"placeholder")
            with patch.dict(os.environ, {"HFSA_REFSEG_CHECKPOINT": str(checkpoint)}, clear=False):
                adapter = RefSegAdapter(
                    get_task_config("refseg"),
                    predictor_factory=lambda **kwargs: _DummyPredictor(),
                )
                with self.assertRaisesRegex(ValueError, "图像数据解码失败"):
                    adapter.predict(
                        {"name": "broken.png", "data_url": "data:image/png;base64,not-base64"},
                        "windmill",
                    )

    def test_adapter_result_is_json_serializable(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best_raw.pt"
            checkpoint.write_bytes(b"placeholder")
            with patch.dict(os.environ, {"HFSA_REFSEG_CHECKPOINT": str(checkpoint)}, clear=False):
                adapter = RefSegAdapter(
                    get_task_config("refseg"),
                    predictor_factory=lambda **kwargs: _DummyPredictor(),
                )
                serialized = json.dumps(adapter.predict(_image_payload(), "windmill"))
            self.assertIn("foreground_ratio", serialized)


if __name__ == "__main__":
    unittest.main()
