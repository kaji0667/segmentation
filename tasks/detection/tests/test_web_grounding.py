"""Regression: Web uses description matching; counting still keeps all boxes."""
import base64
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image

from tasks.detection.grounding import DescriptionEncoder, DescriptionGroundingPredictor
from tasks.routing import get_task_config
from tasks.routing.adapters.detection import DetectionAdapter


class WebGroundingTests(unittest.TestCase):
    def setUp(self):
        self.candidates = torch.tensor([[16., 16., 336., 416., .95, 0.],
                                        [416., 16., 816., 416., .4, 0.]])
        self.maps = [torch.tensor([[[[1., 5.], [1., 5.]]]])]
        self.model = Mock()
        self.model.eval.return_value = self.model
        def forward(*args, **kwargs):
            self.model._last_text_outputs = {"phrase_logits": self.maps}
            return torch.zeros((1, 5, 2))
        self.model.side_effect = forward
        self.encoder = Mock()
        self.encoder.encode.return_value = torch.ones((1, 768)) / np.sqrt(768)
        with patch("tasks.detection.grounding.Path.is_file", return_value=True):
            self.predictor = DescriptionGroundingPredictor(
                "best.pt", device="cpu",
                backend_factory=lambda **kw: SimpleNamespace(pt=True, nn_module=False, stride=32, model=self.model),
                encoder_factory=lambda *args: self.encoder,
            )
        self.nms = patch("tasks.detection.grounding.non_max_suppression",
                         side_effect=lambda *a, **kw: [self.candidates.clone()]).start()
        self.addCleanup(patch.stopall)

    def test_uses_phrase_match_instead_of_highest_detection_confidence(self):
        result = self.predictor.predict(Image.new("RGB", (100, 100)), "the object on the right")
        self.assertEqual(result.candidate_count, 2)
        self.assertEqual(result.boxes.shape, (1, 4))
        np.testing.assert_allclose(result.boxes, [[50, 0, 100, 50]])
        np.testing.assert_allclose(result.scores, [.4])
        self.assertEqual(result.matching_score, 5.)
        kwargs = self.model.call_args.kwargs
        self.assertEqual(kwargs["txt_vec"].shape, (1, 1, 768))
        torch.testing.assert_close(kwargs["txt_dependency_adj"], torch.ones(1, 1, 1))
        self.encoder.encode.assert_called_once_with("the object on the right")
        self.assertEqual(self.nms.call_args.kwargs["conf_thres"], .25)
        self.assertEqual(self.nms.call_args.kwargs["iou_thres"], .7)

    def test_no_candidates_returns_zero_boxes(self):
        self.candidates = torch.empty((0, 6))
        self.maps = []
        result = self.predictor.predict(Image.new("RGB", (100, 100)), "ship")
        self.assertEqual(result.boxes.shape, (0, 4))
        self.assertIsNone(result.matching_score)

    def test_missing_or_invalid_maps_never_fall_back_to_confidence(self):
        for maps in ([], [torch.full((1, 1, 2, 2), float("nan"))], [torch.ones(1, 2, 2, 2)]):
            with self.subTest(maps=maps):
                self.maps = maps
                with self.assertRaises(RuntimeError):
                    self.predictor.predict(Image.new("RGB", (100, 100)), "ship")

    def test_stale_maps_from_previous_request_are_rejected(self):
        self.model.side_effect = lambda *a, **kw: torch.zeros((1, 5, 2))
        self.model._last_text_outputs = {"phrase_logits": self.maps}
        with self.assertRaises(RuntimeError):
            self.predictor.predict(Image.new("RGB", (100, 100)), "ship")

    def test_non_square_geometry_inverts_each_actual_resize_axis(self):
        original = np.zeros((333, 799, 3), dtype=np.uint8)
        tensor, (gh, gw, left, top) = self.predictor._preprocess(original)
        self.assertEqual(tuple(tensor.shape[2:]), (352, 832))
        self.assertNotEqual(gh, gw)
        self.candidates = torch.tensor([[left + 10*gw, top + 20*gh,
                                         left + 700*gw, top + 300*gh, .7, 0.]])
        result = self.predictor.predict(original, "the bridge")
        np.testing.assert_allclose(result.boxes, [[10, 20, 700, 300]], atol=1e-4)

    def test_invalid_candidates_raise_instead_of_drawing_boxes(self):
        self.candidates[0, 0] = float("nan")
        with self.assertRaises(RuntimeError):
            self.predictor.predict(Image.new("RGB", (100, 100)), "ship")


class DescriptionEncoderTests(unittest.TestCase):
    def setUp(self):
        self.core = Mock()
        self.core.tokenizer = SimpleNamespace(context_length=77, encode=lambda text: text.split())
        self.vector = torch.ones((1, 768)) / np.sqrt(768)
        self.core.encode.return_value = self.vector.clone()
        with patch("tasks.detection.grounding.OpenCLIPTextEncoder", return_value=self.core):
            self.encoder = DescriptionEncoder(torch.device("cpu"), "local.pt")

    def test_encodes_bare_complete_description_and_caches_without_renormalizing(self):
        text = "The white airplane on the left."
        first = self.encoder.encode(text)
        self.assertTrue(torch.equal(first, self.vector))
        first.zero_()
        self.assertTrue(torch.equal(self.encoder.encode(text), self.vector))
        self.core.encode.assert_called_once_with([text], return_tokens=False)

    def test_long_text_is_rejected_before_encoding(self):
        with self.assertRaises(ValueError):
            self.encoder.encode(" ".join(["condition"]*76))
        self.core.encode.assert_not_called()


class DetectionAdapterGroundingTests(unittest.TestCase):
    def payload(self):
        buffer = io.BytesIO()
        Image.new("RGB", (100, 100)).save(buffer, "PNG")
        return {"data_url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()}

    def prediction(self, n=1):
        return SimpleNamespace(text="the airplane on the left", boxes=np.ones((n, 4)),
            scores=np.ones(n), visualization=np.zeros((100, 100, 3), dtype=np.uint8),
            original_size=(100, 100), latency_ms=1., candidate_count=4, matching_score=.2)

    def adapter(self, predictor):
        return DetectionAdapter(get_task_config("detection"), predictor_factory=lambda **kw: predictor)

    def test_chinese_translation_keeps_input_and_forwards_model_description(self):
        predictor = Mock()
        predictor.predict.return_value = self.prediction()
        with patch("pathlib.Path.is_file", return_value=True):
            adapter = self.adapter(predictor)
            result = adapter.predict(self.payload(), "左侧的飞机")
        self.assertEqual(predictor.predict.call_args.args[1], "the airplane on the left")
        self.assertEqual(result["summary"]["prompt"], "左侧的飞机")
        self.assertTrue(result["summary"]["prompt_translated"])
        self.assertEqual(result["summary"]["candidate_count"], 4)
        self.assertEqual(len(result["boxes"]), 1)

    def test_multiple_output_boxes_are_an_error_not_a_silent_first_box(self):
        predictor = Mock()
        predictor.predict.return_value = self.prediction(2)
        with patch("pathlib.Path.is_file", return_value=True):
            with self.assertRaises(RuntimeError):
                self.adapter(predictor).predict(self.payload(), "airplane")

    def test_unsupported_chinese_relations_fail_before_model_loading(self):
        factory = Mock()
        adapter = DetectionAdapter(get_task_config("detection"), predictor_factory=factory)
        with self.assertRaises(ValueError):
            adapter.predict(self.payload(), "飞机左侧的船")
        factory.assert_not_called()

    def test_counting_keeps_its_predictor_and_thresholds(self):
        config = get_task_config("counting")
        self.assertEqual(config.runtime_defaults["conf_thres"], .15)
        self.assertEqual(config.runtime_defaults["iou_thres"], .5)
        from tasks.counting.inference import CountingPredictor
        self.assertEqual(CountingPredictor.__module__, "tasks.counting.inference")
        runtime = SimpleNamespace(detect=lambda *args: {
            "boxes": torch.tensor([[80., 80., 240., 240.], [400., 80., 560., 240.]]),
            "scores": torch.tensor([.9, .7]),
        })
        with patch("pathlib.Path.is_file", return_value=True):
            counter = CountingPredictor("best.pt", device="cpu", model_factory=lambda p: object(),
                prompt_encoder_factory=lambda **kw: object(), detector_factory=lambda *args: runtime)
            result = counter.predict(Image.new("RGB", (100, 100)), "ship")
        self.assertEqual(result.count, 2)
        self.assertEqual(result.boxes.shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
