"""Synthetic predictor fixtures exercise routing/formatting, not model accuracy."""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hfsa_adapter import AdapterFailure, HFSAAdapter, parse_question


def request(question, kind="short_text", **kwargs):
    constraint = {"type": kind, **kwargs.pop("constraint", {})}
    return {"question": question, "response_constraint": constraint, **kwargs}


class RoutingTests(unittest.TestCase):
    def test_english_and_chinese_count_keep_appearance(self):
        for question, expected in (
            ("How many small white ships are there in the image?", "small white ships"),
            ("How many ships are in the image?", "ships"),
            ("How many ships are there? Only answer with an integer.", "ships"),
            ("Count ships in the image.", "ships"),
            ("What is the number of ships in the image?", "ships"),
            ("图中有多少艘船？", "船"),
            ("请问图中有多少艘船？只回答数字。", "船"),
            ("图中的飞机有多少架？", "飞机"),
            ("请统计图中飞机的数量。", "飞机"),
        ):
            with self.subTest(question=question):
                plan = parse_question(request(question, "integer"))
                self.assertEqual((plan.task, plan.target), ("counting", expected))

    def test_bbox_routes_and_keeps_referring_qualifiers(self):
        for question, task, target in (
            ("框出图中灰色的小风车。", "refseg", "灰色的小风车"),
            ("Locate the airplane near the bridge.", "refseg", "the airplane near the bridge"),
            ("The gray small windmill", "refseg", "The gray small windmill"),
            ("Detect an airplane in the image.", "detection", "an airplane"),
            ("检测图中的飞机", "detection", "飞机"),
        ):
            plan = parse_question(request(question, "bbox"))
            self.assertEqual((plan.task, plan.target), (task, target))

    def test_known_presence_and_scene_forms(self):
        self.assertEqual(parse_question(request("图中是否有飞机？", "enum")).task, "presence")
        self.assertEqual(parse_question(request("Is there a ship in the image? Answer Yes or No.", "enum")).target, "ship")
        self.assertEqual(parse_question(request("Which scene category best describes the image?", "single_choice")).task, "classification")

    def test_bbox_wrapper_preserves_complete_description(self):
        description = "The small red warehouse at the bottom right. Above it is a road."
        for prefix in (
            "Given a satellite image, identify the bounding box in pixel coordinates. Description: ",
            "Return the bounding box. Please answer with coordinates only.\nDESCRIPTION: ",
        ):
            plan = parse_question(request(prefix + description, "bbox"))
            self.assertEqual((plan.task, plan.target), ("refseg", description.rstrip(".")))

    def test_bbox_wrapper_without_target_or_with_unrelated_task_fails(self):
        with self.assertRaises(AdapterFailure) as context:
            parse_question(request("Return the bounding box. Description: ", "bbox"))
        self.assertEqual(context.exception.code, "missing_target")
        with self.assertRaises(AdapterFailure):
            parse_question(request("What color is it? Description: the roof", "bbox"))
        with self.assertRaises(AdapterFailure):
            parse_question(request("What color is the bounding box? Description: the roof", "bbox"))
        with self.assertRaises(AdapterFailure):
            parse_question(request("Return the bounding box. Description: a warehouse", "short_text"))


    def test_unsupported_semantics_do_not_fall_back(self):
        cases = [
            request("Describe the image."), request("What is the area?", "integer"),
            request("What color is the scene?"),
            request("Is there anything in the image? Answer Yes or No.", "enum"),
            request("How many ships are near the bridge?", "integer"),
            request("How many ships and airplanes are there?", "integer"),
            request("Detect all airplanes.", "bbox"),
            request("What changed?", "enum", constraint={"question_form": "change_region"}),
        ]
        for value in cases:
            with self.subTest(question=value["question"]), self.assertRaises(ValueError):
                parse_question(value)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.image = Path("synthetic.png")
        self.classification = Mock()
        self.classification.class_names = ["airport", "windmill", "bridge"]
        self.classification.predict.return_value = {"predictions": [
            {"class_name": "bridge", "probability": .7},
            {"class_name": "windmill", "probability": .2},
            {"class_name": "airport", "probability": .1},
        ]}
        self.detection = Mock()
        self.detection.predict.return_value = SimpleNamespace(
            detection_count=2, original_size=(40, 80),
            boxes=np.array([[1., 2., 4., 6.], [30.2, 20.8, 70.1, 35.2]]),
            scores=np.array([.2, .8]),
        )
        self.counting = Mock()
        self.counting.predict.return_value = SimpleNamespace(count=3)
        mask = np.zeros((40, 80), dtype=bool)
        mask[10:16, 20:24] = True
        self.refseg = Mock()
        self.refseg.predict.return_value = SimpleNamespace(mask=mask)
        self.translate = Mock(side_effect=lambda text: (text, False))
        self.adapter = HFSAAdapter(self.classification, self.detection, self.counting, self.refseg, self.translate)

    def predict(self, value):
        return self.adapter.predict(value, [self.image])

    def test_scene_choice_uses_image_scores_and_returns_letter(self):
        value = request("Which scene category is shown?", "single_choice",
                        choices={"A": "机场", "B": "风车"}, constraint={"values": ["A", "B"]})
        self.assertEqual(self.predict(value), "B")
        self.classification.predict.assert_called_once_with(self.image, topk=3)
        self.detection.predict.assert_not_called()

    def test_scene_enum_returns_exact_allowed_alias(self):
        value = request("图中属于什么场景？", "enum", constraint={"values": ["wind turbine", "桥梁"]})
        self.assertEqual(self.predict(value), "桥梁")

    def test_scene_rejects_unknown_or_duplicate_options_before_inference(self):
        for values in (["windmill", "forest"], ["windmill", "风车"]):
            with self.assertRaises(ValueError):
                self.predict(request("What scene is this?", "enum", constraint={"values": values}))
        self.classification.predict.assert_not_called()

    def test_count_integer_is_string_and_uses_counting_weights(self):
        self.assertEqual(self.predict(request("图中有多少架飞机？", "integer")), "3")
        self.counting.predict.assert_called_once_with(self.image, "airplane")
        self.detection.predict.assert_not_called()

    def test_count_options_map_actual_count_without_clamping(self):
        self.assertEqual(self.predict(request("How many ships?", "single_choice",
                         choices={"A": "2", "B": "3"}, constraint={"values": ["A", "B"]})), "B")
        with self.assertRaises(ValueError):
            self.predict(request("How many ships?", "integer", constraint={"minimum": 5}))
        with self.assertRaises(ValueError):
            self.predict(request("How many ships?", "enum", constraint={"values": ["1", "2"]}))

    def test_presence_positive_negative_and_choice(self):
        value = request("Are there ships in the image?", "enum", constraint={"values": ["No", "Yes"]})
        self.assertEqual(self.predict(value), "Yes")
        self.detection.predict.return_value.detection_count = 0
        self.assertEqual(self.predict(value), "No")
        self.assertEqual(self.predict(request("图中是否有飞机？", "single_choice",
                         choices={"A": "是", "B": "否"}, constraint={"values": ["A", "B"]})), "B")

    def test_refseg_pixel_box_preserves_qualifiers_and_empty_mask_fails(self):
        value = request("Locate the gray small windmill.", "bbox", image_width=80, image_height=40)
        self.assertEqual(self.predict(value), [20, 10, 24, 16])
        self.refseg.predict.assert_called_once_with(self.image, "the gray small windmill")
        self.refseg.predict.return_value.mask[:] = False
        with self.assertRaises(ValueError):
            self.predict(value)

    def test_wrapped_refseg_request_calls_model_with_full_description(self):
        description = "The red warehouse at the lower left. To its right is a road"
        value = request("Identify the bounding box. Answer with coordinates only. Description: " + description,
                        "bbox", image_width=80, image_height=40)
        self.assertEqual(self.predict(value), [20, 10, 24, 16])
        self.refseg.predict.assert_called_once_with(self.image, description)
        self.detection.predict.assert_not_called()


    def test_detection_uses_highest_confidence_native_box_and_dimensions(self):
        value = request("Detect a ship.", "bbox", image_width=80, image_height=40)
        self.assertEqual(self.predict(value), [30, 20, 71, 36])
        self.refseg.predict.assert_not_called()
        value["image_width"] = 40
        with self.assertRaises(ValueError):
            self.predict(value)

    def test_two_images_rejected_without_running_any_predictor(self):
        with self.assertRaises(ValueError):
            self.adapter.predict(request("How many ships?", "integer"), [self.image, self.image])
        self.counting.predict.assert_not_called()

    def test_abstract_count_target_is_not_sent_to_model(self):
        with self.assertRaises(ValueError):
            self.predict(request("How many colors are there in the image?", "integer"))
        self.counting.predict.assert_not_called()


class DiagnosticTests(unittest.TestCase):
    # Reuse predictor fixtures, without inheriting their test cases a second time.
    setUp = AdapterTests.setUp
    predict = AdapterTests.predict

    def test_expected_failure_is_identified_without_logging_question(self):
        value = request("Describe this private question text.", item_id="TEST_DIAGNOSTIC",
                        request_id="test:diagnostic:1")
        output = io.StringIO()
        with redirect_stderr(output), self.assertRaises(AdapterFailure):
            self.predict(value)
        record = json.loads(output.getvalue())
        self.assertEqual(record["reason"], "unsupported_question_form")
        self.assertEqual(record["item_id"], "TEST_DIAGNOSTIC")
        self.assertIsNone(record["task"])
        self.assertNotIn("private question", output.getvalue())

    def test_unexpected_exception_text_and_paths_remain_private(self):
        self.classification.predict.side_effect = RuntimeError("private-prompt private-key /private/model/path")
        value = request("Which scene category is shown?", item_id="TEST_DIAGNOSTIC")
        output = io.StringIO()
        with redirect_stderr(output), self.assertRaises(RuntimeError):
            self.predict(value)
        record = json.loads(output.getvalue())
        self.assertEqual((record["task"], record["reason"], record["error_type"]),
                         ("classification", "unexpected_exception", "RuntimeError"))
        for secret in ("private-prompt", "private-key", "/private/model/path"):
            self.assertNotIn(secret, output.getvalue())

    def test_success_and_two_image_failure_keep_existing_behavior(self):
        value = request("How many ships?", "integer", item_id="TEST_DIAGNOSTIC")
        output = io.StringIO()
        with redirect_stderr(output):
            self.assertEqual(self.predict(value), "3")
            with self.assertRaises(AdapterFailure):
                self.adapter.predict(value, [self.image, self.image])
        success, failure = map(json.loads, output.getvalue().splitlines())
        self.assertEqual((success["status"], success["task"]), ("succeeded", "counting"))
        self.assertNotIn("answer", success)
        self.assertEqual((failure["reason"], failure["image_count"]), ("unsupported_image_count", 2))
        self.assertEqual(self.counting.predict.call_count, 1)


    def test_failed_request_is_privately_captured_without_credentials_or_gold(self):
        value = request("Describe this private replay question.", "short_text",
                        protocol_version="2.0", item_id="TEST_REPLAY", request_id="test:replay:1")
        value.update(headers={"Authorization": "secret-header"}, api_key="secret-key", answer="secret-gold")
        value["images"] = [{"asset_id": "a" * 32, "sha256": "b" * 64,
                            "mime_type": "image/png", "path": "/secret-path"}]
        value["response_constraint"]["api_key"] = "secret-constraint"
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"HFSA_API_REPLAY_DIR": directory}):
            output = io.StringIO()
            for _ in range(2):
                with redirect_stderr(output), self.assertRaises(AdapterFailure):
                    self.predict(value)
            files = list(Path(directory).glob("*.json"))
            self.assertEqual(len(files), 1)
            saved = files[0].read_text(encoding="utf-8")
            payload = json.loads(saved)
        self.assertEqual(payload["question"], value["question"])
        self.assertNotIn(value["question"], output.getvalue())
        self.assertEqual(payload["response_constraint"], {"type": "short_text"})
        for secret in ("secret-header", "secret-key", "secret-gold", "/secret-path", "secret-constraint"):
            self.assertNotIn(secret, saved)

    def test_capture_failure_does_not_change_successful_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid_directory = Path(directory) / "private-path"
            invalid_directory.write_text("occupied", encoding="utf-8")
            output = io.StringIO()
            with patch.dict("os.environ", {"HFSA_API_REPLAY_DIR": str(invalid_directory)}), redirect_stderr(output):
                self.assertEqual(self.predict(request("How many ships?", "integer")), "3")
        rows = list(map(json.loads, output.getvalue().splitlines()))
        self.assertEqual(rows[0]["event"], "hfsa_api_replay_capture")
        self.assertEqual(rows[-1]["status"], "succeeded")
        self.assertNotIn("private-path", output.getvalue())


if __name__ == "__main__":
    unittest.main()
