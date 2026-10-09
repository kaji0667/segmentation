"""Synthetic predictor fixtures exercise routing/formatting, not model accuracy."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hfsa_adapter import HFSAAdapter, parse_question


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


if __name__ == "__main__":
    unittest.main()
