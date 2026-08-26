from pathlib import Path
import unittest

import torch

from tasks.counting.config import CountingTextConfig
from tasks.counting.evaluation import CountingEvaluator
from train_counting import CountingTrainingApplication
from test_counting import CountingEvaluationApplication
from ultralytics.nn.modules import CountingDetect, Detect
from ultralytics.nn.tasks import DetectionModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CountingIntegrationTest(unittest.TestCase):
    def test_counting_head_preserves_detect_forward(self):
        torch.manual_seed(7)
        base_head = Detect(nc=1, ch=(16, 32, 64)).train()
        counting_head = CountingDetect(nc=1, ch=(16, 32, 64)).train()
        counting_head.load_state_dict(base_head.state_dict(), strict=True)
        features = [
            torch.randn(2, 16, 16, 16),
            torch.randn(2, 32, 8, 8),
            torch.randn(2, 64, 4, 4),
        ]
        expected = base_head([feature.clone() for feature in features])
        actual = counting_head([feature.clone() for feature in features])
        self.assertEqual(len(expected), len(actual))
        for expected_level, actual_level in zip(expected, actual):
            torch.testing.assert_close(actual_level, expected_level)

    def test_counting_yaml_resolves_m_scale_and_head(self):
        alias = PROJECT_ROOT / "ultralytics" / "cfg" / "models" / "v12" / "yolov12m-counting.yaml"
        model = DetectionModel(str(alias), nc=1, verbose=False)
        self.assertEqual(model.yaml.get("scale"), "m")
        self.assertIsInstance(model.model[-1], CountingDetect)

    def test_counting_config_applies_original_task_overrides(self):
        config = CountingTextConfig.build({"enabled": False, "visual_attr_include_geom": True})
        self.assertTrue(config["enabled"])
        self.assertFalse(config["visual_attr_include_geom"])
        self.assertFalse(config["spatial_encoding_enabled"])
        self.assertEqual(config["lambda_relation"], 0.0)
        self.assertEqual(config["lambda_spatial_quadrant"], 0.0)
        self.assertTrue(config["multi_proj_enabled"])
        self.assertTrue(config["film_enabled"])
        self.assertTrue(config["cross_attn_enabled"])

    def test_counting_metrics_match_original_formulas(self):
        evaluator = CountingEvaluator()
        evaluator.update(2, 2, class_name="ship")
        evaluator.update(1, 3, class_name="ship")
        evaluator.update(5, 4, class_name="vehicle")
        metrics = evaluator.compute_metrics()
        self.assertEqual(metrics["total_samples"], 3)
        self.assertAlmostEqual(metrics["exact_match"], 100.0 / 3.0)
        self.assertAlmostEqual(metrics["mae"], 1.0)
        self.assertAlmostEqual(metrics["rmse"], (5.0 / 3.0) ** 0.5)

    def test_counting_cli_defaults_are_project_relative(self):
        train_args = CountingTrainingApplication.build_parser().parse_args([])
        test_args = CountingEvaluationApplication.build_parser().parse_args([])
        for value in (train_args.model, train_args.weights, test_args.weights, test_args.voc_root, test_args.save_dir):
            self.assertNotRegex(str(value), r"^[A-Za-z]:[/\\]")
            self.assertNotIn("/mnt/", str(value))
        self.assertIn("yolov12m-counting.yaml", train_args.model)

    def test_counting_scripts_match_task_script_contract(self):
        for filename in ("train_counting.sh", "test_counting.sh"):
            text = (PROJECT_ROOT / "scripts" / filename).read_text(encoding="utf-8")
            self.assertIn('PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"', text)
            self.assertIn('cd "$PROJECT_ROOT"', text)
            self.assertNotRegex(text, r"[A-Za-z]:[/\\]")
            self.assertNotIn("/mnt/", text)
        train_text = (PROJECT_ROOT / "scripts" / "train_counting.sh").read_text(encoding="utf-8")
        test_text = (PROJECT_ROOT / "scripts" / "test_counting.sh").read_text(encoding="utf-8")
        self.assertIn('exec "$PYTHON_BIN" train_counting.py', train_text)
        self.assertIn("yolov12m-counting.yaml", train_text)
        self.assertIn('exec "$PYTHON_BIN" test_counting.py', test_text)


if __name__ == "__main__":
    unittest.main()
