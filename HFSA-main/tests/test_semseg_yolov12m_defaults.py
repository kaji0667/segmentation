from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_semseg import parse_args
from ultralytics.nn.tasks import yaml_model_load


class YOLOv12mDefaultsTest(unittest.TestCase):
    def test_training_defaults_use_matching_medium_model_and_weights(self):
        with patch.object(sys, "argv", ["train_semseg.py"]):
            args = parse_args()

        self.assertEqual(args.model, "ultralytics/cfg/models/v12/yolov12m-semseg.yaml")
        self.assertEqual(args.weights, "pretrain_model/yolov12m.pt")

    def test_medium_model_alias_resolves_to_m_scale(self):
        model_yaml = yaml_model_load("ultralytics/cfg/models/v12/yolov12m-semseg.yaml")

        self.assertEqual(model_yaml["scale"], "m")
        self.assertTrue(model_yaml["yaml_file"].endswith("yolov12m-semseg.yaml"))


if __name__ == "__main__":
    unittest.main()
