import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classification.metrics import SceneClassificationMetrics
from classification.prepare import VRSBenchSceneDatasetBuilder
from ultralytics.nn.modules import SceneClassifyHead
from ultralytics.nn.tasks import ClassificationModel, guess_model_task


class SceneClassificationIntegrationTest(unittest.TestCase):
    def test_scene_classify_head_preserves_source_tensor_contract(self):
        head = SceneClassifyHead(nc=7, proj_dim=16, hidden_dim=32, dropout=0.1, use_gem=True, ch=(24, 48, 96))
        head.eval()
        outputs = head(
            (
                torch.rand(2, 24, 16, 16),
                torch.rand(2, 48, 8, 8),
                torch.rand(2, 96, 4, 4),
            )
        )
        self.assertEqual(outputs.shape, (2, 7))
        self.assertIn("proj_p3.0.weight", head.state_dict())
        self.assertIn("attn_p5.attention.3.bias", head.state_dict())
        self.assertIn("gem_p4.p", head.state_dict())

    def test_scene_classification_yaml_uses_integrated_head_and_m_scale(self):
        model = ClassificationModel(
            ROOT / "ultralytics/cfg/models/v12/yolov12m-classification.yaml", nc=5, verbose=False
        )
        self.assertIsInstance(model.model[-1], SceneClassifyHead)
        self.assertEqual(model.model[-1].nc, 5)
        self.assertEqual(model.yaml["scale"], "m")
        self.assertEqual(guess_model_task(model.yaml), "classify")

    def test_scene_classification_metrics_match_known_confusion(self):
        metrics = SceneClassificationMetrics(3, ["a", "b", "c"])
        logits = torch.tensor([[4.0, 1.0, 0.0], [0.0, 3.0, 1.0], [0.0, 2.0, 3.0], [2.0, 3.0, 0.0]])
        labels = torch.tensor([0, 1, 2, 0])
        metrics.update(logits, labels)
        report = metrics.compute()
        self.assertEqual(report["evaluated_samples"], 4)
        self.assertEqual(report["top1_accuracy"], 0.75)
        self.assertEqual(report["top5_accuracy"], 1.0)
        self.assertEqual(report["confusion_matrix"], [[1, 1, 0], [0, 1, 0], [0, 0, 1]])

    def test_scene_classification_scripts_are_self_contained_and_project_relative(self):
        for filename in ("train_classification.sh", "test_classification.sh"):
            text = (ROOT / "scripts" / filename).read_text(encoding="utf-8")
            self.assertIn('PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"', text)
            self.assertIn('cd "$PROJECT_ROOT"', text)
            self.assertIn('PYTHON_BIN="${PYTHON_BIN:-python}"', text)
            self.assertNotIn("D:\\", text)
            self.assertNotIn("/mnt/", text)

    def test_vrsbench_scene_selection_keeps_exactly_one_scene_class(self):
        select = VRSBenchSceneDatasetBuilder.select_scene_label
        self.assertEqual(select(["ship", "harbor", "harbor"]), "harbor")
        self.assertIsNone(select(["ship", "vehicle"]))
        self.assertIsNone(select(["bridge", "harbor"]))


if __name__ == "__main__":
    unittest.main()
