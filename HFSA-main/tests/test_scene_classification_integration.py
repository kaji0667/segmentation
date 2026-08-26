import sys
from pathlib import Path
import tempfile
import unittest

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.classification.metrics import SceneClassificationMetrics
from tasks.classification.data import SceneDataModule
from tasks.classification.prepare import VRSBenchSceneDatasetBuilder
from ultralytics.nn.modules import SceneClassifyHead
from ultralytics.nn.tasks import ClassificationModel, guess_model_task
from ultralytics.utils.loss import SceneClassificationLoss


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

    def test_scene_classification_loss_matches_cross_entropy(self):
        logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, 0.2]], requires_grad=True)
        labels = torch.tensor([0, 1])
        actual = SceneClassificationLoss()(logits, labels)
        expected = torch.nn.functional.cross_entropy(logits, labels)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertIsNotNone(logits.grad)

    def test_scene_classification_scripts_are_self_contained_and_project_relative(self):
        for filename in ("train_classification.sh", "test_classification.sh"):
            text = (ROOT / "scripts" / filename).read_text(encoding="utf-8")
            self.assertIn('PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"', text)
            self.assertIn('cd "$PROJECT_ROOT"', text)
            self.assertIn('PYTHON_BIN="${PYTHON_BIN:-python}"', text)
            self.assertNotIn("D:\\", text)
            self.assertNotIn("/mnt/", text)

        train_text = (ROOT / "scripts" / "train_classification.sh").read_text(encoding="utf-8")
        self.assertIn('DATA_DIR="${DATA_DIR:-data/VRSBench_scene}"', train_text)
        self.assertIn('VOC_ROOT="${VOC_ROOT:-data/VRSBench}"', train_text)
        self.assertIn('prepare_classification_data.py', train_text)
        self.assertIn('--output-dir "$DATA_DIR"', train_text)

    def test_vrsbench_scene_selection_keeps_exactly_one_scene_class(self):
        select = VRSBenchSceneDatasetBuilder.select_scene_label
        self.assertEqual(select(["ship", "harbor", "harbor"]), "harbor")
        self.assertIsNone(select(["ship", "vehicle"]))
        self.assertIsNone(select(["bridge", "harbor"]))

    def test_eval_transform_does_not_require_a_dataset_directory(self):
        transform = SceneDataModule.build_transform(32, train=False)
        tensor = transform(Image.new("RGB", (12, 10), color=(128, 64, 32)))
        self.assertEqual(tuple(tensor.shape), (3, 32, 32))

    def test_all_eval_combines_explicit_splits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for split in ("train", "val", "test"):
                class_dir = root / split / "airport"
                class_dir.mkdir(parents=True)
                Image.new("RGB", (8, 8), color=(20, 40, 60)).save(class_dir / f"{split}.jpg")
            loader, classes = SceneDataModule(root, imgsz=16, batch=2, workers=0).build_eval("all")
            self.assertEqual(classes, ["airport"])
            self.assertEqual(len(loader.dataset), 3)


if __name__ == "__main__":
    unittest.main()
