from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TaskPackageLayoutTest(unittest.TestCase):
    def test_three_integrated_tasks_live_under_tasks(self):
        for task in ("refseg", "classification", "counting"):
            self.assertTrue((PROJECT_ROOT / "tasks" / task / "__init__.py").is_file())

        self.assertFalse((PROJECT_ROOT / "classification").exists())
        self.assertFalse((PROJECT_ROOT / "counting").exists())

    def test_root_entries_are_thin_task_dispatchers(self):
        expected_imports = {
            "train_refseg.py": "from tasks.refseg import RefSegTrainingApplication",
            "test_refseg.py": "from tasks.refseg import RefSegEvaluationApplication",
            "train_classification.py": "from tasks.classification.train import SceneClassificationTrainingApplication",
            "test_classification.py": "from tasks.classification.test import SceneClassificationEvaluationApplication",
            "train_counting.py": "from tasks.counting.train import CountingTrainingApplication",
            "test_counting.py": "from tasks.counting.test import CountingEvaluationApplication",
        }
        for filename, expected in expected_imports.items():
            text = (PROJECT_ROOT / filename).read_text(encoding="utf-8")
            self.assertIn(expected, text)

    def test_detection_entries_are_not_reorganized(self):
        self.assertTrue((PROJECT_ROOT / "train.py").is_file())
        self.assertTrue((PROJECT_ROOT / "val.py").is_file())
        self.assertFalse((PROJECT_ROOT / "tasks" / "detection").exists())


if __name__ == "__main__":
    unittest.main()
