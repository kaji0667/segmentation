from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tasks.refseg.checkpoint import save_deployment_checkpoint
from tasks.refseg.engine import parse_args, resolve_checkpoint_path, validate


class _PreviewModel:
    def eval(self):
        return self

    def train(self):
        return self

    def __call__(self, images):
        return torch.full((images.shape[0], 1, images.shape[2], images.shape[3]), -10.0)

    def loss(self, batch, preds):
        return torch.tensor(0.0), torch.zeros(1)


class RefSegScriptsTest(unittest.TestCase):
    def test_eval_only_arguments_are_available(self):
        args = parse_args(["--eval-only", "--checkpoint", "weights/example.pt"])

        self.assertTrue(args.eval_only)
        self.assertEqual(args.checkpoint, "weights/example.pt")

    def test_test_preview_batch_argument_defaults_to_five(self):
        self.assertEqual(parse_args([]).test_preview_batches, 5)
        self.assertEqual(parse_args(["--test-preview-batches", "2"]).test_preview_batches, 2)

    def test_one_checkpoint_delivery_is_the_only_policy(self):
        args = parse_args([])
        self.assertFalse(hasattr(args, "resume"))
        self.assertFalse(hasattr(args, "save_resume_state"))
        engine_text = (PROJECT_ROOT / "tasks" / "refseg" / "engine.py").read_text(encoding="utf-8")
        checkpoint_text = (PROJECT_ROOT / "tasks" / "refseg" / "checkpoint.py").read_text(encoding="utf-8")
        self.assertIn('weights_dir / "best_raw.pt"', engine_text)
        self.assertNotIn('weights_dir / "best.pt"', engine_text)
        self.assertNotIn('training_state', engine_text)
        self.assertNotIn('last.pt', engine_text)
        self.assertNotIn('best.pt', checkpoint_text)

    def test_deployment_checkpoint_contains_only_inference_state(self):
        model = torch.nn.Linear(2, 1)
        args = type("Args", (), {"model": "example.yaml", "imgsz": 512})()

        with tempfile.TemporaryDirectory() as temp_dir:
            weights_dir = Path(temp_dir) / "weights"
            path = weights_dir / "best_raw.pt"
            save_deployment_checkpoint(
                path,
                model,
                epoch=4,
                args=args,
                data={"nc": 1},
                metrics={"best_threshold": 0.7},
                retry_delay=0,
            )

            self.assertEqual([item.name for item in weights_dir.iterdir()], ["best_raw.pt"])
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.assertNotIn("optimizer", checkpoint)
            self.assertNotIn("scheduler", checkpoint)
            self.assertNotIn("training_state", checkpoint)
            self.assertNotIn("rng_state", checkpoint)

    def test_scripts_use_project_relative_paths(self):
        for filename in ("train_refseg.sh", "test_refseg.sh"):
            text = (PROJECT_ROOT / "scripts" / filename).read_text(encoding="utf-8")
            self.assertIn('PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"', text)
            self.assertIn('cd "$PROJECT_ROOT"', text)
            self.assertNotRegex(text, r"[A-Za-z]:[/\\]")
            self.assertNotIn("/mnt/", text)

    def test_training_script_is_self_contained(self):
        text = (PROJECT_ROOT / "scripts" / "train_refseg.sh").read_text(encoding="utf-8")
        self.assertNotIn("run_semseg_preset.sh", text)
        self.assertIn('exec "$PYTHON_BIN" train_refseg.py', text)
        self.assertIn("--val-select-metric miou", text)
        self.assertIn("--freeze-backbone", text)
        self.assertIn('--empty-mask-policy "${EMPTY_MASK_POLICY:-drop}"', text)
        self.assertNotIn("--test-after-train", text)
        self.assertNotIn("TEST_AFTER_TRAIN", text)
        self.assertNotIn("TEST_PREVIEW_BATCHES", text)

    def test_evaluation_script_uses_the_dedicated_entry(self):
        text = (PROJECT_ROOT / "scripts" / "test_refseg.sh").read_text(encoding="utf-8")
        self.assertIn('exec "$PYTHON_BIN" test_refseg.py', text)
        self.assertNotIn("--eval-only", text)
        self.assertIn('--test-preview-batches "${TEST_PREVIEW_BATCHES:-5}"', text)

    def test_validation_saves_only_available_requested_preview_batches(self):
        batch = {
            "img": torch.zeros((1, 3, 4, 4), dtype=torch.float32),
            "mask": torch.zeros((1, 4, 4), dtype=torch.long),
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            preview_paths = [Path(tmp_dir) / f"test_batch{i}_pred.jpg" for i in range(5)]
            with patch("tasks.refseg.engine.save_preview") as save_preview_mock:
                metrics = validate(
                    _PreviewModel(),
                    [batch, batch],
                    torch.device("cpu"),
                    2,
                    255,
                    0,
                    None,
                    np.zeros((2, 3), dtype=np.uint8),
                    val_thresholds=[0.5],
                    preview_paths=preview_paths,
                )

        self.assertEqual(save_preview_mock.call_count, 2)
        self.assertEqual([call.args[0] for call in save_preview_mock.call_args_list], preview_paths[:2])
        self.assertEqual(metrics["preview_files"], [str(path) for path in preview_paths[:2]])

    def test_checkpoint_can_be_resolved_relative_to_project_root(self):
        checkpoint = PROJECT_ROOT / "tests" / "_checkpoint_path_test.pt"
        checkpoint.touch()
        try:
            resolved = resolve_checkpoint_path("tests/_checkpoint_path_test.pt")
            self.assertEqual(resolved, checkpoint.resolve())
        finally:
            checkpoint.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
