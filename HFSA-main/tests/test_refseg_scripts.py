from pathlib import Path
import sys
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from train_semseg import parse_args, resolve_checkpoint_path


class RefSegScriptsTest(unittest.TestCase):
    def test_eval_only_arguments_are_available(self):
        with patch.object(
            sys,
            "argv",
            ["train_semseg.py", "--eval-only", "--checkpoint", "weights/example.pt"],
        ):
            args = parse_args()

        self.assertTrue(args.eval_only)
        self.assertEqual(args.checkpoint, "weights/example.pt")

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
        self.assertIn('exec "$PYTHON_BIN" train_semseg.py', text)
        self.assertIn("--val-select-metric miou", text)
        self.assertIn("--freeze-backbone", text)
        self.assertIn('--empty-mask-policy "${EMPTY_MASK_POLICY:-drop}"', text)

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
