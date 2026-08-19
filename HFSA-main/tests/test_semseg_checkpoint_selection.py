from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_semseg import checkpoint_improvement_flags, select_test_checkpoint


class SemsegCheckpointSelectionTest(unittest.TestCase):
    def test_raw_best_ignores_min_delta(self) -> None:
        improved, raw_improved = checkpoint_improvement_flags(
            fitness=0.5005,
            best_fitness=0.5000,
            best_raw_fitness=0.5000,
            min_delta=0.001,
        )

        self.assertFalse(improved)
        self.assertTrue(raw_improved)

    def test_best_still_respects_min_delta(self) -> None:
        improved, raw_improved = checkpoint_improvement_flags(
            fitness=0.5011,
            best_fitness=0.5000,
            best_raw_fitness=0.5012,
            min_delta=0.001,
        )

        self.assertTrue(improved)
        self.assertFalse(raw_improved)

    def test_test_checkpoint_prefers_raw_best(self) -> None:
        with TemporaryDirectory() as temp_dir:
            weights_dir = Path(temp_dir)
            best_path = weights_dir / "best.pt"
            best_raw_path = weights_dir / "best_raw.pt"
            best_path.touch()

            self.assertEqual(select_test_checkpoint(weights_dir), best_path)

            best_raw_path.touch()
            self.assertEqual(select_test_checkpoint(weights_dir), best_raw_path)


if __name__ == "__main__":
    unittest.main()
