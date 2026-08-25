import argparse
import csv
import tempfile
import unittest
from pathlib import Path

import torch

from train_semseg import recover_training_state, save_checkpoint, trim_uncommitted_results, validate_resume_args


class SemsegCheckpointRecoveryTest(unittest.TestCase):
    def test_legacy_cosine_state_reconstructs_next_epoch_lr(self) -> None:
        original_model = torch.nn.Linear(2, 1)
        original_optimizer = torch.optim.AdamW(original_model.parameters(), lr=1e-4)
        original_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            original_optimizer, T_max=60, eta_min=1e-6
        )
        for _ in range(27):
            original_scheduler.step()
        legacy_optimizer_state = original_optimizer.state_dict()
        original_scheduler.step()
        expected_lr = original_optimizer.param_groups[0]["lr"]

        resumed_model = torch.nn.Linear(2, 1)
        resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-4)
        resumed_optimizer.load_state_dict(legacy_optimizer_state)
        resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            resumed_optimizer, T_max=60, eta_min=1e-6, last_epoch=27
        )

        self.assertAlmostEqual(resumed_optimizer.param_groups[0]["lr"], expected_lr)
        self.assertEqual(resumed_scheduler.last_epoch, 28)

    def test_atomic_checkpoint_contains_resume_state(self) -> None:
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        generator = torch.Generator().manual_seed(42)
        args = argparse.Namespace(data="example.yaml")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "weights" / "last.pt"
            save_checkpoint(
                path,
                model,
                optimizer,
                epoch=3,
                args=args,
                data={"nc": 2},
                metrics={"selection_score": 0.5},
                scheduler=scheduler,
                training_state={"best_fitness": 0.5, "best_raw_fitness": 0.5, "epochs_without_improvement": 0},
                data_generator=generator,
                retry_delay=0,
            )

            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["epoch"], 3)
            self.assertEqual(checkpoint["training_state"]["best_fitness"], 0.5)
            self.assertIsNotNone(checkpoint["scheduler"])
            self.assertIsNotNone(checkpoint["rng_state"]["data_generator"])
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])

    def test_recovery_ignores_rows_newer_than_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            results = Path(temp_dir) / "results.csv"
            with results.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=("epoch", "train_loss", "selection_score"))
                writer.writeheader()
                writer.writerows(
                    (
                        {"epoch": 1, "train_loss": 1.0, "selection_score": 0.5000},
                        {"epoch": 2, "train_loss": 0.9, "selection_score": 0.5005},
                        {"epoch": 3, "train_loss": 0.8, "selection_score": 0.5020},
                        {"epoch": 4, "train_loss": 0.7, "selection_score": 0.9000},
                    )
                )

            state = recover_training_state(results, completed_epoch=3, min_delta=0.001)
            self.assertAlmostEqual(state["best_fitness"], 0.5020)
            self.assertAlmostEqual(state["best_raw_fitness"], 0.5020)
            self.assertEqual(state["epochs_without_improvement"], 0)

            backup = trim_uncommitted_results(results, completed_epoch=3)
            self.assertIsNotNone(backup)
            self.assertTrue(backup.exists())
            with results.open("r", encoding="utf-8") as f:
                self.assertEqual([int(row["epoch"]) for row in csv.DictReader(f)], [1, 2, 3])

    def test_resume_protocol_mismatch_is_rejected(self) -> None:
        current = argparse.Namespace(batch=4, imgsz=512)
        with self.assertRaisesRegex(ValueError, "batch"):
            validate_resume_args(current, {"batch": 8, "imgsz": 512})


if __name__ == "__main__":
    unittest.main()
