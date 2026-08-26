"""Checkpoint persistence and resume policy for referring segmentation."""

from __future__ import annotations

import argparse
import csv
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ultralytics.nn import SemanticSegmentationModel


class RefSegCheckpointManager:
    """Own atomic checkpoint saves, best-model selection, and resume recovery."""

    _RESUME_PROTOCOL_KEYS = (
        "data", "model", "weights", "epochs", "batch", "imgsz", "seed", "deterministic",
        "lr", "weight_decay", "scheduler", "min_lr", "patience", "min_delta",
        "train_head_only", "freeze_backbone", "freeze_neck", "pos_weight_max",
        "loss_small_target_weight", "loss_small_target_area", "loss_tversky_fp_weight",
        "loss_fp_weight", "small_target_boost", "small_target_area", "augment",
        "augment_hflip", "augment_vflip", "augment_color_jitter", "augment_direction_policy",
        "empty_mask_policy", "val_thresholds", "val_select_metric", "val_fbeta",
        "max_batches", "max_val_batches", "text_queries", "text_encoder", "text_model_name",
        "text_pretrained", "text_precision",
    )

    @staticmethod
    def improvement_flags(
        fitness: float,
        best_fitness: float,
        best_raw_fitness: float,
        min_delta: float,
    ) -> Tuple[bool, bool]:
        """Return min-delta and raw checkpoint improvement decisions."""
        return fitness > best_fitness + min_delta, fitness > best_raw_fitness

    @staticmethod
    def select_test_checkpoint(weights_dir: Path) -> Path:
        """Prefer the strict raw-best checkpoint while supporting legacy runs."""
        best_raw_path = weights_dir / "best_raw.pt"
        return best_raw_path if best_raw_path.exists() else weights_dir / "best.pt"

    @staticmethod
    def save(
        path: Path,
        model: SemanticSegmentationModel,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        args: argparse.Namespace,
        data: Dict[str, Any],
        metrics: Dict[str, float],
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        training_state: Optional[Dict[str, Any]] = None,
        data_generator: Optional[torch.Generator] = None,
        retries: int = 5,
        retry_delay: float = 1.0,
    ) -> None:
        """Atomically save a complete restartable training checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "args": vars(args),
            "data": data,
            "metrics": metrics,
            "training_state": training_state or {},
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "data_generator": data_generator.get_state() if data_generator is not None else None,
            },
        }
        attempts = max(int(retries), 1)
        last_error: Optional[BaseException] = None
        for attempt in range(1, attempts + 1):
            fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
            os.close(fd)
            temp_path = Path(temp_name)
            try:
                torch.save(payload, temp_path)
                os.replace(temp_path, path)
                return
            except (OSError, RuntimeError) as exc:
                last_error = exc
                temp_path.unlink(missing_ok=True)
                if attempt >= attempts:
                    break
                print(f"checkpoint save retry {attempt}/{attempts - 1} for {path}: {exc}")
                time.sleep(max(float(retry_delay), 0.0))
        raise RuntimeError(f"Failed to save checkpoint after {attempts} attempt(s): {path}") from last_error

    @staticmethod
    def recover_training_state(results_csv: Path, completed_epoch: int, min_delta: float) -> Dict[str, Any]:
        """Rebuild best scores and patience state from committed result rows."""
        best_fitness = -float("inf")
        best_raw_fitness = -float("inf")
        epochs_without_improvement = 0
        if results_csv.exists():
            with results_csv.open("r", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    row_epoch = int(row["epoch"])
                    if row_epoch > completed_epoch:
                        continue
                    value = row.get("selection_score") or ""
                    fitness = float(value) if value else -float(row["train_loss"])
                    best_raw_fitness = max(best_raw_fitness, fitness)
                    if fitness > best_fitness + float(min_delta):
                        best_fitness = fitness
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
        return {
            "best_fitness": best_fitness,
            "best_raw_fitness": best_raw_fitness,
            "epochs_without_improvement": epochs_without_improvement,
        }

    @staticmethod
    def trim_uncommitted_results(results_csv: Path, completed_epoch: int) -> Optional[Path]:
        """Back up and remove rows newer than the resume checkpoint epoch."""
        if not results_csv.exists():
            return None
        with results_csv.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
        kept = [row for row in rows if int(row["epoch"]) <= completed_epoch]
        if len(kept) == len(rows):
            return None
        backup_path = results_csv.with_name(f"{results_csv.name}.pre_resume_epoch{completed_epoch}.bak")
        suffix = 1
        while backup_path.exists():
            backup_path = results_csv.with_name(
                f"{results_csv.name}.pre_resume_epoch{completed_epoch}.{suffix}.bak"
            )
            suffix += 1
        os.replace(results_csv, backup_path)
        with results_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)
        return backup_path

    @classmethod
    def validate_resume_args(cls, current_args: argparse.Namespace, checkpoint_args: Dict[str, Any]) -> None:
        """Reject resume commands that would silently change the established protocol."""
        mismatches = []
        current = vars(current_args)
        for key in cls._RESUME_PROTOCOL_KEYS:
            if key in checkpoint_args and current.get(key) != checkpoint_args.get(key):
                mismatches.append(f"{key}: current={current.get(key)!r}, checkpoint={checkpoint_args.get(key)!r}")
        if mismatches:
            raise ValueError("Resume protocol mismatch:\n" + "\n".join(mismatches))

    @staticmethod
    def restore_rng_state(checkpoint: Dict[str, Any], data_generator: torch.Generator) -> bool:
        """Restore Python, NumPy, Torch, CUDA, and data-loader generator states."""
        state = checkpoint.get("rng_state")
        if not state:
            return False
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        if torch.cuda.is_available() and state.get("cuda") is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
        if state.get("data_generator") is not None:
            data_generator.set_state(state["data_generator"].cpu())
        return True


checkpoint_improvement_flags = RefSegCheckpointManager.improvement_flags
select_test_checkpoint = RefSegCheckpointManager.select_test_checkpoint
save_checkpoint = RefSegCheckpointManager.save
recover_training_state = RefSegCheckpointManager.recover_training_state
trim_uncommitted_results = RefSegCheckpointManager.trim_uncommitted_results
validate_resume_args = RefSegCheckpointManager.validate_resume_args
restore_rng_state = RefSegCheckpointManager.restore_rng_state


__all__ = (
    "RefSegCheckpointManager",
    "checkpoint_improvement_flags",
    "recover_training_state",
    "restore_rng_state",
    "save_checkpoint",
    "select_test_checkpoint",
    "trim_uncommitted_results",
    "validate_resume_args",
)
