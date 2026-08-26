"""Class-based training, evaluation, and inference for scene classification."""

import csv
import json
import math
import time
from pathlib import Path

import torch
from PIL import Image
from ultralytics.utils.loss import SceneClassificationLoss

from .metrics import SceneClassificationMetrics


class SceneClassificationEvaluator:
    def __init__(self, network, device, class_names):
        self.network = network
        self.device = torch.device(device)
        self.class_names = list(class_names)
        self.criterion = SceneClassificationLoss()

    def evaluate(self, loader, max_batches=0):
        self.network.eval()
        metrics = SceneClassificationMetrics(len(self.class_names), self.class_names)
        loss_sum, sample_count = 0.0, 0
        start = time.perf_counter()
        with torch.no_grad():
            for batch_index, (images, labels) in enumerate(loader):
                if int(max_batches) > 0 and batch_index >= int(max_batches):
                    break
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                logits = self.network(images)
                loss = self.criterion(logits, labels)
                loss_sum += float(loss) * images.shape[0]
                sample_count += images.shape[0]
                metrics.update(logits, labels)
        report = metrics.compute()
        elapsed = time.perf_counter() - start
        report.update(
            {
                "loss": loss_sum / max(sample_count, 1),
                "evaluation_seconds": elapsed,
                "mean_ms_per_sample": elapsed * 1000.0 / max(sample_count, 1),
            }
        )
        return report

    @staticmethod
    def save_report(report, save_dir, checkpoint, split):
        output_dir = Path(save_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(report)
        payload.update({"task": "scene_classification", "checkpoint": str(checkpoint), "split": str(split)})
        path = output_dir / "test_results.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


class SceneClassificationTrainer:
    def __init__(self, network, config, class_names):
        self.network = network
        self.config = config
        self.class_names = list(class_names)
        self.device = torch.device(config.device)
        self.criterion = SceneClassificationLoss()
        self.optimizer = torch.optim.AdamW(
            self.network.head.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

    def _scheduler(self, steps_per_epoch):
        total_steps = max(steps_per_epoch * self.config.epochs, 1)
        warmup_steps = max(steps_per_epoch * self.config.warmup_epochs, 0)

        def lr_lambda(step):
            if warmup_steps and step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    @staticmethod
    def _append_result(path, row):
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def fit(self, train_loader, val_loader):
        output_dir = Path(self.config.save_dir).expanduser().resolve()
        weights_dir = output_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "results.csv"
        effective_steps = min(len(train_loader), self.config.max_train_batches or len(train_loader))
        scheduler = self._scheduler(effective_steps)
        evaluator = SceneClassificationEvaluator(self.network, self.device, self.class_names)
        best_accuracy = -1.0
        for epoch in range(self.config.epochs):
            self.network.train()
            running_loss, samples = 0.0, 0
            for batch_index, (images, labels) in enumerate(train_loader):
                if self.config.max_train_batches and batch_index >= self.config.max_train_batches:
                    break
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                logits = self.network(images)
                loss = self.criterion(logits, labels)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.head.parameters(), self.config.grad_clip_norm)
                self.optimizer.step()
                scheduler.step()
                running_loss += float(loss) * images.shape[0]
                samples += images.shape[0]

            val_metrics = evaluator.evaluate(val_loader, max_batches=self.config.max_val_batches)
            row = {
                "epoch": epoch + 1,
                "train_loss": running_loss / max(samples, 1),
                "val_loss": val_metrics["loss"],
                "top1_accuracy": val_metrics["top1_accuracy"],
                "top5_accuracy": val_metrics["top5_accuracy"],
                "macro_f1": val_metrics["macro_f1"],
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self._append_result(results_path, row)
            payload = self.network.checkpoint_payload(self.class_names, epoch + 1, val_metrics, self.config.to_dict())
            torch.save(payload, weights_dir / "last.pt")
            if val_metrics["top1_accuracy"] > best_accuracy:
                best_accuracy = val_metrics["top1_accuracy"]
                torch.save(payload, weights_dir / "best.pt")
            print(
                f"epoch {epoch + 1}/{self.config.epochs}: train_loss={row['train_loss']:.4f}, "
                f"val_loss={row['val_loss']:.4f}, top1={row['top1_accuracy']:.4f}, macro_f1={row['macro_f1']:.4f}"
            )
        return weights_dir / "best.pt"


class SceneClassificationPredictor:
    def __init__(self, network, transform, class_names, device):
        self.network = network.eval()
        self.transform = transform
        self.class_names = list(class_names)
        self.device = torch.device(device)

    def predict(self, image, topk=5):
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probabilities = self.network(tensor).softmax(dim=1)[0]
        values, indices = probabilities.topk(min(int(topk), len(self.class_names)))
        return [
            {"class_id": int(index), "class_name": self.class_names[int(index)], "probability": float(value)}
            for value, index in zip(values.cpu(), indices.cpu())
        ]
