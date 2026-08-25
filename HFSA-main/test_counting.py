"""Evaluate text-guided object counting on VOC-style VRSBench annotations."""

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import torch
from tqdm import tqdm

from counting.count_infer import CountingImagePreprocessor, CountingPromptEncoder, ObjectCounter
from counting.dataset_vrs import VRSCountingDataset
from counting.evaluate import CountingEvaluator, CountingVisualizer
from ultralytics import YOLO
from ultralytics.utils.ops import scale_boxes


class CountingEvaluationApplication:
    """Class-based evaluator preserving the teammate's positive-query protocol."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Evaluate YOLOv12 text-guided object counting.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument("--weights", type=str, default="runs/counting/counting-VRSBench-ViT-L-14/weights/best.pt", help="Counting checkpoint.")
        parser.add_argument("--voc-root", type=str, default="data/VRSBench", help="VOC-style VRSBench root.")
        parser.add_argument("--dataset", type=str, default="VRSBench", help="Dataset name used in reports.")
        parser.add_argument("--split-file", type=str, default="test.txt", help="Split filename under the VOC root or ImageSets/Main.")
        parser.add_argument("--text-model-name", type=str, default="ViT-L-14", help="OpenCLIP model architecture.")
        parser.add_argument("--text-pretrained", type=str, default="openai", help="OpenCLIP pretrained tag.")
        parser.add_argument("--prompt-template", type=str, default="a satellite remote sensing photo of {class_name}", help="Prompt template containing {class_name}.")
        parser.add_argument("--imgsz", type=int, default=800, help="Inference image size.")
        parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Inference device.")
        parser.add_argument("--conf-thres", type=float, default=0.15, help="Detection confidence threshold.")
        parser.add_argument("--iou-thres", type=float, default=0.50, help="NMS IoU threshold.")
        parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per query.")
        parser.add_argument("--max-samples", type=int, default=0, help="Maximum images to evaluate; 0 means all.")
        parser.add_argument("--max-save-vis", type=int, default=30, help="Maximum query visualizations to save.")
        parser.add_argument("--save-dir", type=str, default="runs/counting/eval", help="Evaluation output directory.")
        return parser

    @classmethod
    def from_cli(cls, argv: Sequence[str] | None = None) -> "CountingEvaluationApplication":
        return cls(cls.build_parser().parse_args(argv))

    @staticmethod
    def _safe_filename(value: str) -> str:
        return "_".join(str(value).replace("/", " ").replace("\\", " ").split())

    def run(self) -> None:
        args = self.args
        weights_path = Path(args.weights).expanduser().resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(f"Counting checkpoint not found: {weights_path}")

        device = torch.device(args.device)
        dataset = VRSCountingDataset(args.voc_root, split_file=args.split_file)
        yolo = YOLO(str(weights_path))
        prompt_encoder = CountingPromptEncoder(
            model_name=args.text_model_name,
            pretrained=args.text_pretrained,
            device=device,
            prompt_template=args.prompt_template,
        )
        counter = ObjectCounter(yolo.model, prompt_encoder=prompt_encoder, device=device)
        preprocessor = CountingImagePreprocessor(args.imgsz)
        evaluator = CountingEvaluator()
        save_dir = Path(args.save_dir).expanduser().resolve()
        visual_dir = save_dir / "visualizations"
        saved_visuals = 0
        processed_images = 0

        print("=" * 72)
        print("Task: text-guided object counting evaluation")
        print(f"Dataset: {args.dataset} ({dataset.voc_root})")
        print(f"Split: {dataset.split_path} ({len(dataset)} listed images)")
        print(f"Weights: {weights_path}")
        print(f"Thresholds: conf={args.conf_thres}, iou={args.iou_thres}, max_det={args.max_det}")
        print("=" * 72)

        for sample in tqdm(dataset, total=len(dataset), desc="Counting Evaluation"):
            if int(args.max_samples) > 0 and processed_images >= int(args.max_samples):
                break
            image = cv2.imread(str(sample.image_path))
            if image is None:
                continue
            image_tensor, _, _ = preprocessor(image)
            processed_images += 1

            for class_name, gt_count in sample.counts.items():
                result = counter.count(
                    image_tensor,
                    class_name,
                    conf_thres=args.conf_thres,
                    iou_thres=args.iou_thres,
                    max_det=args.max_det,
                )
                pred_count = int(result["pred_count"])
                evaluator.update(pred_count, gt_count, class_name=class_name, image_id=sample.image_id)

                if saved_visuals < int(args.max_save_vis):
                    boxes = result["pred_boxes"].clone()
                    if len(boxes) > 0:
                        scale_boxes(image_tensor.shape[2:], boxes, image.shape[:2])
                    visualization = CountingVisualizer.render(
                        image=image,
                        boxes=boxes.numpy(),
                        image_id=sample.image_id,
                        class_name=class_name,
                        gt_count=gt_count,
                        pred_count=pred_count,
                    )
                    filename = f"{self._safe_filename(sample.image_id)}_{self._safe_filename(class_name)}.jpg"
                    CountingVisualizer.save(visualization, visual_dir / filename)
                    saved_visuals += 1

        metrics = evaluator.compute_metrics()
        report_paths = evaluator.save_report(
            save_dir=save_dir,
            dataset=args.dataset,
            split=args.split_file,
            weights=str(weights_path),
        )
        print("=" * 72)
        print(f"Evaluated query pairs: {metrics['total_samples']}")
        print(f"Exact Match: {metrics['exact_match']:.2f}%")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"JSON report: {report_paths['json']}")
        print(f"Text report: {report_paths['text']}")
        print("=" * 72)


def main() -> None:
    CountingEvaluationApplication.from_cli().run()


if __name__ == "__main__":
    main()
