"""Fine-tune YOLO11m on football broadcast data.

Uses SoccerNet annotations (50K+ frames) or custom annotation sets.

Setup:
  1. Download SoccerNet tracking dataset:
     python scripts/download_ground_truth.py --soccer-net-annotations

  2. Run training:
     python scripts/fine_tune_yolo.py --data soccer_net.yaml --epochs 50

The dataset YAML format:
  train: data/soccer_net/train/images
  val: data/soccer_net/val/images
  nc: 2
  names: ["person", "sports ball"]

This script expects:
  - SoccerNet tracking annotations in COCO format
  - Or custom annotations in YOLO format (.txt per image)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger("fine_tune_yolo")


def run_training(args: argparse.Namespace):
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model_path = args.model or "yolo11m.pt"
    logger.info(f"Loading base model: {model_path}")
    model = YOLO(model_path)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr,
        augment=True,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name or "football_finetune",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        cos_lr=True,
        label_smoothing=0.05,
        overlap_mask=False,
        val=True,
    )

    # Export to ONNX for faster inference
    if args.export_onnx:
        best_path = Path(args.project) / (args.name or "football_finetune") / "weights" / "best.pt"
        if best_path.exists():
            model = YOLO(str(best_path))
            model.export(format="onnx", half=True, simplify=True)
            logger.info(f"Exported ONNX to {best_path.with_suffix('.onnx')}")

    logger.info(f"Training complete. Best model: {results.save_dir / 'weights' / 'best.pt'}")


def prepare_soccer_net_annotations(
    output_dir: Path = Path("data/soccer_net"),
    source_dir: Path | None = None,
):
    """Convert SoccerNet tracking annotations to YOLO format.

    Args:
        output_dir: Directory to write YOLO-format dataset.
        source_dir: SoccerNet tracking annotations directory. Falls back to
            default path if not provided.
    """
    import json
    import shutil

    raw_dir = source_dir or Path("data/ground_truth/skillcorner/opendata-master/data")
    if not raw_dir.exists():
        logger.error("SoccerNet data not found. Run download_ground_truth.py first.")
        return

    train_img_dir = output_dir / "train" / "images"
    train_label_dir = output_dir / "train" / "labels"
    val_img_dir = output_dir / "val" / "images"
    val_label_dir = output_dir / "val" / "labels"

    for d in [train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
        d.mkdir(parents=True, exist_ok=True)

    matches = list(raw_dir.glob("*/"))
    import random

    random.shuffle(matches)
    split_idx = int(len(matches) * 0.8)
    train_matches = matches[:split_idx]
    val_matches = matches[split_idx:]

    for match_list, img_dir, label_dir in [
        (train_matches, train_img_dir, train_label_dir),
        (val_matches, val_img_dir, val_label_dir),
    ]:
        for match in match_list:
            annotations_file = match / "annotations.json"
            if not annotations_file.exists():
                continue
            with open(annotations_file) as f:
                annotations = json.load(f)
            for ann in annotations:
                image_path = match / "images" / ann["image_name"]
                if not image_path.exists():
                    continue
                shutil.copy2(image_path, img_dir / ann["image_name"])
                label_path = label_dir / Path(ann["image_name"]).with_suffix(".txt").name
                with open(label_path, "w") as label_f:
                    for bbox in ann.get("detections", []):
                        cls_id = bbox.get("class_id", 0)
                        x_center = (bbox["x1"] + bbox["x2"]) / 2 / ann["width"]
                        y_center = (bbox["y1"] + bbox["y2"]) / 2 / ann["height"]
                        w = (bbox["x2"] - bbox["x1"]) / ann["width"]
                        h = (bbox["y2"] - bbox["y1"]) / ann["height"]
                        label_f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(
        f"train: {train_img_dir.resolve()}\n"
        f"val: {val_img_dir.resolve()}\n"
        f"nc: 2\n"
        f"names: ['person', 'sports ball']\n"
    )
    logger.info(f"Prepared {len(train_matches)} train + {len(val_matches)} val matches")
    logger.info(f"Dataset YAML: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11m on football data")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument("--data", type=str, required=True, help="Dataset YAML path")
    train_parser.add_argument("--model", type=str, default="yolo11m.pt", help="Base model")
    train_parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    train_parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    train_parser.add_argument("--batch", type=int, default=16, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_parser.add_argument("--device", type=str, default="0", help="CUDA device")
    train_parser.add_argument("--workers", type=int, default=4, help="Data workers")
    train_parser.add_argument("--project", type=str, default="runs/train", help="Output dir")
    train_parser.add_argument("--name", type=str, default=None, help="Run name")
    train_parser.add_argument(
        "--export-onnx", action="store_true", help="Export ONNX after training"
    )

    prepare_parser = subparsers.add_parser("prepare", help="Prepare SoccerNet annotations")
    prepare_parser.add_argument("--output-dir", type=str, default="data/soccer_net")
    prepare_parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="SoccerNet annotations directory (overrides default path)",
    )

    args = parser.parse_args()
    if args.command == "train":
        run_training(args)
    elif args.command == "prepare":
        source = Path(args.source) if args.source else None
        prepare_soccer_net_annotations(Path(args.output_dir), source_dir=source)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
