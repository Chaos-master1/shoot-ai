"""Train a YOLO11n ball detector (single class: 'ball').

Pipeline:
  1. Download SoccerNet ball-detection subset or generate synthetic ball images
  2. Convert to YOLO format
  3. Fine-tune yolo11n.pt → ball detector
  4. Export to ONNX for deployment

Usage:
  python scripts/train_ball_detector.py --method synthetic --epochs 30
  python scripts/train_ball_detector.py --method soccernet --epochs 50
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger("train_ball_detector")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ball_detector"
TRAIN_IMG = DATA_DIR / "train" / "images"
TRAIN_LBL = DATA_DIR / "train" / "labels"
VAL_IMG = DATA_DIR / "val" / "images"
VAL_LBL = DATA_DIR / "val" / "labels"


def _ensure_dirs():
    for d in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]:
        d.mkdir(parents=True, exist_ok=True)


def _draw_ball(
    img: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Draw a spherical ball with shading on the image."""
    # Base circle
    cv2.circle(img, (cx, cy), radius, color, -1)
    # Highlight (top-left light source)
    highlight = tuple(min(255, c + 60) for c in color)
    cv2.circle(img, (cx - radius // 3, cy - radius // 3), radius // 3, highlight, -1)
    # Shadow (bottom-right)
    shadow = tuple(max(0, c - 40) for c in color)
    cv2.circle(img, (cx + radius // 4, cy + radius // 4), radius // 4, shadow, -1)
    return img


def generate_synthetic(
    count: int = 500,
    img_size: tuple[int, int] = (640, 640),
    min_radius: int = 6,
    max_radius: int = 18,
    output_dir: Path = TRAIN_IMG,
    label_dir: Path = TRAIN_LBL,
):
    """Generate synthetic ball images on random pitch backgrounds."""
    for i in range(count):
        h, w = img_size
        # Pitch background (green with variation)
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        bg[:, :] = (
            random.randint(40, 90),
            random.randint(120, 200),
            random.randint(20, 60),
        )
        # Add some noise / grass texture
        noise = np.random.randint(-15, 16, (h, w, 3), dtype=np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # Random line markings (white)
        for _ in range(random.randint(0, 5)):
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            cv2.line(
                bg,
                (x1, y1),
                (x1 + random.randint(-80, 80), y1 + random.randint(-80, 80)),
                (200, 200, 200),
                random.randint(1, 3),
            )

        # Ball properties
        radius = random.randint(min_radius, max_radius)
        margin = radius + 2
        cx = random.randint(margin, w - margin)
        cy = random.randint(margin, h - margin)

        # Ball color: white (match ball) or dark (shadow ball)
        ball_color = (240, 240, 240) if random.random() < 0.7 else (30, 30, 30)

        _draw_ball(bg, cx, cy, radius, color=ball_color)

        # Motion blur augmentation
        if random.random() < 0.3:
            ksize = random.choice([3, 5])
            kernel = np.zeros((ksize, ksize))
            kernel[ksize // 2, :] = 1.0 / ksize
            bg = cv2.filter2D(bg, -1, kernel)

        # Varying brightness
        brightness = random.uniform(0.7, 1.3)
        bg = np.clip(bg.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

        # Save image
        fname = f"synth_ball_{i:05d}.jpg"
        cv2.imwrite(str(output_dir / fname), bg)

        # YOLO label: class 0, normalized x_center, y_center, width, height
        x_norm = cx / w
        y_norm = cy / h
        w_norm = (radius * 2) / w
        h_norm = (radius * 2) / h
        label_path = label_dir / Path(fname).with_suffix(".txt")
        label_path.write_text(f"0 {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    logger.info(f"Generated {count} synthetic ball images in {output_dir}")


def prepare_soccernet(
    output_dir: Path = DATA_DIR,
    max_matches: int = 10,
):
    """Download and convert SoccerNet ball annotations to YOLO format.

    Uses the SoccerNet tracking dataset (SkillCorner open-data).
    Falls back to synthetic if download fails.
    """
    raw_dir = PROJECT_ROOT / "data" / "ground_truth" / "skillcorner" / "opendata-master" / "data"
    if not raw_dir.exists():
        logger.warning(
            "SoccerNet data not found at %s. "
            "Run: python scripts/download_ground_truth.py --soccer-net-annotations",
            raw_dir,
        )
        logger.info("Falling back to synthetic-only training.")
        return False

    match_dirs = sorted(raw_dir.glob("*/"))[:max_matches]
    if not match_dirs:
        logger.warning("No SoccerNet matches found.")
        return False

    random.shuffle(match_dirs)
    split = int(len(match_dirs) * 0.8)
    train_matches = match_dirs[:split]
    val_matches = match_dirs[split:]

    import json
    import shutil

    for matches, img_dir, label_dir in [
        (train_matches, TRAIN_IMG, TRAIN_LBL),
        (val_matches, VAL_IMG, VAL_LBL),
    ]:
        for match in matches:
            ann_file = match / "annotations.json"
            if not ann_file.exists():
                continue
            with open(ann_file) as f:
                annotations = json.load(f)
            for ann in annotations:
                src_img = match / "images" / ann["image_name"]
                if not src_img.exists():
                    continue
                dst_img = img_dir / ann["image_name"]
                shutil.copy2(src_img, dst_img)
                label_file = label_dir / Path(ann["image_name"]).with_suffix(".txt")
                with open(label_file, "w") as lf:
                    for bbox in ann.get("detections", []):
                        cls_id = bbox.get("class_id", 0)
                        xc = (bbox["x1"] + bbox["x2"]) / 2.0 / ann["width"]
                        yc = (bbox["y1"] + bbox["y2"]) / 2.0 / ann["height"]
                        bw = (bbox["x2"] - bbox["x1"]) / ann["width"]
                        bh = (bbox["y2"] - bbox["y1"]) / ann["height"]
                        lf.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
    logger.info(f"SoccerNet: {len(train_matches)} train + {len(val_matches)} val matches")
    return True


def write_dataset_yaml():
    """Write dataset.yaml for YOLO training."""
    yaml_path = DATA_DIR / "dataset.yaml"
    yaml_path.write_text(
        f"train: {TRAIN_IMG.resolve()}\nval: {VAL_IMG.resolve()}\nnc: 1\nnames: ['ball']\n"
    )
    logger.info(f"Dataset YAML: {yaml_path}")
    return yaml_path


def run_training(args: argparse.Namespace):
    _ensure_dirs()

    if args.method == "soccernet":
        ok = prepare_soccernet(max_matches=args.max_matches)
        if not ok:
            logger.info("Falling back to synthetic-only data.")
            generate_synthetic(count=args.synthetic_count, img_size=(args.imgsz, args.imgsz))
    elif args.method == "both":
        prepare_soccernet(max_matches=args.max_matches)
        generate_synthetic(count=args.synthetic_count, img_size=(args.imgsz, args.imgsz))
    else:
        generate_synthetic(count=args.synthetic_count, img_size=(args.imgsz, args.imgsz))

    # Also generate validation set
    val_count = max(50, args.synthetic_count // 5)
    generate_synthetic(
        count=val_count,
        img_size=(args.imgsz, args.imgsz),
        output_dir=VAL_IMG,
        label_dir=VAL_LBL,
    )

    yaml_path = write_dataset_yaml()

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model_path = args.model or "yolo11n.pt"
    logger.info(f"Loading base model: {model_path}")
    model = YOLO(model_path)

    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr,
        augment=True,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name or "ball_detector",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        cos_lr=True,
        label_smoothing=0.05,
        val=True,
        patience=args.patience,
    )

    best_path = Path(results.save_dir) / "weights" / "best.pt"
    logger.info(f"Best model: {best_path}")

    # Export to ONNX
    if args.export_onnx:
        if best_path.exists():
            model = YOLO(str(best_path))
            onnx_path = best_path.with_suffix(".onnx")
            model.export(format="onnx", half=True, simplify=True)
            logger.info(f"ONNX exported: {onnx_path}")
        else:
            logger.warning("Best model not found, skipping ONNX export")


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11n ball detector")
    parser.add_argument(
        "--method",
        choices=["synthetic", "soccernet", "both"],
        default="synthetic",
        help="Training data source (default: synthetic)",
    )
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base YOLO model")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="0", help="CUDA device")
    parser.add_argument("--workers", type=int, default=4, help="Data workers")
    parser.add_argument("--project", type=str, default="runs/train", help="Output directory")
    parser.add_argument("--name", type=str, default=None, help="Run name")
    parser.add_argument("--export-onnx", action="store_true", help="Export ONNX after training")
    parser.add_argument("--max-matches", type=int, default=10, help="Max SoccerNet matches to use")
    parser.add_argument(
        "--synthetic-count", type=int, default=500, help="Number of synthetic images to generate"
    )
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
