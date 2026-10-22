"""Train a CNN jersey number classifier on the SoccerNet sn-jersey dataset.

Usage:
    # Download dataset and train (requires SoccerNet pip package)
    python scripts/train_jersey_cnn.py --download --train --output models/jersey_cnn.pt

    # Train from cached dataset only
    python scripts/train_jersey_cnn.py --train --output models/jersey_cnn.pt

The trained model can be loaded via:
    from kawkab.services.jersey_service import JerseyNumberService
    svc = JerseyNumberService(reader="cnn")
    svc.load_cnn_model("models/jersey_cnn.pt")
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    import torch  # for static annotations only; runtime import stays lazy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_jersey_cnn")

CNN_INPUT_SIZE = 28
NUM_CLASSES = 11  # -1 (no digit) + 0-9


def build_model() -> torch.nn.Module:
    # torch is deliberately lazy-imported (heavy; not needed for --help or
    # download-only runs). Static checkers resolve the annotation via the
    # TYPE_CHECKING-only import above.
    import torch
    import torch.nn as nn

    class GNetDeep(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding="same")
            self.bn1 = nn.BatchNorm2d(16)
            self.pool1 = nn.MaxPool2d(3, stride=2)
            self.conv2 = nn.Conv2d(16, 32, 3, padding="same")
            self.bn2 = nn.BatchNorm2d(32)
            self.pool2 = nn.MaxPool2d(3, stride=2)
            self.conv3 = nn.Conv2d(32, 64, 3, padding="same")
            self.bn3 = nn.BatchNorm2d(64)
            self.pool3 = nn.MaxPool2d(3, stride=2)
            self.drop1 = nn.Dropout(0.25)
            self.fc1 = nn.Linear(64 * 3 * 3, 128)
            self.drop2 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128, NUM_CLASSES)

        def forward(self, x):
            # `torch` here is the module imported inside build_model() above --
            # the module-level `import torch.nn as nn` alone used to leave this
            # name undefined (latent NameError on the first forward pass).
            x = torch.relu(self.bn1(self.conv1(x)))
            x = self.pool1(x)
            x = torch.relu(self.bn2(self.conv2(x)))
            x = self.pool2(x)
            x = torch.relu(self.bn3(self.conv3(x)))
            x = self.pool3(x)
            x = self.drop1(x)
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.drop2(x)
            x = self.fc2(x)
            return x

    return GNetDeep()


def download_soccernet_jersey(data_dir: Path) -> bool:
    """Download SoccerNet jersey-2023 dataset via the SoccerNet pip package."""
    try:
        from SoccerNet.Downloader import SoccerNetDownloader as SNdl

        sn_dl = SNdl(LocalDirectory=str(data_dir))
        sn_dl.downloadDataTask(task="jersey-2023", split=["train", "test"])
        logger.info(f"SoccerNet jersey dataset downloaded to {data_dir}")
        return True
    except ImportError:
        logger.error("SoccerNet pip package not installed. Run: pip install SoccerNet")
        return False
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def load_dataset(data_dir: Path, split: str) -> tuple[list[np.ndarray], list[int]]:
    """Load images and labels from the sn-jersey dataset split.

    Dataset structure:
        data_dir/jersey-2023/{split}/image/{player_id}/*.jpg
        data_dir/jersey-2023/{split}/gt.json  # {player_id: jersey_number}
    """
    split_dir = data_dir / "jersey-2023" / split
    image_dir = split_dir / "image"
    gt_path = split_dir / "gt.json"

    if not image_dir.exists() or not gt_path.exists():
        logger.warning(f"Split {split} not found at {image_dir}")
        return [], []

    with open(gt_path) as f:
        ground_truth = json.load(f)

    images: list[np.ndarray] = []
    labels: list[int] = []

    for player_id, jersey_num in ground_truth.items():
        player_dir = image_dir / player_id
        if not player_dir.exists():
            continue
        for img_path in sorted(player_dir.iterdir())[:20]:  # max 20 per player
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            images.append(img)
            labels.append(int(jersey_num))

    logger.info(f"Loaded {len(images)} samples from {split}")
    return images, labels


def prepare_patches(images: list[np.ndarray], labels: list[int]):
    """Isolate digit regions from torso crops and map to digit-level labels.

    Uses the same contour-based digit isolation as jersey_service._isolate_digits.
    """
    digits_list: list[np.ndarray] = []
    digit_labels: list[int] = []

    for img, jersey_num in zip(images, labels, strict=False):
        if jersey_num < 0 or jersey_num > 99:
            continue
        patches = _isolate_digits(img)
        num_str = str(jersey_num)
        for i, patch in enumerate(patches):
            if i >= len(num_str):
                break
            target_digit = int(num_str[i])
            patch_resized = cv2.resize(patch, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
            gray = cv2.cvtColor(patch_resized, cv2.COLOR_BGR2GRAY)
            digits_list.append(gray)
            digit_labels.append(target_digit)

    return digits_list, digit_labels


def _isolate_digits(img: np.ndarray) -> list[np.ndarray]:
    """Mirror of jersey_service._isolate_digits for training."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Sobel(blurred, cv2.CV_8U, 1, 0, ksize=3)
    _, thresh = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digits = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h < 8 or w < 4:
            continue
        if h > img.shape[0] * 0.9 or w > img.shape[1] * 0.8:
            continue
        aspect = w / max(h, 1)
        if aspect < 0.3 or aspect > 1.0:
            continue
        digit = img[y : y + h, x : x + w]
        digits.append(digit)

    digits.sort(
        key=lambda d: (
            cv2.boundingRect(
                cv2.findContours(
                    cv2.threshold(cv2.cvtColor(d, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)[
                        1
                    ],
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )[0]
            )[0][0][0]
            if len(
                cv2.findContours(
                    cv2.threshold(cv2.cvtColor(d, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)[
                        1
                    ],
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )[0]
            )
            > 0
            else 0
        )
    )
    return digits[:3]


def train_model(
    train_patches: list[np.ndarray],
    train_labels: list[int],
    val_patches: list[np.ndarray] | None = None,
    val_labels: list[int] | None = None,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 0.001,
    output_path: str = "models/jersey_cnn.pt",
):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    class DigitDataset(Dataset):
        def __init__(self, patches, labels):
            self.patches = patches
            self.labels = labels

        def __len__(self):
            return len(self.patches)

        def __getitem__(self, idx):
            patch = self.patches[idx].astype(np.float32) / 255.0
            tensor = torch.from_numpy(patch).unsqueeze(0)
            label = self.labels[idx] + 1  # shift: -1->0, 0->1, ..., 9->10
            return tensor, torch.tensor(label, dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(
        DigitDataset(train_patches, train_labels), batch_size=batch_size, shuffle=True
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

        acc = 100.0 * correct / max(total, 1)
        logger.info(
            f"Epoch {epoch + 1}/{epochs}  "
            f"loss={total_loss / max(len(train_loader), 1):.4f}  "
            f"acc={acc:.2f}%"
        )

    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(output))
    logger.info(f"Model saved to {output}")


def main():
    parser = argparse.ArgumentParser(description="Train SoccerNet jersey CNN")
    parser.add_argument("--download", action="store_true", help="Download sn-jersey dataset first")
    parser.add_argument("--train", action="store_true", required=True, help="Train the CNN model")
    parser.add_argument("--data-dir", default="data/soccernet", help="Dataset directory")
    parser.add_argument("--output", default="models/jersey_cnn.pt", help="Output model path")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.download and not download_soccernet_jersey(data_dir):
        return

    if args.train:
        train_imgs, train_labels = load_dataset(data_dir, "train")
        if not train_imgs:
            logger.error("No training data found. Use --download first.")
            return

        logger.info(f"Preparing {len(train_imgs)} training images...")
        train_patches, train_digit_labels = prepare_patches(train_imgs, train_labels)
        logger.info(f"Extracted {len(train_patches)} digit patches for training")

        val_patches = val_labels = None
        val_imgs, val_labels_raw = load_dataset(data_dir, "test")
        if val_imgs:
            val_patches, val_labels = prepare_patches(val_imgs, val_labels_raw)
            logger.info(f"Using {len(val_patches)} validation patches")

        train_model(
            train_patches,
            train_digit_labels,
            val_patches,
            val_labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            output_path=args.output,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
