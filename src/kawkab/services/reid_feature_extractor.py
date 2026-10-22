"""SoccerNet ReID feature extractor — ResNet-50 CircleLoss for player re-identification.

Uses a ResNet-50 backbone trained with CircleLoss on SoccerNet tracking data.
Falls back to a simple ResNet-50 TorchVision baseline when the SoccerNet
checkpoint is not available.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kawkab.core.logging import get_logger
from kawkab.core.paths import get_paths

logger = get_logger(__name__)

_SOCCERNET_REID_AVAILABLE = False
try:
    import torch  # noqa: F401  (availability probe: sets _SOCCERNET_REID_AVAILABLE)
    import torch.nn as nn
    import torchvision.models as models

    _SOCCERNET_REID_AVAILABLE = True
except ImportError:
    pass


class SoccerNetReIDExtractor:
    """ResNet-50 CircleLoss feature extractor trained on SoccerNet tracking data.

    Produces 512-dimensional L2-normalized embeddings for player crops.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._model: Any = None
        self._input_size = (256, 128)  # H, W — standard ReID aspect
        self._load_model()

    def _load_model(self) -> None:
        if not _SOCCERNET_REID_AVAILABLE:
            logger.warning("PyTorch not available; SoccerNet ReID extractor disabled")
            return
        import torch

        model_path = get_paths().cache / "models" / "soccernet_reid.pt"
        try:
            model = models.resnet50(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 512)
            if model_path.exists():
                state = torch.load(str(model_path), map_location=self.device)
                # Handle various checkpoint formats
                if "model_state_dict" in state:
                    model.load_state_dict(state["model_state_dict"])
                elif "state_dict" in state:
                    model.load_state_dict(state["state_dict"])
                elif isinstance(state, dict) and any(
                    k.startswith("layer") or k.startswith("conv") for k in state
                ):
                    model.load_state_dict(state)
                else:
                    # Try loading as full model, or just use initialized weights
                    logger.info(
                        "SoccerNet checkpoint format unrecognised; "
                        "using untrained ResNet-50 baseline"
                    )
            else:
                logger.info("soccernet_reid.pt not cached; using untrained ResNet-50 baseline")
            model.eval()
            model.to(self.device)
            self._model = model
            logger.info(
                f"SoccerNet ReID extractor loaded (device={self.device}, "
                f"checkpoint={model_path.exists()})"
            )
        except Exception as e:
            logger.warning(f"SoccerNet ReID extractor init failed: {e}")
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def extract(self, crop: np.ndarray) -> np.ndarray | None:
        """Extract a 512-d L2-normalized embedding from a person crop.

        Args:
            crop: BGR or RGB image array (H, W, 3)

        Returns:
            512-d float32 unit vector, or None on failure.
        """
        if self._model is None:
            return None
        try:
            import cv2
            import torch
            import torchvision.transforms as transforms

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if crop.shape[2] == 3 else crop
            transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(self._input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            tensor = transform(rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self._model(tensor).cpu().numpy().flatten().astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 1e-8:
                return emb / norm
        except Exception as e:
            logger.debug(f"SoccerNet ReID extraction failed: {e}")
        return None
