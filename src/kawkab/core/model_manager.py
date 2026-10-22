"""Model manager - handles model download, caching, and lazy loading.

Production-grade model management for Kawkab AI:
1. Check if models exist in local cache
2. Download from remote URLs with progress callbacks
3. Validate downloaded files (SHA-256 checksum)
4. Cache management (cleanup old models)
5. Integration with CVService and VRAMManager for sequential loading

This enables future lazy-loading: the installer can be small (~50MB launcher)
and models can be downloaded on first run.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Callable
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from kawkab.core.logging import get_logger
from kawkab.core.paths import get_paths

logger = get_logger(__name__)

# Domain-separated HMAC key for the local model manifest. This is NOT a
# security boundary against a local attacker who can read this file -- it is
# tamper-evidence: a models.json edited by hand (or corrupted on disk) fails
# verification and is discarded, so a swapped checksum can never bless a
# tampered model binary. For an offline desktop app this closes the real
# gap (accidental/casual tampering) without a full PKI.
_MANIFEST_HMAC_KEY = b"kawkab-model-manifest-v1:hmac-sha256"


class ModelManager:
    """Manages AI model download, caching, and validation.

    Models are stored in the user's cache directory under `models/`.
    A manifest file (`models.json`) tracks available models, versions,
    and checksums.
    """

    # Default model URLs with PINNED checksums (verified by download on
    # 2026-09-16). Every download is rejected if its sha256 does not match.
    # ReID weights (OSNet / SoccerNet ResNet-50) are intentionally NOT listed:
    # their historical URLs are dead (boxmot moved to its own TRAINED_URLS on
    # Google Drive; the SoccerNet release asset was removed), and boxmot >= 19
    # -- the pinned dependency -- auto-downloads its ReID weights itself.
    # Shipping only yolo11n keeps the installer small; every other variant is
    # fetched on demand (Settings > AI Models).
    DEFAULT_MODELS = {
        "yolo11n": {
            "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
            "size_mb": 5.4,
            "sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
        },
        "yolo11s": {
            "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
            "size_mb": 18.4,
            "sha256": "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5",
        },
        "yolo11m": {
            "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
            "size_mb": 38.8,
            "sha256": "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95",
        },
        "yolo11l": {
            "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt",
            "size_mb": 49.0,
            "sha256": "9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f",
        },
        "yolo11x": {
            "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt",
            "size_mb": 109.3,
            "sha256": "7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24",
        },
    }

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or get_paths().cache / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache_dir / "models.json"
        self._manifest: dict = {}
        self._load_manifest()

    def _load_manifest(self) -> None:
        """Load the model manifest from disk, verifying its HMAC signature.

        A manifest that fails verification (hand-edited, corrupted, or
        tampered) is discarded so its checksums can never be used to bless a
        swapped model binary.
        """
        if self.manifest_path.exists():
            try:
                data = json.loads(self.manifest_path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load model manifest: {e}")
                self._manifest = {}
                return
            if isinstance(data, dict) and "models" in data and "hmac" in data:
                if self._verify_manifest_hmac(data["models"], data["hmac"]):
                    self._manifest = data["models"]
                else:
                    logger.warning(
                        "Model manifest failed signature verification -- "
                        "discarding (models will re-register on next download)"
                    )
                    self._manifest = {}
            else:
                # Legacy unsigned manifest: accept contents this once; they
                # are re-signed on the next save.
                self._manifest = data if isinstance(data, dict) else {}
        else:
            self._manifest = {}

    @staticmethod
    def _manifest_hmac(models: dict) -> str:
        payload = json.dumps(models, sort_keys=True, separators=(",", ":")).encode()
        return hmac.new(_MANIFEST_HMAC_KEY, payload, hashlib.sha256).hexdigest()

    def _verify_manifest_hmac(self, models: dict, signature: str) -> bool:
        return hmac.compare_digest(self._manifest_hmac(models), str(signature))

    def _save_manifest(self) -> None:
        """Save the model manifest to disk with an HMAC signature."""
        try:
            signed = {
                "models": self._manifest,
                "hmac": self._manifest_hmac(self._manifest),
            }
            self.manifest_path.write_text(json.dumps(signed, indent=2))
        except OSError as e:
            logger.warning(f"Failed to save model manifest: {e}")

    def _compute_sha256(self, path: Path) -> str:
        """Compute SHA-256 checksum of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def get_model_path(self, model_name: str) -> Path | None:
        """Get the local path to a cached model.

        Args:
            model_name: e.g., "yolo11l" or "yolo11l.pt"

        Returns:
            Path to the model file if cached, None otherwise
        """
        name = model_name.replace(".pt", "")
        model_path = self.cache_dir / f"{name}.pt"
        if model_path.exists():
            return model_path
        return None

    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available locally."""
        return self.get_model_path(model_name) is not None

    def download_model(
        self,
        model_name: str,
        progress_callback: Callable[[float, str], None] | None = None,
        force: bool = False,
    ) -> Path:
        """Download a model to the local cache.

        Args:
            model_name: e.g., "yolo11l"
            progress_callback: Called with (progress_0_to_1, message)
            force: Re-download even if already cached

        Returns:
            Path to the downloaded model file

        Raises:
            ValueError: If model_name is not known
            RuntimeError: If download fails
        """
        name = model_name.replace(".pt", "")
        model_path = self.cache_dir / f"{name}.pt"

        if not force and model_path.exists():
            logger.info(f"Model {name} already cached at {model_path}")
            if progress_callback:
                progress_callback(1.0, "Model already cached")
            return model_path

        info = self.DEFAULT_MODELS.get(name)
        if info is None:
            raise ValueError(
                f"Unknown model: {name}. Available: {list(self.DEFAULT_MODELS.keys())}"
            )

        url = info["url"]
        logger.info(f"Downloading {name} from {url}...")

        if progress_callback:
            progress_callback(0.0, f"Downloading {name}...")

        try:
            req = Request(str(url), headers={"User-Agent": "KawkabAI/1.0"})
            with urlopen(req, timeout=300) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 8192

                with open(model_path, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0 and progress_callback:
                            progress = downloaded / total_size
                            mb = downloaded / (1024 * 1024)
                            total_mb = total_size / (1024 * 1024)
                            progress_callback(
                                progress,
                                f"Downloading {name}: {mb:.1f} / {total_mb:.1f} MB",
                            )

        except URLError as e:
            if model_path.exists():
                model_path.unlink()
            raise RuntimeError(f"Failed to download {name}: {e}") from e

        # Verify checksum if available
        expected_sha256 = info.get("sha256")
        if expected_sha256:
            actual_sha256 = self._compute_sha256(model_path)
            if actual_sha256 != expected_sha256:
                model_path.unlink()
                raise RuntimeError(
                    f"Checksum mismatch for {name}: expected {expected_sha256}, got {actual_sha256}"
                )

        # Update manifest
        self._manifest[name] = {
            "path": str(model_path),
            "size_bytes": model_path.stat().st_size,
            "sha256": self._compute_sha256(model_path),
        }
        self._save_manifest()

        logger.info(f"Model {name} downloaded to {model_path}")
        if progress_callback:
            progress_callback(1.0, f"{name} downloaded successfully")

        return model_path

    def ensure_model(
        self,
        model_name: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> Path:
        """Ensure a model is available, downloading if necessary.

        Args:
            model_name: e.g., "yolo11l"
            progress_callback: Called with progress updates

        Returns:
            Path to the model file
        """
        path = self.get_model_path(model_name)
        if path is not None:
            if progress_callback:
                progress_callback(1.0, "Model ready")
            return path
        return self.download_model(model_name, progress_callback)

    def list_cached_models(self) -> list[str]:
        """List all models currently in the cache."""
        return [f.stem for f in self.cache_dir.glob("*.pt") if f.name != "models.json"]

    def cleanup_cache(self, keep_models: list[str] | None = None) -> int:
        """Remove old/unused models from cache.

        Args:
            keep_models: List of model names to keep (e.g., ["yolo11l"])

        Returns:
            Number of files removed
        """
        keep = set(keep_models or [])
        removed = 0
        for f in self.cache_dir.glob("*.pt"):
            if f.stem not in keep:
                try:
                    f.unlink()
                    removed += 1
                    logger.info(f"Removed cached model: {f.name}")
                except OSError as e:
                    logger.warning(f"Failed to remove {f}: {e}")
        return removed

    def get_cache_size_mb(self) -> float:
        """Get total size of cached models in MB."""
        total = 0
        for f in self.cache_dir.glob("*.pt"):
            total += f.stat().st_size
        return total / (1024 * 1024)
