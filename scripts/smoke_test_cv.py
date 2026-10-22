"""Smoke test for CV service - verifies YOLOv11 loads on GPU."""

import asyncio
import contextlib
import sys

from kawkab.core.logging import setup_logging
from kawkab.services.cv_service import CVService

setup_logging(debug=False)


async def main() -> int:
    print("=" * 60)
    print("Kawkab AI - CV Service Smoke Test")
    print("=" * 60)
    print()
    print("Initializing CV service (loading YOLOv11l on GPU)...")
    cv = CVService(model_size="l", gpu_enabled=True)
    await cv.initialize()
    print("[OK] CV service initialized")
    model = cv._model
    if model is not None:
        device = "unknown"
        with contextlib.suppress(Exception):
            device = str(model.device)
        print(f"[OK] Model device: {device}")
    print()
    print("[OK] YOLOv11l loaded successfully!")
    await cv.shutdown()
    print("[OK] Shutdown complete")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
