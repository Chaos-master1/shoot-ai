"""One-command setup for the tracking pipeline.

Creates a virtual environment, installs dependencies, downloads model weights,
and validates CUDA availability.

Usage:
    python scripts/setup.py                  # Interactive
    python scripts/setup.py --auto           # Automatic (skip prompts)
    python scripts/setup.py --minimal        # Minimal install (CPU-only deps)
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("setup")


def run(cmd: list[str], cwd: Path | None = None) -> bool:
    try:
        subprocess.check_call(cmd, cwd=cwd)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {' '.join(cmd)}\n{e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Setup tracking pipeline")
    parser.add_argument("--auto", action="store_true", help="Automatic setup")
    parser.add_argument("--minimal", action="store_true", help="CPU-only minimal install")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    logger.info("=" * 60)
    logger.info("  Football Tracking Pipeline — Setup")
    logger.info("=" * 60)

    # Step 1: Python version check
    py_version = sys.version_info
    if py_version < (3, 10):
        logger.error(f"Python >= 3.10 required (got {py_version.major}.{py_version.minor})")
        sys.exit(1)
    logger.info(f"✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")

    # Step 2: Create venv
    venv_dir = root / ".venv"
    if not venv_dir.exists():
        logger.info("Creating virtual environment...")
        if not run([sys.executable, "-m", "venv", str(venv_dir)]):
            sys.exit(1)
    else:
        logger.info("✓ Virtual environment exists")

    # Determine pip/python paths
    if sys.platform == "win32":
        pip_cmd = str(venv_dir / "Scripts" / "pip")
        python_cmd = str(venv_dir / "Scripts" / "python")
    else:
        pip_cmd = str(venv_dir / "bin" / "pip")
        python_cmd = str(venv_dir / "bin" / "python")

    # Step 3: Upgrade pip
    run([python_cmd, "-m", "pip", "install", "--upgrade", "pip", "-q"])

    # Step 4: Install requirements
    req_file = root / "requirements.txt"
    if req_file.exists():
        logger.info("Installing dependencies...")
        run([pip_cmd, "install", "-r", str(req_file)])

    # Step 5: Install core deps
    core = ["numpy>=1.24", "opencv-python>=4.9.0", "pillow>=10.0"]
    if not args.minimal:
        core += [
            "torch>=2.2.0 --index-url https://download.pytorch.org/whl/cu124",
            "ultralytics>=8.4.0",
            "boxmot>=19.0.0",
        ]
    logger.info("Installing core packages...")
    run([pip_cmd, "install"] + core)

    # Step 6: Install project package
    if (root / "pyproject.toml").exists():
        logger.info("Installing project package...")
        run([pip_cmd, "install", "-e", str(root)])

    # Step 7: Create cache directories
    from kawkab.core.paths import get_paths

    paths = get_paths()
    paths.cache.mkdir(parents=True, exist_ok=True)
    (paths.cache / "models").mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Cache directory: {paths.cache}")

    # Step 8: Validate CUDA
    if not args.minimal:
        logger.info("Validating CUDA...")
        result = subprocess.run(
            [
                python_cmd,
                "-c",
                "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')",
            ],
            capture_output=True,
            text=True,
        )
        cuda_available = result.stdout.strip().split("\n")[0] == "True"
        if cuda_available:
            device_name = result.stdout.strip().split("\n")[1]
            logger.info(f"✓ CUDA available: {device_name}")
        else:
            logger.warning("⚠ CUDA not available — GPU acceleration disabled")

    # Step 9: Download model weights
    logger.info("Downloading model weights...")
    run(
        [
            python_cmd,
            "-c",
            """
from kawkab.core.model_manager import ModelManager
mm = ModelManager()
for model in ['yolo11m', 'osnet_x1_0']:
    try:
        mm.ensure_model(model)
        print(f'  Downloaded {model}')
    except Exception as e:
        print(f'  Failed: {model}: {e}')
""",
        ]
    )

    logger.info("=" * 60)
    logger.info("  Setup complete!")
    logger.info(f"  Virtual env: {venv_dir}")
    logger.info(f"  Run: {python_cmd} -m kawkab track --help")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
