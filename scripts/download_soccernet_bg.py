"""Download SoccerNet tracking-2023 train split in background."""

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

log_dir = os.path.join(os.path.dirname(__file__), "..", "data", "soccernet")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(log_dir, "download_bg.log"), mode="w"),
    ],
)
log = logging.getLogger("soccernet_dl")

try:
    from SoccerNet.Downloader import SoccerNetDownloader

    dl = SoccerNetDownloader(LocalDirectory=log_dir)

    log.info("=" * 60)
    log.info("Starting SoccerNet tracking-2023 train split download")
    log.info("=" * 60)
    dl.downloadDataTask(
        task="tracking-2023", split=["train"], password="SoccerNet", source="OwnCloud"
    )
    log.info("Train split complete!")

    # Also download a few matches from spotting-ball-2025 (smaller, validation)
    log.info("=" * 60)
    log.info("Starting SoccerNet spotting-ball-2025 download")
    log.info("=" * 60)
    dl.downloadDataTask(task="spotting-ball-2025", split=["train"], version="v0")
    log.info("Ball spotting download complete!")

    log.info("=" * 60)
    log.info("ALL DOWNLOADS COMPLETE")
    log.info("=" * 60)
except Exception as e:
    log.error(f"Download failed: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
