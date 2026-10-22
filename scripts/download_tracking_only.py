"""Download SoccerNet tracking-2023 train split, properly handling resumption."""

import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

log_dir = os.path.join(os.path.dirname(__file__), "..", "data", "soccernet")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(log_dir, "tracking_download.log"), mode="w"),
    ],
)
log = logging.getLogger("soccernet_tracking")

# Delete partial zip if it exists to avoid false "already exists"
zip_path = os.path.join(log_dir, "tracking-2023", "train.zip")
if os.path.exists(zip_path):
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    if size_mb < 8000:
        log.warning(
            f"train.zip exists but only {size_mb:.1f} MB (expected ~9580 MB). Deleting for clean download."
        )
        os.remove(zip_path)

from SoccerNet.Downloader import SoccerNetDownloader

dl = SoccerNetDownloader(LocalDirectory=log_dir)

log.info("=" * 60)
log.info("Starting SoccerNet tracking-2023 train split download")
log.info("(this is ~9.58 GB and may take 1-2 hours)")
log.info("=" * 60)

start = time.time()
dl.downloadDataTask(task="tracking-2023", split=["train"], password="SoccerNet", source="OwnCloud")

elapsed = time.time() - start
final_size = os.path.getsize(zip_path) / (1024 * 1024) if os.path.exists(zip_path) else 0
log.info(f"Download complete! Elapsed: {elapsed / 60:.1f} min, Final size: {final_size:.1f} MB")

# Write a sentinel so we know it's done
try:
    with open(os.path.join(log_dir, "tracking-2023", "DOWNLOAD_COMPLETE"), "w") as f:
        f.write(f"Completed at {time.strftime('%Y-%m-%d %H:%M:%S')}, size={final_size:.1f}MB")
except Exception as e:
    log.warning(f"Could not write sentinel: {e}")
