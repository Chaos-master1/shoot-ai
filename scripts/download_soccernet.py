"""Download all SoccerNet data in sequence: tracking, ball action spotting, calibration."""

import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(os.path.dirname(__file__), "..", "data", "soccernet", "download.log"),
            mode="w",
        ),
    ],
)
log = logging.getLogger(__name__)

from SoccerNet.Downloader import SoccerNetDownloader

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "soccernet")
os.makedirs(DATA_DIR, exist_ok=True)

dl = SoccerNetDownloader(LocalDirectory=DATA_DIR)


def download_with_retry(
    task, split, password="SoccerNet", source="OwnCloud", version=None, max_retries=3
):
    for attempt in range(max_retries):
        try:
            log.info(f"Starting download: task={task}, split={split}, source={source}")
            dl.downloadDataTask(
                task=task, split=split, password=password, version=version, source=source
            )
            log.info(f"Completed: task={task}, split={split}")
            return True
        except Exception as e:
            log.error(f"Attempt {attempt + 1}/{max_retries} failed for {task}/{split}: {e}")
            if attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                log.info(f"Retrying in {wait}s...")
                time.sleep(wait)
    log.error(f"All {max_retries} attempts failed for {task}/{split}")
    return False


# 1. Tracking data (from OwnCloud, password-protected)
log.info("=" * 60)
log.info("STEP 1/3: SoccerNet Tracking data")
log.info("=" * 60)
for split in [["train"], ["test"], ["challenge"]]:
    download_with_retry(task="tracking-2023", split=split, password="SoccerNet", source="OwnCloud")

# 2. Ball Action Spotting 2025 (from HuggingFace)
log.info("=" * 60)
log.info("STEP 2/3: SoccerNet Ball Action Spotting 2025")
log.info("=" * 60)
download_with_retry(
    task="spotting-ball-2025", split=["train", "valid", "test", "challenge"], source="HuggingFace"
)

# 3. Calibration data (from OwnCloud)
log.info("=" * 60)
log.info("STEP 3/3: SoccerNet Calibration data")
log.info("=" * 60)
for split in [["train"], ["valid"], ["test"], ["challenge"]]:
    download_with_retry(task="calibration", split=split, password="SoccerNet", source="OwnCloud")

log.info("=" * 60)
log.info("ALL DOWNLOADS COMPLETE")
log.info("=" * 60)
