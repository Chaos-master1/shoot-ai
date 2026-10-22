"""Download ground truth datasets for validation (SkillCorner, Metrica, StatsBomb).

Usage:
    python scripts/download_ground_truth.py --all
    python scripts/download_ground_truth.py --skillcorner
    python scripts/download_ground_truth.py --metrica
    python scripts/download_ground_truth.py --statsbomb
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("download_ground_truth")


def download_skillcorner(output_dir: Path) -> bool:
    """Download SkillCorner open data (10 matches, A-League 2024/25)."""
    url = "https://github.com/SkillCorner/opendata/archive/refs/heads/master.zip"
    import io
    import zipfile

    try:
        import httpx

        resp = httpx.get(url, timeout=120, follow_redirects=True)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            zf.extractall(output_dir)
        extracted = output_dir / "opendata-main"
        if extracted.exists():
            target = output_dir / "skillcorner"
            extracted.rename(target)
        logger.info(f"SkillCorner data downloaded to {output_dir / 'skillcorner'}")
        return True
    except Exception as e:
        logger.error(f"SkillCorner download failed: {e}")
        return False


def download_metrica(output_dir: Path) -> bool:
    """Download Metrica Sports sample data (2 matches with tracking + events)."""
    base = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
    files = [
        "Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
        "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
        "Sample_Game_1/Sample_Game_1_Events.json",
        "Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
        "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv",
        "Sample_Game_2/Sample_Game_2_Events.json",
    ]
    try:
        import httpx

        client = httpx.Client(timeout=60, follow_redirects=True)
        metrica_dir = output_dir / "metrica"
        for fpath in files:
            url = f"{base}/{fpath}"
            local = metrica_dir / fpath
            local.parent.mkdir(parents=True, exist_ok=True)
            resp = client.get(url)
            resp.raise_for_status()
            local.write_bytes(resp.content)
            logger.info(f"Downloaded {local}")
        logger.info(f"Metrica data downloaded to {metrica_dir}")
        return True
    except Exception as e:
        logger.error(f"Metrica download failed: {e}")
        return False


def download_statsbomb_fixtures(output_dir: Path) -> bool:
    """Download StatsBomb World Cup 2022 + EURO 2024 fixture metadata."""
    try:
        import httpx

        client = httpx.Client(timeout=60, follow_redirects=True)

        # Competition JSON
        comp_url = (
            "https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json"
        )
        resp = client.get(comp_url)
        resp.raise_for_status()
        comps = resp.json()

        statsbomb_dir = output_dir / "statsbomb"
        statsbomb_dir.mkdir(parents=True, exist_ok=True)
        (statsbomb_dir / "competitions.json").write_text(json.dumps(comps, indent=2))
        logger.info(f"Saved {len(comps)} competitions to statsbomb/competitions.json")

        # Fetch matches for WC 2022 (comp 43, season 106) and EURO 2024 (comp 55, season 282)
        targets = [(43, 106), (55, 282)]
        match_ids = []
        for comp_id, season_id in targets:
            matches_url = (
                f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/"
                f"{comp_id}/{season_id}.json"
            )
            resp = client.get(matches_url)
            if resp.status_code != 200:
                continue
            matches = resp.json()
            (statsbomb_dir / f"matches_{comp_id}_{season_id}.json").write_text(
                json.dumps(matches, indent=2)
            )
            for m in matches:
                match_ids.append(m["match_id"])
            logger.info(f"Saved {len(matches)} matches for comp {comp_id}")

        # Download events for each match (limit to 10 for size)
        event_dir = statsbomb_dir / "events"
        event_dir.mkdir(exist_ok=True)
        count = 0
        for mid in match_ids[:10]:
            try:
                ev_url = (
                    f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/"
                    f"{mid}.json"
                )
                resp = client.get(ev_url)
                if resp.status_code != 200:
                    continue
                (event_dir / f"{mid}.json").write_bytes(resp.content)
                count += 1
            except Exception:
                continue
        logger.info(f"Downloaded {count} match event files")

        # Convert to ground truth format
        converter(statsbomb_dir, event_dir)
        logger.info(f"StatsBomb data downloaded to {statsbomb_dir}")
        return True
    except Exception as e:
        logger.error(f"StatsBomb download failed: {e}")
        return False


def converter(statsbomb_dir: Path, event_dir: Path):
    """Convert StatsBomb events to ValidationService ground truth JSON."""
    gt_dir = statsbomb_dir / "ground_truth"
    gt_dir.mkdir(exist_ok=True)

    import glob

    for ev_file in glob.glob(str(event_dir / "*.json")):
        try:
            events = json.loads(Path(ev_file).read_text())
        except Exception:
            continue
        gt_events = []
        for ev in events:
            ev_type = ev.get("type", {}).get("name", "").lower()
            ts = ev.get("timestamp", "00:00:00.000")
            # Convert HH:MM:SS.mmm to seconds
            parts = ts.replace(",", ".").split(":")
            secs = (
                float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                if len(parts) == 3
                else 0.0
            )
            team = ev.get("team", {}).get("name", "unknown")
            player_id = ev.get("player", {}).get("id")
            gt_events.append(
                {
                    "event_type": ev_type,
                    "timestamp": secs,
                    "team": team,
                    "player_id": player_id,
                }
            )
        if gt_events:
            match_id = Path(ev_file).stem
            (gt_dir / f"{match_id}_gt.json").write_text(json.dumps(gt_events, indent=2))
            logger.info(f"Converted {len(gt_events)} events for match {match_id}")


def main():
    parser = argparse.ArgumentParser(description="Download ground truth datasets")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--skillcorner", action="store_true")
    parser.add_argument("--metrica", action="store_true")
    parser.add_argument("--statsbomb", action="store_true")
    parser.add_argument("--output", default="data/ground_truth", help="Output directory")
    args = parser.parse_args()

    if not any([args.all, args.skillcorner, args.metrica, args.statsbomb]):
        parser.print_help()
        return

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if args.all or args.skillcorner:
        download_skillcorner(output)
    if args.all or args.metrica:
        download_metrica(output)
    if args.all or args.statsbomb:
        download_statsbomb_fixtures(output)


if __name__ == "__main__":
    main()
