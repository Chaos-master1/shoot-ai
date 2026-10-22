"""Vendor tracking import — the elite interop path (tracking feeds).

Imports external tracking data (SkillCorner JSON, FIFA EPTS XML, Metrica
raw CSVs) as first-class Kawkab matches with ZERO video capture. Frame
positions are persisted through the same ``tracking_frames`` plumbing the
video pipeline uses (``save_tracking_frames_bulk``), plus a provenance
row in ``tracking_imports`` (migration 030). Every downstream model in
the core stack then runs against real 22-player positional data.

Coordinate handling — everything is normalized to Kawkab meters
(105x68, origin at the home goal-line corner) on import:

    - SkillCorner: x in [-1, 1], y in [-1, 1]  (open-data convention)
    - EPTS:        0-100 normalized, meters, or ±1 — auto-detected per frame
    - Metrica:     [0, 1] x [0, 1] normalized (loader converts to meters)

Event<->frame alignment (``align_events``) maps stored event timestamps
to the tracking frames bracketing them, writing ``event_frame_links``
rows so VAEP / pitch control / EPV can pull positional snapshots around
each event. It assumes events and frames share the same match-time clock
(true when both come from the same vendor import; cross-vendor clock
reconciliation is deliberately out of scope here).

Usage:
    from kawkab.services.vendor_tracking_import_service import (
        VendorTrackingImportService,
    )
    svc = VendorTrackingImportService(storage)
    summary = await svc.import_tracking_file("match_skillcorner.json")
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

KAWKAB_X_MAX = 105.0
KAWKAB_Y_MAX = 68.0
DEFAULT_VENDOR_FPS = 25.0
_FRAME_BATCH_SIZE = 500
_DEFAULT_ALIGN_WINDOW_FRAMES = 2

SUPPORTED_VENDORS = ("skillcorner", "epts", "metrica")


# ── Coordinate normalization ────────────────────────────────────────────────


def to_kawkab_meters(
    x: float,
    y: float,
    *,
    convention: str = "auto",
    pitch_length_m: float = KAWKAB_X_MAX,
    pitch_width_m: float = KAWKAB_Y_MAX,
) -> tuple[float, float]:
    """Normalize one vendor (x, y) point to Kawkab meters.

    Conventions:
        "skillcorner" / "unit_pm": x,y in [-1, 1] (SkillCorner open data)
        "normalized_100":          x,y in [0, 100] (some EPTS exports)
        "meters":                  already in pitch meters (FIFA EPTS spec)
        "auto":                    ±1.5 -> unit_pm; anything larger is
                                   treated as meters. A 0-100 feed is
                                   indistinguishable from meters for
                                   in-pitch points, so normalized_100
                                   sources MUST pass the convention
                                   explicitly (the parser defaults cover
                                   each known vendor format).
    """
    if convention in ("skillcorner", "unit_pm"):
        return (x + 1.0) / 2.0 * pitch_length_m, (y + 1.0) / 2.0 * pitch_width_m
    if convention == "normalized_100":
        return x / 100.0 * pitch_length_m, y / 100.0 * pitch_width_m
    if convention == "meters":
        return x, y

    # auto (heuristic): SkillCorner-style unit-plus-minus vs meters. The
    # 0-100 case is intentionally NOT sniffed — see docstring.
    if abs(x) <= 1.5 and abs(y) <= 1.5:
        return (x + 1.0) / 2.0 * pitch_length_m, (y + 1.0) / 2.0 * pitch_width_m
    return x, y


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 256), b""):
            h.update(chunk)
    return h.hexdigest()


def assess_tracking_quality(frames: list[dict], fps: float | None = None) -> dict[str, Any]:
    """Data-quality report for a vendor tracking feed.

    Elite analysts get fired over bad data, not bad math — every vendor
    file carries gaps and glitches, and silently ingesting them poisons
    every downstream model. This reports, over the parsed feed:

    - frame_gaps: adjacent frames farther apart than 2x the median
      interval (missing frames: tracker dropout, vendor cuts).
    - duplicate_timestamps: frames sharing a timestamp (clock glitches).
    - player_count_range: min/max players per frame; a low outlier means
      players fell out of the feed, a high outlier means ghost tracks.
    - out_of_bounds: player positions farther than 3 m outside the pitch
      (real feeds have some; a high rate means a broken unit system or a
      mis-mapped coordinate frame).
    - missing_ball_pct: frames without a ball position.

    Returned in the import response and stored in the provenance record,
    so a downstream consumer can see what came in BEFORE models ran.
    """
    n = len(frames)
    if n == 0:
        return {"frames": 0}

    times = [float(fr.get("timestamp", 0.0) or 0.0) for fr in frames]
    deltas = [b - a for a, b in zip(times, times[1:], strict=False) if b > a]
    median_dt = sorted(deltas)[len(deltas) // 2] if deltas else 0.0
    gap_threshold = (2.0 * median_dt) if median_dt > 0 else 1e9
    frame_gaps = sum(1 for d in deltas if d > gap_threshold)

    seen: dict[float, int] = {}
    for t in times:
        seen[t] = seen.get(t, 0) + 1
    duplicate_timestamps = sum(c - 1 for c in seen.values() if c > 1)

    counts = [len(fr.get("player_detections") or []) for fr in frames]
    out_of_bounds = 0
    missing_ball = 0
    for fr in frames:
        if not fr.get("ball"):
            missing_ball += 1
        for p in fr.get("player_detections") or []:
            if (
                p.get("x", 0.0) < -3.0
                or p.get("x", 0.0) > KAWKAB_X_MAX + 3.0
                or p.get("y", 0.0) < -3.0
                or p.get("y", 0.0) > KAWKAB_Y_MAX + 3.0
            ):
                out_of_bounds += 1

    total_positions = sum(counts) or 1
    return {
        "frames": n,
        "median_frame_interval_s": round(median_dt, 4),
        "frame_gaps": frame_gaps,
        "duplicate_timestamps": duplicate_timestamps,
        "player_count_min": min(counts),
        "player_count_max": max(counts),
        "out_of_bounds_positions": out_of_bounds,
        "out_of_bounds_pct": round(100.0 * out_of_bounds / total_positions, 2),
        "missing_ball_pct": round(100.0 * missing_ball / n, 2),
        "fps": fps,
    }


def detect_vendor(path: str | Path) -> str | None:
    """Sniff the vendor from file content (SkillCorner JSON vs EPTS XML)."""
    p = Path(path)
    if p.suffix.lower() == ".xml":
        try:
            head = p.open("rb").read(4096).decode("utf-8", errors="ignore")
        except OSError:
            return None
        lowered = head.lower()
        if "trainingsession" in lowered or "epts" in lowered:
            return "epts"
        return None
    if p.suffix.lower() == ".json":
        return "skillcorner"
    return None  # metrica needs two CSVs — never auto-detected


# ── Vendor parsers: all return (frames, player_meta, fps) ───────────────────
#
# frames: list of dicts {
#     frame_number: int, timestamp: float (seconds, match clock), period: int,
#     player_detections: [{track_id: int, x, y, speed}],  # Kawkab meters
#     ball: {x, y, z} | None,                              # Kawkab meters
# }
# player_meta: {track_id: {"name": str, "team": "home"|"away", "position": str}}


def parse_skillcorner(
    path: str | Path, *, pitch_length_m: float = KAWKAB_X_MAX, pitch_width_m: float = KAWKAB_Y_MAX
) -> tuple[list[dict], dict[int, dict], float]:
    """SkillCorner JSON (open-data or pro export shape).

    Timestamp scale: SkillCorner "time" is milliseconds since period
    start, but some exports use seconds. The scale is decided from the
    median inter-frame delta — ~1000/fps ms reads as ~40 between frames,
    a seconds clock reads as ~0.04 — which self-calibrates per file
    instead of guessing from absolute magnitude.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        data = {"frames": data}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object or array")

    raw_frames = data.get("frames", data.get("tracking", []))
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError(f"{path}: no frames array found")

    def _raw_time(item: dict) -> float:
        ts_raw = item.get("time", item.get("timestamp", 0.0))
        try:
            return float(ts_raw)
        except (TypeError, ValueError):
            return 0.0

    # Timestamp scale detection (see docstring)
    times = [_raw_time(it) for it in raw_frames if isinstance(it, dict)]
    deltas = [b - a for a, b in zip(times, times[1:], strict=False) if b != a]
    if deltas:
        deltas.sort()
        median_delta = deltas[len(deltas) // 2]
        is_ms = median_delta > 5.0
    else:
        is_ms = (times[0] if times else 0.0) > 10000.0

    # Player metadata: SkillCorner open data lists players with
    # "track_id" matching the per-frame index, plus team/position info.
    player_meta: dict[int, dict] = {}
    for p in data.get("players", []) or []:
        if not isinstance(p, dict):
            continue
        try:
            tid = int(str(p.get("track_id", p.get("id", -1))))
        except (TypeError, ValueError):
            continue
        player_meta[tid] = {
            "name": str(p.get("name", "")),
            "team": str(p.get("side", p.get("team", "home"))).lower(),
            "position": str(p.get("position", "")),
            "jersey_number": p.get("number", p.get("jersey_number")),
        }

    fps = float(data.get("fps", DEFAULT_VENDOR_FPS))
    frames: list[dict] = []
    for i, item in enumerate(raw_frames):
        if not isinstance(item, dict):
            continue
        try:
            frame_number = int(str(item.get("frame_id", item.get("id", item.get("frame", i)))))
            timestamp = _raw_time(item)
            if is_ms:
                timestamp = timestamp / 1000.0
        except (TypeError, ValueError) as exc:
            logger.warning("Skipping SkillCorner frame %d: %s", i, exc)
            continue

        players: list[dict] = []
        for pd in item.get("players", []) or []:
            if not isinstance(pd, dict):
                continue
            try:
                tid = int(str(pd.get("track_id", pd.get("id", -1))))
                x, y = to_kawkab_meters(
                    float(pd.get("x", 0.0)),
                    float(pd.get("y", 0.0)),
                    convention="skillcorner",
                    pitch_length_m=pitch_length_m,
                    pitch_width_m=pitch_width_m,
                )
            except (TypeError, ValueError):
                continue
            players.append(
                {
                    "track_id": tid,
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "speed": float(pd.get("speed", 0.0) or 0.0),
                }
            )

        ball_raw = item.get("ball")
        ball = None
        if isinstance(ball_raw, dict):
            try:
                bx, by = to_kawkab_meters(
                    float(ball_raw.get("x", 0.0)),
                    float(ball_raw.get("y", 0.0)),
                    convention="skillcorner",
                    pitch_length_m=pitch_length_m,
                    pitch_width_m=pitch_width_m,
                )
                ball = {
                    "x": round(bx, 3),
                    "y": round(by, 3),
                    "z": float(ball_raw.get("z", 0.0) or 0.0),
                }
            except (TypeError, ValueError):
                ball = None

        frames.append(
            {
                "frame_number": frame_number,
                "timestamp": timestamp,
                "period": int(item.get("period", 0) or 0),
                "player_detections": players,
                "ball": ball,
            }
        )

    if not frames:
        raise ValueError(f"{path}: parsed zero usable frames")
    return frames, player_meta, fps


def _epts_local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _epts_attr(el: ET.Element, *names: str) -> str:
    """Case-insensitive attribute lookup (EPTS exporters vary in casing:
    PlayerId vs playerId vs player_id)."""
    lowered = {k.lower(): v for k, v in el.attrib.items()}
    for name in names:
        if name in lowered:
            return lowered[name]
    return ""


def parse_epts(
    path: str | Path,
    *,
    pitch_length_m: float = KAWKAB_X_MAX,
    pitch_width_m: float = KAWKAB_Y_MAX,
    convention: str = "meters",
) -> tuple[list[dict], dict[int, dict], float]:
    """FIFA EPTS XML tracking export (namespace-tolerant).

    Walks every element whose local name is 'Frame'; ball is a 'Ball'
    child, players are 'Player' children (PlayerId/x/y attributes or
    children, case-insensitive). Coordinates: FIFA's spec is meters
    (the default); pass convention="normalized_100" for 0-100 exports.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    fps_raw = None
    for el in root.iter():
        if _epts_local(el.tag) == "fps" and el.text:
            try:
                fps_raw = float(el.text)
                break
            except ValueError:
                pass
    fps = fps_raw or DEFAULT_VENDOR_FPS

    player_meta: dict[int, dict] = {}
    frames: list[dict] = []
    first_ts: float | None = None

    for el in root.iter():
        if _epts_local(el.tag) != "frame":
            continue
        attrs = el.attrib
        try:
            frame_number = int(attrs.get("id", attrs.get("frameid", len(frames))))
            ts_raw = attrs.get("utc", attrs.get("timestamp", attrs.get("time", "0")))
            ts_val = float(ts_raw)
            # EPTS utc is ms-since-epoch; convert to relative seconds
            if first_ts is None:
                first_ts = ts_val
            timestamp = (ts_val - (first_ts or 0.0)) / 1000.0 if ts_val > 1e10 else ts_val
        except (TypeError, ValueError) as exc:
            logger.warning("Skipping EPTS frame %d: %s", len(frames), exc)
            continue

        period = 0
        players: list[dict] = []
        ball = None
        for child in el.iter():
            local = _epts_local(child.tag)
            if local == "section" and child.text:
                with contextlib.suppress(ValueError):
                    period = int(child.text)
            elif local == "ball":
                bx = _epts_attr(child, "x") or (child.findtext(".//x") or "0")
                by = _epts_attr(child, "y") or (child.findtext(".//y") or "0")
                bz = _epts_attr(child, "z") or (child.findtext(".//z") or "0")
                try:
                    mx, my = to_kawkab_meters(
                        float(bx),
                        float(by),
                        convention=convention,
                        pitch_length_m=pitch_length_m,
                        pitch_width_m=pitch_width_m,
                    )
                    ball = {"x": round(mx, 3), "y": round(my, 3), "z": float(bz)}
                except (TypeError, ValueError):
                    ball = None
            elif local == "player":
                pid_raw = _epts_attr(child, "playerid", "id") or child.findtext(".//playerid") or ""
                px = _epts_attr(child, "x") or (child.findtext(".//x") or "0")
                py = _epts_attr(child, "y") or (child.findtext(".//y") or "0")
                try:
                    tid = int(float(pid_raw))
                    mx, my = to_kawkab_meters(
                        float(px),
                        float(py),
                        convention=convention,
                        pitch_length_m=pitch_length_m,
                        pitch_width_m=pitch_width_m,
                    )
                except (TypeError, ValueError):
                    continue
                players.append(
                    {
                        "track_id": tid,
                        "x": round(mx, 3),
                        "y": round(my, 3),
                        "speed": float(_epts_attr(child, "speed") or 0.0),
                    }
                )
                if tid not in player_meta:
                    player_meta[tid] = {
                        "name": _epts_attr(child, "name"),
                        "team": (_epts_attr(child, "team") or "home").lower(),
                        "position": "",
                    }

        if players or ball is not None:
            frames.append(
                {
                    "frame_number": frame_number,
                    "timestamp": timestamp,
                    "period": period,
                    "player_detections": players,
                    "ball": ball,
                }
            )

    if not frames:
        raise ValueError(f"{path}: no EPTS Frame elements found — unsupported variant")
    return frames, player_meta, fps


def parse_metrica(
    home_csv: str | Path, away_csv: str | Path
) -> tuple[list[dict], dict[int, dict], float]:
    """Metrica raw CSVs via the existing validation loader (meters out)."""
    from kawkab.core.validation.metrica_loader import load_metrica_match

    mm = load_metrica_match(home_csv, away_csv)
    player_meta: dict[int, dict] = {}
    frames: list[dict] = []
    for fr in mm.frames:
        players: list[dict] = []
        for idx, pos in enumerate(fr.home):
            tid = 1 + idx
            players.append(
                {"track_id": tid, "x": round(pos[0], 3), "y": round(pos[1], 3), "speed": 0.0}
            )
            player_meta.setdefault(tid, {"name": f"Home {idx + 1}", "team": "home", "position": ""})
        for idx, pos in enumerate(fr.away):
            tid = 101 + idx
            players.append(
                {"track_id": tid, "x": round(pos[0], 3), "y": round(pos[1], 3), "speed": 0.0}
            )
            player_meta.setdefault(tid, {"name": f"Away {idx + 1}", "team": "away", "position": ""})
        frames.append(
            {
                "frame_number": fr.frame,
                "timestamp": fr.time_s,
                "period": fr.period,
                "player_detections": players,
                "ball": {"x": round(fr.ball[0], 3), "y": round(fr.ball[1], 3), "z": 0.0}
                if fr.ball
                else None,
            }
        )
    if not frames:
        raise ValueError("Metrica files produced zero frames")
    return frames, player_meta, mm.fps


# ── The import service ──────────────────────────────────────────────────────


class VendorTrackingImportService:
    """Imports a vendor tracking feed into the Kawkab database (no video)."""

    def __init__(self, storage_service: Any) -> None:
        self.storage = storage_service

    async def import_frames(
        self,
        frames: list[dict],
        *,
        player_meta: dict[int, dict],
        vendor: str,
        source_path: str,
        source: str = "file",
        match_id: int | None = None,
        match_name: str | None = None,
        home_team: str | None = None,
        away_team: str | None = None,
        fps: float | None = None,
        pitch_length_m: float = KAWKAB_X_MAX,
        pitch_width_m: float = KAWKAB_Y_MAX,
    ) -> dict[str, Any]:
        """Persist an in-memory frame list through the tracking pipeline.

        This is steps 1-7 of ``import_tracking_file`` factored out so
        parsers that don't start from a file on disk (kloppy-backed
        vendor datasets) land in the exact same tables with the same
        provenance row, quality report, idempotence and metric caching.

        ``frames`` are already in Kawkab meters in the canonical shape
        documented at the top of this module. ``source`` travels through
        the provenance row's metadata so consumers can distinguish a
        file-parsed feed from a library-deserialized one.
        """
        if not frames:
            raise ValueError("no frames to import")

        # 1. Match row (or attach to an existing match)
        if match_id is None:
            home = home_team or f"{vendor.capitalize()} Home"
            away = away_team or f"{vendor.capitalize()} Away"
            name = match_name or f"{home} vs {away} ({vendor} tracking import)"
            match_id = await self.storage.save_match(
                name=name, video_path="", home_team=home, away_team=away
            )
            if not match_id:
                raise RuntimeError("storage refused to create the match row")
        elif match_name or home_team or away_team:
            await self._relabel_match(match_id, match_name, home_team, away_team)

        # 2. Idempotence: same checksum + vendor on the same match -> skip
        checksum = _sha256_file(Path(source_path)) if Path(source_path).exists() else ""
        existing = await self.storage.get_tracking_imports(match_id)
        if any(e.get("vendor") == vendor and e.get("checksum") == checksum for e in existing):
            return {
                "match_id": match_id,
                "vendor": vendor,
                "deduplicated": True,
                "frames_imported": 0,
                "players_registered": 0,
                "events_aligned": 0,
                "checksum": checksum,
            }

        # 3. Persist frames via the SAME plumbing the video pipeline uses
        frame_rows = [
            {
                "frame_number": fr["frame_number"],
                "timestamp": fr["timestamp"],
                "player_detections": [
                    {
                        "track_id": p["track_id"],
                        "x": p["x"],
                        "y": p["y"],
                        "speed": p.get("speed", 0.0),
                    }
                    for p in fr["player_detections"]
                ],
                "ball_detections": (
                    [
                        {
                            "track_id": 999,
                            "x": fr["ball"]["x"],
                            "y": fr["ball"]["y"],
                            "z": fr["ball"].get("z", 0.0),
                        }
                    ]
                    if fr.get("ball")
                    else []
                ),
            }
            for fr in frames
        ]
        frames_saved = 0
        for i in range(0, len(frame_rows), _FRAME_BATCH_SIZE):
            batch = frame_rows[i : i + _FRAME_BATCH_SIZE]
            frames_saved += await self.storage.save_tracking_frames_bulk(match_id, batch)

        # 4. Register players (from metadata, else first frame's track ids)
        registered = 0
        meta = player_meta
        if not meta and frames and frames[0]["player_detections"]:
            meta = {
                p["track_id"]: {"name": f"Player {p['track_id']}", "team": "home", "position": ""}
                for p in frames[0]["player_detections"]
            }
        for tid, pmeta in meta.items():
            team = pmeta.get("team") or "home"
            ok = await self.storage.save_player(
                match_id,
                {
                    "track_id": tid,
                    "name": pmeta.get("name") or f"Player {tid}",
                    "team": team,
                    "position": pmeta.get("position") or None,
                    "jersey_number": pmeta.get("jersey_number"),
                },
            )
            if ok:
                registered += 1

        # 5. Data-quality report — surface what came in BEFORE models run
        quality = assess_tracking_quality(frames, fps=fps)

        # 6. Provenance row (migration 030) — carries the quality report
        # and the source label ("file" parser vs "kloppy" library) so
        # downstream consumers see the feed's condition, not just its size
        await self.storage.save_tracking_import(
            match_id,
            vendor,
            source_path=source_path,
            checksum=checksum,
            fps=fps,
            frame_count=frames_saved,
            pitch_length_m=pitch_length_m,
            pitch_width_m=pitch_width_m,
            metadata={
                "periods": sorted({fr["period"] for fr in frames if fr.get("period")}),
                "players_meta_count": len(meta),
                "source": source,
                "quality": quality,
            },
        )

        # 7. Headline metrics, cached the way the video pipeline does
        await self.storage.save_advanced_metrics(
            match_id, "tracking_frames_total", float(frames_saved), metric_category="import"
        )

        return {
            "match_id": match_id,
            "vendor": vendor,
            "deduplicated": False,
            "frames_imported": frames_saved,
            "players_registered": registered,
            "events_aligned": 0,
            "checksum": checksum,
            "fps": fps,
            "source": source,
            "quality": quality,
        }

    async def import_tracking_file(
        self,
        path: str | Path,
        *,
        vendor: str | None = None,
        match_id: int | None = None,
        match_name: str | None = None,
        home_team: str | None = None,
        away_team: str | None = None,
        away_csv: str | Path | None = None,  # metrica only
        max_frames: int | None = None,
        fps: float | None = None,
        pitch_length_m: float = KAWKAB_X_MAX,
        pitch_width_m: float = KAWKAB_Y_MAX,
    ) -> dict[str, Any]:
        """Import one tracking feed. Creates the match unless match_id given."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"tracking file not found: {path}")

        if vendor is None:
            vendor = "metrica" if away_csv is not None else detect_vendor(path)
        if vendor not in SUPPORTED_VENDORS:
            raise ValueError(f"unsupported vendor {vendor!r}; supported: {SUPPORTED_VENDORS}")

        if vendor == "metrica":
            if away_csv is None:
                raise ValueError("metrica import requires both home and away CSVs (away_csv=)")
            frames, player_meta, parsed_fps = parse_metrica(path, away_csv)
        elif vendor == "epts":
            frames, player_meta, parsed_fps = parse_epts(
                path, pitch_length_m=pitch_length_m, pitch_width_m=pitch_width_m
            )
        else:
            frames, player_meta, parsed_fps = parse_skillcorner(
                path, pitch_length_m=pitch_length_m, pitch_width_m=pitch_width_m
            )

        if fps is None:
            fps = parsed_fps
        if max_frames is not None and max_frames > 0:
            frames = frames[:max_frames]

        # Parse-time provenance: this path parsed a file on disk itself.
        # (kloppy-backed feeds arrive via import_frames with source="kloppy".)
        return await self.import_frames(
            frames,
            player_meta=player_meta,
            vendor=vendor,
            source_path=str(path),
            source="file",
            match_id=match_id,
            match_name=match_name,
            home_team=home_team,
            away_team=away_team,
            fps=fps,
            pitch_length_m=pitch_length_m,
            pitch_width_m=pitch_width_m,
        )

    async def align_events(
        self,
        match_id: int,
        *,
        window_frames: int = _DEFAULT_ALIGN_WINDOW_FRAMES,
        fps: float | None = None,
    ) -> dict[str, Any]:
        """Map stored events to their nearest tracking frames.

        For every event on the match, finds the tracking frame whose
        timestamp is closest to the event's, and (when within
        ``window_frames`` at the feed's fps) writes an ``event_frame_links``
        row. Events without a nearby frame are left unlinked (reported in
        the summary — never silently dropped, per repo invariant).
        """
        if fps is None:
            imports = await self.storage.get_tracking_imports(match_id)
            fps = (imports[0].get("fps") if imports else None) or DEFAULT_VENDOR_FPS

        # Page through all events (get_match_events' default limit is 200)
        events: list[dict] = []
        offset = 0
        while True:
            page = await self.storage.get_match_events(match_id, limit=500, offset=offset)
            if not page:
                break
            events.extend(page)
            if len(page) < 500:
                break
            offset += 500
        if not events:
            return {"match_id": match_id, "events_total": 0, "links_written": 0}

        # Frames in one pass (order guaranteed by frame_number)
        frames = await self.storage.get_tracking_frames(match_id, limit=1_000_000)
        if not frames:
            return {"match_id": match_id, "events_total": len(events), "links_written": 0}

        frame_ts = [(f["frame_number"], float(f.get("timestamp") or 0.0)) for f in frames]
        window_s = window_frames / (fps or DEFAULT_VENDOR_FPS)

        links: list[dict] = []
        matched = 0
        for ev in events:
            ev_ts = ev.get("timestamp")
            if ev_ts is None:
                continue
            try:
                ev_ts = float(ev_ts)
            except (TypeError, ValueError):
                continue
            best = min(frame_ts, key=lambda ft: abs(ft[1] - ev_ts), default=None)
            if best is None or abs(best[1] - ev_ts) > window_s:
                continue
            links.append({"event_id": ev["id"], "frame_number": best[0]})
            matched += 1

        written = 0
        if links:
            for i in range(0, len(links), _FRAME_BATCH_SIZE):
                batch = links[i : i + _FRAME_BATCH_SIZE]
                written += await self.storage.save_event_frame_links_bulk(match_id, batch)

        return {
            "match_id": match_id,
            "events_total": len(events),
            "events_matched": matched,
            "links_written": written,
            "fps": fps,
        }

    async def _relabel_match(
        self,
        match_id: int,
        match_name: str | None,
        home_team: str | None,
        away_team: str | None,
    ) -> None:
        """Best-effort team-label update on an existing match row.

        Only fires when BOTH team names are provided (the existing
        StorageService.update_match_teams signature requires both);
        relabeling is cosmetic and never fails the import.
        """
        try:
            if home_team and away_team and hasattr(self.storage, "update_match_teams"):
                await self.storage.update_match_teams(match_id, home_team, away_team)
        except Exception as exc:
            logger.warning("relabel match %s failed: %s", match_id, exc)
