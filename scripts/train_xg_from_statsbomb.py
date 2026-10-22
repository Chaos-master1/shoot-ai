"""Train xG model coefficients on StatsBomb open data.

Bulk-fetches a large corpus of matches from the StatsBomb open-data repo
(La Liga, Premier League, Champions League, World Cups, Euros), extracts
shot events, trains logistic regression coefficients via batch gradient
descent, runs sanity guards against pathological fits, and saves to
src/kawkab/core/trained_xg_coefficients.json.

The 10 local data/ground_truth/statsbomb/events/3857xxxx.json matches
(WC 2022 + Euro 2024) are HELD OUT of training -- they are the regression
test set (tests/unit/test_regression_xg_models.py), and training on your
test set makes the regression test meaningless.

History: the first version of this script trained on 56 shots from 2
matches, producing a pathological fit (positive distance^2 coefficient ->
xG *increased* with distance; 40m central shot worth 0.98). The sanity
guards below exist so a fit like that can never ship again.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from kawkab.core.xg_trainer import (
    FitShot,
    fit_from_shots,
)

EVENT_DIR = (
    Path(__file__).resolve().parent.parent / "data" / "ground_truth" / "statsbomb" / "events"
)
HOLDOUT_DIR = (
    Path(__file__).resolve().parent.parent / "data" / "ground_truth" / "statsbomb" / "events"
)
CORPUS_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "statsbomb_corpus"
OUTPUT_PATH = SRC_DIR / "kawkab" / "core" / "trained_xg_coefficients.json"
STATSBOMB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
_MAX_RETRIES = 5

# Holdout match ids (WC2022 finals 3857255..3857296 + Euro2024) -- the local
# regression-test set. NEVER train on these.
HOLDOUT_MATCH_IDS = {
    3857255,
    3857271,
    3857272,
    3857273,
    3857274,
    3857275,
    3857276,
    3857277,
    3857278,
    3857296,
}

# (competition_id, season_id, max_matches) -- big-shot-count leagues.
CORPUS_SPECS = [
    (11, 27, 30),  # La Liga 2015/2016
    (11, 1, 30),  # La Liga 2017/2018
    (2, 27, 30),  # Premier League 2015/2016
    (16, 4, 30),  # Champions League 2018/2019
    (16, 1, 30),  # Champions League 2017/2018
    (43, 3, 64),  # FIFA World Cup 2018
    (55, 43, 51),  # UEFA Euro 2020
    (12, 27, 30),  # Serie A 2015/2016
    (9, 27, 30),  # Bundesliga 2015/2016
    (7, 27, 30),  # Ligue 1 2015/2016
]

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
GOAL_CENTER_X = PITCH_LENGTH
GOAL_CENTER_Y = PITCH_WIDTH / 2.0
GOAL_WIDTH = 7.32

MODEL_NAMES = {
    "heuristic": "Heuristic (legacy, hand-set)",
    "enhanced": "Enhanced (hand-set, StatsBomb-calibrated)",
    "trained": "Trained on StatsBomb data (this script)",
}


def _get_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _get_angle(x, y):
    """Angle between the shot direction and the line to the goal CENTER.

    Matches EnhancedXgModel's semantics exactly: angle_deg is the
    deviation from dead central (0 deg = straight at the goal center's
    line, approaching 90 deg = along the goal line to the side), and the
    model uses cos(angle) as the visible-goal fraction with a negative
    coefficient on (1 - cos). The previous version of this helper
    computed the subtended goal-OPENING angle (center + half goal width,
    capped at 90) -- the opposite ordering (central 6m shots got ~18
    deg, goal-line-wide shots got 90 deg), which trained every angle
    coefficient backwards.
    """
    dx = GOAL_CENTER_X - x
    dy = GOAL_CENTER_Y - y
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 0.5:
        return 90.0
    deviation = math.degrees(math.atan2(abs(dy), abs(dx)))
    return min(deviation, 90.0)


def _map_body_part(sb_part):
    m = {
        "Head": "head",
        "Left Foot": "left_foot",
        "Right Foot": "right_foot",
        "Other": "right_foot",
    }
    return m.get(sb_part, "right_foot")


def _map_shot_type(sb_type):
    m = {
        "Open Play": "open_play",
        "Volley": "volley",
        "Half Volley": "half_volley",
        "Free Kick": "free_kick",
        "Penalty": "penalty",
        "Corner": "open_play",
        "Set Piece": "free_kick",
        "Direct Free Kick": "free_kick",
    }
    return m.get(sb_type, "open_play")


def _http_get(url: str) -> bytes | None:
    """GET with retry/backoff. Returns response bytes or None."""
    try:
        import httpx
    except ImportError:
        print("  httpx not available, cannot download")
        return None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                resp = client.get(url)
                if resp.status_code == 429:
                    wait = 2**attempt + random.uniform(0, 1)
                    print(f"  429, retrying in {wait:.1f}s (attempt {attempt}/{_MAX_RETRIES})")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.content
        except Exception as exc:
            if attempt == _MAX_RETRIES:
                print(f"  Failed after {_MAX_RETRIES} attempts: {exc}")
                return None
            wait = 2**attempt + random.uniform(0, 1)
            time.sleep(wait)
    return None


def _discover_corpus_match_ids(max_total: int = 350) -> list[int]:
    """Match ids from CORPUS_SPECS competitions, excluding the holdout set."""
    match_ids: list[int] = []
    for comp_id, season_id, cap in CORPUS_SPECS:
        if len(match_ids) >= max_total:
            break
        url = f"{STATSBOMB_BASE}/matches/{comp_id}/{season_id}.json"
        raw = _http_get(url)
        if raw is None:
            print(f"  Could not list matches for comp {comp_id}/{season_id}")
            continue
        try:
            matches = json.loads(raw)
        except Exception as exc:
            print(f"  Bad match list for {comp_id}/{season_id}: {exc}")
            continue
        ids = [
            m["match_id"] for m in matches if isinstance(m, dict) and m.get("match_id") is not None
        ]
        ids = [i for i in ids if i not in HOLDOUT_MATCH_IDS]
        # Deterministic sample so re-runs use the same corpus.
        random.Random(42).shuffle(ids)
        match_ids.extend(ids[:cap])
        print(f"  comp {comp_id}/{season_id}: {min(len(ids), cap)} matches")
    return match_ids[:max_total]


def _download_corpus(match_ids: list[int]) -> list[Path]:
    """Download (or reuse cached) event files for the corpus matches."""
    CORPUS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, mid in enumerate(match_ids):
        dest = CORPUS_CACHE_DIR / f"{mid}.json"
        if dest.exists() and dest.stat().st_size > 1000:
            paths.append(dest)
            continue
        raw = _http_get(f"{STATSBOMB_BASE}/events/{mid}.json")
        if raw is None:
            continue
        dest.write_bytes(raw)
        paths.append(dest)
        if (i + 1) % 25 == 0:
            print(f"  downloaded {i + 1}/{len(match_ids)} matches...")
    print(f"  corpus: {len(paths)} match files in {CORPUS_CACHE_DIR}")
    return paths


def load_all_shots(event_paths: list[Path]):
    """Extract FitShots from the given StatsBomb event files."""
    shots = []
    for fpath in sorted(event_paths):
        if int(fpath.stem) in HOLDOUT_MATCH_IDS:
            continue  # never train on the regression-test holdout
        try:
            events = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  Skip {fpath.name}: {e}")
            continue
        for ev in events:
            type_info = ev.get("type", {}) or {}
            if type_info.get("name") != "Shot":
                continue
            shot_info = ev.get("shot", {}) or {}
            sb_xg = shot_info.get("statsbomb_xg")
            if sb_xg is None:
                continue
            loc = ev.get("location") or []
            if len(loc) < 2:
                continue
            x, y = float(loc[0]), float(loc[1])
            body_part = _map_body_part((shot_info.get("body_part") or {}).get("name", ""))
            shot_type = _map_shot_type((shot_info.get("type") or {}).get("name", ""))
            is_goal = (shot_info.get("outcome") or {}).get("name") == "Goal"
            under_pressure = ev.get("under_pressure", False)
            freeze_frame = shot_info.get("freeze_frame", [])
            n_opponents = (
                sum(1 for p in freeze_frame if not p.get("teammate", True)) if freeze_frame else 0
            )
            gk_dist = 0.0
            if freeze_frame:
                for p in freeze_frame:
                    if (
                        not p.get("teammate", True)
                        and p.get("position", {}).get("name") == "Goalkeeper"
                    ):
                        ploc = p.get("location", [])
                        if len(ploc) >= 2:
                            gk_dist = _get_distance(x, y, float(ploc[0]), float(ploc[1]))
            is_rebound = False
            is_big_chance = False
            assist_type = "standard"
            if ev.get("pass"):
                pass_type = (ev.get("pass", {}).get("height", {}) or {}).get("name", "")
                if pass_type == "Through Ball":
                    assist_type = "through_ball"
                elif pass_type in ("Cross", "Corner"):
                    assist_type = "cross"

            shots.append(
                FitShot(
                    distance_m=_get_distance(x, y, GOAL_CENTER_X, GOAL_CENTER_Y),
                    angle_deg=_get_angle(x, y),
                    is_header=(body_part == "head"),
                    is_through_ball_assist=(assist_type == "through_ball"),
                    is_cross_assist=(assist_type == "cross"),
                    is_one_on_one=(
                        n_opponents <= 1 and _get_distance(x, y, GOAL_CENTER_X, GOAL_CENTER_Y) < 20
                    ),
                    is_pressed=under_pressure,
                    is_volley=(shot_type in ("volley", "half_volley")),
                    is_free_kick=(shot_type == "free_kick"),
                    gk_distance_m=gk_dist,
                    is_rebound=is_rebound,
                    is_big_chance=is_big_chance,
                    is_goal=is_goal,
                )
            )
    return shots


def _predict_xg(
    coeffs: dict,
    distance_m: float,
    angle_deg: float,
    gk_distance_m: float = 0.0,
    **flags,
) -> float:
    """Evaluate the trained logistic model at an arbitrary shot point.

    Mirrors EnhancedXgModel's feature construction (see xg_model.py),
    INCLUDING the gk_distance terms (gk_distance_m > 0 activates them,
    same as the model). Omitting them in an earlier version made every
    guard evaluation run ~2.9 logits too hot because the fitted
    gk coefficients carry most of the close-range difficulty signal.
    """
    angle_rad = math.radians(max(angle_deg, 0.0))
    gf = math.cos(angle_rad) if angle_rad < math.pi / 2 else 0.0
    d = max(distance_m, 0.5)
    z = coeffs.get("intercept", 0.0)
    z += coeffs.get("distance_m", 0.0) * d
    z += coeffs.get("distance_m_sq", 0.0) * d * d
    z += coeffs.get("angle_sin", 0.0) * (1.0 - gf)
    z += coeffs.get("angle_deg_sq_sin", 0.0) * (1.0 - gf) ** 2
    if gk_distance_m > 0:
        z += coeffs.get("gk_distance_m", 0.0) * gk_distance_m
        z += coeffs.get("gk_distance_m_sq", 0.0) * gk_distance_m * gk_distance_m
    for flag in [
        "is_header",
        "is_through_ball_assist",
        "is_cross_assist",
        "is_one_on_one",
        "is_pressed",
        "is_volley",
        "is_free_kick",
        "is_rebound",
        "is_big_chance",
    ]:
        if flags.get(flag):
            z += coeffs.get(flag, 0.0)
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def sanity_check_coefficients(coeffs: dict) -> list[str]:
    """Guard against pathological fits. Returns list of failure reasons.

    Calibrated against real StatsBomb open-data rates (measured on the
    7.5k-shot corpus this script trains on, cross-checked against
    statsbomb_xg):
    - Empirical goal rate is NOT strictly monotonic in distance: 6-12m
      shots (15.1%) outscore 0-6m shots (10.8%) because close shots are
      disproportionately headers/scrappy box finishes. A small
      near-distance hump is legitimate; what is NEVER legitimate is
      long-range xG comparable to close-range xG (the 56-shot fit gave
      40m central = 0.98).
    - The angle feature must discriminate: at a fixed 11m, a shot from
      near the goal line (deviation ~75deg, tiny visible goal) must be
      worth clearly less than a central one.
    """
    failures: list[str] = []

    xg_6 = _predict_xg(coeffs, 6.0, 0.0, gk_distance_m=max(1.0, 6.0 - 3.0))
    xg_11 = _predict_xg(coeffs, 11.0, 0.0, gk_distance_m=max(1.0, 11.0 - 3.0))
    xg_20 = _predict_xg(coeffs, 20.0, 0.0, gk_distance_m=max(1.0, 20.0 - 3.0))
    xg_30 = _predict_xg(coeffs, 30.0, 0.0, gk_distance_m=max(1.0, 30.0 - 3.0))
    xg_40 = _predict_xg(coeffs, 40.0, 0.0, gk_distance_m=max(1.0, 40.0 - 3.0))

    # 1. Long-range decay: 20m >= 30m >= 40m, and 40m must be small.
    if not (xg_20 >= xg_30 >= xg_40):
        failures.append(
            f"long-range central xG not decreasing: 20m={xg_20:.3f} 30m={xg_30:.3f} 40m={xg_40:.3f}"
        )
    if not (0.002 <= xg_40 <= 0.12):
        failures.append(f"40m central xG implausible: {xg_40:.3f} (expect ~0.01-0.05)")
    # 2. Close range must be clearly better than long range.
    if not (xg_6 > xg_30 and xg_11 > xg_30):
        failures.append(
            f"close-range xG not clearly above 30m xG: "
            f"6m={xg_6:.3f} 11m={xg_11:.3f} 30m={xg_30:.3f}"
        )
    # 3. Reasonable ranges for open-play central shots.
    if not (0.10 <= xg_11 <= 0.50):
        failures.append(
            f"11m central xG implausible: {xg_11:.3f} "
            f"(StatsBomb open-data empirical mean at 6-12m is ~0.15)"
        )
    # 4. Angle discrimination: a near-goal-line wide shot (deviation 80deg
    #    -> cos(80) = 0.17 visible-goal fraction) must be discounted vs a
    #    dead-central shot at the same 11m. The fitted deviation-angle
    #    features carry a weaker signal than subtended-angle encodings
    #    (empirical 6-12m goal% by deviation bucket differs by only a few
    #    points), so this is a directional check (wide clearly lower), not
    #    a strict halving.
    xg_11_wide = _predict_xg(coeffs, 11.0, 80.0, gk_distance_m=max(1.0, 11.0 - 3.0))
    if not (xg_11_wide < 0.85 * xg_11):
        failures.append(
            f"near-goal-line wide shot not discounted: "
            f"11m@80deg={xg_11_wide:.3f} vs 11m central={xg_11:.3f}"
        )
    # 5. Headers harder than feet at the same spot.
    xg_header = _predict_xg(coeffs, 6.0, 0.0, gk_distance_m=max(1.0, 6.0 - 3.0), is_header=True)
    xg_foot = _predict_xg(coeffs, 6.0, 0.0, gk_distance_m=max(1.0, 6.0 - 3.0))
    if xg_header >= xg_foot:
        failures.append(f"header xG ({xg_header:.3f}) >= foot xG ({xg_foot:.3f}) at 6m central")
    # 6. No absurd flag magnitudes (the 56-shot fit had is_header=-4.2).
    for flag in ("is_header", "is_free_kick", "is_pressed", "is_one_on_one"):
        v = coeffs.get(flag, 0.0)
        if abs(v) > 3.0:
            failures.append(f"{flag} coefficient extreme: {v:+.3f} (|coef| > 3.0)")
    return failures


def main():
    print("=== xG Coefficient Trainer (full-corpus) ===")
    print()

    match_ids = _discover_corpus_match_ids()
    if not match_ids:
        print("No corpus matches discoverable (offline?). Aborting -- keeping existing weights.")
        sys.exit(1)
    event_paths = _download_corpus(match_ids)

    shots = load_all_shots(event_paths)
    print(
        f"Loaded {len(shots)} shots from StatsBomb corpus "
        f"({len(event_paths)} matches, holdout excluded)"
    )

    goals = sum(1 for s in shots if s.is_goal)
    print(f"Goals: {goals} ({goals / max(len(shots), 1) * 100:.1f}%)")
    print()

    if len(shots) < 500:
        print(
            f"Too few shots ({len(shots)}), need >= 500 for a stable fit. Aborting -- "
            "keeping existing weights rather than shipping another small-sample fit."
        )
        sys.exit(1)

    train_coeffs = fit_from_shots(shots, model_name="trained")

    failures = sanity_check_coefficients(train_coeffs)
    if failures:
        print("SANITY CHECK FAILED -- refusing to save this fit:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("Sanity checks passed (monotonic distance, angle, body part, free kick).")
    print()

    nonzero = {k: v for k, v in train_coeffs.items() if v != 0.0 or k.startswith("_")}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(nonzero, f, indent=2)
    print(f"Trained coefficients saved to {OUTPUT_PATH}")
    print()

    print("Trained coefficients:")
    for name, val in train_coeffs.items():
        if name.startswith("_"):
            continue
        print(f"  {name:25s} = {val:+.6f}")
    print()
    print("Metadata:")
    print(f"  n_shots  = {train_coeffs.get('_n_shots', '?')}")
    print(f"  goal_rate = {train_coeffs.get('_goal_rate', '?'):.4f}")
    print(f"  model     = {train_coeffs.get('_model_name', '?')}")


if __name__ == "__main__":
    main()
