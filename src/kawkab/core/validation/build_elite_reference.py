"""Build the elite reference population artifact (offline, not at runtime).

The reference population is the bundled StatsBomb open-data cohort: 297
top-flight and international matches with competition, season, date and
team labels. This script turns that corpus into a small, committed
artifact (``src/kawkab/core/elite_reference.json``) holding one row per
*team-match* — the unit a coach can be compared against — carrying both
output metrics (shots, xG, goals) and style metrics (PPDA, field tilt,
progressive passes, final-third entries, high turnovers, pass
completion).

    python -m kawkab.core.validation.build_elite_reference \
        --corpus data/statsbomb_corpus \
        --index data/elite_reference/statsbomb_match_index.json \
        --out src/kawkab/core/elite_reference.json

Rules the build must not break:

1. Both sides of every future comparison are measured with the same
   model AND the same definitions (``core.metric_contract``,
   ``core.style_profile``, ``core.elite_reference.shot_features``), so
   the reference and the user's match are the same measurement.
2. The benchmark xG model is FITTED on this cohort with the reduced
   feature set. Reusing the app's full-feature coefficients without the
   freeze-frame features inflates every shot by roughly 2x (measured:
   0.215 predicted vs 0.105 actual on the same shots).
3. The artifact carries its own provenance: population size, the
   competitions and seasons it covers, the model, the coefficients, the
   metric definitions and the exclusions.

No I/O at import time. Stdlib + the app's own core modules.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kawkab.core import metric_contract
from kawkab.core.elite_reference import (
    BENCHMARK_FEATURES,
    EXCLUDED_FEATURES,
    build_model,
    per_90,
    shot_features,
    shot_xg,
)
from kawkab.core.style_profile import build_profile
from kawkab.core.validation.statsbomb_loader import load_statsbomb_match

DEFAULT_CORPUS = Path("data/statsbomb_corpus")
DEFAULT_INDEX = Path("data/elite_reference/statsbomb_match_index.json")
DEFAULT_OUT = Path("src/kawkab/core/elite_reference.json")

SIDES = ("home", "away")

# Club-name noise that differs between the event feed and the match
# index ("Olympique de Marseille" vs "Marseille").
_NAME_NOISE = re.compile(r"[^a-z0-9]+")


def _normalize_team(name: str) -> str:
    """Lowercase, strip punctuation and club-name noise from a team name."""
    return _NAME_NOISE.sub("", (name or "").lower())


def _side_for_team_names(team_names: set[str], home: str, away: str) -> dict[str, str]:
    """Map each team name seen in the events onto "home" / "away".

    Exact name match first, then a normalized match, then the
    two-team fallback: if one of the two names matches a side exactly,
    the other one is the other side (this is what resolves "Marseille"
    vs "Olympique de Marseille"). A name that cannot be placed is
    dropped rather than mis-attributed.
    """
    mapping: dict[str, str] = {}
    unresolved: set[str] = set()
    for name in team_names:
        if name == home:
            mapping[name] = "home"
        elif name == away:
            mapping[name] = "away"
        else:
            unresolved.add(name)

    if unresolved:
        normalized = {_normalize_team(home): "home", _normalize_team(away): "away"}
        still: set[str] = set()
        for name in unresolved:
            side = normalized.get(_normalize_team(name))
            if side:
                mapping[name] = side
            else:
                still.add(name)
        if still and len(team_names) == 2 and len(mapping) == 1:
            known_side = next(iter(mapping.values()))
            other = "away" if known_side == "home" else "home"
            for name in still:
                mapping[name] = other
        elif still:
            for name in still:
                mapping[name] = ""
    return mapping


def _features_for_shot(shot) -> dict[str, Any]:
    """Benchmark feature dict for one corpus shot."""
    return shot_features(
        distance_m=shot.distance_m,
        angle_deg=shot.angle_deviation_deg,
        body_part=shot.body_part,
        shot_type=shot.shot_type,
        was_pressed=shot.is_pressed,
        is_one_on_one=shot.is_one_on_one,
    )


def _fit_shot(shot):
    """The trainer's feature record for one corpus shot (penalties skipped)."""
    from kawkab.core.xg_trainer import FitShot

    if shot.shot_type == "penalty":
        return None
    return FitShot(
        distance_m=shot.distance_m,
        angle_deg=shot.angle_deviation_deg,
        is_header=(shot.body_part == "head"),
        is_one_on_one=shot.is_one_on_one,
        is_pressed=shot.is_pressed,
        is_volley=False,
        is_free_kick=(shot.shot_type == "free_kick"),
        gk_distance_m=0.0,
        is_rebound=False,
        is_big_chance=False,
        is_goal=shot.is_goal,
    )


def _fit_benchmark_model(fit_shots: list[Any]) -> dict[str, float]:
    """Fit the camera-observable xG model on the cohort's own shots.

    Penalties are excluded (fixed 0.76 xG, not learned) exactly like the
    app's own trainer does.
    """
    from kawkab.core.xg_trainer import fit_from_shots

    return fit_from_shots(fit_shots, model_name="elite-reference-camera")


def _parse_corpus(
    corpus_dir: Path, index: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[Any], list[str]]:
    """One read per file: shots for fitting, style profiles, team labels.

    Events are dropped as soon as a match's profile is computed, so the
    build stays flat in memory on a 297-match corpus.
    """
    parsed: list[dict[str, Any]] = []
    fit_shots: list[Any] = []
    unmatched: list[str] = []

    for path in sorted(corpus_dir.glob("*.json")):
        meta = index.get(path.stem)
        if not meta:
            unmatched.append(path.stem)
            continue
        match = load_statsbomb_match(path)
        home = meta.get("home_team") or ""
        away = meta.get("away_team") or ""

        names = set(match.team_events) | {s.team_name for s in match.shots if s.team_name}
        side_of = _side_for_team_names(names, home, away)

        events_by_side: dict[str, list[dict[str, Any]]] = {s: [] for s in SIDES}
        for name, events in match.team_events.items():
            side = side_of.get(name, "")
            if side:
                events_by_side[side].extend(events)

        profiles = {
            side: build_profile(
                events_by_side[side],
                events_by_side["away" if side == "home" else "home"],
                90.0,
            )
            for side in SIDES
        }

        shots: list[dict[str, Any]] = []
        for shot in match.shots:
            side = side_of.get(shot.team_name, "")
            if not side:
                continue
            shots.append(
                {
                    "side": side,
                    "features": _features_for_shot(shot),
                    "is_goal": shot.is_goal,
                    "sb_xg": float(shot.statsbomb_xg),
                }
            )
            fit_shot = _fit_shot(shot)
            if fit_shot is not None:
                fit_shots.append(fit_shot)

        parsed.append(
            {
                "match_id": str(path.stem),
                "meta": meta,
                "home": home,
                "away": away,
                "shots": shots,
                "profiles": {s: profiles[s]["metrics"] for s in SIDES},
                "styles": profiles,
            }
        )
    return parsed, fit_shots, unmatched


def build_reference(
    corpus_dir: Path | str = DEFAULT_CORPUS,
    index_path: Path | str = DEFAULT_INDEX,
) -> dict[str, Any]:
    """Build the reference artifact dict from the corpus + match index."""
    corpus_dir = Path(corpus_dir)
    index_path = Path(index_path)
    raw_index = json.loads(index_path.read_text())
    index = raw_index.get("matches", raw_index)

    parsed, fit_shots, unmatched = _parse_corpus(corpus_dir, index)
    coefficients = _fit_benchmark_model(fit_shots)
    model = build_model(
        coefficients,
        source="elite reference (fitted on the bundled StatsBomb elite cohort)",
    )

    rows: list[dict[str, Any]] = []
    predictions: list[float] = []
    outcomes: list[int] = []
    statsbomb_xgs: list[float] = []
    team_match_deltas: list[float] = []
    for record in parsed:
        buckets = {side: {"shots": 0, "xg": 0.0, "goals": 0, "sb_xg": 0.0} for side in SIDES}
        for shot in record["shots"]:
            bucket = buckets[shot["side"]]
            bucket["shots"] += 1
            bucket["goals"] += 1 if shot["is_goal"] else 0
            predicted = shot_xg(model, shot["features"])
            bucket["xg"] += predicted
            bucket["sb_xg"] += shot["sb_xg"]
            predictions.append(predicted)
            outcomes.append(1 if shot["is_goal"] else 0)
            statsbomb_xgs.append(shot["sb_xg"])
        for side in SIDES:
            bucket = buckets[side]
            if bucket["shots"]:
                team_match_deltas.append(abs(bucket["xg"] - bucket["sb_xg"]))

        meta = record["meta"]
        minutes = 90.0
        for side in SIDES:
            bucket = buckets[side]
            style = {
                key: (None if value is None else round(float(value), 4))
                for key, value in record["profiles"][side].items()
            }
            rows.append(
                {
                    "match_id": record["match_id"],
                    "date": meta.get("match_date"),
                    "competition": meta.get("competition"),
                    "season": meta.get("season"),
                    "team": record["home"] if side == "home" else record["away"],
                    "opponent": record["away"] if side == "home" else record["home"],
                    "side": side,
                    "minutes": minutes,
                    "shots": bucket["shots"],
                    "xg": round(bucket["xg"], 4),
                    "goals": bucket["goals"],
                    "shots_per_90": round(per_90(bucket["shots"], minutes), 3),
                    "xg_per_90": round(per_90(bucket["xg"], minutes), 4),
                    "goals_per_90": round(per_90(bucket["goals"], minutes), 3),
                    "style": style,
                }
            )

    per_match = {record["match_id"]: record["meta"] for record in parsed}
    competitions = Counter(meta.get("competition") for meta in per_match.values())
    competition_rows = []
    for name, count in sorted(competitions.items(), key=lambda kv: (-kv[1], kv[0] or "")):
        seasons = sorted(
            {
                meta.get("season")
                for meta in per_match.values()
                if meta.get("competition") == name and meta.get("season")
            }
        )
        competition_rows.append({"name": name, "n_matches": count, "seasons": seasons})

    teams = {r["team"] for r in rows if r["team"]}
    dates = sorted(r["date"] for r in rows if r["date"])
    n_shots = sum(len(record["shots"]) for record in parsed)
    n_penalties = sum(
        1
        for record in parsed
        for shot in record["shots"]
        if shot["features"].get("shot_type") == "penalty"
    )
    style_availability = {
        key: sum(1 for row in rows if row["style"].get(key) is not None)
        for key in metric_contract.STYLE_KEYS
    }

    model_card = _model_card(predictions, outcomes, statsbomb_xgs, team_match_deltas, n_penalties)

    return {
        "schema_version": 2,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "generator": "kawkab.core.validation.build_elite_reference",
        "population": {
            "label": "StatsBomb elite cohort",
            "unit": "team-match",
            "n_matches": len(per_match),
            "n_team_matches": len(rows),
            "n_teams": len(teams),
            "competitions": competition_rows,
            "seasons": sorted({r["season"] for r in rows if r["season"]}),
            "date_from": dates[0] if dates else None,
            "date_to": dates[-1] if dates else None,
            "n_shots": n_shots,
            "style_availability": style_availability,
            "source": (
                "StatsBomb open data (https://github.com/statsbomb/open-data) — "
                "events bundled in data/statsbomb_corpus, match metadata from "
                "the upstream open-data match index."
            ),
        },
        "definitions": metric_contract.definitions(),
        "model": {
            "model": "kawkab-elite-reference-xg",
            "coefficients_source": (
                "fitted on the bundled elite cohort with the benchmark feature set only"
            ),
            "features": list(BENCHMARK_FEATURES),
            "excluded_features": list(EXCLUDED_FEATURES),
            "n_fit_shots": int(coefficients.get("_n_shots", 0)),
            "goal_rate": round(coefficients.get("_goal_rate", 0.0), 5),
            "n_penalties_excluded": n_penalties,
            "coefficients": {
                k: round(float(v), 8) for k, v in coefficients.items() if not k.startswith("_")
            },
            "note": (
                "Both the reference cohort and the user's match are measured "
                "with this model and the same reduced feature set, so the xG "
                "below is a ranking quantity, not the app's headline xG."
            ),
        },
        "rows": rows,
        "model_card": model_card,
        "unmatched_matches": unmatched,
    }


def _model_card(
    predictions: list[float],
    outcomes: list[int],
    statsbomb_xgs: list[float],
    team_match_deltas: list[float],
    n_penalties: int,
) -> dict[str, Any]:
    """The benchmark model's own error on the cohort it is fitted on.

    Publishing this is the point: a percentile is only as trustworthy
    as the measurement under it, and the app can state its Brier score,
    log-loss and per-shot disagreement with StatsBomb's own xG instead
    of asking anyone to take the model on faith. In-sample, and the
    card says so.
    """
    import math

    n = len(predictions)
    if n == 0:
        return {"available": False, "reason": "no scored shots"}

    eps = 1e-6
    brier = sum((p - y) ** 2 for p, y in zip(predictions, outcomes, strict=True)) / n
    log_loss = (
        -sum(
            y * math.log(min(max(p, eps), 1 - eps))
            + (1 - y) * math.log(max(1 - min(max(p, eps), 1 - eps), eps))
            for p, y in zip(predictions, outcomes, strict=True)
        )
        / n
    )
    mae = sum(abs(p - s) for p, s in zip(predictions, statsbomb_xgs, strict=True)) / n

    calibration = []
    for index in range(10):
        low, high = index / 10.0, (index + 1) / 10.0
        bucket = [
            (p, y)
            for p, y in zip(predictions, outcomes, strict=True)
            if low <= p < high or (index == 9 and p == 1.0)
        ]
        if not bucket:
            continue
        calibration.append(
            {
                "bin": f"{int(low * 100)}-{int(high * 100)}%",
                "n": len(bucket),
                "predicted": round(sum(p for p, _ in bucket) / len(bucket), 4),
                "observed": round(sum(y for _, y in bucket) / len(bucket), 4),
            }
        )

    return {
        "available": True,
        "model": "kawkab-elite-reference-xg",
        "n_shots": n,
        "n_penalties": n_penalties,
        "brier": round(brier, 5),
        "log_loss": round(log_loss, 5),
        "mae_vs_statsbomb_xg": round(mae, 5),
        "team_match_xg_mae": round(sum(team_match_deltas) / len(team_match_deltas), 4)
        if team_match_deltas
        else None,
        "calibration": calibration,
        "note": (
            "In-sample: the model is fitted on this cohort and scored on the "
            "same shots. The numbers state how closely it reproduces "
            "StatsBomb's xG, not how it will score a new match."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--index", default=str(DEFAULT_INDEX))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    artifact = build_reference(args.corpus, args.index)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=1, sort_keys=True))
    # The model card is published beside the population so its own error
    # can be read (and cited) without loading the 594-row artifact.
    card = {**artifact["model_card"], "generated_at": artifact["generated_at"]}
    out.with_name("model_card.json").write_text(json.dumps(card, indent=1, sort_keys=True))
    population = artifact["population"]
    print(
        f"wrote {out} — {population['n_matches']} matches, "
        f"{population['n_team_matches']} team-matches, "
        f"{population['n_teams']} teams, "
        f"{population['n_shots']} shots, "
        f"style coverage {population['style_availability']}"
    )
    if artifact["unmatched_matches"]:
        print(f"WARNING: {len(artifact['unmatched_matches'])} matches had no index entry")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
