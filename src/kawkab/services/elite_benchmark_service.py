"""Elite reference benchmark — one match measured against elite football.

The product question this answers: *was that a good performance, and
by whose standard?* Kawkab can already produce xG, shot counts and
percentiles inside the squad. What it could not do is put those numbers
next to a defined population of elite football, so every number in the
app was unfalsifiable — "1.4 xG" means nothing without "compared with
whom, measured how, on how many matches".

This service is the measuring side of that answer:

- the population is the committed elite cohort
  (``core/elite_reference.json``: 297 matches, 594 team-matches, 144
  teams, 8 competitions) built offline by
  ``core/validation/build_elite_reference.py``;
- both sides are measured with the SAME model and the SAME reduced
  feature contract, so the percentile compares like with like;
- everything the comparison depends on travels with the answer:
  population size, competitions, seasons, model, feature exclusions,
  and the limitations that a single match cannot escape.

Honesty rules (the same ones the rest of the app keeps):

- a metric that cannot be measured for this match is reported as
  unavailable with a reason — never as zero;
- xG is only computed when EVERY shot in the match carries the
  position features the model needs; a partial sum would understate
  the team and tilt its percentile;
- the population is described, never implied. "Elite cohort", not
  "the league".
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from kawkab.core.elite_reference import (
    METRIC_KEYS,
    block_percentile,
    build_model,
    compare_values,
    per_90,
    percentile_interval,
    shot_features,
    shot_xg,
)
from kawkab.core.metric_contract import (
    OUTPUT_KEYS,
    PLAYER_KEYS,
    PLAYER_METRICS,
    SEASON_KEYS,
    STYLE_KEYS,
)
from kawkab.core.player_positions import POSITION_GROUP_LABELS, position_group
from kawkab.core.player_profile import (
    MIN_PLAYER_MINUTES,
    build_player_profile,
    player_unavailable_reason,
)
from kawkab.core.style_profile import build_profile, unavailable_reason

logger = logging.getLogger(__name__)

REFERENCE_PATH = Path(__file__).resolve().parent.parent / "core" / "elite_reference.json"
PLAYER_REFERENCE_PATH = (
    Path(__file__).resolve().parent.parent / "core" / "elite_player_reference.json"
)

# A single match is 90 minutes of football. The cohort is a sample of
# elite matches, not a league table, and the caveats travel with the
# answer so the UI can show them instead of hiding them.
#
# Each limitation carries a stable key so the UI can translate it; the
# text is the English fallback the UI shows when no translation exists.
LIMITATIONS: tuple[dict[str, str], ...] = (
    {
        "key": "single_match",
        "text": (
            "One match is 90 minutes of football — read the percentile as a "
            "single performance, not a season's worth of form."
        ),
    },
    {
        "key": "sample_not_league",
        "text": (
            "The reference is a sample of elite matches, not every elite match, "
            "and it is not a league average."
        ),
    },
    {
        "key": "reduced_feature_model",
        "text": (
            "Benchmark xG comes from a model fitted on this cohort with "
            "camera-observable features only; it is not the app's headline xG."
        ),
    },
    {
        "key": "in_sample_cohort",
        "text": (
            "The benchmark model is fitted on the cohort it measures, so cohort "
            "values are in-sample and a user's shots are out-of-sample."
        ),
    },
)


PLAYER_LIMITATIONS: tuple[dict[str, str], ...] = (
    {
        "key": "player_minutes_floor",
        "text": (
            "A player needs 30 minutes for a per-90 percentile; shorter "
            "appearances are reported with their minutes instead of ranked."
        ),
    },
    {
        "key": "position_groups",
        "text": (
            "Players are compared only within their position group "
            "(GK/DEF/MID/FWD), never against the whole cohort."
        ),
    },
    {
        "key": "player_cohort_sample",
        "text": (
            "The player population is a sample of elite player-matches from "
            "the bundled StatsBomb cohort, not every elite appearance."
        ),
    },
    {
        "key": "player_in_sample",
        "text": (
            "The benchmark model is fitted on the same cohort, so cohort "
            "player xG is in-sample and a user's player is out-of-sample."
        ),
    },
)

# Season blocks keep the population caveats and drop the one that no
# longer applies (a block is more than one match), replacing it with the
# two the block comparison introduces.
SEASON_LIMITATIONS: tuple[dict[str, str], ...] = tuple(
    limitation for limitation in LIMITATIONS if limitation["key"] != "single_match"
) + (
    {
        "key": "block_not_season",
        "text": (
            "A block of N matches is a form snapshot, not a full season: the "
            "same squad can produce different blocks in different months."
        ),
    },
    {
        "key": "matched_block_size",
        "text": (
            "The percentile compares your N-match average against N-match "
            "blocks drawn from the cohort, so it answers a stricter question "
            "than a single-match comparison."
        ),
    },
)

#: Ratio metrics average directly; per-90 metrics pool by minutes so a
#: 60-minute match cannot weigh as much as a full one.
_RATIO_METRICS = ("ppda", "field_tilt", "pass_completion_pct")


class EliteBenchmarkService:
    """Compare a stored match against the elite reference cohort."""

    def __init__(
        self,
        storage_service: Any,
        reference_path: Path | str | None = None,
        player_reference_path: Path | str | None = None,
    ) -> None:
        self._storage = storage_service
        self._reference_path = Path(reference_path) if reference_path else REFERENCE_PATH
        self._player_reference_path = (
            Path(player_reference_path) if player_reference_path else PLAYER_REFERENCE_PATH
        )
        self._artifact: dict[str, Any] | None = None
        self._player_artifact: dict[str, Any] | None = None
        self._player_loaded = False
        self._model: Any = None
        self._population: dict[str, list[float]] = {}
        self._player_population: dict[str, dict[str, list[float]]] = {}
        self._player_group_counts: dict[str, int] = {}

    # ── reference population ────────────────────────────────────────

    def _load_reference(self) -> dict[str, Any]:
        """Load (once) the artifact and the model built from it."""
        if self._artifact is None:
            artifact = json.loads(self._reference_path.read_text())
            coefficients = (artifact.get("model") or {}).get("coefficients") or {}
            self._model = build_model(
                coefficients,
                source=(artifact.get("model") or {}).get("coefficients_source", "elite reference"),
            )
            population: dict[str, list[float]] = {key: [] for key in METRIC_KEYS}
            for row in artifact.get("rows") or []:
                style = row.get("style") or {}
                for key in OUTPUT_KEYS:
                    value = row.get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        population[key].append(float(value))
                for key in STYLE_KEYS:
                    value = style.get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        population[key].append(float(value))
            self._population = population
            self._artifact = artifact
        return self._artifact

    def _load_player_reference(self) -> dict[str, Any] | None:
        """Load the position-matched player population, or None if absent.

        The team model is loaded first so both artifacts are measured with
        the same coefficients; the player artifact's own model block is
        provenance, not a second model.
        """
        if not self._player_loaded:
            self._player_loaded = True
            if self._player_reference_path.exists():
                artifact = json.loads(self._player_reference_path.read_text())
                population: dict[str, dict[str, list[float]]] = {}
                counts: dict[str, int] = {}
                for row in artifact.get("rows") or []:
                    group = str(row.get("position_group") or "")
                    if not group:
                        continue
                    counts[group] = counts.get(group, 0) + 1
                    metrics = row.get("metrics") or {}
                    bucket = population.setdefault(group, {key: [] for key in PLAYER_KEYS})
                    for key in PLAYER_KEYS:
                        value = metrics.get(key)
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            bucket.setdefault(key, []).append(float(value))
                self._player_artifact = artifact
                self._player_population = population
                self._player_group_counts = counts
        return self._player_artifact

    def population_summary(self) -> dict[str, Any]:
        """The population description the UI shows next to every number."""
        artifact = self._load_reference()
        population = artifact.get("population") or {}
        model = artifact.get("model") or {}
        return {
            "label": population.get("label"),
            "n_matches": population.get("n_matches"),
            "n_team_matches": population.get("n_team_matches"),
            "n_teams": population.get("n_teams"),
            "n_shots": population.get("n_shots"),
            "competitions": population.get("competitions") or [],
            "seasons": population.get("seasons") or [],
            "date_from": population.get("date_from"),
            "date_to": population.get("date_to"),
            "source": population.get("source"),
            "model": model.get("model"),
            "model_features": model.get("features") or [],
            "model_excluded_features": model.get("excluded_features") or [],
            "n_fit_shots": model.get("n_fit_shots"),
            "definitions": artifact.get("definitions") or {},
            "style_availability": population.get("style_availability") or {},
            "model_card": artifact.get("model_card") or {},
        }

    # ── the user's match ────────────────────────────────────────────

    async def _fetch_events(self, match_id: int) -> list[dict[str, Any]]:
        """All events for a match, with parsed metadata."""
        if self._storage is None:
            return []
        out: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = await self._storage.get_match_events(match_id, limit=500, offset=offset)
            if not page:
                break
            for row in page:
                event = dict(row)
                if isinstance(event.get("metadata"), str):
                    try:
                        event["metadata"] = json.loads(event["metadata"])
                    except (json.JSONDecodeError, TypeError):
                        event["metadata"] = {}
                elif not isinstance(event.get("metadata"), dict):
                    event["metadata"] = {}
                out.append(event)
            if len(page) < 500:
                break
            offset += 500
        return out

    @staticmethod
    def _shot_features(metadata: dict[str, Any]) -> dict[str, Any] | None:
        """Benchmark features for a stored shot, or None if unmeasurable.

        The angle convention matters: Kawkab's xG model consumes the
        DEVIATION-from-central angle. Camera-derived matches store it
        under ``angle_to_goal_deg``/``angle_deg``; StatsBomb imports
        store the opening angle under ``angle_deg`` and the deviation
        angle under ``angle_deviation_deg`` — so the deviation key wins
        when both exist.
        """
        distance = metadata.get("distance_m")
        if distance is None:
            distance = metadata.get("distance_to_goal_m")
        angle = None
        for key in ("angle_deviation_deg", "angle_to_goal_deg", "angle_deg"):
            value = metadata.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                angle = float(value)
                break
        if distance is None or angle is None:
            return None
        return shot_features(
            distance_m=float(distance),
            angle_deg=angle,
            body_part=str(metadata.get("body_part") or "right_foot"),
            shot_type=str(metadata.get("shot_type") or "open_play"),
            was_pressed=bool(metadata.get("was_pressed", False)),
            is_one_on_one=bool(metadata.get("is_one_on_one", False)),
        )

    # ── style ───────────────────────────────────────────────────────

    #: Stored type -> the normalized type the style profile consumes.
    _STORED_STYLE_TYPES = {
        "pass": "pass",
        "carry": "carry",
        "interception": "interception",
        "duel": "duel",
        "ball_recovery": "recovery",
        "clearance": "clearance",
        "shot": "shot",
        "foul": "foul",
    }

    @staticmethod
    def _side_key(team: Any, match: dict[str, Any]) -> str | None:
        """Stored team value -> "home"/"away", or None when it cannot map."""
        value = str(team or "")
        if value in ("home", "away"):
            return value
        if value and value == str(match.get("home_team") or ""):
            return "home"
        if value and value == str(match.get("away_team") or ""):
            return "away"
        return None

    def _normalize_stored_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """One stored event -> the shared style event shape (Kawkab meters).

        Positions come from the typed columns when present and from the
        metadata JSON otherwise (``_merge_event_coords`` folds them
        there); ``completed`` is the pass-completion column, which the
        importer also uses for duel outcomes.
        """
        event_type = self._STORED_STYLE_TYPES.get(str(event.get("event_type") or ""))
        if event_type is None:
            return None
        metadata = event.get("metadata") or {}

        def coordinate(*names: str) -> float | None:
            for name in names:
                for source in (event, metadata):
                    value = source.get(name)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        return float(value)
            return None

        completed = event.get("completed")
        if isinstance(completed, int) and not isinstance(completed, bool):
            completed = bool(completed)
        if not isinstance(completed, bool):
            completed = None

        return {
            "type": event_type,
            "x": coordinate("start_x", "x"),
            "y": coordinate("start_y", "y"),
            "end_x": coordinate("end_x"),
            "end_y": coordinate("end_y"),
            "completed": completed if event_type == "pass" else None,
            "won": completed if event_type == "duel" else None,
            "from_track_id": event.get("from_track_id"),
            "is_goal": metadata.get("is_goal")
            if isinstance(metadata.get("is_goal"), bool)
            else None,
        }

    def _style_by_side(
        self, events: list[dict[str, Any]], match: dict[str, Any]
    ) -> tuple[dict[str, dict[str, Any]], str | None]:
        """Style profiles per side, or (empty, why they cannot be built).

        A side with unplaceable team labels would get a profile measured
        against the wrong opponent, which is worse than no profile, so a
        single unmapped event voids the style measurement for the match.
        """
        by_side: dict[str, list[dict[str, Any]]] = {"home": [], "away": []}
        unmapped = False
        for event in events:
            normalized = self._normalize_stored_event(event)
            if normalized is None:
                continue
            side = self._side_key(event.get("team"), match)
            if side is None:
                unmapped = True
                continue
            by_side[side].append(normalized)

        if unmapped:
            return {}, "this match's team labels do not map onto home/away sides"

        minutes = self._minutes(match)
        return {
            side: build_profile(
                by_side[side],
                by_side["away" if side == "home" else "home"],
                minutes,
            )
            for side in ("home", "away")
        }, None

    def _team_buckets(
        self, events: list[dict[str, Any]], match: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """Group shots and goals by the team value stored on the events."""
        buckets: dict[str, dict[str, Any]] = {}
        for event in events:
            team = str(event.get("team") or "unknown")
            bucket = buckets.setdefault(
                team,
                {
                    "shots": 0,
                    "measurable": 0,
                    "xg": 0.0,
                    "goal_events": 0,
                    "flagged_goals": 0,
                    "flagged_shots": 0,
                },
            )
            event_type = event.get("event_type")
            if event_type == "shot":
                bucket["shots"] += 1
                metadata = event.get("metadata") or {}
                features = self._shot_features(metadata)
                if features is not None:
                    bucket["measurable"] += 1
                    bucket["xg"] += shot_xg(self._model, features)
                # Vendor imports record the outcome on the shot itself
                # (StatsBomb: is_goal / shot_outcome); the app's own
                # coding records a separate goal event.
                if "is_goal" in metadata or "shot_outcome" in metadata:
                    bucket["flagged_shots"] += 1
                    if metadata.get("is_goal") is True or metadata.get("shot_outcome") == "Goal":
                        bucket["flagged_goals"] += 1
            elif event_type == "goal":
                bucket["goal_events"] += 1
        return buckets

    @staticmethod
    def _goals_for_side(
        bucket: dict[str, Any], side: str, match: dict[str, Any]
    ) -> tuple[int | None, str | None]:
        """Goals scored by one side, or (None, why it cannot be known).

        Priority: explicit goal events, then the outcome flag on every
        shot (a complete feed with outcomes knows the score exactly,
        including a genuine zero), then the stored score. Own goals are
        the reason the shot count alone is never used.
        """
        if bucket["goal_events"]:
            return bucket["goal_events"], None
        if bucket["flagged_shots"]:
            return bucket["flagged_goals"], None
        score_home, score_away = EliteBenchmarkService._goals_from_score(match)
        if side == "home" and score_home is not None:
            return score_home, None
        if side == "away" and score_away is not None:
            return score_away, None
        return None, "this match records no goals"

    @staticmethod
    def _side_and_label(team: str, match: dict[str, Any]) -> tuple[str, str]:
        """Map a stored team value onto a side and a display name."""
        if team == "home":
            return "home", str(match.get("home_team") or "Home")
        if team == "away":
            return "away", str(match.get("away_team") or "Away")
        return "team", team

    @staticmethod
    def _minutes(match: dict[str, Any]) -> float:
        """Match length in minutes (90 when the match has no duration)."""
        duration = match.get("duration_seconds")
        if isinstance(duration, (int, float)) and duration and duration > 0:
            return max(1.0, min(float(duration) / 60.0, 180.0))
        return 90.0

    @staticmethod
    def _goals_from_score(match: dict[str, Any]) -> tuple[int | None, int | None]:
        """Home/away goals from the stored score, when there is one."""
        home = match.get("score_home")
        away = match.get("score_away")
        if isinstance(home, (int, float)) and isinstance(away, (int, float)):
            return int(home), int(away)
        return None, None

    # ── player benchmark ────────────────────────────────────────────

    def _player_values(
        self, events: list[dict[str, Any]], minutes: float
    ) -> tuple[
        dict[str, float | None],
        dict[str, Any],
        float | None,
        int | None,
        str | None,
        str | None,
    ]:
        """One player's output + style values, with explicit refusal reasons.

        The style metrics reuse ``build_player_profile`` (the same
        predicates as the team profile); xG uses the same benchmark model
        as every other side of the comparison. A shot missing its position
        voids xG for the player instead of silently understating it.
        """
        normalized = [
            value
            for value in (self._normalize_stored_event(e) for e in events)
            if value is not None
        ]
        profile = build_player_profile(normalized, minutes)
        values: dict[str, float | None] = dict(profile["metrics"])

        shot_events = [e for e in events if e.get("event_type") == "shot"]
        features: list[dict[str, Any]] = []
        missing = 0
        goals_known = True
        goals = 0
        for event in shot_events:
            metadata = event.get("metadata") or {}
            feature = self._shot_features(metadata)
            if feature is None:
                missing += 1
            else:
                features.append(feature)
            if isinstance(metadata.get("is_goal"), bool):
                goals += 1 if metadata["is_goal"] else 0
            else:
                goals_known = False

        xg: float | None = None
        xg_reason: str | None = None
        if missing:
            xg_reason = (
                f"{missing} of {len(shot_events)} shots lack the position features the model needs"
            )
        else:
            xg = sum(shot_xg(self._model, feature) for feature in features)
        goals_value = goals if goals_known else None
        goals_reason = None if goals_known else "this match does not record shot outcomes"

        values["shots_per_90"] = per_90(len(shot_events), minutes)
        values["xg_per_90"] = per_90(xg, minutes) if xg is not None else None
        values["goals_per_90"] = per_90(goals_value, minutes) if goals_value is not None else None
        return values, profile, xg, goals_value, xg_reason, goals_reason

    @staticmethod
    def _player_population_payload(artifact: dict[str, Any]) -> dict[str, Any]:
        population = artifact.get("population") or {}
        return {
            "label": population.get("label"),
            "unit": population.get("unit"),
            "n_player_matches": population.get("n_player_matches"),
            "n_players": population.get("n_players"),
            "positions": population.get("positions") or {},
            "min_minutes": population.get("min_minutes", MIN_PLAYER_MINUTES),
            "source": population.get("source"),
        }

    async def _compare_players(
        self,
        events: list[dict[str, Any]],
        match: dict[str, Any],
        match_id: int,
    ) -> dict[str, Any]:
        """Rank every stored player against same-position cohort players."""
        artifact = self._load_player_reference()
        if artifact is None:
            return {
                "available": False,
                "reason": (
                    "the player reference population is not built — rebuild "
                    "core/elite_player_reference.json"
                ),
                "limitations": list(PLAYER_LIMITATIONS),
            }

        getter = getattr(self._storage, "get_match_players", None)
        if not callable(getter):
            return {
                "available": False,
                "reason": "storage does not expose player identity for this match",
                "limitations": list(PLAYER_LIMITATIONS),
            }
        try:
            players = await getter(match_id, limit=100)
        except Exception as exc:  # storage errors become an honest block
            logger.warning("player benchmark could not read players: %s", exc)
            return {
                "available": False,
                "reason": "player identity could not be read from storage",
                "limitations": list(PLAYER_LIMITATIONS),
            }
        if not players:
            return {
                "available": False,
                "reason": "this match has no stored player identities",
                "limitations": list(PLAYER_LIMITATIONS),
            }

        self._load_reference()  # the model must exist before xG is computed
        rows: list[dict[str, Any]] = []
        for raw_player in players:
            player = dict(raw_player)
            track_id = player.get("track_id")
            try:
                track_id = int(track_id) if track_id is not None else None
            except (TypeError, ValueError):
                track_id = None
            side = str(player.get("team") or "")
            position = str(player.get("position") or "")
            group = position_group(position)
            name = str(player.get("name") or "")
            minutes = float(player.get("minutes_played") or 0.0)
            team = (
                match.get("home_team")
                if side == "home"
                else match.get("away_team")
                if side == "away"
                else str(player.get("team") or "")
            )
            row: dict[str, Any] = {
                "name": name,
                "team": team,
                "side": side,
                "position": position,
                "position_group": group,
                "minutes": round(minutes, 1),
                "external_id": player.get("external_id") or "",
                "benchmarked": False,
                "reason": None,
                "population": None,
                "metrics": {},
                "xg": None,
                "goals": None,
            }

            if not name or track_id is None:
                row["reason"] = "no player identity is stored for this row"
            elif group is None:
                row["reason"] = (
                    f"position '{position or 'unknown'}' does not map to a cohort position group"
                )
            elif minutes < MIN_PLAYER_MINUTES:
                row["reason"] = (
                    f"played {minutes:g} minutes — a per-90 percentile needs "
                    f"at least {MIN_PLAYER_MINUTES:g}"
                )
            else:
                player_events = [e for e in events if e.get("from_track_id") == track_id]
                values, profile, xg, goals, xg_reason, goals_reason = self._player_values(
                    player_events, minutes
                )
                population = self._player_population.get(group) or {}
                comparisons = compare_values(
                    values,
                    population,
                    metrics=PLAYER_METRICS,
                    missing_reason="not measurable for this player",
                )
                for key in PLAYER_KEYS:
                    metric = comparisons[key]
                    if metric["value"] is not None:
                        continue
                    if key == "xg_per_90" and xg_reason:
                        metric["reason"] = xg_reason
                    elif key == "goals_per_90" and goals_reason:
                        metric["reason"] = goals_reason
                    elif key in (profile.get("coverage") or {}):
                        metric["reason"] = player_unavailable_reason(key, profile)

                group_count = self._player_group_counts.get(group, 0)
                row.update(
                    {
                        "benchmarked": True,
                        "xg": round(xg, 3) if xg is not None else None,
                        "goals": goals,
                        "metrics": comparisons,
                        "population": {
                            "label": (
                                "elite "
                                f"{POSITION_GROUP_LABELS.get(group, group)} "
                                f"(≥{MIN_PLAYER_MINUTES:g} min)"
                            ),
                            "unit": "player-match",
                            "position_group": group,
                            "n": group_count,
                            "min_minutes": MIN_PLAYER_MINUTES,
                        },
                    }
                )
            rows.append(row)

        rows.sort(
            key=lambda r: (
                not r["benchmarked"],
                -float(r.get("minutes") or 0.0),
                str(r.get("name") or ""),
            )
        )
        return {
            "available": True,
            "rows": rows,
            "n_benchmarked": sum(1 for r in rows if r["benchmarked"]),
            "population": self._player_population_payload(artifact),
            "definitions": artifact.get("definitions") or {},
            "limitations": list(PLAYER_LIMITATIONS),
        }

    async def compare_match(self, match_id: int) -> dict[str, Any]:
        """Benchmark one stored match against the elite cohort."""
        self._load_reference()
        if self._storage is None:
            return {
                "success": False,
                "data_available": False,
                "reason": "storage service unavailable",
            }

        match = await self._storage.get_match(match_id)
        if not match:
            return {
                "success": False,
                "data_available": False,
                "reason": "match not found",
            }

        events = await self._fetch_events(match_id)
        shots = [e for e in events if e.get("event_type") == "shot"]
        if not shots:
            return {
                "success": True,
                "data_available": False,
                "match_id": match_id,
                "reason": "no shots stored for this match — nothing to compare",
                "population": self.population_summary(),
                "limitations": list(LIMITATIONS),
            }

        minutes = self._minutes(match)
        buckets = self._team_buckets(events, match)
        style_by_side, style_note = self._style_by_side(events, match)

        teams: list[dict[str, Any]] = []
        for team, bucket in buckets.items():
            side, label = self._side_and_label(team, match)
            if bucket["shots"] == 0 and not bucket["goal_events"]:
                continue

            xg_value: float | None = None
            xg_reason: str | None = None
            if bucket["shots"] == 0:
                xg_reason = "no shots recorded for this team"
            elif bucket["measurable"] < bucket["shots"]:
                xg_reason = (
                    f"{bucket['shots'] - bucket['measurable']} of {bucket['shots']} shots "
                    "lack the position features the model needs"
                )
            else:
                xg_value = bucket["xg"]

            goals_value, goals_reason = self._goals_for_side(bucket, side, match)

            profile = style_by_side.get(side) if side in ("home", "away") else None
            style_values = (profile or {}).get("metrics") or {}

            user_values: dict[str, float | None] = {
                "shots": float(bucket["shots"]),
                "shots_per_90": per_90(bucket["shots"], minutes),
                "xg": xg_value,
                "xg_per_90": per_90(xg_value, minutes) if xg_value is not None else None,
                "goals": float(goals_value) if goals_value is not None else None,
                "goals_per_90": (per_90(goals_value, minutes) if goals_value is not None else None),
            }
            for key in STYLE_KEYS:
                user_values[key] = style_values.get(key)

            comparisons = compare_values(user_values, self._population)
            if xg_reason:
                comparisons["xg"]["reason"] = xg_reason
                comparisons["xg_per_90"]["reason"] = xg_reason
            if goals_reason:
                comparisons["goals"]["reason"] = goals_reason
                comparisons["goals_per_90"]["reason"] = goals_reason
            for key in STYLE_KEYS:
                if style_values.get(key) is None:
                    comparisons[key]["reason"] = (
                        unavailable_reason(key, profile)
                        if profile is not None
                        else (style_note or "this match has no positioned events for this team")
                    )

            teams.append(
                {
                    "side": side,
                    "team": label,
                    "minutes": minutes,
                    "shots": bucket["shots"],
                    "xg": round(xg_value, 3) if xg_value is not None else None,
                    "goals": goals_value,
                    "metrics": comparisons,
                    "style_coverage": (profile or {}).get("coverage"),
                }
            )

        teams.sort(key=lambda t: (t["side"] != "home", t["team"]))
        try:
            players = await self._compare_players(events, match, match_id)
        except Exception as exc:  # a player block must not break the team card
            logger.warning("player benchmark failed: %s", exc)
            players = {
                "available": False,
                "reason": "the player benchmark could not be computed for this match",
                "limitations": list(PLAYER_LIMITATIONS),
            }
        return {
            "success": True,
            "data_available": True,
            "match_id": match_id,
            "match_name": match.get("name"),
            "match_date": match.get("match_date"),
            "teams": teams,
            "players": players,
            "population": self.population_summary(),
            "limitations": list(LIMITATIONS),
        }

    # ── season block ─────────────────────────────────────────────────

    @staticmethod
    def _modal_home_team(matches: list[dict[str, Any]]) -> str | None:
        """The team the stored matches are mostly about (the analysed club).

        The season convention in this app is that the home side is the
        coach's team (see bridge_season_analytics). Naming the club here
        keeps a mixed match list from silently pooling opponents.
        """
        counts: dict[str, int] = {}
        for match in matches:
            name = str(match.get("home_team") or "")
            if name:
                counts[name] = counts.get(name, 0) + 1
        if not counts:
            return None
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def _season_unavailable(self, reason: str, **extra: Any) -> dict[str, Any]:
        """An honest refusal that still carries the population it refused."""
        return {
            "success": True,
            "data_available": False,
            "reason": reason,
            "population": self.population_summary(),
            **extra,
        }

    @staticmethod
    def _pool(rows: list[dict[str, Any]], key: str, minutes: float) -> float | None:
        """Pool a block metric: minutes-weighted per-90, plain mean for ratios."""
        values = [row["style"].get(key) for row in rows]
        if any(value is None for value in values):
            return None
        if key in _RATIO_METRICS:
            return sum(float(v) for v in values) / len(values)
        if minutes <= 0:
            return None
        return sum(float(v) * row["minutes"] for v, row in zip(values, rows, strict=True)) / minutes

    def _block_metric(
        self, key: str, value: float, block_size: int, draws: int, seed: int
    ) -> dict[str, Any]:
        population = self._population.get(key) or []
        if not population:
            return self._unavailable_metric(
                "reference population empty for this metric", population
            )
        result = block_percentile(value, population, block_size, draws=draws, seed=seed)
        distribution = result["block_distribution"]
        return {
            "value": value,
            "percentile": result["percentile"],
            "interval": percentile_interval(result["percentile"], len(population)),
            "band": result["band"],
            "population": {
                "n": len(population),
                "median": distribution.get("median"),
                "p25": distribution.get("p25"),
                "p75": distribution.get("p75"),
            },
            "block_distribution": distribution,
            "comparable": True,
            "reason": None,
        }

    def _unavailable_metric(
        self, reason: str, population: list[float] | None = None
    ) -> dict[str, Any]:
        return {
            "value": None,
            "percentile": None,
            "interval": None,
            "band": None,
            "population": {"n": len(population or [])},
            "comparable": False,
            "reason": reason,
        }

    async def compare_season(
        self,
        team: str | None = None,
        *,
        min_matches: int = 3,
        draws: int = 2000,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Benchmark a coach's N-match block against same-size elite blocks.

        The unit is the *block*: raw counts are never averaged, per-90
        metrics are pooled by minutes, ratios are averaged, and the
        percentile comes from :func:`block_percentile` against blocks of
        the same size. A block below ``min_matches`` is refused with the
        count rather than reported as a season.
        """
        self._load_reference()
        if self._storage is None:
            return self._season_unavailable("storage service unavailable")

        matches = await self._storage.get_all_matches()
        if not matches:
            return self._season_unavailable("no stored matches")
        if len(matches) < min_matches:
            return self._season_unavailable(
                f"a season block needs at least {min_matches} stored matches (found {len(matches)})"
            )

        team = team or self._modal_home_team(matches)
        if not team:
            return self._season_unavailable("stored matches carry no team names")
        selected = [match for match in matches if str(match.get("home_team") or "") == team]
        if len(selected) < min_matches:
            return self._season_unavailable(
                f"only {len(selected)} stored match(es) for {team} — a block needs "
                f"at least {min_matches}",
                team=team,
                n_matches_stored=len(selected),
            )

        rows: list[dict[str, Any]] = []
        excluded: list[int] = []
        for match in selected:
            match_id = int(match.get("id") or 0)
            events = await self._fetch_events(match_id)
            if not any(e.get("event_type") == "shot" for e in events):
                # A match with no shot events is a missing feed, not a
                # zero-shot performance.
                excluded.append(match_id)
                continue

            bucket = self._team_buckets(events, match).get("home") or {
                "shots": 0,
                "measurable": 0,
                "xg": 0.0,
                "goal_events": 0,
                "flagged_shots": 0,
                "flagged_goals": 0,
            }
            goals, _ = self._goals_for_side(bucket, "home", match)
            xg = bucket["xg"] if bucket["measurable"] == bucket["shots"] else None
            style_by_side, _ = self._style_by_side(events, match)
            profile = style_by_side.get("home")
            rows.append(
                {
                    "match_id": match_id,
                    "match_date": match.get("match_date"),
                    "minutes": self._minutes(match),
                    "shots": bucket["shots"],
                    "xg": xg,
                    "goals": goals,
                    "style": (profile or {}).get("metrics") or {},
                }
            )

        if len(rows) < min_matches:
            return self._season_unavailable(
                f"only {len(rows)} of {len(selected)} stored {team} matches carry "
                f"shot events — a block needs at least {min_matches}",
                team=team,
                n_matches_stored=len(selected),
                excluded_matches=excluded,
            )

        n = len(rows)
        minutes = sum(row["minutes"] for row in rows)
        metrics: dict[str, dict[str, Any]] = {}

        metric_values: dict[str, float | None] = {
            "shots_per_90": per_90(sum(row["shots"] for row in rows), minutes),
            "xg_per_90": (
                per_90(sum(row["xg"] for row in rows), minutes)
                if all(row["xg"] is not None for row in rows)
                else None
            ),
            "goals_per_90": (
                per_90(sum(row["goals"] for row in rows), minutes)
                if all(row["goals"] is not None for row in rows)
                else None
            ),
        }
        for key in STYLE_KEYS:
            metric_values[key] = self._pool(rows, key, minutes)

        reasons = {
            "xg_per_90": (
                "not every match in this block carries the shot positions the model needs"
            ),
            "goals_per_90": "not every match in this block records goals",
        }
        for key in SEASON_KEYS:
            value = metric_values.get(key)
            if value is None:
                metrics[key] = self._unavailable_metric(
                    reasons.get(
                        key,
                        "not every match in this block carries the events this metric needs",
                    ),
                    self._population.get(key),
                )
            else:
                metrics[key] = self._block_metric(key, float(value), n, draws, seed)

        trend: dict[str, Any] = {
            "available": False,
            "reason": "a trend needs at least 4 matches in the block",
        }
        if n >= 4:
            ordered = sorted(rows, key=lambda r: (str(r["match_date"] or ""), r["match_id"]))
            half = n // 2
            before_rows, after_rows = ordered[:half], ordered[half:]
            if all(row["xg"] is not None for row in ordered):
                before_minutes = sum(row["minutes"] for row in before_rows)
                after_minutes = sum(row["minutes"] for row in after_rows)
                before = per_90(sum(row["xg"] for row in before_rows), before_minutes)
                after = per_90(sum(row["xg"] for row in after_rows), after_minutes)
                trend = {
                    "available": True,
                    "metric": "xg_per_90",
                    "split": f"first {half} vs last {n - half} matches",
                    "before": round(before, 3),
                    "after": round(after, 3),
                    "delta": round(after - before, 3),
                    "note": "a direction to review, not a causal verdict",
                }
            else:
                trend = {
                    "available": False,
                    "reason": "the trend needs xG measured in every match of the block",
                }

        return {
            "success": True,
            "data_available": True,
            "team": team,
            "n_matches": n,
            "n_matches_stored": len(selected),
            "excluded_matches": excluded,
            "minutes": round(minutes, 1),
            "metrics": metrics,
            "trend": trend,
            "block": {
                "size": n,
                "draws": draws,
                "seed": seed,
                "method": "percentiles against same-size blocks drawn from the cohort",
            },
            "population": self.population_summary(),
            "limitations": list(SEASON_LIMITATIONS),
        }
