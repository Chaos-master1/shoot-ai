"""Elite reference population — the comparability layer.

Kawkab's numbers are absolute: a team scores 1.24 xG in a match. An
elite program's numbers are *contextual*: the same 1.24 is stated
against a defined population ("better than 74% of 594 elite
team-matches"). Without a population, a number cannot be judged, and
a percentile without a population is decoration.

This module owns the population maths and nothing else. The population
itself is a committed artifact (``elite_reference.json``) built offline
by ``core/validation/build_elite_reference.py`` from the bundled
StatsBomb elite cohort — 297 top-flight / international matches with
competition, season, date and team labels.

Comparability rules baked into this module (they are the reason this
is a module and not a formula in a view):

- the population is a fixed set of *team-matches* (one row per team
  per match), so a team that never shoots is a legitimate member of
  the distribution rather than a missing one;
- the percentile is a mid-rank (ties split the credit), so a match
  with zero shots is reported as "tied with the bottom group" instead
  of a flattering edge;
- every comparison carries the population it was made against
  (n, median, quartiles), because the research is unambiguous that a
  percentile without its population is not a benchmark.

Pure functions only — no file I/O, no service imports. Reading the
artifact and measuring the user's match live in
``services/elite_benchmark_service.py``.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

from kawkab.core.metric_contract import METRIC_KEYS as METRIC_KEYS
from kawkab.core.metric_contract import METRICS

# ── The metric contract ────────────────────────────────────────────────────
#
# The user's match is derived from broadcast video; the reference cohort
# is annotated event data. The two are NOT the same measurement, so the
# benchmark only compares them on the features a camera can produce.
#
# Used on both sides: shot distance, shooting angle (deviation from
# central), body part, shot type, pressure flag, one-on-one flag.
# Excluded on BOTH sides: goalkeeper proximity, rebound, big chance and
# volley — each of them is knowable only from freeze-frame or
# event annotation, and feeding them to one side only would tilt the
# comparison in a direction nobody could audit.
#
# The consequence is honest and worth stating: the benchmark's xG is a
# *reduced-feature* output of the same model the app uses, so its
# absolute value is not the app's headline xG. What the benchmark
# claims is the ranking against a population measured the same way.
BENCHMARK_FEATURES: tuple[str, ...] = (
    "distance_m",
    "angle_deg",
    "is_header",
    "is_free_kick",
    "is_penalty",
    "is_pressed",
    "is_one_on_one",
)

EXCLUDED_FEATURES: tuple[str, ...] = (
    "gk_distance_m",
    "is_rebound",
    "is_big_chance",
    "is_volley",
    "assist_type",
)

_VALID_BODY_PARTS = ("right_foot", "left_foot", "head", "other")
_VALID_SHOT_TYPES = ("open_play", "free_kick", "penalty")


def shot_features(
    *,
    distance_m: float,
    angle_deg: float,
    body_part: str = "right_foot",
    shot_type: str = "open_play",
    was_pressed: bool = False,
    is_one_on_one: bool = False,
) -> dict[str, Any]:
    """Restrict a shot to the benchmark's camera-observable features.

    ``angle_deg`` is the DEVIATION-from-central convention (0 = dead
    central) that the xG model is calibrated for. Shot types outside
    {open play, free kick, penalty} collapse to open play so that a
    camera-derived "volley" and an event-annotated "corner" do not
    silently enter the comparison as different features.
    """
    return {
        "type": "shot",
        "distance_m": float(distance_m),
        "angle_deg": float(angle_deg),
        "body_part": body_part if body_part in _VALID_BODY_PARTS else "right_foot",
        "shot_type": shot_type if shot_type in _VALID_SHOT_TYPES else "open_play",
        "was_pressed": bool(was_pressed),
        "is_one_on_one": bool(is_one_on_one),
    }


def shot_xg(model: Any, features: Mapping[str, Any]) -> float:
    """xG for a benchmark feature dict, using the benchmark model.

    The same model measures the reference cohort and the user's match —
    that identity is what makes the percentile meaningful.
    """
    return float(model.compute(dict(features)))


# Coefficients the xG model reads. A benchmark model that is missing
# one of them cannot be evaluated, so the loader refuses it loudly
# rather than letting a KeyError surface mid-comparison.
REQUIRED_COEFFICIENTS: tuple[str, ...] = (
    "intercept",
    "distance_m",
    "distance_m_sq",
    "angle_sin",
    "angle_deg_sq_sin",
    "is_header",
    "is_through_ball_assist",
    "is_cross_assist",
    "is_one_on_one",
    "is_pressed",
    "is_volley",
    "is_free_kick",
    "gk_distance_m",
    "gk_distance_m_sq",
    "is_rebound",
    "is_big_chance",
)


def build_model(coefficients: Mapping[str, float], source: str = "elite reference"):
    """Build the benchmark xG model from a coefficient mapping.

    The coefficients are fitted on the reference cohort itself (see
    ``core/validation/build_elite_reference.py``) with the same reduced
    feature set the comparison uses, so the numbers the benchmark
    prints are calibrated — not the app's headline xG with features
    silently dropped, which inflates every shot.
    """
    from kawkab.core.xg_model import EnhancedXgModel

    missing = [key for key in REQUIRED_COEFFICIENTS if key not in coefficients]
    if missing:
        raise ValueError(f"benchmark model is missing coefficients: {missing}")
    return EnhancedXgModel(coefficients=dict(coefficients), coeffs_source=source)


def per_90(value: float, minutes: float) -> float:
    """Scale a match total to a 90-minute rate."""
    if minutes <= 0:
        return 0.0
    return float(value) * 90.0 / float(minutes)


def is_finite_number(value: Any) -> bool:
    """True for real numbers only (NaN/inf are not measurements)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

# ``METRIC_KEYS`` is re-exported from ``metric_contract`` (one owner for
# the metric list, its order and its definitions). ``goals`` is in the
# population but is only comparable when the user's match actually
# records goals — otherwise the service reports it as unavailable rather
# than as zero.

# Band boundaries, highest first. ``percentile`` is "the share of the
# population this value is at least as good as", so a high percentile
# is a good result.
_BANDS: tuple[tuple[float, str], ...] = (
    (90.0, "top_decile"),
    (75.0, "top_quartile"),
    (50.0, "above_median"),
    (25.0, "lower_half"),
    (10.0, "bottom_quartile"),
)
_BOTTOM_BAND = "bottom_decile"


def percentile_rank(value: float, population: Sequence[float]) -> float:
    """Percent of ``population`` whose value is at or below ``value``.

    Mid-rank convention: strictly-lower values count fully, ties count
    half. That keeps a zero-shot match tied with every other zero-shot
    match in the cohort instead of being credited with the whole tie
    group (which would read as "top half" for having shot nothing).

    Returns 0.0-100.0. Raises ValueError on an empty population — a
    percentile without a population is not a number this module will
    invent.
    """
    if not population:
        raise ValueError("percentile_rank needs a non-empty population")
    lower = 0
    equal = 0
    for p in population:
        if p < value:
            lower += 1
        elif p == value:
            equal += 1
    return 100.0 * (lower + 0.5 * equal) / len(population)


def block_percentile(
    value: float,
    population: Sequence[float],
    block_size: int,
    *,
    draws: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Percentile of a multi-match block mean against same-size blocks.

    Ranking a six-match average against a distribution of SINGLE
    team-matches is not the same question: an average of six matches has
    far less spread than one match, so the naive comparison overstates
    how extreme the block is. This draws ``draws`` random blocks of
    ``block_size`` team-matches from the population and ranks the value
    against THEIR means — the like-for-like question ("your six matches
    against six-match blocks of elite football").

    Deterministic for a given seed: the same block can never produce two
    different percentiles in one session. ``block_size == 1`` delegates
    to :func:`percentile_rank` (no resampling).
    """
    values = [float(v) for v in population]
    if not values:
        raise ValueError("block_percentile needs a non-empty population")
    if block_size < 1:
        raise ValueError("block_size must be at least 1")

    if block_size == 1:
        percentile = percentile_rank(float(value), values)
        return {
            "percentile": percentile,
            "band": band_for(percentile),
            "block_distribution": {
                **population_stats(values),
                "n_draws": 0,
                "block_size": 1,
            },
            "drawn_with_replacement": False,
        }

    rng = random.Random(seed)
    n = len(values)
    with_replacement = block_size > n
    means: list[float] = []
    for _ in range(draws):
        sample = (
            rng.choices(values, k=block_size)
            if with_replacement
            else rng.sample(values, block_size)
        )
        means.append(sum(sample) / block_size)

    percentile = percentile_rank(float(value), means)
    return {
        "percentile": percentile,
        "band": band_for(percentile),
        "block_distribution": {
            **population_stats(means),
            "n_draws": draws,
            "block_size": block_size,
        },
        "drawn_with_replacement": with_replacement,
    }


def percentile_interval(
    percentile: float, sample_size: int, z: float = 1.96
) -> list[float] | None:
    """A 95% sampling band around a percentile, from its own share.

    A percentile is an estimate of the share of a population a value is
    at least as good as, so its sampling error is the binomial standard
    error of that share. This is what the band means: how precisely the
    cohort size pins the percentile down. It is NOT a confidence
    interval for the team's true ability — that needs more matches, and
    the season block is where that question is answered.
    """
    if sample_size <= 0:
        return None
    p = min(max(percentile / 100.0, 0.0), 1.0)
    standard_error = math.sqrt(p * (1.0 - p) / sample_size)
    return [
        round(100.0 * max(0.0, p - z * standard_error), 1),
        round(100.0 * min(1.0, p + z * standard_error), 1),
    ]


def band_for(percentile: float) -> str:
    """Map a percentile (0-100) to a stable band key."""
    for threshold, key in _BANDS:
        if percentile >= threshold:
            return key
    return _BOTTOM_BAND


def population_stats(values: Sequence[float]) -> dict[str, float | int]:
    """Median / quartiles / mean of a population, for display context."""
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    n = len(ordered)

    def _q(fraction: float) -> float:
        if n == 1:
            return ordered[0]
        pos = fraction * (n - 1)
        low = int(pos)
        high = min(low + 1, n - 1)
        weight = pos - low
        return ordered[low] * (1 - weight) + ordered[high] * weight

    return {
        "n": n,
        "median": _q(0.5),
        "p25": _q(0.25),
        "p75": _q(0.75),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / n,
    }


def compare_values(
    user_values: Mapping[str, float | None],
    population: Mapping[str, Sequence[float]],
) -> dict[str, dict[str, object]]:
    """Compare each metric against its population slice.

    ``user_values`` maps metric key to the user's value (or None when
    the match does not carry that metric). ``population`` maps metric
    key to the reference values for that metric. A metric is compared
    only when BOTH sides exist; otherwise the result carries an
    explicit reason instead of a fabricated zero.

    Direction comes from the contract: PPDA is the one metric where
    lower is better, so its percentile is computed on the inverted
    scale while the reported value and population stay on the real one.
    """
    out: dict[str, dict[str, object]] = {}
    for metric in METRICS:
        key = metric.key
        value = user_values.get(key)
        reference = population.get(key) or []
        if value is None:
            out[key] = {
                "value": None,
                "percentile": None,
                "interval": None,
                "band": None,
                "population": population_stats(reference),
                "comparable": False,
                "reason": "not available for this match",
            }
            continue
        if not reference:
            out[key] = {
                "value": value,
                "percentile": None,
                "interval": None,
                "band": None,
                "population": {"n": 0},
                "comparable": False,
                "reason": "reference population empty for this metric",
            }
            continue
        if metric.higher_is_better:
            percentile = percentile_rank(float(value), reference)
        else:
            percentile = percentile_rank(
                -float(value), [-float(v) for v in reference]
            )
        out[key] = {
            "value": value,
            "percentile": percentile,
            "interval": percentile_interval(percentile, len(reference)),
            "band": band_for(percentile),
            "population": population_stats(reference),
            "comparable": True,
            "reason": None,
        }
    return out
