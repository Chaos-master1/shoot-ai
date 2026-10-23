"""The metric contract — one owner for every benchmarked measurement.

Before this module the same metric existed in four hand-kept copies: the
core key tuple, the builder's definition dict, the service's value dict
and the JS render order. Adding one metric meant touching all four and
hoping they agreed. Now they read from here.

A metric is only comparable when BOTH sides use this definition. That is
why the definition text lives beside the key rather than in a doc: the
UI shows it next to the number, and the artifact records it, so a
percentile can never be read as something other than what was measured.

Pure data. No I/O, no imports from the services.
"""

from __future__ import annotations

from dataclasses import dataclass

# Groups the UI renders as sections. "output" is what a team produced;
# "style" is how it played while producing it.
OUTPUT = "output"
STYLE = "style"


@dataclass(frozen=True)
class MetricSpec:
    """One measured quantity, with the definition both sides obey."""

    key: str
    group: str
    label: str
    unit: str
    digits: int
    higher_is_better: bool
    definition: str

    @property
    def per_90(self) -> bool:
        return self.key.endswith("_per_90")


METRICS: tuple[MetricSpec, ...] = (
    # ── output: what the team produced ────────────────────────────────
    MetricSpec(
        key="shots",
        group=OUTPUT,
        label="Shots",
        unit="count",
        digits=0,
        higher_is_better=True,
        definition="Shots recorded for the team in the match.",
    ),
    MetricSpec(
        key="shots_per_90",
        group=OUTPUT,
        label="Shots / 90",
        unit="per 90",
        digits=1,
        higher_is_better=True,
        definition="Shots scaled to 90 minutes.",
    ),
    MetricSpec(
        key="xg",
        group=OUTPUT,
        label="xG",
        unit="expected goals",
        digits=2,
        higher_is_better=True,
        definition=(
            "Sum of per-shot xG from the benchmark model (camera-observable features only)."
        ),
    ),
    MetricSpec(
        key="xg_per_90",
        group=OUTPUT,
        label="xG / 90",
        unit="per 90",
        digits=2,
        higher_is_better=True,
        definition="Benchmark xG scaled to 90 minutes.",
    ),
    MetricSpec(
        key="goals_per_90",
        group=OUTPUT,
        label="Goals / 90",
        unit="per 90",
        digits=2,
        higher_is_better=True,
        definition="Goals scaled to 90 minutes.",
    ),
    MetricSpec(
        key="goals",
        group=OUTPUT,
        label="Goals",
        unit="count",
        digits=0,
        higher_is_better=True,
        definition="Goals scored by the team in the match.",
    ),
    # ── style: how it played ──────────────────────────────────────────
    MetricSpec(
        key="ppda",
        group=STYLE,
        label="PPDA",
        unit="passes per defensive action",
        digits=2,
        higher_is_better=False,
        definition=(
            "Opponent passes allowed in their own 60% divided by our duels, "
            "interceptions and fouls there (pressure events excluded). Lower "
            "is more aggressive pressing."
        ),
    ),
    MetricSpec(
        key="field_tilt",
        group=STYLE,
        label="Field tilt",
        unit="share",
        digits=3,
        higher_is_better=True,
        definition=(
            "Share of both teams' completed final-third passes that are "
            "ours — an event-based proxy for territory."
        ),
    ),
    MetricSpec(
        key="progressive_passes_per_90",
        group=STYLE,
        label="Progressive passes / 90",
        unit="per 90",
        digits=1,
        higher_is_better=True,
        definition=(
            "Completed passes from the halfway line onward that advance the "
            "ball 10 m or more toward goal, plus any pass into the penalty "
            "area, per 90 minutes."
        ),
    ),
    MetricSpec(
        key="final_third_entries_per_90",
        group=STYLE,
        label="Final-third entries / 90",
        unit="per 90",
        digits=1,
        higher_is_better=True,
        definition=(
            "Completed passes and carries that start outside the final "
            "third and end inside it, per 90 minutes."
        ),
    ),
    MetricSpec(
        key="high_turnovers_per_90",
        group=STYLE,
        label="High turnovers / 90",
        unit="per 90",
        digits=1,
        higher_is_better=True,
        definition=(
            "Possessions won back within 40 m of the opponent's goal "
            "(interceptions, recoveries, won duels), per 90 minutes."
        ),
    ),
    MetricSpec(
        key="pass_completion_pct",
        group=STYLE,
        label="Pass completion",
        unit="%",
        digits=1,
        higher_is_better=True,
        definition="Completed passes divided by attempted passes.",
    ),
)

METRIC_KEYS: tuple[str, ...] = tuple(m.key for m in METRICS)
OUTPUT_KEYS: tuple[str, ...] = tuple(m.key for m in METRICS if m.group == OUTPUT)
STYLE_KEYS: tuple[str, ...] = tuple(m.key for m in METRICS if m.group == STYLE)

#: Metrics a season block may average. Raw counts are excluded: averaging
#: shots or goals across matches measures the fixture list, not the team —
#: the per-90 form and the ratios are what carry over.
SEASON_KEYS: tuple[str, ...] = tuple(
    key for key in METRIC_KEYS if key not in ("shots", "xg", "goals")
)
_BY_KEY = {m.key: m for m in METRICS}


def spec(key: str) -> MetricSpec:
    """The contract for one metric key (KeyError if it is not measured)."""
    return _BY_KEY[key]


def definitions() -> dict[str, dict[str, object]]:
    """The artifact's provenance block: every metric and what it means."""
    return {
        m.key: {
            "group": m.group,
            "label": m.label,
            "unit": m.unit,
            "higher_is_better": m.higher_is_better,
            "definition": m.definition,
        }
        for m in METRICS
    }
