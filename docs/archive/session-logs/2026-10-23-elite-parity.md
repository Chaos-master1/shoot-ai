# 2026-10-23 — Competitor parity program (elite benchmark)

The product gap this pass closed: every number Kawkab produced was
unfalsifiable. "1.4 xG" said nothing about *compared with whom, measured
how, on how many matches*. Hudl/Wyscout/SkillCorner sell that context;
Kawkab had the models but not the population behind them.

Parity here means parity of **measurement**, **comparability** and
**trust** — not of data acquisition. Hudl sells optical tracking and
physical load; Kawkab reads broadcast video and stored events. That gap
is real and is not claimed to be closed.

Invariants this pass added to the codebase:

- the same model and the same metric definitions run on both sides of
  every comparison (`core/metric_contract.py` is the single owner of
  the definitions; core, builder, service and the JS card all read it);
- a percentile never travels without its population;
- an unmeasurable metric returns a reason, never a zero;
- the artifact carries its own provenance and rebuilds deterministically;
- the benchmark model publishes its own error.

## Pass 1 — Style benchmark

- `core/validation/statsbomb_loader.py` now exposes a normalized
  per-team event stream in Kawkab meters (pass, carry, pressure,
  interception, duel, recovery, clearance, shot).
- `core/style_profile.py` — one pure profiler computing PPDA, field
  tilt, progressive passes/90, final-third entries/90, high
  turnovers/90 and pass completion, with a coverage floor per metric and
  an explicit reason below it.
- `core/metric_contract.py` — key, label, unit, group, direction and
  definition for every metric; the artifact, service and UI consume it.
- `core/validation/build_elite_reference.py` + `core/elite_reference.json`
  (schema v2): 297 matches, 594 team-matches, 144 teams, 7,518 shots,
  every style metric present on every row.
- `services/elite_benchmark_service.py` measures the stored match with
  the same definitions and percentiles it per metric.
- Pro Analytics card splits Output and Style, each row carrying its own
  definition as a tooltip.

Two definition bugs were caught by looking at the measured
distributions before shipping: PPDA counted pressure events as
defensive actions (median 2.0 instead of ~8), and progressive passes
were counted from anywhere (median 104 instead of ~50). Both fixed and
asserted against published ranges.

The importer used to map `Ball Recovery` onto `interception`, which made
the stored stream disagree with the cohort on the same measurement.
Fixed at the importer; the recovery analyser already classifies
`ball_recovery` directly.

Parity was checked to the event: one corpus match hand-counted from the
raw StatsBomb JSON agreed exactly with the stored-event profile.

## Pass 2 — Season block

`compare_season` pools a coach's N-match block (per-90 pooled by
minutes, ratios averaged), ranks it with `block_percentile` against
N-match blocks drawn from the cohort, and reports a half-block xG trend.
The season report gained the `data_available` flag it never sent, which
had made every season view say "needs 2 stored matches" instead of
rendering the block.

## Pass 3 — Player benchmark: not shipped

Stored events carry no player reference, and cohort positions never
reach storage, so a position-matched player cohort cannot be built from
the data that exists. Pass 3 needs an identity column on events plus
player-position metadata in the artifact. It was not faked.

## Pass 4 — Uncertainty and own-error trust layer

- Every percentile now carries a 95% sampling band derived from the
  cohort size (`percentile_interval`), labelled in the UI as what it is:
  a band from the population size, not a range for the team.
- `build_elite_reference` computes a model card on the cohort it fits:
  Brier 0.078, log-loss 0.275, MAE vs StatsBomb xG 0.039, calibration
  by decile, team-match xG MAE. It ships inside the artifact and as
  `core/model_card.json` so the error can be cited without loading the
  population.
- The UI shows "Model check: Brier … on 7,518 shots" beside the
  population line, and the in-sample caveat is part of the limitations.

## Verification

- Real import → SQLite → Pro Analytics handler → real renderer (jsdom):
  Barcelona's imported match produced style percentiles with intervals
  and the model card; a 3-match season block produced all nine season
  metrics with intervals and honestly refused the trend (needs 4).
- Cohort distributions asserted against published ranges; artifact
  rebuild is byte-identical apart from `generated_at`.
- Full unit suite (5,555 collected) green; 169 Jest tests green; bundle
  rebuilt; ruff clean.

## Known gap left open

Player-level benchmarking (Pass 3) awaits event-level player identity.
