# Analyst Guide — the elite workflow

This guide walks the daily loop a performance analyst runs in Kawkab:
**get data in → check its quality → trust the models → read the report.**
Every number you can show the coaching staff has a published validation
proof or a model card behind it — this guide shows where those live.

---

## 1. Get match data in

Kawkab ingests three kinds of data, and each has a purpose-built path:

| Data | Source examples | Where to import |
|---|---|---|
| Event data | StatsBomb open data (JSON), Opta F24, Wyscout | `kawkab import` (CLI), Data Providers panel (desktop) |
| Tracking data | Metrica (CSV), SkillCorner, EPTS XML | Data Providers → Vendor Import panel (desktop) |
| Video | Broadcast or tactical footage | Upload tab → YOLO + BoT-SORT tracking pipeline |

### One match (event data)

```bash
# StatsBomb open event file (or Opta F24 / Wyscout XML):
kawkab import path/to/match.json --competition "Premier League" --season-id 2025
```

The importer registers the file's vendor external IDs, so re-importing the
same file (or a vendor's updated revision of it) **deduplicates instead of
creating a second copy of the match**. Check the command output — it prints
`imported` vs `deduplicated` per match.

### A whole season

Point `import` at a directory instead of a file and it becomes a season
import: every vendor file under the directory (recursively) is imported,
tagged with the competition and season you name, and deduplicated against
what's already in the database.

```bash
kawkab import ~/data/statsbomb/open-data/data/events/ \
    --competition "Champions League" --season-id 2025
```

Track progress in the desktop app during long imports: the Vendor Import
panel shows a live per-file progress signal (imported / skipped / failed),
and the same results table appears when the run completes.

### kloppy-backed imports (desktop UI)

**Data Providers → Vendor Import** in the desktop app also exposes the
kloppy library path (available only when `kloppy` is installed — the
Vendor Import panel's availability probe reports the per-provider state
honestly, never a fabricated middle):

- **Import Events via kloppy** — a StatsBomb events JSON through
  kloppy's deserializer. Without a lineup file, lineups are synthesized
  from the events' own Starting XI / Substitution rows. Shots carry the
  vendor's xG labeled `vendor_xg` (source `kloppy` on every event row);
  Kawkab's own xG model is NOT re-run over kloppy imports.
- **Import Tracking via kloppy** — SkillCorner meta + raw JSON/JSONL
  through kloppy, normalized to Kawkab meters and persisted through the
  same frame pipeline (quality report, provenance row) as the file
  parsers.

Imported matches feed the opposition dossier and the training-plan
engine exactly like video- or CLI-imported matches.

### Tracking data (desktop UI)

**Data Providers → Vendor Import** in the desktop app:

1. Pick the match's tracking file(s) — Metrica CSV, SkillCorner JSON,
   or EPTS XML.
2. Select the match the frames belong to (or let the importer create it).
3. Review the **data-quality report** it produces (next section) before
   trusting anything downstream.

---

## 2. Read the data-quality report

Every tracking import produces a quality assessment stored alongside the
data. **Read it before you read any analytics.** It reports:

- **Frame completeness** — % of expected frames actually present; gaps mean
  dropped packets or vendor processing artifacts.
- **Duplicate timestamps** — two frames claiming the same instant; either
  the feed doubled a frame or your clock alignment is off.
- **Player-count anomalies** — frames with more/fewer than 22 detected
  players; sustained anomalies usually mean a substitution the roster
  doesn't know about.
- **Out-of-pitch coordinates** — raw vendor feeds occasionally carry
  coordinates slightly outside the pitch bounds (real noise, not errors);
  the report quantifies how much there is so downstream models can be
  read with that in mind.

Garbage in, garbage out is the #1 way analysts lose a room. If the quality
report looks wrong, fix the import before presenting anything computed
from it.

---

## 3. Trust the models (validation report)

`kawkab validate` runs the full validation suite against the StatsBomb
open-data corpus and writes `docs/validation/validation_report.json` (and
a readable `.md`). The report covers, per model:

**Event models** (validated against real outcomes):

- **xG** — Brier score, ROC-AUC, MAE vs StatsBomb's own xG
- **PSxG** — post-shot calibration
- **xT / carry-xT** — learned from the corpus; sanity-checked for
  plausible gradients toward goal

**Tracking models** (validated on the committed Metrica open tracking
fixture — real vendor data, not synthetic):

- **Pitch control** — Voronoi control shares sanity bands
- **Ball recovery** — recovery counts by pitch zone on real event+
  tracking data
- **Carry-xT** — positive value gradient for the team that actually
  scored in the window
- **Physical load** — average speed and load metrics computed at
  GPS-equivalent sampling, sanity-banded against published elite
  match averages

Every model also has a **model card** (see `docs/model-cards.md`) stating
its inputs, assumptions, and known limitations. Cite the card when someone
asks "how is this number computed?" — it's part of the product, not an
afterthought.

---

## 4. The daily loop, end to end

```bash
# 1. Import this week's matches
kawkab import ./week-12/events/ --competition "Premier League" --season-id 2025

# 2. Build/refresh the validation report (cron-able; runs offline)
kawkab validate

# 3. Skim it for regressions before you present
cat docs/validation/validation_report.md
```

Then in the desktop app: open the match → timeline, possession, pass
network, pitch control, physical load, and the set-piece planner are all
computed from the imported data. Anything you annotate or rate feeds back
into the database (and the audit trail — see the deployment guide if your
club requires full data-provenance records).

---

## 5. Where things live (quick reference)

| Thing | Where |
|---|---|
| Import CLI | `kawkab import --help` |
| Validation CLI | `kawkab validate --help` |
| Benchmarks | `kawkab benchmark --module <name>` |
| Validation report | `docs/validation/validation_report.md` |
| Model cards | `docs/model-cards.md` |
| Tracking fixtures + provenance | `tests/fixtures/tracking/README.md` |
| Deployment / IT guide | `docs/elite-deployment.md` |
