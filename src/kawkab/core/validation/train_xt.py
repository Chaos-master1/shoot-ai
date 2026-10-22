"""Train a reference xT (Expected Threat) grid from the StatsBomb corpus.

Pre-computes the 20x32 xT zone grid from all passes, carries, and shots
in the 297-match StatsBomb open-data corpus, using the exact same
``ExpectedThreatModel`` algorithm the product ships (transition matrix
over pass/carry outcomes + zone scoring probabilities, solved by power
iteration). The output ships as a JSON asset that
``ExpectedThreatModel`` auto-loads as its default grid, so:

    - Cold-start matches (few events) get a sensible league-wide prior
      instead of an all-zero grid.
    - Per-match xT still rebuilds from that match's own events when
      enough data exists (the model's existing behavior).

Direction handling: StatsBomb coordinates are per-possession-team
(each team attacks toward x=120 in its own frame). All actions are
normalized into a single attacking frame (toward x=105) before zone
indexing, so the grid is direction-correct.

Usage:
    PYTHONPATH=src python -m kawkab.core.validation.train_xt \
        --corpus data/statsbomb_corpus --force
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kawkab.core.validation.statsbomb_loader import (
    load_statsbomb_corpus,
    sb_to_meters,
)
from kawkab.core.xt_model import ExpectedThreatModel

KAWKAB_X_MAX = 105.0
KAWKAB_Y_MAX = 68.0


def statsbomb_events_to_kawkab(raw_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert SB Pass/Carry/Shot events into Kawkab event dicts.

    All coordinates normalized to a single attacking frame (toward
    x=105): SB x is already per-attacking-team, so a straight unit->m
    conversion yields direction-correct start/end zones for both teams.
    """
    out: list[dict[str, Any]] = []
    for ev in raw_events:
        type_name = ev.get("type", {}).get("name")
        loc = ev.get("location") or []
        if len(loc) < 2:
            continue
        sx, sy = sb_to_meters(loc)

        if type_name == "Pass":
            end_loc = (ev.get("pass") or {}).get("end_location") or []
            if len(end_loc) < 2:
                continue
            ex, ey = sb_to_meters(end_loc)
            outcome = (ev.get("pass") or {}).get("outcome")
            out.append(
                {
                    "type": "pass",
                    "start_x": sx,
                    "start_y": sy,
                    "end_x": ex,
                    "end_y": ey,
                    "completed": outcome is None,  # SB convention: no outcome = complete
                    "team": "home",  # direction handled by normalization
                }
            )
        elif type_name == "Carry":
            end_loc = (ev.get("carry") or {}).get("end_location") or []
            if len(end_loc) < 2:
                continue
            ex, ey = sb_to_meters(end_loc)
            out.append(
                {
                    "type": "carry",
                    "start_x": sx,
                    "start_y": sy,
                    "end_x": ex,
                    "end_y": ey,
                    "completed": True,
                    "team": "home",
                }
            )
        elif type_name == "Shot":
            shot = ev.get("shot") or {}
            out.append(
                {
                    "type": "shot",
                    "start_x": sx,
                    "start_y": sy,
                    "is_goal": shot.get("outcome", {}).get("name") == "Goal",
                    "team": "home",
                }
            )
    return out


def train_reference_grid(
    corpus_dir: str | Path, *, rows: int = 20, cols: int = 32
) -> dict[str, Any]:
    """Build the league-wide xT grid from every match in the corpus."""
    model = ExpectedThreatModel(rows=rows, cols=cols)
    all_events: list[dict[str, Any]] = []
    n_matches = 0
    for m in load_statsbomb_corpus(corpus_dir):
        raw_path = Path(corpus_dir) / f"{m.match_id}.json"
        with open(raw_path) as f:
            raw = json.load(f)
        events = statsbomb_events_to_kawkab(raw)
        if events:
            all_events.extend(events)
            n_matches += 1

    if not all_events:
        raise ValueError(f"no usable events found in {corpus_dir}")

    model.build_transition_matrix(all_events)
    grid = model.get_zone_grid()
    return {
        "grid": grid,
        "rows": rows,
        "cols": cols,
        "pitch_length": 105.0,
        "pitch_width": 68.0,
        "gamma": model.gamma,
        "n_matches": n_matches,
        "n_actions": len(all_events),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/statsbomb_corpus")
    parser.add_argument("--out", default="src/kawkab/core/trained_xt_grid.json")
    parser.add_argument("--report", default="docs/validation/xt_training_report.json")
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--cols", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        print(f"[xt-train] refusing to overwrite {out_path} (use --force)")
        return 1

    print(f"[xt-train] building reference grid from {args.corpus}")
    result = train_reference_grid(args.corpus, rows=args.rows, cols=args.cols)
    print(
        f"[xt-train] {result['n_matches']} matches, {result['n_actions']} actions "
        f"-> {result['rows']}x{result['cols']} grid"
    )

    payload = dict(result)
    payload["trained_at"] = datetime.now(UTC).isoformat()
    payload["source"] = "StatsBomb open data (CC BY-NC-SA 4.0)"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"[xt-train] grid -> {out_path}")

    # Sanity: monotonicity check — final-third columns must exceed own-half
    grid = result["grid"]
    n_cols = result["cols"]
    own_half = max(max(row[: n_cols // 2]) for row in grid)
    final_third = max(max(row[2 * n_cols // 3 :]) for row in grid)
    print(
        f"[xt-train] max own-half zone xT {own_half:.4f} < final-third {final_third:.4f}: {own_half < final_third}"
    )

    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    with open(rp, "w") as f:
        json.dump(
            {
                **payload,
                "sanity": {
                    "max_own_half": own_half,
                    "max_final_third": final_third,
                    "monotonic": own_half < final_third,
                },
            },
            f,
            indent=2,
        )
    print(f"[xt-train] report -> {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
