"""Passing analysis mixin."""

from __future__ import annotations

from kawkab.core.pass_network import PassNetwork


class PassingMixin:
    def _compute_pass_network(self, events, player_teams: dict[int, str] | None = None):
        """Delegates to core.pass_network.PassNetwork rather than
        re-implementing pass-graph construction -- that version also has
        betweenness and power-iteration eigenvector centrality this one
        never grew. Output is a superset of the old shape: same
        {"nodes": [{"id"}], "edges": [{"source","target",...}]} structure,
        with "weight" renamed to "attempted"/"completed"/"completion_pct"
        per edge (nothing in this codebase reads "weight" downstream).

        Pre-filters to completed passes only, matching this method's
        original (deliberate, tested) semantics: an attempted-but-failed
        pass never reached its target, so it isn't a real connection
        between two players -- PassNetwork itself tracks attempted vs
        completed per edge for callers who want that, but a network of
        *all* attempts (including 0%-complete ones) isn't what "pass
        network" has conventionally meant here.
        """
        completed_events = [e for e in events if e.get("type") == "pass" and e.get("completed")]
        pn = PassNetwork(min_passes=0)
        pn.build(completed_events, player_teams)
        return pn.get_connection_matrix(team=None)

    def detect_line_breaking_passes(self, events, n_lines=3):
        plen = self.pitch_length
        line_breaks = []
        line_positions = [plen * i / (n_lines + 1) for i in range(1, n_lines + 1)]
        for event in events:
            if event.get("type") != "pass":
                continue
            if not event.get("completed", False):
                continue
            metadata = event.get("metadata", {})
            start_x = metadata.get("start_x_pct", 0.5) * plen
            end_x = metadata.get("end_x_pct", 0.6) * plen
            if end_x <= start_x:
                continue
            lines_crossed = 0
            for line_x in line_positions:
                if start_x < line_x <= end_x:
                    lines_crossed += 1
            if lines_crossed >= 2:
                line_breaks.append(
                    {
                        "team": event.get("team", "home"),
                        "player_track_id": event.get("player_track_id"),
                        "start_x_pct": round(metadata.get("start_x_pct", 0.5), 3),
                        "end_x_pct": round(metadata.get("end_x_pct", 0.6), 3),
                        "lines_crossed": lines_crossed,
                        "vertical_gain_pct": round(end_x / plen - start_x / plen, 3),
                    }
                )
        return line_breaks

    def attribute_possession_robust(self, events, frames=None):
        last_known: dict[str, int | None] = {"home": None, "away": None}
        attributed = []
        for event in events:
            team = event.get("team", "home")
            track_id = event.get("player_track_id")
            if track_id is not None:
                last_known[team] = track_id
                attributed.append({**event, "attribution_source": "explicit"})
                continue
            inferred = last_known.get(team)
            if inferred is not None:
                attributed.append(
                    {
                        **event,
                        "player_track_id": inferred,
                        "attribution_source": "last_known",
                    }
                )
            else:
                attributed.append(
                    {
                        **event,
                        "player_track_id": -1,
                        "attribution_source": "unknown",
                    }
                )
        return attributed
