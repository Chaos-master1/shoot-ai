"""Pre-match scouting report generator.

Generates tactical scouting reports about upcoming opponents from
historical match data. Combines StatsBomb / OpenFootball / API-Football
sources to produce:

- Formation preferences and formation changes
- Set-piece tendencies (corners, FKs, throw-ins)
- Top players and their threat rating
- Pressing intensity and PPDA
- Build-up patterns and width usage
- Vulnerability flags (e.g. high xG conceded from crosses)

Designed to be a thin orchestrator — the heavy lifting is done by
existing analysis services (psychology, possession, setpiece).
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class OpponentProfile:
    """Aggregated opponent profile for a scout report."""

    team_name: str
    matches_analyzed: int
    preferred_formation: str
    formation_changes: int
    avg_possession_pct: float
    avg_ppda: float
    pressing_intensity: str
    set_piece_threat: float
    set_piece_vulnerability: float
    width_usage: float
    build_up_style: str
    top_scorers: list[tuple[str, int]]
    top_assisters: list[tuple[str, int]]
    top_xg_contributors: list[tuple[str, float]]
    vulnerability_flags: list[str]
    strength_flags: list[str]
    recommended_tactics: list[str]


class ScoutingService:
    """Generate pre-match scouting reports.

    Args:
        min_matches: Minimum matches required to produce a report.
    """

    def __init__(self, min_matches: int = 3) -> None:
        self.min_matches = min_matches
        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    def analyze(
        self,
        team_name: str,
        matches: list[dict[str, Any]],
    ) -> OpponentProfile:
        """Build scouting profile from historical match data.

        Args:
            team_name: Name of the opponent.
            matches: List of match dicts, each with keys:
              - formation: str
              - possession_pct: float
              - ppda: float
              - set_piece_threat: float
              - set_piece_conceded: float
              - width_usage: float
              - build_up_style: str
              - scorers: list[dict with player/goals]
              - assisters: list[dict with player/assists]
              - xg_contributors: list[dict with player/xg]
        """
        if len(matches) < self.min_matches:
            return self._empty_profile(team_name, len(matches))
        formations = [m.get("formation", "unknown") for m in matches]
        formation_counts: dict[str, int] = {}
        for f in formations:
            formation_counts[f] = formation_counts.get(f, 0) + 1
        preferred = max(formation_counts.items(), key=lambda kv: kv[1])[0]
        formation_changes = sum(
            1 for a, b in zip(formations, formations[1:], strict=False) if a != b
        )
        poss = [m.get("possession_pct", 50.0) for m in matches]
        ppdas = [m.get("ppda", 10.0) for m in matches]
        sp_threats = [m.get("set_piece_threat", 0.0) for m in matches]
        sp_conceded = [m.get("set_piece_conceded", 0.0) for m in matches]
        widths = [m.get("width_usage", 0.5) for m in matches]
        build_styles = [m.get("build_up_style", "mixed") for m in matches]
        avg_poss = statistics.mean(poss)
        avg_ppda = statistics.mean(ppdas)
        avg_sp_threat = statistics.mean(sp_threats)
        avg_sp_conc = statistics.mean(sp_conceded)
        avg_width = statistics.mean(widths)
        press = self._classify_press(avg_ppda)
        build_style = self._classify_build(build_styles)
        scorers = self._aggregate_player_stats(matches, "scorers", "goals")
        assisters = self._aggregate_player_stats(matches, "assisters", "assists")
        xg_players = self._aggregate_player_stats(matches, "xg_contributors", "xg")
        vul_flags = self._build_vulnerabilities(avg_ppda, avg_sp_conc, avg_width, build_style)
        str_flags = self._build_strengths(avg_poss, avg_sp_threat, formation_changes)
        recs = self._recommend_tactics(press, avg_sp_conc, build_style, formation_changes)
        return OpponentProfile(
            team_name=team_name,
            matches_analyzed=len(matches),
            preferred_formation=preferred,
            formation_changes=formation_changes,
            avg_possession_pct=round(avg_poss, 1),
            avg_ppda=round(avg_ppda, 2),
            pressing_intensity=press,
            set_piece_threat=round(avg_sp_threat, 3),
            set_piece_vulnerability=round(avg_sp_conc, 3),
            width_usage=round(avg_width, 3),
            build_up_style=build_style,
            top_scorers=[(n, int(v)) for n, v in scorers[:5]],
            top_assisters=[(n, int(v)) for n, v in assisters[:5]],
            top_xg_contributors=xg_players[:5],
            vulnerability_flags=vul_flags,
            strength_flags=str_flags,
            recommended_tactics=recs,
        )

    def _empty_profile(self, team_name: str, n: int) -> OpponentProfile:
        return OpponentProfile(
            team_name=team_name,
            matches_analyzed=n,
            preferred_formation="unknown",
            formation_changes=0,
            avg_possession_pct=50.0,
            avg_ppda=10.0,
            pressing_intensity="unknown",
            set_piece_threat=0.0,
            set_piece_vulnerability=0.0,
            width_usage=0.5,
            build_up_style="unknown",
            top_scorers=[],
            top_assisters=[],
            top_xg_contributors=[],
            vulnerability_flags=[],
            strength_flags=[],
            recommended_tactics=[f"Need at least {self.min_matches} matches, have {n}."],
        )

    @staticmethod
    def _classify_press(ppda: float) -> str:
        if ppda < 8:
            return "high"
        if ppda < 13:
            return "medium"
        return "low"

    @staticmethod
    def _classify_build(styles: list[str]) -> str:
        if not styles:
            return "unknown"
        counts: dict[str, int] = {}
        for s in styles:
            counts[s] = counts.get(s, 0) + 1
        return max(counts.items(), key=lambda kv: kv[1])[0]

    @staticmethod
    def _aggregate_player_stats(
        matches: list[dict[str, Any]], key: str, count_key: str
    ) -> list[tuple[str, float]]:
        totals: dict[str, float] = {}
        for m in matches:
            for entry in m.get(key, []) or []:
                player = entry.get("player", "unknown")
                value = float(entry.get(count_key, 0))
                totals[player] = totals.get(player, 0.0) + value
        return sorted(totals.items(), key=lambda kv: -kv[1])

    def _build_vulnerabilities(
        self,
        ppda: float,
        sp_conc: float,
        width: float,
        build: str,
    ) -> list[str]:
        flags: list[str] = []
        if ppda > 13:
            flags.append("Low press — vulnerable to quick combinations through midfield")
        if sp_conc > 0.2:
            flags.append("Concedes from set pieces — improve aerial defending")
        if width > 0.7:
            flags.append("Plays very wide — exploit central channel on counter-attacks")
        if build == "short":
            flags.append("Short build-up — high press can force turnovers")
        return flags

    def _build_strengths(
        self,
        poss: float,
        sp_threat: float,
        formation_changes: int,
    ) -> list[str]:
        flags: list[str] = []
        if poss > 60:
            flags.append("Dominant possession — prepare for long defensive phases")
        if sp_threat > 0.25:
            flags.append("Dangerous from set pieces — guard the box on dead balls")
        if formation_changes > 3:
            flags.append("Tactically flexible — pre-match plan must cover multiple shapes")
        return flags

    def _recommend_tactics(
        self,
        press: str,
        sp_conc: float,
        build: str,
        formation_changes: int,
    ) -> list[str]:
        recs: list[str] = []
        if press == "low":
            recs.append("Press high and force turnovers in opponent half")
        if sp_conc > 0.2:
            recs.append("Attack set pieces — opponent is aerially vulnerable")
        if build == "short":
            recs.append("Block passing lanes to the center-backs")
        if formation_changes > 3:
            recs.append("Prepare tactical plan for at least 2 opponent formations")
        if not recs:
            recs.append("Standard preparation: balanced pressing and possession")
        return recs
