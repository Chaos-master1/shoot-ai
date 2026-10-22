"""Opponent Dossier — automated pre-match scouting report.

Combines formation detection, key player analysis, set piece
tendencies, predicted lineup, and scoreline prediction into
a single structured dossier.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FormationTendency:
    formation: str = ""
    count: int = 0
    percentage: float = 0.0
    context: str = ""  # "all", "possession", "defensive"


@dataclass
class KeyPlayer:
    name: str = ""
    position: str = ""
    goals: int = 0
    assists: int = 0
    xg: float = 0.0
    threat_score: float = 0.0
    key_stat: str = ""


@dataclass
class ScorelinePrediction:
    home_score: int = 0
    away_score: int = 0
    home_win_prob: float = 0.0
    draw_prob: float = 0.0
    away_win_prob: float = 0.0
    total_goals_avg: float = 0.0
    both_teams_score_prob: float = 0.0


@dataclass
class OpponentDossier:
    opponent_name: str = ""
    matches_analyzed: int = 0
    formations: list[FormationTendency] = field(default_factory=list)
    predicted_lineup: list[str] = field(default_factory=list)
    key_players: list[KeyPlayer] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    set_piece_tendencies: list[str] = field(default_factory=list)
    build_up_pattern: str = ""
    pressing_style: str = ""
    scoreline_prediction: ScorelinePrediction = field(default_factory=ScorelinePrediction)
    recommended_tactics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "opponent_name": self.opponent_name,
            "matches_analyzed": self.matches_analyzed,
            "formations": [
                {
                    "formation": f.formation,
                    "count": f.count,
                    "percentage": round(f.percentage, 1),
                    "context": f.context,
                }
                for f in self.formations
            ],
            "predicted_lineup": self.predicted_lineup,
            "key_players": [
                {
                    "name": p.name,
                    "position": p.position,
                    "goals": p.goals,
                    "assists": p.assists,
                    "xg": round(p.xg, 2),
                    "threat_score": round(p.threat_score, 2),
                    "key_stat": p.key_stat,
                }
                for p in self.key_players
            ],
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "set_piece_tendencies": self.set_piece_tendencies,
            "build_up_pattern": self.build_up_pattern,
            "pressing_style": self.pressing_style,
            "scoreline_prediction": {
                "home_score": self.scoreline_prediction.home_score,
                "away_score": self.scoreline_prediction.away_score,
                "home_win_prob": round(self.scoreline_prediction.home_win_prob, 2),
                "draw_prob": round(self.scoreline_prediction.draw_prob, 2),
                "away_win_prob": round(self.scoreline_prediction.away_win_prob, 2),
                "total_goals_avg": round(self.scoreline_prediction.total_goals_avg, 2),
                "both_teams_score_prob": round(self.scoreline_prediction.both_teams_score_prob, 2),
            },
            "recommended_tactics": self.recommended_tactics,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Opponent Dossier: {self.opponent_name}",
            f"*Matches analyzed: {self.matches_analyzed}*",
            "",
        ]
        if self.formations:
            lines.append("## Formation Tendencies")
            for f in self.formations:
                lines.append(f"- **{f.formation}**: {f.count}x ({f.percentage:.0f}%) — {f.context}")
            lines.append("")
        if self.predicted_lineup:
            lines.append("## Predicted Lineup")
            for i, lineup_entry in enumerate(self.predicted_lineup, 1):
                lines.append(f"{i}. {lineup_entry}")
            lines.append("")
        if self.key_players:
            lines.append("## Key Players")
            for p in self.key_players:
                desc = (
                    f"⚽ {p.goals}g / 🅰 {p.assists}a / xG {p.xg:.2f}"
                    if p.goals or p.assists
                    else f"xG {p.xg:.2f}"
                )
                lines.append(f"- **{p.name}** ({p.position}) — {desc}")
                if p.key_stat:
                    lines[-1] += f" — {p.key_stat}"
            lines.append("")
        if self.strengths:
            lines.append("## Strengths")
            for s in self.strengths:
                lines.append(f"- ✅ {s}")
            lines.append("")
        if self.weaknesses:
            lines.append("## Weaknesses")
            for w in self.weaknesses:
                lines.append(f"- ⚠️ {w}")
            lines.append("")
        if self.set_piece_tendencies:
            lines.append("## Set Piece Tendencies")
            for s in self.set_piece_tendencies:
                lines.append(f"- {s}")
            lines.append("")
        lines.append("## Build-Up & Pressing")
        lines.append(f"- Build-up: {self.build_up_pattern}")
        lines.append(f"- Pressing: {self.pressing_style}")
        lines.append("")
        sp = self.scoreline_prediction
        lines.append("## Scoreline Prediction")
        lines.append(f"- Predicted: {sp.home_score}-{sp.away_score}")
        lines.append(
            f"- Home win: {sp.home_win_prob:.0%} | Draw: {sp.draw_prob:.0%} | Away win: {sp.away_win_prob:.0%}"
        )
        lines.append(f"- Avg total goals: {sp.total_goals_avg:.2f}")
        lines.append(f"- Both teams score: {sp.both_teams_score_prob:.0%}")
        lines.append("")
        if self.recommended_tactics:
            lines.append("## Recommended Tactics")
            for r in self.recommended_tactics:
                lines.append(f"- {r}")
        return "\n".join(lines)


def _detect_formations(matches: list[dict[str, Any]]) -> list[FormationTendency]:
    form_counts: Counter = Counter()
    for m in matches:
        f = m.get("formation", "unknown")
        if f != "unknown":
            form_counts[f] += 1
    total = sum(form_counts.values())
    if total == 0:
        return [FormationTendency(formation="unknown", count=0, percentage=0, context="all")]
    result = []
    for form, count in form_counts.most_common():
        result.append(
            FormationTendency(
                formation=form,
                count=count,
                percentage=count / total * 100,
                context="all",
            )
        )
    return result


def _predict_lineup(matches: list[dict[str, Any]]) -> list[str]:
    player_minutes: dict[str, float] = {}
    player_position: dict[str, str] = {}
    for m in matches:
        for p in m.get("players", []):
            name = p.get("name", "")
            mins = p.get("minutes_played", 0)
            pos = p.get("position", "sub")
            if name:
                player_minutes[name] = player_minutes.get(name, 0) + mins
                if pos != "sub" and name not in player_position:
                    player_position[name] = pos
    lineup_order = ["GK", "LB", "CB", "CB", "RB", "CDM", "CM", "CM", "LW", "RW", "ST"]
    selected: dict[str, str] = {}
    for pos in lineup_order:
        best = ""
        best_mins = 0.0
        for name, mins in player_minutes.items():
            if name in selected.values():
                continue
            ppos = player_position.get(name, "")
            if (ppos == pos or not ppos) and mins > best_mins:
                best = name
                best_mins = mins
        if best:
            selected[pos] = best
    result = [f"{pos}: {selected.get(pos, 'TBD')}" for pos in lineup_order if pos in selected]
    if not result:
        result = ["4-3-3 formation assumed"]
    return result


def _build_key_players(matches: list[dict[str, Any]]) -> list[KeyPlayer]:
    player_data: dict[str, dict] = {}
    for m in matches:
        for p in m.get("players", []):
            name = p.get("name", "")
            if not name:
                continue
            if name not in player_data:
                player_data[name] = {
                    "goals": 0,
                    "assists": 0,
                    "xg": 0.0,
                    "position": p.get("position", ""),
                    "apps": 0,
                }
            player_data[name]["goals"] += p.get("goals", 0)
            player_data[name]["assists"] += p.get("assists", 0)
            player_data[name]["xg"] += p.get("xg", 0.0)
            player_data[name]["apps"] += 1
        for s in m.get("scorers", []):
            name = s.get("player", "")
            if name and name not in player_data:
                player_data[name] = {"goals": 0, "assists": 0, "xg": 0.0, "position": "", "apps": 0}
            if name:
                player_data[name]["goals"] += s.get("goals", 0)
        for a in m.get("assisters", []):
            name = a.get("player", "")
            if name and name not in player_data:
                player_data[name] = {"goals": 0, "assists": 0, "xg": 0.0, "position": "", "apps": 0}
            if name:
                player_data[name]["assists"] += a.get("assists", 0)
    players = []
    for name, data in player_data.items():
        threat = data["goals"] * 2.0 + data["assists"] * 1.5 + data["xg"] * 3.0
        key_stat = ""
        if data["goals"] >= 3:
            key_stat = f"Top scorer ({data['goals']} goals)"
        elif data["assists"] >= 3:
            key_stat = f"Top creator ({data['assists']} assists)"
        elif data["xg"] > 1.0:
            key_stat = f"High xG ({data['xg']:.2f})"
        players.append(
            KeyPlayer(
                name=name,
                position=data["position"],
                goals=data["goals"],
                assists=data["assists"],
                xg=data["xg"],
                threat_score=threat,
                key_stat=key_stat,
            )
        )
    players.sort(key=lambda p: -p.threat_score)
    return players[:8]


def _classify_pressing(matches: list[dict[str, Any]]) -> str:
    ppdas = [m.get("ppda", 10) for m in matches if m.get("ppda")]
    if not ppdas:
        return "unknown"
    avg = statistics.mean(ppdas)
    if avg < 8:
        return "high intensity"
    if avg < 13:
        return "moderate"
    return "deep / low block"


def _classify_build_up(matches: list[dict[str, Any]]) -> str:
    styles = [m.get("build_up_style", "") for m in matches if m.get("build_up_style")]
    if not styles:
        return "mixed"
    return max(set(styles), key=styles.count)


def _detect_set_piece_tendencies(matches: list[dict[str, Any]]) -> list[str]:
    tendencies = []
    total_corners = sum(m.get("corners_for", 0) for m in matches)
    avg_corners = total_corners / max(1, len(matches))
    sp_threats = [m.get("set_piece_threat", 0) for m in matches]
    avg_sp_threat = statistics.mean(sp_threats) if sp_threats else 0
    sp_conceded = [m.get("set_piece_conceded", 0) for m in matches]
    avg_sp_conc = statistics.mean(sp_conceded) if sp_conceded else 0
    if avg_corners > 5:
        tendencies.append(f"High corner volume ({avg_corners:.1f}/game) — vary marking schemes")
    if avg_sp_threat > 0.25:
        tendencies.append(
            f"Dangerous from set pieces (threat {avg_sp_threat:.3f}) — prioritize blocking"
        )
    if avg_sp_conc > 0.2:
        tendencies.append(f"Vulnerable defending set pieces (concedes {avg_sp_conc:.3f} xG/game)")
    if not tendencies:
        tendencies.append("No significant set piece patterns detected")
    return tendencies


def _predict_scoreline(matches: list[dict[str, Any]]) -> ScorelinePrediction:
    goals_for = [m.get("goals_for", 0) for m in matches]
    goals_against = [m.get("goals_against", 0) for m in matches]
    if not goals_for:
        return ScorelinePrediction()
    avg_for = statistics.mean(goals_for)
    avg_against = statistics.mean(goals_against)
    home_score = round(avg_for)
    away_score = round(avg_against)
    total_goals = avg_for + avg_against
    btts = sum(
        1 for g1, g2 in zip(goals_for, goals_against, strict=False) if g1 > 0 and g2 > 0
    ) / max(1, len(goals_for))
    lam_for = max(0.1, avg_for)
    lam_against = max(0.1, avg_against)
    home_win = 1 - math.exp(-lam_for) * (1 + (1 - math.exp(-lam_against)))
    draw = math.exp(-lam_for) * math.exp(-lam_against) * (1 + lam_for + lam_against)
    away_win = 1 - home_win - draw
    total = home_win + draw + away_win
    if total > 0:
        home_win /= total
        draw /= total
        away_win /= total
    return ScorelinePrediction(
        home_score=home_score,
        away_score=away_score,
        home_win_prob=home_win,
        draw_prob=draw,
        away_win_prob=away_win,
        total_goals_avg=total_goals,
        both_teams_score_prob=btts,
    )


def generate_dossier(
    opponent_name: str,
    matches: list[dict[str, Any]],
    our_team_name: str = "",
) -> OpponentDossier:
    strengths = []
    weaknesses = []
    recs = []

    avg_poss = statistics.mean([m.get("possession_pct", 50) for m in matches]) if matches else 50
    if avg_poss > 58:
        strengths.append("Controls possession — patient build-up")
        recs.append("Stay compact in mid-block, absorb pressure")
    elif avg_poss < 42:
        weaknesses.append("Low possession — may struggle to control games")
        recs.append("Dominate possession, force them to chase")

    ppda_values = [float(m["ppda"]) for m in matches if m.get("ppda") is not None]
    avg_ppda = (
        statistics.mean(ppda_values)
        if ppda_values
        else 10  # 10 = mid-range default when unmeasurable
    )
    if avg_ppda < 8:
        strengths.append("High pressing — aggressive out-of-possession")
        recs.append("Quick vertical passes to bypass press")
    elif avg_ppda > 13:
        weaknesses.append("Low pressing intensity — space to build from back")
        recs.append("Build from back with confidence")

    sp_conc = statistics.mean([m.get("set_piece_conceded", 0) for m in matches]) if matches else 0
    if sp_conc > 0.2:
        weaknesses.append("Vulnerable from set pieces")
        recs.append("Target set piece opportunities")

    formations = _detect_formations(matches)
    _ = [f.formation for f in formations[:3]]
    lineup = _predict_lineup(matches)
    key_players = _build_key_players(matches)
    pressing = _classify_pressing(matches)
    build_up = _classify_build_up(matches)
    sp_tendencies = _detect_set_piece_tendencies(matches)
    scoreline = _predict_scoreline(matches)

    if not strengths:
        strengths.append("Balanced team with no clear weaknesses from available data")
    if not recs:
        recs.append("Standard preparation with focus on own strengths")

    return OpponentDossier(
        opponent_name=opponent_name,
        matches_analyzed=len(matches),
        formations=formations,
        predicted_lineup=lineup,
        key_players=key_players,
        strengths=strengths,
        weaknesses=weaknesses,
        set_piece_tendencies=sp_tendencies,
        build_up_pattern=build_up,
        pressing_style=pressing,
        scoreline_prediction=scoreline,
        recommended_tactics=recs,
    )
