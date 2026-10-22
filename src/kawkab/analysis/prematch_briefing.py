"""Pre-match Manager Briefing — Football Manager/PES style match preview.

Aggregates opponent dossier, team form, injuries, suspensions, referee
profile, head-to-head, weather, and tactical recommendations into a
single structured briefing with JSON and markdown output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


# ── supporting dataclasses ─────────────────────────────────────────


@dataclass
class FormEntry:
    result: str = ""
    opponent: str = ""
    home_score: int = 0
    away_score: int = 0
    competition: str = ""
    date: str = ""

    def to_dict(self) -> dict:
        return {
            "result": self.result,
            "opponent": self.opponent,
            "home_score": self.home_score,
            "away_score": self.away_score,
            "competition": self.competition,
            "date": self.date,
        }


@dataclass
class InjuryEntry:
    player: str = ""
    injury: str = ""
    expected_return: str = ""
    severity: str = ""

    def to_dict(self) -> dict:
        return {
            "player": self.player,
            "injury": self.injury,
            "expected_return": self.expected_return,
            "severity": self.severity,
        }


@dataclass
class PredictedLineupPlayer:
    position: str = ""
    player_name: str = ""
    number: int = 0
    rating: float = 0.0

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "player_name": self.player_name,
            "number": self.number,
            "rating": round(self.rating, 1),
        }


@dataclass
class KeyBattle:
    our_player: str = ""
    opponent_player: str = ""
    importance: str = "medium"
    our_advantage: bool | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "our_player": self.our_player,
            "opponent_player": self.opponent_player,
            "importance": self.importance,
            "our_advantage": self.our_advantage,
            "notes": self.notes,
        }


@dataclass
class TacticalRecommendations:
    suggested_formation: str = ""
    pressing_strategy: str = ""
    attacking_focus: str = ""
    defensive_focus: str = ""
    set_piece_plans: list[str] = field(default_factory=list)
    key_battles: list[KeyBattle] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "suggested_formation": self.suggested_formation,
            "pressing_strategy": self.pressing_strategy,
            "attacking_focus": self.attacking_focus,
            "defensive_focus": self.defensive_focus,
            "set_piece_plans": self.set_piece_plans,
            "key_battles": [b.to_dict() for b in self.key_battles],
            "notes": self.notes,
        }


@dataclass
class MatchPrediction:
    home_score: float = 0.0
    away_score: float = 0.0
    win_probability: float = 0.0
    draw_probability: float = 0.0
    loss_probability: float = 0.0
    btts_probability: float = 0.0
    confidence: str = "medium"

    def to_dict(self) -> dict:
        return {
            "home_score": round(self.home_score, 1),
            "away_score": round(self.away_score, 1),
            "win_probability": round(self.win_probability, 3),
            "draw_probability": round(self.draw_probability, 3),
            "loss_probability": round(self.loss_probability, 3),
            "btts_probability": round(self.btts_probability, 3),
            "confidence": self.confidence,
        }


@dataclass
class PreMatchBriefing:
    """Complete pre-match briefing — the Manager's match preview."""

    # Match info
    home_team: str = ""
    away_team: str = ""
    competition: str = ""
    venue: str = ""
    match_date: str = ""
    kickoff: str = ""
    referee_name: str = ""

    # Our team
    our_team: str = ""
    our_form: list[FormEntry] = field(default_factory=list)
    our_streak: str = ""
    our_injuries: list[InjuryEntry] = field(default_factory=list)
    our_suspensions: list[str] = field(default_factory=list)
    our_top_scorer: str = ""
    our_formation: str = ""
    our_predicted_lineup: list[PredictedLineupPlayer] = field(default_factory=list)
    our_avg_possession: float = 0.0

    # Opponent
    opponent_form: list[FormEntry] = field(default_factory=list)
    opponent_preferred_formation: str = ""
    opponent_pressing: str = ""
    opponent_build_up: str = ""
    opponent_key_players: list[dict] = field(default_factory=list)
    opponent_predicted_lineup: list[PredictedLineupPlayer] = field(default_factory=list)
    opponent_set_piece_tendencies: list[str] = field(default_factory=list)
    opponent_vulnerabilities: list[str] = field(default_factory=list)
    opponent_strengths: list[str] = field(default_factory=list)

    # Head-to-head
    h2h_total_meetings: int = 0
    h2h_home_wins: int = 0
    h2h_away_wins: int = 0
    h2h_draws: int = 0
    h2h_recent: list[dict] = field(default_factory=list)
    h2h_avg_goals: float = 0.0

    # Referee
    referee_avg_cards: float = 0.0
    referee_home_bias: float = 0.0
    referee_foul_threshold: float = 0.0
    referee_inconsistency: float = 0.0

    # Weather
    weather_condition: str = ""
    weather_temperature: float | None = None
    weather_wind: float | None = None
    weather_precipitation: float | None = None

    # Tactical
    tactical: TacticalRecommendations = field(default_factory=TacticalRecommendations)

    # Prediction
    prediction: MatchPrediction = field(default_factory=MatchPrediction)

    # Metadata
    generated_at: str = ""
    briefing_id: str = ""

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now(UTC).isoformat()
        if not self.briefing_id:
            import uuid

            self.briefing_id = uuid.uuid4().hex[:12]

    def to_dict(self) -> dict[str, Any]:
        return {
            "briefing_id": self.briefing_id,
            "generated_at": self.generated_at,
            "match_info": {
                "home_team": self.home_team,
                "away_team": self.away_team,
                "competition": self.competition,
                "venue": self.venue,
                "match_date": self.match_date,
                "kickoff": self.kickoff,
                "referee_name": self.referee_name,
            },
            "our_team": {
                "name": self.our_team,
                "form": [f.to_dict() for f in self.our_form],
                "streak": self.our_streak,
                "injuries": [i.to_dict() for i in self.our_injuries],
                "suspensions": self.our_suspensions,
                "top_scorer": self.our_top_scorer,
                "formation": self.our_formation,
                "predicted_lineup": [p.to_dict() for p in self.our_predicted_lineup],
                "avg_possession": round(self.our_avg_possession, 1),
            },
            "opponent": {
                "name": self.away_team if self.our_team == self.home_team else self.home_team,
                "form": [f.to_dict() for f in self.opponent_form],
                "preferred_formation": self.opponent_preferred_formation,
                "pressing_intensity": self.opponent_pressing,
                "build_up_style": self.opponent_build_up,
                "key_players": self.opponent_key_players[:5],
                "predicted_lineup": [p.to_dict() for p in self.opponent_predicted_lineup],
                "set_piece_tendencies": self.opponent_set_piece_tendencies,
                "vulnerabilities": self.opponent_vulnerabilities[:5],
                "strengths": self.opponent_strengths[:5],
            },
            "head_to_head": {
                "total_meetings": self.h2h_total_meetings,
                "home_wins": self.h2h_home_wins,
                "away_wins": self.h2h_away_wins,
                "draws": self.h2h_draws,
                "recent": self.h2h_recent[-5:],
                "avg_goals_per_game": round(self.h2h_avg_goals, 2),
            },
            "referee": {
                "name": self.referee_name,
                "avg_cards_per_game": round(self.referee_avg_cards, 2),
                "home_bias": round(self.referee_home_bias, 3),
                "foul_threshold": round(self.referee_foul_threshold, 1),
                "inconsistency_score": round(self.referee_inconsistency, 3),
            },
            "weather": {
                "condition": self.weather_condition,
                "temperature": self.weather_temperature,
                "wind_kph": self.weather_wind,
                "precipitation_mm": self.weather_precipitation,
            },
            "tactical": self.tactical.to_dict(),
            "prediction": self.prediction.to_dict(),
        }

    def to_markdown(self) -> str:
        """Generate a printable markdown briefing document."""
        md = []
        md.append(f"# ⚽ Pre-Match Briefing: {self.home_team} vs {self.away_team}")
        md.append("")
        md.append(f"**Competition:** {self.competition}")
        md.append(f"**Date:** {self.match_date} | **Kickoff:** {self.kickoff}")
        md.append(f"**Venue:** {self.venue}")
        md.append(f"**Referee:** {self.referee_name}")
        if self.weather_condition:
            md.append(
                f"**Weather:** {self.weather_condition}"
                + (f" ({self.weather_temperature}°C)" if self.weather_temperature else "")
            )
        md.append("")
        md.append("---")
        md.append("")

        md.append("## 📊 Form Guide")
        md.append("")
        md.append(f"### {self.our_team} — Streak: {self.our_streak}")
        for f in self.our_form[-5:]:
            icon = {"W": "✅", "D": "➖", "L": "❌"}.get(f.result, "➖")
            md.append(f"- {icon} {f.result} {f.home_score}-{f.away_score} vs {f.opponent}")
        md.append("")
        opp_name = self.away_team if self.our_team == self.home_team else self.home_team
        md.append(f"### {opp_name}")
        for f in self.opponent_form[-5:]:
            icon = {"W": "✅", "D": "➖", "L": "❌"}.get(f.result, "➖")
            md.append(f"- {icon} {f.result} {f.home_score}-{f.away_score} vs {f.opponent}")
        md.append("")

        md.append("---")
        md.append("## 🏥 Team News")
        md.append("")
        if self.our_injuries:
            md.append("**Injuries:**")
            for i in self.our_injuries:
                md.append(f"- {i.player} ({i.injury}) — expected back {i.expected_return}")
        if self.our_suspensions:
            md.append("**Suspensions:** " + ", ".join(self.our_suspensions))
        if not self.our_injuries and not self.our_suspensions:
            md.append("No injury or suspension concerns.")
        md.append("")

        if self.our_predicted_lineup:
            md.append("**Predicted XI:**")
            for p in self.our_predicted_lineup:
                md.append(
                    f"  {p.position} — {p.player_name}" + (f" (⭐{p.rating})" if p.rating else "")
                )
        md.append("")

        md.append("---")
        md.append(f"## 🏴󠁧󠁢󠁥󠁮󠁧󠁿 Opponent: {opp_name}")
        md.append("")
        if self.opponent_preferred_formation:
            md.append(f"**Formation:** {self.opponent_preferred_formation}")
        if self.opponent_pressing:
            md.append(f"**Pressing:** {self.opponent_pressing}")
        if self.opponent_build_up:
            md.append(f"**Build-up:** {self.opponent_build_up}")
        md.append("")
        if self.opponent_key_players:
            md.append("**Key Threats:**")
            for kp in self.opponent_key_players[:5]:
                name = kp.get("name", "")
                stat = kp.get("key_stat", "")
                threat = kp.get("threat_score", 0)
                md.append(f"- {name} (threat: {threat}/100) — {stat}")
        md.append("")
        if self.opponent_vulnerabilities:
            md.append("**Vulnerabilities:**")
            for v in self.opponent_vulnerabilities:
                md.append(f"- {v}")
        if self.opponent_strengths:
            md.append("**Strengths:**")
            for s in self.opponent_strengths:
                md.append(f"- {s}")
        md.append("")

        md.append("---")
        md.append("## 🤝 Head-to-Head")
        md.append("")
        md.append(f"Total meetings: **{self.h2h_total_meetings}**")
        md.append(
            f"Home wins: **{self.h2h_home_wins}** | Away wins: **{self.h2h_away_wins}** | Draws: **{self.h2h_draws}**"
        )
        md.append(f"Avg goals per game: **{self.h2h_avg_goals:.2f}**")
        if self.h2h_recent:
            md.append("")
            md.append("**Recent meetings:**")
            for h in self.h2h_recent[-5:]:
                md.append(
                    f"- {h.get('date', '')}: {h.get('home', '')} {h.get('home_score', '')}-{h.get('away_score', '')} {h.get('away', '')}"
                )
        md.append("")

        md.append("---")
        md.append("## 👨‍⚖️ Referee: {self.referee_name}")
        md.append("")
        md.append(f"- Avg cards per game: {self.referee_avg_cards:.1f}")
        md.append(
            f"- Home bias: {self.referee_home_bias:.2f}"
            + (" (slight home favor)" if self.referee_home_bias > 0.05 else "")
        )
        md.append(f"- Foul threshold: {self.referee_foul_threshold:.1f}")
        md.append("")

        md.append("---")
        md.append("## 🧠 Tactical Plan")
        md.append("")
        if self.tactical.suggested_formation:
            md.append(f"**Suggested formation:** {self.tactical.suggested_formation}")
        if self.tactical.pressing_strategy:
            md.append(f"**Pressing strategy:** {self.tactical.pressing_strategy}")
        if self.tactical.attacking_focus:
            md.append(f"**Attacking focus:** {self.tactical.attacking_focus}")
        if self.tactical.defensive_focus:
            md.append(f"**Defensive focus:** {self.tactical.defensive_focus}")
        if self.tactical.set_piece_plans:
            md.append("**Set piece plans:**")
            for sp in self.tactical.set_piece_plans:
                md.append(f"- {sp}")
        if self.tactical.key_battles:
            md.append("**Key battles:**")
            for kb in self.tactical.key_battles:
                icon = "✅" if kb.our_advantage else "⚠️" if kb.our_advantage is False else "➖"
                md.append(f"- {icon} {kb.our_player} vs {kb.opponent_player} ({kb.importance})")
        if self.tactical.notes:
            md.append("**Notes:**")
            for n in self.tactical.notes:
                md.append(f"- {n}")
        md.append("")

        md.append("---")
        md.append("## 📈 Match Prediction")
        md.append("")
        md.append(
            f"**Score:** {self.home_team} {self.prediction.home_score:.1f} — {self.prediction.away_score:.1f} {self.away_team}"
        )
        md.append(f"- Win: **{self.prediction.win_probability * 100:.0f}%**")
        md.append(f"- Draw: **{self.prediction.draw_probability * 100:.0f}%**")
        md.append(f"- Loss: **{self.prediction.loss_probability * 100:.0f}%**")
        md.append(f"- Both teams to score: **{self.prediction.btts_probability * 100:.0f}%**")
        md.append(f"- Confidence: **{self.prediction.confidence}**")
        md.append("")
        md.append("---")
        md.append(f"*Generated: {self.generated_at}*")

        return "\n".join(md)

    def to_html(self) -> str:
        """Generate inline HTML for in-app display."""
        lines = []
        lines.append('<div class="manager-briefing">')

        lines.append('<div class="briefing-header">')
        lines.append(f"<h2>⚽ {self.home_team} vs {self.away_team}</h2>")
        lines.append(
            f'<p class="briefing-meta">{self.competition} — {self.match_date} | {self.kickoff} | {self.venue}</p>'
        )
        lines.append(f'<p class="briefing-meta">Ref: {self.referee_name}')
        if self.weather_condition:
            lines.append(
                f" | 🌤 {self.weather_condition}{f' {self.weather_temperature}°C' if self.weather_temperature else ''}"
            )
        lines.append("</p></div>")

        lines.append('<div class="briefing-grid">')

        lines.append('<div class="briefing-column"><h3>📊 Form</h3>')
        lines.append(f'<h4>{self.our_team} ({self.our_streak})</h4><table class="briefing-table">')
        for f in self.our_form[-5:]:
            lines.append(
                f"<tr><td>{f.result}</td><td>{f.home_score}-{f.away_score}</td><td>vs {f.opponent}</td></tr>"
            )
        lines.append("</table></div>")

        lines.append('<div class="briefing-column"><h3>🏥 Team News</h3>')
        if self.our_injuries:
            for i in self.our_injuries:
                lines.append(f"<p>🩹 {i.player} — {i.injury} ({i.expected_return})</p>")
        else:
            lines.append("<p>✅ No injuries</p>")
        if self.our_predicted_lineup:
            lines.append('<h4>Predicted XI</h4><table class="briefing-table">')
            for p in self.our_predicted_lineup:
                lines.append(f"<tr><td>{p.position}</td><td>{p.player_name}</td></tr>")
            lines.append("</table>")
        lines.append("</div>")

        lines.append('<div class="briefing-column"><h3>🏴 Opponent</h3>')
        _ = self.away_team if self.our_team == self.home_team else self.home_team
        lines.append(
            f"<p>Formation: {self.opponent_preferred_formation} | Press: {self.opponent_pressing}</p>"
        )
        if self.opponent_key_players:
            lines.append("<h4>Key Threats</h4>")
            for kp in self.opponent_key_players[:3]:
                lines.append(f"<p>⚠ {kp.get('name', '')} — {kp.get('key_stat', '')}</p>")
        lines.append("</div>")

        lines.append('<div class="briefing-column"><h3>🧠 Tactical Plan</h3>')
        if self.tactical.suggested_formation:
            lines.append(f"<p><b>Formation:</b> {self.tactical.suggested_formation}</p>")
        if self.tactical.pressing_strategy:
            lines.append(f"<p><b>Press:</b> {self.tactical.pressing_strategy}</p>")
        if self.tactical.attacking_focus:
            lines.append(f"<p><b>Attack:</b> {self.tactical.attacking_focus}</p>")
        if self.tactical.defensive_focus:
            lines.append(f"<p><b>Defend:</b> {self.tactical.defensive_focus}</p>")
        lines.append("</div>")

        lines.append('<div class="briefing-column"><h3>🤝 Head-to-Head</h3>')
        lines.append(
            f"<p>{self.h2h_total_meetings} meetings | {self.h2h_home_wins}W {self.h2h_draws}D {self.h2h_away_wins}L</p>"
        )
        lines.append(f"<p>Avg goals: {self.h2h_avg_goals:.2f}</p>")
        lines.append("</div>")

        lines.append('<div class="briefing-column"><h3>📈 Prediction</h3>')
        lines.append(
            f'<p class="prediction-score">{self.prediction.home_score:.1f} — {self.prediction.away_score:.1f}</p>'
        )
        lines.append(
            f"<p>W {self.prediction.win_probability * 100:.0f}% | D {self.prediction.draw_probability * 100:.0f}% | L {self.prediction.loss_probability * 100:.0f}%</p>"
        )
        lines.append(f"<p>BTTS: {self.prediction.btts_probability * 100:.0f}%</p>")
        lines.append("</div>")

        lines.append("</div></div>")
        return "\n".join(lines)


# ── service ────────────────────────────────────────────────────────


class PreMatchBriefingService:
    """Aggregates all data sources into a pre-match briefing."""

    def __init__(self):
        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    def generate(
        self,
        home_team: str,
        away_team: str,
        our_side: str = "home",
        competition: str = "",
        venue: str = "",
        match_date: str = "",
        kickoff: str = "",
        referee_name: str = "",
        # Our team data
        our_form_data: list[dict] | None = None,
        our_injuries_data: list[dict] | None = None,
        our_suspensions_data: list[str] | None = None,
        our_top_scorer: str = "",
        our_formation: str = "",
        our_predicted_lineup_data: list[dict] | None = None,
        our_avg_possession: float = 0.0,
        # Opponent data
        opponent_form_data: list[dict] | None = None,
        opponent_preferred_formation: str = "",
        opponent_pressing: str = "",
        opponent_build_up: str = "",
        opponent_key_players_data: list[dict] | None = None,
        opponent_predicted_lineup_data: list[dict] | None = None,
        opponent_set_piece_tendencies: list[str] | None = None,
        opponent_vulnerabilities: list[str] | None = None,
        opponent_strengths: list[str] | None = None,
        # H2H data
        h2h_data: list[dict] | None = None,
        # Referee data
        referee_avg_cards: float = 0.0,
        referee_home_bias: float = 0.0,
        referee_foul_threshold: float = 0.0,
        referee_inconsistency: float = 0.0,
        # Weather
        weather_condition: str = "",
        weather_temperature: float | None = None,
        weather_wind: float | None = None,
        weather_precipitation: float | None = None,
        # Tactical
        suggested_formation: str = "",
        pressing_strategy: str = "",
        attacking_focus: str = "",
        defensive_focus: str = "",
        set_piece_plans: list[str] | None = None,
        key_battles_data: list[dict] | None = None,
        tactical_notes: list[str] | None = None,
        # Prediction
        home_score_pred: float = 0.0,
        away_score_pred: float = 0.0,
        win_prob: float = 0.0,
        draw_prob: float = 0.0,
        loss_prob: float = 0.0,
        btts_prob: float = 0.0,
        prediction_confidence: str = "medium",
    ) -> PreMatchBriefing:
        our_team_name = home_team if our_side == "home" else away_team
        _ = away_team if our_side == "home" else home_team

        briefing = PreMatchBriefing(
            home_team=home_team,
            away_team=away_team,
            competition=competition,
            venue=venue,
            match_date=match_date,
            kickoff=kickoff,
            referee_name=referee_name,
            our_team=our_team_name,
            our_formation=our_formation,
            our_top_scorer=our_top_scorer,
            our_avg_possession=our_avg_possession,
            opponent_preferred_formation=opponent_preferred_formation,
            opponent_pressing=opponent_pressing,
            opponent_build_up=opponent_build_up,
            opponent_set_piece_tendencies=opponent_set_piece_tendencies or [],
            opponent_vulnerabilities=opponent_vulnerabilities or [],
            opponent_strengths=opponent_strengths or [],
            referee_avg_cards=referee_avg_cards,
            referee_home_bias=referee_home_bias,
            referee_foul_threshold=referee_foul_threshold,
            referee_inconsistency=referee_inconsistency,
            weather_condition=weather_condition,
            weather_temperature=weather_temperature,
            weather_wind=weather_wind,
            weather_precipitation=weather_precipitation,
        )

        # Parse form
        if our_form_data:
            briefing.our_form = [self._parse_form(f) for f in our_form_data]
            briefing.our_streak = self._compute_streak(briefing.our_form)
        if opponent_form_data:
            briefing.opponent_form = [self._parse_form(f) for f in opponent_form_data]

        # Parse injuries
        if our_injuries_data:
            briefing.our_injuries = [
                InjuryEntry(**{k: v for k, v in i.items() if k in InjuryEntry.__dataclass_fields__})
                for i in our_injuries_data
            ]
        if our_suspensions_data:
            briefing.our_suspensions = our_suspensions_data

        # Parse predicted lineups
        if our_predicted_lineup_data:
            briefing.our_predicted_lineup = [
                PredictedLineupPlayer(
                    **{
                        k: v
                        for k, v in p.items()
                        if k in PredictedLineupPlayer.__dataclass_fields__
                    }
                )
                for p in our_predicted_lineup_data
            ]
        if opponent_predicted_lineup_data:
            briefing.opponent_predicted_lineup = [
                PredictedLineupPlayer(
                    **{
                        k: v
                        for k, v in p.items()
                        if k in PredictedLineupPlayer.__dataclass_fields__
                    }
                )
                for p in opponent_predicted_lineup_data
            ]

        # Key players
        if opponent_key_players_data:
            briefing.opponent_key_players = opponent_key_players_data

        # H2H
        if h2h_data:
            briefing.h2h_total_meetings = len(h2h_data)
            briefing.h2h_recent = h2h_data
            for h in h2h_data:
                hs = h.get("home_score", 0) or 0
                aws = h.get("away_score", 0) or 0
                briefing.h2h_avg_goals += hs + aws
                if our_side == "home":
                    if hs > aws:
                        briefing.h2h_home_wins += 1
                    elif aws > hs:
                        briefing.h2h_away_wins += 1
                    else:
                        briefing.h2h_draws += 1
                else:
                    if aws > hs:
                        briefing.h2h_home_wins += 1
                    elif hs > aws:
                        briefing.h2h_away_wins += 1
                    else:
                        briefing.h2h_draws += 1
            briefing.h2h_avg_goals = briefing.h2h_avg_goals / max(1, len(h2h_data))

        # Tactical
        briefing.tactical = TacticalRecommendations(
            suggested_formation=suggested_formation,
            pressing_strategy=pressing_strategy,
            attacking_focus=attacking_focus,
            defensive_focus=defensive_focus,
            set_piece_plans=set_piece_plans or [],
            key_battles=[
                KeyBattle(**{k: v for k, v in kb.items() if k in KeyBattle.__dataclass_fields__})
                for kb in (key_battles_data or [])
            ],
            notes=tactical_notes or [],
        )

        # Prediction
        briefing.prediction = MatchPrediction(
            home_score=home_score_pred,
            away_score=away_score_pred,
            win_probability=win_prob,
            draw_probability=draw_prob,
            loss_probability=loss_prob,
            btts_probability=btts_prob,
            confidence=prediction_confidence,
        )

        return briefing

    def _parse_form(self, f: dict) -> FormEntry:
        return FormEntry(
            result=str(f.get("result", "D")),
            opponent=str(f.get("opponent", "")),
            home_score=int(f.get("home_score", 0) or 0),
            away_score=int(f.get("away_score", 0) or 0),
            competition=str(f.get("competition", "")),
            date=str(f.get("date", "")),
        )

    def _compute_streak(self, form: list[FormEntry]) -> str:
        if not form:
            return ""
        last = form[-1].result
        count = 0
        for f in reversed(form):
            if f.result == last:
                count += 1
            else:
                break
        streak_names = {"W": "winning", "D": "unbeaten", "L": "losing"}
        name = streak_names.get(last, "unbeaten")
        return f"{count} {name}" if count > 1 else f"Last: {last}"
