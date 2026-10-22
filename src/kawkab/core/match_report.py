"""Day After Match Report — structured post-match analysis.

Produces a configurable report similar to what professional clubs use,
with auto-generated executive summary, key moments, tactical observations,
and areas for improvement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MatchMoment:
    minute: int
    type: str
    description: str
    xg: float | None
    team: str
    player: str | None


@dataclass
class ReportSection:
    title: str
    content: str
    included: bool = True


@dataclass
class ReportTemplate:
    name: str = "standard"
    sections: list[str] = field(
        default_factory=lambda: [
            "executive_summary",
            "xg_flow",
            "key_moments",
            "player_ratings",
            "tactical_observations",
            "notable_patterns",
            "set_pieces",
            "phase_breakdown",
            "areas_for_improvement",
            "what_worked_well",
        ]
    )
    include_charts: bool = True
    include_stat_tables: bool = True
    detail_level: str = "normal"


@dataclass
class DayAfterMatchReport:
    match_id: str
    match_title: str
    date: str
    competition: str
    result: str

    executive_summary: str = ""
    xg_flow_analysis: str = ""
    key_moments: list[MatchMoment] = field(default_factory=list)
    player_ratings_highlights: list[dict] = field(default_factory=list)
    tactical_observations: list[str] = field(default_factory=list)
    notable_patterns: list[str] = field(default_factory=list)
    set_piece_analysis: str = ""
    phase_breakdown: str | None = None
    areas_for_improvement: list[str] = field(default_factory=list)
    what_worked_well: list[str] = field(default_factory=list)

    sections: list[ReportSection] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "match_title": self.match_title,
            "date": self.date,
            "competition": self.competition,
            "result": self.result,
            "executive_summary": self.executive_summary,
            "xg_flow_analysis": self.xg_flow_analysis,
            "key_moments": [
                {
                    "minute": m.minute,
                    "type": m.type,
                    "description": m.description,
                    "xg": m.xg,
                    "team": m.team,
                    "player": m.player,
                }
                for m in self.key_moments
            ],
            "player_ratings_highlights": self.player_ratings_highlights,
            "tactical_observations": self.tactical_observations,
            "notable_patterns": self.notable_patterns,
            "set_piece_analysis": self.set_piece_analysis,
            "phase_breakdown": self.phase_breakdown,
            "areas_for_improvement": self.areas_for_improvement,
            "what_worked_well": self.what_worked_well,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Day After Match Report: {self.match_title}",
            f"**Date:** {self.date}  |  **Competition:** {self.competition}  |  **Result:** {self.result}",
            "",
        ]
        for section in self.sections:
            if not section.included:
                continue
            lines.append(f"## {section.title}")
            lines.append("")
            lines.append(section.content)
            lines.append("")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def _generate_executive_summary(
    match_title: str,
    result: str,
    key_moments: list[MatchMoment],
    xg_timeline: list[dict] | None,
    phase_xg_report: dict | None,
) -> str:
    n_goals = sum(1 for m in key_moments if m.type == "goal")
    n_big_chances = sum(1 for m in key_moments if m.type == "big_chance")
    n_cards = sum(1 for m in key_moments if m.type == "card")

    total_xg_home = 0.0
    total_xg_away = 0.0
    if xg_timeline:
        for entry in xg_timeline:
            team = entry.get("team", "")
            xg = float(entry.get("xg", entry.get("value", 0.0)))
            if team == "home":
                total_xg_home += xg
            else:
                total_xg_away += xg

    summary = f"In {match_title}, the match ended {result}. "
    if n_goals > 0:
        summary += f"There were {n_goals} goal(s) scored"
        if n_big_chances:
            summary += f" from {n_big_chances} big chance(s)"
        summary += ". "
    if total_xg_home > 0 or total_xg_away > 0:
        summary += f"xG summed to {total_xg_home:.2f}–{total_xg_away:.2f}. "
    if n_cards > 0:
        summary += f"{n_cards} card(s) were shown. "

    if phase_xg_report:
        if phase_xg_report.get("home_buildup_xg", 0) > phase_xg_report.get("away_buildup_xg", 0):
            summary += "The home side created more from open play. "
        elif phase_xg_report.get("away_buildup_xg", 0) > phase_xg_report.get("home_buildup_xg", 0):
            summary += "The away side created more from open play. "

    return summary.strip()


def _extract_key_moments(
    events: list[dict],
    home_team: str,
    away_team: str,
) -> list[MatchMoment]:
    moments: list[MatchMoment] = []
    for ev in events:
        minute = int(ev.get("minute", ev.get("timestamp", 0)))
        ev_type = ev.get("type", "")
        team = ev.get("team", "")
        player = ev.get("player", ev.get("player_name", None))
        xg_val: float | None = ev.get("xg", ev.get("metadata", {}).get("xg", None))
        if xg_val is not None:
            xg_val = float(xg_val)

        if ev_type == "goal":
            moments.append(
                MatchMoment(
                    minute=minute,
                    type="goal",
                    description=f"{player or 'Unknown'} scored for {team}",
                    xg=xg_val,
                    team=team,
                    player=player,
                )
            )
        elif ev_type in ("card", "red_card", "yellow_card"):
            card_type = ev.get("card_type", ev_type)
            desc = f"{player or 'Unknown'} received a {card_type} card"
            moments.append(
                MatchMoment(
                    minute=minute,
                    type="card",
                    description=desc,
                    xg=None,
                    team=team,
                    player=player,
                )
            )
        elif ev_type == "penalty" or ev_type == "penalty_goal":
            moments.append(
                MatchMoment(
                    minute=minute,
                    type="penalty",
                    description=f"Penalty {'scored by' if 'goal' in ev_type else 'awarded to'} {team}",
                    xg=0.76 if xg_val is None else xg_val,
                    team=team,
                    player=player,
                )
            )
        elif ev_type == "substitution":
            player_off = ev.get("player_off", player)
            player_on = ev.get("player_on", "")
            desc = f"Substitution: {player_on} replaces {player_off}"
            moments.append(
                MatchMoment(
                    minute=minute,
                    type="substitution",
                    description=desc,
                    xg=None,
                    team=team,
                    player=player_on,
                )
            )

        if xg_val is not None and xg_val > 0.3 and ev_type != "goal":
            desc = f"Big chance for {player or team}"
            if team:
                desc += f" ({team})"
            moments.append(
                MatchMoment(
                    minute=minute,
                    type="big_chance",
                    description=desc,
                    xg=xg_val,
                    team=team,
                    player=player,
                )
            )

    moments.sort(key=lambda m: m.minute)
    return moments


def _generate_tactical_observations(
    events: list[dict],
    phase_xg_report: dict | None,
    tactical_phases: list[dict] | None,
) -> list[str]:
    observations: list[str] = []

    total_passes = sum(1 for ev in events if ev.get("type") == "pass")
    total_tackles = sum(1 for ev in events if ev.get("type") in ("tackle", "defensive_action"))
    total_shots = sum(1 for ev in events if ev.get("type") == "shot")

    if total_passes > 50:
        observations.append(
            f"High pass volume ({total_passes} passes) indicates controlled possession approach."
        )
    elif total_passes < 20:
        observations.append(
            f"Low pass volume ({total_passes}) suggests direct play or disrupted rhythm."
        )
    else:
        observations.append(f"Moderate pass volume ({total_passes}) — balanced approach.")

    if total_tackles > 10:
        observations.append(
            f"High defensive intensity ({total_tackles} tackles/defensive actions)."
        )
    else:
        observations.append(f"Moderate defensive engagement ({total_tackles} defensive actions).")

    if total_shots > 15:
        observations.append(f"High shot volume ({total_shots}) indicates strong offensive output.")
    elif total_shots < 5:
        observations.append(f"Low shot volume ({total_shots}) — struggled to create chances.")

    if phase_xg_report:
        home_buildup = phase_xg_report.get("home_buildup_xg", 0)
        away_buildup = phase_xg_report.get("away_buildup_xg", 0)
        home_set_piece = phase_xg_report.get("home_set_piece_xg", 0)
        away_set_piece = phase_xg_report.get("away_set_piece_xg", 0)

        if home_buildup > away_buildup * 1.5:
            observations.append("Home team dominated open-play creation.")
        elif away_buildup > home_buildup * 1.5:
            observations.append("Away team dominated open-play creation.")

        if home_set_piece > 0.3:
            observations.append(
                f"Home team generated {home_set_piece:.2f} xG from set pieces — a key weapon."
            )
        if away_set_piece > 0.3:
            observations.append(
                f"Away team generated {away_set_piece:.2f} xG from set pieces — a key weapon."
            )

    if tactical_phases:
        for phase in tactical_phases:
            phase_name = phase.get("name", phase.get("phase", ""))
            phase_xg = phase.get("xg", 0)
            if phase_xg and float(phase_xg) > 0.4:
                observations.append(
                    f"Phase '{phase_name}' generated {float(phase_xg):.2f} xG — dominant period."
                )

    return observations


def _generate_notable_patterns(
    events: list[dict],
    pass_network_data: dict | None = None,
) -> list[str]:
    patterns: list[str] = []

    shot_events = [ev for ev in events if ev.get("type") == "shot"]
    if shot_events:
        left_shots = sum(1 for ev in shot_events if ev.get("x", 50) < 33)
        right_shots = sum(1 for ev in shot_events if ev.get("x", 50) > 67)
        central_shots = len(shot_events) - left_shots - right_shots
        total = len(shot_events)
        if total > 0:
            if left_shots / total > 0.4:
                patterns.append(
                    f"Heavy left-flank shot bias ({left_shots}/{total} shots from left)."
                )
            if right_shots / total > 0.4:
                patterns.append(
                    f"Heavy right-flank shot bias ({right_shots}/{total} shots from right)."
                )
            if central_shots / total > 0.5:
                patterns.append(f"Central shot emphasis ({central_shots}/{total} from middle).")

    pass_events = [ev for ev in events if ev.get("type") == "pass"]
    if pass_events:
        forward_passes = sum(
            1 for ev in pass_events if ev.get("direction", "").lower() == "forward"
        )
        backward_passes = sum(
            1 for ev in pass_events if ev.get("direction", "").lower() == "backward"
        )
        _ = sum(1 for ev in pass_events if ev.get("direction", "").lower() in ("lateral", "square"))
        total_p = len(pass_events)
        if total_p > 0 and forward_passes / total_p > 0.5:
            patterns.append(f"Direct passing style ({forward_passes}/{total_p} passes forward).")
        if total_p > 0 and backward_passes / total_p > 0.2:
            patterns.append(
                f"High backward pass rate ({backward_passes}/{total_p}) — possibly recycling possession."
            )

    card_events = [ev for ev in events if ev.get("type") in ("card", "red_card", "yellow_card")]
    if len(card_events) >= 3:
        patterns.append(f"Disciplinary issues ({len(card_events)} cards shown).")

    return patterns


def _generate_set_piece_analysis(events: list[dict]) -> str:
    set_piece_events = [
        ev for ev in events if ev.get("type") in ("corner", "free_kick", "throw_in", "set_piece")
    ]
    n_corners = sum(1 for ev in set_piece_events if ev.get("type") == "corner")
    n_free_kicks = sum(1 for ev in set_piece_events if ev.get("type") == "free_kick")

    shot_from_set_piece = sum(
        1 for ev in events if ev.get("type") == "shot" and ev.get("set_piece", False)
    )
    set_piece_xg = sum(
        float(ev.get("xg", 0.0))
        for ev in events
        if ev.get("type") == "shot" and ev.get("set_piece", False)
    )

    parts: list[str] = []
    if n_corners > 0:
        parts.append(f"{n_corners} corners")
    if n_free_kicks > 0:
        parts.append(f"{n_free_kicks} free kicks")
    if not parts:
        return "Limited set piece activity during the match."

    base = f"Match featured {', '.join(parts)}. "
    base += (
        f"Generated {shot_from_set_piece} shot(s) from set pieces totaling {set_piece_xg:.2f} xG."
    )
    if set_piece_xg > 0.5:
        base += " Set pieces were a significant threat."
    return base


def _generate_areas_for_improvement(
    events: list[dict],
    phase_xg_report: dict | None,
) -> list[str]:
    areas: list[str] = []

    if phase_xg_report:
        if (
            phase_xg_report.get("home_set_piece_xg", 0) > 0
            and phase_xg_report.get("away_set_piece_xg", 0) > 0
        ) and (
            phase_xg_report["home_set_piece_xg"] > phase_xg_report["away_set_piece_xg"] * 2
            or phase_xg_report["away_set_piece_xg"] > phase_xg_report["home_set_piece_xg"] * 2
        ):
            areas.append(
                "Improve set piece defending — conceded significant xG from dead-ball situations."
            )
        if phase_xg_report.get("home_transition_xg", 0) > 0.3:
            areas.append("Tighten transition defense — opponent created from counter-attacks.")

    conceded_goals = [ev for ev in events if ev.get("type") == "goal" and ev.get("team") != "home"]
    if conceded_goals:
        late_goals = [ev for ev in conceded_goals if int(ev.get("minute", 0)) > 75]
        if late_goals:
            areas.append(
                f"Conceded {len(late_goals)} goal(s) after 75' — address late-game concentration."
            )

    failed_clearances = sum(1 for ev in events if ev.get("type") in ("failed_clearance", "error"))
    if failed_clearances > 2:
        areas.append(f"{failed_clearances} defensive errors — improve composure under pressure.")

    if not areas:
        areas.append("No significant areas for improvement identified from available data.")

    return areas


def _generate_what_worked_well(
    events: list[dict],
    phase_xg_report: dict | None,
) -> list[str]:
    positives: list[str] = []

    goals_scored = [ev for ev in events if ev.get("type") == "goal" and ev.get("team") == "home"]
    n_goals = len(goals_scored)
    if n_goals >= 3:
        positives.append(f"Clinical finishing — {n_goals} goals scored.")
    elif n_goals >= 2:
        positives.append(f"Good attacking output — {n_goals} goals scored.")

    if phase_xg_report:
        if phase_xg_report.get("home_buildup_xg", 0) > 0.5:
            positives.append("Effective open-play build-up with dangerous chance creation.")
        if phase_xg_report.get("home_set_piece_xg", 0) > 0.3:
            positives.append("Set pieces provided a reliable attacking threat.")

    clean_sheet = not any(ev.get("type") == "goal" and ev.get("team") != "home" for ev in events)
    if clean_sheet:
        positives.append("Defensive solidity — kept a clean sheet.")

    pass_events = [ev for ev in events if ev.get("type") == "pass"]
    if pass_events:
        completed = sum(1 for ev in pass_events if ev.get("outcome", "complete") == "complete")
        if len(pass_events) > 0 and completed / len(pass_events) > 0.85:
            positives.append(
                f"High pass completion ({completed}/{len(pass_events)}, {100 * completed // len(pass_events)}%)."
            )

    if not positives:
        positives.append("Areas of success not clearly identifiable from available data.")

    return positives


def generate_match_report(
    match_id: str,
    match_meta: dict,
    events: list[dict],
    xg_timeline: list[dict] | None = None,
    player_ratings: list[dict] | None = None,
    phase_xg_report: dict | None = None,
    tactical_phases: list[dict] | None = None,
    narrative: str | None = None,
    template: ReportTemplate | None = None,
) -> DayAfterMatchReport:
    if template is None:
        template = ReportTemplate()

    home_team = match_meta.get("home_team", "Home")
    away_team = match_meta.get("away_team", "Away")

    key_moments = _extract_key_moments(events, home_team, away_team)
    executive_summary = narrative or _generate_executive_summary(
        match_meta.get("title", ""),
        match_meta.get("result", ""),
        key_moments,
        xg_timeline,
        phase_xg_report,
    )
    tactical_observations = _generate_tactical_observations(
        events, phase_xg_report, tactical_phases
    )
    notable_patterns = _generate_notable_patterns(events)
    set_piece_analysis = _generate_set_piece_analysis(events)
    areas_for_improvement = _generate_areas_for_improvement(events, phase_xg_report)
    what_worked_well = _generate_what_worked_well(events, phase_xg_report)

    xg_flow_text = ""
    if xg_timeline:
        max_entry = max(xg_timeline, key=lambda x: float(x.get("xg", x.get("value", 0))))
        xg_flow_text = (
            f"xG flow shows {len(xg_timeline)} shot events. "
            f"Highest xG moment: {float(max_entry.get('xg', max_entry.get('value', 0))):.2f} "
            f"at minute {max_entry.get('minute', max_entry.get('timestamp', 0))}."
        )

    player_highlights: list[dict] = []
    if player_ratings:
        sorted_ratings = sorted(
            player_ratings, key=lambda r: float(r.get("rating", 0)), reverse=True
        )
        for pr in sorted_ratings[:5]:
            player_highlights.append(
                {
                    "player": pr.get("name", pr.get("player_name", "Unknown")),
                    "rating": pr.get("rating", 0),
                    "highlight": pr.get("highlight", pr.get("summary", "")),
                }
            )

    phase_text: str | None = None
    if phase_xg_report:
        phase_text = (
            f"Home build-up xG: {phase_xg_report.get('home_buildup_xg', 0):.2f}, "
            f"Away build-up xG: {phase_xg_report.get('away_buildup_xg', 0):.2f}. "
            f"Home set piece xG: {phase_xg_report.get('home_set_piece_xg', 0):.2f}, "
            f"Away set piece xG: {phase_xg_report.get('away_set_piece_xg', 0):.2f}."
        )

    section_map: dict[str, str] = {
        "executive_summary": executive_summary,
        "xg_flow": xg_flow_text,
        "key_moments": "\n".join(
            f"- **{m.minute}'** [{m.type}] {m.description}"
            + (f" (xG: {m.xg:.2f})" if m.xg is not None else "")
            for m in key_moments
        )
        if key_moments
        else "No key moments identified.",
        "player_ratings": "\n".join(
            f"- {h['player']}: {h['rating']}/10 — {h['highlight']}" for h in player_highlights
        )
        if player_highlights
        else "No player ratings available.",
        "tactical_observations": "\n".join(f"- {o}" for o in tactical_observations),
        "notable_patterns": "\n".join(f"- {p}" for p in notable_patterns),
        "set_pieces": set_piece_analysis,
        "phase_breakdown": phase_text or "Phase breakdown not available.",
        "areas_for_improvement": "\n".join(f"- {a}" for a in areas_for_improvement),
        "what_worked_well": "\n".join(f"- {w}" for w in what_worked_well),
    }

    sections = [
        ReportSection(
            title=_section_title(key),
            content=_truncate_content(section_map.get(key, ""), template.detail_level),
            included=key in template.sections,
        )
        for key in [
            "executive_summary",
            "xg_flow",
            "key_moments",
            "player_ratings",
            "tactical_observations",
            "notable_patterns",
            "set_pieces",
            "phase_breakdown",
            "areas_for_improvement",
            "what_worked_well",
        ]
    ]

    return DayAfterMatchReport(
        match_id=match_id,
        match_title=match_meta.get("title", ""),
        date=match_meta.get("date", ""),
        competition=match_meta.get("competition", ""),
        result=match_meta.get("result", ""),
        executive_summary=executive_summary,
        xg_flow_analysis=xg_flow_text,
        key_moments=key_moments,
        player_ratings_highlights=player_highlights,
        tactical_observations=tactical_observations,
        notable_patterns=notable_patterns,
        set_piece_analysis=set_piece_analysis,
        phase_breakdown=phase_text,
        areas_for_improvement=areas_for_improvement,
        what_worked_well=what_worked_well,
        sections=sections,
    )


def _section_title(key: str) -> str:
    titles = {
        "executive_summary": "Executive Summary",
        "xg_flow": "xG Flow Analysis",
        "key_moments": "Key Moments",
        "player_ratings": "Player Ratings & Highlights",
        "tactical_observations": "Tactical Observations",
        "notable_patterns": "Notable Patterns",
        "set_pieces": "Set Piece Analysis",
        "phase_breakdown": "Phase Breakdown",
        "areas_for_improvement": "Areas for Improvement",
        "what_worked_well": "What Worked Well",
    }
    return titles.get(key, key.replace("_", " ").title())


def _truncate_content(content: str, detail_level: str) -> str:
    if detail_level == "brief":
        lines = content.strip().split("\n")
        if len(lines) > 3:
            return "\n".join(lines[:3]) + "\n*(truncated for brief mode)*"
    return content
