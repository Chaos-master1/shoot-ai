"""Psychology Service - momentum, score-state, and performance psychology.

Tracks psychological factors that influence match performance:
- Score-state transitions (drawing → winning → losing)
- Momentum in 5-min rolling windows (xg diff, possession, shot diff)
- Post-goal regression (research: teams that score often play worse for 5-10 min)
- Late-game psychology (time-wasting when winning 75-90 min)
- Comeback momentum (team scoring after going behind)
- Capitulation (3+ goals conceded in 15 min window)

Uses score events + xG events + possession data to derive insights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


class ScoreState(Enum):
    DRAWING = "drawing"
    WINNING_BY_1 = "winning_by_1"
    WINNING_BY_2_PLUS = "winning_by_2_plus"
    LOSING_BY_1 = "losing_by_1"
    LOSING_BY_2_PLUS = "losing_by_2_plus"


class PsychologyEventType(Enum):
    SCORE_STATE_CHANGE = "score_state_change"
    POST_GOAL_LULL = "post_goal_lull"
    POST_GOAL_BOOST = "post_goal_boost"
    LATE_GAME_DECLINE = "late_game_decline"
    COMEBACK = "comeback"
    CAPITULATION = "capitulation"
    MOMENTUM_SHIFT = "momentum_shift"
    LATE_GAME_TIME_WASTING = "late_game_time_wasting"


@dataclass
class ScoreStateTransition:
    minute: int
    second: int
    team: str
    from_state: ScoreState
    to_state: ScoreState
    trigger_event: str  # "goal", "opponent_goal"


@dataclass
class MomentumPoint:
    minute: float
    home_momentum: float  # -1.0 to 1.0
    away_momentum: float


@dataclass
class PsychologyEvent:
    event_type: PsychologyEventType
    minute: int
    second: int
    team: str
    description: str
    severity: float  # 0.0 to 1.0
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class PsychologyReport:
    match_id: int
    home_team: str
    away_team: str
    score_state_transitions: list[ScoreStateTransition]
    momentum_timeline: list[MomentumPoint]
    psychology_events: list[PsychologyEvent]
    post_goal_lull_count: int
    comeback_count: int
    capitulation_count: int
    avg_late_game_passing_drop: float
    notes: list[str]


class PsychologyService:
    """Multi-faceted match psychology analysis.

    Uses score events, xG events, possession data, and timestamps to
    derive insights about how teams' psychological state affects play.
    """

    POST_GOAL_LULL_WINDOW_S = 600
    MOMENTUM_WINDOW_S = 300
    LATE_GAME_START_MIN = 75
    LATE_GAME_END_MIN = 90
    CAPITULATION_WINDOW_S = 900
    CAPITULATION_GOAL_THRESHOLD = 3

    def __init__(self, llm_service: Any | None = None) -> None:
        self.llm_service = llm_service
        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Main analysis entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        match_id: int,
        home_team: str,
        away_team: str,
        events: list[dict[str, Any]],
    ) -> PsychologyReport:
        """Run full psychology analysis on a match's events.

        Args:
            match_id: match identifier
            home_team, away_team: team names
            events: list of match events with structure:
                {
                    'minute': int, 'second': int, 'type': 'goal'|'shot'|'foul'|'pass'|...,
                    'team': 'home'|'away', 'xg': float (for shots), 'completed': bool (for passes)
                }
        """
        score_state_transitions = self._track_score_states(events, home_team)
        momentum_timeline = self._compute_momentum_timeline(events)
        psychology_events: list[PsychologyEvent] = []
        post_goal_events = self._detect_post_goal_effects(events)
        psychology_events.extend(post_goal_events)
        late_game_events = self._detect_late_game_psychology(events)
        psychology_events.extend(late_game_events)
        comeback_events = self._detect_comebacks(score_state_transitions, events)
        psychology_events.extend(comeback_events)
        capitulation_events = self._detect_capitulation(events)
        psychology_events.extend(capitulation_events)
        psychology_events.sort(key=lambda e: (e.minute, e.second))
        post_goal_lull_count = sum(
            1 for e in psychology_events if e.event_type == PsychologyEventType.POST_GOAL_LULL
        )
        comeback_count = sum(
            1 for e in psychology_events if e.event_type == PsychologyEventType.COMEBACK
        )
        capitulation_count = sum(
            1 for e in psychology_events if e.event_type == PsychologyEventType.CAPITULATION
        )
        avg_late_game_passing_drop = self._compute_late_game_passing_drop(events)
        notes = self._generate_notes(
            post_goal_lull_count,
            comeback_count,
            capitulation_count,
            avg_late_game_passing_drop,
            score_state_transitions,
        )
        return PsychologyReport(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            score_state_transitions=score_state_transitions,
            momentum_timeline=momentum_timeline,
            psychology_events=psychology_events,
            post_goal_lull_count=post_goal_lull_count,
            comeback_count=comeback_count,
            capitulation_count=capitulation_count,
            avg_late_game_passing_drop=avg_late_game_passing_drop,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Score state tracking
    # ------------------------------------------------------------------

    def _track_score_states(
        self, events: list[dict[str, Any]], home_team: str
    ) -> list[ScoreStateTransition]:
        """Track how the score state changes over the match."""
        transitions: list[ScoreStateTransition] = []
        home_goals = 0
        away_goals = 0
        prev_state = self._score_state(home_goals, away_goals)
        for event in events:
            if event.get("type") != "goal":
                continue
            team = event.get("team", "")
            minute = event.get("minute", 0)
            second = event.get("second", 0)
            if team == "home":
                home_goals += 1
            elif team == "away":
                away_goals += 1
            new_state = self._score_state(home_goals, away_goals)
            if new_state != prev_state:
                transitions.append(
                    ScoreStateTransition(
                        minute=minute,
                        second=second,
                        team=team,
                        from_state=prev_state,
                        to_state=new_state,
                        trigger_event="goal",
                    )
                )
                prev_state = new_state
        return transitions

    @staticmethod
    def _score_state(home_goals: int, away_goals: int) -> ScoreState:
        diff = home_goals - away_goals
        if diff == 0:
            return ScoreState.DRAWING
        if diff == 1:
            return ScoreState.WINNING_BY_1
        if diff > 1:
            return ScoreState.WINNING_BY_2_PLUS
        if diff == -1:
            return ScoreState.LOSING_BY_1
        return ScoreState.LOSING_BY_2_PLUS

    # ------------------------------------------------------------------
    # Momentum timeline
    # ------------------------------------------------------------------

    def _compute_momentum_timeline(self, events: list[dict[str, Any]]) -> list[MomentumPoint]:
        """Compute rolling momentum in 5-min windows.

        Momentum formula:
            M = 0.4 * xg_diff + 0.3 * pos_diff + 0.2 * shot_diff + 0.1 * foul_diff
        """
        max_time_s = 90 * 60
        if not events:
            return []
        timeline: list[MomentumPoint] = []
        window_starts = range(0, int(max_time_s) + 1, 60)
        for ws in window_starts:
            we = ws + self.MOMENTUM_WINDOW_S
            window_events = [e for e in events if ws <= self._event_time_s(e) < we]
            if not window_events:
                timeline.append(
                    MomentumPoint(
                        minute=ws / 60.0,
                        home_momentum=0.0,
                        away_momentum=0.0,
                    )
                )
                continue
            home_xg = sum(e.get("xg", 0) for e in window_events if e.get("team") == "home")
            away_xg = sum(e.get("xg", 0) for e in window_events if e.get("team") == "away")
            xg_diff = home_xg - away_xg
            home_shots = sum(
                1 for e in window_events if e.get("type") == "shot" and e.get("team") == "home"
            )
            away_shots = sum(
                1 for e in window_events if e.get("type") == "shot" and e.get("team") == "away"
            )
            shot_diff = home_shots - away_shots
            home_pos = sum(
                1
                for e in window_events
                if e.get("type") == "pass" and e.get("team") == "home" and e.get("completed")
            )
            total_home_passes = sum(
                1 for e in window_events if e.get("type") == "pass" and e.get("team") == "home"
            )
            away_pos = sum(
                1
                for e in window_events
                if e.get("type") == "pass" and e.get("team") == "away" and e.get("completed")
            )
            total_away_passes = sum(
                1 for e in window_events if e.get("type") == "pass" and e.get("team") == "away"
            )
            home_pos_pct = home_pos / max(1, total_home_passes)
            away_pos_pct = away_pos / max(1, total_away_passes)
            pos_diff = home_pos_pct - away_pos_pct
            home_fouls = sum(
                1
                for e in window_events
                if e.get("type") in {"foul", "tackle"} and e.get("team") == "home"
            )
            away_fouls = sum(
                1
                for e in window_events
                if e.get("type") in {"foul", "tackle"} and e.get("team") == "away"
            )
            foul_diff = away_fouls - home_fouls
            home_momentum = 0.4 * xg_diff + 0.3 * pos_diff + 0.2 * shot_diff + 0.1 * foul_diff
            home_momentum = max(-1.0, min(1.0, home_momentum))
            timeline.append(
                MomentumPoint(
                    minute=ws / 60.0,
                    home_momentum=home_momentum,
                    away_momentum=-home_momentum,
                )
            )
        return timeline

    # ------------------------------------------------------------------
    # Post-goal effects
    # ------------------------------------------------------------------

    def _detect_post_goal_effects(self, events: list[dict[str, Any]]) -> list[PsychologyEvent]:
        """Detect post-goal lulls (regression) and boosts.

        For each goal, compare team performance in 10-min window after goal
        to overall match average. Significant drop = lull; significant rise = boost.
        """
        lulls: list[PsychologyEvent] = []
        team_passes_overall: dict[str, list[bool]] = {"home": [], "away": []}
        for e in events:
            if e.get("type") == "pass" and "completed" in e:
                team = e.get("team", "")
                if team in team_passes_overall:
                    team_passes_overall[team].append(e["completed"])
        overall_pass_rates: dict[str, float] = {
            team: (sum(passes) / len(passes)) if passes else 0.5
            for team, passes in team_passes_overall.items()
        }
        for i, event in enumerate(events):
            if event.get("type") != "goal":
                continue
            team = event.get("team", "")
            if team not in overall_pass_rates:
                continue
            t0 = self._event_time_s(event)
            after_passes: list[bool] = []
            for later in events[i + 1 :]:
                if self._event_time_s(later) - t0 > self.POST_GOAL_LULL_WINDOW_S:
                    break
                if (
                    later.get("type") == "pass"
                    and later.get("team") == team
                    and "completed" in later
                ):
                    after_passes.append(later["completed"])
            if len(after_passes) < 5:
                continue
            after_rate = sum(after_passes) / len(after_passes)
            drop = overall_pass_rates[team] - after_rate
            if drop > 0.12:
                lulls.append(
                    PsychologyEvent(
                        event_type=PsychologyEventType.POST_GOAL_LULL,
                        minute=event.get("minute", 0),
                        second=event.get("second", 0),
                        team=team,
                        description=(
                            f"Post-goal lull for {team}: passing accuracy dropped "
                            f"{drop * 100:.1f}% in next 10 min after scoring"
                        ),
                        severity=min(1.0, drop * 5),
                        data={"drop_pct": drop, "window_s": self.POST_GOAL_LULL_WINDOW_S},
                    )
                )
            elif drop < -0.08:
                lulls.append(
                    PsychologyEvent(
                        event_type=PsychologyEventType.POST_GOAL_BOOST,
                        minute=event.get("minute", 0),
                        second=event.get("second", 0),
                        team=team,
                        description=(
                            f"Post-goal boost for {team}: passing accuracy rose "
                            f"{-drop * 100:.1f}% in next 10 min after scoring"
                        ),
                        severity=min(1.0, -drop * 5),
                        data={"boost_pct": -drop, "window_s": self.POST_GOAL_LULL_WINDOW_S},
                    )
                )
        return lulls

    # ------------------------------------------------------------------
    # Late game psychology
    # ------------------------------------------------------------------

    def _detect_late_game_psychology(self, events: list[dict[str, Any]]) -> list[PsychologyEvent]:
        """Detect late-game psychology: time-wasting, declining focus.

        Heuristics:
        - In 75-90 min with a team leading: drop in passing accuracy,
          more fouls (time-wasting), fewer shots
        """
        late_events: list[PsychologyEvent] = []
        late_game = [
            e
            for e in events
            if self.LATE_GAME_START_MIN <= e.get("minute", 0) <= self.LATE_GAME_END_MIN
        ]
        if not late_game:
            return late_events
        for team in ("home", "away"):
            team_late = [e for e in late_game if e.get("team") == team]
            if not team_late:
                continue
            passes = [e for e in team_late if e.get("type") == "pass" and "completed" in e]
            if not passes:
                continue
            late_pass_rate = sum(p["completed"] for p in passes) / len(passes)
            early_events = [
                e
                for e in events
                if e.get("minute", 0) < 60
                and e.get("team") == team
                and e.get("type") == "pass"
                and "completed" in e
            ]
            if not early_events:
                continue
            early_pass_rate = sum(e["completed"] for e in early_events) / len(early_events)
            drop = early_pass_rate - late_pass_rate
            if drop > 0.10:
                late_events.append(
                    PsychologyEvent(
                        event_type=PsychologyEventType.LATE_GAME_DECLINE,
                        minute=75,
                        second=0,
                        team=team,
                        description=(
                            f"Late-game decline for {team}: passing accuracy dropped "
                            f"{drop * 100:.1f}% in final 15 min"
                        ),
                        severity=min(1.0, drop * 5),
                        data={"drop_pct": drop},
                    )
                )
            fouls = [e for e in team_late if e.get("type") in {"foul", "tackle"}]
            if len(fouls) >= 3 and drop > 0.05:
                late_events.append(
                    PsychologyEvent(
                        event_type=PsychologyEventType.LATE_GAME_TIME_WASTING,
                        minute=80,
                        second=0,
                        team=team,
                        description=(
                            f"Late-game time-wasting pattern: {len(fouls)} fouls in final 15 min"
                        ),
                        severity=min(1.0, len(fouls) / 5),
                        data={"fouls_count": len(fouls)},
                    )
                )
        return late_events

    # ------------------------------------------------------------------
    # Comeback detection
    # ------------------------------------------------------------------

    def _detect_comebacks(
        self, transitions: list[ScoreStateTransition], events: list[dict[str, Any]]
    ) -> list[PsychologyEvent]:
        """Detect comeback moments: team scores after being 1+ goal down."""
        comebacks: list[PsychologyEvent] = []
        for tr in transitions:
            was_behind = tr.from_state in {ScoreState.LOSING_BY_1, ScoreState.LOSING_BY_2_PLUS}
            became_drawing_or_winning = tr.to_state in {
                ScoreState.DRAWING,
                ScoreState.WINNING_BY_1,
                ScoreState.WINNING_BY_2_PLUS,
            }
            if was_behind and became_drawing_or_winning:
                comebacks.append(
                    PsychologyEvent(
                        event_type=PsychologyEventType.COMEBACK,
                        minute=tr.minute,
                        second=tr.second,
                        team=tr.team,
                        description=(
                            f"Comeback by {tr.team}: scored after being behind, "
                            f"moved from {tr.from_state.value} to {tr.to_state.value}"
                        ),
                        severity=0.8,
                    )
                )
        return comebacks

    # ------------------------------------------------------------------
    # Capitulation detection
    # ------------------------------------------------------------------

    def _detect_capitulation(self, events: list[dict[str, Any]]) -> list[PsychologyEvent]:
        """Detect capitulation: 3+ goals conceded in 15-min window."""
        capitulations: list[PsychologyEvent] = []
        for i, event in enumerate(events):
            if event.get("type") != "goal":
                continue
            t0 = self._event_time_s(event)
            losing_team = event.get("team", "")
            if losing_team == "home":
                losing_team = "away"
            elif losing_team == "away":
                losing_team = "home"
            else:
                continue
            later_goals: list[dict[str, Any]] = []
            for later in events[i + 1 :]:
                if self._event_time_s(later) - t0 > self.CAPITULATION_WINDOW_S:
                    break
                if later.get("type") == "goal" and later.get("team") == losing_team:
                    later_goals.append(later)
            if len(later_goals) >= 2:
                capitulations.append(
                    PsychologyEvent(
                        event_type=PsychologyEventType.CAPITULATION,
                        minute=event.get("minute", 0),
                        second=event.get("second", 0),
                        team=losing_team,
                        description=(
                            f"Capitulation pattern: {losing_team} conceded "
                            f"{len(later_goals)} additional goals within 15 min"
                        ),
                        severity=min(1.0, len(later_goals) / 3),
                        data={"conceded_count": len(later_goals)},
                    )
                )
        return capitulations

    # ------------------------------------------------------------------
    # Late game passing drop summary
    # ------------------------------------------------------------------

    def _compute_late_game_passing_drop(self, events: list[dict[str, Any]]) -> float:
        drops: list[float] = []
        for team in ("home", "away"):
            early = [
                e
                for e in events
                if e.get("minute", 0) < 60
                and e.get("team") == team
                and e.get("type") == "pass"
                and "completed" in e
            ]
            late = [
                e
                for e in events
                if self.LATE_GAME_START_MIN <= e.get("minute", 0) <= self.LATE_GAME_END_MIN
                and e.get("team") == team
                and e.get("type") == "pass"
                and "completed" in e
            ]
            if early and late:
                early_rate = sum(e["completed"] for e in early) / len(early)
                late_rate = sum(e["completed"] for e in late) / len(late)
                drops.append(max(0.0, early_rate - late_rate))
        return sum(drops) / len(drops) if drops else 0.0

    def _generate_notes(
        self,
        post_goal_lull_count: int,
        comeback_count: int,
        capitulation_count: int,
        avg_late_game_drop: float,
        transitions: list[ScoreStateTransition],
    ) -> list[str]:
        notes: list[str] = []
        if post_goal_lull_count >= 2:
            notes.append(
                f"{post_goal_lull_count} post-goal lulls detected — "
                "team(s) showed regression after scoring (psychology research supports this)"
            )
        if comeback_count > 0:
            notes.append(f"{comeback_count} comeback(s) detected — strong mental resilience shown")
        if capitulation_count > 0:
            notes.append(
                f"{capitulation_count} capitulation pattern(s) detected — "
                "team(s) lost focus after conceding"
            )
        if avg_late_game_drop > 0.1:
            notes.append(
                f"Average late-game passing drop: {avg_late_game_drop * 100:.1f}% — "
                "typical time-management behavior when leading"
            )
        if not notes:
            notes.append("No significant psychological patterns detected")
        return notes

    @staticmethod
    def _event_time_s(event: dict[str, Any]) -> float:
        return float(event.get("minute", 0)) * 60.0 + float(event.get("second", 0))
