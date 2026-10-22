"""Match scripting — scripted match scenarios for training/analysis."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScriptedEvent:
    minute: float = 0.0
    event_type: str = "pass"
    team: str = "home"
    start_x: float = 50.0
    start_y: float = 34.0
    end_x: float = 60.0
    end_y: float = 34.0
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "minute": round(self.minute, 1),
            "event_type": self.event_type,
            "team": self.team,
            "start_x": round(self.start_x, 1),
            "start_y": round(self.start_y, 1),
            "end_x": round(self.end_x, 1),
            "end_y": round(self.end_y, 1),
            "attributes": self.attributes,
        }


@dataclass
class ScriptedPhase:
    name: str = ""
    start_minute: float = 0.0
    end_minute: float = 5.0
    team_focus: str = "both"
    intensity: float = 0.5  # 0 = slow, 1 = high press
    description: str = ""
    events: list[ScriptedEvent] = field(default_factory=list)
    coaching_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start_minute": round(self.start_minute, 1),
            "end_minute": round(self.end_minute, 1),
            "team_focus": self.team_focus,
            "intensity": round(self.intensity, 2),
            "description": self.description,
            "events": [e.to_dict() for e in self.events],
            "coaching_notes": self.coaching_notes,
        }


@dataclass
class MatchScript:
    title: str = ""
    home_team: str = "Home"
    away_team: str = "Away"
    home_formation: str = "4-3-3"
    away_formation: str = "4-4-2"
    phases: list[ScriptedPhase] = field(default_factory=list)
    final_score: tuple[int, int] = (0, 0)
    total_events: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_formation": self.home_formation,
            "away_formation": self.away_formation,
            "phases": [p.to_dict() for p in self.phases],
            "final_score": list(self.final_score),
            "total_events": self.total_events,
        }


def generate_possession_phase(
    team: str = "home",
    start_minute: float = 0.0,
    duration_minutes: float = 3.0,
    intensity: float = 0.5,
    attacking_direction: str = "right",
) -> ScriptedPhase:
    """Generate a possession phase with build-up and progression."""
    events = []
    base_x = 30 if team == "home" else 75
    goal_x = 105 if team == "home" else 0
    direction = 1 if attacking_direction == "right" else -1
    if team == "away":
        direction *= -1

    pitch_length = 105.0
    pitch_width = 68.0
    delta = direction * intensity * 15.0
    n_events = max(5, int(duration_minutes * 2))

    for i in range(n_events):
        minute = start_minute + (i / n_events) * duration_minutes
        progress = i / n_events
        is_last = i == n_events - 1

        sx = base_x + direction * progress * 40 + random.uniform(-5, 5)
        sy = 34 + random.uniform(-15, 15)

        if is_last:
            event_type = "shot"
            ex = goal_x + random.uniform(-5, 5)
            ey = 34 + random.uniform(-15, 15)
            completed = random.random() < 0.05
        else:
            event_type = "pass"
            ex = sx + delta + random.uniform(-5, 5)
            ey = sy + random.uniform(-10, 10)
            completed = random.random() < 0.85

        sx = max(0, min(pitch_length, sx))
        sy = max(0, min(pitch_width, sy))
        ex = max(0, min(pitch_length, ex))
        ey = max(0, min(pitch_width, ey))

        events.append(
            ScriptedEvent(
                minute=minute,
                event_type=event_type,
                team=team,
                start_x=sx,
                start_y=sy,
                end_x=ex,
                end_y=ey,
                attributes={"completed": completed, "intensity": round(intensity, 2)},
            )
        )

    return ScriptedPhase(
        name=f"{'Home' if team == 'home' else 'Away'} Possession",
        start_minute=start_minute,
        end_minute=start_minute + duration_minutes,
        team_focus=team,
        intensity=intensity,
        description="Patient possession phase with progressive build-up",
        events=events,
        coaching_notes="Focus on maintaining shape and creating passing lanes",
    )


def generate_pressing_phase(
    pressing_team: str = "home",
    start_minute: float = 0.0,
    duration_minutes: float = 2.0,
    intensity: float = 0.8,
) -> ScriptedPhase:
    """Generate a high-pressing phase."""
    events = []
    pitch_length = 105.0
    pitch_width = 68.0

    for i in range(int(duration_minutes * 3)):
        minute = start_minute + (i / (duration_minutes * 3)) * duration_minutes
        base_x = 70 if pressing_team == "home" else 35
        sx = base_x + random.uniform(-20, 20)
        sy = random.uniform(10, 58)
        ex = sx + random.uniform(-15, 15) * intensity
        ey = sy + random.uniform(-10, 10) * intensity

        events.append(
            ScriptedEvent(
                minute=minute,
                event_type="tackle" if random.random() < 0.3 else "run",
                team=pressing_team,
                start_x=max(0, min(pitch_length, sx)),
                start_y=max(0, min(pitch_width, sy)),
                end_x=max(0, min(pitch_length, ex)),
                end_y=max(0, min(pitch_width, ey)),
                attributes={"intensity": round(intensity, 2), "pressing": True},
            )
        )

    return ScriptedPhase(
        name=f"{'Home' if pressing_team == 'home' else 'Away'} High Press",
        start_minute=start_minute,
        end_minute=start_minute + duration_minutes,
        team_focus=pressing_team,
        intensity=intensity,
        description="High-intensity pressing phase forcing turnovers",
        events=events,
        coaching_notes="Coordinate pressing triggers; maintain compactness",
    )


def generate_counter_attack_phase(
    attacking_team: str = "home",
    start_minute: float = 0.0,
) -> ScriptedPhase:
    """Generate a fast counter-attack scenario."""
    events = []
    pitch_length = 105.0
    pitch_width = 68.0

    # Recover ball deep
    recovery_x = 15 if attacking_team == "home" else 90
    recovery_y = random.uniform(15, 53)

    events.append(
        ScriptedEvent(
            minute=start_minute,
            event_type="tackle",
            team=attacking_team,
            start_x=recovery_x,
            start_y=recovery_y,
            end_x=recovery_x + random.uniform(5, 15),
            end_y=recovery_y + random.uniform(-5, 5),
            attributes={"completed": True},
        )
    )

    # Quick vertical pass
    direction = 1 if attacking_team == "home" else -1
    events.append(
        ScriptedEvent(
            minute=start_minute + 0.1,
            event_type="pass",
            team=attacking_team,
            start_x=recovery_x,
            start_y=recovery_y,
            end_x=recovery_x + direction * 40 + random.uniform(-5, 5),
            end_y=recovery_y + random.uniform(-10, 10),
            attributes={"completed": True, "type": "through_ball"},
        )
    )

    # Run with ball / second pass
    mid_x = recovery_x + direction * 40
    mid_y = recovery_y
    events.append(
        ScriptedEvent(
            minute=start_minute + 0.3,
            event_type="pass" if random.random() < 0.5 else "carry",
            team=attacking_team,
            start_x=mid_x,
            start_y=mid_y,
            end_x=mid_x + direction * 25 + random.uniform(-3, 3),
            end_y=mid_y + random.uniform(-8, 8),
            attributes={"completed": True},
        )
    )

    # Shot
    shot_x = mid_x + direction * 25
    shot_y = mid_y + random.uniform(-10, 10)
    goal_x = 105 if attacking_team == "home" else 0
    events.append(
        ScriptedEvent(
            minute=start_minute + 0.5,
            event_type="shot",
            team=attacking_team,
            start_x=max(0, min(pitch_length, shot_x)),
            start_y=max(0, min(pitch_width, shot_y)),
            end_x=goal_x,
            end_y=34 + random.uniform(-15, 15),
            attributes={"is_goal": random.random() < 0.3},
        )
    )

    return ScriptedPhase(
        name="Counter Attack",
        start_minute=start_minute,
        end_minute=start_minute + 1.0,
        team_focus=attacking_team,
        intensity=1.0,
        description="Fast transition from defensive recovery to shot",
        events=events,
        coaching_notes="Speed of transition; decision-making in final third",
    )


def generate_match_script(
    template: str = "balanced",
    home_team: str = "Home",
    away_team: str = "Away",
) -> MatchScript:
    """Generate a full match script from a template.

    Templates: balanced, home_dominant, away_dominant, defensive, high_pressing, counter_attack
    """
    configs = {
        "balanced": {"home_poss": 3, "away_poss": 3, "press": 2, "counter": 1},
        "home_dominant": {"home_poss": 5, "away_poss": 1, "press": 3, "counter": 1},
        "away_dominant": {"home_poss": 1, "away_poss": 5, "press": 1, "counter": 3},
        "defensive": {"home_poss": 2, "away_poss": 2, "press": 4, "counter": 2},
        "high_pressing": {"home_poss": 2, "away_poss": 2, "press": 6, "counter": 1},
        "counter_attack": {"home_poss": 2, "away_poss": 3, "press": 2, "counter": 4},
    }
    cfg = configs.get(template, configs["balanced"])

    phases = []
    current_minute = 0.0

    for _ in range(cfg["home_poss"]):
        duration = random.uniform(2, 5)
        phases.append(
            generate_possession_phase(
                "home", current_minute, duration, random.uniform(0.3, 0.7), "right"
            )
        )
        current_minute += duration + random.uniform(0.5, 1.5)

    for _ in range(cfg["away_poss"]):
        duration = random.uniform(2, 5)
        phases.append(
            generate_possession_phase(
                "away", current_minute, duration, random.uniform(0.3, 0.7), "left"
            )
        )
        current_minute += duration + random.uniform(0.5, 1.5)

    for _ in range(cfg["press"]):
        duration = random.uniform(1.5, 3)
        team = "home" if random.random() < 0.5 else "away"
        phases.append(
            generate_pressing_phase(team, current_minute, duration, random.uniform(0.7, 1.0))
        )
        current_minute += duration + random.uniform(0.3, 1.0)

    for _ in range(cfg["counter"]):
        team = "home" if random.random() < 0.5 else "away"
        phases.append(generate_counter_attack_phase(team, current_minute))
        current_minute += 1.5 + random.uniform(0.5, 1.0)

    total_events = sum(len(p.events) for p in phases)
    return MatchScript(
        title=f"{template.replace('_', ' ').title()} Match",
        home_team=home_team,
        away_team=away_team,
        phases=sorted(phases, key=lambda p: p.start_minute),
        total_events=total_events,
    )
