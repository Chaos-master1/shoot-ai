"""Football Rules Service - encodes IFAB Laws of the Game.

Provides:
- 17 Laws of the Game as structured YAML
- Event classification (foul → free kick / penalty / yellow card)
- Offside detection
- Restart type lookup
- Human-readable law summaries

The Laws of the Game are maintained by IFAB (International Football Association
Board) and updated annually. The YAML file ships with v2024/2025 rules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from kawkab.core.game_constants import GAME
from kawkab.core.logging import get_logger

logger = get_logger(__name__)


class Law(Enum):
    FIELD = 1
    BALL = 2
    PLAYERS = 3
    EQUIPMENT = 4
    REFEREE = 5
    OTHER_MATCH = 6
    DURATION = 7
    START_RESTART = 8
    BALL_IN_OUT = 9
    DETERMINING_OUTCOME = 10
    OFFSIDE = 11
    FOULS = 12
    FREE_KICKS = 13
    PENALTY = 14
    THROW_IN = 15
    GOAL_KICK = 16
    CORNER = 17


class RestartType(Enum):
    GOAL_KICK = "goal_kick"
    CORNER_KICK = "corner_kick"
    THROW_IN = "throw_in"
    DIRECT_FREE_KICK = "direct_free_kick"
    INDIRECT_FREE_KICK = "indirect_free_kick"
    PENALTY_KICK = "penalty_kick"
    DROP_BALL = "drop_ball"
    KICK_OFF = "kick_off"


@dataclass
class RuleReference:
    law: int
    law_name: str
    restart: RestartType | None
    description: str
    card_likely: str  # "", "yellow", "red", "yellow_or_red"


@dataclass
class OffsideCheck:
    is_offside: bool
    attacker_track_id: int
    attacker_x: float
    second_last_defender_x: float
    ball_x: float
    explanation: str


class FootballRulesService:
    """Service for the 17 IFAB Laws of the Game.

    Loads rules from `knowledge/rules/laws_of_the_game.yaml` and provides
    helpers for event classification, offside detection, and restart lookup.
    """

    PITCH_LENGTH = GAME.PITCH_LENGTH_M  # meters (IFAB standard)
    PITCH_WIDTH = GAME.PITCH_WIDTH_M
    PENALTY_AREA_DEPTH = 16.5
    PENALTY_AREA_WIDTH = 40.32
    GOAL_AREA_DEPTH = 5.5
    GOAL_AREA_WIDTH = 18.32
    PENALTY_SPOT_DISTANCE = 11.0

    def __init__(self, rules_yaml_path: str | None = None) -> None:
        self._laws: dict[int, dict[str, Any]] = {}
        if rules_yaml_path is None:
            rules_yaml_path = self._default_rules_path()
        self._load_rules(rules_yaml_path)

    def _default_rules_path(self) -> str:
        paths_to_try = [
            Path(__file__).parent.parent / "knowledge" / "rules" / "laws_of_the_game.yaml",
            Path.cwd() / "src" / "kawkab" / "knowledge" / "rules" / "laws_of_the_game.yaml",
            Path.cwd() / "kawkab" / "knowledge" / "rules" / "laws_of_the_game.yaml",
        ]
        for p in paths_to_try:
            if p.exists():
                return str(p)
        return ""

    def _load_rules(self, path: str) -> None:
        if not path or not os.path.exists(path):
            logger.warning(f"Rules YAML not found: {path}; using embedded defaults")
            self._laws = self._embedded_defaults()
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            self._laws = {
                int(law["number"]): law for law in data.get("laws", []) if "number" in law
            }
            logger.info(f"Loaded {len(self._laws)} laws from {path}")
        except Exception as e:
            logger.warning(f"Failed to load rules YAML: {e}; using defaults")
            self._laws = self._embedded_defaults()

    def _embedded_defaults(self) -> dict[int, dict[str, Any]]:
        """Minimal law summaries used when YAML is missing."""
        return {
            1: {
                "number": 1,
                "name": "The Field of Play",
                "summary": "Rectangular pitch, 100-110m x 64-75m",
            },
            2: {
                "number": 2,
                "name": "The Ball",
                "summary": "Spherical, 68-70cm circumference, 410-450g",
            },
            3: {
                "number": 3,
                "name": "The Players",
                "summary": "11 per team, match may not start/end with fewer than 7",
            },
            4: {
                "number": 4,
                "name": "The Players' Equipment",
                "summary": "Kit, boots, shin guards, no jewelry",
            },
            5: {
                "number": 5,
                "name": "The Referee",
                "summary": "Enforces Laws, controls match, has final decision",
            },
            6: {
                "number": 6,
                "name": "The Other Match Officials",
                "summary": "Two assistant referees, fourth official, VAR",
            },
            7: {
                "number": 7,
                "name": "The Duration of the Match",
                "summary": "Two 45-min halves, 15-min break",
            },
            8: {
                "number": 8,
                "name": "The Start and Restart of Play",
                "summary": "Kick-off at center, dropped ball for stoppages",
            },
            9: {
                "number": 9,
                "name": "The Ball In and Out of Play",
                "summary": "Out when fully over goal line or touchline",
            },
            10: {
                "number": 10,
                "name": "Determining the Outcome",
                "summary": "Goal when ball fully crosses goal line",
            },
            11: {
                "number": 11,
                "name": "Offside",
                "summary": "Offside if nearer to opponents' goal line than ball and second-last opponent",
            },
            12: {
                "number": 12,
                "name": "Fouls and Misconduct",
                "summary": "Direct free kick + caution for careless, reckless, excessive force tackles",
            },
            13: {
                "number": 13,
                "name": "Free Kicks",
                "summary": "Direct (can score) and indirect (need second touch)",
            },
            14: {
                "number": 14,
                "name": "The Penalty Kick",
                "summary": "Awarded for fouls in penalty area; from 11m spot",
            },
            15: {
                "number": 15,
                "name": "The Throw-In",
                "summary": "Restart when ball crosses touchline; both hands from behind head",
            },
            16: {
                "number": 16,
                "name": "The Goal Kick",
                "summary": "Restart when ball crosses goal line last touched by attacker",
            },
            17: {
                "number": 17,
                "name": "The Corner Kick",
                "summary": "Restart when ball crosses goal line last touched by defender",
            },
        }

    @property
    def available(self) -> bool:
        return len(self._laws) >= 17

    def get_law_summary(self, law_number: int) -> dict[str, Any]:
        return self._laws.get(law_number, {})

    def get_all_laws(self) -> list[dict[str, Any]]:
        return sorted(self._laws.values(), key=lambda x: x.get("number", 0))

    def classify_event(
        self,
        event_type: str,
        location_x: float,
        location_y: float,
        side: str = "home",
        pitch_length: float = PITCH_LENGTH,
        pitch_width: float = PITCH_WIDTH,
    ) -> RuleReference:
        """Classify an event according to the Laws of the Game.

        Args:
            event_type: 'foul', 'ball_out', 'handball', 'offside', etc.
            location_x: 0..pitch_length (0 = home goal line, pitch_length = away goal line)
            location_y: 0..pitch_width
            side: 'home' or 'away' (which team the event is in favor of)
        """
        et = event_type.lower().strip()
        if et in {"foul", "tackle", "challenge", "trip", "push"}:
            return self._classify_foul(location_x, location_y, side, pitch_length)
        if et in {"ball_out", "out_of_play", "out"}:
            return self._classify_ball_out(location_x, location_y, side, pitch_length, pitch_width)
        if et in {"handball"}:
            return self._classify_handball(location_x, location_y, side, pitch_length)
        if et == "offside":
            return RuleReference(
                law=11,
                law_name="Offside",
                restart=RestartType.INDIRECT_FREE_KICK,
                description="Offside offense results in indirect free kick for opposing team from where offense occurred.",
                card_likely="yellow",
            )
        if et in {"goal", "score"}:
            return RuleReference(
                law=10,
                law_name="Determining the Outcome",
                restart=None,
                description="Goal scored when whole ball crosses goal line between posts and under crossbar.",
                card_likely="",
            )
        return RuleReference(
            law=0,
            law_name="Unknown",
            restart=None,
            description=f"Event type '{event_type}' not classified by rules service.",
            card_likely="",
        )

    def _classify_foul(self, x: float, y: float, side: str, pitch_length: float) -> RuleReference:
        if side == "home":
            penalty_x = pitch_length - self.PENALTY_AREA_DEPTH
        else:
            penalty_x = self.PENALTY_AREA_DEPTH
        in_penalty_area = abs(x - penalty_x) < self.PENALTY_AREA_DEPTH
        if in_penalty_area:
            return RuleReference(
                law=14,
                law_name="The Penalty Kick",
                restart=RestartType.PENALTY_KICK,
                description=f"Foul by {('defending' if in_penalty_area else 'attacking')} team inside penalty area → penalty kick.",
                card_likely="yellow_or_red",
            )
        return RuleReference(
            law=12,
            law_name="Fouls and Misconduct",
            restart=RestartType.DIRECT_FREE_KICK,
            description="Direct free kick awarded for foul outside penalty area.",
            card_likely="yellow",
        )

    def _classify_ball_out(
        self, x: float, y: float, side: str, pitch_length: float, pitch_width: float
    ) -> RuleReference:
        margin = 2.0
        on_touchline = y < margin or y > pitch_width - margin
        on_goal_line = x < margin or x > pitch_length - margin
        if on_touchline:
            return RuleReference(
                law=15,
                law_name="The Throw-In",
                restart=RestartType.THROW_IN,
                description="Ball out over touchline → throw-in to opposing team from where it crossed.",
                card_likely="",
            )
        if on_goal_line:
            last_touch = side
            if last_touch == "away":
                return RuleReference(
                    law=16,
                    law_name="The Goal Kick",
                    restart=RestartType.GOAL_KICK,
                    description="Ball out over home goal line last touched by away team → goal kick for home.",
                    card_likely="",
                )
            return RuleReference(
                law=17,
                law_name="The Corner Kick",
                restart=RestartType.CORNER_KICK,
                description="Ball out over home goal line last touched by home team → corner kick for away.",
                card_likely="",
            )
        return RuleReference(
            law=9,
            law_name="The Ball In and Out of Play",
            restart=None,
            description="Ball position ambiguous; not clearly over a line.",
            card_likely="",
        )

    def _classify_handball(
        self, x: float, y: float, side: str, pitch_length: float
    ) -> RuleReference:
        if side == "home":
            penalty_x = pitch_length - self.PENALTY_AREA_DEPTH
        else:
            penalty_x = self.PENALTY_AREA_DEPTH
        in_penalty_area = abs(x - penalty_x) < self.PENALTY_AREA_DEPTH
        if in_penalty_area:
            return RuleReference(
                law=14,
                law_name="The Penalty Kick",
                restart=RestartType.PENALTY_KICK,
                description="Deliberate handball in own penalty area → penalty kick + caution for offender.",
                card_likely="yellow_or_red",
            )
        return RuleReference(
            law=12,
            law_name="Fouls and Misconduct",
            restart=RestartType.DIRECT_FREE_KICK,
            description="Deliberate handball → direct free kick for opposing team.",
            card_likely="yellow",
        )

    def is_offside(
        self,
        attacker_x: float,
        second_last_defender_x: float,
        ball_x: float,
        attacking_direction: str = "right",
    ) -> OffsideCheck:
        """Check IFAB offside Law 11.

        An attacker is offside if nearer to the opponents' goal line than BOTH
        the ball AND the second-last opponent. The goalkeeper is usually the
        last opponent; the second-last is the next defender back.

        Args:
            attacker_x: pitch x position of attacker (0=home goal, PITCH_LENGTH=away goal)
            second_last_defender_x: x position of second-last defender
            ball_x: x position of the ball
            attacking_direction: 'right' (attacking toward x=pitch_length) or 'left'

        Returns:
            OffsideCheck with is_offside, positions, and explanation
        """
        if attacking_direction == "right":
            attacker_ahead = attacker_x > second_last_defender_x
            ball_ahead = attacker_x > ball_x
        else:
            attacker_ahead = attacker_x < second_last_defender_x
            ball_ahead = attacker_x < ball_x
        is_offside = attacker_ahead and ball_ahead
        if is_offside:
            explanation = (
                f"Attacker at x={attacker_x:.1f} is beyond both "
                f"second-last defender (x={second_last_defender_x:.1f}) and "
                f"ball (x={ball_x:.1f}). Offside position — IFAB Law 11."
            )
        else:
            explanation = (
                f"Attacker at x={attacker_x:.1f} is not beyond both defender and ball. "
                f"Onside — IFAB Law 11."
            )
        return OffsideCheck(
            is_offside=is_offside,
            attacker_track_id=0,
            attacker_x=attacker_x,
            second_last_defender_x=second_last_defender_x,
            ball_x=ball_x,
            explanation=explanation,
        )

    def get_restart_for_event(
        self,
        event_type: str,
        location_x: float = 0.0,
        location_y: float = 0.0,
        side: str = "home",
    ) -> RestartType:
        ref = self.classify_event(event_type, location_x, location_y, side)
        return ref.restart or RestartType.DROP_BALL
