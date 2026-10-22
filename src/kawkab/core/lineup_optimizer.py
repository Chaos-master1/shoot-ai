"""Lineup optimizer — suggests optimal player placement based on formation,
player roles, and opponent analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kawkab.core.game_constants import GAME

try:
    from ortools.sat.python import cp_model

    _ORTOOLS_AVAILABLE = True
except ImportError:
    _ORTOOLS_AVAILABLE = False

PITCH_LENGTH = GAME.PITCH_LENGTH_M
PITCH_WIDTH = GAME.PITCH_WIDTH_M

# Standard formation templates: list of (position_name, x_ratio, y_ratio)
# x_ratio: 0.0 = own goal line, 1.0 = opponent goal line
# y_ratio: 0.0 = left touchline, 1.0 = right touchline (as viewed from above)

FORMATION_TEMPLATES: dict[str, list[tuple[str, float, float]]] = {
    "4-4-2": [
        ("GK", 0.05, 0.5),
        ("LB", 0.25, 0.15),
        ("CB", 0.25, 0.38),
        ("CB", 0.25, 0.62),
        ("RB", 0.25, 0.85),
        ("LM", 0.50, 0.15),
        ("CM", 0.50, 0.38),
        ("CM", 0.50, 0.62),
        ("RM", 0.50, 0.85),
        ("ST", 0.75, 0.35),
        ("ST", 0.75, 0.65),
    ],
    "4-3-3": [
        ("GK", 0.05, 0.5),
        ("LB", 0.25, 0.15),
        ("CB", 0.25, 0.38),
        ("CB", 0.25, 0.62),
        ("RB", 0.25, 0.85),
        ("CM", 0.55, 0.25),
        ("CDM", 0.45, 0.5),
        ("CM", 0.55, 0.75),
        ("LW", 0.80, 0.15),
        ("ST", 0.80, 0.5),
        ("RW", 0.80, 0.85),
    ],
    "3-5-2": [
        ("GK", 0.05, 0.5),
        ("CB", 0.20, 0.25),
        ("CB", 0.20, 0.5),
        ("CB", 0.20, 0.75),
        ("LWB", 0.45, 0.08),
        ("CM", 0.50, 0.3),
        ("CDM", 0.45, 0.5),
        ("CM", 0.50, 0.7),
        ("RWB", 0.45, 0.92),
        ("ST", 0.75, 0.35),
        ("ST", 0.75, 0.65),
    ],
    "4-2-3-1": [
        ("GK", 0.05, 0.5),
        ("LB", 0.25, 0.15),
        ("CB", 0.25, 0.38),
        ("CB", 0.25, 0.62),
        ("RB", 0.25, 0.85),
        ("CDM", 0.40, 0.35),
        ("CDM", 0.40, 0.65),
        ("LW", 0.70, 0.15),
        ("CAM", 0.65, 0.5),
        ("RW", 0.70, 0.85),
        ("ST", 0.80, 0.5),
    ],
    "3-4-3": [
        ("GK", 0.05, 0.5),
        ("CB", 0.20, 0.3),
        ("CB", 0.20, 0.5),
        ("CB", 0.20, 0.7),
        ("LM", 0.50, 0.12),
        ("CM", 0.50, 0.35),
        ("CM", 0.50, 0.65),
        ("RM", 0.50, 0.88),
        ("LW", 0.80, 0.15),
        ("ST", 0.80, 0.5),
        ("RW", 0.80, 0.85),
    ],
    "5-3-2": [
        ("GK", 0.05, 0.5),
        ("CB", 0.18, 0.15),
        ("CB", 0.18, 0.38),
        ("CB", 0.18, 0.5),
        ("CB", 0.18, 0.62),
        ("CB", 0.18, 0.85),
        ("CM", 0.45, 0.25),
        ("CM", 0.45, 0.5),
        ("CM", 0.45, 0.75),
        ("ST", 0.75, 0.35),
        ("ST", 0.75, 0.65),
    ],
    "4-1-4-1": [
        ("GK", 0.05, 0.5),
        ("LB", 0.25, 0.15),
        ("CB", 0.25, 0.38),
        ("CB", 0.25, 0.62),
        ("RB", 0.25, 0.85),
        ("CDM", 0.40, 0.5),
        ("LM", 0.60, 0.12),
        ("CM", 0.60, 0.35),
        ("CM", 0.60, 0.65),
        ("RM", 0.60, 0.88),
        ("ST", 0.85, 0.5),
    ],
}

# Position role groupings for player-slot matching
POSITION_ROLES: dict[str, list[str]] = {
    "GK": ["goalkeeper", "gk", "keeper"],
    "CB": ["centre_back", "center_back", "cb", "defender_central", "central_defender"],
    "LB": ["left_back", "lb", "full_back", "fullback"],
    "RB": ["right_back", "rb", "full_back", "fullback"],
    "LWB": ["left_wing_back", "lwb", "wing_back", "wingback"],
    "RWB": ["right_wing_back", "rwb", "wing_back", "wingback"],
    "CDM": ["defensive_midfielder", "cdm", "holding_midfielder", "defensive_mid"],
    "CM": ["centre_midfielder", "central_midfielder", "cm", "midfielder_central"],
    "CAM": ["attacking_midfielder", "cam", "playmaker", "attacking_mid"],
    "LM": ["left_midfielder", "lm", "left_mid"],
    "RM": ["right_midfielder", "rm", "right_mid"],
    "LW": ["left_winger", "lw", "left_wing", "winger"],
    "RW": ["right_winger", "rw", "right_wing", "winger"],
    "ST": ["striker", "st", "forward", "centre_forward", "cf", "attacker"],
}


@dataclass
class PlayerSlot:
    position_name: str
    x: float
    y: float
    role: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position_name,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "role": self.role,
        }


@dataclass
class LineupSuggestion:
    formation: str
    slots: list[PlayerSlot] = field(default_factory=list)
    description: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "formation": self.formation,
            "slots": [s.to_dict() for s in self.slots],
            "description": self.description,
            "confidence": round(self.confidence, 2),
        }


@dataclass
class OptimizerResult:
    suggestions: list[LineupSuggestion] = field(default_factory=list)
    best_formation: str = ""
    best_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "suggestions": [s.to_dict() for s in self.suggestions],
            "best_formation": self.best_formation,
            "best_confidence": round(self.best_confidence, 2),
        }


class LineupOptimizer:
    """Suggests optimal lineups based on formations, player roles, and match context.

    Usage:
        opt = LineupOptimizer()
        suggestion = opt.suggest_lineup(formation="4-4-2", players=player_list)
        comparison = opt.compare_formations(["4-4-2", "4-3-3", "4-2-3-1"], opponent="3-5-2")
    """

    PITCH_LENGTH = GAME.PITCH_LENGTH_M
    PITCH_WIDTH = GAME.PITCH_WIDTH_M

    def __init__(self) -> None:
        self._templates = FORMATION_TEMPLATES
        self._roles = POSITION_ROLES

    def suggest_lineup(
        self,
        formation: str = "4-4-2",
        players: list[dict[str, Any]] | None = None,
        opponent_formation: str = "4-4-2",
        home_away: str = "home",
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        attacking_direction: str = "right",
    ) -> LineupSuggestion:
        """Suggest a lineup for a given formation.

        Args:
            formation: Desired formation name (e.g. "4-4-2").
            players: Optional list of player dicts with keys:
                     "track_id", "role" (position name or role string), "rating" (0-1).
            opponent_formation: Opponent's expected formation.
            home_away: "home" or "away".
            pitch_length, pitch_width: Pitch dimensions in meters.
            attacking_direction: Which direction the team attacks ("right" or "left").
                                 Affects formation mirroring.

        Returns:
            LineupSuggestion with optimal slot positions.
        """
        template = self._templates.get(formation)
        if template is None:
            supported = ", ".join(sorted(self._templates))
            return LineupSuggestion(
                formation=formation,
                description=f"Unsupported formation '{formation}'. Supported: {supported}",
            )

        slots: list[PlayerSlot] = []
        for pos_name, x_ratio, y_ratio in template:
            x = x_ratio * pitch_length
            y = y_ratio * pitch_width
            if home_away == "away":
                x = self._mirror_for_away(x, attacking_direction)
            slots.append(PlayerSlot(position_name=pos_name, x=x, y=y))

        # Assign players to slots — try MILP first, fall back to greedy
        if players and len(slots) > 0:
            if _ORTOOLS_AVAILABLE and len(players) >= len(slots):
                try:
                    milp_assigned = self._assign_via_milp(slots, players)
                    if milp_assigned is not None:
                        slots = milp_assigned
                    else:
                        slots = self._assign_players_to_slots(
                            slots, players, pitch_length, pitch_width
                        )
                except Exception:
                    slots = self._assign_players_to_slots(slots, players, pitch_length, pitch_width)
            else:
                slots = self._assign_players_to_slots(slots, players, pitch_length, pitch_width)

        # Adjust for opponent formation
        adj = self._formation_strength_vs(formation, opponent_formation)
        def_strength, att_strength = adj
        overall = (def_strength + att_strength) / 2.0
        confidence = max(0.0, min(1.0, overall))

        # Build tactical description
        desc = self._build_description(formation, opponent_formation, def_strength, att_strength)

        return LineupSuggestion(
            formation=formation,
            slots=slots,
            description=desc,
            confidence=confidence,
        )

    def _assign_via_milp(
        self,
        slots: list[PlayerSlot],
        players: list[dict[str, Any]],
    ) -> list[PlayerSlot] | None:
        """Assign players to slots using MILP (CP-SAT) for optimal role+rating fit.

        Builds a binary assignment model:
            x[p,s] = 1 if player p is assigned to slot s
            constraints: each player <= 1 slot, each slot == 1 player
            objective: maximize sum(x[p,s] * match_score(p,s))
        where match_score = _role_match_score * (1 + 0.1 * rating).

        Returns enriched slot list on success, or None if the solver fails.
        """
        num_players = len(players)
        num_slots = len(slots)

        model = cp_model.CpModel()

        x: dict[tuple[int, int], cp_model.IntVar] = {}
        score_terms: list[cp_model.LinearExpr] = []

        for p_idx, player in enumerate(players):
            role = str(player.get("role", "")).lower()
            rating = float(player.get("rating", 0.5))

            for s_idx, slot in enumerate(slots):
                var = model.NewBoolVar(f"x_{p_idx}_{s_idx}")
                x[p_idx, s_idx] = var

                match_score = self._role_match_score(slot.position_name, role) * (
                    1.0 + 0.1 * rating
                )
                int_coeff = max(1, int(round(match_score * 10000)))
                score_terms.append(var * int_coeff)

        # Each player gets at most one slot
        for p_idx in range(num_players):
            model.Add(sum(x[p_idx, s_idx] for s_idx in range(num_slots)) <= 1)

        # Each slot gets exactly one player
        for s_idx in range(num_slots):
            model.Add(sum(x[p_idx, s_idx] for p_idx in range(num_players)) == 1)

        model.Maximize(sum(score_terms))

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None

        assigned_players: set[int] = set()
        slot_to_player: dict[int, int] = {}

        for p_idx in range(num_players):
            for s_idx in range(num_slots):
                if solver.Value(x[p_idx, s_idx]) == 1:
                    slot_to_player[s_idx] = p_idx
                    assigned_players.add(p_idx)
                    break

        enriched: list[PlayerSlot] = []
        for s_idx, slot in enumerate(slots):
            if s_idx in slot_to_player:
                p = players[slot_to_player[s_idx]]
                pid = p.get("track_id", -1)
                pname = p.get("name", p.get("display_name", f"Player {pid}"))
                enriched.append(
                    PlayerSlot(
                        position_name=slot.position_name,
                        x=slot.x,
                        y=slot.y,
                        role=pname,
                    )
                )
            else:
                enriched.append(
                    PlayerSlot(
                        position_name=slot.position_name,
                        x=slot.x,
                        y=slot.y,
                        role=slot.position_name,
                    )
                )

        for p_idx, player in enumerate(players):
            if p_idx not in assigned_players:
                pid = player.get("track_id", -1)
                pname = player.get("name", player.get("display_name", f"Player {pid}"))
                enriched.append(
                    PlayerSlot(
                        position_name="SUB",
                        x=0.0,
                        y=0.0,
                        role=pname,
                    )
                )

        return enriched

    def compare_formations(
        self,
        formations: list[str],
        opponent_formation: str = "4-4-2",
    ) -> OptimizerResult:
        """Compare multiple formations against a given opponent and rank them.

        Args:
            formations: List of formation names to evaluate.
            opponent_formation: The opponent's expected formation.

        Returns:
            OptimizerResult with ranked suggestions.
        """
        if not formations:
            return OptimizerResult()

        suggestions: list[LineupSuggestion] = []
        for fm in formations:
            def_strength, att_strength = self._formation_strength_vs(fm, opponent_formation)
            overall = (def_strength + att_strength) / 2.0
            confidence = max(0.0, min(1.0, overall))
            desc = self._build_description(fm, opponent_formation, def_strength, att_strength)
            template = self._templates.get(fm, [])
            slots = [
                PlayerSlot(position_name=pos, x=xr * self.PITCH_LENGTH, y=yr * self.PITCH_WIDTH)
                for pos, xr, yr in template
            ]
            suggestions.append(
                LineupSuggestion(
                    formation=fm,
                    slots=slots,
                    description=desc,
                    confidence=confidence,
                )
            )

        suggestions.sort(key=lambda s: s.confidence, reverse=True)
        best = suggestions[0]
        return OptimizerResult(
            suggestions=suggestions,
            best_formation=best.formation,
            best_confidence=best.confidence,
        )

    def _formation_strength_vs(self, formation: str, opponent: str) -> tuple[float, float]:
        """Return (defensive_strength, attacking_strength) vs opponent.

        Uses a simple geometric heuristic:
        - Defensive strength: number of defensive players (GK, CB, FB, CDM)
          relative to opponent attackers.
        - Attacking strength: number of attacking players (ST, W, CAM)
          relative to opponent defenders.
        """

        def _count_positions(fm: str, categories: dict[str, set[str]]) -> float:
            template = self._templates.get(fm, [])
            count = 0.0
            for pos_name, _, _ in template:
                for _cat, members in categories.items():
                    if pos_name in members:
                        count += 1.0
                        break
            return count

        defensive_roles = {
            "defenders": {"GK", "CB", "LB", "RB", "LWB", "RWB"},
            "midfield_def": {"CDM"},
        }
        attacking_roles = {
            "attackers": {"ST", "LW", "RW", "CAM"},
            "midfield_att": {"CM", "LM", "RM"},
        }

        own_def_count = _count_positions(formation, defensive_roles)
        opp_att_count = _count_positions(opponent, attacking_roles)

        own_att_count = _count_positions(formation, attacking_roles)
        opp_def_count = _count_positions(opponent, defensive_roles)

        def_strength = 0.5
        att_strength = 0.5

        if opp_att_count > 0:
            def_strength = min(own_def_count / (opp_att_count + 1.0) * 1.2, 1.0)
        if opp_def_count > 0:
            att_strength = min(own_att_count / (opp_def_count + 1.0) * 1.2, 1.0)

        return (def_strength, att_strength)

    def _assign_players_to_slots(
        self,
        slots: list[PlayerSlot],
        players: list[dict[str, Any]],
        pitch_length: float,
        pitch_width: float,
    ) -> list[PlayerSlot]:
        """Assign players to the best-fitting slots based on role match.

        Players not assigned to any slot are returned as their own slots
        (unassigned), while assigned slots adopt the player's details.
        """
        if not players or not slots:
            return slots

        enriched: list[PlayerSlot] = []
        assigned_players: set[int] = set()

        for slot in slots:
            best_player = None
            best_score = -1.0

            for p in players:
                pid = p.get("track_id", -1)
                if pid < 0:
                    continue
                if pid in assigned_players:
                    continue

                role = str(p.get("role", "")).lower()
                score = self._role_match_score(slot.position_name, role)
                if score > best_score:
                    best_score = score
                    best_player = p

            if best_player is not None and best_score > 0.1:
                pid = best_player.get("track_id", -1)
                assigned_players.add(pid)
                pname = best_player.get("name", best_player.get("display_name", f"Player {pid}"))
                enriched.append(
                    PlayerSlot(
                        position_name=slot.position_name,
                        x=slot.x,
                        y=slot.y,
                        role=pname,
                    )
                )
            else:
                enriched.append(
                    PlayerSlot(
                        position_name=slot.position_name,
                        x=slot.x,
                        y=slot.y,
                        role=slot.position_name,
                    )
                )

        # Append any unassigned players
        for p in players:
            pid = p.get("track_id", -1)
            if pid >= 0 and pid not in assigned_players:
                pname = p.get("name", p.get("display_name", f"Player {pid}"))
                enriched.append(
                    PlayerSlot(
                        position_name="SUB",
                        x=0.0,
                        y=0.0,
                        role=pname,
                    )
                )

        return enriched

    @staticmethod
    def _role_match_score(slot_pos: str, player_role: str) -> float:
        """Score how well a player's role matches a slot position (0..1)."""
        if not player_role:
            return 0.0

        # Direct match
        if player_role == slot_pos.lower():
            return 1.0

        # Check role group
        for canon, aliases in POSITION_ROLES.items():
            if canon == slot_pos:
                if player_role in [a.lower() for a in aliases]:
                    return 0.9
                for alias in aliases:
                    if alias.lower() in player_role or player_role in alias.lower():
                        return 0.7
            if (canonical := _canonical_role(player_role)) and canonical == canon:
                return 0.85

        # Partial fuzzy
        slot_lower = slot_pos.lower()
        if slot_lower in player_role or player_role in slot_lower:
            return 0.5

        # Positional family
        families = {
            "defender": {"CB", "LB", "RB", "LWB", "RWB"},
            "midfielder": {"CDM", "CM", "CAM", "LM", "RM"},
            "forward": {"ST", "LW", "RW"},
        }
        for _family, members in families.items():
            if slot_pos in members:
                role_canon = _canonical_role(player_role)
                if role_canon in members:
                    return 0.4

        return 0.0

    @staticmethod
    def _build_description(
        formation: str,
        opponent: str,
        def_strength: float,
        att_strength: float,
    ) -> str:
        """Generate a tactical description for the formation matchup."""
        parts: list[str] = [f"{formation} vs {opponent}"]

        if def_strength > 0.7:
            parts.append("strong defensive coverage")
        elif def_strength < 0.4:
            parts.append("vulnerable defensively")

        if att_strength > 0.7:
            parts.append("high attacking threat")
        elif att_strength < 0.4:
            parts.append("limited attacking presence")

        if def_strength > 0.6 and att_strength > 0.6:
            parts.append("well-balanced")
        elif def_strength > att_strength + 0.2:
            parts.append("defensive-minded")
        elif att_strength > def_strength + 0.2:
            parts.append("attack-minded")

        return " | ".join(parts)

    @staticmethod
    def _mirror_for_away(x: float, attacking_direction: str = "right") -> float:
        """Mirror x-coordinate for the away team.

        When the home team attacks right (default), the away team attacks left,
        so their formation is mirrored across the pitch via PITCH_LENGTH - x.
        The attacking_direction parameter is provided for API consistency
        (e.g. if the home team attacks left, the away team attacks right and no
        mirror is needed — callers should handle this by skipping the mirror call).
        """
        return PITCH_LENGTH - x


def _canonical_role(role: str) -> str | None:
    """Map a role string to its canonical position name, if possible."""
    r = role.lower()
    for canon, aliases in POSITION_ROLES.items():
        if canon.lower() == r:
            return canon
        for alias in aliases:
            if alias.lower() == r:
                return canon
    return None
