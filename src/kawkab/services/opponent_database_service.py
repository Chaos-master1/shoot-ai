"""Opponent database — store opponent profiles, tactical tendencies, head-to-head history."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime

from kawkab.core import paths as kawkab_paths
from kawkab.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OpponentProfile:
    id: str
    team_name: str
    league: str = ""
    country: str = ""
    formation_tendencies: list[str] = field(default_factory=list)
    pressing_style: str = ""
    attacking_patterns: list[str] = field(default_factory=list)
    defensive_vulnerabilities: list[str] = field(default_factory=list)
    set_piece_routines: list[str] = field(default_factory=list)
    key_players: list[dict] = field(default_factory=list)
    notes: str = ""
    match_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class MatchUpRecord:
    id: str
    opponent_id: str
    our_team: str
    date: str
    competition: str = ""
    home_away: str = "home"
    our_score: int = 0
    their_score: int = 0
    our_possession: float = 0.0
    their_possession: float = 0.0
    our_shots: int = 0
    their_shots: int = 0
    our_xg: float = 0.0
    their_xg: float = 0.0
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class OpponentDatabaseService:
    """Store and analyze opponent profiles with tactical tendencies and head-to-head history."""

    def __init__(self, data_dir: str | None = None) -> None:
        self._profiles: dict[str, OpponentProfile] = {}
        self._matchups: dict[str, MatchUpRecord] = {}
        # Per-user app-data directory, NOT the source tree: profiles are
        # runtime user data and used to be written into src/kawkab/../data
        # (untracked artifacts inside the repo, lost on reinstall for
        # packaged apps where that tree is read-only).
        if data_dir is None:
            data_dir = str(kawkab_paths.get_paths().appdata / "data" / "opponents")
        self._data_dir = data_dir
        self._load_data()

    def _data_path(self, *parts: str) -> str:
        path = os.path.join(self._data_dir, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def _load_data(self) -> None:
        profiles_file = self._data_path("profiles.json")
        matchups_file = self._data_path("matchups.json")
        try:
            if os.path.exists(profiles_file):
                with open(profiles_file, encoding="utf-8") as f:
                    data = json.load(f)
                for p in data:
                    self._profiles[p["id"]] = OpponentProfile(**p)
            if os.path.exists(matchups_file):
                with open(matchups_file, encoding="utf-8") as f:
                    data = json.load(f)
                for m in data:
                    self._matchups[m["id"]] = MatchUpRecord(**m)
        except Exception as e:
            logger.warning(f"Failed to load opponent data: {e}")

    def _save_profiles(self) -> None:
        data = [vars(p) for p in self._profiles.values()]
        with open(self._data_path("profiles.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def _save_matchups(self) -> None:
        data = [vars(m) for m in self._matchups.values()]
        with open(self._data_path("matchups.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def list_profiles(self) -> list[dict]:
        results = []
        for p in self._profiles.values():
            results.append(
                {
                    "id": p.id,
                    "team_name": p.team_name,
                    "league": p.league,
                    "country": p.country,
                    "formation": ", ".join(p.formation_tendencies),
                    "pressing_style": p.pressing_style,
                    "matches": len(p.match_ids),
                    "updated_at": p.updated_at,
                }
            )
        results.sort(key=lambda x: str(x["team_name"]))  # type: ignore[arg-type,return-value]
        return results

    def get_profile(self, profile_id: str) -> dict | None:
        p = self._profiles.get(profile_id)
        if not p:
            return None
        return {
            "id": p.id,
            "team_name": p.team_name,
            "league": p.league,
            "country": p.country,
            "formation_tendencies": p.formation_tendencies,
            "pressing_style": p.pressing_style,
            "attacking_patterns": p.attacking_patterns,
            "defensive_vulnerabilities": p.defensive_vulnerabilities,
            "set_piece_routines": p.set_piece_routines,
            "key_players": p.key_players,
            "notes": p.notes,
            "match_ids": p.match_ids,
            "created_at": p.created_at,
            "updated_at": p.updated_at,
        }

    def create_profile(self, team_name: str, league: str = "", country: str = "") -> dict:
        import uuid

        pid = str(uuid.uuid4())[:8]
        profile = OpponentProfile(id=pid, team_name=team_name, league=league, country=country)
        self._profiles[pid] = profile
        self._save_profiles()
        return {"id": pid, "team_name": team_name}

    def update_profile(self, profile_id: str, updates: dict) -> bool:
        p = self._profiles.get(profile_id)
        if not p:
            return False
        for key, val in updates.items():
            if hasattr(p, key) and key not in ("id", "created_at"):
                setattr(p, key, val)
        p.updated_at = datetime.now(UTC).isoformat()
        self._save_profiles()
        return True

    def delete_profile(self, profile_id: str) -> bool:
        if profile_id in self._profiles:
            del self._profiles[profile_id]
            self._save_profiles()
            return True
        return False

    def add_matchup(
        self,
        opponent_id: str,
        our_team: str,
        date: str,
        competition: str = "",
        home_away: str = "home",
        our_score: int = 0,
        their_score: int = 0,
        our_xg: float = 0.0,
        their_xg: float = 0.0,
        notes: str = "",
    ) -> dict:
        import uuid

        mid = str(uuid.uuid4())[:8]
        record = MatchUpRecord(
            id=mid,
            opponent_id=opponent_id,
            our_team=our_team,
            date=date,
            competition=competition,
            home_away=home_away,
            our_score=our_score,
            their_score=their_score,
            our_xg=our_xg,
            their_xg=their_xg,
            notes=notes,
        )
        self._matchups[mid] = record
        self._save_matchups()

        if opponent_id in self._profiles:
            p = self._profiles[opponent_id]
            p.match_ids.append(str(mid))
            p.updated_at = datetime.now(UTC).isoformat()
            self._save_profiles()

        return {"id": mid}

    def get_matchups(self, opponent_id: str) -> list[dict]:
        results = []
        for m in self._matchups.values():
            if m.opponent_id == opponent_id:
                results.append(
                    {
                        "id": m.id,
                        "date": m.date,
                        "competition": m.competition,
                        "home_away": m.home_away,
                        "score": f"{m.our_score} - {m.their_score}",
                        "our_xg": m.our_xg,
                        "their_xg": m.their_xg,
                        "notes": m.notes,
                    }
                )
        results.sort(key=lambda x: str(x["date"]), reverse=True)  # type: ignore[arg-type,return-value]
        return results

    def generate_scouting_report(self, opponent_id: str) -> str:
        profile = self.get_profile(profile_id=opponent_id)
        if not profile:
            return "Opponent not found."

        matchups = self.get_matchups(opponent_id)
        total_matches = len(matchups)
        wins = sum(
            1 for m in matchups if int(m["score"].split(" - ")[0]) > int(m["score"].split(" - ")[1])
        )
        losses = sum(
            1 for m in matchups if int(m["score"].split(" - ")[0]) < int(m["score"].split(" - ")[1])
        )
        draws = total_matches - wins - losses

        report = f"""# Scouting Report: {profile["team_name"]}

## Overview
- **League**: {profile.get("league", "N/A")}
- **Country**: {profile.get("country", "N/A")}
- **Head-to-Head**: {wins}W / {draws}D / {losses}L ({total_matches} matches)

## Tactical Profile
- **Preferred Formations**: {", ".join(profile.get("formation_tendencies", [])) or "N/A"}
- **Pressing Style**: {profile.get("pressing_style", "N/A")}
- **Attacking Patterns**: {", ".join(profile.get("attacking_patterns", [])) or "N/A"}
- **Defensive Vulnerabilities**: {", ".join(profile.get("defensive_vulnerabilities", [])) or "N/A"}
- **Set Piece Routines**: {", ".join(profile.get("set_piece_routines", [])) or "N/A"}

## Key Players"""
        for kp in profile.get("key_players", []):
            report += f"\n- **{kp.get('name', 'Unknown')}** ({kp.get('position', 'N/A')}) — {kp.get('notes', '')}"

        report += """

## Match History"""
        for m in matchups[:5]:
            report += f"\n- {m['date']}: {m.get('competition', 'N/A')} — {m['score']} (xG: {m.get('our_xg', '?')} - {m.get('their_xg', '?')})"

        if profile.get("notes"):
            report += f"\n\n## Additional Notes\n{profile['notes']}"

        report += "\n\n## Game Plan Recommendations\n- Press their weakness in transition\n- Target full-backs in wide areas\n- Stay compact to counter their attacking patterns\n- Defend set pieces with zonal marking"

        return report
