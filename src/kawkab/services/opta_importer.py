from __future__ import annotations

import contextlib
import logging
import xml.etree.ElementTree as ET
from datetime import datetime

from kawkab.services.data_provider_base import (
    BaseDataProvider,
    ProviderEvent,
    ProviderLineup,
    ProviderMatch,
)

logger = logging.getLogger(__name__)


class OptaF7Importer(BaseDataProvider):
    """Imports Opta F7 XML match data.

    Opta F7 format is the standard XML feed for football event data.
    Can import from raw XML or a .xml/.zip file path.
    """

    def __init__(self, data_dir: str = "") -> None:
        self._data_dir = data_dir

    def get_provider_name(self) -> str:
        return "opta_f7"

    # ── XML parsing ──

    def parse_match_xml(self, xml_content: str) -> ProviderMatch:
        root = ET.fromstring(xml_content)
        ns = self._ns(root.tag)
        match_data = root.find(f".//{ns}MatchData")
        if match_data is None:
            match_data = root
        match_info = match_data.find(f".//{ns}MatchInfo")
        home_team = ""
        away_team = ""
        home_score = None
        away_score = None
        competition = ""
        season = ""
        match_date = datetime.now()
        match_id = match_data.get("uID", "")
        if match_id.startswith("m"):
            match_id = match_id[1:]

        if match_info is not None:
            competition = self._get_text(match_info, ns, "Competition", "CompetitionName")
            season = self._get_text(match_info, ns, "Season", "SeasonName")
            date_str = self._get_text(match_info, ns, "Date")
            if date_str:
                try:
                    match_date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    with contextlib.suppress(ValueError):
                        match_date = datetime.strptime(date_str, "%Y-%m-%d")

        team_elements = match_data.findall(f".//{ns}Team")
        for team_el in team_elements:
            team_ref = team_el.get("TeamRef", "")
            name = self._get_text(team_el, ns, "Name")
            score_el = team_el.find(f"{ns}TeamStat")

            if "Home" in team_ref or "home" in team_ref.lower():
                home_team = name
            elif "Away" in team_ref or "away" in team_ref.lower():
                away_team = name

            if score_el is not None:
                score_str = self._get_text(score_el, ns, "Value")
                if "Home" in team_ref or "home" in team_ref.lower():
                    home_score = int(score_str) if score_str and score_str.isdigit() else None
                elif "Away" in team_ref or "away" in team_ref.lower():
                    away_score = int(score_str) if score_str and score_str.isdigit() else None

        return ProviderMatch(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            competition=competition,
            season=season,
            date=match_date,
            home_score=home_score,
            away_score=away_score,
        )

    def parse_event_xml(self, xml_content: str) -> list[ProviderEvent]:
        root = ET.fromstring(xml_content)
        ns = self._ns(root.tag)
        events: list[ProviderEvent] = []
        match_data = root.find(f".//{ns}MatchData")
        if match_data is None:
            match_data = root
        match_id = match_data.get("uID", "")
        if match_id.startswith("m"):
            match_id = match_id[1:]

        game_events = match_data.find(f".//{ns}GameEvents")
        if game_events is None:
            game_events = match_data

        for ev_el in game_events.findall(f".//{ns}Event"):
            event_id = ev_el.get("id", "")
            event_type = ev_el.get("type_id", "")
            team_id = ev_el.get("team_id", "")
            player_id = ev_el.get("player_id", "")
            timestamp = float(ev_el.get("time_min", 0)) * 60 + float(ev_el.get("time_sec", 0))
            outcome = ev_el.get("outcome", "1") == "1"

            x = float(ev_el.get("x", 0))
            y = float(ev_el.get("y", 0))
            end_x = float(ev_el.get("end_x", 0))
            end_y = float(ev_el.get("end_y", 0))

            qualifiers = {}
            for qual in ev_el.findall(f".//{ns}Q"):
                q_id = qual.get("q_id", "")
                q_val = qual.get("value", "")
                qualifiers[q_id] = q_val

            events.append(
                ProviderEvent(
                    event_id=event_id,
                    match_id=match_id,
                    timestamp=timestamp,
                    type=self._opta_type_name(event_type),
                    team=team_id,
                    player=player_id,
                    x=x,
                    y=y,
                    end_x=end_x,
                    end_y=end_y,
                    outcome=outcome,
                    extra={"type_id": event_type, "qualifiers": qualifiers},
                )
            )

        return events

    def parse_lineup_xml(self, xml_content: str) -> list[ProviderLineup]:
        root = ET.fromstring(xml_content)
        ns = self._ns(root.tag)
        lineups: list[ProviderLineup] = []
        match_data = root.find(f".//{ns}MatchData")
        if match_data is None:
            match_data = root
        match_id = match_data.get("uID", "")
        if match_id.startswith("m"):
            match_id = match_id[1:]

        team_elements = match_data.findall(f".//{ns}Team")

        for team_el in team_elements:
            team_ref = team_el.get("TeamRef", "")
            team_name = self._get_text(team_el, ns, "Name")
            formation = ""
            players_list: list[dict] = []

            for player_el in team_el.findall(f".//{ns}Player"):
                player_ref = player_el.get("PlayerRef", "")
                shirt_number = player_el.get("ShirtNumber", "")
                position = player_el.get("Position", "")
                status = player_el.get("Status", "")
                players_list.append(
                    {
                        "player_ref": player_ref,
                        "shirt_number": int(shirt_number)
                        if shirt_number and shirt_number.isdigit()
                        else 0,
                        "position": position,
                        "status": status,
                    }
                )
                if formation and position in ("Goalkeeper", "Defender", "Midfielder", "Forward"):
                    pass

            formation_el = team_el.find(f"{ns}Formation")
            if formation_el is not None:
                formation = formation_el.text or ""

            lineups.append(
                ProviderLineup(
                    match_id=match_id,
                    team=team_name or team_ref,
                    formation=formation,
                    players=players_list,
                )
            )

        return lineups

    # ── Async interface ──

    async def search_matches(
        self,
        team: str | None = None,
        competition: str | None = None,
        season: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
    ) -> list[ProviderMatch]:
        logger.warning(
            "OptaF7Importer.search_matches: local file mode - provide match XML files directly"
        )
        return []

    async def get_match_events(self, match_id: str) -> list[ProviderEvent]:
        logger.warning("OptaF7Importer.get_match_events: use parse_event_xml(xml_content) directly")
        return []

    async def get_match_lineups(self, match_id: str) -> list[ProviderLineup]:
        logger.warning(
            "OptaF7Importer.get_match_lineups: use parse_lineup_xml(xml_content) directly"
        )
        return []

    def get_rate_limit_info(self) -> dict:
        return {"requests_per_min": 0, "daily_limit": 0, "type": "local_file"}

    # ── Helpers ──

    @staticmethod
    def _ns(tag: str) -> str:
        idx = tag.find("}")
        if idx >= 0:
            return "{" + tag[1:idx] + "}"
        return ""

    @staticmethod
    def _get_text(element: ET.Element, ns: str, *path_parts: str) -> str:
        current = element
        for part in path_parts:
            child = current.find(f"{ns}{part}")
            if child is None:
                child = current.find(part)
            if child is None:
                return ""
            current = child
        return current.text or ""

    @staticmethod
    def _opta_type_name(type_id: str) -> str:
        mapping = {
            "1": "pass",
            "2": "offside_pass",
            "3": "take_on",
            "4": "foul",
            "5": "out_of_play",
            "6": "corner",
            "7": "cross",
            "8": "shot_off_target",
            "9": "shot_on_target",
            "10": "shot_post",
            "11": "goal",
            "12": "attempt_saved",
            "13": "card",
            "14": "substitution",
            "15": "free_kick",
            "16": "goal_kick",
            "17": "throw_in",
            "18": "tackle",
            "19": "interception",
            "20": "clearance",
            "21": "block",
            "22": "save",
            "23": "penalty",
            "24": "own_goal",
            "25": "miss",
            "26": "key_pass",
            "27": "through_ball",
            "28": "dribble",
            "29": "turnover",
        }
        return mapping.get(type_id, f"unknown_{type_id}")
