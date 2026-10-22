"""EasySoccerData wrapper - scrapes Sofascore, FBref, Promiedos for live data.

No API key required. Silent degradation: returns empty results on failure.
Sofascore module is stable (PyPI v0.0.8); FBref in development.
"""

from __future__ import annotations

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


class EasySoccerService:
    """Wrapper around EasySoccerData's Sofascore client.

    Provides live event data, match details, player info, and incidents
    from Sofascore without any API key. Falls back gracefully on failure.
    """

    def __init__(self) -> None:
        self._client = None
        self._available = False

    def _get_client(self):
        if self._client is None:
            try:
                import esd

                self._client = esd.SofascoreClient()
                self._available = True
            except ImportError:
                logger.warning(
                    "EasySoccerData not installed. Install with: pip install EasySoccerData"
                )
                return None
            except Exception as e:
                logger.warning(f"Failed to initialize EasySoccerData: {e}")
                return None
        return self._client

    def check_available(self) -> bool:
        """Check if EasySoccerData is installed and functional."""
        client = self._get_client()
        if client is None:
            return False
        try:
            events = client.get_events(live=True)
            return events is not None and len(events) > 0
        except Exception:
            return False

    def get_live_events(self) -> list[dict]:
        """Get currently live matches from Sofascore."""
        client = self._get_client()
        if client is None:
            return []
        try:
            events = client.get_events(live=True)
            return [
                {
                    "id": e.get("id"),
                    "home_team": e.get("home_team", ""),
                    "away_team": e.get("away_team", ""),
                    "home_score": e.get("home_score"),
                    "away_score": e.get("away_score"),
                    "status": e.get("status", ""),
                    "league_name": e.get("league_name", ""),
                    "current_minute": e.get("current_minute"),
                }
                for e in (events or [])
            ]
        except Exception as ex:
            logger.warning(f"EasySoccerData get_live_events failed: {ex}")
            return []

    def get_event(self, event_id: int) -> dict | None:
        """Get detailed match info from Sofascore."""
        client = self._get_client()
        if client is None:
            return None
        try:
            e = client.get_event(event_id)
            if e is None:
                return None
            return {
                "id": e.get("id"),
                "home_team": e.get("home_team", ""),
                "away_team": e.get("away_team", ""),
                "home_score": e.get("home_score"),
                "away_score": e.get("away_score"),
                "status": e.get("status", ""),
                "venue": e.get("venue", ""),
                "league_name": e.get("league_name", ""),
                "event_date": str(e.get("event_date", "")),
            }
        except Exception as ex:
            logger.warning(f"EasySoccerData get_event failed: {ex}")
            return None

    def get_match_incidents(self, event_id: int) -> list[dict]:
        """Get goals, cards, substitutions for a match."""
        client = self._get_client()
        if client is None:
            return []
        try:
            incidents = client.get_match_incidents(event_id)
            return [
                {
                    "type": i.get("type", ""),
                    "team": i.get("team", ""),
                    "player": i.get("player", ""),
                    "minute": i.get("minute"),
                    "score_home": i.get("score_home"),
                    "score_away": i.get("score_away"),
                }
                for i in (incidents or [])
            ]
        except Exception as ex:
            logger.warning(f"EasySoccerData get_match_incidents failed: {ex}")
            return []

    def get_player(self, player_id: int) -> dict | None:
        """Get player info from Sofascore."""
        client = self._get_client()
        if client is None:
            return None
        try:
            p = client.get_player(player_id)
            if p is None:
                return None
            return {
                "id": p.get("id"),
                "name": p.get("name", ""),
                "position": p.get("position", ""),
                "jersey_number": p.get("jersey_number"),
                "nationality": p.get("nationality", ""),
                "date_of_birth": str(p.get("date_of_birth", "")),
            }
        except Exception as ex:
            logger.warning(f"EasySoccerData get_player failed: {ex}")
            return None

    def search_events(self, date: str = "today") -> list[dict]:
        """Get scheduled events for a date (YYYY-MM-DD or 'today')."""
        client = self._get_client()
        if client is None:
            return []
        try:
            events = client.get_events(date=date)
            return [
                {
                    "id": e.get("id"),
                    "home_team": e.get("home_team", ""),
                    "away_team": e.get("away_team", ""),
                    "home_score": e.get("home_score"),
                    "away_score": e.get("away_score"),
                    "status": e.get("status", ""),
                    "league_name": e.get("league_name", ""),
                }
                for e in (events or [])
            ]
        except Exception as ex:
            logger.warning(f"EasySoccerData search_events failed: {ex}")
            return []
