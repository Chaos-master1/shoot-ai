"""Handler for external API bridge methods - all football-data.org, Bzzoiro, API-Football,
StatsBomb, TheSportsDB, OpenFootball, EasySoccer integrations."""

from __future__ import annotations

import json

from kawkab.core.logging import get_logger
from kawkab.core.security import ErrorSanitizer, SecurityValidator
from kawkab.ui.bridge_handlers.base import BridgeHandlerBase

logger = get_logger(__name__)


class ExternalHandler(BridgeHandlerBase):
    """Handles all external football data API operations for Bridge."""

    # ── service accessors ────────────────────────────────────────

    @property
    def football_data_service(self):
        return self._services.get("football_data_service")

    @property
    def bzzoiro_service(self):
        return self._services.get("bzzoiro_service")

    @property
    def easy_soccer_service(self):
        return self._services.get("easy_soccer_service")

    @property
    def api_football_service(self):
        return self._services.get("api_football_service")

    @property
    def thesportsdb_service(self):
        return self._services.get("thesportsdb_service")

    @property
    def statsbomb_service(self):
        return self._services.get("statsbomb_service")

    @property
    def openfootball_service(self):
        return self._services.get("openfootball_service")

    @property
    def player_profile_service(self):
        return self._services.get("player_profile_service")

    @property
    def storage_service(self):
        return self._services.get("storage_service")

    # ================================================================
    # football-data.org
    # ================================================================

    async def check_football_data_status(self):
        self._check_rate_limit()
        if self.football_data_service is None:
            return json.dumps({"available": False, "error": "Service not initialized"})
        try:
            status = await self.football_data_service.check_status()
            return json.dumps(status)
        except Exception as e:
            logger.error(f"check_football_data_status failed: {e}")
            return json.dumps({"available": False, "error": ErrorSanitizer.sanitize_error(e)})

    async def search_football_team(self, query):
        if self.football_data_service is None:
            return json.dumps({"teams": []})
        try:
            query = SecurityValidator.sanitize_string(query, max_length=100)
            teams = await self.football_data_service.search_team(query)
            return json.dumps({"teams": teams})
        except Exception as e:
            logger.error(f"search_football_team failed: {e}")
            return json.dumps({"teams": []})

    async def import_football_team_squad(self, match_id, api_team_id, side):
        if self.football_data_service is None or self.player_profile_service is None:
            return json.dumps({"success": False, "error": "Required service not available"})
        try:
            match_id = SecurityValidator.validate_match_id(match_id)
            team_id = int(api_team_id)
            side = SecurityValidator.sanitize_string(side, max_length=10)
            if side not in ("home", "away"):
                return json.dumps({"success": False, "error": "side must be 'home' or 'away'"})

            squad = await self.football_data_service.import_team_squad(team_id, side)
            created = []
            skipped = 0

            existing = await self.player_profile_service.get_all_profiles(team=side)
            existing_nums = {p.jersey_number for p in existing if p.jersey_number is not None}

            for player_data in squad:
                if player_data["jersey_number"] in existing_nums:
                    skipped += 1
                    continue
                try:
                    profile = await self.player_profile_service.create_profile(**player_data)
                    created.append(
                        {
                            "profile_id": profile.id,
                            "name": player_data["display_name"],
                            "jersey": player_data["jersey_number"],
                            "position": player_data["preferred_position"],
                        }
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to create profile for {player_data['display_name']}: {e}"
                    )

            if side == "home":
                await self.storage_service.update_match_football_data(
                    match_id, football_data_home_team_id=team_id
                )
            else:
                await self.storage_service.update_match_football_data(
                    match_id, football_data_away_team_id=team_id
                )

            return json.dumps(
                {
                    "success": True,
                    "created": created,
                    "skipped": skipped,
                }
            )
        except Exception as e:
            logger.error(f"import_football_team_squad failed: {e}")
            return json.dumps({"success": False, "error": ErrorSanitizer.sanitize_error(e)})

    async def verify_match_with_api(self, match_id, api_match_id):
        if self.football_data_service is None:
            return json.dumps({"error": "FootballDataService not available"})
        try:
            match_id = SecurityValidator.validate_match_id(match_id)
            match = await self.storage_service.get_match(match_id)
            if not match:
                return json.dumps({"error": "Match not found"})

            events = await self.storage_service.get_match_events(match_id)
            shot_events = [e for e in events if e.get("type") == "goal"]
            detected_home = sum(1 for e in shot_events if e.get("team") == "home")
            detected_away = sum(1 for e in shot_events if e.get("team") == "away")

            result = await self.football_data_service.verify_match(
                api_match_id, detected_home, detected_away
            )
            if result is None:
                return json.dumps({"error": "Could not fetch match data from API"})

            await self.storage_service.update_match_football_data(
                match_id, api_match_id=api_match_id
            )

            return json.dumps({"success": True, **result})
        except Exception as e:
            logger.error(f"verify_match_with_api failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_football_standings(self, competition_code):
        if self.football_data_service is None:
            return json.dumps({"error": "FootballDataService not available"})
        try:
            code = SecurityValidator.sanitize_string(competition_code, max_length=10)
            standings = await self.football_data_service.get_standings(code)
            if standings is None:
                return json.dumps({"error": "Could not fetch standings"})
            return json.dumps({"success": True, "standings": standings})
        except Exception as e:
            logger.error(f"get_football_standings failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_football_competitions(self):
        if self.football_data_service is None:
            return json.dumps({"competitions": []})
        try:
            comps = await self.football_data_service.get_competitions()
            return json.dumps({"competitions": comps})
        except Exception as e:
            logger.error(f"get_football_competitions failed: {e}")
            return json.dumps({"competitions": []})

    async def get_football_team_matches(self, api_team_id, date_from, date_to):
        if self.football_data_service is None:
            return json.dumps({"matches": []})
        try:
            matches = await self.football_data_service.get_team_matches(
                api_team_id, date_from=date_from or None, date_to=date_to or None
            )
            return json.dumps({"matches": matches})
        except Exception as e:
            logger.error(f"get_football_team_matches failed: {e}")
            return json.dumps({"matches": []})

    # ================================================================
    # Bzzoiro
    # ================================================================

    async def check_bzzoiro_status(self):
        if self.bzzoiro_service is None:
            return json.dumps({"available": False, "error": "Service not initialized"})
        try:
            status = await self.bzzoiro_service.check_status()
            return json.dumps(status)
        except Exception as e:
            logger.error(f"check_bzzoiro_status failed: {e}")
            return json.dumps({"available": False, "error": ErrorSanitizer.sanitize_error(e)})

    async def search_bzzoiro_team(self, query):
        if self.bzzoiro_service is None:
            return json.dumps({"teams": []})
        try:
            query = SecurityValidator.sanitize_string(query, max_length=100)
            teams = await self.bzzoiro_service.search_team(query)
            return json.dumps({"teams": teams})
        except Exception as e:
            logger.error(f"search_bzzoiro_team failed: {e}")
            return json.dumps({"teams": []})

    async def get_bzzoiro_team_squad(self, team_id):
        if self.bzzoiro_service is None:
            return json.dumps({"players": []})
        try:
            squad = await self.bzzoiro_service.get_team_squad(team_id)
            return json.dumps({"players": squad})
        except Exception as e:
            logger.error(f"get_bzzoiro_team_squad failed: {e}")
            return json.dumps({"players": []})

    async def import_bzzoiro_team_squad(self, match_id, team_id, side):
        if self.bzzoiro_service is None or self.player_profile_service is None:
            return json.dumps({"success": False, "error": "Required service not available"})
        try:
            match_id = SecurityValidator.validate_match_id(match_id)
            side = SecurityValidator.sanitize_string(side, max_length=10)
            if side not in ("home", "away"):
                return json.dumps({"success": False, "error": "side must be 'home' or 'away'"})
            squad = await self.bzzoiro_service.get_team_squad(team_id)
            created = []
            skipped = 0
            existing = await self.player_profile_service.get_all_profiles(team=side)
            existing_nums = {p.jersey_number for p in existing if p.jersey_number is not None}
            for p in squad:
                jersey = p.get("jersey_number")
                if jersey is not None and jersey in existing_nums:
                    skipped += 1
                    continue
                try:
                    profile = await self.player_profile_service.create_profile(
                        display_name=p.get("name"),
                        jersey_number=jersey,
                        preferred_position=p.get("position"),
                        nationality=p.get("nationality"),
                        date_of_birth=p.get("date_of_birth"),
                        team=side,
                        bzzoiro_person_id=p.get("id"),
                        bzzoiro_team_id=team_id,
                    )
                    created.append(
                        {
                            "profile_id": profile.id,
                            "name": p.get("name"),
                            "jersey": jersey,
                            "position": p.get("position"),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to create profile: {e}")
            if side == "home":
                await self.storage_service.update_match_bzzoiro(
                    match_id, bzzoiro_home_team_id=team_id
                )
            else:
                await self.storage_service.update_match_bzzoiro(
                    match_id, bzzoiro_away_team_id=team_id
                )
            return json.dumps({"success": True, "created": created, "skipped": skipped})
        except Exception as e:
            logger.error(f"import_bzzoiro_team_squad failed: {e}")
            return json.dumps({"success": False, "error": ErrorSanitizer.sanitize_error(e)})

    async def verify_match_bzzoiro(self, match_id, bzzoiro_event_id):
        if self.bzzoiro_service is None:
            return json.dumps({"error": "BzzoiroService not available"})
        try:
            match_id = SecurityValidator.validate_match_id(match_id)
            match = await self.storage_service.get_match(match_id)
            if not match:
                return json.dumps({"error": "Match not found"})
            detail = await self.bzzoiro_service.get_match_detail(bzzoiro_event_id)
            if detail is None:
                return json.dumps({"error": "Could not fetch match from Bzzoiro"})
            events = await self.storage_service.get_match_events(match_id)
            shot_events = [e for e in events if e.get("type") == "goal"]
            detected_home = sum(1 for e in shot_events if e.get("team") == "home")
            detected_away = sum(1 for e in shot_events if e.get("team") == "away")
            api_home = detail.get("home_score") or 0
            api_away = detail.get("away_score") or 0
            match_name = detail.get("home_team", "") + " vs " + detail.get("away_team", "")
            match_ok = (detected_home == api_home) and (detected_away == api_away)
            await self.storage_service.update_match_bzzoiro(
                match_id, bzzoiro_event_id=bzzoiro_event_id
            )
            return json.dumps(
                {
                    "success": True,
                    "match": match_name,
                    "api_score": f"{api_home}-{api_away}",
                    "detected_score": f"{detected_home}-{detected_away}",
                    "match_ok": match_ok,
                }
            )
        except Exception as e:
            logger.error(f"verify_match_bzzoiro failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_bzzoiro_standings(self, league_id):
        if self.bzzoiro_service is None:
            return json.dumps({"standings": []})
        try:
            standings = await self.bzzoiro_service.get_standings(league_id)
            return json.dumps({"standings": standings})
        except Exception as e:
            logger.error(f"get_bzzoiro_standings failed: {e}")
            return json.dumps({"standings": []})

    async def get_bzzoiro_leagues(self):
        if self.bzzoiro_service is None:
            return json.dumps({"leagues": []})
        try:
            leagues = await self.bzzoiro_service.get_leagues()
            return json.dumps({"leagues": leagues})
        except Exception as e:
            logger.error(f"get_bzzoiro_leagues failed: {e}")
            return json.dumps({"leagues": []})

    async def get_bzzoiro_team_matches(self, team_id, date_from, date_to):
        if self.bzzoiro_service is None:
            return json.dumps({"matches": []})
        try:
            matches = await self.bzzoiro_service.get_team_matches(
                team_id, date_from=date_from or None, date_to=date_to or None
            )
            return json.dumps({"matches": matches})
        except Exception as e:
            logger.error(f"get_bzzoiro_team_matches failed: {e}")
            return json.dumps({"matches": []})

    async def get_bzzoiro_live(self):
        if self.bzzoiro_service is None:
            return json.dumps({"matches": []})
        try:
            matches = await self.bzzoiro_service.get_live_events()
            return json.dumps({"matches": matches})
        except Exception as e:
            logger.error(f"get_bzzoiro_live failed: {e}")
            return json.dumps({"matches": []})

    async def get_bzzoiro_predictions(self, event_id):
        if self.bzzoiro_service is None:
            return json.dumps({"error": "BzzoiroService not available"})
        try:
            preds = await self.bzzoiro_service.get_predictions(event_id)
            if preds is None:
                return json.dumps({"error": "No predictions available"})
            return json.dumps({"predictions": preds})
        except Exception as e:
            logger.error(f"get_bzzoiro_predictions failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_bzzoiro_match_stats(self, event_id):
        if self.bzzoiro_service is None:
            return json.dumps({"error": "BzzoiroService not available"})
        try:
            stats = await self.bzzoiro_service.get_match_stats(event_id)
            if stats is None:
                return json.dumps({"error": "No stats available"})
            return json.dumps({"stats": stats})
        except Exception as e:
            logger.error(f"get_bzzoiro_match_stats failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # EasySoccerData (Sofascore)
    # ================================================================

    async def check_easy_soccer_status(self):
        if self.easy_soccer_service is None:
            return json.dumps({"available": False, "error": "Service not initialized"})
        try:
            ok = self.easy_soccer_service.check_available()
            return json.dumps({"available": ok})
        except Exception as e:
            logger.error(f"check_easy_soccer_status failed: {e}")
            return json.dumps({"available": False, "error": ErrorSanitizer.sanitize_error(e)})

    async def get_easy_soccer_live(self):
        if self.easy_soccer_service is None:
            return json.dumps({"matches": []})
        try:
            events = self.easy_soccer_service.get_live_events()
            return json.dumps({"matches": events})
        except Exception as e:
            logger.error(f"get_easy_soccer_live failed: {e}")
            return json.dumps({"matches": []})

    async def get_easy_soccer_event(self, event_id):
        if self.easy_soccer_service is None:
            return json.dumps({"error": "EasySoccerData not available"})
        try:
            detail = self.easy_soccer_service.get_event(event_id)
            if detail is None:
                return json.dumps({"error": "Event not found"})
            return json.dumps({"event": detail})
        except Exception as e:
            logger.error(f"get_easy_soccer_event failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_easy_soccer_incidents(self, event_id):
        if self.easy_soccer_service is None:
            return json.dumps({"incidents": []})
        try:
            incidents = self.easy_soccer_service.get_match_incidents(event_id)
            return json.dumps({"incidents": incidents})
        except Exception as e:
            logger.error(f"get_easy_soccer_incidents failed: {e}")
            return json.dumps({"incidents": []})

    async def get_easy_soccer_player(self, player_id):
        if self.easy_soccer_service is None:
            return json.dumps({"error": "EasySoccerData not available"})
        try:
            player = self.easy_soccer_service.get_player(player_id)
            if player is None:
                return json.dumps({"error": "Player not found"})
            return json.dumps({"player": player})
        except Exception as e:
            logger.error(f"get_easy_soccer_player failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def search_easy_soccer_events(self, date):
        if self.easy_soccer_service is None:
            return json.dumps({"events": []})
        try:
            events = self.easy_soccer_service.search_events(date)
            return json.dumps({"events": events})
        except Exception as e:
            logger.error(f"search_easy_soccer_events failed: {e}")
            return json.dumps({"events": []})

    # ================================================================
    # API-Football (api-sports.io)
    # ================================================================

    async def check_apifootball_status(self):
        if self.api_football_service is None:
            return json.dumps({"available": False, "error": "Service not initialized"})
        try:
            status = await self.api_football_service.check_status()
            return json.dumps(status)
        except Exception as e:
            logger.error(f"check_apifootball_status failed: {e}")
            return json.dumps({"available": False, "error": ErrorSanitizer.sanitize_error(e)})

    async def search_apifootball_team(self, query):
        if self.api_football_service is None:
            return json.dumps({"teams": []})
        try:
            query = SecurityValidator.sanitize_string(query, max_length=100)
            teams = await self.api_football_service.search_team(query)
            return json.dumps({"teams": teams})
        except Exception as e:
            logger.error(f"search_apifootball_team failed: {e}")
            return json.dumps({"teams": []})

    async def import_apifootball_squad(self, match_id, team_id, side):
        if self.api_football_service is None or self.player_profile_service is None:
            return json.dumps({"success": False, "error": "Required service not available"})
        try:
            match_id = SecurityValidator.validate_match_id(match_id)
            side = SecurityValidator.sanitize_string(side, max_length=10)
            if side not in ("home", "away"):
                return json.dumps({"success": False, "error": "side must be 'home' or 'away'"})
            squad = await self.api_football_service.get_team_squad(team_id)
            created = []
            skipped = 0
            existing = await self.player_profile_service.get_all_profiles(team=side)
            existing_nums = {p.jersey_number for p in existing if p.jersey_number is not None}
            for p in squad:
                jersey = p.get("jersey_number")
                if jersey is not None and jersey in existing_nums:
                    skipped += 1
                    continue
                try:
                    profile = await self.player_profile_service.create_profile(
                        display_name=p.get("name"),
                        jersey_number=jersey,
                        preferred_position=p.get("position"),
                        team=side,
                        apifb_person_id=p.get("id"),
                        apifb_team_id=team_id,
                    )
                    created.append(
                        {
                            "profile_id": profile.id,
                            "name": p.get("name"),
                            "jersey": jersey,
                            "position": p.get("position"),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to create profile: {e}")
            if side == "home":
                await self.storage_service.update_match_apifootball(
                    match_id, apifb_home_team_id=team_id
                )
            else:
                await self.storage_service.update_match_apifootball(
                    match_id, apifb_away_team_id=team_id
                )
            return json.dumps({"success": True, "created": created, "skipped": skipped})
        except Exception as e:
            logger.error(f"import_apifootball_squad failed: {e}")
            return json.dumps({"success": False, "error": ErrorSanitizer.sanitize_error(e)})

    async def get_apifootball_standings(self, league_id, season=2024):
        if self.api_football_service is None:
            return json.dumps({"standings": []})
        try:
            standings = await self.api_football_service.get_standings(league_id, season)
            return json.dumps({"standings": standings})
        except Exception as e:
            logger.error(f"get_apifootball_standings failed: {e}")
            return json.dumps({"standings": []})

    async def get_apifootball_fixtures(self, team_id, season):
        if self.api_football_service is None:
            return json.dumps({"matches": []})
        try:
            matches = await self.api_football_service.get_fixtures(team_id, season, last=5)
            return json.dumps({"matches": matches})
        except Exception as e:
            logger.error(f"get_apifootball_fixtures failed: {e}")
            return json.dumps({"matches": []})

    async def get_apifootball_fixture_detail(self, fixture_id):
        if self.api_football_service is None:
            return json.dumps({"error": "ApiFootballService not available"})
        try:
            detail = await self.api_football_service.get_fixture_detail(fixture_id)
            if detail is None:
                return json.dumps({"error": "Fixture not found"})
            return json.dumps({"fixture": detail})
        except Exception as e:
            logger.error(f"get_apifootball_fixture_detail failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_apifootball_predictions(self, fixture_id):
        if self.api_football_service is None:
            return json.dumps({"error": "ApiFootballService not available"})
        try:
            preds = await self.api_football_service.get_predictions(fixture_id)
            if preds is None:
                return json.dumps({"error": "No predictions available"})
            return json.dumps({"predictions": preds})
        except Exception as e:
            logger.error(f"get_apifootball_predictions failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_apifootball_live(self):
        if self.api_football_service is None:
            return json.dumps({"matches": []})
        try:
            matches = await self.api_football_service.get_live_fixtures()
            return json.dumps({"matches": matches})
        except Exception as e:
            logger.error(f"get_apifootball_live failed: {e}")
            return json.dumps({"matches": []})

    async def verify_match_apifootball(self, match_id, fixture_id):
        if self.api_football_service is None:
            return json.dumps({"error": "ApiFootballService not available"})
        try:
            match_id = SecurityValidator.validate_match_id(match_id)
            match = await self.storage_service.get_match(match_id)
            if not match:
                return json.dumps({"error": "Match not found"})
            detail = await self.api_football_service.get_fixture_detail(fixture_id)
            if detail is None:
                return json.dumps({"error": "Could not fetch fixture"})
            events = await self.storage_service.get_match_events(match_id)
            shot_events = [e for e in events if e.get("type") == "goal"]
            detected_home = sum(1 for e in shot_events if e.get("team") == "home")
            detected_away = sum(1 for e in shot_events if e.get("team") == "away")
            api_home = detail.get("home_score") or 0
            api_away = detail.get("away_score") or 0
            match_name = detail.get("home_team", "") + " vs " + detail.get("away_team", "")
            match_ok = (detected_home == api_home) and (detected_away == api_away)
            await self.storage_service.update_match_apifootball(
                match_id, apifb_fixture_id=fixture_id
            )
            return json.dumps(
                {
                    "success": True,
                    "match": match_name,
                    "api_score": f"{api_home}-{api_away}",
                    "detected_score": f"{detected_home}-{detected_away}",
                    "match_ok": match_ok,
                }
            )
        except Exception as e:
            logger.error(f"verify_match_apifootball failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # TheSportsDB
    # ================================================================

    async def check_thesportsdb_status(self):
        if self.thesportsdb_service is None:
            return json.dumps({"available": False})
        try:
            leagues = await self.thesportsdb_service.get_all_leagues()
            available = len(leagues) > 0
            if available:
                self.thesportsdb_service._available = True
            return json.dumps({"available": available})
        except Exception:
            return json.dumps({"available": False})

    async def search_thesportsdb_team(self, query):
        if self.thesportsdb_service is None:
            return json.dumps({"teams": []})
        try:
            teams = await self.thesportsdb_service.search_teams(query)
            return json.dumps(
                {
                    "teams": [
                        {
                            "id": t.id,
                            "name": t.name,
                            "league": t.league_name,
                            "league_id": t.league_id,
                            "badge": t.badge_url,
                            "stadium": t.stadium,
                            "location": t.location,
                            "formed_year": t.formed_year,
                            "api_football_id": t.api_football_id,
                        }
                        for t in teams
                    ]
                }
            )
        except Exception as e:
            logger.error(f"search_thesportsdb_team failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_thesportsdb_standings(self, league_id):
        if self.thesportsdb_service is None:
            return json.dumps({"standings": []})
        try:
            standings = await self.thesportsdb_service.get_standings(league_id)
            return json.dumps(
                {
                    "standings": [
                        {
                            "rank": s.rank,
                            "team": s.team_name,
                            "team_id": s.team_id,
                            "badge": s.badge_url,
                            "played": s.played,
                            "won": s.won,
                            "drawn": s.drawn,
                            "lost": s.lost,
                            "goals_for": s.goals_for,
                            "goals_against": s.goals_against,
                            "goal_diff": s.goal_diff,
                            "points": s.points,
                            "form": s.form,
                            "description": s.description,
                        }
                        for s in standings
                    ]
                }
            )
        except Exception as e:
            logger.error(f"get_thesportsdb_standings failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_thesportsdb_team_events_last(self, team_id):
        if self.thesportsdb_service is None:
            return json.dumps({"events": []})
        try:
            events = await self.thesportsdb_service.get_team_events_last(team_id)
            return json.dumps(
                {
                    "events": [
                        {
                            "id": e.id,
                            "event": e.event_name,
                            "home": e.home_team,
                            "away": e.away_team,
                            "home_score": e.home_score,
                            "away_score": e.away_score,
                            "round": e.round,
                            "date": e.date,
                            "time": e.time,
                            "league": e.league_name,
                        }
                        for e in events
                    ]
                }
            )
        except Exception as e:
            logger.error(f"get_thesportsdb_team_events_last failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_thesportsdb_team_events_next(self, team_id):
        if self.thesportsdb_service is None:
            return json.dumps({"events": []})
        try:
            events = await self.thesportsdb_service.get_team_events_next(team_id)
            return json.dumps(
                {
                    "events": [
                        {
                            "id": e.id,
                            "event": e.event_name,
                            "home": e.home_team,
                            "away": e.away_team,
                            "home_score": e.home_score,
                            "away_score": e.away_score,
                            "round": e.round,
                            "date": e.date,
                            "time": e.time,
                            "league": e.league_name,
                        }
                        for e in events
                    ]
                }
            )
        except Exception as e:
            logger.error(f"get_thesportsdb_team_events_next failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_thesportsdb_team_info(self, team_id):
        if self.thesportsdb_service is None:
            return json.dumps({"team": None})
        try:
            team = await self.thesportsdb_service.get_team(team_id)
            if not team:
                return json.dumps({"team": None})
            return json.dumps(
                {
                    "team": {
                        "id": team.id,
                        "name": team.name,
                        "alternate_name": team.alternate_name,
                        "league": team.league_name,
                        "league_id": team.league_id,
                        "badge": team.badge_url,
                        "stadium": team.stadium,
                        "capacity": team.stadium_capacity,
                        "location": team.location,
                        "formed_year": team.formed_year,
                        "description": team.description[:500] if team.description else "",
                        "api_football_id": team.api_football_id,
                    }
                }
            )
        except Exception as e:
            logger.error(f"get_thesportsdb_team_info failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # StatsBomb
    # ================================================================

    async def check_statsbomb_status(self):
        if self.statsbomb_service is None:
            return json.dumps({"available": False})
        try:
            comps = await self.statsbomb_service.get_competitions()
            available = len(comps) > 0
            return json.dumps(
                {
                    "available": available,
                    "competitions": len(comps),
                }
            )
        except Exception:
            return json.dumps({"available": False})

    async def get_statsbomb_competitions(self):
        if self.statsbomb_service is None:
            return json.dumps({"competitions": []})
        try:
            comps = await self.statsbomb_service.get_competitions()
            return json.dumps(
                {
                    "competitions": [
                        {
                            "competition_id": c.competition_id,
                            "season_id": c.season_id,
                            "name": c.competition_name,
                            "country": c.country_name,
                            "season": c.season_name,
                            "gender": c.competition_gender,
                            "international": c.competition_international,
                            "youth": c.competition_youth,
                            "has_360": c.has_360,
                        }
                        for c in comps
                    ]
                }
            )
        except Exception as e:
            logger.error(f"get_statsbomb_competitions failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_statsbomb_matches(self, competition_id, season_id):
        if self.statsbomb_service is None:
            return json.dumps({"matches": []})
        try:
            matches = await self.statsbomb_service.get_matches(competition_id, season_id)
            return json.dumps(
                {
                    "matches": [
                        {
                            "match_id": m.match_id,
                            "home": m.home_team,
                            "away": m.away_team,
                            "home_score": m.home_score,
                            "away_score": m.away_score,
                            "date": m.match_date,
                            "stage": m.competition_stage,
                            "stadium": m.stadium,
                            "has_360": m.has_360,
                        }
                        for m in matches
                    ]
                }
            )
        except Exception as e:
            logger.error(f"get_statsbomb_matches failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_statsbomb_events(self, match_id):
        if self.statsbomb_service is None:
            return json.dumps({"events": [], "summary": {}})
        try:
            events = await self.statsbomb_service.get_events(match_id)
            shots = [e for e in events if e.event_type == "Shot"]
            passes = [e for e in events if e.event_type == "Pass"]
            total_xg = sum(e.xg for e in shots if e.xg is not None)
            teams = {e.team for e in events if e.team}
            return json.dumps(
                {
                    "summary": {
                        "total_events": len(events),
                        "shots": len(shots),
                        "passes": len(passes),
                        "total_xg": round(total_xg, 3),
                        "teams": sorted(teams),
                    },
                    "shots": [
                        {
                            "minute": s.minute,
                            "team": s.team,
                            "player": s.player,
                            "xg": s.xg,
                            "outcome": s.outcome,
                            "body_part": s.shot_body_part,
                            "type": s.shot_type,
                        }
                        for s in shots[:20]
                    ],
                }
            )
        except Exception as e:
            logger.error(f"get_statsbomb_events failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_statsbomb_lineups(self, match_id):
        if self.statsbomb_service is None:
            return json.dumps({"lineups": []})
        try:
            lineups = await self.statsbomb_service.get_lineups(match_id)
            return json.dumps(
                {
                    "lineups": [
                        {
                            "team": lineup.team_name,
                            "team_id": lineup.team_id,
                            "players": lineup.players,
                        }
                        for lineup in lineups
                    ]
                }
            )
        except Exception as e:
            logger.error(f"get_statsbomb_lineups failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def import_statsbomb_match(self, match_id):
        if self.statsbomb_service is None:
            return json.dumps({"imported": 0, "error": "StatsBomb service not available"})
        storage = self._services.get("storage_service")
        if storage is None:
            return json.dumps({"imported": 0, "error": "Storage service not available"})
        try:
            count = await self.statsbomb_service.import_match_to_db(int(match_id), storage)
            return json.dumps({"imported": count, "match_id": int(match_id)})
        except Exception as e:
            logger.error(f"import_statsbomb_match failed: {e}")
            return json.dumps({"imported": 0, "error": ErrorSanitizer.sanitize_error(e)})

    async def search_statsbomb_team(self, team_name):
        if self.statsbomb_service is None or not team_name.strip():
            return json.dumps({"matches": []})
        try:
            matches = await self.statsbomb_service.search_team_matches(team_name)
            return json.dumps(
                {
                    "matches": [
                        {
                            "match_id": m.match_id,
                            "competition": m.competition_name,
                            "season": m.season_name,
                            "home": m.home_team,
                            "away": m.away_team,
                            "home_score": m.home_score,
                            "away_score": m.away_score,
                            "date": m.match_date,
                        }
                        for m in matches[:30]
                    ]
                }
            )
        except Exception as e:
            logger.error(f"search_statsbomb_team failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # OpenFootball
    # ================================================================

    async def check_openfootball_status(self):
        if self.openfootball_service is None:
            return json.dumps({"available": False})
        try:
            sample = await self.openfootball_service.get_matches("en.1", "2024-25")
            available = len(sample) > 0
            return json.dumps(
                {
                    "available": available,
                    "competitions": len(self.openfootball_service.get_competitions()),
                    "sample_matches": len(sample),
                }
            )
        except Exception:
            return json.dumps({"available": False})

    async def get_openfootball_competitions(self):
        if self.openfootball_service is None:
            return json.dumps({"competitions": []})
        comps = self.openfootball_service.get_competitions()
        return json.dumps(
            {"competitions": [{"id": c.id, "name": c.name, "seasons": c.seasons} for c in comps]}
        )

    async def get_openfootball_matches(self, competition_id, season):
        if self.openfootball_service is None:
            return json.dumps({"matches": []})
        try:
            matches = await self.openfootball_service.get_matches(competition_id, season)
            return json.dumps(
                {
                    "matches": [
                        {
                            "competition": m.competition,
                            "round": m.round,
                            "date": m.date,
                            "time": m.time,
                            "home": m.home_team,
                            "away": m.away_team,
                            "home_score": m.home_score,
                            "away_score": m.away_score,
                            "ht_home": m.half_time_home,
                            "ht_away": m.half_time_away,
                        }
                        for m in matches
                    ]
                }
            )
        except Exception as e:
            logger.error(f"get_openfootball_matches failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def search_openfootball_team(self, team_name):
        if self.openfootball_service is None or not team_name.strip():
            return json.dumps({"matches": []})
        try:
            matches = await self.openfootball_service.search_team_matches(team_name)
            return json.dumps(
                {
                    "matches": [
                        {
                            "competition": m.competition,
                            "season": m.season,
                            "round": m.round,
                            "date": m.date,
                            "home": m.home_team,
                            "away": m.away_team,
                            "home_score": m.home_score,
                            "away_score": m.away_score,
                        }
                        for m in matches[:40]
                    ]
                }
            )
        except Exception as e:
            logger.error(f"search_openfootball_team failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_openfootball_worldcup(self, year):
        if self.openfootball_service is None:
            return json.dumps({"matches": [], "years": []})
        try:
            matches = await self.openfootball_service.get_worldcup_matches(year)
            return json.dumps(
                {
                    "years": self.openfootball_service.get_all_worldcup_years(),
                    "matches": [
                        {
                            "round": m.round,
                            "date": m.date,
                            "home": m.home_team,
                            "away": m.away_team,
                            "home_score": m.home_score,
                            "away_score": m.away_score,
                        }
                        for m in matches
                    ],
                }
            )
        except Exception as e:
            logger.error(f"get_openfootball_worldcup failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})
