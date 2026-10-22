"""Handler for the recruitment & scouting bridge methods — shortlist,
contracts, scout search, opponent database, scouting network, Transfermarkt."""

from __future__ import annotations

import json

from kawkab.core.logging import get_logger
from kawkab.core.security import ErrorSanitizer, SecurityValidator
from kawkab.ui.bridge_handlers.base import BridgeHandlerBase

logger = get_logger(__name__)


class RecruitmentHandler(BridgeHandlerBase):
    """Recruitment-hub bridge surface: scout/shortlist/contracts/opponents."""

    @property
    def storage_service(self):
        return self._services.get("storage_service")

    # ================================================================
    # Wave E — Scout Portal
    # ================================================================

    async def scout_search_players(self, query, position=""):
        try:
            self._check_rate_limit()
            # Try real player search database first
            try:
                import os

                from kawkab.core.player_search import SearchCriteria, search_players

                db_path = os.path.join(
                    os.path.dirname(__file__), "..", "..", "..", "data", "players.json"
                )
                if os.path.exists(db_path):
                    with open(db_path) as f:
                        import json as _json

                        player_db = _json.load(f)
                else:
                    player_db = []
                criteria = SearchCriteria(limit=20)
                if position:
                    criteria.positions = [position.upper()]
                results = search_players(criteria, player_db) if player_db else []
                query_lower = query.lower().strip()
                if query_lower:
                    results = [r for r in results if query_lower in (r.player_name or "").lower()]
                player_list = []
                for r in results:
                    player_list.append(
                        {
                            "track_id": abs(hash(r.player_id)) % 100000,
                            "name": r.player_name,
                            "position": r.position,
                            "team": r.team if hasattr(r, "team") else "",
                            "age": r.age,
                            "matches": r.stats.get("matches", 0),
                            "goals": r.stats.get("goals", 0),
                            "assists": r.stats.get("assists", 0),
                            "xg": r.stats.get("xg", 0),
                            "passes": r.stats.get("passes", 0),
                            "tackles": r.stats.get("tackles", 0),
                        }
                    )
                return json.dumps({"results": player_list, "total": len(player_list)})
            except Exception as exc:
                # Honest failure: no fabricated players. A coaching tool that
                # invents Haaland stats on an internal error is worse than an
                # empty result -- the coach would brief a scout report on
                # fiction. Surface the error; the UI renders zero results.
                logger.error(f"scout_search_players failed: {exc}")
                return json.dumps(
                    {
                        "results": [],
                        "total": 0,
                        "error": ErrorSanitizer.sanitize_error(exc),
                    }
                )
        except Exception as e:
            logger.error(f"scout_search_players failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def search_external_player(self, query, position=""):
        """Best-effort enrichment: merge live external-provider results into
        the local scout search (called by the frontend alongside, not
        instead of, ``scout_search_players``).

        None of the configured external providers (football-data.org,
        API-Football, Bzzoiro, TheSportsDB, EasySoccerData) expose a
        player-name search endpoint on their free tiers — each only
        supports team search or player-lookup-by-ID. Rather than fabricate
        a cross-provider name-matching heuristic this build can't validate
        against live data, this honestly returns an empty player list so
        the frontend's existing "best-effort, silently degrade" merge path
        (see app-scout.js) is a genuine no-op instead of a broken bridge
        call. Revisit if a provider adds player-name search.
        """
        try:
            self._check_rate_limit()
            query = SecurityValidator.sanitize_string(query, max_length=200) if query else ""
            return json.dumps({"players": [], "query": query, "position": position})
        except Exception as e:
            logger.error(f"search_external_player failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_shortlist(self):
        try:
            self._check_rate_limit()
            storage = self.storage_service
            players = await storage.get_shortlist() if storage else []
            return json.dumps({"players": players, "total": len(players)})
        except Exception as e:
            logger.error(f"get_shortlist failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def add_shortlist_entry(self, entry_json):
        try:
            self._check_rate_limit()
            entry = json.loads(entry_json or "{}")
            if not entry.get("player_id") or not entry.get("player_name"):
                return json.dumps(
                    {"success": False, "error": "player_id and player_name are required"}
                )
            entry.setdefault("status", "shortlisted")
            storage = self.storage_service
            if not storage:
                return json.dumps({"success": False, "error": "Storage not initialized"})
            entry_id = await storage.save_shortlist_entry(entry)
            return json.dumps({"success": entry_id > 0, "id": entry_id})
        except Exception as e:
            logger.error(f"add_shortlist_entry failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def update_shortlist_entry(self, entry_id, updates_json):
        try:
            self._check_rate_limit()
            updates = json.loads(updates_json or "{}")
            storage = self.storage_service
            if not storage:
                return json.dumps({"success": False, "error": "Storage not initialized"})
            ok = await storage.update_shortlist_entry(int(entry_id), updates)
            return json.dumps({"success": ok})
        except Exception as e:
            logger.error(f"update_shortlist_entry failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def delete_shortlist_entry(self, entry_id):
        try:
            self._check_rate_limit()
            storage = self.storage_service
            if not storage:
                return json.dumps({"success": False, "error": "Storage not initialized"})
            ok = await storage.delete_shortlist_entry(int(entry_id))
            return json.dumps({"success": ok})
        except Exception as e:
            logger.error(f"delete_shortlist_entry failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_contracts(self):
        try:
            self._check_rate_limit()
            storage = self.storage_service
            if not storage:
                return json.dumps({"contracts": [], "expiring_soon": []})
            contracts = await storage.get_contracts()
            expiring = await storage.get_contracts_expiring_soon(90)
            return json.dumps({"contracts": contracts, "expiring_soon": expiring})
        except Exception as e:
            logger.error(f"get_contracts failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def add_contract(self, contract_json):
        try:
            self._check_rate_limit()
            contract = json.loads(contract_json or "{}")
            required = ("player_profile_id", "player_name", "start_date", "end_date")
            missing = [k for k in required if not contract.get(k)]
            if missing:
                return json.dumps(
                    {"success": False, "error": f"missing required fields: {missing}"}
                )
            storage = self.storage_service
            if not storage:
                return json.dumps({"success": False, "error": "Storage not initialized"})
            contract_id = await storage.save_contract(contract)
            return json.dumps({"success": contract_id > 0, "id": contract_id})
        except Exception as e:
            logger.error(f"add_contract failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_contract_alerts(self):
        try:
            self._check_rate_limit()
            storage = self.storage_service
            if not storage:
                return json.dumps({"alerts": []})
            expiring = await storage.get_contracts_expiring_soon(180)
            alerts = [
                {
                    "level": "critical" if self._days_until(c["end_date"]) <= 30 else "warning",
                    "player": c["player_name"],
                    "end_date": c["end_date"],
                    "message": f"Contract ends {c['end_date']}",
                }
                for c in expiring
            ]
            return json.dumps({"alerts": alerts})
        except Exception as e:
            logger.error(f"get_contract_alerts failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    @staticmethod
    def _days_until(date_str):
        try:
            from datetime import date

            y, m, d = str(date_str).split("-")
            return (date(int(y), int(m), int(d)) - date.today()).days
        except Exception:
            return 9999

    async def recruit_from_search(self, player_id, source, player_json):
        """One-click recruit: an external search result becomes a shortlist row.

        Bridges the external scouting sources (Transfermarkt / scout network)
        into the local shortlist so a coach can act on what they find without
        retyping it.
        """
        try:
            self._check_rate_limit()
            data = json.loads(player_json or "{}")
            entry = {
                "player_id": f"{source}:{player_id}",
                "player_name": data.get("player_name") or data.get("name") or str(player_id),
                "position": data.get("position", ""),
                "team": data.get("team") or data.get("club", ""),
                "league": data.get("league", ""),
                "priority": "medium",
                "status": "scouted",
                "notes": f"Recruited from {source} search",
                "scout_rating": float(data.get("scout_rating", 0.0) or 0.0),
                "estimated_value": data.get("estimated_value") or data.get("market_value"),
                "age": data.get("age"),
                "nationality": data.get("nationality", ""),
            }
            return await self.add_shortlist_entry(json.dumps(entry))
        except Exception as e:
            logger.error(f"recruit_from_search failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Phase 13 — Opponent Database + Scouting Network + Transfermarkt
    # ================================================================

    def _get_opponent_db(self):
        svc = self._services.get("opponent_database_service")
        if svc is None:
            from kawkab.services.opponent_database_service import OpponentDatabaseService

            svc = OpponentDatabaseService()
            self._services["opponent_database_service"] = svc
        return svc

    def _get_scouting_network(self):
        svc = self._services.get("scouting_network_service")
        if svc is None:
            from kawkab.services.scouting_network_service import ScoutingNetworkService

            svc = ScoutingNetworkService()
            self._services["scouting_network_service"] = svc
        return svc

    def _get_transfermarkt(self):
        svc = self._services.get("transfermarkt_integration_service")
        if svc is None:
            from kawkab.services.transfermarkt_integration_service import (
                TransfermarktIntegrationService,
            )

            svc = TransfermarktIntegrationService()
            self._services["transfermarkt_integration_service"] = svc
        return svc

    async def opponent_list(self):
        try:
            svc = self._get_opponent_db()
            return json.dumps({"success": True, "profiles": svc.list_profiles()})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def opponent_get(self, profile_id):
        try:
            svc = self._get_opponent_db()
            profile = svc.get_profile(str(profile_id))
            if profile:
                matchups = svc.get_matchups(str(profile_id))
                return json.dumps({"success": True, "profile": profile, "matchups": matchups})
            return json.dumps({"success": False, "error": "Not found"})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def opponent_create(self, team_name, league, country):
        try:
            svc = self._get_opponent_db()
            result = svc.create_profile(str(team_name), str(league or ""), str(country or ""))
            return json.dumps({"success": True, "profile": result})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def opponent_update(self, profile_id, updates_json):
        try:
            svc = self._get_opponent_db()
            updates = json.loads(updates_json)
            ok = svc.update_profile(str(profile_id), updates)
            return json.dumps({"success": ok})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def opponent_delete(self, profile_id):
        try:
            svc = self._get_opponent_db()
            ok = svc.delete_profile(str(profile_id))
            return json.dumps({"success": ok})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def opponent_add_matchup(
        self,
        profile_id,
        our_team,
        date,
        competition,
        home_away,
        our_score,
        their_score,
        our_xg,
        their_xg,
        notes,
    ):
        try:
            svc = self._get_opponent_db()
            result = svc.add_matchup(
                opponent_id=str(profile_id),
                our_team=str(our_team),
                date=str(date),
                competition=str(competition or ""),
                home_away=str(home_away or "home"),
                our_score=int(our_score or 0),
                their_score=int(their_score or 0),
                our_xg=float(our_xg or 0.0),
                their_xg=float(their_xg or 0.0),
                notes=str(notes or ""),
            )
            return json.dumps({"success": True, "matchup": result})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def opponent_scouting_report(self, profile_id):
        try:
            svc = self._get_opponent_db()
            report = svc.generate_scouting_report(str(profile_id))
            return json.dumps({"success": True, "report": report})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def scout_network_search(self, query, position, min_age, max_age, league, min_rating):
        try:
            svc = self._get_scouting_network()
            results = svc.search_players(
                query=str(query or ""),
                position=str(position or ""),
                min_age=int(min_age or 0),
                max_age=int(max_age or 99),
                league=str(league or ""),
                min_rating=float(min_rating or 0.0),
            )
            return json.dumps({"success": True, "players": results})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def scout_network_add(
        self,
        name,
        position,
        club,
        league,
        rating,
        strengths_json,
        weaknesses_json,
        scout_notes,
        submitted_by,
        tags_json,
    ):
        try:
            svc = self._get_scouting_network()
            strengths = json.loads(strengths_json) if strengths_json else []
            weaknesses = json.loads(weaknesses_json) if weaknesses_json else []
            tags = json.loads(tags_json) if tags_json else []
            result = svc.add_player(
                name=str(name),
                position=str(position or ""),
                club=str(club or ""),
                league=str(league or ""),
                rating=float(rating or 0.0),
                strengths=strengths,
                weaknesses=weaknesses,
                scout_notes=str(scout_notes or ""),
                submitted_by=str(submitted_by or ""),
                tags=tags,
            )
            return json.dumps({"success": True, "player": result})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def scout_network_get(self, player_id):
        try:
            svc = self._get_scouting_network()
            player = svc.get_player(str(player_id))
            if player:
                return json.dumps({"success": True, "player": player})
            return json.dumps({"success": False, "error": "Not found"})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def scout_network_delete(self, player_id):
        try:
            svc = self._get_scouting_network()
            ok = svc.delete_player(str(player_id))
            return json.dumps({"success": ok})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def scout_network_stats(self):
        try:
            svc = self._get_scouting_network()
            stats = svc.get_stats()
            return json.dumps({"success": True, "stats": stats})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def transfermarkt_search(self, name):
        try:
            svc = self._get_transfermarkt()
            results = svc.search_player(str(name))
            # Honest states: the service never fabricates players (the old
            # demo-data path returned "Demo FC" squads for any query), so
            # an empty list means exactly that — surface the reason.
            return json.dumps(
                {
                    "success": True,
                    "results": results,
                    "data_available": bool(results),
                    "provider_status": svc.provider_status(),
                }
            )
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def transfermarkt_get(self, player_id):
        try:
            svc = self._get_transfermarkt()
            details = svc.get_player_details(int(player_id))
            return json.dumps(
                {
                    "success": True,
                    "details": details,
                    "data_available": bool(details.get("data_available", True)),
                }
            )
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def transfermarkt_squad(self, club_name):
        try:
            svc = self._get_transfermarkt()
            squad = svc.get_club_squad(str(club_name))
            return json.dumps(
                {
                    "success": True,
                    "squad": squad,
                    "data_available": bool(squad),
                    "provider_status": svc.provider_status(),
                }
            )
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})
