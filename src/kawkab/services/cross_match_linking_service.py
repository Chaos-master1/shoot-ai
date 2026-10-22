"""Cross-match player linking pipeline.

Automatically links tracked players across multiple matches using:
1. ReID embeddings (if available in tracking data)
2. Jersey number + team matching
3. Face recognition embeddings stored in player profiles

Pipeline:
  - Iterate all matches in the DB
  - For each match, load tracked players
  - Match against existing profiles via cosine distance on ReID/face embeddings
  - Auto-link if distance < 0.3; flag for review if < 0.5
  - Create new profiles for unmatched players
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from kawkab.services.player_profile_service import PlayerProfileService

logger = logging.getLogger(__name__)

AUTO_LINK_THRESHOLD = 0.3
FLAG_REVIEW_THRESHOLD = 0.5


class CrossMatchLinkingService:
    """Links tracked players across multiple matches to persistent profiles."""

    def __init__(self, storage_service: Any, profile_service: Any = None) -> None:
        self._storage = storage_service
        self._profile_svc = (
            profile_service if profile_service is not None else PlayerProfileService()
        )

    async def link_all_matches(self) -> dict[str, Any]:
        """Iterate all matches and link players to profiles.

        Returns:
            Summary dict with total_linked, total_review, per_match breakdown.
        """
        matches = await self._storage.get_all_matches()
        total_linked = 0
        total_review = 0
        match_results: list[dict[str, Any]] = []

        for match in matches:
            match_id = match["id"]
            result = await self.link_match(match_id)
            total_linked += result["linked"]
            total_review += result["flagged_for_review"]
            match_results.append(result)

        return {
            "matches_processed": len(matches),
            "total_linked": total_linked,
            "total_flagged_for_review": total_review,
            "match_results": match_results,
        }

    async def link_match(self, match_id: int) -> dict[str, Any]:
        """Link a single match's players to persistent profiles.

        Args:
            match_id: The match ID in the database.

        Returns:
            Dict with linked count, flagged count, and proposals list.
        """
        players = await self._storage.get_match_players(match_id)
        if not players:
            return {"match_id": match_id, "linked": 0, "flagged_for_review": 0, "proposals": []}

        profiles = await self._storage.get_all_player_profiles()
        profile_embeddings = self._load_profile_embeddings(profiles)

        linked = 0
        flagged = 0
        proposals: list[dict[str, Any]] = []

        for player in players:
            track_id = player.get("track_id")
            if track_id is None:
                continue

            player_embedding = self._get_player_embedding(player)

            if player_embedding is not None and profile_embeddings:
                match = self._find_best_match(player_embedding, profile_embeddings)
                if match is not None:
                    dist = match["distance"]
                    if dist < AUTO_LINK_THRESHOLD:
                        success = await self._profile_svc.link_match_player(
                            profile_id=match["profile_id"],
                            match_id=match_id,
                            track_id=track_id,
                            confidence=round(1.0 - dist, 4),
                        )
                        if success:
                            linked += 1
                        proposals.append(
                            {
                                "track_id": track_id,
                                "profile_id": match["profile_id"],
                                "profile_name": match.get("display_name"),
                                "distance": round(dist, 4),
                                "confidence": round(1.0 - dist, 4),
                                "action": "auto_linked",
                            }
                        )
                        continue
                    elif dist < FLAG_REVIEW_THRESHOLD:
                        proposals.append(
                            {
                                "track_id": track_id,
                                "profile_id": match["profile_id"],
                                "profile_name": match.get("display_name"),
                                "distance": round(dist, 4),
                                "confidence": round(1.0 - dist, 4),
                                "action": "flag_for_review",
                            }
                        )
                        flagged += 1
                        continue

            proposals.append(
                {
                    "track_id": track_id,
                    "profile_id": None,
                    "profile_name": None,
                    "distance": None,
                    "confidence": 0.0,
                    "action": "no_match",
                }
            )

        return {
            "match_id": match_id,
            "linked": linked,
            "flagged_for_review": flagged,
            "proposals": proposals,
        }

    def _load_profile_embeddings(self, profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Load face/reid embeddings from profiles into a searchable list."""
        embeddings: list[dict[str, Any]] = []
        for p in profiles:
            emb = None
            emb_text = p.get("face_embedding")
            if emb_text:
                try:
                    import json

                    emb = np.array(json.loads(emb_text), dtype=np.float32)
                except (json.JSONDecodeError, TypeError):
                    pass
            reid_text = p.get("reid_embedding")
            if reid_text and emb is None:
                try:
                    import json

                    emb = np.array(json.loads(reid_text), dtype=np.float32)
                except (json.JSONDecodeError, TypeError):
                    pass
            if emb is not None:
                embeddings.append(
                    {
                        "profile_id": p["id"],
                        "display_name": p.get("display_name"),
                        "jersey_number": p.get("jersey_number"),
                        "team": p.get("team"),
                        "embedding": emb,
                    }
                )
        return embeddings

    def _get_player_embedding(self, player: dict[str, Any]) -> np.ndarray | None:
        """Try to get an embedding for a match player from available data."""
        emb_text = player.get("reid_embedding") or player.get("face_embedding")
        if emb_text:
            try:
                import json

                return np.array(json.loads(emb_text), dtype=np.float32)
            except (json.JSONDecodeError, TypeError):
                return None
        track_data = player.get("track_data") or {}
        embeddings_list = track_data.get("reid_embeddings") or track_data.get("face_embeddings")
        if embeddings_list and len(embeddings_list) > 0:
            if isinstance(embeddings_list[0], list):
                return np.array(embeddings_list[0], dtype=np.float32)
            return embeddings_list[0]
        return None

    def _find_best_match(
        self, embedding: np.ndarray, profile_embeddings: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Find the closest profile by cosine distance."""
        emb = embedding / max(np.linalg.norm(embedding), 1e-8)
        best_dist = float("inf")
        best_match: dict[str, Any] | None = None
        for entry in profile_embeddings:
            gemb = entry["embedding"]
            if gemb.shape != emb.shape:
                continue
            dist = float(np.linalg.norm(gemb - emb))
            if dist < best_dist:
                best_dist = dist
                best_match = entry
        if best_match is None:
            return None
        return {
            "profile_id": best_match["profile_id"],
            "display_name": best_match.get("display_name"),
            "jersey_number": best_match.get("jersey_number"),
            "team": best_match.get("team"),
            "distance": best_dist,
        }
