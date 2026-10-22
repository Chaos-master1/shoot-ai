"""Tests for CrossMatchLinkingService."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from tests.conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()
_svc = load_service_module("cross_match_test", "cross_match_linking_service.py")
CrossMatchLinkingService = _svc.CrossMatchLinkingService
AUTO_LINK_THRESHOLD = _svc.AUTO_LINK_THRESHOLD
FLAG_REVIEW_THRESHOLD = _svc.FLAG_REVIEW_THRESHOLD


def _make_embedding(dim: int = 512, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    emb = rng.randn(dim).astype(np.float32)
    return emb / max(np.linalg.norm(emb), 1e-8)


class FakeRow:
    def __init__(self, data: dict):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data.items())

    def keys(self):
        return self._data.keys()

    def get(self, key, default=None):
        return self._data.get(key, default)


@pytest.fixture
def mock_storage():
    storage = MagicMock()
    storage.get_all_matches = AsyncMock(return_value=[])
    storage.get_match_players = AsyncMock(return_value=[])
    storage.get_all_player_profiles = AsyncMock(return_value=[])
    return storage


@pytest.fixture
def mock_profile_svc():
    """Return a mock PlayerProfileService for injection."""
    mock_inst = MagicMock()
    mock_inst.link_match_player = AsyncMock(return_value=True)
    return mock_inst


class TestCrossMatchLinkingService:
    @pytest.mark.asyncio
    async def test_initialization(self, mock_storage, mock_profile_svc):
        svc = CrossMatchLinkingService(mock_storage)
        assert svc._storage is mock_storage
        assert svc._profile_svc is not None

    @pytest.mark.asyncio
    async def test_link_match_no_players(self, mock_storage, mock_profile_svc):
        mock_storage.get_match_players.return_value = []
        svc = CrossMatchLinkingService(mock_storage, profile_service=mock_profile_svc)
        result = await svc.link_match(1)
        assert result["linked"] == 0
        assert result["flagged_for_review"] == 0
        assert len(result["proposals"]) == 0

    @pytest.mark.asyncio
    async def test_link_match_no_profiles_creates_proposal(self, mock_storage, mock_profile_svc):
        emb = _make_embedding(seed=42)
        emb_b64 = json.dumps(emb.tolist())
        mock_storage.get_match_players.return_value = [
            {
                "track_id": 101,
                "name": "Player A",
                "team": "home",
                "face_embedding": emb_b64,
                "reid_embedding": None,
                "jersey_number": 10,
                "track_data": None,
            }
        ]
        mock_storage.get_all_player_profiles.return_value = []
        svc = CrossMatchLinkingService(mock_storage, profile_service=mock_profile_svc)
        result = await svc.link_match(1)
        assert result["linked"] == 0
        assert result["flagged_for_review"] == 0
        assert len(result["proposals"]) == 1
        assert result["proposals"][0]["action"] == "no_match"

    @pytest.mark.asyncio
    async def test_link_match_with_matching_embedding(self, mock_storage, mock_profile_svc):
        emb = _make_embedding(seed=42)
        emb_b64 = json.dumps(emb.tolist())
        match_emb = emb + 0.005 * np.random.RandomState(99).randn(512).astype(np.float32)
        match_emb = match_emb / max(np.linalg.norm(match_emb), 1e-8)
        match_emb_b64 = json.dumps(match_emb.tolist())

        mock_storage.get_match_players.return_value = [
            {
                "track_id": 101,
                "name": "Player A",
                "team": "home",
                "face_embedding": emb_b64,
                "reid_embedding": None,
                "jersey_number": 10,
                "track_data": None,
            }
        ]
        mock_storage.get_all_player_profiles.return_value = [
            {
                "id": 1,
                "display_name": "Existing Player",
                "jersey_number": 10,
                "team": "home",
                "face_embedding": match_emb_b64,
                "reid_embedding": None,
            }
        ]

        svc = CrossMatchLinkingService(mock_storage, profile_service=mock_profile_svc)
        result = await svc.link_match(1)
        assert result["linked"] >= 1
        assert len(result["proposals"]) == 1
        assert result["proposals"][0]["action"] == "auto_linked"

    @pytest.mark.asyncio
    async def test_flag_for_review_threshold(self, mock_storage, mock_profile_svc):
        emb = _make_embedding(seed=1)
        emb_b64 = json.dumps(emb.tolist())
        diff_emb = np.random.RandomState(2).randn(512).astype(np.float32)
        diff_emb = diff_emb / max(np.linalg.norm(diff_emb), 1e-8)
        diff_emb_b64 = json.dumps(diff_emb.tolist())

        mock_storage.get_match_players.return_value = [
            {
                "track_id": 201,
                "name": "Player B",
                "team": "away",
                "face_embedding": emb_b64,
                "reid_embedding": None,
                "jersey_number": 7,
                "track_data": None,
            }
        ]
        mock_storage.get_all_player_profiles.return_value = [
            {
                "id": 2,
                "display_name": "Different Player",
                "jersey_number": 7,
                "team": "away",
                "face_embedding": diff_emb_b64,
                "reid_embedding": None,
            }
        ]

        svc = CrossMatchLinkingService(mock_storage, profile_service=mock_profile_svc)
        result = await svc.link_match(1)

        if len(result["proposals"]) > 0:
            p = result["proposals"][0]
            assert p["action"] in ("flag_for_review", "no_match")

    @pytest.mark.asyncio
    async def test_empty_match_no_players(self, mock_storage, mock_profile_svc):
        mock_storage.get_match_players.return_value = []
        svc = CrossMatchLinkingService(mock_storage, profile_service=mock_profile_svc)
        result = await svc.link_match(42)
        assert result["match_id"] == 42
        assert result["linked"] == 0
        assert result["flagged_for_review"] == 0

    @pytest.mark.asyncio
    async def test_link_match_db_error(self, mock_storage, mock_profile_svc):
        mock_storage.get_match_players.side_effect = Exception("DB error")
        svc = CrossMatchLinkingService(mock_storage, profile_service=mock_profile_svc)
        with pytest.raises(Exception, match="DB error"):
            await svc.link_match(1)

    @pytest.mark.asyncio
    async def test_link_all_matches_multiple(self, mock_storage, mock_profile_svc):
        emb1 = _make_embedding(seed=10)
        emb1_b64 = json.dumps(emb1.tolist())
        emb2 = _make_embedding(seed=20)
        emb2_b64 = json.dumps(emb2.tolist())

        mock_storage.get_all_matches.return_value = [
            {"id": 1, "name": "Match 1"},
            {"id": 2, "name": "Match 2"},
        ]

        def get_players_side_effect(match_id):
            if match_id == 1:
                return [
                    {
                        "track_id": 301,
                        "name": "P1",
                        "team": "home",
                        "face_embedding": emb1_b64,
                        "reid_embedding": None,
                        "jersey_number": 5,
                        "track_data": None,
                    }
                ]
            return [
                {
                    "track_id": 401,
                    "name": "P2",
                    "team": "away",
                    "face_embedding": emb2_b64,
                    "reid_embedding": None,
                    "jersey_number": 9,
                    "track_data": None,
                }
            ]

        mock_storage.get_match_players = AsyncMock(side_effect=get_players_side_effect)
        mock_storage.get_all_player_profiles.return_value = []

        svc = CrossMatchLinkingService(mock_storage, profile_service=mock_profile_svc)
        summary = await svc.link_all_matches()
        assert summary["matches_processed"] == 2
        assert summary["total_linked"] == 0

    @pytest.mark.asyncio
    async def test_link_all_matches_empty(self, mock_storage, mock_profile_svc):
        mock_storage.get_all_matches.return_value = []
        svc = CrossMatchLinkingService(mock_storage, profile_service=mock_profile_svc)
        summary = await svc.link_all_matches()
        assert summary["matches_processed"] == 0
        assert summary["total_linked"] == 0
        assert summary["total_flagged_for_review"] == 0
        assert summary["match_results"] == []

    @pytest.mark.asyncio
    async def test_auto_link_vs_flag_thresholds(self, mock_storage, mock_profile_svc):
        emb = _make_embedding(seed=5)
        emb_b64 = json.dumps(emb.tolist())
        close_emb = emb + 0.002 * np.random.RandomState(55).randn(512).astype(np.float32)
        close_emb = close_emb / max(np.linalg.norm(close_emb), 1e-8)
        close_b64 = json.dumps(close_emb.tolist())
        far_emb = np.random.RandomState(99).randn(512).astype(np.float32)
        far_emb = far_emb / max(np.linalg.norm(far_emb), 1e-8)
        far_b64 = json.dumps(far_emb.tolist())

        mock_storage.get_match_players.return_value = [
            {
                "track_id": 501,
                "name": "Close",
                "team": "home",
                "face_embedding": emb_b64,
                "reid_embedding": None,
                "jersey_number": 1,
                "track_data": None,
            },
            {
                "track_id": 502,
                "name": "Far",
                "team": "home",
                "face_embedding": far_b64,
                "reid_embedding": None,
                "jersey_number": 2,
                "track_data": None,
            },
        ]
        mock_storage.get_all_player_profiles.return_value = [
            {
                "id": 10,
                "display_name": "Profile A",
                "jersey_number": 1,
                "team": "home",
                "face_embedding": close_b64,
                "reid_embedding": None,
            },
        ]

        svc = CrossMatchLinkingService(mock_storage, profile_service=mock_profile_svc)
        result = await svc.link_match(1)
        linked = sum(1 for p in result["proposals"] if p["action"] == "auto_linked")
        no_match = sum(1 for p in result["proposals"] if p["action"] == "no_match")
        assert linked == 1
        assert no_match == 1

    def test_get_player_face_embedding(self):
        emb = _make_embedding(seed=7)
        emb_b64 = json.dumps(emb.tolist())
        player = {"track_id": 601, "face_embedding": emb_b64, "reid_embedding": None}
        storage = MagicMock()
        svc = CrossMatchLinkingService(storage)
        extracted = svc._get_player_embedding(player)
        assert extracted is not None
        assert np.allclose(extracted, emb)

    def test_get_player_reid_embedding(self):
        emb = _make_embedding(seed=8)
        emb_b64 = json.dumps(emb.tolist())
        player = {"track_id": 701, "face_embedding": None, "reid_embedding": emb_b64}
        storage = MagicMock()
        svc = CrossMatchLinkingService(storage)
        extracted = svc._get_player_embedding(player)
        assert extracted is not None
        assert np.allclose(extracted, emb)

    def test_load_profile_embeddings(self):
        emb = _make_embedding(seed=9)
        emb_b64 = json.dumps(emb.tolist())
        profiles = [
            {
                "id": 100,
                "display_name": "P1",
                "jersey_number": 3,
                "team": "home",
                "face_embedding": emb_b64,
                "reid_embedding": None,
            },
            {
                "id": 101,
                "display_name": "P2",
                "jersey_number": 4,
                "team": "away",
                "face_embedding": None,
                "reid_embedding": None,
            },
        ]
        storage = MagicMock()
        svc = CrossMatchLinkingService(storage)
        loaded = svc._load_profile_embeddings(profiles)
        assert len(loaded) == 1
        assert loaded[0]["profile_id"] == 100

    def test_find_best_match(self):
        emb = _make_embedding(seed=11)
        close_emb = emb + 0.01 * np.random.RandomState(111).randn(512).astype(np.float32)
        close_emb = close_emb / max(np.linalg.norm(close_emb), 1e-8)
        profile_embs = [
            {"profile_id": 200, "display_name": "Target", "embedding": close_emb},
            {
                "profile_id": 201,
                "display_name": "Far",
                "embedding": np.random.RandomState(222).randn(512).astype(np.float32),
            },
        ]
        storage = MagicMock()
        svc = CrossMatchLinkingService(storage)
        match = svc._find_best_match(emb, profile_embs)
        assert match is not None
        assert match["profile_id"] == 200

    def test_find_best_match_empty(self):
        storage = MagicMock()
        svc = CrossMatchLinkingService(storage)
        match = svc._find_best_match(np.array([1.0, 2.0]), [])
        assert match is None
