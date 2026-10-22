"""Tests for PlayerProfileService."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()


class FakeRow:
    """Simulates sqlite3.Row with dict-like access."""

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


def _make_row(data: dict):
    return FakeRow(data)


@pytest.fixture(scope="module")
def pp_mod():
    return load_service_module(
        "kawkab.services.player_profile_service", "player_profile_service.py"
    )


class TestPlayerProfileDataclasses:
    def test_player_profile_defaults(self, pp_mod):
        p = pp_mod.PlayerProfile(id=1, global_id="g1")
        assert p.id == 1
        assert p.global_id == "g1"
        assert p.is_active is True
        assert p.team == "home"

    def test_player_profile_all_fields(self, pp_mod):
        p = pp_mod.PlayerProfile(
            id=1,
            global_id="g1",
            display_name="Messi",
            jersey_number=10,
            preferred_position="FW",
            height_cm=170,
            weight_kg=72,
            dominant_foot="left",
            date_of_birth="1987-06-24",
            nationality="Argentina",
            photo_path="/photos/messi.jpg",
            team="home",
            is_active=True,
            created_at="2024-01-01",
            updated_at="2024-01-02",
            matches_played=500,
            total_distance_km=450.0,
            avg_max_speed_kmh=32.0,
            total_goals=700,
            total_assists=300,
            pass_accuracy_avg=0.85,
        )
        assert p.display_name == "Messi"
        assert p.total_goals == 700

    def test_player_match_appearance_defaults(self, pp_mod):
        a = pp_mod.PlayerMatchAppearance(
            match_id=1,
            match_name="M1",
            match_date="2024-01-01",
            opponent="Opp",
            team="home",
            jersey_number=10,
            position="FW",
        )
        assert a.minutes_played == 0.0
        assert a.xg == 0.0


class TestPlayerProfileService:
    def _mock_db(self, svc, rows_by_query: dict):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor

        def _execute(sql, params=None):
            for snippet, result in rows_by_query.items():
                if snippet in sql:
                    if result is None:
                        cursor.fetchone.return_value = None
                        cursor.fetchall.return_value = []
                    elif isinstance(result, list):
                        cursor.fetchall.return_value = [_make_row(r) for r in result]
                        cursor.fetchone.return_value = _make_row(result[0]) if result else None
                    elif isinstance(result, dict):
                        cursor.fetchone.return_value = _make_row(result)
                        cursor.fetchall.return_value = [_make_row(result)]
                    break
            cursor.lastrowid = 42
            return cursor

        cursor.execute = _execute
        conn.commit = MagicMock()
        svc._conn = conn
        return svc

    @pytest.mark.asyncio
    async def test_create_profile(self, pp_mod):
        svc = pp_mod.PlayerProfileService()
        svc = self._mock_db(
            svc,
            {
                "FROM player_profiles WHERE id = ?": {
                    "id": 42,
                    "global_id": "g1",
                    "display_name": "Test",
                    "jersey_number": 10,
                    "preferred_position": "FW",
                    "height_cm": 180,
                    "weight_kg": 75,
                    "dominant_foot": "right",
                    "date_of_birth": "2000-01-01",
                    "nationality": "Test",
                    "team": "home",
                    "is_active": 1,
                    "photo_path": None,
                    "created_at": "2024-01-01",
                    "updated_at": None,
                },
            },
        )
        profile = await svc.create_profile(
            global_id="g1",
            display_name="Test",
            jersey_number=10,
            preferred_position="FW",
            height_cm=180,
            weight_kg=75,
            dominant_foot="right",
            team="home",
        )
        assert profile is not None
        assert profile.display_name == "Test"
        assert profile.jersey_number == 10

    @pytest.mark.asyncio
    async def test_create_profile_auto_global_id(self, pp_mod, monkeypatch):
        svc = pp_mod.PlayerProfileService()
        svc = self._mock_db(
            svc,
            {
                "FROM player_profiles WHERE id = ?": {
                    "id": 42,
                    "global_id": "auto_123",
                    "display_name": "Auto",
                    "jersey_number": 7,
                    "preferred_position": "MF",
                    "height_cm": None,
                    "weight_kg": None,
                    "dominant_foot": None,
                    "date_of_birth": None,
                    "nationality": None,
                    "team": "home",
                    "is_active": 1,
                    "photo_path": None,
                    "created_at": "2024-01-01",
                    "updated_at": None,
                },
            },
        )
        profile = await svc.create_profile(display_name="Auto")
        assert profile is not None

    @pytest.mark.asyncio
    async def test_get_profile_found(self, pp_mod):
        svc = pp_mod.PlayerProfileService()
        svc = self._mock_db(
            svc,
            {
                "FROM player_profiles WHERE id = ?": {
                    "id": 1,
                    "global_id": "g1",
                    "display_name": "Found",
                    "jersey_number": 9,
                    "preferred_position": "ST",
                    "height_cm": 185,
                    "weight_kg": 80,
                    "dominant_foot": "right",
                    "date_of_birth": "1995-05-05",
                    "nationality": "BR",
                    "team": "home",
                    "is_active": 1,
                    "photo_path": None,
                    "created_at": "2024-01-01",
                    "updated_at": None,
                },
            },
        )
        profile = await svc.get_profile(1)
        assert profile is not None
        assert profile.display_name == "Found"

    @pytest.mark.asyncio
    async def test_get_profile_not_found(self, pp_mod):
        svc = pp_mod.PlayerProfileService()
        svc = self._mock_db(
            svc,
            {
                "FROM player_profiles WHERE id = ?": None,
            },
        )
        profile = await svc.get_profile(999)
        assert profile is None

    @pytest.mark.asyncio
    async def test_get_profile_by_global_id(self, pp_mod):
        svc = pp_mod.PlayerProfileService()
        svc = self._mock_db(
            svc,
            {
                "FROM player_profiles WHERE global_id = ?": {
                    "id": 2,
                    "global_id": "ext_1",
                    "display_name": "External",
                    "jersey_number": 5,
                    "preferred_position": "DF",
                    "height_cm": 190,
                    "weight_kg": 85,
                    "dominant_foot": "left",
                    "date_of_birth": "1990-03-15",
                    "nationality": "DE",
                    "team": "away",
                    "is_active": 1,
                    "photo_path": None,
                    "created_at": "2024-01-01",
                    "updated_at": None,
                },
            },
        )
        profile = await svc.get_profile_by_global_id("ext_1")
        assert profile is not None
        assert profile.nationality == "DE"

    @pytest.mark.asyncio
    async def test_get_profile_by_global_id_not_found(self, pp_mod):
        svc = pp_mod.PlayerProfileService()
        svc = self._mock_db(
            svc,
            {
                "FROM player_profiles WHERE global_id = ?": None,
            },
        )
        profile = await svc.get_profile_by_global_id("nonexistent")
        assert profile is None

    @pytest.mark.asyncio
    async def test_get_all_profiles(self, pp_mod):
        svc = pp_mod.PlayerProfileService()
        svc = self._mock_db(
            svc,
            {
                "FROM player_profiles WHERE is_active = 1": [
                    {
                        "id": 1,
                        "global_id": "g1",
                        "display_name": "P1",
                        "jersey_number": 10,
                        "preferred_position": "FW",
                        "height_cm": 180,
                        "weight_kg": 75,
                        "dominant_foot": "right",
                        "date_of_birth": "2000-01-01",
                        "nationality": "BR",
                        "team": "home",
                        "is_active": 1,
                        "photo_path": None,
                        "created_at": "2024-01-01",
                        "updated_at": None,
                    },
                    {
                        "id": 2,
                        "global_id": "g2",
                        "display_name": "P2",
                        "jersey_number": 4,
                        "preferred_position": "DF",
                        "height_cm": 185,
                        "weight_kg": 80,
                        "dominant_foot": "left",
                        "date_of_birth": "1998-02-02",
                        "nationality": "AR",
                        "team": "home",
                        "is_active": 1,
                        "photo_path": None,
                        "created_at": "2024-01-01",
                        "updated_at": None,
                    },
                ],
            },
        )
        profiles = await svc.get_all_profiles()
        assert len(profiles) == 2

    @pytest.mark.asyncio
    async def test_get_all_profiles_by_team(self, pp_mod):
        svc = pp_mod.PlayerProfileService()
        svc = self._mock_db(
            svc,
            {
                "FROM player_profiles WHERE team = ? AND is_active = 1": [
                    {
                        "id": 3,
                        "global_id": "g3",
                        "display_name": "P3",
                        "jersey_number": 7,
                        "preferred_position": "MF",
                        "height_cm": 175,
                        "weight_kg": 70,
                        "dominant_foot": "right",
                        "date_of_birth": "2001-03-03",
                        "nationality": "PT",
                        "team": "away",
                        "is_active": 1,
                        "photo_path": None,
                        "created_at": "2024-01-01",
                        "updated_at": None,
                    },
                ],
            },
        )
        profiles = await svc.get_all_profiles(team="away")
        assert len(profiles) == 1

    @pytest.mark.asyncio
    async def test_update_profile(self, pp_mod):
        svc = pp_mod.PlayerProfileService()
        svc = self._mock_db(
            svc,
            {
                "FROM player_profiles WHERE id = ?": {
                    "id": 1,
                    "global_id": "g1",
                    "display_name": "Updated",
                    "jersey_number": 11,
                    "preferred_position": "LW",
                    "height_cm": 180,
                    "weight_kg": 75,
                    "dominant_foot": "right",
                    "date_of_birth": "2000-01-01",
                    "nationality": "BR",
                    "team": "home",
                    "is_active": 1,
                    "photo_path": None,
                    "created_at": "2024-01-01",
                    "updated_at": "2024-06-01",
                },
            },
        )
        profile = await svc.update_profile(1, display_name="Updated", jersey_number=11)
        assert profile is not None
        assert profile.display_name == "Updated"

    @pytest.mark.asyncio
    async def test_update_profile_invalid_field(self, pp_mod):
        svc = pp_mod.PlayerProfileService()
        svc = self._mock_db(
            svc,
            {
                "FROM player_profiles WHERE id = ?": {
                    "id": 1,
                    "global_id": "g1",
                    "display_name": "Same",
                    "jersey_number": 10,
                    "preferred_position": "FW",
                    "height_cm": 180,
                    "weight_kg": 75,
                    "dominant_foot": "right",
                    "date_of_birth": "2000-01-01",
                    "nationality": "BR",
                    "team": "home",
                    "is_active": 1,
                    "photo_path": None,
                    "created_at": "2024-01-01",
                    "updated_at": None,
                },
            },
        )
        profile = await svc.update_profile(1, invalid_field="ignored")
        assert profile is not None

    @pytest.mark.asyncio
    async def test_link_match_player(self, pp_mod):
        svc = pp_mod.PlayerProfileService()
        conn = MagicMock()
        cursor = MagicMock()
        cursor.execute.return_value = cursor
        conn.cursor.return_value = cursor
        conn.commit = MagicMock()
        svc._conn = conn
        result = await svc.link_match_player(1, 101, track_id=5, confidence=0.8)
        assert result is True

    @pytest.mark.asyncio
    async def test_link_match_player_failure(self, pp_mod):
        svc = pp_mod.PlayerProfileService()
        conn = MagicMock()
        cursor = MagicMock()
        cursor.execute.side_effect = Exception("DB error")
        conn.cursor.return_value = cursor
        svc._conn = conn
        result = await svc.link_match_player(1, 101)
        assert result is False

    @pytest.mark.asyncio
    async def test_get_profile_appearances(self, pp_mod):
        svc = pp_mod.PlayerProfileService()
        svc = self._mock_db(
            svc,
            {
                "FROM player_match_links l": [
                    {
                        "match_id": 1,
                        "match_name": "M1",
                        "match_date": "2024-01-01",
                        "opponent": "Opp",
                        "team": "home",
                        "jersey_number": 10,
                        "position": "FW",
                        "distance_covered_m": 5000.0,
                        "max_speed_kmh": 30.0,
                        "avg_speed_kmh": 7.0,
                        "passes_attempted": 40,
                        "passes_completed": 35,
                        "shots": 3,
                        "tackles": 2,
                    },
                ],
            },
        )
        apps = await svc.get_profile_appearances(1)
        assert len(apps) == 1
        assert apps[0].match_name == "M1"
        assert apps[0].distance_covered_m == 5000.0

    @pytest.mark.asyncio
    async def test_get_profile_appearances_empty(self, pp_mod):
        svc = pp_mod.PlayerProfileService()
        svc = self._mock_db(
            svc,
            {
                "FROM player_match_links l": [],
            },
        )
        apps = await svc.get_profile_appearances(999)
        assert apps == []

    @pytest.mark.asyncio
    async def test_compute_career_stats(self, pp_mod):
        svc = pp_mod.PlayerProfileService()
        svc = self._mock_db(
            svc,
            {
                "FROM player_match_links l": {
                    "matches_played": 3,
                    "total_distance": 15000.0,
                    "avg_max_speed": 30.0,
                    "avg_speed": 7.5,
                    "total_shots": 10,
                    "total_passes_attempted": 150,
                    "total_passes_completed": 130,
                    "total_tackles": 8,
                },
            },
        )
        stats = await svc.compute_career_stats(1)
        assert stats["matches_played"] == 3
        assert stats["total_distance_km"] == 15.0
        assert stats["pass_accuracy"] > 0.8

    @pytest.mark.asyncio
    async def test_compute_career_stats_no_data(self, pp_mod):
        svc = pp_mod.PlayerProfileService()
        svc = self._mock_db(
            svc,
            {
                "FROM player_match_links l": None,
            },
        )
        stats = await svc.compute_career_stats(999)
        assert stats == {}

    @pytest.mark.asyncio
    async def test_compute_career_stats_zero_passes(self, pp_mod):
        svc = pp_mod.PlayerProfileService()
        svc = self._mock_db(
            svc,
            {
                "FROM player_match_links l": {
                    "matches_played": 1,
                    "total_distance": 1000.0,
                    "avg_max_speed": 25.0,
                    "avg_speed": 5.0,
                    "total_shots": 0,
                    "total_passes_attempted": 0,
                    "total_passes_completed": 0,
                    "total_tackles": 0,
                },
            },
        )
        stats = await svc.compute_career_stats(1)
        assert stats["pass_accuracy"] == 0.0

    @pytest.mark.asyncio
    async def test_auto_link_by_jersey(self, pp_mod):
        svc = pp_mod.PlayerProfileService()
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor

        call_count = [0]

        def _execute(sql, params=None):
            call_count[0] += 1
            if call_count[0] == 1:
                cursor.fetchall.return_value = [
                    _make_row({"track_id": 5, "jersey_number": 10, "team": "home"}),
                    _make_row({"track_id": 6, "jersey_number": None, "team": "home"}),
                ]
            elif call_count[0] == 2:
                cursor.fetchone.return_value = _make_row(
                    {"id": 1, "display_name": "P1", "jersey_number": 10}
                )
            return cursor

        cursor.execute = _execute
        svc._conn = conn
        proposals = await svc.auto_link_by_jersey(1)
        assert len(proposals) == 1
        assert proposals[0]["profile_id"] == 1
        assert proposals[0]["confidence"] == 0.7
        assert proposals[0]["method"] == "jersey_number"

    @pytest.mark.asyncio
    async def test_auto_link_by_jersey_no_match(self, pp_mod):
        svc = pp_mod.PlayerProfileService()
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor

        def _execute(sql, params=None):
            cursor.fetchall.return_value = [
                _make_row({"track_id": 7, "jersey_number": 99, "team": "home"}),
            ]
            cursor.fetchone.return_value = None
            return cursor

        cursor.execute = _execute
        svc._conn = conn
        proposals = await svc.auto_link_by_jersey(1)
        assert len(proposals) == 0

    @pytest.mark.asyncio
    async def test_auto_link_by_jersey_team_filter(self, pp_mod):
        svc = pp_mod.PlayerProfileService()
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor

        def _execute(sql, params=None):
            if "AND team = ?" in sql:
                cursor.fetchall.return_value = [
                    _make_row({"track_id": 8, "jersey_number": 7, "team": "away"}),
                ]
                cursor.fetchone.return_value = _make_row(
                    {"id": 2, "display_name": "P2", "jersey_number": 7}
                )
            else:
                cursor.fetchall.return_value = []
            return cursor

        cursor.execute = _execute
        svc._conn = conn
        proposals = await svc.auto_link_by_jersey(1, team="away")
        assert len(proposals) == 1

    @pytest.mark.asyncio
    async def test_get_team_roster(self, pp_mod):
        svc = pp_mod.PlayerProfileService()
        svc = self._mock_db(
            svc,
            {
                "FROM player_profiles WHERE team = ? AND is_active = 1": [
                    {
                        "id": 1,
                        "global_id": "g1",
                        "display_name": "R1",
                        "jersey_number": 1,
                        "preferred_position": "GK",
                        "height_cm": 190,
                        "weight_kg": 85,
                        "dominant_foot": "right",
                        "date_of_birth": "1995-01-01",
                        "nationality": "BR",
                        "team": "home",
                        "is_active": 1,
                        "photo_path": None,
                        "created_at": "2024-01-01",
                        "updated_at": None,
                    },
                ],
            },
        )
        roster = await svc.get_team_roster("home")
        assert len(roster) == 1
        assert roster[0].preferred_position == "GK"

    @pytest.mark.asyncio
    async def test_close_clears_connection(self, pp_mod):
        svc = pp_mod.PlayerProfileService()
        conn = MagicMock()
        svc._conn = conn
        await svc.close()
        conn.close.assert_called_once()
        assert svc._conn is None
