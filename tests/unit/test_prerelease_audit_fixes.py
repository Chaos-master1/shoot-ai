"""Regression tests for the pre-release audit fixes (2026-09-16).

Covers four real defects found in the full-codebase audit:
1. EnhancedXgModel.compute_single cached results across instances.
2. AnalysisServiceCore.analyze_match discarded computed team stats + away PPDA.
3. OAuth HTTP calls had no timeout (indefinite login hangs).
4. decrypt_dict failed silently on medical fields.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.core.xg_model import EnhancedXgFeatures, EnhancedXgModel
from kawkab.services.analysis_service import AnalysisService
from kawkab.services.cv_service import Detection, FrameDetections, MatchTrackData


def _features(distance: float = 11.0, angle: float = 0.0) -> EnhancedXgFeatures:
    return EnhancedXgFeatures(distance_m=distance, angle_deg=angle)


class TestXgModelCacheIsolation:
    """compute_single must be memoized per instance, not across instances.

    A class-level lru_cache keyed on the features dataclass alone leaked
    results between models with different coefficients -- the model
    comparison path constructs several instances and would have served
    one model's xG for another.
    """

    @staticmethod
    def _model(intercept: float) -> EnhancedXgModel:
        zero = dict.fromkeys(
            (
                "distance_m",
                "distance_m_sq",
                "angle_sin",
                "angle_deg_sq_sin",
                "is_header",
                "is_through_ball_assist",
                "is_cross_assist",
                "is_one_on_one",
                "is_pressed",
                "is_volley",
                "is_free_kick",
                "gk_distance_m",
                "gk_distance_m_sq",
                "is_rebound",
                "is_big_chance",
            ),
            0.0,
        )
        zero["intercept"] = intercept
        return EnhancedXgModel(coefficients=zero)

    def test_same_features_different_models_differ(self):
        m1 = self._model(-2.0)
        m2 = self._model(2.0)
        f = _features()
        v1 = m1.compute_single(f)
        v2 = m2.compute_single(f)
        assert v2 > v1, "models with different intercepts must not share cached results"
        # And repeated calls stay consistent per instance.
        assert m1.compute_single(f) == v1
        assert m2.compute_single(f) == v2

    def test_repeat_calls_are_memoized_not_recomputed(self):
        m = self._model(-1.0)
        f = _features()
        first = m.compute_single(f)
        with patch.object(
            m,
            "_compute_single_uncached",
            side_effect=AssertionError("must be cached"),
        ):
            assert m.compute_single(f) == first


class TestAnalysisTeamStatsWired:
    """analyze_match must return the team stats it computes and keep both PPDA values."""

    @staticmethod
    def _track_data() -> MatchTrackData:
        frames = []
        for i in range(10):
            ball = Detection(
                bbox=(100, 300, 105, 305),
                confidence=0.9,
                class_id=32,
                class_name="sports ball",
                track_id=99,
            )
            # Players move each frame so _compute_player_stats records
            # real displacement -- the fixture must exercise the physical
            # stats that _compute_team_stats aggregates.
            home_p = Detection(
                bbox=(110 + i * 8, 300, 130 + i * 8, 340),
                confidence=0.9,
                class_id=0,
                class_name="person",
                track_id=1,
            )
            away_p = Detection(
                bbox=(400 + i * 6, 300, 420 + i * 6, 340),
                confidence=0.9,
                class_id=0,
                class_name="person",
                track_id=2,
            )
            frames.append(
                FrameDetections(
                    frame_number=i,
                    timestamp=i * 0.1,
                    detections=[ball, home_p, away_p],
                    image_width=1280,
                    image_height=720,
                )
            )
        return MatchTrackData(
            match_id=1,
            fps=10.0,
            total_frames=10,
            duration_seconds=1.0,
            frames=frames,
            track_registry={1: {}, 2: {}, 99: {}},
            player_teams={1: "home", 2: "away"},
        )

    @pytest.mark.asyncio
    async def test_team_stats_carry_computed_values(self):
        analysis = await AnalysisService().analyze_match(self._track_data(), match_id=1)
        # distance_covered_km is only ever set by _compute_team_stats;
        # before the fix the returned analysis used fresh zeroed objects.
        assert analysis.home_team.distance_covered_km > 0.0
        assert analysis.away_team.distance_covered_km > 0.0
        # passes_attempted must be consistent with detected events
        n_passes = sum(1 for e in analysis.events if e.get("type") == "pass")
        if n_passes:
            assert (
                analysis.home_team.passes_attempted + analysis.away_team.passes_attempted
                == n_passes
            )

    @pytest.mark.asyncio
    async def test_ppda_breakdown_populated_for_both_teams(self):
        analysis = await AnalysisService().analyze_match(self._track_data(), match_id=1)
        assert set(analysis.ppda_breakdown) == {"home", "away"}
        for side in ("home", "away"):
            breakdown = analysis.ppda_breakdown[side]
            assert isinstance(breakdown, dict)
            assert "ppda" in breakdown and "intensity" in breakdown


def _oauth_provider():
    from kawkab.cloud.oauth import OAuthProvider, OAuthProviderConfig

    cfg = OAuthProviderConfig(
        client_id="id",
        client_secret="secret",
        authorize_url="https://auth.example.com/auth",
        token_url="https://auth.example.com/token",
        userinfo_url="https://auth.example.com/userinfo",
        scopes=["openid"],
    )
    return OAuthProvider(cfg)


class TestOAuthTimeouts:
    """Every OAuth HTTP call must pass an explicit timeout."""

    def test_exchange_code_passes_timeout(self):
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"access_token": "t"}
        with patch("httpx.post", return_value=resp) as m:
            _oauth_provider().exchange_code("code", "http://localhost/cb")
        assert m.call_args.kwargs.get("timeout") is not None

    def test_get_userinfo_passes_timeout(self):
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"sub": "u1"}
        with patch("httpx.get", return_value=resp) as m:
            _oauth_provider().get_userinfo("tok")
        assert m.call_args.kwargs.get("timeout") is not None

    def test_refresh_token_passes_timeout(self):
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"access_token": "t"}
        with patch("httpx.post", return_value=resp) as m:
            _oauth_provider().refresh_token("rt")
        assert m.call_args.kwargs.get("timeout") is not None


class TestDecryptDictLogsFailures:
    """A failed decrypt on a medical field must not be silent.

    Patched on encryption.logger directly: the test-suite stubs replace
    loguru and kawkab.core.logging wholesale, so neither caplog nor a
    loguru sink can observe the records.
    """

    def test_bad_ciphertext_field_is_left_and_logged(self):
        from kawkab.core import encryption

        init_fernet = encryption.init_fernet
        decrypt_dict = encryption.decrypt_dict
        init_fernet("ca1fca1f" * 16)
        with patch.object(encryption.logger, "warning") as warn_mock:
            result = decrypt_dict({"diagnosis": "not-a-valid-fernet-token"}, ["diagnosis"])
        assert result["diagnosis"] == "not-a-valid-fernet-token"
        warn_mock.assert_called_once()
        call_text = str(warn_mock.call_args)
        assert "decrypt failed" in call_text and "diagnosis" in call_text

    def test_roundtrip_still_clean(self):
        from kawkab.core import encryption

        init_fernet = encryption.init_fernet
        encrypt_dict = encryption.encrypt_dict
        decrypt_dict = encryption.decrypt_dict
        init_fernet("ca1fca1f" * 16)
        enc = encrypt_dict({"diagnosis": "concussion"}, ["diagnosis"])
        with patch.object(encryption.logger, "warning") as warn_mock:
            dec = decrypt_dict(enc, ["diagnosis"])
        assert dec["diagnosis"] == "concussion"
        warn_mock.assert_not_called()
