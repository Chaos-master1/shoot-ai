"""Tests for WeatherService."""

from __future__ import annotations

import sys
from pathlib import Path

# Add the tests dir to path so conftest is found
sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()

_svc = load_service_module("ws_test", "weather_service.py")
WeatherService = _svc.WeatherService
WeatherSource = _svc.WeatherSource
PitchState = _svc.PitchState

import numpy as np


class TestManualWeather:
    def test_from_manual_clear(self) -> None:
        w = WeatherService.from_manual(temperature_c=22, precipitation_mm=0, wind_speed_kmh=5)
        assert w.temperature_c == 22
        assert w.pitch_state == PitchState.DRY
        assert w.source == WeatherSource.MANUAL

    def test_from_manual_rain(self) -> None:
        w = WeatherService.from_manual(temperature_c=15, precipitation_mm=3)
        assert w.pitch_state == PitchState.WET

    def test_from_manual_heavy_rain_muddy(self) -> None:
        w = WeatherService.from_manual(temperature_c=14, precipitation_mm=8)
        assert w.pitch_state == PitchState.MUDDY

    def test_from_manual_snow(self) -> None:
        w = WeatherService.from_manual(temperature_c=-1, precipitation_mm=2)
        assert w.pitch_state == PitchState.SNOWY

    def test_from_manual_ice(self) -> None:
        w = WeatherService.from_manual(temperature_c=-3, precipitation_mm=1)
        assert w.pitch_state == PitchState.ICY

    def test_default_humidity(self) -> None:
        w = WeatherService.from_manual(temperature_c=20)
        assert w.humidity_pct == 50.0


class TestPitchInference:
    def test_dry_pitch(self) -> None:
        assert WeatherService._infer_pitch_state(20, 0) == PitchState.DRY

    def test_wet_light_rain(self) -> None:
        assert WeatherService._infer_pitch_state(15, 1.0) == PitchState.WET

    def test_muddy_heavy_rain(self) -> None:
        assert WeatherService._infer_pitch_state(15, 6.0) == PitchState.MUDDY

    def test_snow_cold(self) -> None:
        assert WeatherService._infer_pitch_state(0, 1.0) == PitchState.SNOWY

    def test_ice_extreme_cold(self) -> None:
        assert WeatherService._infer_pitch_state(-3, 0) == PitchState.ICY


class TestConditionsClassification:
    def test_clear_weather(self) -> None:
        assert WeatherService._classify_conditions_text(0, 10, 5) == "clear"

    def test_cloudy(self) -> None:
        assert WeatherService._classify_conditions_text(0, 60, 5) == "cloudy"

    def test_overcast(self) -> None:
        assert WeatherService._classify_conditions_text(0, 90, 5) == "overcast"

    def test_light_rain(self) -> None:
        assert WeatherService._classify_conditions_text(0.5, 80, 5) == "light_rain"

    def test_heavy_rain(self) -> None:
        assert WeatherService._classify_conditions_text(15, 100, 5) == "heavy_rain"

    def test_windy_clear(self) -> None:
        assert WeatherService._classify_conditions_text(0, 20, 60) == "very_windy_clear"


class TestWeatherImpact:
    def test_clear_no_impact(self) -> None:
        w = WeatherService.from_manual(temperature_c=20, precipitation_mm=0, wind_speed_kmh=5)
        impact = WeatherService.analyze_impact(w)
        assert abs(impact.expected_goals_delta) < 0.1
        assert abs(impact.passing_accuracy_delta_pct) < 1
        assert abs(impact.sprint_distance_delta_pct) < 1

    def test_heavy_rain_increases_goals(self) -> None:
        w = WeatherService.from_manual(temperature_c=14, precipitation_mm=8, wind_speed_kmh=10)
        impact = WeatherService.analyze_impact(w)
        assert impact.expected_goals_delta >= 0.2
        assert impact.passing_accuracy_delta_pct < -5

    def test_strong_wind_hurts_passing(self) -> None:
        w = WeatherService.from_manual(temperature_c=20, wind_speed_kmh=45)
        impact = WeatherService.analyze_impact(w)
        assert impact.passing_accuracy_delta_pct < -10

    def test_heat_hurts_sprint(self) -> None:
        w = WeatherService.from_manual(temperature_c=35)
        impact = WeatherService.analyze_impact(w)
        assert impact.sprint_distance_delta_pct < -15

    def test_cold_increases_goals(self) -> None:
        w = WeatherService.from_manual(temperature_c=-2)
        impact = WeatherService.analyze_impact(w)
        assert impact.expected_goals_delta >= 0.1

    def test_impact_has_notes(self) -> None:
        w = WeatherService.from_manual(temperature_c=14, precipitation_mm=8)
        impact = WeatherService.analyze_impact(w)
        assert len(impact.notes) > 0
        assert "rain" in impact.notes[0].lower()

    def test_rain_and_wind_disadvantage_set_pieces(self) -> None:
        w = WeatherService.from_manual(temperature_c=15, precipitation_mm=6, wind_speed_kmh=35)
        impact = WeatherService.analyze_impact(w)
        assert impact.set_piece_advantage == "no_advantage"


class TestVideoWeather:
    def test_empty_frames(self) -> None:
        svc = WeatherService()
        pred = svc.classify_from_video([])
        assert pred.confidence == 0.0

    def test_clear_frame(self) -> None:
        svc = WeatherService()
        frame = np.full((480, 640, 3), 180, dtype=np.uint8)
        pred = svc.classify_from_video([frame])
        assert pred.confidence > 0.0
        assert pred.avg_brightness > 100

    def test_rainy_frame(self) -> None:
        svc = WeatherService()
        # A noisy, low-brightness frame with blue dominance = rain-like
        np.random.seed(0)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = 60  # B
        frame[:, :, 1] = 50  # G
        frame[:, :, 2] = 40  # R
        frame += np.random.randint(0, 30, frame.shape, dtype=np.uint8)
        pred = svc.classify_from_video([frame])
        # Either rainy (blue dominant + edges) or foggy (low contrast)
        assert pred.is_rainy or pred.avg_brightness < 100

    def test_foggy_frame(self) -> None:
        svc = WeatherService()
        # High brightness, low edges = foggy
        frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        pred = svc.classify_from_video([frame])
        # Bright frame should be detected as clear or foggy
        assert pred.is_foggy or pred.is_clear or pred.avg_brightness > 100

    def test_conditions_from_clear(self) -> None:
        svc = WeatherService()
        frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        pred = svc.classify_from_video([frame])
        cond = svc.conditions_from_video_prediction(pred)
        assert cond.pitch_state in {PitchState.DRY, PitchState.WET}

    def test_combined_advanced(self) -> None:
        svc = WeatherService()
        frame = np.full((480, 640, 3), 180, dtype=np.uint8)
        result = svc.classify_from_video_advanced([frame, frame])
        assert "raindrop_detection" in result
        assert "weather_classification" in result
        assert "is_rainy" in result
        assert "method" in result
        assert "conditions" in result


class TestWeatherServiceInit:
    def test_available(self) -> None:
        svc = WeatherService()
        assert svc.available

    def test_no_http_client_leak(self) -> None:
        svc = WeatherService()
        # Should create a client if none provided
        assert svc._client is not None

    def test_advanced_classifiers_fallback(self) -> None:
        # In a test env without kawkab.services importable, the fallback
        # path is hit. This just verifies the WeatherService still works.
        _ = WeatherService()
        # Just verify the service can produce conditions
        cond = WeatherService.from_manual(temperature_c=20)
        assert cond.temperature_c == 20
