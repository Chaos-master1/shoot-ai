"""Weather service for match conditions and impact analysis.

Three data sources:
1. Open-Meteo public API (free, no key) - historical and forecast
2. Manual entry from user
3. In-video weather detection (frame-by-frame CNN classifier)

Output:
- WeatherConditions per match
- WeatherImpact predictions based on research
- Stored in DB for multi-match analysis

Research-backed weather → performance effects:
- Rain: more goals (avg +0.3 goals), more aerial duels, less passing accuracy
- Heavy wind (>30 km/h): -15% long passing accuracy
- Heat (>30C): -20% sprint distance
- Cold (<0C): more goals, more physical play
- Snow: highly variable, but generally lower scores
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import httpx
import numpy as np

from kawkab.core.logging import get_logger

if TYPE_CHECKING:
    from kawkab.services.raindrop_detection_service import RaindropDetectionService
    from kawkab.services.weather_image_classifier import WeatherImageClassifier

logger = get_logger(__name__)

CACHE_TTL_LONG = 86400

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"


class PitchState(Enum):
    DRY = "dry"
    WET = "wet"
    MUDDY = "muddy"
    SNOWY = "snowy"
    ICY = "icy"
    UNKNOWN = "unknown"


class WeatherSource(Enum):
    MANUAL = "manual"
    OPEN_METEO_ARCHIVE = "open_meteo_archive"
    OPEN_METEO_FORECAST = "open_meteo_forecast"
    VIDEO_CLASSIFIER = "video_classifier"


@dataclass
class WeatherConditions:
    temperature_c: float
    feels_like_c: float
    precipitation_mm: float
    wind_speed_kmh: float
    wind_direction_deg: float
    humidity_pct: float
    cloud_cover_pct: float
    conditions: str
    pitch_state: PitchState
    is_daylight: bool
    source: WeatherSource
    recorded_at: float = field(default_factory=time.time)


@dataclass
class WeatherImpact:
    expected_goals_delta: float
    passing_accuracy_delta_pct: float
    sprint_distance_delta_pct: float
    set_piece_advantage: str
    notes: list[str]


@dataclass
class VideoWeatherPrediction:
    is_rainy: bool
    is_foggy: bool
    is_snowy: bool
    is_clear: bool
    is_dusk_dawn: bool
    confidence: float
    avg_brightness: float


class WeatherService:
    """Multi-source weather for Kawkab AI matches.

    Sources (in priority order):
    1. Manual entry (highest priority — user override)
    2. Open-Meteo public API (free, no key)
    3. In-video classifier (frame brightness/edge analysis, no model needed)
    """

    PITCH_TEMP_THRESHOLDS = {
        "snow": 0.0,
        "ice": -2.0,
    }

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self._client = http_client or httpx.AsyncClient(timeout=20.0)
        self._owns_client = http_client is None
        self._cache: dict[str, tuple[float, WeatherConditions]] = {}
        self._video_classifier_available = False
        self._try_init_video_classifier()
        self._raindrop_service: RaindropDetectionService | None = None
        self._weather_classifier: WeatherImageClassifier | None = None
        self._try_init_advanced_classifiers()

    def _try_init_video_classifier(self) -> None:
        try:
            self._video_classifier_available = True
            logger.info("Video weather classifier (torch) available")
        except Exception:
            logger.info("torch not available; video classifier will use OpenCV-only fallback")

    def _try_init_advanced_classifiers(self) -> None:
        try:
            from kawkab.services.raindrop_detection_service import RaindropDetectionService
            from kawkab.services.weather_image_classifier import WeatherImageClassifier

            self._raindrop_service = RaindropDetectionService()
            self._weather_classifier = WeatherImageClassifier()
            logger.info("Advanced weather classifiers loaded (raindrop + multi-class)")
        except Exception as e:
            logger.info(f"Advanced weather classifiers not available: {e}")

    @property
    def available(self) -> bool:
        return True

    @property
    def has_video_classifier(self) -> bool:
        return self._video_classifier_available

    async def close(self) -> None:
        if self._owns_client and self._client:
            await self._client.aclose()

    def _cache_key(self, lat: float, lon: float, date: str) -> str:
        return f"{lat:.4f},{lon:.4f},{date}"

    def _is_cache_valid(self, key: str) -> bool:
        if key not in self._cache:
            return False
        ts, _ = self._cache[key]
        return time.monotonic() < ts

    # ------------------------------------------------------------------
    # Manual entry
    # ------------------------------------------------------------------

    @staticmethod
    def from_manual(
        temperature_c: float,
        precipitation_mm: float = 0.0,
        wind_speed_kmh: float = 0.0,
        humidity_pct: float = 50.0,
        conditions: str = "clear",
    ) -> WeatherConditions:
        """Build WeatherConditions from manual user input."""
        pitch = WeatherService._infer_pitch_state(temperature_c, precipitation_mm)
        return WeatherConditions(
            temperature_c=temperature_c,
            feels_like_c=temperature_c,
            precipitation_mm=precipitation_mm,
            wind_speed_kmh=wind_speed_kmh,
            wind_direction_deg=0.0,
            humidity_pct=humidity_pct,
            cloud_cover_pct=0.0,
            conditions=conditions,
            pitch_state=pitch,
            is_daylight=True,
            source=WeatherSource.MANUAL,
        )

    @staticmethod
    def _infer_pitch_state(temperature_c: float, precipitation_mm: float) -> PitchState:
        if temperature_c <= -2.0:
            return PitchState.ICY
        if temperature_c <= 0.5 and precipitation_mm > 0.5:
            return PitchState.SNOWY
        if precipitation_mm > 5.0:
            return PitchState.MUDDY
        if precipitation_mm > 0.0:
            return PitchState.WET
        return PitchState.DRY

    # ------------------------------------------------------------------
    # Open-Meteo API
    # ------------------------------------------------------------------

    async def fetch_conditions(
        self,
        latitude: float,
        longitude: float,
        date: str,
        is_forecast: bool = False,
    ) -> WeatherConditions | None:
        """Fetch weather for a lat/lon/date from Open-Meteo.

        Args:
            latitude, longitude: pitch location
            date: ISO date string (YYYY-MM-DD)
            is_forecast: if True, use forecast API; else use historical archive
        """
        key = self._cache_key(latitude, longitude, date + ("_f" if is_forecast else ""))
        if self._is_cache_valid(key):
            return self._cache[key][1]

        base = OPEN_METEO_FORECAST if is_forecast else OPEN_METEO_ARCHIVE
        params: dict[str, float | str] = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": date,
            "end_date": date,
            "hourly": "temperature_2m,precipitation,wind_speed_10m,wind_direction_10m,relative_humidity_2m,cloud_cover,is_day",
            "timezone": "UTC",
        }
        try:
            r = await self._client.get(base, params=params)
            if r.status_code != 200:
                logger.warning(f"Open-Meteo {r.status_code}")
                return None
            data = r.json()
            if "hourly" not in data:
                return None
            hourly = data["hourly"]
            times = hourly.get("time", [])
            if not times:
                return None
            if is_forecast:
                target_idx = self._find_closest_time_index(times)
            else:
                target_idx = min(15, len(times) - 1)
            if target_idx is None or target_idx >= len(times):
                return None
            temp = hourly["temperature_2m"][target_idx] if hourly.get("temperature_2m") else 15.0
            precip = hourly.get("precipitation", [0.0] * len(times))[target_idx] or 0.0
            wind = hourly.get("wind_speed_10m", [0.0] * len(times))[target_idx] or 0.0
            wind_dir = hourly.get("wind_direction_10m", [0.0] * len(times))[target_idx] or 0.0
            humidity = hourly.get("relative_humidity_2m", [50.0] * len(times))[target_idx] or 50.0
            cloud = hourly.get("cloud_cover", [0.0] * len(times))[target_idx] or 0.0
            is_day = (
                hourly.get("is_day", [1] * len(times))[target_idx] if hourly.get("is_day") else 1
            )
            source = (
                WeatherSource.OPEN_METEO_FORECAST
                if is_forecast
                else WeatherSource.OPEN_METEO_ARCHIVE
            )
            conditions_str = self._classify_conditions_text(precip, cloud, wind)
            pitch = self._infer_pitch_state(temp, precip)
            result = WeatherConditions(
                temperature_c=float(temp),
                feels_like_c=float(temp),
                precipitation_mm=float(precip),
                wind_speed_kmh=float(wind),
                wind_direction_deg=float(wind_dir),
                humidity_pct=float(humidity),
                cloud_cover_pct=float(cloud),
                conditions=conditions_str,
                pitch_state=pitch,
                is_daylight=bool(is_day),
                source=source,
            )
            self._cache[key] = (time.monotonic() + CACHE_TTL_LONG, result)
            return result
        except Exception as e:
            logger.warning(f"Open-Meteo fetch failed: {e}")
            return None

    def _find_closest_time_index(self, times: list[str]) -> int | None:
        """For forecast, find hour closest to typical match time (3pm UTC)."""
        target_hour = 15
        for i, t in enumerate(times):
            if "T" in t and int(t.split("T")[1].split(":")[0]) == target_hour:
                return i
        return 0

    @staticmethod
    def _classify_conditions_text(precip_mm: float, cloud_pct: float, wind_kmh: float) -> str:
        if precip_mm > 10:
            return "heavy_rain"
        if precip_mm > 2:
            return "rain"
        if precip_mm > 0.1:
            return "light_rain"
        if cloud_pct > 80:
            return "overcast"
        if cloud_pct > 40:
            return "cloudy"
        if wind_kmh > 50:
            return "very_windy_clear"
        if wind_kmh > 30:
            return "windy"
        return "clear"

    # ------------------------------------------------------------------
    # In-video weather classifier (frame analysis)
    # ------------------------------------------------------------------

    def classify_from_video(self, frames: list[np.ndarray]) -> VideoWeatherPrediction:
        """Classify weather conditions from a sample of match frames.

        Uses simple CV heuristics (no model required) so it works without
        ultralytics/torch. For higher accuracy, a ResNet18 trained on
        weather classes could be plugged in (the framework is ready).
        """
        if not frames:
            return VideoWeatherPrediction(
                is_rainy=False,
                is_foggy=False,
                is_snowy=False,
                is_clear=False,
                is_dusk_dawn=False,
                confidence=0.0,
                avg_brightness=0.0,
            )
        brightness_vals = []
        edge_density_vals = []
        blue_dominance_vals = []
        for f in frames[:30]:
            if f is None or f.size == 0:
                continue
            try:
                gray = (
                    float(f[:, :, 0].mean()) * 0.299
                    + float(f[:, :, 1].mean()) * 0.587
                    + float(f[:, :, 2].mean()) * 0.114
                )
                brightness_vals.append(gray)
                edges = self._edge_density(f)
                edge_density_vals.append(edges)
                blue_dom = float(f[:, :, 0].mean()) - float(f[:, :, 2].mean())
                blue_dominance_vals.append(blue_dom)
            except Exception:
                continue
        if not brightness_vals:
            return VideoWeatherPrediction(
                is_rainy=False,
                is_foggy=False,
                is_snowy=False,
                is_clear=False,
                is_dusk_dawn=False,
                confidence=0.0,
                avg_brightness=0.0,
            )
        avg_brightness = float(np.mean(brightness_vals))
        avg_edge = float(np.mean(edge_density_vals)) if edge_density_vals else 0.0
        avg_blue_dom = float(np.mean(blue_dominance_vals))
        is_rainy = avg_edge > 0.18 and avg_blue_dom > 5
        is_foggy = avg_brightness > 180 and avg_edge < 0.08
        is_snowy = avg_brightness > 200 and avg_edge < 0.06
        is_dusk_dawn = 50 < avg_brightness < 130
        is_clear = avg_brightness > 130 and avg_edge < 0.15
        confidence = 0.5
        if is_clear:
            confidence = 0.7
        if is_rainy or is_snowy:
            confidence = 0.6
        if is_foggy:
            confidence = 0.65
        return VideoWeatherPrediction(
            is_rainy=is_rainy,
            is_foggy=is_foggy,
            is_snowy=is_snowy,
            is_clear=is_clear,
            is_dusk_dawn=is_dusk_dawn,
            confidence=confidence,
            avg_brightness=avg_brightness,
        )

    @staticmethod
    def _edge_density(frame: np.ndarray) -> float:
        try:
            import cv2

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            edges = cv2.Canny(gray, 80, 200)
            return float(edges.mean()) / 255.0
        except Exception:
            return 0.0

    def conditions_from_video_prediction(self, pred: VideoWeatherPrediction) -> WeatherConditions:
        if pred.is_rainy:
            return self.from_manual(temperature_c=15.0, precipitation_mm=3.0, conditions="rain")
        if pred.is_snowy:
            return self.from_manual(temperature_c=-1.0, precipitation_mm=1.0, conditions="snow")
        if pred.is_foggy:
            return self.from_manual(temperature_c=12.0, humidity_pct=95, conditions="fog")
        if pred.is_clear:
            return self.from_manual(temperature_c=20.0, conditions="clear")
        return self.from_manual(temperature_c=18.0, conditions="unknown")

    # ------------------------------------------------------------------
    # Advanced: raindrop + multi-class video classifier (tobybreckon-style)
    # ------------------------------------------------------------------

    @property
    def has_raindrop_detector(self) -> bool:
        return self._raindrop_service is not None

    @property
    def has_multi_class_classifier(self) -> bool:
        return self._weather_classifier is not None

    def classify_from_video_advanced(self, frames: list[np.ndarray]) -> dict[str, Any]:
        """Combined weather detection using raindrop + multi-class classifier.

        Returns dict with:
            - raindrop_detection: RaindropDetection
            - weather_classification: WeatherClassification
            - conditions: WeatherConditions (best estimate)
        """
        from dataclasses import asdict

        result: dict[str, Any] = {
            "raindrop_detection": None,
            "weather_classification": None,
            "conditions": None,
            "is_rainy": False,
            "method": "opencv_only",
        }
        raindrop_result = None
        if self._raindrop_service is not None:
            try:
                raindrop_result = self._raindrop_service.detect(frames)
                result["raindrop_detection"] = asdict(raindrop_result)
            except Exception as e:
                logger.warning(f"Raindrop detection failed: {e}")
        weather_cls = None
        if self._weather_classifier is not None:
            try:
                weather_cls = self._weather_classifier.classify_batch(frames)
                result["weather_classification"] = asdict(weather_cls)
            except Exception as e:
                logger.warning(f"Weather classification failed: {e}")
        is_rainy = (raindrop_result is not None and raindrop_result.is_rainy) or (
            weather_cls is not None and weather_cls.predicted_class == "rainy"
        )
        result["is_rainy"] = is_rainy
        conditions = None
        if weather_cls is not None:
            conditions = self.from_manual(
                temperature_c=15.0 if is_rainy else 20.0,
                precipitation_mm=3.0 if is_rainy else 0.0,
                conditions=weather_cls.predicted_class,
            )
            result["conditions"] = {
                "temperature_c": conditions.temperature_c,
                "precipitation_mm": conditions.precipitation_mm,
                "wind_speed_kmh": conditions.wind_speed_kmh,
                "humidity_pct": conditions.humidity_pct,
                "conditions": conditions.conditions,
                "pitch_state": conditions.pitch_state.value,
                "source": "video_classifier",
            }
        if raindrop_result is not None and weather_cls is not None:
            result["method"] = "raindrop+multiclass"
        elif raindrop_result is not None:
            result["method"] = "raindrop"
        elif weather_cls is not None:
            result["method"] = "multiclass"
        return result

    # ------------------------------------------------------------------
    # Impact analysis
    # ------------------------------------------------------------------

    @staticmethod
    def analyze_impact(conditions: WeatherConditions) -> WeatherImpact:
        notes: list[str] = []
        goals_delta = 0.0
        passing_delta = 0.0
        sprint_delta = 0.0
        set_piece_advantage = "neutral"
        if conditions.precipitation_mm > 5.0:
            goals_delta += 0.3
            passing_delta -= 12.0
            sprint_delta -= 5.0
            set_piece_advantage = "neutral"
            notes.append("Heavy rain: more goals expected, lower passing accuracy")
        elif conditions.precipitation_mm > 1.0:
            goals_delta += 0.15
            passing_delta -= 5.0
            notes.append("Light rain: slight goal boost, marginally lower passing")
        if conditions.wind_speed_kmh > 40:
            passing_delta -= 15.0
            notes.append("Strong wind: significantly reduces long passing accuracy")
            set_piece_advantage = "no_advantage"
        elif conditions.wind_speed_kmh > 25:
            passing_delta -= 7.0
            notes.append("Moderate wind: reduces long passing")
        if conditions.temperature_c > 32:
            sprint_delta -= 20.0
            notes.append("Extreme heat: significantly reduces sprint distance")
        elif conditions.temperature_c > 28:
            sprint_delta -= 10.0
            notes.append("Hot weather: reduced sprint distance")
        if conditions.temperature_c < 0:
            goals_delta += 0.2
            notes.append("Cold conditions: typically see more goals")
        if conditions.temperature_c < 5:
            sprint_delta -= 8.0
            notes.append("Cold weather: reduced sprint performance")
        if conditions.precipitation_mm > 0 and conditions.wind_speed_kmh > 30:
            set_piece_advantage = "no_advantage"
        return WeatherImpact(
            expected_goals_delta=round(goals_delta, 2),
            passing_accuracy_delta_pct=round(passing_delta, 1),
            sprint_distance_delta_pct=round(sprint_delta, 1),
            set_piece_advantage=set_piece_advantage,
            notes=notes or ["Conditions: no significant performance impact expected"],
        )
