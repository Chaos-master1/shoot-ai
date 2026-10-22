"""Application and tracking-pipeline configuration.

Two distinct concerns share this module because both are conventionally
called "config" and `kawkab.app` imports `get_settings` from here:

- ``Settings`` / ``get_settings()`` — app-level runtime settings (desktop
  app identity, GPU/model defaults, external API keys, LLM provider).
  Sourced from environment variables / `.env`, documented in
  `.env.example`. Used by `kawkab.app.MainWindow` to construct every
  backend service.
- ``TrackingConfig`` and friends — CV tracking-pipeline tuning parameters
  (detection thresholds, tracker settings, filter/stitch tuning). Used by
  the `kawkab track`/`batch` CLI pipeline, loaded from YAML/JSON.

Usage:
    settings = get_settings()  # app-level, env-driven

    cfg = TrackingConfig.load("configs/broadcast.yaml")  # or use defaults:
    cfg = TrackingConfig()
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from kawkab import __version__ as _app_version


class Settings(BaseSettings):
    """App-level runtime settings, overridable via environment variables or `.env`.

    Field names match their env var names uppercased (pydantic-settings
    default), except where noted with an explicit ``validation_alias`` —
    see `.env.example` for the full documented list.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App identity
    app_name: str = "Kawkab AI"
    app_version: str = _app_version

    # Core (KAWKAB_-prefixed in .env.example)
    lang: str = Field(default="en", validation_alias="KAWKAB_LANG")
    debug: bool = Field(default=False, validation_alias="KAWKAB_DEBUG")
    data_dir: str = Field(default="", validation_alias="KAWKAB_DATA_DIR")

    # CV / tracking defaults — mirror the CVService/PoseAnalysisService
    # constructor defaults so an unconfigured Settings() never overrides
    # a service's own considered default.
    model_size: str = "auto"
    pose_model_size: str = "n"
    confidence_threshold: float = 0.4
    iou_threshold: float = 0.5
    gpu_enabled: bool = True
    frame_skip: int = 6
    auto_detect_gpu_tier: bool = True

    # Video enhancement
    enable_upscaling: bool = False
    enable_interpolation: bool = False

    # External football data providers (read-only, optional, free tiers)
    football_data_api_key: str = Field(default="", validation_alias="FOOTBALL_DATA_API_KEY")
    apifootball_api_key: str = Field(default="", validation_alias="APIFOOTBALL_API_KEY")
    bzzoiro_api_key: str = Field(default="", validation_alias="BZZOIRO_API_KEY")
    thesportsdb_api_key: str = Field(default="", validation_alias="THESPORTSDB_API_KEY")

    # LLM provider (local Ollama by default — mirrors LLMConfig's defaults)
    llm_provider: str = Field(default="ollama", validation_alias="LLM_PROVIDER")
    llm_api_key: str = Field(default="", validation_alias="LLM_API_KEY")
    ollama_model: str = Field(default="ministral-3:14b", validation_alias="OLLAMA_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_URL")
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", validation_alias="OPENAI_MODEL")

    # Optional service flags
    enable_face_recognition: bool = Field(default=True, validation_alias="ENABLE_FACE_RECOGNITION")
    enable_weather_detection: bool = Field(
        default=True, validation_alias="ENABLE_WEATHER_DETECTION"
    )
    enable_realtime_analysis: bool = Field(
        default=False, validation_alias="ENABLE_REALTIME_ANALYSIS"
    )

    # Physics engine (set-piece simulation)
    fluidx3d_path: str = Field(default="", validation_alias="FLUIDX3D_PATH")
    enable_physics_sim: bool = Field(default=False, validation_alias="ENABLE_PHYSICS_SIM")

    # Security
    session_timeout_minutes: int = Field(default=0, validation_alias="SESSION_TIMEOUT_MINUTES")


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings singleton (cached after first call)."""
    return Settings()


@dataclass
class DetectionConfig:
    confidence_threshold: float = 0.4
    ball_confidence_threshold: float = 0.15
    iou_threshold: float = 0.5
    max_bbox_area_ratio: float = 0.15
    min_bbox_area_ratio: float = 0.002
    classes: list[int] = field(default_factory=lambda: [0, 32])
    ball_size_min_px: int = 4
    ball_size_max_px: int = 12
    ball_circularity_min: float = 0.6
    tile_overlap: float = 0.2
    enable_tiling: bool = False


@dataclass
class TrackingConfig:
    max_age: int = 30
    min_hits: int = 3
    w_association_emb: float = 0.75
    iou_match_thresh: float = 0.8
    reid_embedding_cap: int = 8
    reid_sample_rate: int = 30
    enable_camera_motion_compensation: bool = False


@dataclass
class FilterConfig:
    expected_player_count: int = 22
    max_keep_top_n: int = 28
    min_track_lifetime_frames: int = 30
    broadcast_frag_ratio_threshold: float = 0.2
    broadcast_stage1_divisor: int = 3000
    broadcast_stage1_pct: float = 0.02
    singlecam_stage1_pct: float = 1.0
    broadcast_stage3_min_segments: int = 2
    broadcast_stage3_min_pct: float = 0.15
    broadcast_stage3_top_k_buffer: int = 3


@dataclass
class StitchConfig:
    spatial_threshold_px: float = 50.0
    temporal_gap_max: float = 2.0
    gap_multiplier: float = 1.5
    overlap_ratio: float = 0.3
    color_distance_threshold: float = 70.0
    face_distance_threshold: float = 0.6
    reid_similarity_threshold: float = 0.6
    reid_few_emb_threshold: float = 0.7
    reid_many_emb_threshold: float = 0.65
    color_few_samples_threshold: float = 70.0
    color_many_samples_threshold: float = 55.0


@dataclass
class CameraCutConfig:
    hue_bins: int = 32
    sat_bins: int = 8
    threshold: float = 0.35
    min_cut_interval: float = 0.5
    sample_every_n: int = 6
    segment_min_frames: int = 6


@dataclass
class PitchDetectionConfig:
    min_line_length: int = 80
    max_line_gap: int = 12
    canny_low: int = 50
    canny_high: int = 150
    hough_threshold: int = 80
    min_confidence: float = 0.15


@dataclass
class ColorConfig:
    pitch_green_lower: list[int] = field(default_factory=lambda: [25, 40, 40])
    pitch_green_upper: list[int] = field(default_factory=lambda: [90, 255, 255])
    n_clusters: int = 3
    white_threshold: int = 230
    black_threshold: int = 30
    min_color_samples: int = 3
    color_sample_rate_hz: float = 2.0
    jpeg_ocr_sample_rate: int = 30


@dataclass
class EventDetectionConfig:
    goal_line_x_ratio: float = 0.05
    min_pass_duration: float = 0.3
    max_pass_duration: float = 6.0
    min_pass_px: float = 50.0
    min_pass_straightness: float = 0.5
    min_shot_px: float = 60.0
    max_shot_duration: float = 1.5
    min_shot_straightness: float = 0.3
    ball_conf_min: float = 0.3
    segment_gap_time: float = 0.5
    segment_max_jump_px: float = 300.0
    shot_dedup_window: float = 2.0
    pass_dedup_window: float = 1.0


@dataclass
class PerformanceConfig:
    frame_skip: int = 6
    checkpoint_interval: int = 500
    enable_checkpoint: bool = False
    gpu_enabled: bool = True
    half_precision: bool = True
    enable_streaming: bool = False
    batch_reid: bool = True


@dataclass
class TrackingConfigRoot:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    stitch: StitchConfig = field(default_factory=StitchConfig)
    camera_cut: CameraCutConfig = field(default_factory=CameraCutConfig)
    pitch: PitchDetectionConfig = field(default_factory=PitchDetectionConfig)
    color: ColorConfig = field(default_factory=ColorConfig)
    event: EventDetectionConfig = field(default_factory=EventDetectionConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    @classmethod
    def load(cls, path: str | Path) -> TrackingConfigRoot:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        with open(p) as f:
            data = json.load(f) if p.suffix == ".json" else _load_yaml(p)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> TrackingConfigRoot:
        root = cls()
        sections: list[tuple[str, type[Any]]] = [
            ("detection", DetectionConfig),
            ("tracking", TrackingConfig),
            ("filter", FilterConfig),
            ("stitch", StitchConfig),
            ("camera_cut", CameraCutConfig),
            ("pitch", PitchDetectionConfig),
            ("color", ColorConfig),
            ("event", EventDetectionConfig),
            ("performance", PerformanceConfig),
        ]
        for section_name, section_cls in sections:
            if section_name in data:
                section_data = data[section_name]
                current = getattr(root, section_name)
                fields = section_cls.__dataclass_fields__
                for field_name in fields:
                    if field_name in section_data:
                        setattr(current, field_name, section_data[field_name])
        return root


def _load_yaml(path: Path) -> dict:
    """Minimal YAML loader — just key:value lines, no nesting."""
    import re

    result: dict[str, Any] = {}
    current_section: str | None = None
    section_data: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        section_match = re.match(r"^(\w+):\s*$", line)
        if section_match:
            if current_section and section_data:
                result[current_section] = dict(section_data)
                section_data = {}
            current_section = section_match.group(1)
            continue
        kv_match = re.match(r"^(\w+):\s*(.+)$", line)
        if kv_match and current_section:
            k, v = kv_match.group(1), kv_match.group(2).strip()
            section_data[k] = _parse_value(v)
    if current_section and section_data:
        result[current_section] = section_data
    return result


def _parse_value(v: str) -> Any:
    if v.lower() in ("true", "yes"):
        return True
    if v.lower() in ("false", "no"):
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    if v.startswith("[") and v.endswith("]"):
        return [_parse_value(x.strip()) for x in v[1:-1].split(",")]
    return v
