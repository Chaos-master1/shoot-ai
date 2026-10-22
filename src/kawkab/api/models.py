"""Pydantic models for the REST API v1."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MatchOut(BaseModel):
    id: int
    name: str
    video_path: str
    home_team: str | None = None
    away_team: str | None = None
    duration: float = 0.0
    # fps/total_frames are only populated once a match has actually been
    # through the analysis pipeline (save_match() doesn't set them at
    # creation) -- a non-Optional type with a numeric default doesn't
    # help once the DB value is explicitly NULL: Pydantic v2 only applies
    # a field's default when the key is *absent*, not when it's present
    # as None, so GET /api/v1/matches/{id} raised a 500 on any match
    # that existed but hadn't been analyzed yet. None now means "not
    # analyzed yet", not a fabricated 0.
    fps: float | None = None
    total_frames: int | None = None
    created_at: str = ""


class MatchListOut(BaseModel):
    matches: list[MatchOut]
    total: int


class EventOut(BaseModel):
    id: int
    match_id: int
    event_type: str
    timestamp: float
    team: str = ""
    from_track_id: int = 0
    x: float = 0.0
    y: float = 0.0
    end_x: float = 0.0
    end_y: float = 0.0


class PlayerOut(BaseModel):
    track_id: int
    match_id: int
    name: str = ""
    team: str = ""
    jersey_number: int = 0


class ShotAnalysisOut(BaseModel):
    match_id: int
    shots: list[dict]
    total_shots: int
    total_goals: int
    total_xg: float


class TacticalShapesOut(BaseModel):
    formation_home: str = ""
    formation_away: str = ""
    shapes: dict = Field(default_factory=dict)
    support_angles: dict = Field(default_factory=dict)


class PressingOut(BaseModel):
    home_block: str = ""
    away_block: str = ""
    home_ppda: float = 0.0
    away_ppda: float = 0.0
    pressing_triggers: int = 0


class PlayerRatingOut(BaseModel):
    track_id: int
    name: str = ""
    rating: float = 0.0
    pass_accuracy: float = 0.0
    shot_impact: float = 0.0
    tackles: int = 0


class SquadSummaryOut(BaseModel):
    team: str
    players: list[PlayerRatingOut]


class MatchReportOut(BaseModel):
    match_id: int
    summary: str = ""
    key_moments: list[dict] = Field(default_factory=list)
    tactical_observations: list[str] = Field(default_factory=list)
    areas_for_improvement: list[str] = Field(default_factory=list)


class LlmQueryIn(BaseModel):
    match_id: int
    question: str


class LlmQueryOut(BaseModel):
    answer: str
    model_used: str = ""


class CalibrationOut(BaseModel):
    model_name: str
    total_xg: float
    actual_goals: int
    calibration_error: float
    brier_score: float
    log_loss: float
    shots_evaluated: int
    status: str = ""


class ModelComparisonOut(BaseModel):
    models: list[dict]
    best_model: str
    cv_summary: dict = Field(default_factory=dict)


class FitnessOut(BaseModel):
    player_name: str = ""
    total_distance: float = 0.0
    max_speed: float = 0.0
    sprints: int = 0
    high_intensity_distance: float = 0.0
    workload_score: float = 0.0


class RecruitmentSearchIn(BaseModel):
    position: str = ""
    min_age: int = 16
    max_age: int = 40
    league: str = ""
    stat_thresholds: dict = Field(default_factory=dict)
    limit: int = 20


class TransferFeeEstimateOut(BaseModel):
    player_name: str = ""
    estimated_fee: float = 0.0
    fee_range_low: float = 0.0
    fee_range_high: float = 0.0
    confidence: str = "medium"


class GamePlanOut(BaseModel):
    opponent: str
    formation_recommendation: str = ""
    key_players_to_neutralize: list[str] = Field(default_factory=list)
    set_piece_plan: str = ""
    scoreline_prediction: str = ""


class MonitoringDashboardOut(BaseModel):
    models: dict
    total_evaluations: int
    active_alerts: list[dict]


class WebhookCreateIn(BaseModel):
    url: str
    secret: str = ""
    events: list[str] = Field(default_factory=list)


class WebhookOut(BaseModel):
    id: int
    url: str
    events: list[str]
    is_active: bool = True
    created_at: str = ""


class ModelCardOut(BaseModel):
    name: str
    description: str
    version: str = ""
    model_type: str = ""
    input_features: list[str] = []
    output_description: str = ""
    training_data: str = ""
    known_limitations: list[str] = []
    tested_range: str = ""
    failure_modes: list[str] = []
    test_count: int = 0
    last_validated: str = ""
