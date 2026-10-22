"""Service layer for Kawkab AI.

Services are async-capable, dependency-injectable components that handle
specific domains: CV, enhancement, analysis, reasoning, LLM, storage, etc.
"""

from kawkab.services.advanced_event_detection_service import AdvancedEventDetectionService
from kawkab.services.analysis_service import AnalysisService
from kawkab.services.anomaly_detection_service import AnomalyDetectionService
from kawkab.services.audio_service import AudioService
from kawkab.services.batch_service import BatchJob, BatchService, BatchStatus
from kawkab.services.benchmark_service import BenchmarkResult, BenchmarkService
from kawkab.services.clip_extraction_service import (
    ClipLibraryService,
    ClipPlaylist,
    VideoClip,
)
from kawkab.services.clip_service import ClipExtractionService
from kawkab.services.cv_service import CVService, FrameDetections, MatchTrackData
from kawkab.services.data_export_service import DataExportService
from kawkab.services.enhancement_service import EnhancementService
from kawkab.services.face_recognition_service import FaceRecognitionService
from kawkab.services.homography_service import HomographyMatrix, HomographyService
from kawkab.services.knowledge_service import KnowledgeService
from kawkab.services.lightglue_homography_service import LightGlueHomographyService
from kawkab.services.llm_service import LLMConfig, LLMService
from kawkab.services.multi_match_analysis_service import MultiMatchAnalysisService
from kawkab.services.physical_load_service import PhysicalLoadService
from kawkab.services.player_profile_service import PlayerProfile, PlayerProfileService
from kawkab.services.pressure_metrics_service import PressureMetricsService
from kawkab.services.quality_scoring_service import QualityScoringService
from kawkab.services.reasoning_service import DiagnosisReport, ReasoningService
from kawkab.services.storage_service import StorageService
from kawkab.services.training_plan_service import TrainingPlan, TrainingPlanGenerator
from kawkab.services.validation_service import (
    EventGroundTruth,
    ValidationReport,
    ValidationResult,
    ValidationService,
)
from kawkab.services.visualization_service import VisualizationService
from kawkab.services.vram_manager import ModelPriority, VRAMManager

try:
    from kawkab.services.norfair_tracker import NorfairTracker
except ImportError:
    NorfairTracker = None  # type: ignore
from kawkab.services.api_football_service import ApiFootballService
from kawkab.services.bzzoiro_service import BzzoiroService
from kawkab.services.card_detection_service import (
    AudioCardSignal,
    CardDetectionService,
    CardEvent,
    CardSource,
    CardType,
)
from kawkab.services.easy_soccer_service import EasySoccerService
from kawkab.services.feedback_service import FeedbackService
from kawkab.services.fluidx3d_service import CfdResult, FluidX3DService
from kawkab.services.football_data_service import FootballDataService
from kawkab.services.football_rules_service import (
    FootballRulesService,
    Law,
    OffsideCheck,
    RestartType,
    RuleReference,
)
from kawkab.services.goalkeeper_service import (
    GoalkeeperAction,
    GoalkeeperService,
    GoalkeeperStats,
)
from kawkab.services.mujoco_ball_service import (
    MuJoCoBallService,
    TrajectoryPoint,
    TrajectoryResult,
)
from kawkab.services.openfootball_service import OpenFootballDataService
from kawkab.services.periodization_service import (
    CongestionLevel,
    CyclePhase,
    PeriodizationReport,
    PeriodizationService,
    WeekSummary,
)
from kawkab.services.pitch_detector import (
    CalibrationGuess,
    PitchDetector,
)
from kawkab.services.player_development_service import (
    PlayerDevelopmentReport,
    PlayerDevelopmentService,
    PlayerMatchStat,
    PlayerTrend,
    TrendDirection,
)
from kawkab.services.pose_analysis_service import (
    ActivitySegment,
    FallEvent,
    PoseAnalysisService,
    PoseResult,
)
from kawkab.services.positioning_service import (
    PositioningReport,
    PositioningService,
    Run,
    RunType,
)
from kawkab.services.possession_service import (
    PlayerPossessionStats,
    PossessionChain,
    PossessionReport,
    PossessionService,
)
from kawkab.services.psychology_service import (
    MomentumPoint,
    PsychologyEvent,
    PsychologyEventType,
    PsychologyReport,
    PsychologyService,
    ScoreState,
    ScoreStateTransition,
)
from kawkab.services.raindrop_detection_service import (
    RaindropDetection,
    RaindropDetectionService,
)
from kawkab.services.realtime_service import (
    AlertKind,
    AlertRule,
    AlertSeverity,
    CallbackSubscriber,
    ConsoleSubscriber,
    LowConfidenceAlertRule,
    LowFpsAlertRule,
    RealtimeEvent,
    RealtimeService,
    RealtimeSubscriber,
    ShotAlertRule,
    StreamStats,
)
from kawkab.services.roboflow_sports_service import RoboflowSportsService
from kawkab.services.scouting_service import (
    OpponentProfile,
    ScoutingService,
)
from kawkab.services.setpiece_service import (
    SetPieceEvent,
    SetPieceReport,
    SetPieceService,
    SetPieceStats,
)
from kawkab.services.statsbomb_service import StatsBombService
from kawkab.services.substitution_service import (
    SubstitutionEvent,
    SubstitutionImpact,
    SubstitutionReport,
    SubstitutionService,
)
from kawkab.services.thesportsdb_service import TheSportsDBService
from kawkab.services.video_review_service import (
    Annotation,
    AnnotationKind,
    Clip,
    ClipTag,
    ReviewSession,
    VideoReviewService,
)
from kawkab.services.weather_image_classifier import (
    WEATHER_CLASSES,
    WeatherClassification,
    WeatherImageClassifier,
    compute_features,
)
from kawkab.services.weather_service import (
    PitchState,
    VideoWeatherPrediction,
    WeatherConditions,
    WeatherImpact,
    WeatherService,
    WeatherSource,
)
from kawkab.services.workload_service import (
    RiskLevel,
    WorkloadRecord,
    WorkloadReport,
    WorkloadService,
    WorkloadSource,
)

__all__ = [
    "CVService",
    "MatchTrackData",
    "FrameDetections",
    "EnhancementService",
    "AnalysisService",
    "LLMService",
    "LLMConfig",
    "KnowledgeService",
    "StorageService",
    "AudioService",
    "ReasoningService",
    "DiagnosisReport",
    "ClipExtractionService",
    "ClipLibraryService",
    "VideoClip",
    "ClipPlaylist",
    "TrainingPlanGenerator",
    "TrainingPlan",
    "HomographyService",
    "HomographyMatrix",
    "VRAMManager",
    "ModelPriority",
    "PlayerProfileService",
    "PlayerProfile",
    "MultiMatchAnalysisService",
    "DataExportService",
    "VisualizationService",
    "AnomalyDetectionService",
    "QualityScoringService",
    "AdvancedEventDetectionService",
    "PhysicalLoadService",
    "PressureMetricsService",
    "LightGlueHomographyService",
    "BenchmarkService",
    "BenchmarkResult",
    "RealtimeService",
    "RealtimeEvent",
    "RealtimeSubscriber",
    "StreamStats",
    "AlertKind",
    "AlertSeverity",
    "AlertRule",
    "ShotAlertRule",
    "LowFpsAlertRule",
    "LowConfidenceAlertRule",
    "CallbackSubscriber",
    "ConsoleSubscriber",
    "ValidationService",
    "ValidationReport",
    "ValidationResult",
    "EventGroundTruth",
    "BatchService",
    "BatchJob",
    "BatchStatus",
    "FaceRecognitionService",
    "NorfairTracker",
    "PoseAnalysisService",
    "PoseResult",
    "ActivitySegment",
    "FallEvent",
    "MuJoCoBallService",
    "TrajectoryResult",
    "TrajectoryPoint",
    "FluidX3DService",
    "CfdResult",
    "WeatherService",
    "WeatherConditions",
    "WeatherImpact",
    "WeatherSource",
    "PitchState",
    "VideoWeatherPrediction",
    "PsychologyService",
    "PsychologyReport",
    "PsychologyEvent",
    "ScoreStateTransition",
    "MomentumPoint",
    "ScoreState",
    "PsychologyEventType",
    "FootballRulesService",
    "Law",
    "RestartType",
    "RuleReference",
    "OffsideCheck",
    "CardDetectionService",
    "CardEvent",
    "CardType",
    "CardSource",
    "AudioCardSignal",
    "RaindropDetectionService",
    "RaindropDetection",
    "WeatherImageClassifier",
    "WeatherClassification",
    "WEATHER_CLASSES",
    "compute_features",
    "SetPieceService",
    "SetPieceEvent",
    "SetPieceStats",
    "SetPieceReport",
    "GoalkeeperService",
    "GoalkeeperAction",
    "GoalkeeperStats",
    "SubstitutionService",
    "SubstitutionEvent",
    "SubstitutionImpact",
    "SubstitutionReport",
    "PossessionService",
    "PossessionChain",
    "PlayerPossessionStats",
    "PossessionReport",
    "FootballDataService",
    "BzzoiroService",
    "EasySoccerService",
    "FeedbackService",
    "ApiFootballService",
    "TheSportsDBService",
    "StatsBombService",
    "OpenFootballDataService",
    "RoboflowSportsService",
    "RealtimeService",
    "RealtimeEvent",
    "RealtimeSubscriber",
    "StreamStats",
    "AlertKind",
    "AlertSeverity",
    "AlertRule",
    "ShotAlertRule",
    "LowFpsAlertRule",
    "LowConfidenceAlertRule",
    "CallbackSubscriber",
    "ConsoleSubscriber",
    "PositioningService",
    "PositioningReport",
    "Run",
    "RunType",
    "PlayerDevelopmentService",
    "PlayerDevelopmentReport",
    "PlayerMatchStat",
    "PlayerTrend",
    "TrendDirection",
    "WorkloadService",
    "WorkloadReport",
    "WorkloadRecord",
    "WorkloadSource",
    "RiskLevel",
    "ScoutingService",
    "OpponentProfile",
    "VideoReviewService",
    "ReviewSession",
    "Clip",
    "Annotation",
    "AnnotationKind",
    "ClipTag",
    "PitchDetector",
    "CalibrationGuess",
    "PeriodizationService",
    "PeriodizationReport",
    "WeekSummary",
    "CyclePhase",
    "CongestionLevel",
]
