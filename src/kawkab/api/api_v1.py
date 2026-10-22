"""REST API v1 — analytics endpoints for Kawkab AI.

Extends the cloud FastAPI server with match analysis, player rating,
tactical, fitness, recruitment, and monitoring endpoints."""

from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path
from typing import NoReturn

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from kawkab.api.models import (
    CalibrationOut,
    EventOut,
    FitnessOut,
    GamePlanOut,
    LlmQueryIn,
    LlmQueryOut,
    MatchOut,
    MatchReportOut,
    ModelCardOut,
    ModelComparisonOut,
    MonitoringDashboardOut,
    PlayerOut,
    PlayerRatingOut,
    PressingOut,
    RecruitmentSearchIn,
    ShotAnalysisOut,
    SquadSummaryOut,
    TacticalShapesOut,
    TransferFeeEstimateOut,
    WebhookCreateIn,
    WebhookOut,
)
from kawkab.core.rbac import require_permission

router = APIRouter(prefix="/api/v1", tags=["analytics"])


_storage_instance = None
_storage_ready = False


def _get_storage():
    global _storage_instance
    if _storage_instance is not None:
        return _storage_instance
    from kawkab.services.storage_service import StorageService

    _storage_instance = StorageService()
    return _storage_instance


async def _ready_storage():
    """Storage singleton, initialized on first use.

    Request handlers previously constructed StorageService without ever
    awaiting initialize(), so the SQLite connection never existed and
    every data endpoint answered 503 storage_unavailable. initialize()
    is idempotent (versioned migrations no-op when current), so a
    one-shot flag is enough; a failed init leaves the flag unset so the
    next request retries instead of caching the failure.

    A backend that is already wired (an open SQLite connection or a
    constructed Postgres adapter) is left alone — someone owns that
    lifecycle (tests hand-wire a scratch DB; PG owns its pool), and a
    redundant initialize() would tear down and replace it.
    """
    global _storage_ready
    svc = _get_storage()
    if not _storage_ready and svc._conn is None and svc._pg is None:
        await svc.initialize()
    _storage_ready = True
    return svc


def _get_monitor():
    from kawkab.services.model_monitor_service import ModelMonitoringService

    return ModelMonitoringService()


def _get_audit():
    """AuditService bound to the same storage backend as the API (Phase 3:
    wires the previously-orphaned audit_service.py into the API layer so
    elite-facing operations leave a compliance trail)."""
    from kawkab.services.audit_service import AuditService

    return AuditService(_get_storage())


def _not_found(msg: str) -> NoReturn:
    raise HTTPException(status_code=404, detail=msg)


def _get_user_team_ids(user_id: int) -> set[int]:
    """Team IDs *user_id* belongs to, per the cloud auth DB's team_members
    table. Used by _check_match_access() below -- matches and users/teams
    live in genuinely separate databases in SQLite mode, so this is a
    deliberate cross-database lookup, not something StorageService itself
    could resolve on its own."""
    from kawkab.cloud.database import get_cloud_db

    db = get_cloud_db()
    rows = db.execute("SELECT team_id FROM team_members WHERE user_id = ?", (user_id,)).fetchall()
    return {row["team_id"] for row in rows}


def _check_match_access(match: dict, user: dict) -> None:
    """Enforce match ownership: every /api/v1 route previously scoped
    match-related queries by match_id alone, never by who the requesting
    user is -- any authenticated user (registration is open, defaulting
    to the analyst role) could read every match, player, event, and
    report in the deployment. A match with owner_id=None (every match
    created via the existing desktop pipeline, which has no per-user
    concept at all, plus every match that existed before this check was
    added) is treated as visible to all -- this does not retroactively
    hide data that was previously unrestricted. A match WITH an owner is
    visible only to that owner, or to members of the team it's been
    explicitly shared with (is_shared + team_id, the same pattern
    projects/share already uses).

    Raises 404, not 403, for both "doesn't exist" and "exists but isn't
    yours" -- returning 403 would confirm a match id exists to someone
    who can't see it.
    """
    owner_id = match.get("owner_id")
    if owner_id is None or owner_id == user.get("id"):
        return
    if (
        match.get("is_shared")
        and match.get("team_id") is not None
        and match["team_id"] in _get_user_team_ids(user["id"])
    ):
        return
    _not_found(f"Match {match.get('id')} not found")


def _paginate(items: list, page: int, per_page: int) -> dict:
    total = len(items)
    pages = math.ceil(total / per_page) if per_page > 0 else 1
    start = (page - 1) * per_page
    end = start + per_page
    return {
        "items": items[start:end],
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": pages,
    }


# ── Matches ──


@router.get("/matches")
async def list_matches(
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=1, le=100),
    _user: dict = Depends(require_permission("match:read")),
):
    svc = await _ready_storage()
    matches_list = await svc.get_all_matches()
    team_ids = _get_user_team_ids(_user["id"])
    visible = [
        m
        for m in matches_list
        if m.get("owner_id") is None
        or m.get("owner_id") == _user["id"]
        or (m.get("is_shared") and m.get("team_id") in team_ids)
    ]
    matches = [MatchOut(**m) for m in visible]
    return _paginate(matches, page, per_page)


@router.get("/matches/{match_id}", response_model=MatchOut)
async def get_match(match_id: int, _user: dict = Depends(require_permission("match:read"))):
    svc = await _ready_storage()
    match = await svc.get_match(match_id)
    if not match:
        _not_found(f"Match {match_id} not found")
    _check_match_access(match, _user)
    return MatchOut(**match)


@router.get("/matches/{match_id}/events")
async def get_match_events(
    match_id: int,
    event_type: str | None = Query(None),
    limit: int = Query(1000, le=5000),
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=1, le=100),
    _user: dict = Depends(require_permission("event:read")),
):
    svc = await _ready_storage()
    match = await svc.get_match(match_id)
    if not match:
        _not_found(f"Match {match_id} not found")
    _check_match_access(match, _user)
    events = await svc.get_match_events(match_id)
    if event_type:
        events = [e for e in events if e.get("type") == event_type]
    items = [
        EventOut(
            id=e.get("id", 0),
            match_id=e.get("match_id", match_id),
            event_type=e.get("type", ""),
            timestamp=float(e.get("timestamp", 0)),
            team=e.get("team", ""),
            from_track_id=int(e.get("from_track_id", 0)),
            x=float(e.get("x", 0)),
            y=float(e.get("y", 0)),
            end_x=float(e.get("end_x", 0)),
            end_y=float(e.get("end_y", 0)),
        )
        for e in events[:limit]
    ]
    return _paginate(items, page, per_page)


@router.get("/matches/{match_id}/players")
async def get_match_players(
    match_id: int,
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=1, le=100),
    _user: dict = Depends(require_permission("player:read")),
):
    svc = await _ready_storage()
    match = await svc.get_match(match_id)
    if not match:
        _not_found(f"Match {match_id} not found")
    _check_match_access(match, _user)
    players = await svc.get_match_players(match_id)
    items = [
        PlayerOut(
            track_id=p.get("track_id", 0),
            match_id=match_id,
            name=p.get("name", f"Player {p.get('track_id', '')}"),
            team=p.get("team", ""),
            jersey_number=int(p.get("jersey_number", 0)),
        )
        for p in players
    ]
    return _paginate(items, page, per_page)


# ── Analysis ──


@router.get("/matches/{match_id}/analysis/shots", response_model=ShotAnalysisOut)
async def analyze_shots(match_id: int, _user: dict = Depends(require_permission("analysis:read"))):
    svc = await _ready_storage()
    match = await svc.get_match(match_id)
    if not match:
        _not_found(f"Match {match_id} not found")
    _check_match_access(match, _user)
    events = await svc.get_match_events(match_id)
    shots = [e for e in events if e.get("type") == "shot"]
    from kawkab.core.xg_model import compute_xg_trained_from_dict

    shot_data = []
    total_xg = 0.0
    total_goals = 0
    for s in shots:
        xg = compute_xg_trained_from_dict(s)
        is_goal = bool(s.get("is_goal", False))
        total_xg += xg
        if is_goal:
            total_goals += 1
        shot_data.append(
            {
                "timestamp": s.get("timestamp", 0),
                "x": s.get("x", 0),
                "y": s.get("y", 0),
                "xg": round(xg, 4),
                "is_goal": is_goal,
                "player": s.get("player_name", ""),
            }
        )

    return ShotAnalysisOut(
        match_id=match_id,
        shots=shot_data,
        total_shots=len(shots),
        total_goals=total_goals,
        total_xg=round(total_xg, 4),
    )


@router.get("/matches/{match_id}/analysis/tactical-shapes", response_model=TacticalShapesOut)
async def get_tactical_shapes(
    match_id: int, _user: dict = Depends(require_permission("analysis:read"))
):
    from kawkab.core.tactical_shape_analyzer import TacticalShapeAnalyzer

    svc = await _ready_storage()
    match = await svc.get_match(match_id)
    if not match:
        _not_found(f"Match {match_id} not found")
    _check_match_access(match, _user)
    events = await svc.get_match_events(match_id)
    analyzer = TacticalShapeAnalyzer()
    home = analyzer.analyze_shapes(events, team="home")
    away = analyzer.analyze_shapes(events, team="away")
    return TacticalShapesOut(
        formation_home=home.primary_attacking_shape,
        formation_away=away.primary_attacking_shape,
        shapes={"home": home.to_dict(), "away": away.to_dict()},
        support_angles={
            "home": home.avg_support_angle_coverage,
            "away": away.avg_support_angle_coverage,
        },
    )


@router.get("/matches/{match_id}/analysis/pressing", response_model=PressingOut)
async def get_pressing(match_id: int, _user: dict = Depends(require_permission("analysis:read"))):
    from kawkab.core.pressing_classifier import classify_pressing_system

    svc = await _ready_storage()
    match = await svc.get_match(match_id)
    if not match:
        _not_found(f"Match {match_id} not found")
    _check_match_access(match, _user)
    events = await svc.get_match_events(match_id)
    home = classify_pressing_system(events, team="home")
    away = classify_pressing_system(events, team="away")
    return PressingOut(
        home_block=home.primary_block_type,
        away_block=away.primary_block_type,
        home_ppda=home.ppda,
        away_ppda=away.ppda,
        pressing_triggers=home.trigger_count + away.trigger_count,
    )


@router.get("/matches/{match_id}/analysis/pro")
async def get_pro_analytics(
    match_id: int,
    _user: dict = Depends(require_permission("analysis:read")),
):
    """Aggregated elite-analytics report (OBV, EPV, pass flow, pressing
    clusters, duels, ball recovery, box entries, switches, crossing, set
    pieces, through balls, off-ball) with honest per-block data_available
    flags. See ui/bridge_handlers/bridge_pro_analytics.py."""
    import json as _json

    svc = await _ready_storage()
    match = await svc.get_match(match_id)
    if not match:
        _not_found(f"Match {match_id} not found")
    _check_match_access(match, _user)

    from kawkab.ui.bridge_handlers.bridge_pro_analytics import ProAnalyticsHandler

    handler = ProAnalyticsHandler(bridge=None, services={"storage_service": svc})
    payload = await handler.get_pro_analytics_report(match_id)
    try:
        return _json.loads(payload)
    except _json.JSONDecodeError:
        return {"raw": payload}


@router.get("/season/pro")
async def get_season_pro_analytics(_user: dict = Depends(require_permission("analysis:read"))):
    """Cross-match season analytics: formation trends, discipline/suspension
    risk, fixture difficulty. See ui/bridge_handlers/bridge_season_analytics.py."""
    import json as _json

    svc = await _ready_storage()

    from kawkab.ui.bridge_handlers.bridge_season_analytics import SeasonAnalyticsHandler

    handler = SeasonAnalyticsHandler(bridge=None, services={"storage_service": svc})
    payload = await handler.get_season_pro_report()
    try:
        return _json.loads(payload)
    except _json.JSONDecodeError:
        return {"raw": payload}


@router.get("/matches/{match_id}/analysis/report", response_model=MatchReportOut)
async def get_match_report(
    match_id: int, _user: dict = Depends(require_permission("analysis:read"))
):
    from kawkab.core.tactical_report import generate_tactical_report

    svc = await _ready_storage()
    match = await svc.get_match(match_id)
    if not match:
        _not_found(f"Match {match_id} not found")
    _check_match_access(match, _user)
    events = await svc.get_match_events(match_id)
    report = generate_tactical_report(
        events,
        match_id=match_id,
        home_team=match.get("home_team") or "Home",
        away_team=match.get("away_team") or "Away",
    )
    # summary/key_moments/areas_for_improvement have no corresponding field on
    # TacticalReport -- left at their model defaults rather than fabricated.
    return MatchReportOut(match_id=match_id, tactical_observations=report.key_tactical_observations)


# ── AI / LLM ──


@router.post("/matches/{match_id}/ai/ask", response_model=LlmQueryOut)
async def ask_llm(
    match_id: int, body: LlmQueryIn, _user: dict = Depends(require_permission("analysis:read"))
):
    from kawkab.services.llm_service import LLMConfig, LLMService

    svc = await _ready_storage()
    match = await svc.get_match(match_id)
    if not match:
        _not_found(f"Match {match_id} not found")
    _check_match_access(match, _user)
    events = await svc.get_match_events(match_id)
    context = json.dumps({"match_id": match_id, "events_count": len(events)}, indent=2)
    llm = LLMService(LLMConfig(provider="ollama"))
    answer = await llm.generate(
        f"You are a football analysis assistant. Match context:\n{context}\n\nQuestion: {body.question}"
    )
    return LlmQueryOut(
        answer=answer, model_used=llm.model_name if hasattr(llm, "model_name") else "default"
    )


# ── Player Ratings ──


@router.get("/matches/{match_id}/ratings", response_model=list[SquadSummaryOut])
async def get_player_ratings(
    match_id: int, _user: dict = Depends(require_permission("player:read"))
):
    from kawkab.services.rating_service import RatingService

    svc = await _ready_storage()
    match = await svc.get_match(match_id)
    if not match:
        _not_found(f"Match {match_id} not found")
    _check_match_access(match, _user)
    events = await svc.get_match_events(match_id)
    players = await svc.get_match_players(match_id)
    rating_svc = RatingService()
    ratings = rating_svc.compute_ratings(events, players)

    teams: dict[str, list[PlayerRatingOut]] = {}
    for r in ratings:
        team = r.get("team", "unknown")
        if team not in teams:
            teams[team] = []
        teams[team].append(
            PlayerRatingOut(
                track_id=r.get("track_id", 0),
                name=r.get("name", f"Player {r.get('track_id', '')}"),
                rating=float(r.get("rating", 0)),
                pass_accuracy=float(r.get("pass_accuracy", 0)),
                shot_impact=float(r.get("shot_impact", 0)),
                tackles=int(r.get("tackles", 0)),
            )
        )
    return [SquadSummaryOut(team=t, players=ps) for t, ps in teams.items()]


# ── Calibration & Model Comparison ──


@router.get("/matches/{match_id}/calibration", response_model=CalibrationOut)
async def get_calibration(
    match_id: int, _user: dict = Depends(require_permission("analysis:read"))
):
    from kawkab.core.calibration import ModelCalibrator

    svc = await _ready_storage()
    match = await svc.get_match(match_id)
    if not match:
        _not_found(f"Match {match_id} not found")
    _check_match_access(match, _user)
    events = await svc.get_match_events(match_id)
    calibrator = ModelCalibrator()
    report = calibrator.generate_calibration_report(events)
    return CalibrationOut(
        model_name="heuristic",
        total_xg=report.get("total_xg", 0),
        actual_goals=report.get("actual_goals", 0),
        calibration_error=report.get("calibration_error", 0),
        brier_score=report.get("brier_score", 0),
        log_loss=report.get("log_loss", 0),
        shots_evaluated=report.get("n_shots", 0),
        status=report.get("status", ""),
    )


@router.post("/model-comparison", response_model=ModelComparisonOut)
async def compare_models(
    shots: list[dict],
    n_folds: int = Query(5, ge=0, le=10),
    _user: dict = Depends(require_permission("analysis:run")),
):
    from kawkab.core.model_comparison import compare_xg_models

    report = compare_xg_models(shots, n_folds=n_folds, compute_feature_importance=True)
    return ModelComparisonOut(
        models=report.to_dict().get("models", []),
        best_model=report.best_model,
        cv_summary=report.cv_summary,
    )


# ── Fitness / Wearables ──


@router.get("/players/{track_id}/fitness", response_model=FitnessOut)
async def get_player_fitness(
    track_id: int,
    match_id: int = Query(..., description="Match ID"),
    _user: dict = Depends(require_permission("medical:read")),
):
    # PhysicalLoadService/WorkloadService need raw per-frame tracking data and
    # day-by-day season history respectively -- neither is derivable from
    # match events. GPS sessions and the acwr_daily table (populated by the
    # GPS import path) are the only real, already-measured source for this.
    svc = await _ready_storage()
    match = await svc.get_match(match_id)
    if not match:
        _not_found(f"Match {match_id} not found")
    _check_match_access(match, _user)
    players = await svc.get_match_players(match_id)
    player = next((p for p in players if p.get("track_id") == track_id), None)
    if not player:
        _not_found(f"Player {track_id} not found in match {match_id}")
    sessions = await svc.get_gps_sessions(match_id)
    session = next((s for s in sessions if s.get("player_id") == track_id), None)
    acwr_history = await svc.get_player_acwr(track_id, limit=1)
    total_distance = float(session.get("total_distance_m", 0)) if session is not None else 0.0
    max_speed = float(session.get("max_speed_kmh", 0)) if session is not None else 0.0
    return FitnessOut(
        player_name=player.get("name", f"Player {track_id}"),
        total_distance=total_distance,
        max_speed=max_speed,
        workload_score=float(acwr_history[0].get("acwr", 0)) if acwr_history else 0.0,
    )


# ── Recruitment ──


@router.post("/recruitment/search")
async def search_players(
    body: RecruitmentSearchIn, _user: dict = Depends(require_permission("recruitment:read"))
):
    try:
        from kawkab.services.player_search import PlayerSearchService

        search_svc = PlayerSearchService()
        results = search_svc.search(
            position=body.position,
            min_age=body.min_age,
            max_age=body.max_age,
            league=body.league,
            stat_thresholds=body.stat_thresholds,
            limit=body.limit,
        )
        return {"results": results, "total": len(results)}
    except ImportError:
        return {"results": [], "total": 0, "note": "PlayerSearchService not available"}


@router.get("/recruitment/transfer-fee/{player_name}", response_model=TransferFeeEstimateOut)
async def estimate_transfer_fee(
    player_name: str, _user: dict = Depends(require_permission("recruitment:read"))
):
    try:
        # estimate_player_transfer_fee takes age/position/performance, not a player name:
        # this endpoint has no per-player stats to feed it, so report a not-found-style
        # zero estimate rather than crashing with a TypeError.
        fee_data: dict = {
            "estimated_fee": 0,
            "fee_range_low": 0,
            "fee_range_high": 0,
            "confidence": "unavailable",
        }
        return TransferFeeEstimateOut(
            player_name=player_name,
            estimated_fee=fee_data.get("estimated_fee", 0),
            fee_range_low=fee_data.get("fee_range_low", 0),
            fee_range_high=fee_data.get("fee_range_high", 0),
            confidence=fee_data.get("confidence", "medium"),
        )
    except Exception:
        return TransferFeeEstimateOut(player_name=player_name)


@router.get("/recruitment/shortlist")
async def get_shortlist(_user: dict = Depends(require_permission("recruitment:read"))):
    try:
        from kawkab.services.shortlist_service import ShortlistService

        shortlist_svc = ShortlistService()
        return shortlist_svc.get_shortlist()
    except (ImportError, AttributeError):
        return []


# ── Game Plan ──


@router.get("/game-plan/{match_id}/vs/{opponent}", response_model=GamePlanOut)
async def get_game_plan(
    match_id: int, opponent: str, _user: dict = Depends(require_permission("analysis:read"))
):
    from kawkab.core.game_plan import GamePlanGenerator

    svc = await _ready_storage()
    match = await svc.get_match(match_id)
    if not match:
        _not_found(f"Match {match_id} not found")
    _check_match_access(match, _user)
    events = await svc.get_match_events(match_id)
    generator = GamePlanGenerator()
    plan = generator.generate(events, opponent=opponent)
    return GamePlanOut(**plan)


# ── Monitoring ──


@router.get("/monitoring/dashboard", response_model=MonitoringDashboardOut)
async def get_monitoring_dashboard(_user: dict = Depends(require_permission("admin:settings"))):
    monitor = _get_monitor()
    dashboard = monitor.get_monitoring_dashboard()
    return MonitoringDashboardOut(**dashboard)


@router.get("/monitoring/drift")
async def get_drift_alerts(_user: dict = Depends(require_permission("admin:settings"))):
    monitor = _get_monitor()
    alerts = monitor.monitor.detect_drift()
    return {
        "alerts": [
            {
                "model_name": a.model_name,
                "metric": a.metric,
                "message": a.message,
                "severity": a.severity,
            }
            for a in alerts
        ],
        "total": len(alerts),
    }


# ── Webhooks ──


@router.post("/webhooks", response_model=WebhookOut)
async def create_webhook(
    body: WebhookCreateIn, _user: dict = Depends(require_permission("admin:settings"))
):
    from kawkab.services.webhook_service import WebhookService

    wh_svc = WebhookService()
    wh = wh_svc.register(body.url, body.secret, body.events)
    return WebhookOut(**wh)


@router.get("/webhooks", response_model=list[WebhookOut])
async def list_webhooks(_user: dict = Depends(require_permission("admin:settings"))):
    from kawkab.services.webhook_service import WebhookService

    wh_svc = WebhookService()
    return [WebhookOut(**wh) for wh in wh_svc.list_all()]


@router.delete("/webhooks/{webhook_id}")
async def delete_webhook(
    webhook_id: int, _user: dict = Depends(require_permission("admin:settings"))
):
    from kawkab.services.webhook_service import WebhookService

    wh_svc = WebhookService()
    wh_svc.unregister(webhook_id)
    return {"ok": True}


# ── Season Summary ──


@router.get("/season/summary")
async def get_season_summary(_user: dict = Depends(require_permission("match:read"))):
    from kawkab.core.season_aggregator import SeasonAggregator

    aggregator = SeasonAggregator()
    return aggregator.aggregate_team_season([])


# ── Coding Tags ──


@router.get("/matches/{match_id}/coding/tags")
async def get_coding_tags(
    match_id: int,
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=1, le=100),
    _user: dict = Depends(require_permission("tag:read")),
):
    from kawkab.services.storage_service import StorageService

    svc = StorageService()
    match = await svc.get_match(match_id)
    if not match:
        _not_found(f"Match {match_id} not found")
    _check_match_access(match, _user)
    tags = await svc.get_coding_tags(match_id)
    return _paginate(tags, page, per_page)


@router.get("/matches/{match_id}/coding/tags/stats")
async def get_coding_stats(match_id: int, _user: dict = Depends(require_permission("tag:read"))):
    from kawkab.services.storage_service import StorageService

    svc = StorageService()
    match = await svc.get_match(match_id)
    if not match:
        _not_found(f"Match {match_id} not found")
    _check_match_access(match, _user)
    return await svc.get_coding_tag_stats(match_id)


@router.get("/matches/{match_id}/coding/tags/type/{tag_type}")
async def get_coding_tags_by_type(
    match_id: int, tag_type: str, _user: dict = Depends(require_permission("tag:read"))
):
    from kawkab.services.storage_service import StorageService

    svc = StorageService()
    match = await svc.get_match(match_id)
    if not match:
        _not_found(f"Match {match_id} not found")
    _check_match_access(match, _user)
    return await svc.get_coding_tags_by_type(match_id, tag_type)


# ── Injury / Medical ──


@router.get("/players/{player_id}/injury-risk")
async def get_player_injury_risk(
    player_id: int, _user: dict = Depends(require_permission("medical:read"))
):
    from kawkab.core.injury_risk import InjuryRiskPredictor

    svc = await _ready_storage()
    acwr_history = await svc.get_player_acwr(player_id, limit=1)
    if not acwr_history:
        return {
            "player_id": player_id,
            "data_available": False,
            "reason": "no GPS/workload data on file for this player",
        }
    latest = acwr_history[0]
    pred = InjuryRiskPredictor()
    risk = pred.predict_injury_risk({"acwr": latest.get("acwr", 1.0)})
    return {
        "player_id": player_id,
        "data_available": True,
        "acwr": latest.get("acwr"),
        "acwr_category": latest.get("load_category"),
        **risk,
    }


@router.get("/squad/{team_id}/injury-report")
async def get_squad_injury_report(
    team_id: int, _user: dict = Depends(require_permission("medical:read"))
):
    svc = await _ready_storage()
    return await svc.get_squad_injury_report(team_id)


# ── Streaming ──


@router.get("/streaming/status")
async def get_streaming_status(_user: dict = Depends(require_permission("match:read"))):
    return {"status": "idle"}


@router.post("/streaming/start")
async def start_streaming(
    source: str = "", _user: dict = Depends(require_permission("analysis:run"))
):
    return {"status": "started", "source": source}


@router.post("/streaming/stop")
async def stop_streaming(_user: dict = Depends(require_permission("analysis:run"))):
    return {"status": "stopped"}


# ── Collaboration ──


@router.get("/collaboration/sessions")
async def get_collab_sessions(_user: dict = Depends(require_permission("admin:settings"))):
    from kawkab.cloud.server import connected_clients

    return {
        "sessions": [
            {"project_id": pid, "clients": len(clients)}
            for pid, clients in connected_clients.items()
        ],
        "total_sessions": len(connected_clients),
    }


# ── Model Cards ──


@router.get("/model-cards", response_model=list[ModelCardOut])
async def list_model_cards():
    from kawkab.core.model_card_registry import list_model_cards

    return [ModelCardOut(**m.__dict__) for m in list_model_cards()]


@router.get("/model-cards/{name}", response_model=ModelCardOut)
async def get_model_card(name: str):
    from kawkab.core.model_card_registry import get_model_card

    card = get_model_card(name)
    if not card:
        raise HTTPException(404, f"Model card '{name}' not found")
    return ModelCardOut(**card.__dict__)


# ── Health ──


@router.get("/feedback")
async def get_all_feedback(
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=1, le=100),
    _user: dict = Depends(require_permission("admin:settings")),
):
    from kawkab.services.storage_service import StorageService

    svc = StorageService()
    feedback = await svc.get_all_feedback()
    return _paginate(feedback, page, per_page)


@router.get("/issues")
async def get_all_issues(
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=1, le=100),
    _user: dict = Depends(require_permission("admin:settings")),
):
    from kawkab.services.storage_service import StorageService

    svc = StorageService()
    issues = await svc.get_all_issues()
    return _paginate(issues, page, per_page)


@router.get("/playlists")
async def get_playlists(
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=1, le=100),
    _user: dict = Depends(require_permission("match:read")),
):
    from kawkab.services.storage_service import StorageService

    svc = StorageService()
    playlists = await svc.get_playlists()
    return _paginate(playlists, page, per_page)


@router.get("/matches/{match_id}/reports")
async def get_reports(
    match_id: int,
    language: str = Query(""),
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=1, le=100),
    _user: dict = Depends(require_permission("analysis:read")),
):
    from kawkab.services.storage_service import StorageService

    svc = StorageService()
    match = await svc.get_match(match_id)
    if not match:
        _not_found(f"Match {match_id} not found")
    _check_match_access(match, _user)
    reports = await svc.get_reports(match_id, language)
    return _paginate(reports, page, per_page)


@router.get("/health")
async def api_health():
    return {"status": "ok", "api_version": "v1"}


# ── Vendor tracking import (elite interop path) ──


class TrackingImportIn(BaseModel):
    file_path: str = ""
    vendor: str = ""  # skillcorner | epts | metrica (auto-detected if empty)
    match_id: int | None = None  # attach to an existing match
    match_name: str = ""
    home_team: str = ""
    away_team: str = ""
    away_csv: str = ""  # metrica only: the away CSV path
    max_frames: int = 0  # 0 = all frames
    fps: float | None = None


class TrackingImportOut(BaseModel):
    success: bool
    match_id: int
    vendor: str
    deduplicated: bool
    frames_imported: int
    players_registered: int
    checksum: str
    match_name: str = ""
    source: str = "tracking"


@router.post("/matches/import/tracking", response_model=TrackingImportOut)
async def import_tracking_match(
    body: TrackingImportIn,
    _user: dict = Depends(require_permission("analysis:run")),
):
    """Import a vendor tracking feed (SkillCorner JSON / EPTS XML / Metrica CSVs)
    as a Kawkab match — zero video capture. RBAC-gated to analysis:run;
    file_path validated against the SecurityValidator allowlist, same as
    every other local-file API."""
    from kawkab.core.security import SecurityValidator
    from kawkab.services.vendor_tracking_import_service import VendorTrackingImportService

    if not body.file_path:
        raise HTTPException(400, "file_path is required")
    try:
        SecurityValidator.validate_data_file_path(body.file_path)
        if body.away_csv:
            SecurityValidator.validate_data_file_path(body.away_csv)
    except Exception as exc:
        raise HTTPException(400, f"file_path rejected: {exc}") from exc
    for p in (body.file_path, body.away_csv):
        if p and not Path(p).exists():
            raise HTTPException(404, f"file not found: {p}")

    storage = await _ready_storage()
    svc = VendorTrackingImportService(storage)
    try:
        summary = await svc.import_tracking_file(
            body.file_path,
            vendor=body.vendor or None,
            match_id=body.match_id,
            match_name=body.match_name or None,
            home_team=body.home_team or None,
            away_team=body.away_team or None,
            away_csv=body.away_csv or None,
            max_frames=body.max_frames or None,
            fps=body.fps,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    # Audit trail (only on a fresh, non-dedup import)
    if not summary.get("deduplicated"):
        _get_audit().log_event(
            "match.imported",
            "match",
            str(summary["match_id"]),
            details={
                "vendor": summary["vendor"],
                "frames_imported": summary["frames_imported"],
                "source_file": body.file_path,
            },
            user=str(_user.get("sub", "local")),
        )

    return TrackingImportOut(
        success=True,
        match_id=summary["match_id"],
        vendor=summary["vendor"],
        deduplicated=summary.get("deduplicated", False),
        frames_imported=summary["frames_imported"],
        players_registered=summary["players_registered"],
        checksum=summary.get("checksum", ""),
        match_name=summary.get("match_name", ""),
    )


# ── Vendor event-data import (Opta F24 / Wyscout) ──


class EventImportIn(BaseModel):
    vendor: str  # opta | wyscout
    file_path: str
    f7_path: str = ""  # opta only: paired F7 match-info XML
    match_id: int | None = None
    match_name: str = ""
    home_team: str = ""
    away_team: str = ""


class EventImportOut(BaseModel):
    success: bool
    match_id: int
    vendor: str
    events_imported: int
    events_skipped: int
    players_registered: int
    shots: int
    goals: int
    xg_total: float
    match_name: str = ""


@router.post("/matches/import/events", response_model=EventImportOut)
async def import_vendor_events(
    body: EventImportIn,
    _user: dict = Depends(require_permission("analysis:run")),
):
    """Import a vendor event-data file (Opta F24 XML / Wyscout JSON) as a
    Kawkab match — zero video capture. RBAC-gated to analysis:run;
    file_path validated against the SecurityValidator allowlist."""
    from kawkab.core.security import SecurityValidator
    from kawkab.services.vendor_event_import_service import VendorEventImportService

    vendor = (body.vendor or "").lower()
    if vendor not in ("opta", "wyscout"):
        raise HTTPException(400, "vendor must be 'opta' or 'wyscout'")
    try:
        SecurityValidator.validate_data_file_path(body.file_path)
        if body.f7_path:
            SecurityValidator.validate_data_file_path(body.f7_path)
    except Exception as exc:
        raise HTTPException(400, f"file_path rejected: {exc}") from exc
    for p in (body.file_path, body.f7_path):
        if p and not Path(p).exists():
            raise HTTPException(404, f"file not found: {p}")

    storage = await _ready_storage()
    svc = VendorEventImportService(storage)
    try:
        if vendor == "opta":
            summary = await svc.import_opta_f24(
                body.file_path,
                body.f7_path or None,
                match_name=body.match_name or None,
                home_team=body.home_team or None,
                away_team=body.away_team or None,
                match_id=body.match_id,
            )
        else:
            summary = await svc.import_wyscout(
                body.file_path,
                match_name=body.match_name or None,
                home_team=body.home_team or None,
                away_team=body.away_team or None,
                match_id=body.match_id,
            )
        _get_audit().log_event(
            "match.imported",
            "match",
            str(summary["match_id"]),
            details={
                "vendor": vendor,
                "events_imported": summary["events_imported"],
                "source_file": body.file_path,
            },
            user=str(_user.get("sub", "local")),
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return EventImportOut(
        success=True,
        match_id=summary["match_id"],
        vendor=summary["vendor"],
        events_imported=summary["events_imported"],
        events_skipped=summary["events_skipped"],
        players_registered=summary["players_registered"],
        shots=summary["shots"],
        goals=summary["goals"],
        xg_total=summary["xg_total"],
        match_name=summary.get("match_name", ""),
    )


# ── StatsBomb event-data import (elite interop path) ──


class StatsBombImportIn(BaseModel):
    events_json: str = ""
    file_path: str = ""
    match_name: str = ""
    home_team: str = ""
    away_team: str = ""


class StatsBombImportOut(BaseModel):
    success: bool
    match_id: int
    events_imported: int
    events_skipped: int
    players_registered: int
    shots: int
    goals: int
    xg_total: float
    match_name: str
    source: str = "statsbomb"


class AuditLogOut(BaseModel):
    success: bool
    events: list[dict]
    total: int


@router.get("/audit/events", response_model=AuditLogOut)
async def get_audit_events(
    action: str | None = None,
    entity_type: str | None = None,
    limit: int = 100,
    offset: int = 0,
    _user: dict = Depends(require_permission("analysis:read")),
):
    """Query the hash-chained audit trail (who imported/ran/exported what
    and when). RBAC-gated to analysis:read; analyst-level minimum."""
    audit = _get_audit()
    events = audit.get_events(
        action=action, entity_type=entity_type, limit=min(limit, 500), offset=max(offset, 0)
    )
    return AuditLogOut(success=True, events=events, total=len(events))


@router.post("/matches/import/statsbomb", response_model=StatsBombImportOut)
async def import_statsbomb_match(
    body: StatsBombImportIn,
    _user: dict = Depends(require_permission("analysis:run")),
):
    """Import a StatsBomb events file (path or inline JSON) as a Kawkab match.

    Lets clubs run the full Kawkab analytics stack on their existing
    StatsBomb event data with zero video capture. RBAC-gated to
    analysis:run; file_path is validated against the SecurityValidator's
    directory allowlist, same as every other local-file API.
    """
    import json as _json
    from pathlib import Path as _Path

    from kawkab.core.security import SecurityValidator
    from kawkab.services.statsbomb_import_service import StatsBombImportService

    storage = await _ready_storage()

    tmp_json_path = ""
    try:
        if body.file_path:
            try:
                SecurityValidator.validate_data_file_path(body.file_path)
            except Exception as exc:
                raise HTTPException(400, f"file_path rejected: {exc}") from exc
            source_path = _Path(body.file_path)
            if not source_path.exists():
                raise HTTPException(404, f"file not found: {body.file_path}")
        elif body.events_json:
            import tempfile as _tempfile

            data = _json.loads(body.events_json)
            if not isinstance(data, list):
                raise HTTPException(400, "events_json must be a JSON array of events")
            fd, tmp_json_path = _tempfile.mkstemp(suffix=".json")
            with open(fd, "w") as f:
                _json.dump(data, f)
            source_path = _Path(tmp_json_path)
        else:
            raise HTTPException(400, "provide either file_path or events_json")

        svc = StatsBombImportService(storage)
        summary = await svc.import_match(
            source_path,
            match_name=body.match_name or None,
            home_team=body.home_team or None,
            away_team=body.away_team or None,
        )
        _get_audit().log_event(
            "match.imported",
            "match",
            str(summary["match_id"]),
            details={
                "vendor": "statsbomb",
                "events_imported": summary["events_imported"],
                "source_file": str(source_path),
            },
            user=str(_user.get("sub", "local")),
        )
        return StatsBombImportOut(
            success=True,
            match_id=summary["match_id"],
            events_imported=summary["events_imported"],
            events_skipped=summary["events_skipped"],
            players_registered=summary["players_registered"],
            shots=summary["shots"],
            goals=summary["goals"],
            xg_total=summary["xg_total"],
            match_name=summary["match_name"],
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — API boundary
        raise HTTPException(500, f"import failed: {exc}") from exc
    finally:
        if tmp_json_path:
            with contextlib.suppress(OSError):
                _Path(tmp_json_path).unlink()


class SeasonImportIn(BaseModel):
    directory: str
    competition: str | None = None
    season_id: int | None = None
    match_date: str | None = None
    max_matches: int | None = None


class SeasonImportOut(BaseModel):
    success: bool
    directory: str
    total_files: int
    eligible: int
    imported: int
    skipped_already: int
    skipped_not_events: int
    failed: int
    matches: list[dict]
    source: str = "statsbomb"


@router.post("/matches/import/statsbomb/season", response_model=SeasonImportOut)
async def import_statsbomb_season(
    body: SeasonImportIn,
    _user: dict = Depends(require_permission("analysis:run")),
):
    """Bulk-import a whole directory of StatsBomb event files as one season.

    The season-scale workflow elite analysts live in: one call turns a
    folder of vendor match files into deduplicated matches tagged with
    competition/date/season (migration 031 external-ID registry makes
    re-runs idempotent). RBAC-gated to analysis:run; the directory is
    validated against the same documents-dir allowlist as every other
    local-file API.
    """
    from kawkab.core.security import SecurityValidator
    from kawkab.services.season_import_service import SeasonImportService

    storage = await _ready_storage()
    try:
        try:
            SecurityValidator.validate_directory_path(body.directory)
        except Exception as exc:
            raise HTTPException(400, f"directory rejected: {exc}") from exc

        svc = SeasonImportService(storage)
        summary = await svc.import_statsbomb_directory(
            body.directory,
            competition=body.competition,
            season_id=body.season_id,
            match_date=body.match_date,
            max_matches=body.max_matches,
        )
        _get_audit().log_event(
            "season.imported",
            "match",
            body.directory,
            details={
                "vendor": "statsbomb",
                "imported": summary["imported"],
                "skipped_already": summary["skipped_already"],
                "failed": summary["failed"],
            },
            user=str(_user.get("sub", "local")),
        )
        return SeasonImportOut(
            success=True,
            directory=str(summary["directory"]),
            **{k: v for k, v in summary.items() if k != "directory"},
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — API boundary
        raise HTTPException(500, f"season import failed: {exc}") from exc
