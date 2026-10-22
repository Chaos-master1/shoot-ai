"""Tests for anomaly detection service."""

import pytest
from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()

_mod = load_service_module("anomaly_test", "anomaly_detection_service.py")
AnomalyDetectionService = _mod.AnomalyDetectionService
Anomaly = _mod.Anomaly


# ---------------------------------------------------------------------------
# Fake data helpers
# ---------------------------------------------------------------------------


class FakePlayer:
    def __init__(self, max_speed_kmh=0, distance_covered_m=0):
        self.max_speed_kmh = max_speed_kmh
        self.distance_covered_m = distance_covered_m


class FakeTeam:
    def __init__(self, possession_pct=50):
        self.possession_pct = possession_pct


class FakeTrackData:
    def __init__(self, tracking_metrics=None):
        self.tracking_metrics = tracking_metrics or {}


class FakeAnalysis:
    def __init__(self, players=None, home_team=None, away_team=None, formations=None):
        self.players = players or {}
        self.home_team = home_team
        self.away_team = away_team
        self.formations = formations or {}


# ===================================================================
# Physical anomaly detection
# ===================================================================


class TestPhysicalAnomalies:
    def test_speed_above_max_detected(self):
        svc = AnomalyDetectionService()
        analysis = FakeAnalysis(players={"p1": FakePlayer(max_speed_kmh=45.0)})
        anomalies = svc._check_physical_stats(analysis)
        assert len(anomalies) == 1
        a = anomalies[0]
        assert a.category == "physical"
        assert a.severity == "critical"
        assert a.metric == "max_speed"
        assert "Usain Bolt" in a.description

    def test_speed_normal_no_anomaly(self):
        svc = AnomalyDetectionService()
        analysis = FakeAnalysis(players={"p1": FakePlayer(max_speed_kmh=35.0)})
        anomalies = svc._check_physical_stats(analysis)
        assert len(anomalies) == 0

    def test_distance_above_max_detected(self):
        svc = AnomalyDetectionService()
        analysis = FakeAnalysis(players={"p1": FakePlayer(distance_covered_m=20000.0)})
        anomalies = svc._check_physical_stats(analysis)
        assert len(anomalies) == 1
        a = anomalies[0]
        assert a.category == "physical"
        assert a.severity == "high"
        assert a.metric == "distance_covered"

    def test_distance_normal_no_anomaly(self):
        svc = AnomalyDetectionService()
        analysis = FakeAnalysis(players={"p1": FakePlayer(distance_covered_m=10000.0)})
        anomalies = svc._check_physical_stats(analysis)
        assert len(anomalies) == 0

    def test_multiple_physical_anomalies_on_same_player(self):
        svc = AnomalyDetectionService()
        analysis = FakeAnalysis(
            players={"p1": FakePlayer(max_speed_kmh=50.0, distance_covered_m=18000.0)}
        )
        anomalies = svc._check_physical_stats(analysis)
        assert len(anomalies) == 2
        metrics = {a.metric for a in anomalies}
        assert metrics == {"max_speed", "distance_covered"}


# ===================================================================
# Tracking quality checks
# ===================================================================


class TestTrackingQuality:
    def test_too_few_tracks_detected(self):
        svc = AnomalyDetectionService()
        track_data = FakeTrackData({"validated_player_tracks": 5})
        anomalies = svc._check_tracking_quality(track_data)
        assert len(anomalies) >= 1
        high_anomalies = [a for a in anomalies if a.metric == "validated_tracks"]
        assert len(high_anomalies) == 1
        assert high_anomalies[0].severity == "high"

    def test_too_many_tracks_detected(self):
        svc = AnomalyDetectionService()
        track_data = FakeTrackData({"validated_player_tracks": 50})
        anomalies = svc._check_tracking_quality(track_data)
        assert len(anomalies) >= 1
        track_anomalies = [a for a in anomalies if a.metric == "validated_tracks"]
        assert len(track_anomalies) == 1
        assert track_anomalies[0].severity == "medium"

    def test_fragmentation_above_max_detected(self):
        svc = AnomalyDetectionService()
        track_data = FakeTrackData({"fragmentation_rate": 8.0})
        anomalies = svc._check_tracking_quality(track_data)
        assert len(anomalies) >= 1
        frag_anomalies = [a for a in anomalies if a.metric == "fragmentation_rate"]
        assert len(frag_anomalies) == 1
        assert frag_anomalies[0].severity == "high"

    def test_poor_tracking_quality_detected(self):
        svc = AnomalyDetectionService()
        track_data = FakeTrackData({"tracking_quality": "poor"})
        anomalies = svc._check_tracking_quality(track_data)
        assert len(anomalies) >= 1
        qual_anomalies = [a for a in anomalies if a.metric == "tracking_quality"]
        assert len(qual_anomalies) == 1
        assert qual_anomalies[0].severity == "high"

    def test_very_poor_tracking_quality_detected(self):
        svc = AnomalyDetectionService()
        track_data = FakeTrackData({"tracking_quality": "very_poor"})
        anomalies = svc._check_tracking_quality(track_data)
        qual_anomalies = [a for a in anomalies if a.metric == "tracking_quality"]
        assert len(qual_anomalies) == 1

    def test_no_tracking_quality_issues(self):
        svc = AnomalyDetectionService()
        track_data = FakeTrackData(
            {
                "validated_player_tracks": 25,
                "fragmentation_rate": 2.0,
                "tracking_quality": "good",
            }
        )
        anomalies = svc._check_tracking_quality(track_data)
        assert len(anomalies) == 0

    def test_tracking_metrics_missing_no_crash(self):
        svc = AnomalyDetectionService()
        track_data = FakeTrackData({})
        anomalies = svc._check_tracking_quality(track_data)
        assert all(a.category == "tracking" for a in anomalies)
        assert len([a for a in anomalies if a.metric == "validated_tracks"]) == 1

    def test_tracking_quality_excellent_no_anomaly(self):
        svc = AnomalyDetectionService()
        track_data = FakeTrackData({"tracking_quality": "excellent"})
        anomalies = svc._check_tracking_quality(track_data)
        qual_anomalies = [a for a in anomalies if a.metric == "tracking_quality"]
        assert len(qual_anomalies) == 0


# ===================================================================
# Team / statistical outlier checks
# ===================================================================


class TestTeamStats:
    def test_extreme_possession_split_detected(self):
        svc = AnomalyDetectionService()
        analysis = FakeAnalysis(
            home_team=FakeTeam(possession_pct=80),
            away_team=FakeTeam(possession_pct=20),
        )
        anomalies = svc._check_team_stats(analysis)
        assert len(anomalies) == 1
        a = anomalies[0]
        assert a.category == "team"
        assert a.severity == "medium"
        assert a.metric == "possession_split"

    def test_normal_possession_no_anomaly(self):
        svc = AnomalyDetectionService()
        analysis = FakeAnalysis(
            home_team=FakeTeam(possession_pct=55),
            away_team=FakeTeam(possession_pct=45),
        )
        anomalies = svc._check_team_stats(analysis)
        assert len(anomalies) == 0

    def test_home_team_missing_no_anomaly(self):
        svc = AnomalyDetectionService()
        analysis = FakeAnalysis(home_team=None)
        anomalies = svc._check_team_stats(analysis)
        assert len(anomalies) == 0

    def test_away_team_missing_no_anomaly(self):
        svc = AnomalyDetectionService()
        analysis = FakeAnalysis(away_team=None)
        anomalies = svc._check_team_stats(analysis)
        assert len(anomalies) == 0

    def test_borderline_possession_40_percent_diff_no_anomaly(self):
        svc = AnomalyDetectionService()
        analysis = FakeAnalysis(
            home_team=FakeTeam(possession_pct=70),
            away_team=FakeTeam(possession_pct=30),
        )
        anomalies = svc._check_team_stats(analysis)
        assert len(anomalies) == 0

    def test_extreme_possession_above_40_pct_detected(self):
        svc = AnomalyDetectionService()
        analysis = FakeAnalysis(
            home_team=FakeTeam(possession_pct=71),
            away_team=FakeTeam(possession_pct=29),
        )
        anomalies = svc._check_team_stats(analysis)
        assert len(anomalies) == 1


# ===================================================================
# Formation checks
# ===================================================================


class TestFormationChecks:
    def test_unusual_formation_detected(self):
        svc = AnomalyDetectionService()
        analysis = FakeAnalysis(formations={"home": {"formation": "2-3-5"}})
        anomalies = svc._check_formations(analysis)
        assert len(anomalies) == 1
        a = anomalies[0]
        assert a.category == "team"
        assert a.severity == "low"
        assert a.metric == "formation"

    def test_unknown_formation_ignored(self):
        svc = AnomalyDetectionService()
        analysis = FakeAnalysis(formations={"home": {"formation": "unknown"}})
        anomalies = svc._check_formations(analysis)
        assert len(anomalies) == 0

    def test_standard_formation_no_anomaly(self):
        svc = AnomalyDetectionService()
        analysis = FakeAnalysis(formations={"home": {"formation": "4-3-3"}})
        anomalies = svc._check_formations(analysis)
        assert len(anomalies) == 0

    def test_both_teams_unusual_formations(self):
        svc = AnomalyDetectionService()
        analysis = FakeAnalysis(
            formations={
                "home": {"formation": "1-4-5"},
                "away": {"formation": "2-7-1"},
            }
        )
        anomalies = svc._check_formations(analysis)
        assert len(anomalies) == 2

    def test_formations_key_missing_no_crash(self):
        svc = AnomalyDetectionService()
        analysis = FakeAnalysis(formations={})
        anomalies = svc._check_formations(analysis)
        assert len(anomalies) == 0


# ===================================================================
# Event anomaly detection
# ===================================================================


class TestEventAnomalies:
    def test_too_many_shots_detected(self):
        svc = AnomalyDetectionService()
        events = [{"type": "shot", "metadata": {}} for _ in range(60)]
        anomalies = svc._check_events(events)
        shot_anomalies = [a for a in anomalies if a.metric == "shot_count"]
        assert len(shot_anomalies) == 1
        assert shot_anomalies[0].severity == "medium"

    def test_too_few_passes_detected(self):
        svc = AnomalyDetectionService()
        events = [{"type": "pass", "metadata": {}} for _ in range(2)]
        anomalies = svc._check_events(events)
        pass_anomalies = [a for a in anomalies if a.metric == "pass_count"]
        assert len(pass_anomalies) == 1
        assert pass_anomalies[0].severity == "high"

    def test_impossible_pass_distance_detected(self):
        svc = AnomalyDetectionService()
        events = [{"type": "pass", "metadata": {"distance_m": 120.0}}]
        anomalies = svc._check_events(events)
        dist_anomalies = [a for a in anomalies if a.metric == "pass_distance"]
        assert len(dist_anomalies) == 1
        assert dist_anomalies[0].severity == "medium"

    def test_normal_pass_distance_no_anomaly(self):
        svc = AnomalyDetectionService()
        events = [{"type": "pass", "metadata": {"distance_m": 30.0}}]
        anomalies = svc._check_events(events)
        dist_anomalies = [a for a in anomalies if a.metric == "pass_distance"]
        assert len(dist_anomalies) == 0

    def test_empty_events_no_anomaly(self):
        svc = AnomalyDetectionService()
        anomalies = svc._check_events([])
        assert len(anomalies) == 0

    def test_only_reports_first_impossible_pass_once(self):
        svc = AnomalyDetectionService()
        events = [
            {"type": "pass", "metadata": {"distance_m": 120.0}},
            {"type": "pass", "metadata": {"distance_m": 200.0}},
            {"type": "pass", "metadata": {"distance_m": 30.0}},
        ]
        anomalies = svc._check_events(events)
        dist_anomalies = [a for a in anomalies if a.metric == "pass_distance"]
        assert len(dist_anomalies) == 1

    def test_non_pass_events_skipped_for_distance_check(self):
        svc = AnomalyDetectionService()
        events = [{"type": "shot", "metadata": {"distance_m": 200.0}}]
        anomalies = svc._check_events(events)
        dist_anomalies = [a for a in anomalies if a.metric == "pass_distance"]
        assert len(dist_anomalies) == 0


# ===================================================================
# Full pipeline (detect_anomalies)
# ===================================================================


class TestDetectAnomalies:
    @pytest.mark.asyncio
    async def test_full_pipeline_all_sections(self):
        svc = AnomalyDetectionService()
        track_data = FakeTrackData({"validated_player_tracks": 5})
        analysis = FakeAnalysis(
            players={"p1": FakePlayer(max_speed_kmh=50.0)},
            home_team=FakeTeam(possession_pct=80),
            away_team=FakeTeam(possession_pct=20),
            formations={"away": {"formation": "2-3-5"}},
        )
        events = [{"type": "shot", "metadata": {}} for _ in range(60)]
        anomalies = await svc.detect_anomalies(
            track_data=track_data, analysis=analysis, events=events
        )
        assert len(anomalies) >= 4
        categories = {a.category for a in anomalies}
        assert "tracking" in categories
        assert "physical" in categories
        assert "team" in categories
        assert "events" in categories
        severities = [a.severity for a in anomalies]
        critical_first = severities.index("critical")
        high_after = any(s in ("high", "medium", "low") for s in severities[:critical_first])
        assert not high_after, "critical should be sorted first"

    @pytest.mark.asyncio
    async def test_no_data_returns_empty(self):
        svc = AnomalyDetectionService()
        anomalies = await svc.detect_anomalies()
        assert len(anomalies) == 0

    @pytest.mark.asyncio
    async def test_no_players_no_physical_anomalies(self):
        svc = AnomalyDetectionService()
        analysis = FakeAnalysis(players={})
        anomalies = await svc.detect_anomalies(analysis=analysis)
        physical = [a for a in anomalies if a.category == "physical"]
        assert len(physical) == 0

    @pytest.mark.asyncio
    async def test_all_normal_data_no_anomalies(self):
        svc = AnomalyDetectionService()
        track_data = FakeTrackData(
            {
                "validated_player_tracks": 25,
                "fragmentation_rate": 2.0,
                "tracking_quality": "good",
            }
        )
        analysis = FakeAnalysis(
            players={"p1": FakePlayer(max_speed_kmh=32.0, distance_covered_m=10000.0)},
            home_team=FakeTeam(possession_pct=52),
            away_team=FakeTeam(possession_pct=48),
            formations={"home": {"formation": "4-3-3"}},
        )
        events = [{"type": "pass", "metadata": {"distance_m": 30.0}} for _ in range(10)] + [
            {"type": "shot", "metadata": {}} for _ in range(5)
        ]
        anomalies = await svc.detect_anomalies(
            track_data=track_data, analysis=analysis, events=events
        )
        assert len(anomalies) == 0


# ===================================================================
# Quality report generation
# ===================================================================


class TestGenerateQualityReport:
    @pytest.mark.asyncio
    async def test_quality_report_clean_data(self):
        svc = AnomalyDetectionService()
        report = await svc.generate_quality_report([])
        assert report["overall_score"] == 1.0
        assert report["total_issues"] == 0
        assert report["passes"] is True

    @pytest.mark.asyncio
    async def test_quality_report_deducts_for_severity(self):
        svc = AnomalyDetectionService()
        anomalies = [
            Anomaly("physical", "critical", "speed", "<=40", "45", "desc", "rec"),
            Anomaly("tracking", "high", "tracks", ">=18", "10", "desc", "rec"),
            Anomaly("tracking", "medium", "frag", "<=5", "8", "desc", "rec"),
            Anomaly("team", "low", "formation", "std", "2-3-5", "desc", "rec"),
        ]
        report = await svc.generate_quality_report(anomalies)
        assert report["total_issues"] == 4
        assert report["critical"] == 1
        assert report["high"] == 1
        assert report["medium"] == 1
        assert report["low"] == 1
        assert report["passes"] is False
        assert report["overall_score"] < 1.0

    @pytest.mark.asyncio
    async def test_quality_report_issues_structured(self):
        svc = AnomalyDetectionService()
        anomalies = [
            Anomaly("physical", "critical", "speed", "<=40", "45", "Too fast", "Slow down"),
        ]
        report = await svc.generate_quality_report(anomalies)
        assert len(report["issues"]) == 1
        issue = report["issues"][0]
        assert issue["category"] == "physical"
        assert issue["severity"] == "critical"
        assert issue["metric"] == "speed"
        assert issue["expected"] == "<=40"
        assert issue["actual"] == "45"
        assert issue["description"] == "Too fast"
        assert issue["recommendation"] == "Slow down"

    @pytest.mark.asyncio
    async def test_quality_report_score_floor_zero(self):
        svc = AnomalyDetectionService()
        anomalies = [
            Anomaly("physical", "critical", "s", "r", "v", "d", "r"),
            Anomaly("physical", "critical", "s", "r", "v", "d", "r"),
            Anomaly("physical", "critical", "s", "r", "v", "d", "r"),
            Anomaly("physical", "critical", "s", "r", "v", "d", "r"),
        ]
        report = await svc.generate_quality_report(anomalies)
        assert report["overall_score"] >= 0.0


# ===================================================================
# Anomaly dataclass
# ===================================================================


class TestAnomalyDataclass:
    def test_anomaly_creation(self):
        a = Anomaly(
            "tracking",
            "high",
            "validated_tracks",
            ">= 18",
            "10",
            "Only 10 tracks",
            "Check video quality",
        )
        assert a.category == "tracking"
        assert a.severity == "high"
        assert a.metric == "validated_tracks"
        assert a.expected_range == ">= 18"
        assert a.actual_value == "10"
        assert a.description == "Only 10 tracks"
        assert a.recommendation == "Check video quality"

    def test_anomaly_default_fields(self):
        a = Anomaly("events", "low", "passes", ">= 50", "0", "desc", "rec")
        assert a.category == "events"
        assert a.severity == "low"
