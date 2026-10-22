"""Phase 6 Sprint 2: Live Tagging Dashboard tests."""

import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
SRC = BASE / "src"
WEB_JS = BASE / "src" / "kawkab" / "web" / "js"
WEB_CSS = BASE / "src" / "kawkab" / "web" / "css"
INDEX = BASE / "src" / "kawkab" / "web" / "index.html"
BRIDGE_ANALYSIS = BASE / "src" / "kawkab" / "ui" / "bridge_handlers" / "bridge_analysis.py"
BRIDGE_LIVE = BASE / "src" / "kawkab" / "ui" / "bridge_handlers" / "bridge_live.py"
BRIDGE = BASE / "src" / "kawkab" / "ui" / "bridge.py"
SERVICE = BASE / "src" / "kawkab" / "services" / "live_tagging_service.py"


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class TestBridgeAnalysisMethods:
    """Deliverable 1: verify get_live_kpis / get_live_pitch_map / get_live_xg_chart exist."""

    def test_get_live_kpis_exists(self):
        code = _read(BRIDGE_LIVE)
        assert "async def get_live_kpis" in code

    def test_get_live_pitch_map_exists(self):
        code = _read(BRIDGE_LIVE)
        assert "async def get_live_pitch_map" in code

    def test_get_live_xg_chart_exists(self):
        code = _read(BRIDGE_LIVE)
        assert "async def get_live_xg_chart" in code

    def test_get_live_kpis_returns_json(self):
        code = _read(BRIDGE_LIVE)
        assert (
            "return json.dumps({" in code.split("async def get_live_kpis")[1].split("async def")[0]
        )

    def test_get_live_pitch_map_returns_json(self):
        code = _read(BRIDGE_LIVE)
        assert "home_events" in code.split("async def get_live_pitch_map")[1].split("async def")[0]

    def test_get_live_xg_chart_returns_json(self):
        code = _read(BRIDGE_LIVE)
        assert "timeline" in code.split("async def get_live_xg_chart")[1].split("async def")[0]

    def test_get_live_kpis_has_possession_key(self):
        code = _read(BRIDGE_LIVE)
        body = code.split("async def get_live_kpis")[1].split("async def")[0]
        assert '"possession_pct"' in body
        assert '"shots"' in body
        assert '"shots_ontarget"' in body
        assert '"goals"' in body
        assert '"xg"' in body
        assert '"xg_diff"' in body
        assert '"period"' in body
        assert '"team_stats"' in body

    def test_get_live_pitch_map_has_zones(self):
        code = _read(BRIDGE_LIVE)
        body = code.split("async def get_live_pitch_map")[1].split("async def")[0]
        assert "home_hot_zones" in body
        assert "away_hot_zones" in body

    def test_get_live_xg_chart_has_cumulative(self):
        code = _read(BRIDGE_LIVE)
        body = code.split("async def get_live_xg_chart")[1].split("async def")[0]
        assert "cumulative_home" in body
        assert "cumulative_away" in body


class TestBridgeSlots:
    """Deliverable 1: verify @Slot delegators in bridge.py."""

    def test_get_live_kpis_slot_exists(self):
        code = _read(BRIDGE)
        assert "async def get_live_kpis" in code

    def test_get_live_pitch_map_slot_exists(self):
        code = _read(BRIDGE)
        assert "async def get_live_pitch_map" in code

    def test_get_live_xg_chart_slot_exists(self):
        code = _read(BRIDGE)
        assert "async def get_live_xg_chart" in code

    def test_get_live_kpis_slot_delegates(self):
        code = _read(BRIDGE)
        assert "self._live.get_live_kpis(" in code

    def test_get_live_pitch_map_slot_delegates(self):
        code = _read(BRIDGE)
        assert "self._live.get_live_pitch_map(" in code

    def test_get_live_xg_chart_slot_delegates(self):
        code = _read(BRIDGE)
        assert "self._live.get_live_xg_chart(" in code

    def test_slots_have_correct_signature(self):
        code = _read(BRIDGE)
        assert "@Slot(str, result=str)" in code


class TestFrontendKpiDashboard:
    """Deliverable 2: KPI cards in index.html."""

    def test_kpi_dashboard_section_exists(self):
        html = _read(INDEX)
        assert "live-kpi-dashboard" in html

    def test_kpi_possession_element(self):
        html = _read(INDEX)
        assert "live-possession" in html

    def test_kpi_shots_element(self):
        html = _read(INDEX)
        assert "live-shots" in html

    def test_kpi_xg_element(self):
        html = _read(INDEX)
        assert "live-xg" in html

    def test_kpi_xg_diff_element(self):
        html = _read(INDEX)
        assert "live-xg-diff" in html

    def test_kpi_goals_element(self):
        html = _read(INDEX)
        assert "live-goals" in html

    def test_dashboard_toggle_button(self):
        html = _read(INDEX)
        assert "live-dashboard-toggle" in html
        assert "Dashboard" in html

    def test_kpi_section_starts_hidden(self):
        html = _read(INDEX)
        # The kpi-grid should have class hidden initially
        assert 'class="kpi-grid hidden"' in html

    def test_all_kpi_cards_have_kpi_card_class(self):
        html = _read(INDEX)
        cards = re.findall(r'<div class="kpi-card"', html)
        # We have 5 KPI cards
        assert len(cards) >= 5


class TestFrontendPitchMap:
    """Deliverable 3: SVG pitch map."""

    def test_pitch_svg_exists(self):
        html = _read(INDEX)
        assert "live-pitch-svg" in html

    def test_pitch_events_overlay_exists(self):
        html = _read(INDEX)
        assert "live-pitch-events" in html

    def test_pitch_map_container_exists(self):
        html = _read(INDEX)
        assert "live-pitch-map-container" in html

    def test_svg_has_pitch_dimensions(self):
        html = _read(INDEX)
        assert 'viewBox="0 0 1050 680"' in html

    def test_svg_has_penalty_areas(self):
        html = _read(INDEX)
        assert 'stroke="#fff"' in html
        assert 'fill="#2d5a27"' in html

    def test_pitch_map_panel_exists(self):
        html = _read(INDEX)
        assert "Pitch Map" in html


class TestFrontendXgChart:
    """Deliverable 4: xG Chart.js canvas."""

    def test_xg_chart_canvas_exists(self):
        html = _read(INDEX)
        assert "live-xg-chart-canvas" in html

    def test_xg_chart_panel_exists(self):
        html = _read(INDEX)
        assert "xG Timeline" in html

    def test_chart_export_bar_exists(self):
        html = _read(INDEX)
        assert "chart-export-bar" in html


class TestLiveDashboardJS:
    """Verify JavaScript functions exist."""

    def test_init_live_dashboard_exists(self):
        code = _read(WEB_JS / "app-ai.js")
        assert "function initLiveDashboard" in code

    def test_update_live_dashboard_exists(self):
        code = _read(WEB_JS / "app-ai.js")
        assert "function updateLiveDashboard" in code

    def test_render_live_pitch_map_exists(self):
        code = _read(WEB_JS / "app-ai.js")
        assert "function renderLivePitchMap" in code

    def test_render_live_xg_chart_exists(self):
        code = _read(WEB_JS / "app-ai.js")
        assert "function renderLiveXgChart" in code

    def test_get_live_kpis_called(self):
        code = _read(WEB_JS / "app-ai.js")
        assert "get_live_kpis" in code

    def test_get_live_pitch_map_called(self):
        code = _read(WEB_JS / "app-ai.js")
        assert "get_live_pitch_map" in code

    def test_get_live_xg_chart_called(self):
        code = _read(WEB_JS / "app-ai.js")
        assert "get_live_xg_chart" in code

    def test_dashboard_toggle_in_vars(self):
        code = _read(WEB_JS / "app-ai.js")
        assert "live-dashboard-toggle" in code or "liveDashboardToggle" in code

    def test_live_xg_chart_instance_var(self):
        code = _read(WEB_JS / "app-ai.js")
        assert "_liveXgChartInstance" in code

    def test_pitch_map_home_away_events(self):
        code = _read(WEB_JS / "app-ai.js")
        assert "home_events" in code
        assert "away_events" in code

    def test_chart_destroy_on_stop(self):
        code = _read(WEB_JS / "app-ai.js")
        assert "_liveXgChartInstance.destroy" in code

    def test_init_live_dashboard_called_on_start(self):
        code = _read(WEB_JS / "app-ai.js")
        assert "initLiveDashboard" in code
        # Called when session starts
        assert "initLiveDashboard()" in code


class TestCSS:
    """Verify CSS styles exist."""

    def test_live_dashboard_panels_style(self):
        css = _read(WEB_CSS / "main.css")
        assert ".live-dashboard-panels" in css

    def test_pitch_map_container_style(self):
        css = _read(WEB_CSS / "main.css")
        assert ".pitch-map-container" in css

    def test_pitch_svg_style(self):
        css = _read(WEB_CSS / "main.css")
        assert ".pitch-svg" in css

    def test_pitch_event_marker_style(self):
        css = _read(WEB_CSS / "main.css")
        assert ".pitch-event-marker" in css

    def test_dashboard_panels_responsive(self):
        css = _read(WEB_CSS / "main.css")
        assert ".live-dashboard-panels" in css
        assert "grid-template-columns" in css


class TestLiveServiceIntegration:
    """Verify the live tagging service is callable from the new methods."""

    def test_service_import(self):
        code = _read(SERVICE)
        assert "class LiveTaggingService" in code
        assert "get_all_tags" in code

    def test_service_has_home_team_attr(self):
        code = _read(SERVICE)
        assert "home_team" in code

    def test_dashboard_panels_hidden_initially(self):
        html = _read(INDEX)
        assert 'id="live-dashboard-panels" class="live-dashboard-panels hidden"' in html


class TestKpiDashboardLiveUpdate:
    """Verify live update integration."""

    def test_update_live_dashboard_updates_kpis(self):
        code = _read(WEB_JS / "app-ai.js")
        kpi_ids = ["live-possession", "live-shots", "live-xg", "live-xg-diff", "live-goals"]
        for kid in kpi_ids:
            assert kid in code, f"KPI id '{kid}' not referenced in JS"

    def test_update_live_dashboard_updates_pitch_map(self):
        code = _read(WEB_JS / "app-ai.js")
        assert "renderLivePitchMap" in code

    def test_update_live_dashboard_updates_xg_chart(self):
        code = _read(WEB_JS / "app-ai.js")
        assert "renderLiveXgChart" in code

    def test_pitch_map_draws_circles(self):
        code = _read(WEB_JS / "app-ai.js")
        assert "createElementNS" in code
        assert "http://www.w3.org/2000/svg" in code

    def test_xg_chart_creates_chart_js(self):
        code = _read(WEB_JS / "app-ai.js")
        assert "new Chart(ctx" in code

    def test_pitch_map_has_goal_marker(self):
        code = _read(WEB_JS / "app-ai.js")
        assert "ev.type === 'goal'" in code or "goal" in code

    def test_pitch_map_has_shot_marker(self):
        code = _read(WEB_JS / "app-ai.js")
        assert "ev.type === 'shot'" in code or "shot" in code

    def test_pitch_map_has_pass_marker(self):
        code = _read(WEB_JS / "app-ai.js")
        assert "ev.type === 'pass'" in code or "pass" in code

    def test_pitch_map_markers_have_titles(self):
        code = _read(WEB_JS / "app-ai.js")
        assert "title" in code
        assert "formatLiveTime" in code
