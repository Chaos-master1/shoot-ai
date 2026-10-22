"""Handler for export bridge methods (exportCSV, exportJSON, exportPDF, extractVideoClips)."""

from __future__ import annotations

import json
from pathlib import Path

from kawkab.core.logging import get_logger
from kawkab.core.paths import get_paths
from kawkab.core.security import ErrorSanitizer, SecurityValidator
from kawkab.ui.bridge_handlers.base import BridgeHandlerBase

logger = get_logger(__name__)


class ExportHandler(BridgeHandlerBase):
    """Handles data export operations for Bridge."""

    @property
    def data_export_service(self):
        return self._services.get("data_export_service")

    @property
    def clip_service(self):
        return self._services.get("clip_service")

    @property
    def storage_service(self):
        return self._services.get("storage_service")

    def _check_rate_limit(self, category: str = "export") -> None:
        if self._rate_limiter is not None and not self._rate_limiter.acquire(category):
            raise RuntimeError(f"Rate limit exceeded for {category}")

    async def export_match_csv(self, match_id_str):
        try:
            self._check_rate_limit("export")
            match_id = SecurityValidator.validate_match_id(match_id_str)
            if self.data_export_service is None:
                return json.dumps({"error": "DataExportService not available"})
            path = await self.data_export_service.export_match_csv(match_id)
            return json.dumps({"success": True, "path": str(path)})
        except Exception as e:
            logger.error(f"Export CSV failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def export_match_json(self, match_id_str):
        try:
            self._check_rate_limit("export")
            match_id = SecurityValidator.validate_match_id(match_id_str)
            if self.data_export_service is None:
                return json.dumps({"error": "DataExportService not available"})
            path = await self.data_export_service.export_match_json(match_id)
            return json.dumps({"success": True, "path": str(path)})
        except Exception as e:
            logger.error(f"Export JSON failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def export_report_pdf(self, match_id, language):
        self._check_rate_limit("export")
        import html as html_mod
        from datetime import datetime

        try:
            match_id = SecurityValidator.validate_match_id(match_id)
            language = SecurityValidator.sanitize_string(language, max_length=10)
            if language not in ("en", "ar"):
                language = "en"

            match = await self.storage_service.get_match(match_id)
            if not match:
                return json.dumps({"error": "Match not found"})

            events = await self.storage_service.get_match_events(match_id)
            report_text = ""
            try:
                stored_reports = await self.storage_service.get_reports(match_id, language)
                if stored_reports and len(stored_reports) > 0:
                    report_text = stored_reports[0].get("report_text", "")
            except Exception:
                report_text = ""

            shot_events = [e for e in events if e.get("type") == "shot"]
            pass_events = [e for e in events if e.get("type") == "pass"]

            home_shots = sum(1 for e in shot_events if e.get("team") == "home")
            away_shots = sum(1 for e in shot_events if e.get("team") == "away")
            home_passes = sum(1 for e in pass_events if e.get("team") == "home")
            away_passes = sum(1 for e in pass_events if e.get("team") == "away")
            home_on_target = sum(
                1 for e in shot_events if e.get("team") == "home" and e.get("on_target")
            )
            away_on_target = sum(
                1 for e in shot_events if e.get("team") == "away" and e.get("on_target")
            )

            match_name = html_mod.escape(match.get("name", "Unnamed Match"))
            match_date = match.get("match_date", datetime.now().strftime("%Y-%m-%d"))

            is_rtl = language == "ar"
            doc_dir = "rtl" if is_rtl else "ltr"
            title = "تقرير المباراة" if is_rtl else "Match Report"

            html_content = f"""<!DOCTYPE html>
<html lang="{language}" dir="{doc_dir}">
<head><meta charset="UTF-8"><title>{title} - {match_name}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #1e293b; background: #fff; padding: 2rem; max-width: 900px; margin: 0 auto; line-height: 1.6; }}
h1 {{ font-size: 1.75rem; color: #2563eb; margin-bottom: 0.25rem; }}
h2 {{ font-size: 1.25rem; color: #334155; margin: 1.5rem 0 0.75rem; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.25rem; }}
h3 {{ font-size: 1rem; color: #475569; margin: 1rem 0 0.5rem; }}
.meta {{ color: #64748b; font-size: 0.875rem; margin-bottom: 1.5rem; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; }}
.card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; }}
.card h3 {{ color: #2563eb; margin-top: 0; }}
.stat-row {{ display: flex; justify-content: space-between; padding: 0.35rem 0; border-bottom: 1px solid #f1f5f9; font-size: 0.9rem; }}
.stat-label {{ color: #64748b; }}
.stat-value {{ font-weight: 600; }}
.report {{ white-space: pre-wrap; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; font-size: 0.9rem; line-height: 1.7; }}
.footer {{ margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e2e8f0; font-size: 0.75rem; color: #94a3b8; text-align: center; }}
</style></head>
<body>
<h1>{title}</h1>
<p class="meta">{html_mod.escape(match_name)} &middot; {match_date}</p>

<div class="grid">
  <div class="card">
    <h3>{"الفريق المضيف" if is_rtl else "Home Team"}</h3>
    <div class="stat-row"><span class="stat-label">{"التسديدات" if is_rtl else "Shots"}</span><span class="stat-value">{home_shots}</span></div>
    <div class="stat-row"><span class="stat-label">{"على المرمى" if is_rtl else "On Target"}</span><span class="stat-value">{home_on_target}</span></div>
    <div class="stat-row"><span class="stat-label">{"التمريرات" if is_rtl else "Passes"}</span><span class="stat-value">{home_passes}</span></div>
  </div>
  <div class="card">
    <h3>{"الفريق الضيف" if is_rtl else "Away Team"}</h3>
    <div class="stat-row"><span class="stat-label">{"التسديدات" if is_rtl else "Shots"}</span><span class="stat-value">{away_shots}</span></div>
    <div class="stat-row"><span class="stat-label">{"على المرمى" if is_rtl else "On Target"}</span><span class="stat-value">{away_on_target}</span></div>
    <div class="stat-row"><span class="stat-label">{"التمريرات" if is_rtl else "Passes"}</span><span class="stat-value">{away_passes}</span></div>
  </div>
</div>

<h2>{"تقرير المدرب" if is_rtl else "Coach Report"}</h2>
<div class="report">{html_mod.escape(report_text) if report_text else ("لم يتم إنشاء تقرير بعد" if is_rtl else "No report generated yet")}</div>

<div class="footer">{"تم الإنشاء بواسطة" if is_rtl else "Generated by"} Kawkab AI &middot; {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
</body></html>"""

            output_dir = get_paths().exports
            output_path = output_dir / f"report_{match_id}_{language}.html"
            output_path.write_text(html_content, encoding="utf-8")
            return json.dumps({"success": True, "path": str(output_path)})
        except Exception as e:
            logger.error(f"export_report_pdf failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def export_match_statsbomb(self, match_id_str, file_path):
        try:
            self._check_rate_limit("export")
            match_id = SecurityValidator.validate_match_id(match_id_str)
            if self.data_export_service is None:
                return json.dumps({"error": "DataExportService not available"})
            path = await self.data_export_service.export_statsbomb_compatible(match_id)
            output = Path(file_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            return json.dumps({"success": True, "path": str(output)})
        except Exception as e:
            logger.error(f"Export StatsBomb failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def extract_event_clips(self, match_id):
        try:
            self._check_rate_limit("export")
            match_id = SecurityValidator.validate_match_id(match_id)
            if self.clip_service is None:
                return json.dumps({"error": "ClipExtractionService not available"})
            match = await self.storage_service.get_match(match_id)
            if not match or not match.get("video_path"):
                return json.dumps({"error": "No video found for match"})
            events = await self.storage_service.get_match_events(match_id)
            shot_events = [e for e in events if e.get("type") == "shot"]
            if not shot_events:
                return json.dumps({"error": "No shot events to extract"})
            clip_events = [
                {"timestamp": e["timestamp"], "type": "shot", "team": e.get("team", "unknown")}
                for e in shot_events
            ]
            clips = await self.clip_service.extract_event_clips(
                video_path=Path(match["video_path"]),
                events=clip_events,
                context_seconds=3.0,
            )
            return json.dumps({"success": True, "clips": clips})
        except Exception as e:
            logger.error(f"extract_event_clips failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})
