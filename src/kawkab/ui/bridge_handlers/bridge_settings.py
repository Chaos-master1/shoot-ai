"""Handler for the Settings bridge methods — app info, GPU tier, YOLO
variant control, model cache manager, and the one-call settings overview."""

from __future__ import annotations

import json

from kawkab.core.logging import get_logger
from kawkab.core.security import ErrorSanitizer
from kawkab.ui.bridge_handlers.base import BridgeHandlerBase

logger = get_logger(__name__)


class SettingsHandler(BridgeHandlerBase):
    """Settings workspace bridge surface (see app-settings.js)."""

    # ================================================================
    # Settings — model cache manager + app overview
    # ================================================================

    def _get_model_manager(self):
        try:
            from kawkab.core.model_manager import ModelManager

            return self._services.get("model_manager") or ModelManager()
        except Exception:
            return None

    async def get_model_cache_info(self):
        """Cache size, per-model availability, and download candidates.

        Powers the Settings model-cache manager UI: which models are on disk,
        how much space they use, and which variants can be fetched.
        """
        try:
            mm = self._get_model_manager()
            if mm is None:
                return json.dumps({"error": "ModelManager unavailable"})
            cached = {}
            for name in mm.list_cached_models():
                path = mm.get_model_path(name)
                cached[name] = {
                    "size_mb": round(path.stat().st_size / (1024 * 1024), 1) if path else 0,
                }
            variants = []
            for name, info in mm.DEFAULT_MODELS.items():
                variants.append(
                    {
                        "name": name,
                        "size_mb": info.get("size_mb", 0),
                        "cached": name in cached,
                        "checksum_pinned": info.get("sha256") is not None,
                    }
                )
            return json.dumps(
                {
                    "cache_size_mb": round(mm.get_cache_size_mb(), 1),
                    "models": variants,
                    "current_variant": (
                        self._services.get("cv_service").model_size
                        if self._services.get("cv_service")
                        and hasattr(self._services.get("cv_service"), "model_size")
                        else None
                    ),
                }
            )
        except Exception as e:
            logger.error(f"get_model_cache_info failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def download_model_slot(self, model_name):
        """Download a model into the local cache (blocking; called from UI)."""
        try:
            mm = self._get_model_manager()
            if mm is None:
                return json.dumps({"success": False, "error": "ModelManager unavailable"})
            path = mm.download_model(str(model_name))
            return json.dumps({"success": True, "path": str(path), "name": str(model_name)})
        except ValueError as e:
            return json.dumps({"success": False, "error": str(e)})
        except Exception as e:
            logger.error(f"download_model failed: {e}")
            return json.dumps({"success": False, "error": ErrorSanitizer.sanitize_error(e)})

    async def delete_cached_model(self, model_name):
        """Remove one model from the cache (keep-lists nothing)."""
        try:
            mm = self._get_model_manager()
            if mm is None:
                return json.dumps({"success": False, "error": "ModelManager unavailable"})
            removed = mm.cleanup_cache(keep_models=[str(model_name)])
            return json.dumps({"success": True, "removed": removed})
        except Exception as e:
            logger.error(f"delete_cached_model failed: {e}")
            return json.dumps({"success": False, "error": ErrorSanitizer.sanitize_error(e)})

    async def get_settings_overview(self):
        """One-call Settings payload: app info, GPU, model variant, cache size."""
        try:
            info = json.loads(await self.get_app_info())
            gpu = json.loads(await self.get_gpu_tier())
            current = json.loads(await self.get_current_yolo_variant())
            recommended = json.loads(await self.get_recommended_yolo_variant())
            cache = json.loads(await self.get_model_cache_info())
            return json.dumps(
                {
                    "app": {k: info.get(k) for k in ("name", "version", "platform", "python")},
                    "gpu": {k: gpu.get(k) for k in ("backend", "tier") if k in gpu},
                    "model": {
                        "current": current.get("variant"),
                        "recommended": recommended.get("recommended"),
                        "tier": recommended.get("tier"),
                    },
                    "cache_size_mb": cache.get("cache_size_mb", 0),
                    "checksum_pinned": any(
                        m.get("checksum_pinned") for m in cache.get("models", [])
                    ),
                }
            )
        except Exception as e:
            logger.error(f"get_settings_overview failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def generate_scout_report_pdf(self, track_id, match_id=0):
        try:
            self._check_rate_limit()
            from kawkab.core.scout_reports import generate_scout_report

            report = generate_scout_report(track_id, match_id)
            return json.dumps(
                {"report": report.to_dict() if hasattr(report, "to_dict") else str(report)}
            )
        except Exception as e:
            logger.error(f"generate_scout_report failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_app_info(self):
        return json.dumps(
            {
                "version": "0.13.0",
                "name": "Kawkab AI",
                "platform": __import__("platform").platform(),
                "python": __import__("sys").version,
                "description": "Private offline AI football coach",
            }
        )

    # ── P0-B2: YOLO variant control ────────────────────────────────

    async def get_recommended_yolo_variant(self):
        """Return the recommended YOLO variant for the current GPU tier."""
        try:
            from kawkab.core.gpu_acceleration import detect_gpu_tier, recommend_yolo_variant

            tier = detect_gpu_tier()
            variant = recommend_yolo_variant(tier)
            return json.dumps({"success": True, "tier": tier, "recommended": variant})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_current_yolo_variant(self):
        """Return the current YOLO variant in use."""
        try:
            cv = self._services.get("cv_service")
            variant = cv.model_size if cv and hasattr(cv, "model_size") else "l"
            return json.dumps({"success": True, "variant": variant})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def set_yolo_variant(self, variant: str):
        """Set the YOLO variant for the next analysis."""
        try:
            valid = {"n", "s", "m", "l", "x"}
            if variant not in valid:
                return json.dumps(
                    {
                        "success": False,
                        "error": f"Invalid variant '{variant}'. Must be one of {valid}",
                    }
                )
            cv = self._services.get("cv_service")
            if cv and hasattr(cv, "model_size"):
                cv.model_size = variant
                logger.info(f"YOLO variant set to yolo11{variant}")
                return json.dumps({"success": True, "variant": variant})
            return json.dumps({"success": False, "error": "CV service not available"})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_gpu_tier(self):
        """Return detected GPU tier and VRAM info."""
        try:
            from kawkab.core.gpu_acceleration import detect_gpu, detect_gpu_tier

            backend = detect_gpu()
            tier = detect_gpu_tier()
            info = {"backend": backend, "tier": tier}
            if backend == "cuda":
                try:
                    import subprocess

                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=name,memory.total,driver_version",
                            "--format=csv,noheader",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        parts = result.stdout.strip().split(", ")
                        info["gpu_name"] = parts[0] if len(parts) > 0 else "unknown"
                        info["vram_mb"] = parts[1] if len(parts) > 1 else "unknown"
                        info["driver"] = parts[2] if len(parts) > 2 else "unknown"
                except Exception:
                    pass
            return json.dumps({"success": True, "info": info})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})
