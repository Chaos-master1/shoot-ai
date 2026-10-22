"""Handler for collaboration, cloud sync/OAuth, AI assistant v2, marketplace,
telestration and live-stream capture bridge methods."""

from __future__ import annotations

import json

from kawkab.core.logging import get_logger
from kawkab.core.security import ErrorSanitizer, SecurityValidator
from kawkab.ui.bridge_handlers.base import BridgeHandlerBase

logger = get_logger(__name__)


class CloudCollabHandler(BridgeHandlerBase):
    """Collaboration + cloud + AI-v2 + marketplace + telestration/stream surface."""

    @property
    def storage_service(self):
        return self._services.get("storage_service")

    @property
    def roboflow_sports_service(self):
        return self._services.get("roboflow_sports_service")

    # ================================================================
    # Sprint 3 — Collaboration Service
    # ================================================================

    async def create_collab_user(self, username, display_name, role="analyst"):
        try:
            svc = self._services.get("collaboration_service")
            if svc is None:
                from kawkab.services.collaboration_service import CollaborationService

                svc = CollaborationService()
                self._services["collaboration_service"] = svc
            return svc.create_user(username, display_name, role)
        except Exception as e:
            logger.error(f"create_collab_user failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_collab_users(self):
        try:
            svc = self._services.get("collaboration_service")
            if svc is None:
                return json.dumps({"users": [], "total": 0})
            return svc.get_users()
        except Exception as e:
            logger.error(f"get_collab_users failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def delete_collab_user(self, user_id):
        try:
            svc = self._services.get("collaboration_service")
            if svc is None:
                return json.dumps({"error": "Service not initialized"})
            return svc.delete_user(user_id)
        except Exception as e:
            logger.error(f"delete_collab_user failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def add_comment(self, match_id, event_id, user_id, text):
        try:
            svc = self._services.get("collaboration_service")
            if svc is None:
                from kawkab.services.collaboration_service import CollaborationService

                svc = CollaborationService()
                self._services["collaboration_service"] = svc
            return svc.add_comment(match_id, event_id, user_id, text)
        except Exception as e:
            logger.error(f"add_comment failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_comments(self, match_id, event_id=0):
        try:
            svc = self._services.get("collaboration_service")
            if svc is None:
                return json.dumps({"comments": [], "total": 0})
            return svc.get_comments(match_id, event_id)
        except Exception as e:
            logger.error(f"get_comments failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def delete_comment(self, comment_id):
        try:
            svc = self._services.get("collaboration_service")
            if svc is None:
                return json.dumps({"error": "Service not initialized"})
            return svc.delete_comment(comment_id)
        except Exception as e:
            logger.error(f"delete_comment failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def export_project(self, match_id):
        try:
            storage = self.storage_service
            if storage is None:
                return json.dumps({"error": "Storage not available"})
            match = await storage.get_match(match_id)
            if not match:
                return json.dumps({"error": "Match not found"})
            events = await storage.get_match_events(match_id)
            match["events"] = events
            svc = self._services.get("collaboration_service")
            if svc is None:
                from kawkab.services.collaboration_service import CollaborationService

                svc = CollaborationService()
                self._services["collaboration_service"] = svc
            return svc.export_project(match)
        except Exception as e:
            logger.error(f"export_project failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def import_project(self, project_json):
        try:
            svc = self._services.get("collaboration_service")
            if svc is None:
                from kawkab.services.collaboration_service import CollaborationService

                svc = CollaborationService()
                self._services["collaboration_service"] = svc
            return svc.import_project(project_json)
        except Exception as e:
            logger.error(f"import_project failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_activity_feed(self, limit=50):
        try:
            svc = self._services.get("collaboration_service")
            if svc is None:
                return json.dumps({"activities": [], "total": 0})
            return svc.get_activity_feed(limit)
        except Exception as e:
            logger.error(f"get_activity_feed failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_event_comments(self, match_id, event_id):
        try:
            svc = self._services.get("collaboration_service")
            if svc is None:
                return json.dumps({"comments": [], "total": 0})
            return svc.get_event_comments(match_id, event_id)
        except Exception as e:
            logger.error(f"get_event_comments failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_mentions(self, username):
        try:
            svc = self._services.get("collaboration_service")
            if svc is None:
                return json.dumps({"mentions": [], "total": 0, "unread": 0})
            return svc.get_mentions(username)
        except Exception as e:
            logger.error(f"get_mentions failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def mark_mention_read(self, mention_id):
        try:
            svc = self._services.get("collaboration_service")
            if svc is None:
                return json.dumps({"error": "Service not initialized"})
            return svc.mark_mention_read(mention_id)
        except Exception as e:
            logger.error(f"mark_mention_read failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def rf_draw_pitch(self, scale):
        import base64

        if self.roboflow_sports_service is None or not self.roboflow_sports_service.available:
            return json.dumps({"error": "roboflow/sports not installed"})
        try:
            import cv2

            img = self.roboflow_sports_service.draw_pitch(scale=scale)
            if img is None:
                return json.dumps({"error": "draw_pitch returned None"})
            _, buf = cv2.imencode(".png", img)
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            return json.dumps({"success": True, "image_b64": b64, "shape": list(img.shape)})
        except Exception as e:
            logger.error(f"rf_draw_pitch failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Phase 10 — Telestration v2
    # ================================================================

    async def tel_layer_add(self, layer_id, name=""):
        try:
            svc = self._get_telestration()
            return svc.add_layer(layer_id, name)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def tel_layer_remove(self, layer_id):
        try:
            svc = self._get_telestration()
            return svc.remove_layer(layer_id)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def tel_layer_toggle(self, layer_id):
        try:
            svc = self._get_telestration()
            return svc.toggle_layer_visibility(layer_id)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def tel_layer_opacity(self, layer_id, opacity):
        try:
            svc = self._get_telestration()
            return svc.set_layer_opacity(layer_id, opacity)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def tel_get_layers(self):
        try:
            svc = self._get_telestration()
            return svc.get_layers()
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def tel_save_preset(self, name, layers_json):
        try:
            svc = self._get_telestration()
            return svc.save_preset(name, layers_json)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def tel_load_preset(self, name):
        try:
            svc = self._get_telestration()
            return svc.load_preset(name)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def tel_list_presets(self):
        try:
            svc = self._get_telestration()
            return svc.list_presets()
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def tel_delete_preset(self, name):
        try:
            svc = self._get_telestration()
            return svc.delete_preset(name)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def tel_export_video(self, video_path, layers_json, output_path=""):
        try:
            svc = self._get_telestration()
            return svc.export_annotated_video(video_path, layers_json, output_path)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _get_telestration(self):
        svc = self._services.get("telestration_service")
        if svc is None:
            from kawkab.services.telestration_service import TelestrationService

            svc = TelestrationService()
            self._services["telestration_service"] = svc
        return svc

    # ================================================================
    # Phase 9 — Live Stream Capture
    # ================================================================

    async def stream_start_capture(self, url, stream_id="", output_filename=""):
        try:
            # ffmpeg -i <url>: block file:// URLs (arbitrary local file reads),
            # and validate any output_filename the caller supplies.
            u = str(url or "").strip()
            if not u or u.lower().startswith("file:"):
                return json.dumps({"error": "invalid or unsupported stream URL"})
            if output_filename:
                safe = SecurityValidator.sanitize_string(str(output_filename), max_length=200)
                if safe != str(output_filename):
                    return json.dumps({"error": "invalid output filename"})
                # sanitize_string's default allowlist keeps "/", "\\" and ".",
                # so traversal sequences survive it -- reject them explicitly:
                # the filename is joined onto the capture output dir.
                fn = str(output_filename)
                if "/" in fn or "\\" in fn or ".." in fn:
                    return json.dumps({"error": "invalid output filename"})
            svc = self._services.get("live_stream_service")
            if svc is None:
                from kawkab.services.live_stream_service import LiveStreamCaptureService

                svc = LiveStreamCaptureService()
                self._services["live_stream_service"] = svc
            return svc.start_capture(url, stream_id, output_filename)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def stream_stop_capture(self, stream_id):
        try:
            svc = self._services.get("live_stream_service")
            if svc is None:
                return json.dumps({"error": "No stream service"})
            return svc.stop_capture(stream_id)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def stream_get_status(self, stream_id):
        try:
            svc = self._services.get("live_stream_service")
            if svc is None:
                return json.dumps({"error": "No stream service"})
            return svc.get_stream_status(stream_id)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def stream_list(self):
        try:
            svc = self._services.get("live_stream_service")
            if svc is None:
                return json.dumps({"streams": []})
            return svc.list_streams()
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def stream_add_marker(self, stream_id, label=""):
        try:
            svc = self._services.get("live_stream_service")
            if svc is None:
                return json.dumps({"error": "No stream service"})
            return svc.add_chapter_marker(stream_id, label)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def stream_list_recordings(self):
        try:
            svc = self._services.get("live_stream_service")
            if svc is None:
                return json.dumps({"recordings": []})
            return svc.list_recordings()
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def stream_detect_source(self, url):
        try:
            from kawkab.services.live_stream_service import LiveStreamCaptureService

            svc = LiveStreamCaptureService()
            return json.dumps({"source_type": svc.detect_source_type(url)})
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ================================================================
    # Phase 8 — Cloud Sync
    # ================================================================

    async def cloud_check_health(self):
        try:
            svc = self._get_cloud_sync()
            return svc.check_health()
        except Exception as e:
            return json.dumps({"error": str(e), "status": "offline"})

    async def cloud_register(self, username, email, password, display_name=""):
        try:
            svc = self._get_cloud_sync()
            return svc.register(username, email, password, display_name)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def cloud_login(self, email, password):
        try:
            svc = self._get_cloud_sync()
            return svc.login(email, password)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def cloud_logout(self):
        try:
            svc = self._get_cloud_sync()
            return svc.logout()
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def cloud_get_me(self):
        try:
            svc = self._get_cloud_sync()
            return svc.get_me()
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def cloud_is_logged_in(self):
        try:
            svc = self._get_cloud_sync()
            return svc.is_logged_in()
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def cloud_create_team(self, name, description=""):
        try:
            svc = self._get_cloud_sync()
            return svc.create_team(name, description)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def cloud_list_teams(self):
        try:
            svc = self._get_cloud_sync()
            return svc.list_teams()
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def cloud_invite_member(self, team_id, email):
        try:
            svc = self._get_cloud_sync()
            return svc.invite_member(team_id, email)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def cloud_accept_invite(self, token):
        try:
            svc = self._get_cloud_sync()
            return svc.accept_invite(token)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def cloud_sync_push(self, device_id, operations_json):
        try:
            svc = self._get_cloud_sync()
            ops = json.loads(operations_json)
            return svc.sync_push(device_id, ops)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def cloud_sync_pull(self, device_id):
        try:
            svc = self._get_cloud_sync()
            return svc.sync_pull(device_id)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def cloud_oauth_authorize_url(self, provider, redirect_uri=""):
        try:
            svc = self._get_cloud_sync()
            return svc.oauth_authorize_url(provider, redirect_uri)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def cloud_oauth_exchange(self, provider, code, state):
        try:
            svc = self._get_cloud_sync()
            return svc.oauth_exchange(provider, code, state)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def cloud_oauth_providers(self):
        try:
            svc = self._get_cloud_sync()
            return svc.oauth_providers()
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def cloud_start_server(self, port=8741):
        try:
            import threading

            from kawkab.cloud.server import start

            t = threading.Thread(target=start, args=("0.0.0.0", port), daemon=True)
            t.start()
            return json.dumps(
                {"ok": True, "port": port, "message": f"Cloud server started on port {port}"}
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def cloud_server_status(self):
        try:
            import httpx

            resp = httpx.get("http://localhost:8741/health", timeout=3.0)
            return json.dumps({"running": resp.status_code == 200, "details": resp.json()})
        except Exception:
            return json.dumps({"running": False})

    def _get_cloud_sync(self):
        svc = self._services.get("cloud_sync_service")
        if svc is None:
            from kawkab.services.cloud_sync_service import CloudSyncService

            svc = CloudSyncService()
            self._services["cloud_sync_service"] = svc
        return svc

    # ================================================================
    # Phase 12 — AI Coach Assistant v2
    # ================================================================

    def _get_ai_v2(self):
        svc = self._services.get("ai_assistant_v2_service")
        if svc is None:
            from kawkab.services.ai_assistant_v2_service import AIAssistantV2Service

            svc = AIAssistantV2Service(llm_service=self._services.get("llm_service"))
            self._services["ai_assistant_v2_service"] = svc
        return svc

    async def ai_v2_create_conv(self, match_id, title):
        try:
            svc = self._get_ai_v2()
            conv = svc.create_conversation(
                match_id=int(match_id) if match_id else None,
                title=str(title or "New Chat"),
            )
            return json.dumps({"success": True, "conv_id": conv.id, "title": conv.title})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def ai_v2_list_convs(self, match_id):
        try:
            svc = self._get_ai_v2()
            convs = svc.list_conversations(match_id=int(match_id) if match_id else None)
            return json.dumps({"success": True, "conversations": convs})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def ai_v2_delete_conv(self, conv_id):
        try:
            svc = self._get_ai_v2()
            ok = svc.delete_conversation(str(conv_id))
            return json.dumps({"success": ok})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def ai_v2_ask(self, conv_id, question, match_context, language):
        try:
            svc = self._get_ai_v2()
            answer = await svc.ask(
                conv_id=str(conv_id),
                question=str(question),
                match_context=str(match_context or ""),
                language=str(language or "en"),
            )
            return json.dumps({"success": True, "answer": answer})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def ai_v2_tactical_suggestion(self, topic, match_context, language):
        try:
            svc = self._get_ai_v2()
            answer = await svc.get_tactical_suggestion(
                topic=str(topic),
                match_context=str(match_context or ""),
                language=str(language or "en"),
            )
            return json.dumps({"success": True, "answer": answer})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def ai_v2_auto_report(self, match_id, language):
        try:
            svc = self._get_ai_v2()
            mid = int(match_id)
            events = await self.storage_service.get_match_events(mid)
            match_data = await self.storage_service.get_match(mid) or {}
            match_data["event_count"] = len(events)

            event_summary = {}
            for ev in events:
                et = ev.get("event_type", "unknown")
                event_summary[et] = event_summary.get(et, 0) + 1
            match_data["event_breakdown"] = event_summary

            report = await svc.generate_automated_report(
                match_id=mid,
                match_data=match_data,
                language=str(language or "en"),
            )
            return json.dumps({"success": True, "report": report})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Phase 15 — Community Marketplace
    # ================================================================

    def _get_marketplace(self):
        svc = self._services.get("marketplace_service")
        if svc is None:
            from kawkab.services.marketplace_service import MarketplaceService

            svc = MarketplaceService()
            self._services["marketplace_service"] = svc
        return svc

    async def marketplace_list(self, item_type, category, query, source):
        try:
            svc = self._get_marketplace()
            items = svc.list_items(
                item_type=str(item_type or ""),
                category=str(category or ""),
                query=str(query or ""),
                source=str(source or ""),
            )
            return json.dumps({"success": True, "items": items})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def marketplace_get(self, item_id):
        try:
            svc = self._get_marketplace()
            item = svc.get_item(str(item_id))
            if item:
                return json.dumps({"success": True, "item": item})
            return json.dumps({"success": False, "error": "Not found"})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def marketplace_add(
        self, item_type, name, description, author, category, tags_json, data, source
    ):
        try:
            svc = self._get_marketplace()
            tags = json.loads(tags_json) if tags_json else []
            result = svc.add_item(
                item_type=str(item_type),
                name=str(name),
                description=str(description or ""),
                author=str(author or ""),
                category=str(category or ""),
                tags=tags,
                data=str(data or ""),
                source=str(source or "local"),
            )
            return json.dumps({"success": True, "item": result})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def marketplace_rate(self, item_id, rating):
        try:
            svc = self._get_marketplace()
            ok = svc.rate_item(str(item_id), float(rating))
            return json.dumps({"success": ok})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def marketplace_delete(self, item_id):
        try:
            svc = self._get_marketplace()
            ok = svc.delete_item(str(item_id))
            return json.dumps({"success": ok})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def marketplace_stats(self):
        try:
            svc = self._get_marketplace()
            stats = svc.get_stats()
            return json.dumps({"success": True, "stats": stats})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def marketplace_categories(self, item_type):
        try:
            svc = self._get_marketplace()
            cats = svc.get_categories(str(item_type or ""))
            return json.dumps({"success": True, "categories": cats})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})
