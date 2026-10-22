"""Handler for the tactical whiteboard bridge methods — states, annotations,
formations, templates and SVG export."""

from __future__ import annotations

import json

from kawkab.core.logging import get_logger
from kawkab.core.security import ErrorSanitizer
from kawkab.ui.bridge_handlers.base import BridgeHandlerBase

logger = get_logger(__name__)


class WhiteboardHandler(BridgeHandlerBase):
    """Tactical whiteboard bridge surface."""

    @property
    def storage_service(self):
        return self._services.get("storage_service")

    @property
    def tactical_whiteboard(self):
        if not hasattr(self, "_tactical_wb"):
            from kawkab.analysis.tactical_whiteboard import TacticalWhiteboard

            self._tactical_wb = TacticalWhiteboard()
        return self._tactical_wb

    # ================================================================
    # Tactical Whiteboard
    # ================================================================

    async def check_whiteboard_status(self):
        wb = self.tactical_whiteboard
        return json.dumps({"available": wb.available})

    async def whiteboard_create(self, name, formation_home):
        try:
            wb = self.tactical_whiteboard
            state = wb.create_state(name=name or "", formation_home=formation_home or "")
            return json.dumps(state.to_dict())
        except Exception as e:
            logger.error(f"whiteboard_create failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def whiteboard_get(self, state_id):
        try:
            wb = self.tactical_whiteboard
            state = wb.get_state(state_id)
            if state is None:
                return json.dumps({"error": "State not found"})
            return json.dumps(state.to_dict())
        except Exception as e:
            logger.error(f"whiteboard_get failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def whiteboard_list(self):
        try:
            wb = self.tactical_whiteboard
            return json.dumps(wb.list_states())
        except Exception as e:
            logger.error(f"whiteboard_list failed: {e}")
            return json.dumps([])

    async def whiteboard_delete(self, state_id):
        try:
            wb = self.tactical_whiteboard
            ok = wb.delete_state(state_id)
            return json.dumps({"ok": ok})
        except Exception as e:
            logger.error(f"whiteboard_delete failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def whiteboard_update(self, state_id, data_json):
        try:
            wb = self.tactical_whiteboard
            data = json.loads(data_json) if data_json else {}
            state = wb.update_state(state_id, data)
            if state is None:
                return json.dumps({"error": "State not found"})
            return json.dumps(state.to_dict())
        except Exception as e:
            logger.error(f"whiteboard_update failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def whiteboard_add_annotation(self, state_id, annotation_json):
        try:
            wb = self.tactical_whiteboard
            ann = json.loads(annotation_json) if annotation_json else {}
            result = wb.add_annotation(state_id, ann)
            if result is None:
                return json.dumps({"error": "State not found"})
            return json.dumps(result.to_dict())
        except Exception as e:
            logger.error(f"whiteboard_add_annotation failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def whiteboard_remove_annotation(self, state_id, annotation_id):
        try:
            wb = self.tactical_whiteboard
            ok = wb.remove_annotation(state_id, annotation_id)
            return json.dumps({"ok": ok})
        except Exception as e:
            logger.error(f"whiteboard_remove_annotation failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def whiteboard_clear_annotations(self, state_id):
        try:
            wb = self.tactical_whiteboard
            ok = wb.clear_annotations(state_id)
            return json.dumps({"ok": ok})
        except Exception as e:
            logger.error(f"whiteboard_clear_annotations failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def whiteboard_set_formation(self, state_id, formation_name, team):
        try:
            wb = self.tactical_whiteboard
            players = wb.set_players_from_formation(state_id, formation_name, team)
            if players is None:
                return json.dumps({"error": "State or formation not found"})
            return json.dumps({"players": players})
        except Exception as e:
            logger.error(f"whiteboard_set_formation failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def whiteboard_move_player(self, state_id, player_index, x, y, team):
        try:
            wb = self.tactical_whiteboard
            ok = wb.move_player(state_id, int(player_index), float(x), float(y), team or "home")
            return json.dumps({"ok": ok})
        except Exception as e:
            logger.error(f"whiteboard_move_player failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def whiteboard_list_templates(self):
        try:
            wb = self.tactical_whiteboard
            return json.dumps(wb.list_templates())
        except Exception as e:
            logger.error(f"whiteboard_list_templates failed: {e}")
            return json.dumps([])

    async def whiteboard_get_template(self, name):
        try:
            wb = self.tactical_whiteboard
            positions = wb.get_template(name)
            if positions is None:
                return json.dumps({"error": "Template not found"})
            return json.dumps({"positions": positions})
        except Exception as e:
            logger.error(f"whiteboard_get_template failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def whiteboard_generate_svg(self, state_id, width, height):
        try:
            wb = self.tactical_whiteboard
            svg = wb.generate_svg(state_id, width=int(width or 600), height=int(height or 400))
            if svg is None:
                return json.dumps({"error": "State not found"})
            return json.dumps({"svg": svg})
        except Exception as e:
            logger.error(f"whiteboard_generate_svg failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def whiteboard_generate_player_run(self, start_x, start_y, end_x, end_y, color, label):
        try:
            wb = self.tactical_whiteboard
            ann = wb.generate_player_run(
                float(start_x),
                float(start_y),
                float(end_x),
                float(end_y),
                color=color or "#f39c12",
                label=label or "",
            )
            return json.dumps(ann.to_dict())
        except Exception as e:
            logger.error(f"whiteboard_generate_player_run failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def whiteboard_generate_pass(self, start_x, start_y, end_x, end_y, color):
        try:
            wb = self.tactical_whiteboard
            ann = wb.generate_pass(
                float(start_x),
                float(start_y),
                float(end_x),
                float(end_y),
                color=color or "#2ecc71",
            )
            return json.dumps(ann.to_dict())
        except Exception as e:
            logger.error(f"whiteboard_generate_pass failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})
