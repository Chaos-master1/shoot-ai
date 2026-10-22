"""Opposition intelligence bridge handler (Phase D).

Surfaces the previously-dead dossier builder over real stored data,
the set-play library, and the vendor-import honest state. The
kloppy-backed vendor import follows the Transfermarkt precedent: no
free public API + ToS-restricted scraping = an explicit no-provider
state, never fabricated opponent data.
"""

from __future__ import annotations

import json

from kawkab.core.logging import get_logger
from kawkab.core.security import ErrorSanitizer
from kawkab.services.opposition_service import OppositionService
from kawkab.ui.bridge_handlers.base import BridgeHandlerBase

logger = get_logger(__name__)


class OppositionHandler(BridgeHandlerBase):
    """Slots for opposition intelligence (Phase D)."""

    async def get_opponent_dossier(self, opponent_team: str):
        """Build an opposition dossier from stored matches/events.

        Returns analysis over what the club actually has on record —
        with explicit unknowns where the data does not support a claim.
        """
        try:
            self._check_rate_limit("training")
            kb = await self._knowledge()
            report = await OppositionService(self._services["storage_service"], kb).build_dossier(
                str(opponent_team or "").strip()
            )
            return json.dumps(report, ensure_ascii=False)
        except Exception as e:
            logger.error(f"get_opponent_dossier failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_vendor_import_status(self):
        """Honest state of external match-data vendors (kloppy et al.)."""
        try:
            self._check_rate_limit("training")
            report = await OppositionService(
                self._services["storage_service"]
            ).vendor_import_status()
            return json.dumps(report, ensure_ascii=False)
        except Exception as e:
            logger.error(f"get_vendor_import_status failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_set_play_library(self):
        """Surface the knowledge base's set-play library."""
        try:
            self._check_rate_limit("training")
            kb = await self._knowledge()
            report = await OppositionService(
                self._services["storage_service"], kb
            ).set_play_library()
            return json.dumps(report, ensure_ascii=False)
        except Exception as e:
            logger.error(f"get_set_play_library failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})
