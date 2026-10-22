"""Base class for wearable data parsers.

Every vendor parser (Catapult CSV, STATSports GPX/CSV/XML, Polar HR, FIT, TCX)
inherits from :class:`BaseWearableParser` and implements ``parse``. This keeps
the parser surface uniform so :class:`~kawkab.services.wearables.service.WearableImportService`
can treat them interchangeably and so future parsers (WIMU, Kinexon, ...) slot
in without changing any caller.

Design rules:
- Parsers never raise on a malformed row; they skip + log at WARNING. A parser
  only raises on truly fatal conditions (file not found, wrong format entirely).
- Parsers return a populated :class:`~kawkab.services.wearables.models.WearableSession`
  with ``finalize()`` already called.
- Parsers never touch storage or the network — pure file → in-memory object.
"""

from __future__ import annotations

import abc
from pathlib import Path

from kawkab.core.logging import get_logger
from kawkab.services.wearables.models import WearableSession

logger = get_logger(__name__)


class BaseWearableParser(abc.ABC):
    """Abstract base for vendor-specific wearable parsers.

    Subclasses set :attr:`device_type` and implement :meth:`parse`.
    """

    #: Vendor key written into ``WearableSession.device_type``.
    device_type: str = "unknown"

    #: File extensions (lowercase, with dot) this parser handles.
    supported_extensions: tuple[str, ...] = ()

    @abc.abstractmethod
    def parse(self, file_path: str) -> WearableSession:
        """Parse the file at ``file_path`` into a WearableSession.

        Must call ``session.finalize()`` before returning.
        Must not raise on a single bad row (skip + log).
        """
        raise NotImplementedError

    # -- shared helpers ----------------------------------------------------

    @staticmethod
    def _check_file(file_path: str) -> Path:
        """Resolve and verify the file exists. Raises FileNotFoundError if not."""
        p = Path(file_path)
        if not p.is_file():
            raise FileNotFoundError(f"Wearable file not found: {file_path}")
        return p

    @staticmethod
    def _parse_float(val) -> float | None:
        """Robust float parse — returns None for empty/garbage values."""
        if val is None or val == "":
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_int(val) -> int | None:
        if val is None or val == "":
            return None
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return None

    @classmethod
    def supports(cls, file_path: str) -> bool:
        """True if this parser claims the file's extension."""
        from os.path import splitext

        ext = splitext(file_path)[1].lower()
        return ext in cls.supported_extensions


__all__ = ["BaseWearableParser"]
