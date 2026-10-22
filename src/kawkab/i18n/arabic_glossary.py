"""Arabic football terminology glossary loader.

Loads ``docs/translations/ar.yml`` and exposes lookup helpers that
the in-app translator uses to ensure domain terms are translated
consistently (rather than machine-translated ad-hoc).

YAML support is optional — when PyYAML is not available, a minimal
hand-rolled parser is used (the glossary file format is simple).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GlossaryEntry:
    """A single glossary term."""

    en: str
    ar: str
    transliteration: str = ""
    definition: str = ""


class ArabicGlossary:
    """Lookup Arabic translations for football terms.

    Args:
        glossary_path: Path to the YAML file.
    """

    DEFAULT_PATH = (
        Path(__file__).resolve().parent.parent.parent.parent / "docs" / "translations" / "ar.yml"
    )

    def __init__(self, glossary_path: Path | None = None) -> None:
        self.glossary_path = glossary_path or self.DEFAULT_PATH
        self._entries: dict[str, GlossaryEntry] = {}
        self._load()

    def _load(self) -> None:
        if not self.glossary_path.exists():
            logger.warning("Glossary file not found: %s", self.glossary_path)
            return
        text = self.glossary_path.read_text(encoding="utf-8")
        try:
            import yaml

            data = yaml.safe_load(text)
        except ImportError:
            data = self._parse_simple(text)
        if not isinstance(data, dict) or "terms" not in data:
            logger.warning("Glossary file is malformed")
            return
        terms = data["terms"]
        if not isinstance(terms, dict):
            return
        for key, entry in terms.items():
            if isinstance(entry, dict):
                self._entries[key] = GlossaryEntry(
                    en=entry.get("en") or key,
                    ar=entry.get("ar", ""),
                    transliteration=entry.get("transliteration", ""),
                    definition=entry.get("definition", ""),
                )

    def _parse_simple(self, text: str) -> dict[str, Any]:
        result: dict[str, Any] = {"terms": {}}
        current_section: dict[str, dict] | None = None
        current_term: dict | None = None
        for line in text.splitlines():
            if not line.strip() or line.strip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            content = line.strip()
            if indent == 0 and content.endswith(":"):
                section_name = content[:-1]
                if section_name == "terms":
                    current_section = {}
                    result["terms"] = current_section
                continue
            if current_section is None:
                continue
            if indent == 2 and content.endswith(":"):
                term_key = content[:-1]
                current_term = {}
                current_section[term_key] = current_term
                _ = term_key
                continue
            if current_term is None:
                continue
            if indent >= 4 and ":" in content:
                key, _, value = content.partition(":")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                current_term[key] = value
        return result

    def get(self, key: str) -> GlossaryEntry | None:
        return self._entries.get(key)

    def translate(self, en_term: str) -> str | None:
        """Look up Arabic translation for an English term key."""
        entry = self._entries.get(en_term)
        if entry is None:
            return None
        return entry.ar

    def search(self, query: str, limit: int = 10) -> list[GlossaryEntry]:
        """Search entries by English, Arabic, or transliteration substring."""
        q = query.lower()
        results: list[GlossaryEntry] = []
        for entry in self._entries.values():
            if q in entry.en.lower() or q in entry.ar.lower() or q in entry.transliteration.lower():
                results.append(entry)
                if len(results) >= limit:
                    break
        return results

    def all_entries(self) -> list[GlossaryEntry]:
        return list(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries


_singleton: ArabicGlossary | None = None


def get_glossary(path: Path | None = None) -> ArabicGlossary:
    """Singleton-style accessor for the Arabic glossary."""
    global _singleton
    if _singleton is None:
        _singleton = ArabicGlossary(path)
    return _singleton
