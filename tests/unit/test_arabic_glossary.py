"""Tests for ArabicGlossary."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()
_g = load_service_module("gl_test", "arabic_glossary.py", subdir="i18n")
ArabicGlossary = _g.ArabicGlossary
GlossaryEntry = _g.GlossaryEntry

import pytest


@pytest.fixture
def glossary() -> ArabicGlossary:
    return ArabicGlossary()


class TestGlossary:
    def test_loads_file(self, glossary: ArabicGlossary) -> None:
        assert len(glossary) > 0

    def test_contains_basic_terms(self, glossary: ArabicGlossary) -> None:
        for key in ["match", "goal", "save", "pass", "offside"]:
            assert key in glossary, f"Missing basic term: {key}"

    def test_translate_known(self, glossary: ArabicGlossary) -> None:
        assert glossary.translate("goal") == "هدف"
        assert glossary.translate("save") == "تصدي"
        assert glossary.translate("penalty") == "ضربة جزاء"

    def test_translate_unknown(self, glossary: ArabicGlossary) -> None:
        assert glossary.translate("nonexistent_term_xyz") is None

    def test_get_returns_entry(self, glossary: ArabicGlossary) -> None:
        entry = glossary.get("yellow_card")
        assert entry is not None
        assert entry.en == "Yellow card"
        assert entry.ar == "بطاقة صفراء"
        assert entry.transliteration == "biṭāqa ṣafrāʾ"

    def test_entry_has_definition(self, glossary: ArabicGlossary) -> None:
        entry = glossary.get("offside")
        assert entry is not None
        assert "closer to the goal" in entry.definition

    def test_search_english(self, glossary: ArabicGlossary) -> None:
        results = glossary.search("corner")
        assert any("corner" in e.en.lower() for e in results)

    def test_search_arabic(self, glossary: ArabicGlossary) -> None:
        results = glossary.search("هدف")
        assert any("هدف" in e.ar for e in results)

    def test_search_transliteration(self, glossary: ArabicGlossary) -> None:
        results = glossary.search("hadaf")
        assert any("hadaf" in e.transliteration.lower() for e in results)

    def test_search_limit(self, glossary: ArabicGlossary) -> None:
        results = glossary.search("a", limit=3)
        assert len(results) <= 3

    def test_all_entries(self, glossary: ArabicGlossary) -> None:
        entries = glossary.all_entries()
        assert isinstance(entries, list)
        assert len(entries) == len(glossary)

    def test_contains(self, glossary: ArabicGlossary) -> None:
        assert "goal" in glossary
        assert "xyz_unknown" not in glossary

    def test_no_arabic_text_in_transliteration_only(self, glossary: ArabicGlossary) -> None:
        for entry in glossary.all_entries():
            if entry.transliteration:
                for ch in entry.transliteration:
                    assert ord(ch) < 128 or ch in "āīūĀĪŌḍḥṣṭẓġšḫʿʾʿ", (
                        f"non-ASCII in transliteration: {entry.transliteration}"
                    )

    def test_at_least_50_terms(self, glossary: ArabicGlossary) -> None:
        assert len(glossary) >= 50, f"Only {len(glossary)} terms found"

    def test_all_entries_have_arabic(self, glossary: ArabicGlossary) -> None:
        for entry in glossary.all_entries():
            assert entry.ar, f"Missing Arabic for: {entry.en}"

    def test_unique_keys(self, glossary: ArabicGlossary) -> None:
        keys = list(glossary._entries.keys())
        assert len(keys) == len(set(keys))


class TestSimpleParser:
    def test_parse_minimal(self) -> None:
        text = """
terms:
  goal:
    en: "Goal"
    ar: "هدف"
  save:
    en: "Save"
    ar: "تصدي"
"""
        g = ArabicGlossary.__new__(ArabicGlossary)
        g._entries = {}
        data = g._parse_simple(text)
        assert "terms" in data
        assert "goal" in data["terms"]
        assert data["terms"]["goal"]["ar"] == "هدف"
