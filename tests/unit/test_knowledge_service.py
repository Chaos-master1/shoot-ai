"""Tests for KnowledgeService — tactical rule and drill knowledge base."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()


# ---------------------------------------------------------------------------
# YAML stub (PyYAML may not be in test env)
# ---------------------------------------------------------------------------

_PREV_YAML: types.ModuleType | None = None
_PREV_YAML_SAFE_LOAD = None


def _install_yaml_stub():
    global _PREV_YAML, _PREV_YAML_SAFE_LOAD
    if "yaml" in sys.modules:
        _PREV_YAML = sys.modules["yaml"]
        _PREV_YAML_SAFE_LOAD = getattr(_PREV_YAML, "safe_load", None)
        return
    _PREV_YAML = None
    yaml_mod = types.ModuleType("yaml")
    from unittest.mock import MagicMock

    yaml_mod.safe_load = MagicMock(return_value={})
    sys.modules["yaml"] = yaml_mod


def _restore_yaml() -> None:
    global _PREV_YAML, _PREV_YAML_SAFE_LOAD
    if _PREV_YAML is not None:
        _PREV_YAML.safe_load = _PREV_YAML_SAFE_LOAD
        sys.modules["yaml"] = _PREV_YAML
    else:
        sys.modules.pop("yaml", None)
    _PREV_YAML = None


def _patch_paths_with_knowledge_base():
    """Add knowledge_base attribute to the conftest _Paths stub."""
    paths_mod = sys.modules.get("kawkab.core.paths")
    if paths_mod is None:
        return
    orig_get_paths = getattr(paths_mod, "_orig_get_paths", None)
    if orig_get_paths is None:
        orig_get_paths = paths_mod.get_paths
        paths_mod._orig_get_paths = orig_get_paths

    def _patched_get_paths():
        p = orig_get_paths()
        p.__dict__["knowledge_base"] = Path(__file__).parent / "_kb_test"
        return p

    paths_mod.get_paths = _patched_get_paths


_install_yaml_stub()
_patch_paths_with_knowledge_base()


@pytest.fixture(autouse=True, scope="module")
def _cleanup_yaml_after_module():
    yield
    _restore_yaml()


_mod = load_service_module("know_test", "knowledge_service.py")

TacticalRule = _mod.TacticalRule
Drill = _mod.Drill
KnowledgeService = _mod.KnowledgeService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kb_paths(tmp_path):
    """Patch get_paths so knowledge_base points to tmp_path/knowledge_base."""
    kb_root = tmp_path / "knowledge_base"
    (kb_root / "tactics").mkdir(parents=True, exist_ok=True)
    (kb_root / "drills").mkdir(parents=True, exist_ok=True)

    class _FakePaths:
        knowledge_base = kb_root

    with patch("know_test.get_paths", return_value=_FakePaths()):
        yield kb_root


# ===================================================================
# TacticalRule
# ===================================================================


class TestTacticalRule:
    def test_from_yaml(self, tmp_path):
        import yaml

        yaml.safe_load = MagicMock(
            return_value={
                "rule": {
                    "id": "R001",
                    "category": "defensive",
                    "subcategory": "pressing",
                    "severity": "high",
                    "names": {"en": "High Press Trap"},
                    "description": {"en": "Team loses shape"},
                    "pattern_signature": {"type": "pressing_breakdown"},
                    "hypotheses": [{"condition": "opponent_plays_wide", "action": "shift"}],
                    "recommended_drills": ["D001"],
                    "sources": ["UEFA 2023"],
                }
            }
        )
        f = tmp_path / "rule.yaml"
        f.write_text("dummy")
        rule = TacticalRule.from_yaml(f)
        assert rule.rule_id == "R001"
        assert rule.category == "defensive"
        assert rule.severity == "high"
        assert rule.names["en"] == "High Press Trap"
        assert rule.pattern_signature["type"] == "pressing_breakdown"

    def test_from_yaml_defaults(self, tmp_path):
        import yaml

        yaml.safe_load = MagicMock(
            return_value={
                "rule": {"id": "R002", "category": "offensive"},
            }
        )
        f = tmp_path / "rule.yaml"
        f.write_text("dummy")
        rule = TacticalRule.from_yaml(f)
        assert rule.rule_id == "R002"
        assert rule.subcategory == ""
        assert rule.severity == "medium"
        assert rule.names == {}
        assert rule.hypotheses == []
        assert rule.recommended_drills == []
        assert rule.sources == []

    def test_from_yaml_top_level_fallback(self, tmp_path):
        import yaml

        yaml.safe_load = MagicMock(
            return_value={
                "id": "R003",
                "category": "transition",
            }
        )
        f = tmp_path / "rule.yaml"
        f.write_text("dummy")
        rule = TacticalRule.from_yaml(f)
        assert rule.rule_id == "R003"


# ===================================================================
# Drill
# ===================================================================


class TestDrill:
    def test_from_yaml(self, tmp_path):
        import yaml

        yaml.safe_load = MagicMock(
            return_value={
                "drill_id": "D001",
                "name": "Rondo 4v2",
                "category": "possession",
                "targets": ["passing_accuracy"],
                "duration_min": 12,
                "players_required": 6,
                "intensity": "high",
                "equipment": ["cones", "balls"],
                "space": "20x20",
                "setup": "Place cones in a square",
                "rules": ["two_touch"],
                "progressions": ["add neutral"],
                "regressions": ["increase grid"],
                "coaching_points": ["head up"],
                "addresses_problems": ["R001_loss"],
                "source": "Pep",
                "video_reference": "https://example.com/rondo",
            }
        )
        f = tmp_path / "drill.yaml"
        f.write_text("dummy")
        drill = Drill.from_yaml(f)
        assert drill.drill_id == "D001"
        assert drill.name == "Rondo 4v2"
        assert drill.category == "possession"
        assert drill.duration_min == 12
        assert drill.players_required == 6
        assert drill.intensity == "high"

    def test_from_yaml_list(self, tmp_path):
        import yaml

        yaml.safe_load = MagicMock(
            return_value=[
                {"drill_id": "D002", "name": "Shadow Play"},
                {"drill_id": "D003", "name": "Box Exercise"},
            ]
        )
        f = tmp_path / "drill.yaml"
        f.write_text("dummy")
        drill = Drill.from_yaml(f)
        assert drill.drill_id == "D002"
        assert drill.name == "Shadow Play"

    def test_from_yaml_defaults(self, tmp_path):
        import yaml

        yaml.safe_load = MagicMock(
            return_value={
                "drill_id": "D004",
                "name": "Test",
            }
        )
        f = tmp_path / "drill.yaml"
        f.write_text("dummy")
        drill = Drill.from_yaml(f)
        assert drill.category == "general"
        assert drill.duration_min == 15
        assert drill.players_required == 6
        assert drill.intensity == "medium"
        assert drill.progressions == []
        assert drill.coaching_points == []
        assert drill.source == ""


# ===================================================================
# KnowledgeService initialize
# ===================================================================


class TestKnowledgeServiceInit:
    def test_default_state(self):
        ks = KnowledgeService()
        assert ks._rules == {}
        assert ks._drills == {}
        assert ks._initialized is False

    def test_stats_empty(self):
        ks = KnowledgeService()
        assert ks.stats == {"rules": 0, "drills": 0}


class TestInitialize:
    @pytest.mark.asyncio
    async def test_loads_yaml_and_yml(self, kb_paths):
        import yaml

        yaml.safe_load = MagicMock(
            side_effect=[
                {"rule": {"id": "R001", "category": "defensive"}},
                {"rule": {"id": "R002", "category": "offensive"}},
                {"drill_id": "D001", "name": "Test Drill"},
            ]
        )
        (kb_paths / "tactics" / "a.yaml").write_text("dummy")
        (kb_paths / "tactics" / "b.yml").write_text("dummy")
        (kb_paths / "drills" / "c.yaml").write_text("dummy")
        ks = KnowledgeService()
        await ks.initialize()
        assert len(ks._rules) == 2
        assert len(ks._drills) == 1
        assert ks._initialized is True

    @pytest.mark.asyncio
    async def test_skip_if_already_initialized(self, kb_paths):
        ks = KnowledgeService()
        ks._initialized = True
        await ks.initialize()
        assert len(ks._rules) == 0

    @pytest.mark.asyncio
    async def test_skips_corrupt_yaml_files(self, kb_paths):
        import yaml

        yaml.safe_load = MagicMock(side_effect=Exception("yaml parse error"))
        (kb_paths / "tactics" / "bad.yaml").write_text("}")
        ks = KnowledgeService()
        await ks.initialize()
        assert len(ks._rules) == 0
        assert ks._initialized is True

    @pytest.mark.asyncio
    async def test_missing_directories(self, tmp_path):
        class FakePaths:
            knowledge_base = tmp_path / "nonexistent"

        with patch("know_test.get_paths", return_value=FakePaths()):
            ks = KnowledgeService()
            await ks.initialize()
            assert len(ks._rules) == 0
            assert len(ks._drills) == 0
            assert ks._initialized is True


class TestGetRule:
    @pytest.mark.asyncio
    async def test_get_existing(self, kb_paths):
        import yaml

        yaml.safe_load = MagicMock(
            return_value={
                "rule": {"id": "R001", "category": "defensive"},
            }
        )
        (kb_paths / "tactics" / "a.yaml").write_text("dummy")
        ks = KnowledgeService()
        await ks.initialize()
        rule = ks.get_rule("R001")
        assert rule is not None
        assert rule.rule_id == "R001"

    def test_get_nonexistent(self):
        ks = KnowledgeService()
        assert ks.get_rule("FAKE") is None


class TestGetDrill:
    @pytest.mark.asyncio
    async def test_get_existing(self, kb_paths):
        import yaml

        yaml.safe_load = MagicMock(
            return_value={
                "drill_id": "D001",
                "name": "Test",
            }
        )
        (kb_paths / "drills" / "a.yaml").write_text("dummy")
        ks = KnowledgeService()
        await ks.initialize()
        drill = ks.get_drill("D001")
        assert drill is not None
        assert drill.drill_id == "D001"

    def test_get_nonexistent(self):
        ks = KnowledgeService()
        assert ks.get_drill("FAKE") is None


class TestGetAll:
    @pytest.mark.asyncio
    async def test_get_all_rules(self, kb_paths):
        import yaml

        yaml.safe_load = MagicMock(
            side_effect=[
                {"rule": {"id": "R001", "category": "defensive"}},
                {"rule": {"id": "R002", "category": "offensive"}},
            ]
        )
        (kb_paths / "tactics" / "a.yaml").write_text("dummy")
        (kb_paths / "tactics" / "b.yaml").write_text("dummy")
        ks = KnowledgeService()
        await ks.initialize()
        all_rules = ks.get_all_rules()
        assert len(all_rules) == 2

    @pytest.mark.asyncio
    async def test_get_all_drills(self, kb_paths):
        import yaml

        yaml.safe_load = MagicMock(
            side_effect=[
                {"drill_id": "D001", "name": "One"},
                {"drill_id": "D002", "name": "Two"},
            ]
        )
        (kb_paths / "drills" / "a.yaml").write_text("dummy")
        (kb_paths / "drills" / "b.yaml").write_text("dummy")
        ks = KnowledgeService()
        await ks.initialize()
        all_drills = ks.get_all_drills()
        assert len(all_drills) == 2


class TestFindRulesForPattern:
    @pytest.mark.asyncio
    async def test_match_type(self, kb_paths):
        import yaml

        yaml.safe_load = MagicMock(
            side_effect=[
                {
                    "rule": {
                        "id": "R001",
                        "category": "defensive",
                        "pattern_signature": {"type": "pressing_breakdown"},
                    }
                },
                {
                    "rule": {
                        "id": "R002",
                        "category": "offensive",
                        "pattern_signature": {"type": "counter_attack"},
                    }
                },
            ]
        )
        (kb_paths / "tactics" / "a.yaml").write_text("dummy")
        (kb_paths / "tactics" / "b.yaml").write_text("dummy")
        ks = KnowledgeService()
        await ks.initialize()
        results = ks.find_rules_for_pattern("pressing_breakdown")
        assert len(results) == 1
        assert results[0].rule_id == "R001"

    @pytest.mark.asyncio
    async def test_with_category_filter(self, kb_paths):
        import yaml

        yaml.safe_load = MagicMock(
            side_effect=[
                {
                    "rule": {
                        "id": "R001",
                        "category": "defensive",
                        "pattern_signature": {"type": "zone_loss"},
                    }
                },
                {
                    "rule": {
                        "id": "R002",
                        "category": "offensive",
                        "pattern_signature": {"type": "zone_loss"},
                    }
                },
            ]
        )
        (kb_paths / "tactics" / "a.yaml").write_text("dummy")
        (kb_paths / "tactics" / "b.yaml").write_text("dummy")
        ks = KnowledgeService()
        await ks.initialize()
        results = ks.find_rules_for_pattern("zone_loss", category="defensive")
        assert len(results) == 1
        assert results[0].rule_id == "R001"

    def test_empty_when_not_initialized(self):
        ks = KnowledgeService()
        assert ks.find_rules_for_pattern("anything") == []


class TestFindDrillsForProblem:
    @pytest.mark.asyncio
    async def test_finds_matching_drills(self, kb_paths):
        import yaml

        yaml.safe_load = MagicMock(
            side_effect=[
                {"drill_id": "D001", "name": "One", "addresses_problems": ["loss_under_press"]},
                {"drill_id": "D002", "name": "Two", "addresses_problems": ["poor_finishing"]},
            ]
        )
        (kb_paths / "drills" / "a.yaml").write_text("dummy")
        (kb_paths / "drills" / "b.yaml").write_text("dummy")
        ks = KnowledgeService()
        await ks.initialize()
        results = ks.find_drills_for_problem("loss_under_press")
        assert len(results) == 1
        assert results[0].drill_id == "D001"

    def test_empty_when_not_initialized(self):
        ks = KnowledgeService()
        assert ks.find_drills_for_problem("anything") == []


class TestStats:
    @pytest.mark.asyncio
    async def test_counts_after_initialize(self, kb_paths):
        import yaml

        yaml.safe_load = MagicMock(
            side_effect=[
                {"rule": {"id": "R001", "category": "defensive"}},
                {"drill_id": "D001", "name": "A"},
                {"drill_id": "D002", "name": "B"},
            ]
        )
        (kb_paths / "tactics" / "a.yaml").write_text("dummy")
        (kb_paths / "drills" / "a.yaml").write_text("dummy")
        (kb_paths / "drills" / "b.yaml").write_text("dummy")
        ks = KnowledgeService()
        await ks.initialize()
        assert ks.stats == {"rules": 1, "drills": 2}
