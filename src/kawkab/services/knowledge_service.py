"""Knowledge base service - loads tactical rules and drills from YAML.

Current inventory: 40 rule files across 5 categories (defensive, offensive,
transitions, individual, meta) and 24 drills. Counted from the files by
scripts/check_docs_truth.py — do not state larger numbers here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from kawkab.core.logging import get_logger
from kawkab.core.paths import get_paths

logger = get_logger(__name__)


@dataclass
class TacticalRule:
    """A tactical rule for diagnosing football problems."""

    rule_id: str
    category: str
    subcategory: str
    severity: str
    names: dict[str, str]  # localized names
    description: dict[str, str]
    pattern_signature: dict[str, Any]
    hypotheses: list[dict[str, Any]]
    recommended_drills: list[str]
    sources: list[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> TacticalRule:
        """Load a tactical rule from a YAML file."""
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        rule_data = data.get("rule", data)
        return cls(
            rule_id=rule_data["id"],
            category=rule_data["category"],
            subcategory=rule_data.get("subcategory", ""),
            severity=rule_data.get("severity", "medium"),
            names=rule_data.get("names", {}),
            description=rule_data.get("description", {}),
            pattern_signature=rule_data.get("pattern_signature", {}),
            hypotheses=rule_data.get("hypotheses", []),
            recommended_drills=rule_data.get("recommended_drills", []),
            sources=rule_data.get("sources", []),
        )


@dataclass
class Drill:
    """A training drill."""

    drill_id: str
    name: str
    category: str
    targets: list[str]
    duration_min: int
    players_required: int
    intensity: str
    equipment: list[str]
    space: str
    setup: str
    rules: list[str]
    progressions: list[str] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)
    coaching_points: list[str] = field(default_factory=list)
    addresses_problems: list[str] = field(default_factory=list)
    source: str = ""
    video_reference: str = ""

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> Drill:
        """Load a drill from a YAML file."""
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        drill_list = data if isinstance(data, list) else [data]
        d = drill_list[0]

        return cls(
            drill_id=d["drill_id"],
            name=d["name"],
            category=d.get("category", "general"),
            targets=d.get("targets", []),
            duration_min=d.get("duration_min", 15),
            players_required=d.get("players_required", 6),
            intensity=d.get("intensity", "medium"),
            equipment=d.get("equipment", []),
            space=d.get("space", ""),
            setup=d.get("setup", ""),
            rules=d.get("rules", []),
            progressions=d.get("progressions", []),
            regressions=d.get("regressions", []),
            coaching_points=d.get("coaching_points", []),
            addresses_problems=d.get("addresses_problems", []),
            source=d.get("source", ""),
            video_reference=d.get("video_reference", ""),
        )


class KnowledgeService:
    """Manages the tactical knowledge base."""

    def __init__(self) -> None:
        self._rules: dict[str, TacticalRule] = {}
        self._drills: dict[str, Drill] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Load all knowledge base files from disk."""
        if self._initialized:
            return

        paths = get_paths()
        kb_root = paths.knowledge_base

        logger.info(f"Loading knowledge base from {kb_root}")

        rule_count = 0
        for yaml_file in (kb_root / "tactics").rglob("*.yaml"):
            try:
                rule = TacticalRule.from_yaml(yaml_file)
                self._rules[rule.rule_id] = rule
                rule_count += 1
            except Exception as e:
                logger.warning(f"Failed to load rule {yaml_file}: {e}")

        for yml_file in (kb_root / "tactics").rglob("*.yml"):
            try:
                rule = TacticalRule.from_yaml(yml_file)
                self._rules[rule.rule_id] = rule
                rule_count += 1
            except Exception as e:
                logger.warning(f"Failed to load rule {yml_file}: {e}")

        drill_count = 0
        for yaml_file in (kb_root / "drills").rglob("*.yaml"):
            try:
                drill = Drill.from_yaml(yaml_file)
                self._drills[drill.drill_id] = drill
                drill_count += 1
            except Exception as e:
                logger.warning(f"Failed to load drill {yaml_file}: {e}")

        for yml_file in (kb_root / "drills").rglob("*.yml"):
            try:
                drill = Drill.from_yaml(yml_file)
                self._drills[drill.drill_id] = drill
                drill_count += 1
            except Exception as e:
                logger.warning(f"Failed to load drill {yml_file}: {e}")

        logger.info(f"Knowledge base loaded: {rule_count} rules, {drill_count} drills")
        self._initialized = True

    def get_rule(self, rule_id: str) -> TacticalRule | None:
        """Get a tactical rule by ID."""
        return self._rules.get(rule_id)

    def get_drill(self, drill_id: str) -> Drill | None:
        """Get a drill by ID."""
        return self._drills.get(drill_id)

    def find_rules_for_pattern(
        self, pattern_type: str, category: str | None = None
    ) -> list[TacticalRule]:
        """Find rules that match a given pattern type.

        Args:
            pattern_type: Type of pattern (e.g., "zone_based_goal_concession")
            category: Optional category filter (e.g., "defensive")

        Returns:
            List of matching rules
        """
        results = []
        for rule in self._rules.values():
            if category and rule.category != category:
                continue
            sig = rule.pattern_signature
            if sig.get("type") == pattern_type:
                results.append(rule)
        return results

    def find_drills_for_problem(self, problem_signature: str) -> list[Drill]:
        """Find drills that address a specific problem.

        Args:
            problem_signature: Problem identifier (e.g., "possession_loss_under_press")

        Returns:
            List of matching drills
        """
        results = []
        for drill in self._drills.values():
            if drill.addresses_problems and problem_signature in drill.addresses_problems:
                results.append(drill)
        return results

    def get_all_rules(self) -> list[TacticalRule]:
        """Get all loaded rules."""
        return list(self._rules.values())

    def get_all_drills(self) -> list[Drill]:
        """Get all loaded drills."""
        return list(self._drills.values())

    @property
    def stats(self) -> dict[str, int]:
        """Get knowledge base statistics."""
        return {
            "rules": len(self._rules),
            "drills": len(self._drills),
        }
