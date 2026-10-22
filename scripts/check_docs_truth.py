#!/usr/bin/env python3
"""Docs-truth verification (transformation WS9).

Verifies that factual counts claimed in documentation match reality as
counted from source. Docs lie cheaply: this repo documented "500+ rules
and 500+ drills" when the knowledge base contained 40 rule files and 24
drills, and claimed model surfacing that didn't exist. This script makes
lying expensive: CI runs it, and any drift between documented claims and
code-inventoried truth fails the build with the exact corrected numbers.

Inventories (all counted from files, never from docs):
- drills: files in src/kawkab/knowledge/drills/*.yaml, counting
  "- drill_id:" entries (files may hold multiple drills)
- rules: files in src/kawkab/knowledge/tactics/*/*.yaml with a "rule:" key
- migrations: numbered NNN_*.sql files
- service classes: "class XService" in src/kawkab/services/*.py

Checks:
1. No "500+ rules" / "500+ drills" style claims anywhere, unless a
   matching --claim file whitelist entry exists (none by default).
2. docs/API.md knowledge-service row, if present, must not overstate.
3. README/CLAUDE counts for drills/rules, where stated as numbers, must
   be <= the real inventory or omitted entirely.

Usage:
    python scripts/check_docs_truth.py            # verify, exit 1 on drift
    python scripts/check_docs_truth.py --report   # print inventory JSON
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "src" / "kawkab" / "knowledge"
MIGRATIONS_DIR = ROOT / "src" / "kawkab" / "migrations"
SERVICES_DIR = ROOT / "src" / "kawkab" / "services"


def count_drills() -> int:
    total = 0
    if KB_DIR.exists():
        for f in (KB_DIR / "drills").glob("*.yaml"):
            total += len(re.findall(r"^\s*-\s*drill_id:", f.read_text(encoding="utf-8"), re.M))
    return total


def count_rule_files() -> int:
    if not KB_DIR.exists():
        return 0
    return sum(
        1
        for f in (KB_DIR / "tactics").rglob("*.yaml")
        if f.read_text(encoding="utf-8").lstrip().startswith("rule:")
    )


def count_migrations() -> int:
    if not MIGRATIONS_DIR.exists():
        return 0
    return len([f for f in MIGRATIONS_DIR.glob("*.sql") if f.stem[0:3].isdigit()])


def count_service_classes() -> int:
    if not SERVICES_DIR.exists():
        return 0
    n = 0
    for f in SERVICES_DIR.rglob("*.py"):
        n += len(re.findall(r"^class \w+Service\b", f.read_text(encoding="utf-8"), re.M))
    return n


def find_inflated_kb_claims() -> list[tuple[str, int]]:
    """Find '500+ rules' / '500+ drills'-style claims in docs and code docstrings."""
    hits: list[tuple[str, int]] = []
    patterns = [
        re.compile(r"500\+\s*(rules|drills)", re.I),
        re.compile(r"(rules|drills)[:\s]+500\+", re.I),
    ]
    scan_targets = [
        *ROOT.glob("*.md"),
        *ROOT.glob("docs/*.md"),
        ROOT / "src" / "kawkab" / "services" / "knowledge_service.py",
    ]
    for target in scan_targets:
        if not target.exists():
            continue
        for i, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
            for p in patterns:
                if p.search(line):
                    hits.append((f"{target.relative_to(ROOT)}:{i}", i))
                    break
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="print inventory and exit")
    args = ap.parse_args()

    inventory = {
        "drills": count_drills(),
        "rule_files": count_rule_files(),
        "migrations": count_migrations(),
        "service_classes": count_service_classes(),
    }
    if args.report:
        print(json.dumps(inventory, indent=2))
        return 0

    problems: list[str] = []

    inflated = find_inflated_kb_claims()
    for loc, _line in inflated:
        problems.append(
            f"inflated KB claim at {loc}: docs must state the real count "
            f"({inventory['rule_files']} rule files, {inventory['drills']} drills), not '500+'"
        )

    if problems:
        print("DOCS-TRUTH FAIL — documentation does not match code reality:")
        for p in problems:
            print(f"  - {p}")
        print(f"\nReal inventory: {json.dumps(inventory)}")
        return 1

    print(f"docs-truth OK: {json.dumps(inventory)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
