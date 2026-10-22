#!/usr/bin/env python3
"""Compile .po locale files into .json dictionaries for the frontend.

Usage:
    python scripts/compile_i18n.py

Reads locales/*.po and writes locales/*.json files that can be fetched
by the web frontend at runtime.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

LOCALES_DIR = Path(__file__).resolve().parent.parent / "locales"
PO_PATTERN = re.compile(r'^msgid "(.+)"\n^msgstr "(.+)"', re.MULTILINE)


def parse_po(text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for m in PO_PATTERN.finditer(text):
        key = m.group(1)
        value = m.group(2).replace("\\n", "\n")
        entries[key] = value
    return entries


def main() -> int:
    if not LOCALES_DIR.is_dir():
        print(f"ERROR: locales directory not found at {LOCALES_DIR}", file=sys.stderr)
        return 1

    for po_path in sorted(LOCALES_DIR.glob("*.po")):
        lang = po_path.stem
        text = po_path.read_text(encoding="utf-8")
        entries = parse_po(text)
        if not entries:
            print(f"WARNING: no entries parsed from {po_path.name}", file=sys.stderr)
            continue

        json_path = LOCALES_DIR / f"{lang}.json"
        json_path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  {po_path.name} -> {json_path.name}  ({len(entries)} keys)")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
