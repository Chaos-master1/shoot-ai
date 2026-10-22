"""Parse transcript for player name mentions and link to known squad rosters.

Usage:
    python scripts/parse_transcript_names.py France vs Sweden_match_transcript.txt --squad squad.json

Outputs:
    - Named events (when a player name is spoken, with timestamp)
    - Can cross-reference with jersey OCR results
"""

import json
import re
from pathlib import Path
from typing import Any

# Common football name patterns
NAME_PATTERN = re.compile(
    r"\b([A-Z][a-zéèêëàâùûüôöîïç]{2,}(?:[\s-][A-Z][a-zéèêëàâùûüôöîïç]{2,})?)\b"
)

# Frequent non-player words that look like names
SKIP_WORDS = {
    "Thank",
    "Please",
    "Remain",
    "Standing",
    "Very",
    "Much",
    "Able",
    "National",
    "Anthem",
    "France",
    "Sweden",
    "England",
    "Germany",
    "Spain",
    "Italy",
    "Brazil",
    "Argentina",
    "Portugal",
    "Netherlands",
    "World",
    "Cup",
    "Qualifying",
    "League",
    "Team",
    "Match",
    "Game",
    "Half",
    "Time",
    "Goal",
    "Shot",
    "Save",
    "Pass",
    "Foul",
    "Corner",
    "Yellow",
    "Red",
    "Card",
    "Substitute",
    "Injury",
    "Offside",
    "Referee",
    "Manager",
    "Coach",
    "Stadium",
    "Today",
    "Tonight",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
    "Hello",
    "Welcome",
    "Here",
    "There",
    "Where",
    "What",
    "When",
    "Really",
    "Just",
    "Over",
    "Under",
    "About",
    "After",
    "Before",
    "During",
    "Without",
    "Because",
    "While",
}


def parse_transcript(transcript_path: Path) -> list[dict[str, Any]]:
    """Parse timestamped transcript lines.

    Format: [start-end] text
    """
    segments = []
    pattern = re.compile(r"\[(\d+\.?\d*)s-(\d+\.?\d*)s\]\s*(.+)")
    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                segments.append(
                    {
                        "start": float(m.group(1)),
                        "end": float(m.group(2)),
                        "text": m.group(3).strip(),
                    }
                )
    return segments


def extract_name_mentions(
    segments: list[dict[str, Any]],
    known_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Find player name mentions in transcript segments.

    Args:
        segments: Parsed transcript segments.
        known_names: Set of known player names to filter by.
                     If None, uses heuristic detection.

    Returns:
        List of {name, timestamp, segment_text}
    """
    mentions = []
    for seg in segments:
        words = NAME_PATTERN.findall(seg["text"])
        for word in words:
            word_clean = word.strip().rstrip(".,!?;:")
            if not word_clean or word_clean in SKIP_WORDS:
                continue
            if len(word_clean) < 3:
                continue
            if known_names and word_clean not in known_names:
                continue
            mentions.append(
                {
                    "name": word_clean,
                    "timestamp": seg["start"],
                    "end": seg["end"],
                    "context": seg["text"],
                }
            )
    return mentions


def build_squad_lookup(roster_path: Path) -> dict[str, dict[int, str]]:
    """Build name lookup from squad JSON.

    Format: {"team_name": [{"name": "Player Name", "number": 10}, ...]}
    """
    with open(roster_path) as f:
        roster = json.load(f)
    lookup: dict[str, dict[int, str]] = {}
    all_names: set[str] = set()
    for team, players in roster.items():
        lookup[team] = {}
        for p in players:
            lookup[team][p["number"]] = p["name"]
            all_names.add(p["name"])
    return lookup, all_names


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Parse transcript for player names")
    parser.add_argument("transcript", type=Path, help="Transcript file")
    parser.add_argument("--squad", type=Path, help="Squad roster JSON")
    args = parser.parse_args()

    segments = parse_transcript(args.transcript)
    print(f"Parsed {len(segments)} segments from {args.transcript.name}")

    known_names = None
    if args.squad:
        lookup, names = build_squad_lookup(args.squad)
        known_names = names
        print(f"Loaded {len(names)} known player names from {args.squad.name}")

    mentions = extract_name_mentions(segments, known_names)

    # Filter unique names and count
    name_counts: dict[str, int] = {}
    for m in mentions:
        name_counts[m["name"]] = name_counts.get(m["name"], 0) + 1

    # Deduplicate: only first mention per name
    seen_names: set[str] = set()
    unique_mentions = []
    for m in mentions:
        if m["name"] not in seen_names:
            unique_mentions.append(m)
            seen_names.add(m["name"])

    print(f"\nUnique player mentions ({len(unique_mentions)}):")
    for m in unique_mentions:
        print(f'  {m["name"]} @ {m["timestamp"]:.1f}s — "{m["context"]}"')

    print("\nName frequency:")
    for name, count in sorted(name_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}x")


if __name__ == "__main__":
    main()
