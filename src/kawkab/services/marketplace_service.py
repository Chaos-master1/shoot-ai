"""Community marketplace — drill sharing, tactical templates, and plugin store.

Items are stored locally with optional cloud sync. Categories:
- drills: Training drills with instructions, diagrams, difficulty level
- templates: Tactical templates (formations, set pieces, game plans)
- plugins: Extensions/scripts that enhance the platform
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal


@dataclass
class MarketplaceItem:
    id: str
    item_type: Literal["drill", "template", "plugin"]
    name: str
    description: str = ""
    author: str = ""
    version: str = "1.0.0"
    category: str = ""
    tags: list[str] = field(default_factory=list)
    rating: float = 0.0
    download_count: int = 0
    data: str = ""  # JSON blob with the actual content
    preview_image: str = ""  # base64 or path
    source: Literal["local", "community"] = "local"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


# Built-in sample items for the marketplace
SAMPLE_DRILLS = [
    {
        "name": "Rondo 4v2 Possession",
        "description": "Classic 4v2 rondo to improve quick passing, movement off the ball, and pressing in tight spaces.",
        "category": "possession",
        "tags": ["rondo", "possession", "passing", "pressing"],
        "data": json.dumps(
            {
                "duration_min": 10,
                "players": 6,
                "area": "15x15m",
                "equipment": ["cones", "bibs"],
                "instructions": [
                    "Mark a 15x15m square with cones.",
                    "4 attackers position on the outside, 2 defenders in the middle.",
                    "Attackers pass the ball among themselves while defenders try to intercept.",
                    "If a defender wins the ball, switch roles.",
                    "Progress to 1-touch passing.",
                ],
                "variations": ["Limit to 2-touch", "Add neutral player", "Increase to 5v3"],
            }
        ),
        "rating": 4.5,
    },
    {
        "name": "Finishing Under Pressure",
        "description": "High-intensity finishing drill combining passing combinations with a final shot under defensive pressure.",
        "category": "finishing",
        "tags": ["finishing", "shooting", "pressure", "combinations"],
        "data": json.dumps(
            {
                "duration_min": 15,
                "players": 8,
                "area": "Penalty box + midfield",
                "equipment": ["cones", "goals", "balls"],
                "instructions": [
                    "Set up a passing circuit leading to the edge of the box.",
                    "Player A passes to B who lays off to C sprinting into the box.",
                    "C shoots first time at goal with a defender giving chase.",
                    "Rotate positions every 5 minutes.",
                    "Focus on one-touch finishing and composure.",
                ],
                "variations": ["Add goalkeeper", "Cross from wide instead of through middle"],
            }
        ),
        "rating": 4.2,
    },
    {
        "name": "Defensive Shape & Compactness",
        "description": "Team defensive drill focusing on maintaining compact shape, cover, and balance during opposition build-up.",
        "category": "defense",
        "tags": ["defense", "shape", "compactness", "team"],
        "data": json.dumps(
            {
                "duration_min": 20,
                "players": 10,
                "area": "Half pitch",
                "equipment": ["cones", "bibs", "goals"],
                "instructions": [
                    "Set up in a 4-4-2 defensive block on half pitch.",
                    "Attackers build up from the back with midfield support.",
                    "Defensive team must maintain compact shape (max 35m vertical, 40m horizontal).",
                    "Focus on shifting as a unit and closing passing lanes.",
                    "Coach calls 'press' trigger to execute coordinated press.",
                ],
                "variations": ["3-5-2 shape", "High press vs mid block"],
            }
        ),
        "rating": 4.7,
    },
    {
        "name": "Transition to Attack",
        "description": "Drill for quick transition from defense to attack after winning possession, emphasizing vertical passes and forward runs.",
        "category": "transitions",
        "tags": ["transitions", "counter-attack", "speed", "forward runs"],
        "data": json.dumps(
            {
                "duration_min": 15,
                "players": 10,
                "area": "Full pitch",
                "equipment": ["cones", "bibs", "goals"],
                "instructions": [
                    "Split into two teams of 5 with goalkeepers.",
                    "Team A attacks, Team B defends.",
                    "When Team B wins possession, they must transition within 3 passes to a shot on goal.",
                    "Team A must immediately transition to defend.",
                    "Emphasize forward passing and sprinting to support.",
                ],
                "variations": ["Limit to 2 touches", "Add neutral player"],
            }
        ),
        "rating": 4.4,
    },
    {
        "name": "Wide Play & Crossing",
        "description": "Develop wide attacking play with overlapping full-backs, quality crosses, and organized box finishing.",
        "category": "attack",
        "tags": ["wide play", "crossing", "overlap", "finishing"],
        "data": json.dumps(
            {
                "duration_min": 20,
                "players": 12,
                "area": "Full pitch width, final third",
                "equipment": ["cones", "bibs", "goals", "balls"],
                "instructions": [
                    "Wide players start wide, full-backs deep.",
                    "Midfielder switches play to the winger.",
                    "Full-back overlaps, receives pass, and crosses into the box.",
                    "3 attackers make runs: near post, far post, edge of box.",
                    "Rotate sides after 5 reps.",
                ],
                "variations": ["Cut-back passes instead of crosses", "Add defenders"],
            }
        ),
        "rating": 4.0,
    },
]

SAMPLE_TEMPLATES = [
    {
        "name": "4-3-3 Possession Build-up",
        "description": "Complete tactical template for build-up play in a 4-3-3 formation with player roles and movement patterns.",
        "category": "formation",
        "tags": ["4-3-3", "build-up", "possession", "tactical"],
        "data": json.dumps(
            {
                "formation": "4-3-3",
                "phases": {
                    "build_up": "CBs split wide, full-books push high, DM drops between CBs, CM create diamonds, wingers stretch the pitch.",
                    "midfield": "8 and 10 rotate positions, 6 dictates tempo, full-books provide width in final third.",
                    "final_third": "Wingers 1v1, full-books overlap, 8 arrives late at far post, 10 occupies half-spaces.",
                },
                "player_roles": {
                    "GK": "Sweeper keeper, comfortable with feet",
                    "CB": "Ball-playing, good passing range",
                    "FB": "High energy, good crossing, tactical discipline",
                    "DM": "Metronome, defensive screen, short passing",
                    "CM": "Box-to-box, arrives in box, pressing trigger",
                    "W": "1v1 specialist, dribbling, cut inside, crossing",
                    "CF": "Link-up play, dropping deep, finishing",
                },
                "strengths": ["Control of possession", "Overloads in midfield", "Width in attack"],
                "weaknesses": ["Counter-attack vulnerable", "Requires high fitness from FBs"],
            }
        ),
        "rating": 4.8,
    },
    {
        "name": "3-5-2 Counter-Attack",
        "description": "Compact 3-5-2 setup designed for quick transitions and counter-attacking football.",
        "category": "formation",
        "tags": ["3-5-2", "counter-attack", "compact", "transitions"],
        "data": json.dumps(
            {
                "formation": "3-5-2",
                "phases": {
                    "defensive": "Back 5 (3 CB + 2 WB) compact, midfield 3 screen, 2 forwards stay high.",
                    "transition": "Win ball → vertical pass to forward → support runner from midfield in 3 passes.",
                    "attack": "WBs provide width, 2 forwards combine, AM arrives late.",
                },
                "player_roles": {
                    "CB": "Strong 1v1, good passing, aggressive",
                    "WB": "Endless stamina, cross, defend, attack",
                    "CM": "Box-to-box, duel winner, simple passing",
                    "AM": "Creative, final pass, second striker",
                    "CF": "Hold-up play, pace, finishing",
                },
                "strengths": [
                    "Defensive solidity",
                    "Quick transitions",
                    "Numerical advantage in midfield",
                ],
                "weaknesses": ["Vulnerable to wide overloads", "CBs isolated 1v1"],
            }
        ),
        "rating": 4.3,
    },
]

SAMPLE_PLUGINS = [
    {
        "name": "Expected Threat (xT) Heatmap Overlay",
        "description": "Visual overlay showing xT zone values as a pitch heatmap. Color-coded from low (blue) to high threat (red).",
        "category": "visualization",
        "tags": ["xT", "heatmap", "visualization", "overlay"],
        "data": json.dumps({"type": "visualization", "hook": "analysis", "version": "1.0"}),
        "rating": 4.1,
    },
    {
        "name": "Auto-Formation Detection",
        "description": "Automatically detects team formation from event data using k-means clustering of average player positions.",
        "category": "analysis",
        "tags": ["formation", "analysis", "auto-detect", "machine-learning"],
        "data": json.dumps({"type": "analysis", "hook": "post_analysis", "version": "1.0"}),
        "rating": 4.6,
    },
    {
        "name": "Export to Video + Graphics",
        "description": "Exports match analysis as a broadcast-style video with stats overlay, goal graphics, and telestration animations.",
        "category": "export",
        "tags": ["export", "video", "graphics", "broadcast"],
        "data": json.dumps({"type": "export", "hook": "export", "version": "2.0"}),
        "rating": 4.4,
    },
]


class MarketplaceService:
    """Community marketplace for sharing drills, templates, and plugins."""

    def __init__(self) -> None:
        self._items: dict[str, MarketplaceItem] = {}
        self._data_file = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "marketplace.json"
        )
        self._load_data()
        self._seed_samples()

    def _load_data(self) -> None:
        os.makedirs(os.path.dirname(self._data_file), exist_ok=True)
        try:
            if os.path.exists(self._data_file):
                with open(self._data_file, encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    self._items[item["id"]] = MarketplaceItem(**item)
        except Exception as e:
            import logging

            logging.warning(f"Failed to load marketplace data: {e}")

    def _save_data(self) -> None:
        data = [vars(item) for item in self._items.values()]
        with open(self._data_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def _seed_samples(self) -> None:
        import uuid

        for sample in SAMPLE_DRILLS:
            name = sample["name"]
            exists = any(
                item.name == name and item.item_type == "drill" for item in self._items.values()
            )
            if not exists:
                item = MarketplaceItem(
                    id=str(uuid.uuid4())[:8],
                    item_type="drill",
                    source="community",
                    **dict(sample.items()),  # type: ignore[arg-type]
                )
                self._items[item.id] = item

        for sample in SAMPLE_TEMPLATES:
            name = sample["name"]
            exists = any(
                item.name == name and item.item_type == "template" for item in self._items.values()
            )
            if not exists:
                item = MarketplaceItem(
                    id=str(uuid.uuid4())[:8],
                    item_type="template",
                    source="community",
                    **dict(sample.items()),  # type: ignore[arg-type]
                )
                self._items[item.id] = item

        for sample in SAMPLE_PLUGINS:
            name = sample["name"]
            exists = any(
                item.name == name and item.item_type == "plugin" for item in self._items.values()
            )
            if not exists:
                item = MarketplaceItem(
                    id=str(uuid.uuid4())[:8],
                    item_type="plugin",
                    source="community",
                    **dict(sample.items()),  # type: ignore[arg-type]
                )
                self._items[item.id] = item

        self._save_data()

    def list_items(
        self, item_type: str = "", category: str = "", query: str = "", source: str = ""
    ) -> list[dict]:
        query = query.lower().strip()
        results: list[dict[str, Any]] = []
        for item in self._items.values():
            if item_type and item.item_type != item_type:
                continue
            if category and item.category != category:
                continue
            if query and query not in item.name.lower() and query not in item.description.lower():
                continue
            if source and item.source != source:
                continue
            results.append(
                {
                    "id": item.id,
                    "item_type": item.item_type,
                    "name": item.name,
                    "description": item.description[:120],
                    "author": item.author or "Community",
                    "version": item.version,
                    "category": item.category,
                    "tags": item.tags,
                    "rating": item.rating,
                    "download_count": item.download_count,
                    "source": item.source,
                    "created_at": item.created_at,
                }
            )
        results.sort(key=lambda x: x["rating"], reverse=True)
        return results

    def get_item(self, item_id: str) -> dict | None:
        item = self._items.get(item_id)
        if not item:
            return None
        return dict(vars(item).items())

    def add_item(
        self,
        item_type: Literal["drill", "template", "plugin"],
        name: str,
        description: str = "",
        author: str = "",
        category: str = "",
        tags: list[str] | None = None,
        data: str = "",
        source: Literal["local", "community"] = "local",
    ) -> dict:
        import uuid

        item_id = str(uuid.uuid4())[:8]
        item = MarketplaceItem(
            id=item_id,
            item_type=item_type,
            name=name,
            description=description,
            author=author,
            category=category,
            tags=tags or [],
            data=data,
            source=source,
        )
        self._items[item_id] = item
        self._save_data()
        return {"id": item_id, "name": name, "item_type": item_type}

    def rate_item(self, item_id: str, rating: float) -> bool:
        item = self._items.get(item_id)
        if not item:
            return False
        item.rating = round((item.rating + rating) / 2, 1)
        item.download_count += 1
        self._save_data()
        return True

    def delete_item(self, item_id: str) -> bool:
        if item_id in self._items:
            del self._items[item_id]
            self._save_data()
            return True
        return False

    def get_categories(self, item_type: str = "") -> list[str]:
        cats = set()
        for item in self._items.values():
            if item_type and item.item_type != item_type:
                continue
            if item.category:
                cats.add(item.category)
        return sorted(cats)

    def get_stats(self) -> dict:
        drills = sum(1 for i in self._items.values() if i.item_type == "drill")
        templates = sum(1 for i in self._items.values() if i.item_type == "template")
        plugins = sum(1 for i in self._items.values() if i.item_type == "plugin")
        return {
            "total": len(self._items),
            "drills": drills,
            "templates": templates,
            "plugins": plugins,
        }
