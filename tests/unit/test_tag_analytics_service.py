from __future__ import annotations

from kawkab.services.tag_analytics_service import (
    DEFAULT_TAG_TEMPLATES,
    _format_timecode,
    _guess_category,
    _parse_timecode,
    compute_tag_analytics,
    export_tags_sportscode,
    import_tags_sportscode,
    tags_from_csv,
    tags_to_csv,
)


class TestTagTemplates:
    def test_default_templates_count(self):
        assert len(DEFAULT_TAG_TEMPLATES) == 25

    def test_all_categories_present(self):
        cats = {t.category for t in DEFAULT_TAG_TEMPLATES}
        assert "attack" in cats
        assert "defense" in cats
        assert "mistake" in cats
        assert "set_piece" in cats

    def test_all_have_colors(self):
        for t in DEFAULT_TAG_TEMPLATES:
            assert t.color.startswith("#")

    def test_shortcuts_unique(self):
        shortcuts = [t.shortcut for t in DEFAULT_TAG_TEMPLATES if t.shortcut]
        assert len(shortcuts) == len(set(shortcuts))


class TestComputeTagAnalytics:
    def test_empty_tags_returns_empty(self):
        analytics = compute_tag_analytics([])
        assert analytics.total_tags == 0

    def test_counts_categories(self):
        tags = [
            {"timestamp": 1, "type": "Shot", "category": "attack", "player_name": "P1"},
            {"timestamp": 2, "type": "Tackle", "category": "defense", "player_name": "P2"},
            {"timestamp": 3, "type": "Shot", "category": "attack", "player_name": "P1"},
        ]
        analytics = compute_tag_analytics(tags)
        assert analytics.total_tags == 3
        assert analytics.by_category["attack"] == 2
        assert analytics.by_category["defense"] == 1

    def test_counts_by_type(self):
        tags = [
            {"timestamp": 1, "type": "Shot", "category": "attack"},
            {"timestamp": 2, "type": "Pass", "category": "attack"},
        ]
        analytics = compute_tag_analytics(tags)
        assert analytics.by_type["Shot"] == 1
        assert analytics.by_type["Pass"] == 1

    def test_counts_by_player(self):
        tags = [
            {"timestamp": 1, "type": "Shot", "player_name": "Player A"},
            {"timestamp": 2, "type": "Pass", "player_name": "Player A"},
            {"timestamp": 3, "type": "Tackle", "player_name": "Player B"},
        ]
        analytics = compute_tag_analytics(tags)
        assert analytics.by_player["Player A"] == 2
        assert analytics.by_player["Player B"] == 1

    def test_counts_by_period(self):
        tags = [
            {"timestamp": 100, "period": "1st Half", "type": "Shot", "category": "attack"},
            {"timestamp": 2000, "period": "2nd Half", "type": "Pass", "category": "attack"},
        ]
        analytics = compute_tag_analytics(tags)
        assert analytics.by_period["1st Half"] == 1
        assert analytics.by_period["2nd Half"] == 1

    def test_co_occurrence_detected(self):
        tags = [
            {"timestamp": 1, "type": "Pass", "category": "attack"},
            {"timestamp": 2, "type": "Shot", "category": "attack"},
        ]
        analytics = compute_tag_analytics(tags, window_size_seconds=5.0)
        assert len(analytics.co_occurrence) > 0
        assert (
            analytics.co_occurrence["Pass"]["Shot"] >= 1
            or analytics.co_occurrence["Shot"]["Pass"] >= 1
        )

    def test_no_co_occurrence_outside_window(self):
        tags = [
            {"timestamp": 1, "type": "Pass", "category": "attack"},
            {"timestamp": 100, "type": "Shot", "category": "attack"},
        ]
        analytics = compute_tag_analytics(tags, window_size_seconds=5.0)
        # co_occurrence might still be defaultdict, but counts should be 0
        total_co = sum(sum(v.values()) for v in analytics.co_occurrence.values())
        assert total_co == 0

    def test_patterns_detected(self):
        tags = [
            {"timestamp": 1, "type": "Pass", "category": "attack"},
            {"timestamp": 2, "type": "Through Ball", "category": "attack"},
            {"timestamp": 3, "type": "Shot", "category": "attack"},
        ]
        analytics = compute_tag_analytics(tags, window_size_seconds=10.0)
        patterns = [p for p in analytics.patterns if p["count"] >= 1]
        assert len(patterns) >= 1

    def test_timeline_sorted(self):
        tags = [
            {"timestamp": 10, "type": "B", "category": "defense"},
            {"timestamp": 1, "type": "A", "category": "attack"},
        ]
        analytics = compute_tag_analytics(tags)
        assert analytics.timeline[0]["timestamp"] == 1
        assert analytics.timeline[1]["timestamp"] == 10

    def test_to_dict_serializable(self):
        tags = [{"timestamp": 1, "type": "Shot", "category": "attack", "player_name": "P1"}]
        analytics = compute_tag_analytics(tags)
        d = analytics.to_dict()
        assert d["total_tags"] == 1
        assert d["by_category"]["attack"] == 1
        assert isinstance(d["timeline"], list)


class TestCSVImportExport:
    def test_tags_to_csv_headers(self):
        tags = [
            {
                "timestamp": 1.0,
                "type": "Shot",
                "category": "attack",
                "player_name": "P1",
                "period": "1st",
                "notes": "Goal",
            }
        ]
        csv_text = tags_to_csv(tags)
        assert "timestamp" in csv_text
        assert "type" in csv_text
        assert "Shot" in csv_text

    def test_tags_to_csv_roundtrip(self):
        original = [
            {
                "timestamp": 1.5,
                "type": "Shot",
                "category": "attack",
                "player_name": "P1",
                "period": "1st",
                "notes": "",
            },
            {
                "timestamp": 10.0,
                "type": "Tackle",
                "category": "defense",
                "player_name": "P2",
                "period": "1st",
                "notes": "Won",
            },
        ]
        csv_text = tags_to_csv(original)
        restored = tags_from_csv(csv_text)
        assert len(restored) == 2
        assert restored[0]["type"] == "Shot"
        assert restored[1]["player_name"] == "P2"

    def test_tags_from_csv_empty_string(self):
        result = tags_from_csv("")
        assert result == []

    def test_tags_from_csv_missing_fields(self):
        csv_text = "timestamp,type\n1.0,Shot\n2.0,Pass"
        result = tags_from_csv(csv_text)
        assert len(result) == 2


class TestSportscodeImportExport:
    def test_export_sportscode_format(self):
        tags = [{"timestamp": 90.5, "type": "Shot", "notes": "Goal", "period": "1st"}]
        csv_text = export_tags_sportscode(tags)
        assert "Code" in csv_text
        assert "Time" in csv_text
        assert "Shot" in csv_text

    def test_import_sportscode_roundtrip(self):
        csv_text = "Code,Time,Notes,Period\nShot,00:01:30.500,Goal,1st\nPass,00:02:00.000,,1st"
        result = import_tags_sportscode(csv_text)
        assert len(result) == 2
        assert result[0]["type"] == "Shot"
        assert abs(result[0]["timestamp"] - 90.5) < 0.01
        assert result[1]["type"] == "Pass"

    def test_import_sportscode_empty(self):
        result = import_tags_sportscode("Code,Time,Notes,Period\n")
        assert result == []


class TestFormatTimecode:
    def test_format_zero(self):
        assert _format_timecode(0) == "00:00:00.000"

    def test_format_90_seconds(self):
        assert _format_timecode(90.5) == "00:01:30.500"

    def test_format_3600(self):
        assert _format_timecode(3600) == "01:00:00.000"

    def test_parse_timecode(self):
        assert abs(_parse_timecode("00:01:30.500") - 90.5) < 0.01

    def test_parse_timecode_malformed(self):
        assert _parse_timecode("invalid") == 0.0


class TestGuessCategory:
    def test_shot_is_attack(self):
        assert _guess_category("Shot") == "attack"

    def test_tackle_is_defense(self):
        assert _guess_category("Tackle") == "defense"

    def test_foul_is_mistake(self):
        assert _guess_category("Foul") == "mistake"

    def test_corner_is_set_piece(self):
        assert _guess_category("Corner") == "set_piece"

    def test_unknown_is_set_piece(self):
        assert _guess_category("Unknown") == "set_piece"

    def test_case_insensitive(self):
        assert _guess_category("shot") == "attack"
