"""Tests for opponent dossier auto-generator."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.analysis.opponent_dossier import (
    FormationTendency,
    ScorelinePrediction,
    _build_key_players,
    _classify_build_up,
    _classify_pressing,
    _detect_formations,
    _detect_set_piece_tendencies,
    _predict_lineup,
    _predict_scoreline,
    generate_dossier,
)


def _sample_match(overrides: dict | None = None) -> dict:
    base = {
        "formation": "4-3-3",
        "possession_pct": 55.0,
        "ppda": 9.0,
        "goals_for": 2,
        "goals_against": 1,
        "corners_for": 6,
        "set_piece_threat": 0.15,
        "set_piece_conceded": 0.12,
        "width_usage": 0.6,
        "build_up_style": "mixed",
        "players": [
            {
                "name": "Keeper",
                "position": "GK",
                "minutes_played": 90,
                "goals": 0,
                "assists": 0,
                "xg": 0,
            },
            {
                "name": "Defender1",
                "position": "CB",
                "minutes_played": 90,
                "goals": 0,
                "assists": 0,
                "xg": 0.02,
            },
            {
                "name": "Mid1",
                "position": "CM",
                "minutes_played": 85,
                "goals": 1,
                "assists": 0,
                "xg": 0.3,
            },
            {
                "name": "Striker1",
                "position": "ST",
                "minutes_played": 80,
                "goals": 2,
                "assists": 1,
                "xg": 1.5,
            },
            {
                "name": "Winger1",
                "position": "RW",
                "minutes_played": 75,
                "goals": 0,
                "assists": 2,
                "xg": 0.8,
            },
            {
                "name": "Sub1",
                "position": "sub",
                "minutes_played": 15,
                "goals": 0,
                "assists": 0,
                "xg": 0,
            },
        ],
        "scorers": [{"player": "Striker1", "goals": 2}],
        "assisters": [{"player": "Winger1", "assists": 1}],
    }
    if overrides:
        base.update(overrides)
    return base


def test_detect_formations():
    matches = [
        _sample_match({"formation": "4-3-3"}),
        _sample_match({"formation": "4-3-3"}),
        _sample_match({"formation": "4-4-2"}),
    ]
    forms = _detect_formations(matches)
    assert len(forms) >= 2
    assert forms[0].formation == "4-3-3"
    assert forms[0].count == 2


def test_detect_formations_unknown():
    assert _detect_formations([{}, {}])[0].formation == "unknown"


def test_predict_lineup():
    matches = [
        _sample_match(),
        _sample_match({}),
    ]
    lineup = _predict_lineup(matches)
    assert len(lineup) > 0
    assert any("Keeper" in ln for ln in lineup)


def test_predict_lineup_empty():
    assert _predict_lineup([]) == ["4-3-3 formation assumed"]


def test_build_key_players():
    matches = [
        _sample_match(),
        _sample_match(
            {
                "scorers": [{"player": "Striker1", "goals": 1}],
                "assisters": [{"player": "Mid1", "assists": 2}],
            }
        ),
    ]
    players = _build_key_players(matches)
    assert len(players) > 0
    top = players[0]
    assert top.goals > 0 or top.assists > 0 or top.xg > 0


def test_key_player_threat_scoring():
    """Players with more goals/assists get higher threat scores."""
    matches = [
        {
            "players": [
                {
                    "name": "Star",
                    "position": "ST",
                    "minutes_played": 90,
                    "goals": 5,
                    "assists": 3,
                    "xg": 4.0,
                },
                {
                    "name": "RolePlayer",
                    "position": "CM",
                    "minutes_played": 90,
                    "goals": 0,
                    "assists": 0,
                    "xg": 0.1,
                },
            ],
            "scorers": [{"player": "Star", "goals": 5}],
            "assisters": [{"player": "Star", "assists": 3}],
        }
    ]
    players = _build_key_players(matches)
    assert len(players) == 2
    assert players[0].name == "Star"
    assert players[0].threat_score > players[1].threat_score


def test_classify_pressing():
    assert _classify_pressing([{"ppda": 6}, {"ppda": 7}]) == "high intensity"
    assert _classify_pressing([{"ppda": 10}, {"ppda": 11}]) == "moderate"
    assert _classify_pressing([{"ppda": 15}, {"ppda": 16}]) == "deep / low block"
    assert _classify_pressing([]) == "unknown"


def test_classify_build_up():
    matches = [{"build_up_style": "short"}, {"build_up_style": "short"}, {"build_up_style": "long"}]
    assert _classify_build_up(matches) == "short"
    assert _classify_build_up([]) == "mixed"


def test_set_piece_tendencies():
    matches = [
        _sample_match({"corners_for": 8, "set_piece_threat": 0.35, "set_piece_conceded": 0.3})
    ]
    tend = _detect_set_piece_tendencies(matches)
    assert len(tend) >= 2
    assert any("high" in t.lower() for t in tend)


def test_set_piece_tendencies_empty():
    tend = _detect_set_piece_tendencies(
        [{"corners_for": 2, "set_piece_threat": 0.05, "set_piece_conceded": 0.02}]
    )
    assert any("No significant" in t for t in tend)


def test_predict_scoreline():
    matches = [_sample_match() for _ in range(3)]
    sp = _predict_scoreline(matches)
    assert sp.home_score >= 0
    assert sp.away_score >= 0
    assert abs(sp.home_win_prob + sp.draw_prob + sp.away_win_prob - 1.0) < 0.01


def test_predict_scoreline_empty():
    sp = _predict_scoreline([])
    assert sp.home_score == 0 and sp.away_score == 0


def test_generate_dossier():
    matches = [_sample_match() for _ in range(5)]
    dossier = generate_dossier("Test Opponent", matches, "Our Team")
    assert dossier.opponent_name == "Test Opponent"
    assert dossier.matches_analyzed == 5
    assert len(dossier.formations) > 0
    assert len(dossier.key_players) > 0
    assert len(dossier.strengths) > 0 or len(dossier.weaknesses) > 0


def test_dossier_to_dict():
    matches = [_sample_match() for _ in range(3)]
    dossier = generate_dossier("FC Opponent", matches)
    d = dossier.to_dict()
    assert d["opponent_name"] == "FC Opponent"
    assert "formations" in d
    assert "key_players" in d
    assert "scoreline_prediction" in d


def test_dossier_to_markdown():
    matches = [_sample_match() for _ in range(3)]
    dossier = generate_dossier("Markdown Test", matches)
    md = dossier.to_markdown()
    assert "# Opponent Dossier: Markdown Test" in md
    assert "Formation Tendencies" in md
    assert "Key Players" in md
    assert "Scoreline Prediction" in md
    assert "Recommended Tactics" in md


def test_dossier_with_high_possession_strength():
    matches = [_sample_match({"possession_pct": 68.0}) for _ in range(3)]
    dossier = generate_dossier("Possession Team", matches)
    assert any("possession" in s.lower() for s in dossier.strengths)


def test_dossier_with_low_possession_weakness():
    matches = [_sample_match({"possession_pct": 35.0}) for _ in range(3)]
    dossier = generate_dossier("Direct Team", matches)
    assert any("possession" in w.lower() for w in dossier.weaknesses)


def test_dossier_with_high_press():
    matches = [_sample_match({"ppda": 5.0}) for _ in range(3)]
    dossier = generate_dossier("Press Team", matches)
    assert any("pressing" in s.lower() for s in dossier.strengths)


def test_dossier_with_set_piece_vulnerability():
    matches = [_sample_match({"set_piece_conceded": 0.35}) for _ in range(3)]
    dossier = generate_dossier("SP Weak", matches)
    assert any("set piece" in w.lower() for w in dossier.weaknesses)


def test_formation_tendency_to_dict():
    ft = FormationTendency(formation="4-4-2", count=3, percentage=60.0, context="all")
    d = ft.__dict__
    assert d["formation"] == "4-4-2"


def test_scoreline_prediction_default():
    sp = ScorelinePrediction()
    assert sp.home_score == 0
    assert sp.draw_prob == 0.0


def test_generate_dossier_empty_matches():
    dossier = generate_dossier("Unknown", [])
    assert dossier.matches_analyzed == 0
    assert len(dossier.strengths) > 0
