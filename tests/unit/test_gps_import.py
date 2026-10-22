"""Tests for GPS/IMU import parsers (Catapult, STATSports, Kinexon)."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from kawkab.services.gps_import import (
    compute_session_summary,
    import_gps_file,
    parse_catapult_csv,
    parse_kinexon_json,
    parse_statsports_csv,
)

CATAPULT_CSV = """Time,Speed (m/s),Accel X,Accel Y,Accel Z,Heart Rate,Distance (m),Player Load,Metabolic Power (W/kg),Lat,Lon
00:00:00.000,0.0,0.01,-0.02,0.99,72,0.0,0.0,0.0,51.5,-0.12
00:00:00.100,2.5,0.12,-0.15,1.02,75,0.25,0.5,3.2,51.5,-0.12
00:00:00.200,5.0,0.25,-0.30,1.10,78,0.50,1.0,6.5,51.5,-0.12
00:00:00.300,7.5,0.40,-0.45,1.15,82,0.75,1.5,10.0,51.5,-0.12
"""


STATSPORTS_CSV = """Time (s),Speed (km/h),Accel X (g),Accel Y (g),Accel Z (g),HR (bpm),Distance (m),Player Load,Metabolic Power
0.0,0.0,0.001,-0.002,0.101,72,0.0,0.0,0.0
0.1,9.0,0.012,-0.015,0.104,75,0.25,0.5,3.2
0.2,18.0,0.025,-0.030,0.112,78,0.50,1.0,6.5
0.3,27.0,0.040,-0.045,0.117,82,0.75,1.5,10.0
"""


def _kinexon_sample(t, speed, accel, hr=72, p_load=0.5, x=0, y=0):
    return {
        "timestamp": f"00:00:{t:06.3f}",
        "speed": speed,
        "acceleration": accel,
        "heart_rate": hr,
        "player_load": p_load,
        "position": {"x": x, "y": y},
        "distance": speed * 0.1,
    }


KINEXON_JSON = json.dumps(
    {
        "sessions": [
            {
                "start_time": "2026-07-12T14:00:00",
                "samples": [
                    _kinexon_sample(0.000, 0.0, 0.0, 72, 0.0),
                    _kinexon_sample(0.100, 2.5, 0.5, 75, 0.5),
                    _kinexon_sample(0.200, 5.0, 1.0, 78, 1.0),
                    _kinexon_sample(0.300, 7.5, 1.5, 82, 1.5),
                ],
            }
        ]
    }
)


# ─── Catapult CSV ──────────────────────────────────────────────────────────


def test_catapult_basic():
    samples = parse_catapult_csv(CATAPULT_CSV)
    assert len(samples) == 4
    assert samples[0]["timestamp"] == 0.0
    assert samples[1]["timestamp"] == 0.1
    assert samples[2]["speed_ms"] == 5.0
    assert samples[3]["speed_ms"] == 7.5


def test_catapult_columns():
    samples = parse_catapult_csv(CATAPULT_CSV)
    s = samples[1]
    assert s["heart_rate"] == 75
    assert s["distance"] == 0.25
    assert s["player_load"] == 0.5
    assert s["metabolic_power"] == 3.2
    assert s["lat"] == 51.5
    assert s["lon"] == -0.12
    assert s["accel_x"] == 0.12
    assert s["accel_y"] == -0.15
    assert s["accel_z"] == 1.02


def test_catapult_speed_zone():
    samples = parse_catapult_csv(CATAPULT_CSV)
    # 0.0 m/s -> walking (zone 1)
    assert samples[0]["speed_zone"] == 1
    # 2.5 m/s -> jogging (zone 2)
    assert samples[1]["speed_zone"] == 2
    # 5.0 m/s -> running (zone 3)
    assert samples[2]["speed_zone"] == 3
    # 7.5 m/s -> sprinting (zone 5)
    assert samples[3]["speed_zone"] == 5


def test_catapult_empty():
    """Empty CSV should produce empty list."""
    samples = parse_catapult_csv("")
    assert samples == []


def test_catapult_bytes():
    """Parse bytes as well as string."""
    samples = parse_catapult_csv(CATAPULT_CSV.encode("utf-8"))
    assert len(samples) == 4


# ─── STATSports CSV ────────────────────────────────────────────────────────


def test_statsports_basic():
    samples = parse_statsports_csv(STATSPORTS_CSV)
    assert len(samples) == 4
    assert samples[0]["timestamp"] == 0.0


def test_statsports_speed_conversion():
    """STATSports provides km/h, should convert to m/s."""
    samples = parse_statsports_csv(STATSPORTS_CSV)
    assert samples[1]["speed_ms"] == pytest.approx(2.5, rel=0.01)
    assert samples[2]["speed_ms"] == pytest.approx(5.0, rel=0.01)
    assert samples[3]["speed_ms"] == pytest.approx(7.5, rel=0.01)


def test_statsports_accel_g_conversion():
    """Accel in g should convert to m/s²."""
    samples = parse_statsports_csv(STATSPORTS_CSV)
    s1 = samples[1]
    assert s1["accel_x"] == pytest.approx(0.012 * 9.81, rel=0.01)
    assert s1["accel_y"] == pytest.approx(-0.015 * 9.81, rel=0.01)
    assert s1["accel_z"] == pytest.approx(0.104 * 9.81, rel=0.01)
    assert s1["heart_rate"] == 75
    assert s1["distance"] == 0.25


def test_statsports_empty():
    assert parse_statsports_csv("") == []


# ─── Kinexon JSON ──────────────────────────────────────────────────────────


def test_kinexon_basic():
    samples = parse_kinexon_json(KINEXON_JSON)
    assert len(samples) == 4
    assert samples[0]["timestamp"] == 0.0


def test_kinexon_fields():
    samples = parse_kinexon_json(KINEXON_JSON)
    s1 = samples[1]
    assert s1["speed_ms"] == 2.5
    assert s1["acceleration"] == 0.5
    assert s1["heart_rate"] == 75
    assert s1["player_load"] == 0.5
    assert s1["x_m"] == 0
    assert s1["y_m"] == 0


def test_kinexon_speed_zone():
    samples = parse_kinexon_json(KINEXON_JSON)
    assert samples[0]["speed_zone"] == 1
    assert samples[1]["speed_zone"] == 2
    assert samples[2]["speed_zone"] == 3
    assert samples[3]["speed_zone"] == 5


def test_kinexon_json_bytes():
    samples = parse_kinexon_json(KINEXON_JSON.encode("utf-8"))
    assert len(samples) == 4


def test_kinexon_empty():
    assert parse_kinexon_json("{}") == []


def test_kinexon_single_session():
    data = json.dumps(
        {
            "start_time": "2026-07-12T14:00:00",
            "samples": [{"timestamp": 0.0, "speed": 1.0, "distance": 0.1}],
        }
    )
    samples = parse_kinexon_json(data)
    assert len(samples) == 1


# ─── Session Summary ───────────────────────────────────────────────────────


def test_compute_summary():
    samples = parse_catapult_csv(CATAPULT_CSV)
    summary = compute_session_summary(samples)
    assert summary["total_distance_m"] > 0
    assert summary["sample_count"] == 4
    assert summary["max_speed_kmh"] > 0
    assert summary["avg_speed_kmh"] > 0
    assert "distance_by_zone" in summary


def test_compute_summary_no_heart_rate():
    hr_csv = """Time,Speed (m/s),Heart Rate,Distance (m)
00:00:00.000,0.0,,0.0
00:00:00.100,2.5,155,0.25
"""
    summary = compute_session_summary(parse_catapult_csv(hr_csv))
    assert summary["max_heart_rate"] == 155
    assert summary["avg_heart_rate"] == 155.0  # only second row has non-None HR
    assert summary["sample_count"] == 2


def test_compute_summary_empty():
    assert compute_session_summary([]) == {}


def test_compute_summary_sprint_count():
    sprint_csv = """Time,Speed (m/s),Distance (m)
00:00:00.000,0.0,0.0
00:00:00.100,8.0,0.8  # sprint
00:00:00.200,9.0,0.9  # sprint
00:00:00.300,2.0,0.2
00:00:00.400,8.5,0.85  # sprint (new burst)
"""
    samples = parse_catapult_csv(sprint_csv)
    summary = compute_session_summary(samples)
    assert summary["sprint_count"] >= 1  # at least one sprint burst detected
    assert summary["acceleration_count"] >= 0
    assert summary["deceleration_count"] >= 0


# ─── Auto-detect ───────────────────────────────────────────────────────────


def test_import_gps_file_catapult(tmp_path):
    f = tmp_path / "catapult_export.csv"
    f.write_text(CATAPULT_CSV)
    samples = import_gps_file(str(f))
    assert len(samples) == 4


def test_import_gps_file_kinexon(tmp_path):
    f = tmp_path / "kinexon_export.json"
    f.write_text(KINEXON_JSON)
    samples = import_gps_file(str(f))
    assert len(samples) == 4


def test_import_gps_file_unsupported(tmp_path):
    f = tmp_path / "data.xyz"
    f.write_text("unknown")
    try:
        import_gps_file(str(f))
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass
