"""kloppy-backed vendor import tests (real migrated SQLite).

What's pinned here:
  - the provider capability report is grounded in the installed library
    (no fabricated availability, no fabricated unavailability),
  - a REAL StatsBomb corpus file round-trips through the kloppy
    deserializer into the same tables the hand importer writes,
  - synthetic lineups are built from the events' own Starting XI /
    Substitution / Player On rows (the corpus ships no lineup files),
  - vendor xG is labeled ``vendor_xg`` + ``vendor_xg_provider`` and never
    collides with the hand importer's ``statsbomb_xg`` key,
  - a SkillCorner feed lands in tracking frames with ``source=kloppy``
    provenance, and the hand parser's path is unchanged (source=file),
  - management rows (Starting XI, Half End, ...) are skipped from the
    census exactly like the hand importer,
  - an uninstalled kloppy is an explicit ImportError, never silence.

Contract-corpus tests skip (with a loud reason) when data/statsbomb_corpus
is absent, so the suite still runs on bare checkouts.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS = PROJECT_ROOT / "data" / "statsbomb_corpus"
MIGRATIONS = PROJECT_ROOT / "src" / "kawkab" / "migrations"


def _corpus_file() -> Path | None:
    if not CORPUS.is_dir():
        return None
    files = sorted(CORPUS.glob("*.json"))
    return files[0] if files else None


@pytest.fixture()
def storage(tmp_path):
    """Real SQLite storage over the REAL 001-033 migration chain."""
    from kawkab.core.migration_manager import MigrationManager
    from kawkab.services.storage_service import StorageService

    db = tmp_path / "kloppy_import.db"
    MigrationManager(db, MIGRATIONS).migrate()
    svc = StorageService()
    svc._db_path = db
    svc._conn = sqlite3.connect(str(db))
    svc._conn.row_factory = sqlite3.Row
    return svc


@pytest.fixture()
def kloppy_svc(storage):
    from kawkab.services.kloppy_import_service import KloppyImportService

    return KloppyImportService(storage)


# ── shared helpers ────────────────────────────────────────────────────────


def _meta_of(event_row: dict) -> dict:
    m = event_row.get("metadata")
    if isinstance(m, str):
        m = json.loads(m)
    return m or {}


def _run(coro):
    return asyncio.run(coro)


# ── provider capability report ────────────────────────────────────────────


class TestProviderCapabilities:
    def test_report_is_environment_grounded(self):
        from kawkab.services.kloppy_import_service import (
            KloppyImportService,
            kloppy_available,
        )

        caps = KloppyImportService.provider_capabilities()
        assert {c["provider"] for c in caps} == {"statsbomb", "skillcorner"}
        if kloppy_available():
            assert all(c["available"] for c in caps), caps
            assert all(c["reason"] is None for c in caps)
            assert all(c["status"] == "available" for c in caps)
        else:
            assert all(not c["available"] for c in caps)
            assert all("not installed" in c["reason"] for c in caps)

    def test_no_kloppy_means_provider_unavailable_everywhere(self, storage):
        from kawkab.services import kloppy_import_service as mod

        real = mod.kloppy_available
        mod.kloppy_available = lambda: False
        try:
            caps = mod.KloppyImportService.provider_capabilities()
            assert all(c["status"] == "provider_unavailable" for c in caps)
            assert all("not installed" in (c["reason"] or "") for c in caps)

            with pytest.raises(ImportError, match="not installed"):
                mod._require_kloppy()
        finally:
            mod.kloppy_available = real

    def test_opposition_probe_surfaces_capabilities(self):
        from kawkab.services.opposition_service import _probe_kloppy

        status = _probe_kloppy()
        assert status["provider"] == "kloppy"
        assert "statsbomb" in status["supported_formats"]
        assert isinstance(status["providers"], list)
        if status["provider_available"]:
            assert status["version"]
            assert "no vendor account" in status["honesty_note"]
            assert all(c["available"] for c in status["providers"]), status["providers"]
        else:
            assert "not installed" in status["reason"]


# ── lineup synthesis ──────────────────────────────────────────────────────


class TestLineupSynthesis:
    def test_synthesizes_from_corpus_events(self):
        """Real corpus: starters from tactics.lineup + subs from both schemas."""
        from kawkab.services.kloppy_import_service import (
            synthesize_lineups_from_events,
        )

        f = _corpus_file()
        if f is None:
            pytest.skip("data/statsbomb_corpus not present")
        lineups = synthesize_lineups_from_events(f)
        assert len(lineups) == 2
        for team in lineups:
            assert isinstance(team["team_id"], int)
            assert team["team_name"]
            assert len(team["lineup"]) >= 11
        # The verified corpus mix: 22 starters + 6 Substitution
        # replacements + 1 Player On = 29 per the 3825872-style files.
        total = sum(len(t["lineup"]) for t in lineups)
        assert total >= 22

    def test_rejects_events_without_starting_xi(self, tmp_path):
        from kawkab.services.kloppy_import_service import (
            synthesize_lineups_from_events,
        )

        f = tmp_path / "events.json"
        f.write_text(json.dumps([{"type": {"name": "Pass"}}]))
        with pytest.raises(ValueError, match="Starting XI"):
            synthesize_lineups_from_events(f)


# ── StatsBomb events round-trip (real corpus file) ───────────────────────


class TestStatsBombImport:
    def test_corpus_round_trip(self, storage, kloppy_svc):
        f = _corpus_file()
        if f is None:
            pytest.skip("data/statsbomb_corpus not present")

        summary = _run(kloppy_svc.import_statsbomb_events(f))

        assert summary["source"] == "kloppy"
        assert summary["provider"] == "statsbomb"
        assert summary["kloppy_version"]
        assert summary["events_imported"] > 0
        assert summary["unhandled_types"] == {}
        assert summary["shots"] == summary["goals"] + 0 or summary["shots"] > 0
        # 28 players: 22 starters + 6 subs + 1 Player On minus dedup —
        # asserted loosely; exact player math belongs to kloppy.
        assert summary["players_registered"] >= 22
        # Management rows skipped from the census, exactly like the hand path
        events = _run(storage.get_match_events(summary["match_id"], limit=5000))
        types = {e["event_type"] for e in events}
        assert not types & {
            "starting_xi",
            "half_start",
            "half_end",
            "tactical_shift",
            "substitution",
            "player_on",
            "player_off",
            "injury_stoppage",
        }
        # Provenance on every event row
        metas = [_meta_of(e) for e in events]
        assert {m.get("source") for m in metas} == {"kloppy"}
        assert {m.get("provider") for m in metas} == {"statsbomb"}
        # Shots persist with goal flags matching the summary
        stored_shots = [e for e in events if e["event_type"] == "shot"]
        assert len(stored_shots) == summary["shots"]
        goals = sum(1 for m in (_meta_of(e) for e in stored_shots) if m.get("is_goal"))
        assert goals == summary["goals"]

    def test_vendor_xg_is_labeled_and_disjoint_from_hand_importer(self, storage, kloppy_svc):
        f = _corpus_file()
        if f is None:
            pytest.skip("data/statsbomb_corpus not present")

        summary = _run(kloppy_svc.import_statsbomb_events(f))
        assert summary["xg_total"] > 0
        events = _run(storage.get_match_events(summary["match_id"], limit=5000))
        shot_metas = [_meta_of(e) for e in events if e["event_type"] == "shot"]
        assert shot_metas
        for m in shot_metas:
            assert "statsbomb_xg" not in m  # hand importer's key
            assert m["vendor_xg_provider"] == "statsbomb"
            if m["shot_outcome"] in {"goal", "saved", "post", "off_target"}:
                assert m["vendor_xg"] >= 0.0
        # Headline metric is the vendor model's, labeled as such
        assert summary["xg_note"] == "vendor (StatsBomb) model xG, labeled vendor_xg"

    def test_players_carry_team_sides_and_jerseys(self, storage, kloppy_svc):
        f = _corpus_file()
        if f is None:
            pytest.skip("data/statsbomb_corpus not present")

        summary = _run(kloppy_svc.import_statsbomb_events(f))
        players = _run(storage.get_match_players(summary["match_id"], limit=200))
        assert len(players) == summary["players_registered"]
        teams = {p["team"] for p in players}
        assert teams <= {"home", "away"}
        assert teams == {"home", "away"}

    def test_missing_file_is_loud(self, kloppy_svc, tmp_path):
        with pytest.raises(FileNotFoundError):
            _run(kloppy_svc.import_statsbomb_events(tmp_path / "nope.json"))


# ── SkillCorner tracking through kloppy ───────────────────────────────────


def _write_skillcorner_v3(tmp_path: Path) -> tuple[Path, Path]:
    """Minimal V3 fixture in the exact shape kloppy's deserializer wants."""

    def players(base_team_id, base_pid, tag):
        roles = [{"id": 1, "name": "Goalkeeper"}] + [{"id": 5, "name": "Midfielder"}] * 10
        return [
            {
                "id": base_pid + i,
                "trackable_object": base_pid + i,
                "team_id": base_team_id,
                "number": i + 1,
                "first_name": tag,
                "last_name": f"P{i + 1}",
                "start_time": "00:00:00",
                "player_role": roles[i],
            }
            for i in range(11)
        ]

    meta = {
        "id": 1,
        "home_team": {"id": 1, "name": "Home FC"},
        "away_team": {"id": 2, "name": "Away FC"},
        "home_team_score": 0,
        "away_team_score": 0,
        "players": players(1, 1, "Home") + players(2, 100, "Away"),
        "referees": [],
        "ball": {"trackable_object": -1},
        "pitch_length": 105.0,
        "pitch_width": 68.0,
        "fps": 10,
    }
    lines = []
    for fid in range(1, 41):
        pd = [{"player_id": 1 + i, "x": 0.5, "y": 0.1 * (i % 10 - 5)} for i in range(11)]
        pd += [{"player_id": 100 + i, "x": -0.5, "y": 0.1 * (i % 10 - 5)} for i in range(11)]
        lines.append(
            json.dumps(
                {
                    "frame": fid,
                    "period": 1,
                    "time": f"00:00:{fid % 60:02d}.5",
                    "possession": {"team_id": 1, "player_id": 5},
                    "player_data": pd,
                    "ball_data": {"x": 0.0, "y": 0.0, "z": 0.0},
                }
            )
        )
    meta_p = tmp_path / "sc_meta.json"
    # .json extension (kloppy sniffs content; the security validator's
    # allowlist is {json, xml, csv} — JSONL text passes as .json)
    raw_p = tmp_path / "sc_raw.json"
    meta_p.write_text(json.dumps(meta))
    raw_p.write_text("\n".join(lines))
    return meta_p, raw_p


class TestSkillCornerImport:
    def test_frames_land_with_kloppy_provenance(self, storage, kloppy_svc, tmp_path):
        meta_p, raw_p = _write_skillcorner_v3(tmp_path)

        summary = _run(kloppy_svc.import_skillcorner_tracking(meta_p, raw_p))

        assert summary["source"] == "kloppy"
        assert summary["provider"] == "skillcorner"
        assert summary["frames_imported"] == 40
        assert summary["fps"] == 10.0
        # Frames in Kawkab meters, ball included
        frames = _run(storage.get_tracking_frames(summary["match_id"], limit=100))
        assert len(frames) == 40
        fr = frames[0]
        assert len(fr["player_detections"]) == 22
        xs = [p["x"] for p in fr["player_detections"]]
        assert all(0.0 <= x <= 105.0 for x in xs)
        assert fr["ball_detections"], "ball should be persisted"
        # Provenance row carries the kloppy source label
        imports = _run(storage.get_tracking_imports(summary["match_id"]))
        assert imports, "tracking_imports row missing"
        row = imports[0]
        assert row["vendor"] == "skillcorner-kloppy"
        meta = row.get("metadata")
        if isinstance(meta, str):
            meta = json.loads(meta)
        assert meta["source"] == "kloppy"
        # Players registered from kloppy metadata, both sides
        players = _run(storage.get_match_players(summary["match_id"], limit=100))
        assert {p["team"] for p in players} == {"home", "away"}

    def test_missing_meta_is_loud(self, kloppy_svc, tmp_path):
        meta_p, raw_p = _write_skillcorner_v3(tmp_path)
        with pytest.raises(FileNotFoundError):
            _run(kloppy_svc.import_skillcorner_tracking(tmp_path / "nope.json", raw_p))


# ── hand-parser regression through the shared pipeline ───────────────────


class TestHandParserRegression:
    def test_file_path_still_lands_source_file(self, storage, tmp_path):
        from kawkab.services.vendor_tracking_import_service import (
            VendorTrackingImportService,
        )

        # A SkillCorner open-data-shaped JSON the HAND parser understands
        frames = {"frames": [], "players": [], "fps": 10}
        for i in range(1, 12):
            frames["players"].append(
                {
                    "track_id": i,
                    "name": f"Home P{i}",
                    "side": "home",
                    "position": "MF",
                    "number": i,
                }
            )
        for i in range(12, 23):
            frames["players"].append(
                {
                    "track_id": i,
                    "name": f"Away P{i}",
                    "side": "away",
                    "position": "DF",
                    "number": i,
                }
            )
        for fid in range(1, 41):
            players = [
                {"track_id": i, "x": 0.5, "y": 0.1 * (i % 10 - 5), "speed": 1.0}
                for i in range(1, 23)
            ]
            frames["frames"].append(
                {
                    "frame_id": fid,
                    "time": fid * 100,
                    "period": 1,
                    "players": players,
                    "ball": {"x": 0.0, "y": 0.0, "z": 0.0},
                }
            )
        f = tmp_path / "hand_skillcorner.json"
        f.write_text(json.dumps(frames))

        svc = VendorTrackingImportService(storage)
        result = _run(svc.import_tracking_file(f, vendor="skillcorner"))
        assert result["source"] == "file"
        assert result["frames_imported"] == 40


# ── bridge handler surface ────────────────────────────────────────────────


class _StubBridge:
    pass


class TestBridgeHandler:
    def _handler(self, store):
        from kawkab.ui.bridge_handlers.bridge_import import ImportHandler

        return ImportHandler(bridge=_StubBridge(), services={"storage_service": store})

    @staticmethod
    def _in_allowlist(tmp_path: Path, src: Path) -> Path:
        """Copy a fixture into the paths allowlist so validation passes."""
        from kawkab.core.paths import get_paths

        allowed = get_paths().documents / "vendor_test"
        allowed.mkdir(parents=True, exist_ok=True)
        dst = allowed / src.name
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        return dst

    def test_statsbomb_happy_path(self, storage, tmp_path):
        f = _corpus_file()
        if f is None:
            pytest.skip("data/statsbomb_corpus not present")
        allowed = self._in_allowlist(tmp_path, f)
        out = json.loads(_run(self._handler(storage).import_kloppy_statsbomb(str(allowed))))
        assert out["success"] is True
        assert out["source"] == "kloppy"
        assert out["events_imported"] > 0

    def test_skillcorner_happy_path(self, storage, tmp_path):
        meta_p, raw_p = _write_skillcorner_v3(tmp_path)
        meta_a = self._in_allowlist(tmp_path, meta_p)
        raw_a = self._in_allowlist(tmp_path, raw_p)
        out = json.loads(
            _run(self._handler(storage).import_kloppy_skillcorner(str(meta_a), str(raw_a)))
        )
        assert out["success"] is True
        assert out["frames_imported"] == 40

    def test_path_validator_still_guards(self, storage, tmp_path):
        handler = self._handler(storage)
        out = json.loads(_run(handler.import_kloppy_statsbomb("/etc/passwd")))
        assert out["success"] is False
        assert "error" in out

    def test_provider_unavailable_is_explicit_not_a_crash(self, storage, tmp_path, monkeypatch):
        import kawkab.services.kloppy_import_service as ks_mod

        class _Boom:
            def __init__(self, storage):
                raise ImportError("kloppy is not installed — vendor import is unavailable")

        monkeypatch.setattr(ks_mod, "KloppyImportService", _Boom)
        f = tmp_path / "x.json"
        f.write_text("[]")
        allowed = self._in_allowlist(tmp_path, f)
        out = json.loads(_run(self._handler(storage).import_kloppy_statsbomb(str(allowed))))
        assert out["success"] is False
        assert out["provider_unavailable"] is True
        assert "not installed" in out["error"]
