from __future__ import annotations

import pytest

from kawkab.services.data_provider_base import (
    DataProviderRegistry,
    ProviderEvent,
    ProviderMatch,
)
from kawkab.services.opta_importer import OptaF7Importer


class TestDataProviderRegistry:
    def test_register_and_get(self):
        registry = DataProviderRegistry()
        provider = OptaF7Importer()
        registry.register(provider)
        assert registry.get("opta_f7") is provider

    def test_list_providers(self):
        registry = DataProviderRegistry()
        registry.register(OptaF7Importer())
        assert "opta_f7" in registry.list_providers()

    def test_get_unknown_provider(self):
        registry = DataProviderRegistry()
        assert registry.get("unknown") is None

    def test_unregister(self):
        registry = DataProviderRegistry()
        provider = OptaF7Importer()
        registry.register(provider)
        registry.unregister("opta_f7")
        assert registry.get("opta_f7") is None

    def test_same_provider_overwrites(self):
        registry = DataProviderRegistry()
        p1 = OptaF7Importer(data_dir="/a")
        p2 = OptaF7Importer(data_dir="/b")
        registry.register(p1)
        registry.register(p2)
        assert len(registry.list_providers()) == 1
        assert registry.get("opta_f7") is p2


class TestProviderMatch:
    def test_default_values(self):
        m = ProviderMatch(
            match_id="1",
            home_team="A",
            away_team="B",
            competition="C",
            season="S",
            date=__import__("datetime").datetime.now(),
        )
        assert m.home_score is None
        assert m.status == "scheduled"

    def test_with_score(self):
        m = ProviderMatch(
            match_id="1",
            home_team="A",
            away_team="B",
            competition="C",
            season="S",
            date=__import__("datetime").datetime.now(),
            home_score=2,
            away_score=1,
        )
        assert m.home_score == 2
        assert m.away_score == 1

    def test_extra_fields(self):
        e = ProviderEvent(event_id="e1", match_id="m1", timestamp=10.0, type="pass", team="T1")
        assert e.x == 0.0
        assert e.outcome is True
        assert e.extra == {}


class TestOptaImporter:
    def test_provider_name(self):
        imp = OptaF7Importer()
        assert imp.get_provider_name() == "opta_f7"

    def test_rate_limit_info(self):
        imp = OptaF7Importer()
        info = imp.get_rate_limit_info()
        assert info["type"] == "local_file"

    @pytest.mark.asyncio
    async def test_search_matches_returns_empty(self):
        imp = OptaF7Importer()
        result = await imp.search_matches()
        assert result == []

    def test_opta_type_name_known(self):
        assert OptaF7Importer._opta_type_name("1") == "pass"
        assert OptaF7Importer._opta_type_name("11") == "goal"
        assert OptaF7Importer._opta_type_name("22") == "save"

    def test_opta_type_name_unknown(self):
        assert OptaF7Importer._opta_type_name("999") == "unknown_999"

    def test_parse_basic_match_xml(self):
        xml = """<?xml version="1.0"?>
<MatchData uID="m12345">
  <MatchInfo>
    <Date>2024-03-15T20:00:00</Date>
    <Competition><CompetitionName>Premier League</CompetitionName></Competition>
    <Season><SeasonName>2023/2024</SeasonName></Season>
  </MatchInfo>
  <Team TeamRef="Home1">
    <Name>Home Team</Name>
  </Team>
  <Team TeamRef="Away1">
    <Name>Away Team</Name>
  </Team>
</MatchData>"""
        imp = OptaF7Importer()
        match = imp.parse_match_xml(xml)
        assert match.match_id == "12345"
        assert match.home_team == "Home Team"
        assert match.away_team == "Away Team"

    def test_parse_match_with_scores(self):
        xml = """<?xml version="1.0"?>
<MatchData uID="m99">
  <Team TeamRef="Home1">
    <Name>FC Home</Name>
    <TeamStat><Value>3</Value></TeamStat>
  </Team>
  <Team TeamRef="Away1">
    <Name>FC Away</Name>
    <TeamStat><Value>1</Value></TeamStat>
  </Team>
</MatchData>"""
        imp = OptaF7Importer()
        match = imp.parse_match_xml(xml)
        assert match.home_team == "FC Home"
        assert match.away_team == "FC Away"

    def test_parse_events_xml(self):
        xml = """<?xml version="1.0"?>
<MatchData uID="m1">
  <GameEvents>
    <Event id="ev1" type_id="1" team_id="t1" player_id="p1" time_min="10" time_sec="30" x="50" y="30" end_x="80" end_y="40" outcome="1"/>
    <Event id="ev2" type_id="11" team_id="t1" player_id="p2" time_min="25" time_sec="0" x="90" y="50" outcome="1"/>
  </GameEvents>
</MatchData>"""
        imp = OptaF7Importer()
        events = imp.parse_event_xml(xml)
        assert len(events) == 2
        assert events[0].type == "pass"
        assert events[0].team == "t1"
        assert events[0].player == "p1"
        assert abs(events[0].timestamp - 630.0) < 0.1
        assert events[1].type == "goal"

    def test_parse_lineup_xml(self):
        xml = """<?xml version="1.0"?>
<MatchData uID="m1">
  <Team TeamRef="t1">
    <Name>FC Test</Name>
    <Formation>4-4-2</Formation>
    <Player PlayerRef="p1" ShirtNumber="10" Position="Midfielder" Status="Start"/>
    <Player PlayerRef="p2" ShirtNumber="9" Position="Forward" Status="Start"/>
  </Team>
</MatchData>"""
        imp = OptaF7Importer()
        lineups = imp.parse_lineup_xml(xml)
        assert len(lineups) == 1
        assert lineups[0].formation == "4-4-2"
        assert len(lineups[0].players) == 2
        assert lineups[0].players[0]["position"] == "Midfielder"

    def test_empty_xml_returns_graceful(self):
        xml = """<?xml version="1.0"?><MatchData uID="m1"></MatchData>"""
        imp = OptaF7Importer()
        match = imp.parse_match_xml(xml)
        assert match.match_id == "1"
        events = imp.parse_event_xml(xml)
        assert events == []
        lineups = imp.parse_lineup_xml(xml)
        assert lineups == []
