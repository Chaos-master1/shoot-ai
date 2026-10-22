"""Tests for pass network analysis."""

from kawkab.core.pass_network import PassNetwork, PassNetworkResult, compute_pass_network


class TestPassNetwork:
    def test_build_empty_events(self):
        pn = PassNetwork()
        pn.build([])
        matrix = pn.get_connection_matrix()
        assert matrix["nodes"] == []
        assert matrix["edges"] == []

    def test_build_simple_pass(self):
        pn = PassNetwork()
        events = [
            {
                "type": "pass",
                "from_track_id": 1,
                "to_track_id": 2,
                "completed": True,
                "team": "home",
            },
        ]
        pn.build(events, {1: "home", 2: "home"})
        matrix = pn.get_connection_matrix()
        assert len(matrix["edges"]) == 1
        assert matrix["edges"][0]["attempted"] == 1
        assert matrix["edges"][0]["completion_pct"] == 100.0

    def test_multiple_passes_same_edge(self):
        pn = PassNetwork()
        events = [
            {"type": "pass", "from_track_id": 1, "to_track_id": 2, "completed": True},
            {"type": "pass", "from_track_id": 1, "to_track_id": 2, "completed": False},
            {"type": "pass", "from_track_id": 1, "to_track_id": 2, "completed": True},
        ]
        pn.build(events, {1: "home", 2: "home"})
        matrix = pn.get_connection_matrix("home")
        assert matrix["edges"][0]["attempted"] == 3
        assert matrix["edges"][0]["completed"] == 2
        assert matrix["edges"][0]["completion_pct"] == 66.7

    def test_min_passes_filter(self):
        pn = PassNetwork(min_passes=3)
        events = [
            {"type": "pass", "from_track_id": 1, "to_track_id": 2, "completed": True},
            {"type": "pass", "from_track_id": 1, "to_track_id": 2, "completed": True},
        ]
        pn.build(events, {1: "home", 2: "home"})
        matrix = pn.get_connection_matrix()
        assert len(matrix["edges"]) == 0

    def test_team_filter(self):
        pn = PassNetwork()
        events = [
            {"type": "pass", "from_track_id": 1, "to_track_id": 2, "completed": True},
            {"type": "pass", "from_track_id": 3, "to_track_id": 4, "completed": True},
        ]
        pn.build(events, {1: "home", 2: "home", 3: "away", 4: "away"})
        home_matrix = pn.get_connection_matrix("home")
        assert len(home_matrix["edges"]) == 1
        assert home_matrix["edges"][0]["source"] == 1
        away_matrix = pn.get_connection_matrix("away")
        assert len(away_matrix["edges"]) == 1

    def test_strongest_links(self):
        pn = PassNetwork()
        events = [
            {"type": "pass", "from_track_id": 1, "to_track_id": 2, "completed": True},
            {"type": "pass", "from_track_id": 1, "to_track_id": 2, "completed": True},
            {"type": "pass", "from_track_id": 1, "to_track_id": 2, "completed": True},
            {"type": "pass", "from_track_id": 1, "to_track_id": 3, "completed": True},
        ]
        pn.build(events, {1: "home", 2: "home", 3: "home"})
        links = pn.get_strongest_links("home", top_n=2)
        assert len(links) == 2
        assert links[0]["source"] == 1
        assert links[0]["target"] == 2

    def test_centrality(self):
        pn = PassNetwork()
        events = [
            {"type": "pass", "from_track_id": 1, "to_track_id": 2, "completed": True},
            {"type": "pass", "from_track_id": 1, "to_track_id": 3, "completed": True},
            {"type": "pass", "from_track_id": 2, "to_track_id": 1, "completed": True},
        ]
        pn.build(events, {1: "home", 2: "home", 3: "home"})
        cent = pn.compute_centrality("home")
        assert 1 in cent
        assert cent[1] >= cent[2]
        assert cent[2] >= cent[3]

    def test_density(self):
        pn = PassNetwork()
        events = [
            {"type": "pass", "from_track_id": 1, "to_track_id": 2, "completed": True},
            {"type": "pass", "from_track_id": 2, "to_track_id": 1, "completed": True},
        ]
        pn.build(events, {1: "home", 2: "home"})
        density = pn.compute_network_density("home", n_players=3)
        assert density == 0.3333

    def test_to_team_report(self):
        pn = PassNetwork()
        events = [
            {"type": "pass", "from_track_id": 1, "to_track_id": 2, "completed": True},
        ]
        pn.build(events, {1: "home", 2: "home"})
        report = pn.to_team_report("home")
        assert "connection_matrix" in report
        assert "strongest_links" in report
        assert "centrality" in report
        assert "density" in report

    def test_get_connection_matrix_no_team_filter_empty_teams(self):
        pn = PassNetwork()
        events = [
            {"type": "pass", "from_track_id": 1, "to_track_id": 2, "completed": True},
        ]
        pn.build(events, {})
        matrix = pn.get_connection_matrix()
        assert len(matrix["edges"]) == 1

    def test_disconnected_graph(self):
        pn = PassNetwork()
        events = [
            {"type": "pass", "from_track_id": 1, "to_track_id": 2, "completed": True},
            {"type": "pass", "from_track_id": 3, "to_track_id": 4, "completed": True},
        ]
        pn.build(events, {1: "home", 2: "home", 3: "home", 4: "home"})
        matrix = pn.get_connection_matrix("home")
        assert len(matrix["edges"]) == 2
        assert matrix["nodes"][0]["id"] == 1
        assert matrix["nodes"][2]["id"] == 3

    def test_centrality_no_edges(self):
        pn = PassNetwork()
        cent = pn.compute_centrality("home")
        assert cent == {}

    def test_density_zero_players(self):
        pn = PassNetwork()
        density = pn.compute_network_density("home", n_players=0)
        assert density == 0.0

    def test_all_passes_failed(self):
        pn = PassNetwork()
        events = [
            {"type": "pass", "from_track_id": 1, "to_track_id": 2, "completed": False},
            {"type": "pass", "from_track_id": 1, "to_track_id": 2, "completed": False},
        ]
        pn.build(events, {1: "home", 2: "home"})
        matrix = pn.get_connection_matrix("home")
        assert matrix["edges"][0]["completion_pct"] == 0.0
        assert matrix["edges"][0]["completed"] == 0

    def test_eigenvector_centrality_present_in_result(self):
        result = compute_pass_network(
            [
                {"type": "pass", "from_track_id": 1, "to_track_id": 2, "completed": True},
                {"type": "pass", "from_track_id": 2, "to_track_id": 3, "completed": True},
                {"type": "pass", "from_track_id": 3, "to_track_id": 1, "completed": True},
            ],
            {1: "home", 2: "home", 3: "home"},
            team="home",
        )
        assert isinstance(result, PassNetworkResult)
        assert "eigenvector_centrality" in result.__dataclass_fields__
        assert len(result.eigenvector_centrality) == 3

    def test_eigenvector_centrality_high_degree_player_scores_higher(self):
        pn = PassNetwork()
        events = [
            {"type": "pass", "from_track_id": 1, "to_track_id": 2, "completed": True},
            {"type": "pass", "from_track_id": 1, "to_track_id": 3, "completed": True},
            {"type": "pass", "from_track_id": 1, "to_track_id": 4, "completed": True},
            {"type": "pass", "from_track_id": 2, "to_track_id": 1, "completed": True},
            {"type": "pass", "from_track_id": 3, "to_track_id": 1, "completed": True},
            {"type": "pass", "from_track_id": 4, "to_track_id": 1, "completed": True},
        ]
        pn.build(events, {1: "home", 2: "home", 3: "home", 4: "home"})
        ev = pn.compute_eigenvector_centrality("home")
        assert float(ev["1"]) > float(ev["2"])
        assert float(ev["1"]) > float(ev["3"])
        assert float(ev["1"]) > float(ev["4"])

    def test_eigenvector_centrality_values_between_zero_and_one(self):
        pn = PassNetwork()
        events = [
            {"type": "pass", "from_track_id": 1, "to_track_id": 2, "completed": True},
            {"type": "pass", "from_track_id": 2, "to_track_id": 3, "completed": True},
            {"type": "pass", "from_track_id": 3, "to_track_id": 1, "completed": True},
        ]
        pn.build(events, {1: "home", 2: "home", 3: "home"})
        ev = pn.compute_eigenvector_centrality("home")
        for val in ev.values():
            assert 0.0 <= val <= 1.0
