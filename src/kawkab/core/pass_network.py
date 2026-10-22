"""Pass network analysis for football analytics.

Computes passing connection matrices, network metrics, and
key passing pairings per team.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PassNetworkResult:
    connection_matrix: dict[str, Any]
    strongest_links: list[dict[str, Any]]
    centrality: dict[int, float]
    betweenness: dict[int, float]
    density: float
    eigenvector_centrality: dict[str, float]


def compute_pass_network(
    events: list[dict[str, Any]],
    player_teams: dict[int, str] | None = None,
    team: str | None = None,
    min_passes: int = 0,
) -> PassNetworkResult:
    pn = PassNetwork(min_passes)
    pn.build(events, player_teams)
    return PassNetworkResult(
        connection_matrix=pn.get_connection_matrix(team),
        strongest_links=pn.get_strongest_links(team) if team else [],
        centrality=pn.compute_centrality(team) if team else {},
        betweenness=pn.compute_betweenness(team) if team else {},
        density=pn.compute_network_density(team) if team else 0.0,
        eigenvector_centrality=pn.compute_eigenvector_centrality(team) if team else {},
    )


class PassNetwork:
    """Pass network analysis for a single match.

    Builds a directed graph of passes between players and computes
    network-level metrics.

    Args:
        min_passes: Minimum passes for an edge to be included.
    """

    def __init__(self, min_passes: int = 0) -> None:
        self.min_passes = min_passes
        self._edges: dict[tuple[int, int], dict[str, int]] = {}
        self._player_teams: dict[int, str] = {}

    def build(
        self,
        events: list[dict[str, Any]],
        player_teams: dict[int, str] | None = None,
    ) -> None:
        raw: dict[tuple[int, int], dict[str, int]] = defaultdict(
            lambda: {"attempted": 0, "completed": 0}
        )
        self._player_teams = player_teams or {}

        for ev in events:
            if ev.get("type") != "pass":
                continue
            src = ev.get("from_track_id")
            dst = ev.get("to_track_id")
            if src is None or dst is None:
                continue
            key = (src, dst)
            raw[key]["attempted"] += 1
            if ev.get("completed", True):
                raw[key]["completed"] += 1

        self._edges = {k: v for k, v in raw.items() if v["attempted"] >= self.min_passes}

    def get_connection_matrix(self, team: str | None = None) -> dict[str, Any]:
        nodes: set[int] = set()
        edges_out: list[dict[str, Any]] = []

        for (src, dst), stats in self._edges.items():
            if team is not None:
                src_team = self._player_teams.get(src)
                if src_team != team:
                    continue
            nodes.add(src)
            nodes.add(dst)
            completion_pct = (
                round((stats["completed"] / stats["attempted"]) * 100, 1)
                if stats["attempted"] > 0
                else 0.0
            )
            edges_out.append(
                {
                    "source": src,
                    "target": dst,
                    "attempted": stats["attempted"],
                    "completed": stats["completed"],
                    "completion_pct": completion_pct,
                }
            )

        return {
            "nodes": [{"id": n} for n in sorted(nodes)],
            "edges": sorted(edges_out, key=lambda e: -e["attempted"]),
        }

    def get_strongest_links(self, team: str, top_n: int = 5) -> list[dict[str, Any]]:
        links = []
        for (src, dst), stats in self._edges.items():
            src_team = self._player_teams.get(src)
            if src_team != team:
                continue
            completion_pct = (
                round((stats["completed"] / stats["attempted"]) * 100, 1)
                if stats["attempted"] > 0
                else 0.0
            )
            links.append(
                {
                    "source": src,
                    "target": dst,
                    "attempted": stats["attempted"],
                    "completed": stats["completed"],
                    "completion_pct": completion_pct,
                }
            )

        links.sort(key=lambda x: -x["attempted"])
        return links[:top_n]

    def compute_centrality(self, team: str) -> dict[int, float]:
        degree: dict[int, int] = defaultdict(int)
        max_degree = 0

        for (src, dst), stats in self._edges.items():
            src_team = self._player_teams.get(src)
            if src_team != team:
                continue
            total = stats["attempted"]
            degree[src] += total
            degree[dst] += total
            max_degree = max(max_degree, degree[src], degree[dst])

        if max_degree == 0:
            return {}

        return {tid: round(val / max_degree, 4) for tid, val in degree.items()}

    def compute_network_density(self, team: str, n_players: int = 11) -> float:
        team_edges = set()
        for (src, dst), stats in self._edges.items():
            if self._player_teams.get(src) == team and stats["attempted"] > 0:
                team_edges.add((src, dst))

        max_possible = n_players * (n_players - 1)
        if max_possible == 0:
            return 0.0

        return round(len(team_edges) / max_possible, 4)

    def compute_betweenness(self, team: str) -> dict[int, float]:
        """Compute betweenness centrality using BFS on shortest paths.

        Higher betweenness = more passes flow through this player,
        indicating their role as a key connector in the network.

        Args:
            team: "home" or "away"

        Returns:
            dict mapping track_id -> betweenness score (normalized 0-1)
        """
        adjacency: dict[int, list[int]] = defaultdict(list)
        for (src, dst), stats in self._edges.items():
            if self._player_teams.get(src) != team:
                continue
            if stats["attempted"] == 0:
                continue
            adjacency[src].append(dst)
            adjacency[dst].append(src)

        players = list(adjacency.keys())
        if len(players) < 2:
            return {}

        betweenness: dict[int, float] = defaultdict(float)

        for s in players:
            stack: list[int] = []
            paths: list[list[int]] = [[] for _ in range(max(players) + 1 if players else 0)]
            sigma: dict[int, int] = defaultdict(int)
            d: dict[int, int] = {}
            for v in players:
                if paths and v < len(paths):
                    paths[v] = []
            sigma[s] = 1
            d[s] = 0
            stack.append(s)
            queue = [s]
            idx = 0
            while idx < len(queue):
                v = queue[idx]
                idx += 1
                stack.append(v)
                for w in adjacency.get(v, []):
                    if w not in d:
                        queue.append(w)
                        d[w] = d[v] + 1
                    if d[w] == d[v] + 1:
                        sigma[w] += sigma[v]
                        if paths and w < len(paths):
                            paths[w].append(v)

            delta: dict[int, float] = defaultdict(float)
            while stack:
                w = stack.pop()
                for v in paths[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                if w != s:
                    betweenness[w] += delta[w]

        if not betweenness:
            return {}

        max_val = max(betweenness.values())
        if max_val == 0:
            return dict.fromkeys(betweenness, 0.0)

        return {tid: round(val / max_val, 4) for tid, val in betweenness.items()}

    def compute_eigenvector_centrality(
        self, team: str, max_iter: int = 100, tol: float = 1e-6
    ) -> dict[str, float]:
        player_set: set[int] = set()
        edge_counts: dict[tuple[int, int], int] = {}
        for (src, dst), stats in self._edges.items():
            if self._player_teams.get(src) != team:
                continue
            player_set.add(src)
            player_set.add(dst)
            key = (src, dst) if src < dst else (dst, src)
            edge_counts[key] = edge_counts.get(key, 0) + stats["attempted"]

        players = sorted(player_set)
        n = len(players)
        if n == 0:
            return {}

        idx_map = {p: i for i, p in enumerate(players)}
        A = np.zeros((n, n), dtype=np.float64)
        for (src, dst), cnt in edge_counts.items():
            i, j = idx_map[src], idx_map[dst]
            A[i, j] = cnt
            A[j, i] = cnt

        v = np.ones(n, dtype=np.float64) / np.sqrt(n)
        for _ in range(max_iter):
            v_new = A @ v
            norm = np.linalg.norm(v_new)
            if norm == 0:
                break
            v_new = v_new / norm
            if np.linalg.norm(v_new - v) < tol:
                v = v_new
                break
            v = v_new

        v_min, v_max = v.min(), v.max()
        v = (v - v_min) / (v_max - v_min) if v_max > v_min else np.zeros_like(v)

        return {str(pid): round(float(v[idx]), 4) for pid, idx in idx_map.items()}

    def to_team_report(self, team: str) -> dict[str, Any]:
        """Generate a complete pass network report for a team."""
        return {
            "connection_matrix": self.get_connection_matrix(team),
            "strongest_links": self.get_strongest_links(team),
            "centrality": self.compute_centrality(team),
            "betweenness": self.compute_betweenness(team),
            "density": self.compute_network_density(team),
            "eigenvector_centrality": self.compute_eigenvector_centrality(team),
        }
