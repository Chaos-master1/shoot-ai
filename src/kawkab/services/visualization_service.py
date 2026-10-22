"""Visualization service - generate professional charts and diagrams.

Generates:
1. Heatmaps - player/team position density, action heatmaps
2. Pass networks - weighted graph of passing connections
3. Pass sonars - polar plot of pass directions and distances
4. Formation diagrams - pitch visualization of detected formations

All outputs are saved as PNG files in the exports directory.
Uses matplotlib, mplsoccer, and networkx for professional-quality visuals.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from kawkab.core.logging import get_logger
from kawkab.core.paths import get_paths

logger = get_logger(__name__)


class VisualizationService:
    """Generates professional football analytics visualizations."""

    def __init__(self) -> None:
        self._exports_dir = get_paths().exports
        self._exports_dir.mkdir(parents=True, exist_ok=True)
        logger.info("VisualizationService initialized")

    async def generate_heatmap(
        self,
        positions: list[tuple[float, float]],
        title: str = "Position Heatmap",
        output_name: str = "heatmap.png",
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        bins: int = 20,
    ) -> Path | None:
        """Generate a 2D heatmap from player/team positions.

        Args:
            positions: List of (x, y) positions in pitch meters
            title: Chart title
            output_name: Output filename
            pitch_length: Pitch length in meters
            pitch_width: Pitch width in meters
            bins: Number of bins for the histogram

        Returns:
            Path to the generated PNG file, or None if generation failed
        """
        try:
            import matplotlib.pyplot as plt
            from daimon_runtime import setup_plot
        except ImportError:
            logger.error("matplotlib not available for visualization")
            return None

        if not positions:
            logger.warning("No positions provided for heatmap")
            return None

        try:
            setup_plot({"runDir": str(self._exports_dir)})

            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.hist2d(xs, ys, bins=bins, cmap="hot", range=[[0, pitch_length], [0, pitch_width]])
            ax.set_xlabel("Pitch Length (m)")
            ax.set_ylabel("Pitch Width (m)")
            ax.set_title(title)
            ax.set_xlim(0, pitch_length)
            ax.set_ylim(0, pitch_width)

            # Draw pitch lines (simplified)
            ax.axhline(pitch_width / 2, color="white", linewidth=1, alpha=0.5)
            ax.axvline(pitch_length / 2, color="white", linewidth=1, alpha=0.5)
            ax.axvline(pitch_length * 0.15, color="white", linewidth=1, alpha=0.3)
            ax.axvline(pitch_length * 0.85, color="white", linewidth=1, alpha=0.3)

            output_path = self._exports_dir / output_name
            plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="black")
            plt.close(fig)
            logger.info(f"Generated heatmap: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Heatmap generation failed: {e}")
            return None

    async def generate_pass_network(
        self,
        pass_events: list[dict[str, Any]],
        player_positions: dict[int, tuple[float, float]],
        title: str = "Pass Network",
        output_name: str = "pass_network.png",
    ) -> Path | None:
        """Generate a pass network visualization.

        Args:
            pass_events: List of pass events with from_track_id and to_track_id
            player_positions: Dict mapping track_id to (x, y) average position
            title: Chart title
            output_name: Output filename
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            from daimon_runtime import setup_plot
        except ImportError:
            logger.error("networkx or matplotlib not available")
            return None

        if not pass_events or not player_positions:
            return None

        try:
            setup_plot({"runDir": str(self._exports_dir)})

            pass_graph = nx.DiGraph()
            edge_weights: defaultdict[tuple[Any, Any], int] = defaultdict(int)

            for event in pass_events:
                if event.get("type") != "pass" or not event.get("completed"):
                    continue
                src = event.get("from_track_id")
                dst = event.get("to_track_id")
                if src is None or dst is None:
                    continue
                edge = (src, dst)
                edge_weights[edge] += 1

            for (src, dst), weight in edge_weights.items():
                pass_graph.add_edge(src, dst, weight=weight)
                if src not in pass_graph.nodes:
                    pass_graph.add_node(src)
                if dst not in pass_graph.nodes:
                    pass_graph.add_node(dst)

            if len(pass_graph.nodes) == 0:
                return None

            # Use actual pitch positions if available, else spring layout
            pos = {
                node: player_positions.get(node, (50, 34))  # default center
                for node in pass_graph.nodes
            }

            # Fallback for missing positions
            if len(pos) < len(pass_graph.nodes):
                spring_pos = nx.spring_layout(pass_graph, seed=42)
                for node in pass_graph.nodes:
                    if node not in pos:
                        pos[node] = (spring_pos[node][0] * 50 + 25, spring_pos[node][1] * 30 + 17)

            fig, ax = plt.subplots(figsize=(12, 8))

            # Draw pitch background
            ax.set_xlim(0, 105)
            ax.set_ylim(0, 68)
            ax.set_facecolor("#2d5a27")
            ax.axhline(34, color="white", linewidth=1, alpha=0.5)
            ax.axvline(52.5, color="white", linewidth=1, alpha=0.5)

            # Draw edges with width proportional to pass count
            max_weight = max(edge_weights.values()) if edge_weights else 1
            for (src, dst), weight in edge_weights.items():
                width = 1 + (weight / max_weight) * 5
                alpha = min(0.3 + (weight / max_weight) * 0.7, 0.9)
                nx.draw_networkx_edges(
                    pass_graph,
                    pos,
                    edgelist=[(src, dst)],
                    width=width,
                    alpha=alpha,
                    edge_color="white",
                    arrows=True,
                    arrowsize=15,
                    ax=ax,
                    connectionstyle="arc3,rad=0.1",
                )

            # Draw nodes
            node_sizes = [300 + edge_weights.get((n, n), 0) * 100 for n in pass_graph.nodes]
            nx.draw_networkx_nodes(
                pass_graph, pos, node_size=node_sizes, node_color="gold", alpha=0.8, ax=ax
            )
            nx.draw_networkx_labels(pass_graph, pos, font_size=10, font_color="black", ax=ax)

            ax.set_title(title, color="white", fontsize=14)
            ax.set_xlabel("Pitch Length (m)", color="white")
            ax.set_ylabel("Pitch Width (m)", color="white")
            ax.tick_params(colors="white")

            output_path = self._exports_dir / output_name
            plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
            plt.close(fig)
            logger.info(f"Generated pass network: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Pass network generation failed: {e}")
            return None

    async def generate_pass_sonar(
        self,
        pass_events: list[dict[str, Any]],
        player_positions: dict[int, tuple[float, float]],
        title: str = "Pass Sonar",
        output_name: str = "pass_sonar.png",
    ) -> Path | None:
        """Generate a pass sonar (polar plot) showing pass directions and distances.

        Uses soccerplots for professional rendering when available,
        falls back to matplotlib polar plots.
        """
        try:
            import matplotlib.pyplot as plt  # noqa: F401  (availability probe + fallback renderer)
        except ImportError:
            logger.error("matplotlib not available")
            return None

        if not pass_events or not player_positions:
            return None

        # Try soccerplots for enhanced rendering
        has_soccerplots = False
        try:
            from soccerplots.radar_chart import Radar  # noqa: F401  (availability probe)

            has_soccerplots = True
        except ImportError:
            pass

        try:
            from daimon_runtime import setup_plot

            setup_plot({"runDir": str(self._exports_dir)})

            # Compute pass angles and distances by player
            player_passes = defaultdict(list)
            for event in pass_events:
                if event.get("type") != "pass" or not event.get("completed"):
                    continue
                src = event.get("from_track_id")
                if src is None or src not in player_positions:
                    continue
                # Use metadata for end position if available
                meta = event.get("metadata", {})
                end_x = meta.get("end_x")
                end_y = meta.get("end_y")
                if end_x is not None and end_y is not None:
                    start_x, start_y = player_positions[src]
                    dx = end_x - start_x
                    dy = end_y - start_y
                    distance = math.sqrt(dx * dx + dy * dy)
                    angle = math.atan2(dy, dx)
                    player_passes[src].append((angle, distance))

            if not player_passes:
                return None

            if has_soccerplots and len(player_passes) <= 6:
                return self._render_pass_sonar_soccerplots(player_passes, title, output_name)
            return self._render_pass_sonar_matplotlib(player_passes, title, output_name)
        except Exception as e:
            logger.error(f"Pass sonar generation failed: {e}")
            return None

    async def generate_formation_diagram(
        self,
        formation: dict[str, Any],
        title: str = "Formation",
        output_name: str = "formation.png",
    ) -> Path | None:
        """Generate a formation diagram on a pitch background.

        Args:
            formation: Dict with defenders, midfielders, attackers, and line_height
            title: Chart title
            output_name: Output filename
        """
        try:
            import matplotlib.pyplot as plt
            from daimon_runtime import setup_plot
        except ImportError:
            return None

        try:
            setup_plot({"runDir": str(self._exports_dir)})

            fig, ax = plt.subplots(figsize=(10, 7))
            ax.set_xlim(0, 105)
            ax.set_ylim(0, 68)
            ax.set_facecolor("#2d5a27")
            ax.axhline(34, color="white", linewidth=1.5, alpha=0.6)
            ax.axvline(52.5, color="white", linewidth=1.5, alpha=0.6)
            ax.axvline(16.5, color="white", linewidth=1, alpha=0.4)
            ax.axvline(88.5, color="white", linewidth=1, alpha=0.4)

            # Draw players by position
            def draw_line(players: list, y_line: float, color: str) -> None:
                n = len(players)
                if n == 0:
                    return
                spacing = 68 / (n + 1)
                for i, pid in enumerate(players):
                    y = spacing * (i + 1)
                    ax.scatter(
                        [y_line], [y], s=400, c=color, edgecolors="white", linewidths=2, zorder=5
                    )
                    ax.annotate(
                        str(pid),
                        (y_line, y),
                        fontsize=9,
                        ha="center",
                        va="center",
                        color="white",
                        fontweight="bold",
                    )

            defenders = formation.get("defenders", [])
            midfielders = formation.get("midfielders", [])
            attackers = formation.get("attackers", [])
            line_height = formation.get("line_height_m", 30)

            draw_line(defenders, line_height, "#e74c3c")
            draw_line(midfielders, line_height + 20, "#f39c12")
            draw_line(attackers, line_height + 40, "#2ecc71")

            ax.set_title(
                f"{title} ({formation.get('formation', 'unknown')})", color="white", fontsize=14
            )
            ax.set_xlabel("Pitch Length (m)", color="white")
            ax.set_ylabel("Pitch Width (m)", color="white")
            ax.tick_params(colors="white")

            output_path = self._exports_dir / output_name
            plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
            plt.close(fig)
            logger.info(f"Generated formation diagram: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Formation diagram generation failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Pass sonar rendering backends
    # ------------------------------------------------------------------

    def _render_pass_sonar_matplotlib(
        self,
        player_passes: dict[int, list[tuple[float, float]]],
        title: str,
        output_name: str,
    ) -> Path | None:
        """Render pass sonar using matplotlib polar plots."""
        import matplotlib.pyplot as plt

        n_players = len(player_passes)
        cols = min(4, n_players)
        rows = math.ceil(n_players / cols)

        fig, axes = plt.subplots(
            rows, cols, figsize=(4 * cols, 4 * rows), subplot_kw={"polar": True}
        )
        axes = [axes] if n_players == 1 else axes.flatten() if rows > 1 else axes

        for idx, (player_id, passes) in enumerate(player_passes.items()):
            if idx >= len(axes):
                break
            ax = axes[idx]
            angles = [p[0] for p in passes]
            distances = [p[1] for p in passes]
            n_bins = 8
            bin_width = 2 * math.pi / n_bins
            bin_counts = [0] * n_bins
            bin_distances = [0.0] * n_bins
            for angle, dist in zip(angles, distances, strict=False):
                bin_idx = int((angle + math.pi) / bin_width) % n_bins
                bin_counts[bin_idx] += 1
                bin_distances[bin_idx] += dist
            avg_distances = [
                bin_distances[i] / bin_counts[i] if bin_counts[i] > 0 else 0 for i in range(n_bins)
            ]
            max_dist = max(avg_distances) if max(avg_distances) > 0 else 1
            normalized = [d / max_dist for d in avg_distances]
            theta = [i * bin_width - math.pi for i in range(n_bins)]
            theta += [theta[0]]
            normalized += [normalized[0]]
            ax.fill(theta, normalized, alpha=0.3, color="blue")
            ax.plot(theta, normalized, color="blue", linewidth=2)
            ax.set_title(f"Player {player_id}", fontsize=10)
            ax.set_ylim(0, 1.2)
            ax.set_yticks([])

        for idx in range(len(player_passes), len(axes) if hasattr(axes, "__len__") else 1):
            if hasattr(axes, "__getitem__"):
                axes[idx].axis("off")

        fig.suptitle(title, fontsize=14)
        output_path = self._exports_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Generated pass sonar (matplotlib): {output_path}")
        return output_path

    def _render_pass_sonar_soccerplots(
        self,
        player_passes: dict[int, list[tuple[float, float]]],
        title: str,
        output_name: str,
    ) -> Path | None:
        """Render pass sonar using soccerplots for professional-quality output."""
        import matplotlib.pyplot as plt
        from soccerplots.radar_chart import Radar

        n_players = len(player_passes)
        cols = min(3, n_players)
        rows = math.ceil(n_players / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = [axes] if n_players == 1 else axes.flatten() if rows > 1 else axes

        for idx, (player_id, passes) in enumerate(player_passes.items()):
            if idx >= len(axes):
                break
            _ = axes[idx]
            angles = [p[0] for p in passes]
            distances = [p[1] for p in passes]
            n_bins = 8
            bin_width = 2 * math.pi / n_bins
            bin_counts = [0] * n_bins
            bin_distances = [0.0] * n_bins
            for angle, dist in zip(angles, distances, strict=False):
                bin_idx = int((angle + math.pi) / bin_width) % n_bins
                bin_counts[bin_idx] += 1
                bin_distances[bin_idx] += dist
            avg_distances = [
                bin_distances[i] / bin_counts[i] if bin_counts[i] > 0 else 0 for i in range(n_bins)
            ]
            max_dist = max(avg_distances) if max(avg_distances) > 0 else 1
            normalized = [d / max_dist for d in avg_distances]
            theta = [i * bin_width - math.pi for i in range(n_bins)]
            _ = [math.degrees(t) for t in theta]
            params = [f"{i * 45}°" for i in range(n_bins)]
            ranges = [(0, 1.2)] * n_bins
            values = normalized

            radar = Radar()
            fig_radar, ax_radar = radar.plot(
                params=params,
                ranges=ranges,
                values=[values],
                title={"title": f"Player {player_id}", "color": "#1a1a1a"},
                alphas=[0.3],  # type: ignore
            )

        for idx in range(len(player_passes), len(axes)):
            axes[idx].axis("off")

        fig.suptitle(title, fontsize=14, y=1.02)
        output_path = self._exports_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Generated pass sonar (soccerplots): {output_path}")
        return output_path
