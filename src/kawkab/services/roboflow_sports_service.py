"""Service wrapper for the roboflow/sports Python package.

Provides soccer-specific CV utilities: pitch drawing, Voronoi pitch control,
ball annotation/tracking, and view transformation.
Installs via: pip install git+https://github.com/roboflow/sports.git

The package is optional — if not installed, the service gracefully degrades
and reports not available. All methods return None or empty results in that case.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


class RoboflowSportsService:
    """Wrapper for roboflow/sports soccer CV utilities.

    The underlying package (https://github.com/roboflow/sports) provides:
    - `sports.annotators.soccer`: pitch drawing, point overlays, Voronoi diagrams
    - `sports.common.ball`: BallAnnotator, BallTracker
    - `sports.common.team`: TeamClassifier (when available)
    - `sports.common.view`: ViewTransformer
    - `sports.configs.soccer`: SoccerPitchConfiguration
    """

    def __init__(self) -> None:
        self._available = False
        self._soccer_config = None
        self._soccer_annotators = None
        self._ball_module = None
        self._team_module = None
        self._view_module = None
        self._try_import()

    def _try_import(self) -> None:
        try:
            from sports.annotators import soccer as soccer_annotators  # type: ignore
            from sports.common import ball as ball_module  # type: ignore
            from sports.configs.soccer import SoccerPitchConfiguration  # type: ignore

            self._soccer_config = SoccerPitchConfiguration
            self._soccer_annotators = soccer_annotators
            self._ball_module = ball_module

            try:
                from sports.common import team as team_module  # type: ignore

                self._team_module = team_module
            except Exception:
                logger.info("roboflow/sports: team module not available")
            try:
                from sports.common import view as view_module  # type: ignore

                self._view_module = view_module
            except Exception:
                logger.info("roboflow/sports: view module not available")

            self._available = True
            logger.info("roboflow/sports loaded")
        except Exception as e:
            logger.info(f"roboflow/sports not available: {e}")
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    @property
    def has_team_classifier(self) -> bool:
        return self._team_module is not None

    @property
    def has_view_transformer(self) -> bool:
        return self._view_module is not None

    # ------------------------------------------------------------------
    # Pitch drawing
    # ------------------------------------------------------------------

    def draw_pitch(
        self,
        background_color: tuple[int, int, int] = (34, 139, 34),
        line_color: tuple[int, int, int] = (255, 255, 255),
        padding: int = 50,
        line_thickness: int = 4,
        point_radius: int = 8,
        scale: float = 0.1,
    ) -> np.ndarray | None:
        if not self._available or self._soccer_annotators is None:
            return None
        try:
            import supervision as sv  # type: ignore

            return self._soccer_annotators.draw_pitch(
                config=self._soccer_config(),
                background_color=sv.Color(*background_color),
                line_color=sv.Color(*line_color),
                padding=padding,
                line_thickness=line_thickness,
                point_radius=point_radius,
                scale=scale,
            )
        except Exception as e:
            logger.warning(f"roboflow/sports draw_pitch failed: {e}")
            return None

    def draw_points_on_pitch(
        self,
        xy: np.ndarray,
        face_color: tuple[int, int, int] = (255, 0, 0),
        edge_color: tuple[int, int, int] = (0, 0, 0),
        radius: int = 10,
        thickness: int = 2,
        padding: int = 50,
        scale: float = 0.1,
    ) -> np.ndarray | None:
        if not self._available or self._soccer_annotators is None:
            return None
        try:
            import supervision as sv  # type: ignore

            return self._soccer_annotators.draw_points_on_pitch(
                config=self._soccer_config(),
                xy=xy,
                face_color=sv.Color(*face_color),
                edge_color=sv.Color(*edge_color),
                radius=radius,
                thickness=thickness,
                padding=padding,
                scale=scale,
            )
        except Exception as e:
            logger.warning(f"roboflow/sports draw_points_on_pitch failed: {e}")
            return None

    def draw_paths_on_pitch(
        self,
        paths: list[np.ndarray],
        color: tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
        padding: int = 50,
        scale: float = 0.1,
    ) -> np.ndarray | None:
        if not self._available or self._soccer_annotators is None:
            return None
        try:
            import supervision as sv  # type: ignore

            return self._soccer_annotators.draw_paths_on_pitch(
                config=self._soccer_config(),
                paths=paths,
                color=sv.Color(*color),
                thickness=thickness,
                padding=padding,
                scale=scale,
            )
        except Exception as e:
            logger.warning(f"roboflow/sports draw_paths_on_pitch failed: {e}")
            return None

    def draw_voronoi(
        self,
        team_1_xy: np.ndarray,
        team_2_xy: np.ndarray,
        team_1_color: tuple[int, int, int] = (255, 0, 0),
        team_2_color: tuple[int, int, int] = (255, 255, 255),
        opacity: float = 0.5,
        padding: int = 50,
        scale: float = 0.1,
    ) -> np.ndarray | None:
        if not self._available or self._soccer_annotators is None:
            return None
        try:
            import supervision as sv  # type: ignore

            return self._soccer_annotators.draw_pitch_voronoi_diagram(
                config=self._soccer_config(),
                team_1_xy=team_1_xy,
                team_2_xy=team_2_xy,
                team_1_color=sv.Color(*team_1_color),
                team_2_color=sv.Color(*team_2_color),
                opacity=opacity,
                padding=padding,
                scale=scale,
            )
        except Exception as e:
            logger.warning(f"roboflow/sports draw_voronoi failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Ball utilities
    # ------------------------------------------------------------------

    def create_ball_annotator(
        self, radius: int, buffer_size: int = 5, thickness: int = 2
    ) -> Any | None:
        if not self._available or self._ball_module is None:
            return None
        try:
            return self._ball_module.BallAnnotator(
                radius=radius, buffer_size=buffer_size, thickness=thickness
            )
        except Exception as e:
            logger.warning(f"roboflow/sports create_ball_annotator failed: {e}")
            return None

    def create_ball_tracker(self, buffer_size: int = 10) -> Any | None:
        if not self._available or self._ball_module is None:
            return None
        try:
            return self._ball_module.BallTracker(buffer_size=buffer_size)
        except Exception as e:
            logger.warning(f"roboflow/sports create_ball_tracker failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Team classifier (if module available)
    # ------------------------------------------------------------------

    def create_team_classifier(self, device: str = "cpu") -> Any | None:
        if not self._available or self._team_module is None:
            return None
        try:
            cls = getattr(self._team_module, "TeamClassifier", None)
            if cls is None:
                return None
            try:
                return cls(device=device)
            except TypeError:
                return cls()
        except Exception as e:
            logger.warning(f"roboflow/sports create_team_classifier failed: {e}")
            return None

    # ------------------------------------------------------------------
    # View transformer
    # ------------------------------------------------------------------

    def create_view_transformer(self, source: np.ndarray, target: np.ndarray) -> Any | None:
        if not self._available or self._view_module is None:
            return None
        try:
            cls = getattr(self._view_module, "ViewTransformer", None)
            if cls is None:
                return None
            return cls(source=source, target=target)
        except Exception as e:
            logger.warning(f"roboflow/sports create_view_transformer failed: {e}")
            return None
