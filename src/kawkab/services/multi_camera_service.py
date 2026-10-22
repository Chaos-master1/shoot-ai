"""Multi-camera fusion — merges tracks from N calibrated cameras into a shared 3D pitch space.

Architecture:
- Each camera has its own homography H (pixel → pitch)
- Each camera produces independent track IDs
- The fusion service maps all tracks to shared pitch coordinates
- Identity matching across cameras uses: pitch position proximity + appearance ReID + temporal consistency
- Output: unified track list with global IDs projected onto the canonical pitch

Professional use case:
- 2-8 camera setup around the pitch
- Each camera covers 1/4 to 1/2 of the pitch (partial overlap)
- Fusion enables: full-pitch tracking, occlusion handling, multi-angle event review
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from kawkab.core.game_constants import GAME
from kawkab.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CameraView:
    """A single camera's calibration and current detections."""

    camera_id: str
    homography: np.ndarray  # 3x3 homography matrix (pixel → pitch)
    confidence: float = 1.0  # calibration confidence 0-1
    timestamp: float = 0.0
    tracks: list[dict] = field(default_factory=list)  # pixel-space tracks from this camera


@dataclass
class FusedTrack:
    """A globally-unique player track after fusion."""

    global_id: int
    pitch_x: float
    pitch_y: float
    speed: float = 0.0
    heading: float = 0.0
    confidence: float = 0.0
    camera_sources: list[str] = field(default_factory=list)
    last_seen_camera: str = ""
    last_seen_timestamp: float = 0.0


class MultiCameraFusion:
    """Fuses player tracks from multiple calibrated cameras into global pitch-space tracks.

    Algorithm:
    1. Map each camera's pixel-space tracks to pitch coordinates via homography
    2. Compute pairwise pitch-distance matrix between all camera track pairs
    3. Run Hungarian assignment to match tracks across cameras
    4. Merge matched tracks via weighted average (weight = camera confidence)
    5. Maintain global ID persistence using ReID features and track history
    6. Handle occlusions: if track missing from one camera, continue track from others

    Parameters:
        max_distance: Maximum pitch-space distance (meters) for cross-camera matching (default 3.0)
        min_cameras: Minimum cameras that must see a track to promote it (default 1)
        max_occlusion_frames: Frames to keep a track alive after all cameras lose it (default 90)
    """

    def __init__(
        self,
        max_distance: float = 3.0,
        min_cameras: int = 1,
        max_occlusion_frames: int = 90,
    ):
        self.max_distance = max_distance
        self.min_cameras = min_cameras
        self.max_occlusion_frames = max_occlusion_frames
        self.cameras: dict[str, CameraView] = {}
        self.fused_tracks: dict[int, FusedTrack] = {}
        self._next_global_id = 1
        self._track_history: dict[int, list[tuple[float, float, float]]] = {}
        self._frame_count = 0
        logger.info(
            f"MultiCameraFusion initialized: max_distance={max_distance}m, "
            f"min_cameras={min_cameras}, max_occlusion={max_occlusion_frames}"
        )

    def register_camera(
        self, camera_id: str, homography: np.ndarray, confidence: float = 1.0
    ) -> None:
        """Register a calibrated camera for fusion.

        Args:
            camera_id: Unique camera identifier
            homography: 3x3 homography matrix (pixel → pitch)
            confidence: Calibration confidence 0-1
        """
        self.cameras[camera_id] = CameraView(
            camera_id=camera_id,
            homography=homography,
            confidence=confidence,
        )
        logger.info(f"Camera {camera_id} registered (confidence={confidence:.2f})")

    def update(self, camera_id: str, tracks: list[dict], timestamp: float) -> list[FusedTrack]:
        """Ingest tracks from a camera and return the fused global track list.

        Each track dict should have:
            - track_id: int (per-camera local ID)
            - bbox: (x1, y1, x2, y2) pixel coordinates
            - confidence: float 0-1
            - class_name: str (e.g. "person", "sports ball")
            - reid_embedding: np.ndarray (optional, for appearance matching)

        Args:
            camera_id: Source camera identifier
            tracks: List of track dicts from this camera
            timestamp: Current frame timestamp in seconds

        Returns:
            List of FusedTrack objects with global IDs
        """
        if camera_id not in self.cameras:
            logger.warning(f"Camera {camera_id} not registered, ignoring update")
            return list(self.fused_tracks.values())

        self._frame_count += 1
        camera = self.cameras[camera_id]
        camera.timestamp = timestamp
        camera.tracks = tracks

        _ = self._project_to_pitch(camera_id, tracks)
        self._match_across_cameras()
        self._handle_occlusions()
        self._cull_stale_tracks()

        return list(self.fused_tracks.values())

    BALL_CLASSES = {"sports ball", "ball"}

    def _project_to_pitch(self, camera_id: str, tracks: list[dict]) -> list[dict]:
        """Project pixel-space tracks from a camera to pitch coordinates.

        Filters out ball tracks (class_name in BALL_CLASSES).
        """
        camera = self.cameras[camera_id]
        H = camera.homography
        pitch_tracks = []
        for track in tracks:
            if track.get("class_name", "person") in self.BALL_CLASSES:
                continue
            bbox = track.get("bbox", (0, 0, 0, 0))
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            pt = np.array([cx, cy, 1.0])
            transformed = H @ pt
            if abs(transformed[2]) < 1e-8:
                pitch_x, pitch_y = 0.0, 0.0
            else:
                pitch_x = float(transformed[0] / transformed[2])
                pitch_y = float(transformed[1] / transformed[2])
            pitch_x = max(0.0, min(GAME.PITCH_LENGTH_M, pitch_x))
            pitch_y = max(0.0, min(GAME.PITCH_WIDTH_M, pitch_y))
            pitch_tracks.append(
                {
                    "camera_id": camera_id,
                    "local_track_id": track.get("track_id"),
                    "pitch_x": pitch_x,
                    "pitch_y": pitch_y,
                    "confidence": track.get("confidence", 0.5),
                    "reid_embedding": track.get("reid_embedding"),
                    "bbox": track.get("bbox"),
                }
            )
        return pitch_tracks

    def _match_across_cameras(self) -> list[list[tuple[str, int]]]:
        """Match tracks across all registered cameras using Hungarian assignment.

        Returns:
            List of track groups, each group is a list of (camera_id, local_track_id) tuples
            that represent the same physical player.
        """
        all_pitch_tracks: list[dict] = []
        for cid, cam in self.cameras.items():
            projected = self._project_to_pitch(cid, cam.tracks)
            for pt in projected:
                pt["camera_id"] = cid
                all_pitch_tracks.append(pt)

        track_groups: list[list[tuple[str, int]]] = []

        if len(all_pitch_tracks) < 2:
            track_groups = [[(pt["camera_id"], pt["local_track_id"])] for pt in all_pitch_tracks]
            self._merge_tracks(track_groups)
            return track_groups

        n = len(all_pitch_tracks)
        cost_matrix = np.full((n, n), self.max_distance * 2, dtype=np.float64)

        for i in range(n):
            for j in range(n):
                if i == j or all_pitch_tracks[i]["camera_id"] == all_pitch_tracks[j]["camera_id"]:
                    cost_matrix[i, j] = self.max_distance * 2
                else:
                    dx = all_pitch_tracks[i]["pitch_x"] - all_pitch_tracks[j]["pitch_x"]
                    dy = all_pitch_tracks[i]["pitch_y"] - all_pitch_tracks[j]["pitch_y"]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist <= self.max_distance:
                        cost_matrix[i, j] = dist
                    else:
                        cost_matrix[i, j] = self.max_distance * 2

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        used_rows: set[int] = set()
        used_cols: set[int] = set()

        for r, c in zip(row_ind, col_ind, strict=False):
            if cost_matrix[r, c] > self.max_distance:
                continue
            if r in used_rows or c in used_cols:
                continue
            used_rows.add(r)
            used_cols.add(c)
            group = [
                (all_pitch_tracks[r]["camera_id"], all_pitch_tracks[r]["local_track_id"]),
                (all_pitch_tracks[c]["camera_id"], all_pitch_tracks[c]["local_track_id"]),
            ]
            track_groups.append(group)

        # Add unmatched tracks as singleton groups
        for i, pt in enumerate(all_pitch_tracks):
            if i not in used_rows and i not in used_cols:
                track_groups.append([(pt["camera_id"], pt["local_track_id"])])

        self._merge_tracks(track_groups)
        return track_groups

    def _merge_tracks(self, track_groups: list[list[tuple[str, int]]]) -> None:
        """Merge matched track groups into fused tracks.

        Uses weighted average (by camera confidence) for position.
        """
        for group in track_groups:
            if not group:
                continue

            positions: list[tuple[float, float, float, str]] = []
            for cid, local_tid in group:
                camera = self.cameras.get(cid)
                if camera is None:
                    continue
                projected = self._project_to_pitch(
                    cid, [t for t in camera.tracks if t.get("track_id") == local_tid]
                )
                if projected:
                    pt = projected[0]
                    positions.append((pt["pitch_x"], pt["pitch_y"], camera.confidence, cid))

            if not positions:
                continue

            total_weight = sum(p[2] for p in positions)
            if total_weight < 1e-8:
                avg_x = sum(p[0] for p in positions) / len(positions)
                avg_y = sum(p[1] for p in positions) / len(positions)
                avg_conf = sum(p[2] for p in positions) / len(positions)
            else:
                avg_x = sum(p[0] * p[2] for p in positions) / total_weight
                avg_y = sum(p[1] * p[2] for p in positions) / total_weight
                avg_conf = total_weight / len(positions)

            camera_sources = [p[3] for p in positions]
            last_cam = camera_sources[-1] if camera_sources else ""

            # Try to match to existing fused track
            best_id = None
            best_dist = self.max_distance
            for gid, ft in self.fused_tracks.items():
                dx = ft.pitch_x - avg_x
                dy = ft.pitch_y - avg_y
                d = np.sqrt(dx * dx + dy * dy)
                if d < best_dist:
                    best_dist = d
                    best_id = gid

            if best_id is not None:
                ft = self.fused_tracks[best_id]
                ft.pitch_x = avg_x
                ft.pitch_y = avg_y
                ft.confidence = avg_conf
                ft.camera_sources = list(set(ft.camera_sources + camera_sources))
                ft.last_seen_camera = last_cam
                ft.last_seen_timestamp = self._frame_count
                if best_id in self._track_history:
                    self._track_history[best_id].append((avg_x, avg_y, self._frame_count))
            else:
                gid = self._next_global_id
                self._next_global_id += 1
                ft = FusedTrack(
                    global_id=gid,
                    pitch_x=avg_x,
                    pitch_y=avg_y,
                    confidence=avg_conf,
                    camera_sources=camera_sources,
                    last_seen_camera=last_cam,
                    last_seen_timestamp=float(self._frame_count),
                )
                self.fused_tracks[gid] = ft
                self._track_history[gid] = [(avg_x, avg_y, self._frame_count)]

    def _assign_global_ids(self) -> None:
        """Re-assign global IDs to maintain consistency across frames."""
        pass  # IDs are assigned during _merge_tracks

    def _handle_occlusions(self) -> None:
        """Handle tracks that are temporarily lost by all cameras.

        Keeps the track alive for max_occlusion_frames using last known position.
        """
        current_frame = self._frame_count
        for gid in list(self.fused_tracks.keys()):
            ft = self.fused_tracks[gid]
            frames_since_seen = current_frame - ft.last_seen_timestamp
            if frames_since_seen > 0 and frames_since_seen <= self.max_occlusion_frames:
                history = self._track_history.get(gid, [])
                ft.confidence *= 0.95
                if len(history) >= 2:
                    x_prev, y_prev, t_prev = history[-2]
                    x_last, y_last, t_last = history[-1]
                    dt = t_last - t_prev if t_last > t_prev else 1.0
                    vx = (x_last - x_prev) / dt
                    vy = (y_last - y_prev) / dt
                    ft.pitch_x = max(0.0, min(GAME.PITCH_LENGTH_M, ft.pitch_x + vx))
                    ft.pitch_y = max(0.0, min(GAME.PITCH_WIDTH_M, ft.pitch_y + vy))
                    ft.speed = np.sqrt(vx * vx + vy * vy)
                self._track_history[gid].append((ft.pitch_x, ft.pitch_y, self._frame_count))

    def _cull_stale_tracks(self) -> None:
        """Remove tracks that haven't been seen for too long."""
        current_frame = self._frame_count
        stale_ids = [
            gid
            for gid, ft in self.fused_tracks.items()
            if current_frame - ft.last_seen_timestamp > self.max_occlusion_frames
        ]
        for gid in stale_ids:
            del self.fused_tracks[gid]
            self._track_history.pop(gid, None)
        if stale_ids:
            logger.debug(f"Culled {len(stale_ids)} stale tracks")

    def get_pitch_coverage(self) -> dict:
        """Return pitch coverage statistics per camera.

        Returns:
            Dict with camera_id -> {covered_area_pct, track_count, avg_confidence}
        """
        coverage: dict[str, dict] = {}
        for cid, cam in self.cameras.items():
            projected = self._project_to_pitch(cid, cam.tracks)
            if not projected:
                coverage[cid] = {
                    "covered_area_pct": 0.0,
                    "track_count": 0,
                    "avg_confidence": 0.0,
                }
                continue
            xs = np.array([p["pitch_x"] for p in projected])
            ys = np.array([p["pitch_y"] for p in projected])
            if len(xs) < 2:
                coverage[cid] = {
                    "covered_area_pct": 0.0,
                    "track_count": len(projected),
                    "avg_confidence": float(np.mean([p["confidence"] for p in projected])),
                }
                continue
            x_range = max(xs) - min(xs)
            y_range = max(ys) - min(ys)
            pitch_area = GAME.PITCH_LENGTH_M * GAME.PITCH_WIDTH_M
            covered_area = x_range * y_range
            coverage[cid] = {
                "covered_area_pct": min(100.0, covered_area / pitch_area * 100.0),
                "track_count": len(projected),
                "avg_confidence": float(np.mean([p["confidence"] for p in projected])),
            }
        return coverage

    def get_camera_contributions(self) -> dict[str, float]:
        """Return each camera's contribution to the fused track set.

        Returns:
            Dict of camera_id -> fraction of fused tracks that include this camera
        """
        if not self.fused_tracks:
            return dict.fromkeys(self.cameras, 0.0)
        contributions: dict[str, float] = {}
        for cid in self.cameras:
            count = sum(1 for ft in self.fused_tracks.values() if cid in ft.camera_sources)
            contributions[cid] = count / len(self.fused_tracks)
        return contributions

    def reset(self) -> None:
        """Reset all fusion state."""
        self.fused_tracks.clear()
        self._track_history.clear()
        self._next_global_id = 1
        self._frame_count = 0
        for cam in self.cameras.values():
            cam.tracks = []
            cam.timestamp = 0.0
        logger.info("MultiCameraFusion reset")

    def save_state(self, path: str | Path) -> None:
        """Save fusion state to a JSON file.

        Args:
            path: Path to save file
        """
        path = Path(path)
        state = {
            "max_distance": self.max_distance,
            "min_cameras": self.min_cameras,
            "max_occlusion_frames": self.max_occlusion_frames,
            "next_global_id": self._next_global_id,
            "frame_count": self._frame_count,
            "cameras": {
                cid: {
                    "homography": cam.homography.tolist(),
                    "confidence": cam.confidence,
                }
                for cid, cam in self.cameras.items()
            },
            "fused_tracks": {
                str(gid): {
                    "pitch_x": ft.pitch_x,
                    "pitch_y": ft.pitch_y,
                    "speed": ft.speed,
                    "heading": ft.heading,
                    "confidence": ft.confidence,
                    "camera_sources": ft.camera_sources,
                    "last_seen_camera": ft.last_seen_camera,
                    "last_seen_timestamp": ft.last_seen_timestamp,
                }
                for gid, ft in self.fused_tracks.items()
            },
            "track_history": {
                str(gid): [(x, y, t) for x, y, t in hist]
                for gid, hist in self._track_history.items()
            },
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=_json_default)
        logger.info(f"Fusion state saved to {path}")

    def load_state(self, path: str | Path) -> None:
        """Load fusion state from a JSON file.

        Args:
            path: Path to save file
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"Fusion state file not found: {path}")
            return
        with open(path) as f:
            state = json.load(f)
        self.max_distance = state.get("max_distance", self.max_distance)
        self.min_cameras = state.get("min_cameras", self.min_cameras)
        self.max_occlusion_frames = state.get("max_occlusion_frames", self.max_occlusion_frames)
        self._next_global_id = state.get("next_global_id", 1)
        self._frame_count = state.get("frame_count", 0)
        self.cameras.clear()
        for cid, cdata in state.get("cameras", {}).items():
            self.cameras[cid] = CameraView(
                camera_id=cid,
                homography=np.array(cdata["homography"], dtype=np.float64),
                confidence=cdata.get("confidence", 1.0),
            )
        self.fused_tracks.clear()
        for gid_str, ftdata in state.get("fused_tracks", {}).items():
            gid = int(gid_str)
            self.fused_tracks[gid] = FusedTrack(
                global_id=gid,
                pitch_x=ftdata["pitch_x"],
                pitch_y=ftdata["pitch_y"],
                speed=ftdata.get("speed", 0.0),
                heading=ftdata.get("heading", 0.0),
                confidence=ftdata.get("confidence", 0.0),
                camera_sources=ftdata.get("camera_sources", []),
                last_seen_camera=ftdata.get("last_seen_camera", ""),
                last_seen_timestamp=ftdata.get("last_seen_timestamp", 0.0),
            )
        self._track_history.clear()
        for gid_str, hist in state.get("track_history", {}).items():
            self._track_history[int(gid_str)] = [(x, y, t) for x, y, t in hist]
        logger.info(f"Fusion state loaded from {path}")


def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
