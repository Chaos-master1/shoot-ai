"""Face recognition service for player identification using InsightFace.

Pipelines:
  - Build gallery: upload team roster photos → ArcFace embeddings → DB
  - Identify: detect faces from YOLO player crops → embed → match vs gallery
  - Link: automatic track_id → player_profile matching across matches
"""

from __future__ import annotations

import json
import logging
from typing import Any

import cv2
import numpy as np

from kawkab.core.paths import get_paths

logger = logging.getLogger(__name__)

# InsightFace auto-downloads model packs to ~/.insightface/models/
# We use buffalo_l: SCRFD face detector + ArcFace W50K recognizer
INSIGHTFACE_MODEL_PACK = "buffalo_l"
FACE_EMBEDDING_DIM = 512
MATCH_THRESHOLD = 0.45  # cosine distance threshold for positive match


class FaceRecognitionService:
    """Player face detection, embedding, and gallery matching."""

    def __init__(self) -> None:
        self._detector: Any = None
        self._recognizer: Any = None
        self._gallery: list[dict[str, Any]] = []
        self._gallery_loaded = False

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _ensure_models(self) -> None:
        if self._detector is not None:
            return
        try:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(
                name=INSIGHTFACE_MODEL_PACK,
                root=str(get_paths().models / "insightface"),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0, det_thresh=0.4, det_size=(640, 640))
            self._app = app
            logger.info(
                "InsightFace loaded (model pack=%s, det_size=640)",
                INSIGHTFACE_MODEL_PACK,
            )
        except Exception as e:
            logger.debug("InsightFace init failed: %s", e)
            self._app = None

    @property
    def available(self) -> bool:
        try:
            import insightface  # noqa: F401

            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Face detection + embedding
    # ------------------------------------------------------------------

    def detect_faces(self, img: np.ndarray) -> list[dict[str, Any]]:
        """Detect faces in an image, return list with bbox + embedding + confidence."""
        self._ensure_models()
        if self._app is None:
            return []

        faces = self._app.get(img)
        results = []
        for face in faces:
            bbox = face.bbox.astype(int).tolist()
            embedding = face.normed_embedding.tolist()
            det_score = float(face.det_score)
            results.append(
                {
                    "bbox": bbox,
                    "embedding": embedding,
                    "confidence": det_score,
                    "embedding_np": face.normed_embedding,
                }
            )
        return results

    def get_embedding(self, img: np.ndarray) -> np.ndarray | None:
        """Get the face embedding for the largest face in an image."""
        faces = self.detect_faces(img)
        if not faces:
            return None
        largest = max(
            faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1])
        )
        return largest["embedding_np"]

    # ------------------------------------------------------------------
    # Gallery management
    # ------------------------------------------------------------------

    def build_gallery(self, profiles: list[dict]) -> None:
        """Load player profiles with embeddings into memory.

        Args:
            profiles: List of player profile dicts from storage_service.get_all_player_profiles()
        """
        self._gallery = []
        for p in profiles:
            emb_text = p.get("face_embedding")
            if not emb_text:
                continue
            try:
                emb = np.array(json.loads(emb_text), dtype=np.float32)
            except (json.JSONDecodeError, TypeError):
                continue
            self._gallery.append(
                {
                    "profile_id": p["id"],
                    "global_id": p.get("global_id", ""),
                    "display_name": p.get("display_name", ""),
                    "jersey_number": p.get("jersey_number"),
                    "team": p.get("team", "home"),
                    "embedding": emb,
                    "confidence": p.get("face_confidence", 0.0),
                }
            )
        self._gallery_loaded = True
        logger.info("Face gallery loaded: %d profiles", len(self._gallery))

    def add_to_gallery(
        self,
        profile_id: int,
        embedding: np.ndarray,
        confidence: float,
    ) -> None:
        """Add a profile to the in-memory gallery."""
        self._gallery.append(
            {
                "profile_id": profile_id,
                "embedding": embedding,
                "confidence": confidence,
            }
        )
        self._gallery_loaded = True

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match_face(
        self, embedding: np.ndarray, threshold: float = MATCH_THRESHOLD
    ) -> dict[str, Any] | None:
        """Match an embedding against the gallery, return best match or None."""
        if not self._gallery_loaded:
            return None
        if not self._gallery:
            return None

        emb = embedding / max(np.linalg.norm(embedding), 1e-8)
        best_dist = float("inf")
        best_match = None

        for entry in self._gallery:
            gemb = entry["embedding"]
            if gemb.shape != emb.shape:
                continue
            dist = float(np.linalg.norm(gemb - emb))
            if dist < best_dist:
                best_dist = dist
                best_match = entry

        if best_match is None or best_dist > threshold:
            return None
        return {
            "profile_id": best_match["profile_id"],
            "display_name": best_match["display_name"],
            "jersey_number": best_match["jersey_number"],
            "team": best_match["team"],
            "distance": round(best_dist, 4),
            "confidence": round(1.0 - best_dist, 4),
        }

    def identify_player_from_crop(self, player_crop: np.ndarray) -> dict[str, Any] | None:
        """Identify a player from a cropped image of their upper body/face."""
        emb = self.get_embedding(player_crop)
        if emb is None:
            return None
        return self.match_face(emb)

    # ------------------------------------------------------------------
    # Bulk identification across match data
    # ------------------------------------------------------------------

    def identify_players_in_match(
        self,
        profiles: list[dict],
        track_data,
        frame_indices: list[int] | None = None,
    ) -> dict[int, dict[str, Any]]:
        """Try to identify each tracked player across sampled frames."""
        self._ensure_models()
        self.build_gallery(profiles)
        if not self._gallery:
            logger.warning("Empty face gallery, skipping match identification")
            return {}

        if frame_indices is None:
            total = len(track_data.frames)
            step = max(1, total // 30)
            frame_indices = list(range(0, total, step))

        identified: dict[int, dict[str, Any]] = {}

        for idx in frame_indices:
            if idx >= len(track_data.frames):
                continue
            frame_det = track_data.frames[idx]
            if not frame_det.detections:
                continue

            for det in frame_det.detections:
                if det.class_name != "person" or det.track_id is None:
                    continue
                if det.track_id in identified:
                    continue

                bbox = [int(v) for v in det.bbox]
                crop = self._crop_face_region(frame_det, bbox, track_data)
                if crop is None:
                    continue

                result = self.identify_player_from_crop(crop)
                if result:
                    identified[det.track_id] = result
                    logger.debug(
                        "Identified track %d as %s (dist=%.4f)",
                        det.track_id,
                        result["display_name"],
                        result["distance"],
                    )

        logger.info(
            "Match identification: %d / %d players identified",
            len(identified),
            len(
                {
                    d.track_id
                    for f in track_data.frames
                    for d in f.detections
                    if d.class_name == "person" and d.track_id is not None
                }
            ),
        )
        return identified

    def _crop_face_region(self, frame_det, bbox: list[int], track_data) -> np.ndarray | None:
        """Extract the upper-body region from a frame."""

        w = frame_det.image_width
        h = frame_det.image_height
        x1, y1, x2, y2 = bbox

        face_y1 = max(0, y1)
        face_y2 = min(h, y1 + int((y2 - y1) * 0.4))
        face_x1 = max(0, x1)
        face_x2 = min(w, x2)
        if face_x2 <= face_x1 or face_y2 <= face_y1:
            return None

        cap = cv2.VideoCapture(str(track_data.video_path))
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_det.frame_number)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None

        return frame[face_y1:face_y2, face_x1:face_x2]
