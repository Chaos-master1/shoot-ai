"""Abstract tracker interface for interchangeable tracking backends.

Allows swapping between boxmot (DeepOCSORT, BoT-SORT, ByteTrack, etc.)
and Norfair without changing pipeline code.

Usage:
    tracker = TrackerRegistry.create("deepocsort", cfg)
    tracks = tracker.update(detections, frame)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class TrackedObject:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str = "person"


class BaseTracker(ABC):
    @abstractmethod
    def update(self, detections: np.ndarray, frame: np.ndarray) -> list[TrackedObject]: ...

    @abstractmethod
    def reset(self) -> None: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class TrackerRegistry:
    _registry: dict[str, type[BaseTracker]] = {}

    @classmethod
    def register(cls, name: str, tracker_cls: type[BaseTracker]):
        cls._registry[name] = tracker_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseTracker:
        name = name.lower()
        if name in cls._registry:
            return cls._registry[name](**kwargs)
        raise ValueError(f"Unknown tracker: {name}. Available: {list(cls._registry.keys())}")

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry.keys())
