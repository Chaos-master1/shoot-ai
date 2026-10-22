from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from kawkab.core.logging import get_logger
from kawkab.core.physical_metrics import PhysicalMetricsAnalyzer
from kawkab.services.wearable_import_service import WearableDataPoint

logger = get_logger(__name__)


@dataclass
class UnifiedPhysiologicalData:
    timestamp_s: float = 0.0
    video_speed_ms: float | None = None
    wearable_speed_ms: float | None = None
    heart_rate_bpm: float | None = None
    acceleration_ms2: float | None = None
    metabolic_power_w_kg: float | None = None
    source: str = "video"  # "video", "wearable", "merged"

    def to_dict(self):
        return {
            "t": round(self.timestamp_s, 1),
            "v_spd": round(self.video_speed_ms, 2) if self.video_speed_ms is not None else None,
            "w_spd": round(self.wearable_speed_ms, 2)
            if self.wearable_speed_ms is not None
            else None,
            "hr": round(self.heart_rate_bpm, 1) if self.heart_rate_bpm is not None else None,
            "acc": round(self.acceleration_ms2, 2) if self.acceleration_ms2 is not None else None,
            "mp": round(self.metabolic_power_w_kg, 2)
            if self.metabolic_power_w_kg is not None
            else None,
            "src": self.source,
        }


@dataclass
class MergedPhysiologyReport:
    player_id: int = 0
    total_duration_s: float = 0.0
    video_total_distance_m: float = 0.0
    wearable_total_distance_m: float | None = None
    avg_hr_bpm: float | None = None
    peak_hr_bpm: float | None = None
    hr_zones: dict = field(default_factory=dict)
    merged_timeline: list = field(default_factory=list)
    video_metrics: dict | None = None
    wearable_summary: dict | None = None
    correlation_speed_r: float | None = None

    def to_dict(self):
        return {
            "player_id": self.player_id,
            "duration_s": round(self.total_duration_s, 1),
            "video_distance_m": round(self.video_total_distance_m, 1),
            "wearable_distance_m": round(self.wearable_total_distance_m, 1)
            if self.wearable_total_distance_m is not None
            else None,
            "avg_hr": round(self.avg_hr_bpm, 1) if self.avg_hr_bpm is not None else None,
            "peak_hr": round(self.peak_hr_bpm, 1) if self.peak_hr_bpm is not None else None,
            "hr_zones": self.hr_zones,
            "timeline_points": len(self.merged_timeline),
            "video_metrics": self.video_metrics,
            "wearable_summary": self.wearable_summary,
            "correlation_speed_r": round(self.correlation_speed_r, 3)
            if self.correlation_speed_r is not None
            else None,
        }


class PhysiologicalMergeService:
    def __init__(self):
        self._analyzer = PhysicalMetricsAnalyzer()

    def merge(
        self,
        player_id: int,
        video_trajectory: list[tuple[float, float, float]],
        wearable_data: list[WearableDataPoint],
        body_mass_kg: float = 75.0,
    ) -> str:
        try:
            video_metrics = self._analyzer.analyze_player(video_trajectory, body_mass_kg)
            report = MergedPhysiologyReport(player_id=player_id)
            report.video_metrics = video_metrics.to_dict()
            if wearable_data:
                w_times = np.array([d.timestamp_s for d in wearable_data])
                w_speeds = np.array(
                    [d.speed_ms if d.speed_ms is not None else np.nan for d in wearable_data]
                )
                w_hr = np.array(
                    [
                        d.heart_rate_bpm if d.heart_rate_bpm is not None else np.nan
                        for d in wearable_data
                    ]
                )
                report.wearable_total_distance_m = float(
                    np.nansum([d.distance_m for d in wearable_data if d.distance_m is not None])
                )
                valid_hr = w_hr[~np.isnan(w_hr)]
                if len(valid_hr) > 0:
                    report.avg_hr_bpm = float(np.mean(valid_hr))
                    report.peak_hr_bpm = float(np.max(valid_hr))
                    report.hr_zones = {
                        "zone1_50_60": int(
                            np.sum(
                                (valid_hr >= 0.5 * np.max(valid_hr))
                                & (valid_hr < 0.6 * np.max(valid_hr))
                            )
                        ),
                        "zone2_60_70": int(
                            np.sum(
                                (valid_hr >= 0.6 * np.max(valid_hr))
                                & (valid_hr < 0.7 * np.max(valid_hr))
                            )
                        ),
                        "zone3_70_80": int(
                            np.sum(
                                (valid_hr >= 0.7 * np.max(valid_hr))
                                & (valid_hr < 0.8 * np.max(valid_hr))
                            )
                        ),
                        "zone4_80_90": int(
                            np.sum(
                                (valid_hr >= 0.8 * np.max(valid_hr))
                                & (valid_hr < 0.9 * np.max(valid_hr))
                            )
                        ),
                        "zone5_90_100": int(np.sum(valid_hr >= 0.9 * np.max(valid_hr))),
                    }
                report.wearable_summary = {
                    "point_count": len(wearable_data),
                    "duration_s": round(float(w_times[-1] - w_times[0]), 1)
                    if len(w_times) > 1
                    else 0,
                }
                v_times = np.array([t[0] for t in video_trajectory])
                v_speeds = np.array([t[1] for t in video_trajectory])
                if len(v_speeds) > 1 and len(w_speeds) > 1:
                    v_interp = np.interp(w_times, v_times, v_speeds)
                    valid_mask = ~np.isnan(w_speeds) & ~np.isnan(v_interp)
                    if np.sum(valid_mask) > 2:
                        vr = np.corrcoef(v_interp[valid_mask], w_speeds[valid_mask])[0, 1]
                        report.correlation_speed_r = float(vr)
                timeline = []
                v_idx = 0
                for _w_idx, wdp in enumerate(wearable_data):
                    while (
                        v_idx < len(video_trajectory) - 1
                        and video_trajectory[v_idx + 1][0] < wdp.timestamp_s
                    ):
                        v_idx += 1
                    vs = video_trajectory[v_idx][1] if v_idx < len(video_trajectory) else None
                    udp = UnifiedPhysiologicalData(
                        timestamp_s=wdp.timestamp_s,
                        video_speed_ms=vs,
                        wearable_speed_ms=wdp.speed_ms,
                        heart_rate_bpm=wdp.heart_rate_bpm,
                        acceleration_ms2=wdp.acceleration_ms2,
                        source="merged",
                    )
                    if wdp.speed_ms is not None:
                        udp.metabolic_power_w_kg = 3.6 * wdp.speed_ms + 1.2 * (
                            wdp.acceleration_ms2**2 if wdp.acceleration_ms2 else 0
                        )
                    timeline.append(udp.to_dict())
                report.merged_timeline = timeline
            report.total_duration_s = (
                float(video_trajectory[-1][0] - video_trajectory[0][0])
                if len(video_trajectory) > 1
                else 0
            )
            report.video_total_distance_m = float(video_metrics.total_distance_m)
            return json.dumps({"report": report.to_dict(), "ok": True})
        except Exception as e:
            logger.error(f"merge failed: {e}")
            return json.dumps({"error": str(e)})
