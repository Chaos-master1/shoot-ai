from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CorrelationResult:
    event_type: str = ""
    pre_event_avg_speed: float = 0.0
    post_event_avg_speed: float = 0.0
    pre_event_avg_hr: float | None = None
    post_event_avg_hr: float | None = None
    speed_delta_pct: float = 0.0
    hr_delta_pct: float | None = None
    sample_count: int = 0


@dataclass
class PhysioTacticalReport:
    correlations: list = field(default_factory=list)
    fatigue_periods: list = field(default_factory=list)
    high_intensity_bursts: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "correlations": [
                {
                    "event_type": c.event_type,
                    "pre_speed": round(c.pre_event_avg_speed, 2),
                    "post_speed": round(c.post_event_avg_speed, 2),
                    "speed_delta_pct": round(c.speed_delta_pct, 1),
                    "pre_hr": round(c.pre_event_avg_hr, 1) if c.pre_event_avg_hr else None,
                    "post_hr": round(c.post_event_avg_hr, 1) if c.post_event_avg_hr else None,
                    "hr_delta_pct": round(c.hr_delta_pct, 1) if c.hr_delta_pct else None,
                    "sample_count": c.sample_count,
                }
                for c in self.correlations
            ],
            "fatigue_periods": self.fatigue_periods[:20],
            "high_intensity_bursts": self.high_intensity_bursts[:20],
            "summary": self.summary,
        }


class PhysioTacticalCorrelationService:
    def analyze(
        self,
        events: list[dict],
        speed_timeline: list[dict],
        hr_timeline: list[dict] | None = None,
        window_s: float = 5.0,
    ) -> str:
        try:
            report = PhysioTacticalReport()
            speeds = np.array([s.get("v_spd", s.get("speed", 0)) or 0 for s in speed_timeline])
            times = np.array([s.get("t", s.get("timestamp", 0)) for s in speed_timeline])
            hr_values = None
            hr_times = None
            if hr_timeline:
                hr_values = np.array(
                    [h.get("hr", h.get("heart_rate", 0)) or 0 for h in hr_timeline]
                )
                hr_times = np.array([h.get("t", h.get("timestamp", 0)) for h in hr_timeline])
            event_types = {e.get("type", "unknown") for e in events}
            for etype in event_types:
                type_events = [e for e in events if e.get("type") == etype]
                pre_speeds = []
                post_speeds = []
                pre_hrs = []
                post_hrs = []
                for ev in type_events:
                    et = ev.get("timestamp", 0)
                    mask_pre = (times >= et - window_s) & (times < et)
                    mask_post = (times > et) & (times <= et + window_s)
                    if np.sum(mask_pre) > 0:
                        pre_speeds.append(float(np.mean(speeds[mask_pre])))
                    if np.sum(mask_post) > 0:
                        post_speeds.append(float(np.mean(speeds[mask_post])))
                    if hr_values is not None and hr_times is not None:
                        hm_pre = (hr_times >= et - window_s) & (hr_times < et)
                        hm_post = (hr_times > et) & (hr_times <= et + window_s)
                        if np.sum(hm_pre) > 0:
                            pre_hrs.append(float(np.mean(hr_values[hm_pre])))
                        if np.sum(hm_post) > 0:
                            post_hrs.append(float(np.mean(hr_values[hm_post])))
                if pre_speeds or post_speeds:
                    avg_pre = float(np.mean(pre_speeds)) if pre_speeds else 0
                    avg_post = float(np.mean(post_speeds)) if post_speeds else 0
                    delta = ((avg_post - avg_pre) / avg_pre * 100) if avg_pre > 0 else 0
                    c = CorrelationResult(
                        event_type=etype,
                        pre_event_avg_speed=avg_pre,
                        post_event_avg_speed=avg_post,
                        speed_delta_pct=delta,
                        sample_count=min(len(pre_speeds), len(post_speeds)),
                    )
                    if pre_hrs and post_hrs:
                        c.pre_event_avg_hr = float(np.mean(pre_hrs))
                        c.post_event_avg_hr = float(np.mean(post_hrs))
                        c.hr_delta_pct = (
                            (c.post_event_avg_hr - c.pre_event_avg_hr) / c.pre_event_avg_hr * 100
                            if c.pre_event_avg_hr > 0
                            else 0
                        )
                    report.correlations.append(c)
            fatigue_windows = []
            for i in range(
                0, len(times), max(1, int(30 / (times[1] - times[0]) if len(times) > 1 else 1))
            ):
                if i + 30 >= len(times):
                    break
                window_speeds = speeds[i : i + 30]
                avg = float(np.mean(window_speeds))
                fatigue_windows.append(
                    {"start_s": round(float(times[i]), 1), "avg_speed": round(avg, 2)}
                )
            if fatigue_windows:
                overall_avg = float(np.mean([f["avg_speed"] for f in fatigue_windows]))
                report.fatigue_periods = [
                    f for f in fatigue_windows if f["avg_speed"] < overall_avg * 0.8
                ]
            hi_bursts = []
            in_burst = False
            burst_start = 0.0
            for i in range(len(speeds)):
                if speeds[i] > 5.5:
                    if not in_burst:
                        burst_start = float(times[i])
                        in_burst = True
                else:
                    if in_burst:
                        hi_bursts.append(
                            {
                                "start_s": burst_start,
                                "end_s": float(times[i - 1]) if i > 0 else burst_start,
                            }
                        )
                        in_burst = False
            if in_burst:
                hi_bursts.append({"start_s": burst_start, "end_s": float(times[-1])})
            report.high_intensity_bursts = hi_bursts
            report.summary = {
                "total_events_analyzed": len(events),
                "speed_points": len(speed_timeline),
                "hr_points": len(hr_timeline) if hr_timeline else 0,
                "correlation_count": len(report.correlations),
                "fatigue_period_count": len(report.fatigue_periods),
                "hi_burst_count": len(hi_bursts),
            }
            return json.dumps({"report": report.to_dict(), "ok": True})
        except Exception as e:
            logger.error(f"physio_tactical_analyze failed: {e}")
            return json.dumps({"error": str(e)})
