"""HRV (Heart Rate Variability) analysis service.

Computes time-domain (RMSSD, SDNN, pNN50) and frequency-domain (LF/HF ratio)
HRV metrics from R-R interval data. Used for fatigue monitoring and recovery tracking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HRVResult:
    rmssd_ms: float = 0.0
    sdnn_ms: float = 0.0
    pnn50_pct: float = 0.0
    lf_hf_ratio: float = 0.0
    mean_hr_bpm: float = 0.0
    status: str = "unknown"


class HRVService:
    def compute_hrv(self, rr_intervals: list[float]) -> HRVResult:
        if len(rr_intervals) < 5:
            return HRVResult(status="insufficient_data")

        # Time domain
        diffs = [rr_intervals[i + 1] - rr_intervals[i] for i in range(len(rr_intervals) - 1)]
        squared_diffs = [d * d for d in diffs]
        rmssd = math.sqrt(sum(squared_diffs) / len(squared_diffs)) if diffs else 0.0
        mean_rr = sum(rr_intervals) / len(rr_intervals)
        sdnn = math.sqrt(sum((rr - mean_rr) ** 2 for rr in rr_intervals) / len(rr_intervals))
        nn50 = sum(1 for d in diffs if abs(d) > 50)
        pnn50 = (nn50 / len(diffs) * 100) if diffs else 0.0
        mean_hr = 60000.0 / mean_rr if mean_rr > 0 else 0.0

        from numpy import array, fft

        rr_arr = array(rr_intervals)
        n = len(rr_arr)
        if n >= 30:
            fft_vals = fft.rfft(rr_arr - rr_arr.mean())
            freqs = fft.rfftfreq(n, d=mean_rr / 1000.0)
            lf_mask = (freqs >= 0.04) & (freqs < 0.15)
            hf_mask = (freqs >= 0.15) & (freqs < 0.4)
            lf_power = sum(abs(fft_vals[lf_mask]) ** 2) if any(lf_mask) else 0.0
            hf_power = sum(abs(fft_vals[hf_mask]) ** 2) if any(hf_mask) else 0.0
            lf_hf = lf_power / hf_power if hf_power > 0 else 0.0
        else:
            lf_hf = 0.0

        return HRVResult(
            rmssd_ms=round(rmssd, 2),
            sdnn_ms=round(sdnn, 2),
            pnn50_pct=round(pnn50, 1),
            lf_hf_ratio=round(lf_hf, 2),
            mean_hr_bpm=round(mean_hr, 1),
            status="ok",
        )
