"""Biological maturation (PHV) estimation — Phase C youth layer.

Implements the Mirwald et al. (2002) somatic maturity-offset equation
for boys and is deliberately honest about what it does not implement:

- **Boys only, for now.** The male coefficients below are verified
  against two independent published sources. The female equation was
  NOT verifiable from authoritative sources during implementation, so
  female inputs return ``not_implemented`` with guidance — a guessed
  coefficient would be fabrication in exactly the sense this
  transformation exists to eliminate.
- **Estimates, not measurements.** The equation has a standard error
  of roughly ±0.6 years and is most informative within ~1-2 years of
  PHV; validation work (Malina & Kozieł) shows predictions track
  chronological age closely. Every result ships that caveat.
- **Flags, never labels.** Classification bands are relative to PHV
  timing for bio-banding conversations and load management around the
  growth window — never talent-selection verdicts.

Units: height/sitting height in centimeters, age in decimal years.
Reference: Mirwald, Baxter-Jones, Bailey & Beunen (2002), Med Sci
Sports Exerc 34(4):689-694 (European-ancestry reference samples).
"""

from __future__ import annotations

from typing import Any

from kawkab.core.logging import get_logger

logger = get_logger(__name__)

# Mirwald (2002) boys: offset = -9.236 + 0.0002708*LL*SH - 0.001663*CA*LL
#                        + 0.007216*CA*SH + 0.02292*CA
# (LL = standing height - sitting height; CA in decimal years; cm units)
_MALE_INTERCEPT = -9.236
_MALE_LL_SH = 0.0002708
_MALE_CA_LL = -0.001663
_MALE_CA_SH = 0.007216
_MALE_CA = 0.02292

# Validity context shipped with every result.
_VALIDITY_NOTES = [
    "standard error of the estimate is roughly ±0.59 years; treat the "
    "offset as an estimate, not a measurement",
    "most informative within ~1-2 years of the actual PHV event; "
    "interpretation degrades away from that window",
    "validation studies (Malina & Kozieł) show predicted offset tracks "
    "chronological age closely (r ≈ 0.95) — compare players against "
    "their own age group, not across ages",
    "equation derived from European-ancestry reference samples "
    "(Mirwald et al. 2002); apply caution to other populations",
]


def _classification(offset: float) -> dict[str, Any]:
    """PHV-relative band from the maturity offset (years)."""
    if offset < -1.0:
        return {"band": "pre_phv", "label": "Before peak height velocity"}
    if offset <= 1.0:
        label = "Circa peak height velocity"
        if abs(offset) <= 0.5:
            return {"band": "circum_phv", "label": label, "peak_growth_window": True}
        return {"band": "circum_phv", "label": label, "peak_growth_window": False}
    return {"band": "post_phv", "label": "After peak height velocity"}


class MaturationService:
    """Somatic maturity-offset estimation and growth-window load flags."""

    def estimate_offset(
        self,
        age_years: float,
        standing_height_cm: float,
        sitting_height_cm: float,
        sex: str = "male",
    ) -> dict[str, Any]:
        """Estimate maturity offset and PHV age from anthropometrics."""
        result: dict[str, Any] = {
            "age_years": age_years,
            "standing_height_cm": standing_height_cm,
            "sitting_height_cm": sitting_height_cm,
            "sex": str(sex).lower(),
            "maturity_offset": None,
            "classification": "not_computable",
            "band": None,
            "provenance": {
                "method": "Mirwald et al. (2002) somatic maturity offset",
                "validity_notes": list(_VALIDITY_NOTES),
            },
        }

        # Input sanity — bad measurements must fail loudly, not compute.
        if not (5.0 <= float(age_years) <= 25.0):
            result["provenance"]["error"] = "age outside 5-25 years — check input"
            return result
        if not (60.0 < float(standing_height_cm) < 230.0):
            result["provenance"]["error"] = "standing height implausible — check units (cm)"
            return result
        if not (30.0 < float(sitting_height_cm) < float(standing_height_cm)):
            result["provenance"]["error"] = (
                "sitting height must be positive and below standing height"
            )
            return result

        if result["sex"] == "female":
            result["provenance"]["error"] = (
                "female maturity-offset coefficients were not verifiable from "
                "authoritative sources during implementation; enter measurements "
                "anyway to store them, but do not use this estimate for girls"
            )
            result["classification"] = "not_implemented"
            return result
        if result["sex"] != "male":
            result["provenance"]["error"] = f"unknown sex value: {sex!r} (male | female)"
            return result

        leg_length = float(standing_height_cm) - float(sitting_height_cm)
        ca = float(age_years)
        offset = (
            _MALE_INTERCEPT
            + _MALE_LL_SH * leg_length * float(sitting_height_cm)
            + _MALE_CA_LL * ca * leg_length
            + _MALE_CA_SH * ca * float(sitting_height_cm)
            + _MALE_CA * ca
        )
        cls = _classification(offset)
        result["maturity_offset"] = round(offset, 2)
        result["leg_length_cm"] = round(leg_length, 1)
        result["classification"] = cls["label"]
        result["band"] = cls["band"]
        result["peak_growth_window"] = cls.get("peak_growth_window", False)
        result["estimated_age_at_phv"] = round(ca - offset, 2)
        return result

    def growth_load_flags(
        self,
        maturity_offset: float | None,
        minutes_last_28d: int | None = None,
        age_phase: str = "",
    ) -> dict[str, Any]:
        """Load-management advisories around the growth window.

        Flags are for conversations about managing minutes and load for
        players in their peak growth window — never selection blocks.
        """
        flags: list[dict[str, Any]] = []
        if maturity_offset is None:
            return {
                "flags": flags,
                "provenance": {"note": "no maturity estimate available"},
            }
        cls = _classification(maturity_offset)
        if cls.get("peak_growth_window"):
            flags.append(
                {
                    "type": "peak_growth_window",
                    "note": (
                        "Player is at peak height velocity: growth plates are "
                        "vulnerable and load tolerance varies widely. Favor "
                        "monitoring and conservative load progression."
                    ),
                }
            )
            if minutes_last_28d is not None and minutes_last_28d > 360:
                flags.append(
                    {
                        "type": "high_minutes_in_growth_window",
                        "note": (
                            f"{minutes_last_28d} minutes in the last 28 days while in "
                            "the peak growth window — review match/training load "
                            "balance (advisory only)"
                        ),
                    }
                )
        if age_phase and maturity_offset is not None and maturity_offset > 1.0:
            flags.append(
                {
                    "type": "bio_band_offset",
                    "note": (
                        f"Post-PHV while recorded age phase is '{age_phase}' — "
                        "consider bio-banded comparison before conclusions about "
                        "physical performance"
                    ),
                }
            )
        return {"flags": flags, "band": cls["band"]}
