"""Math and physics utilities ported from zalo/MathUtilities to Python.

Exports:
- kabsch_align / kabsch_rotation: optimal rigid alignment (Kabsch algorithm)
- hungarian: optimal assignment (linear sum assignment)
- SpatialHash2D / SpatialHash3D: O(1) neighbor lookups

All algorithms are reference implementations adapted from the
zalo/MathUtilities repository (Unlicense). The C#/Unity code was
re-implemented in pure Python with NumPy.
"""

from kawkab.utils.hungarian import hungarian, hungarian_match
from kawkab.utils.kabsch import (
    apply_rigid_transform,
    kabsch_align,
    kabsch_align_2d,
    kabsch_rotation,
)
from kawkab.utils.spatial_hash import SpatialHash2D, SpatialHash3D, bulk_insert_2d

__all__ = [
    "kabsch_align",
    "kabsch_align_2d",
    "kabsch_rotation",
    "apply_rigid_transform",
    "hungarian",
    "hungarian_match",
    "SpatialHash2D",
    "SpatialHash3D",
    "bulk_insert_2d",
]
