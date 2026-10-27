"""Angle-convention normalisation.

The cricket biomechanics literature uses two mutually inverse conventions for joint angles:

``straight_180``
    Full extension is 180 deg, flexion decreases from there. Used by
    ``peploe2019technique`` ("straight = 180 deg, flexed < 180 deg") and by
    ``pullshot_malaysia`` ("180 deg was considered as a full extension of joint angle and
    zero 0 deg as a full flexion").

``flexion_from_0``
    Full extension is 0 deg, flexion increases from there. Common in clinical sources, and
    the convention behind statements such as "knee flexion <30 deg at front-foot contact
    elevates lumbar loading".

Mixing them silently inverts a grade: a well-braced 165 deg front knee read under the wrong
convention becomes a heavily flexed one. Every reference-template metric must therefore
declare its ``convention``, and ``template_validator`` rejects any that does not.

Internally the pipeline computes angles in ``straight_180``. Template ranges are converted
into that internal convention at scoring time, never the other way round.
"""

from __future__ import annotations

from enum import Enum
from typing import Tuple


class AngleConvention(str, Enum):
    STRAIGHT_180 = "straight_180"
    FLEXION_FROM_0 = "flexion_from_0"


#: The convention all internal computation uses.
INTERNAL = AngleConvention.STRAIGHT_180


class ConventionError(ValueError):
    pass


def to_internal(value: float, convention: AngleConvention | str) -> float:
    """Convert a single angle from a stated convention into the internal one."""
    conv = AngleConvention(convention)
    if conv is INTERNAL:
        return value
    if conv is AngleConvention.FLEXION_FROM_0:
        return 180.0 - value
    raise ConventionError(f"Unhandled convention {conv!r}")


def range_to_internal(
    low: float,
    high: float,
    convention: AngleConvention | str,
) -> Tuple[float, float]:
    """Convert an expected ``[low, high]`` band into the internal convention.

    The inverting conversion swaps the bounds, so the result is re-ordered. Forgetting this
    is the exact bug the module exists to prevent - an inverted band silently passes every
    value it should fail.
    """
    a = to_internal(low, convention)
    b = to_internal(high, convention)
    return (a, b) if a <= b else (b, a)


def describe(convention: AngleConvention | str) -> str:
    """Human-readable gloss, for validator errors and template documentation."""
    conv = AngleConvention(convention)
    return {
        AngleConvention.STRAIGHT_180: "straight = 180 deg, flexed < 180 deg",
        AngleConvention.FLEXION_FROM_0: "straight = 0 deg, flexion increases",
    }[conv]
