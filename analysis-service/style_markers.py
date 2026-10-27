"""Technical style markers: classified and reported, never scored.

**Status: specified stub.**

These are reliably observable properties of a player's technique that are **not correctness
axes**. They exist as a separate category from Tier A/B/C because their exclusion from
scoring is not about measurement error - it is about the literature showing that the
coaching orthodoxy is wrong.

The backlift is the clearest case. ``noorbhai_lbbt`` found:

* 77% of successful batsmen, and 90% of IPL batsmen, use a **lateral** backlift - lifting
  toward second slip with the face toward the off side
* 83% of coaches teach the **straight** backlift
* more than 70% of *uncoached* players naturally adopt lateral; more than 70% of *coached*
  players adopt straight

A template encoding "straight backlift = correct" would therefore mark down the technique
that most elite players actually use. The app reports what it sees and leaves the judgement
to the player and their coach.

The same logic applies to trigger movement, whose prevalence is format-dependent (Test
56.1%, ODI 47.4%, T20I 73.3%, IPL 76% - ``trigger_formats``), so absence is not a fault.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from pose_extraction import PoseSequence


class BackliftDirection(str, Enum):
    LATERAL = "lateral"
    STRAIGHT = "straight"
    INDETERMINATE = "indeterminate"


class TriggerType(str, Enum):
    NONE = "none"
    BACK_AND_ACROSS = "back_and_across"
    FORWARD_PRESS = "forward_press"
    TAP = "tap"
    OTHER = "other"


class StanceWidth(str, Enum):
    NARROW = "narrow"
    MEDIUM = "medium"
    WIDE = "wide"


#: Neutral display strings. Deliberately free of any evaluative word - no "good", no
#: "correct", no "textbook". The feedback layer additionally wraps these with an explicit
#: "this is a style choice, not a fault".
DISPLAY = {
    BackliftDirection.LATERAL: "toward the slips (lateral)",
    BackliftDirection.STRAIGHT: "toward the stumps (straight)",
    BackliftDirection.INDETERMINATE: "not clear from this angle",
    TriggerType.NONE: "none detected",
    TriggerType.BACK_AND_ACROSS: "back and across",
    TriggerType.FORWARD_PRESS: "forward press",
    TriggerType.TAP: "bat tap",
    TriggerType.OTHER: "present, type unclear",
    StanceWidth.NARROW: "narrow",
    StanceWidth.MEDIUM: "medium",
    StanceWidth.WIDE: "wide",
}


@dataclass
class StyleMarkers:
    backlift: BackliftDirection
    trigger: TriggerType
    stance: Optional[StanceWidth]
    backlift_confidence: float
    trigger_confidence: float

    def to_display(self) -> Dict[str, str]:
        """Map to ``{metric_key: display_string}`` for :func:`feedback.generate`."""
        out = {
            "batting.backlift_direction": DISPLAY[self.backlift],
            "batting.trigger_present": DISPLAY[self.trigger],
        }
        if self.stance is not None:
            out["batting.stance_width"] = DISPLAY[self.stance]
        return out


#: Stance-width band edges, as fractions of stature. These are **display conventions, not
#: norms** - ``battingsetup_review`` describes stance qualitatively without publishing
#: normative widths. They exist only to turn a number into a word for a descriptor that is
#: never scored. Do not promote them into a reference template.
STANCE_NARROW_MAX = 0.20
STANCE_WIDE_MIN = 0.35

#: Minimum lateral wrist travel, in stature fractions, before a backlift counts as lateral
#: rather than straight. Also a display convention pending a sourced threshold.
BACKLIFT_LATERAL_MIN = 0.04


def detect_backlift_direction(
    seq: PoseSequence,
    bc_frame: int,
    ds_frame: int,
    *,
    handedness: str,
    has_front_on: bool,
) -> tuple[BackliftDirection, float]:
    """Classify backlift direction.

    Requires a front-on or behind view. On a true side-on camera the lateral component of
    the backlift points almost directly at or away from the lens - precisely the axis a
    monocular side-on view cannot resolve - so this returns ``INDETERMINATE`` with zero
    confidence rather than guessing. A wrong backlift classification would be shown to the
    player as a fact about their technique.
    """
    if not has_front_on:
        return BackliftDirection.INDETERMINATE, 0.0

    import numpy as np

    from pose_extraction import KEYPOINT_INDEX, to_lead_rear

    wrist = KEYPOINT_INDEX[to_lead_rear("lead_wrist", handedness)]
    start = seq.xy[bc_frame, wrist]
    top = seq.xy[ds_frame, wrist]
    if np.isnan(start).any() or np.isnan(top).any():
        return BackliftDirection.INDETERMINATE, 0.0

    from normalisation import estimate_scale

    stature = estimate_scale(seq).stature_px
    lateral = abs(float(top[0] - start[0])) / stature

    confidence = float(
        min(
            seq.confidence[bc_frame, wrist],
            seq.confidence[ds_frame, wrist],
        )
    )
    direction = (
        BackliftDirection.LATERAL
        if lateral >= BACKLIFT_LATERAL_MIN
        else BackliftDirection.STRAIGHT
    )
    return direction, confidence


def classify_trigger(
    seq: PoseSequence,
    stance_frame: int,
    ss_frame: int,
    *,
    handedness: str,
    detected: bool,
) -> tuple[TriggerType, float]:
    """Classify a trigger movement that phase detection has already located.

    Detection lives in :mod:`phase_detection`, which separates a trigger from a stride by
    net displacement - a trigger returns to where it started, a stride does not. This
    function only names the type.

    ``TAP`` is not returned: it is a bat movement with no foot displacement, and the bat is
    not tracked.
    """
    if not detected:
        return TriggerType.NONE, 1.0

    import numpy as np

    from pose_extraction import KEYPOINT_INDEX, to_lead_rear

    ankle = KEYPOINT_INDEX[to_lead_rear("lead_ankle", handedness)]
    start = seq.xy[stance_frame, ankle]
    mid = seq.xy[(stance_frame + ss_frame) // 2, ankle]
    if np.isnan(start).any() or np.isnan(mid).any():
        return TriggerType.OTHER, 0.3

    # Player faces +x, so a foot moving back is negative along the crease axis.
    moved_back = float(mid[0] - start[0]) < 0
    confidence = float(
        min(seq.confidence[stance_frame, ankle], seq.confidence[ss_frame, ankle])
    )
    return (
        TriggerType.BACK_AND_ACROSS if moved_back else TriggerType.FORWARD_PRESS
    ), confidence


def detect_stance_width(
    seq: PoseSequence,
    stance_frame: int,
    *,
    stature_px: float,
) -> tuple[StanceWidth, float]:
    """Band stance width relative to stature.

    Band edges are display conventions, not published norms - see the constants above. This
    is a descriptor and carries no score weight, so the edges only decide which word appears
    beside the player's own measurement.
    """
    import numpy as np

    from pose_extraction import KEYPOINT_INDEX

    left = seq.xy[stance_frame, KEYPOINT_INDEX["left_ankle"]]
    right = seq.xy[stance_frame, KEYPOINT_INDEX["right_ankle"]]
    if np.isnan(left).any() or np.isnan(right).any() or stature_px <= 0:
        raise ValueError("stance width unavailable: ankle keypoints missing")

    width = abs(float(left[0] - right[0])) / stature_px
    confidence = float(
        min(
            seq.confidence[stance_frame, KEYPOINT_INDEX["left_ankle"]],
            seq.confidence[stance_frame, KEYPOINT_INDEX["right_ankle"]],
        )
    )

    if width < STANCE_NARROW_MAX:
        return StanceWidth.NARROW, confidence
    if width > STANCE_WIDE_MIN:
        return StanceWidth.WIDE, confidence
    return StanceWidth.MEDIUM, confidence
