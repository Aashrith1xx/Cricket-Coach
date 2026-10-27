"""Bowling metric computation.

What is absent matters as much as what is here. There is **no** function for shoulder
counter-rotation, elbow extension, arm elevation, shoulder alignment or action-type
classification:

* ICC's legality threshold is 15 deg of elbow extension; measured elbow error is 22.7 deg.
* The mixed-action risk threshold is 30 deg of shoulder counter-rotation, derived from a
  variable whose error is 28.5 deg - larger than the threshold itself.

Telling a bowler their action is safe on that basis could contribute to a lumbar stress
fracture, and telling them it is unsafe would be an unfounded medical claim. Both are
refused. If you are here to add one of those, read ``docs/ACCURACY.md`` first.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from metrics._common import (
    angle_at,
    angle_change_between,
    interval_measure,
    measure,
    position_at,
)
from normalisation import BodyScale
from phase_detection import PhaseTimeline
from pose_extraction import KEYPOINT_INDEX, PoseSequence, to_lead_rear
from uncertainty import Measurement

# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------


def back_to_front_foot_interval(timeline: PhaseTimeline) -> Optional[Measurement]:
    """BFC to FFC - the delivery stride duration.

    The most reliable bowling measurement available: both endpoints are foot contacts.
    ``elitefastbowlers2024`` associates reduced back-foot contact time with faster release.
    """
    return interval_measure("bowling.interval_bfc_ffc", timeline, "BFC", "FFC")


def front_foot_to_release_interval(timeline: PhaseTimeline) -> Optional[Measurement]:
    """FFC to BR.

    Depends on BR, which without ball tracking is proxied by peak wrist speed and is the
    weakest event in the bowling pipeline. That confidence is propagated rather than
    assumed - several metrics anchor to BR and would otherwise inherit unearned precision.
    """
    return interval_measure("bowling.interval_ffc_br", timeline, "FFC", "BR")


def runup_tempo(timeline: PhaseTimeline) -> Optional[Measurement]:
    """Ratio of the penultimate stride to the delivery stride.

    A ratio rather than an absolute duration, so it is scale-free and comparable between
    bowlers with very different run-up lengths.
    """
    penultimate = timeline.interval_s("PRE_DELIVERY_STRIDE", "BFC")
    delivery = timeline.interval_s("BFC", "FFC")
    if penultimate is None or delivery is None or delivery <= 0:
        return None

    confidence = min(
        timeline.interval_confidence("PRE_DELIVERY_STRIDE", "BFC"),
        timeline.interval_confidence("BFC", "FFC"),
    )
    return measure(
        "bowling.runup_final_stride_tempo",
        penultimate / delivery,
        fps=timeline.fps,
        confidence=confidence,
        units="ratio",
    )


# ---------------------------------------------------------------------------
# Angles - sagittal plane only
# ---------------------------------------------------------------------------


def _front_leg(handedness: str):
    return (
        to_lead_rear("lead_hip", handedness),
        to_lead_rear("lead_knee", handedness),
        to_lead_rear("lead_ankle", handedness),
    )


def front_knee_at_ffc(
    seq: PoseSequence, timeline: PhaseTimeline, *, handedness: str
) -> Optional[Measurement]:
    """Front knee angle at front-foot contact, ``straight_180``.

    Reported as a technique variable only. The well-known <30 deg flexion threshold comes
    from lumbar-loading research, and presenting it as a target would launder a safety
    judgement into a form score.
    """
    hip, knee, ankle = _front_leg(handedness)
    result = angle_at(seq, timeline, "FFC", hip, knee, ankle)
    if result is None:
        return None
    angle, confidence = result
    return measure(
        "bowling.front_knee_angle_ffc", angle, fps=timeline.fps, confidence=confidence, units="deg"
    )


def front_knee_at_release(
    seq: PoseSequence, timeline: PhaseTimeline, *, handedness: str
) -> Optional[Measurement]:
    """Front knee angle at release. ``elitefastbowlers2024``: correlates with release speed
    at r = 0.40, faster bowlers releasing with a more extended front knee."""
    hip, knee, ankle = _front_leg(handedness)
    result = angle_at(seq, timeline, "BR", hip, knee, ankle)
    if result is None:
        return None
    angle, confidence = result
    return measure(
        "bowling.front_knee_angle_br", angle, fps=timeline.fps, confidence=confidence, units="deg"
    )


def front_knee_bracing(
    seq: PoseSequence, timeline: PhaseTimeline, *, handedness: str
) -> Optional[Measurement]:
    """Knee angle change from contact to release - bracing versus collapse.

    The highest-value bowling metric available: a difference of two well-measured sagittal
    angles, so systematic error partly cancels, and it captures the mechanism the literature
    associates with release speed. Positive means the knee extended.
    """
    hip, knee, ankle = _front_leg(handedness)
    result = angle_change_between(seq, timeline, "FFC", "BR", hip, knee, ankle)
    if result is None:
        return None
    delta, confidence = result
    return measure(
        "bowling.front_knee_change_ffc_br",
        delta,
        fps=timeline.fps,
        confidence=confidence,
        units="deg",
    )


def _trunk_angle_from_vertical(seq: PoseSequence, frame: int, axis: int) -> float:
    """Angle of the trunk from vertical, in the plane given by ``axis``.

    Trunk is the hip-midpoint to shoulder-midpoint vector; using midpoints rather than one
    side halves the effect of a single badly-tracked joint.
    """
    idx = [KEYPOINT_INDEX[n] for n in ("left_hip", "right_hip")]
    hips = np.nanmean(seq.xy[frame, idx], axis=0)
    idx = [KEYPOINT_INDEX[n] for n in ("left_shoulder", "right_shoulder")]
    shoulders = np.nanmean(seq.xy[frame, idx], axis=0)

    if np.isnan(hips).any() or np.isnan(shoulders).any():
        return float("nan")

    horizontal = shoulders[axis] - hips[axis]
    vertical = hips[1] - shoulders[1]  # positive when upright, since y grows downward
    if vertical == 0:
        return float("nan")
    return float(np.degrees(np.arctan2(horizontal, vertical)))


def trunk_forward_flexion(
    seq: PoseSequence, timeline: PhaseTimeline
) -> Optional[Measurement]:
    """Forward trunk lean at release, degrees from vertical.

    Sagittal plane, so it lies in the image plane on the side-on camera the capture protocol
    specifies - the one trunk angle this geometry measures well. Reported in the
    ``straight_180`` sense by adding to 180 so that upright is 180, matching the internal
    convention used everywhere else.
    """
    frame = timeline.frame_of("BR")
    if frame is None:
        return None

    lo = max(0, frame - 2)
    hi = min(seq.n_frames, frame + 3)
    angles = [_trunk_angle_from_vertical(seq, f, 0) for f in range(lo, hi)]
    angles = [a for a in angles if np.isfinite(a)]
    if not angles:
        return None

    idx = [KEYPOINT_INDEX[n] for n in ("left_hip", "right_hip", "left_shoulder", "right_shoulder")]
    kp_conf = float(np.nanmean(seq.confidence[lo:hi, idx]))
    confidence = min(kp_conf, timeline.events["BR"].confidence)

    return measure(
        "bowling.trunk_forward_flexion_br",
        180.0 - abs(float(np.mean(angles))),
        fps=timeline.fps,
        confidence=confidence,
        units="deg",
    )


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------


def delivery_stride_length(
    seq: PoseSequence, timeline: PhaseTimeline, scale: BodyScale
) -> Optional[Measurement]:
    """Ankle-to-ankle separation at front-foot contact, in stature fractions."""
    left = position_at(seq, timeline, "FFC", "left_ankle")
    right = position_at(seq, timeline, "FFC", "right_ankle")
    if left is None or right is None:
        return None

    separation = abs(float(left[0][0] - right[0][0]))
    confidence = min(left[1], right[1]) * scale.confidence
    return measure(
        "bowling.delivery_stride_length_norm",
        scale.to_stature_fraction(separation),
        fps=timeline.fps,
        confidence=confidence,
        units="frac_stature",
    )


def head_over_front_foot(
    seq: PoseSequence, timeline: PhaseTimeline, scale: BodyScale, *, handedness: str
) -> Optional[Measurement]:
    """Horizontal offset of the head from the front ankle at contact."""
    ankle = to_lead_rear("lead_ankle", handedness)
    head = position_at(seq, timeline, "FFC", "nose")
    foot = position_at(seq, timeline, "FFC", ankle)
    if head is None or foot is None:
        return None

    offset = float(head[0][0] - foot[0][0]) / scale.stature_px
    confidence = min(head[1], foot[1]) * scale.confidence
    return measure(
        "bowling.head_over_front_foot_ffc",
        offset,
        fps=timeline.fps,
        confidence=confidence,
        units="frac_stature",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def compute_all(
    seq: PoseSequence,
    timeline: PhaseTimeline,
    scale: BodyScale,
    *,
    handedness: str,
    allowed: Optional[set] = None,
) -> List[Measurement]:
    """Compute every bowling metric available for this clip."""
    candidates = {
        "bowling.interval_bfc_ffc": lambda: back_to_front_foot_interval(timeline),
        "bowling.interval_ffc_br": lambda: front_foot_to_release_interval(timeline),
        "bowling.runup_final_stride_tempo": lambda: runup_tempo(timeline),
        "bowling.front_knee_angle_ffc": lambda: front_knee_at_ffc(
            seq, timeline, handedness=handedness
        ),
        "bowling.front_knee_angle_br": lambda: front_knee_at_release(
            seq, timeline, handedness=handedness
        ),
        "bowling.front_knee_change_ffc_br": lambda: front_knee_bracing(
            seq, timeline, handedness=handedness
        ),
        "bowling.trunk_forward_flexion_br": lambda: trunk_forward_flexion(seq, timeline),
        "bowling.delivery_stride_length_norm": lambda: delivery_stride_length(
            seq, timeline, scale
        ),
        "bowling.head_over_front_foot_ffc": lambda: head_over_front_foot(
            seq, timeline, scale, handedness=handedness
        ),
    }

    out: List[Measurement] = []
    for key, fn in candidates.items():
        if allowed is not None and key not in allowed:
            continue
        result = fn()
        if result is not None:
            out.append(result)
    return out
