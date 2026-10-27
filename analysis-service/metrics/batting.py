"""Batting metric computation.

Metric selection is driven by ``docs/ACCURACY.md``, not by what a coaching manual would
list. In particular the three strongest predictors of bat speed - pelvis-thorax separation,
lead elbow extension and wrist uncocking - are Tier C and deliberately absent, because they
sit in the worst-measured categories (22.7-28.9 deg RMSE). What remains is sagittal-plane
lower-body geometry, head kinematics, and timing.

Every function returns ``None`` when its events or keypoints are unavailable. See
``_common`` for why that is preferable to any fallback.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from metrics._common import (
    angle_at,
    angle_change_between,
    displacement_between,
    interval_measure,
    measure,
    position_at,
)
from normalisation import BodyScale, centre_of_mass
from phase_detection import PhaseTimeline
from pose_extraction import PoseSequence, head_point, to_lead_rear
from uncertainty import Measurement

# ---------------------------------------------------------------------------
# Temporal - highest score weight, most reliable
# ---------------------------------------------------------------------------


def stride_duration(timeline: PhaseTimeline) -> Optional[Measurement]:
    """SS to SE. Among the most reliable measurements the system makes: both endpoints are
    foot events, the same problem gait analysis solves at ~24-25 ms error."""
    return interval_measure("batting.stride_duration_ss_se", timeline, "SS", "SE")


def stride_to_impact_interval(timeline: PhaseTimeline) -> Optional[Measurement]:
    """SE to IMP. ``frontfootdrive1999`` reports the front foot planted ~0.06 s before
    impact for a cover drive.

    Depends on IMP, which is proxied from peak wrist speed rather than observed, so the
    interval inherits that lower confidence via :func:`interval_measure`."""
    return interval_measure("batting.interval_se_imp", timeline, "SE", "IMP")


def backlift_tempo(timeline: PhaseTimeline) -> Optional[Measurement]:
    """BC to DS - how long the backlift takes."""
    return interval_measure("batting.tempo_bc_ds", timeline, "BC", "DS")


def trigger_timing(timeline: PhaseTimeline) -> Optional[Measurement]:
    """Lead time from the trigger movement to the start of the stride.

    Positive, since the trigger precedes the stride. Trigger *presence* is a descriptor
    handled in :mod:`style_markers` and must never be graded - prevalence ranges from 47% in
    ODI cricket to 76% in the IPL, so absence is a style, not a fault."""
    return interval_measure("batting.trigger_timing", timeline, "TRIGGER", "SS")


# ---------------------------------------------------------------------------
# Angles - sagittal plane only
# ---------------------------------------------------------------------------


def lead_knee_angle_at_impact(
    seq: PoseSequence, timeline: PhaseTimeline, *, handedness: str
) -> Optional[Measurement]:
    """Lead knee angle at contact, ``straight_180`` convention.

    The best-measured joint available (7.87 deg RMSE), and ``pullshot_malaysia`` found
    lower-body angles to be the ones that discriminate skill."""
    hip = to_lead_rear("lead_hip", handedness)
    knee = to_lead_rear("lead_knee", handedness)
    ankle = to_lead_rear("lead_ankle", handedness)

    result = angle_at(seq, timeline, "IMP", hip, knee, ankle)
    if result is None:
        return None
    angle, confidence = result
    return measure(
        "batting.lead_knee_angle_imp", angle, fps=timeline.fps, confidence=confidence, units="deg"
    )


def lead_knee_extension(
    seq: PoseSequence, timeline: PhaseTimeline, *, handedness: str
) -> Optional[Measurement]:
    """Change in lead knee angle from stride end to contact. Defined by
    ``peploe2019technique``. Differencing cancels part of the systematic pose error."""
    hip = to_lead_rear("lead_hip", handedness)
    knee = to_lead_rear("lead_knee", handedness)
    ankle = to_lead_rear("lead_ankle", handedness)

    result = angle_change_between(seq, timeline, "SE", "IMP", hip, knee, ankle)
    if result is None:
        return None
    delta, confidence = result
    return measure(
        "batting.lead_knee_extension_se_imp",
        delta,
        fps=timeline.fps,
        confidence=confidence,
        units="deg",
    )


# ---------------------------------------------------------------------------
# Head - the strongest measurable batting discriminator
# ---------------------------------------------------------------------------


def head_lift(
    seq: PoseSequence, timeline: PhaseTimeline, scale: BodyScale
) -> Optional[Measurement]:
    """Vertical head movement through the downswing. Positive means the head rose.

    ``mann2013head`` establishes head-ball coupling as the strongest measurable batting
    discriminator (elite 2.4 +/- 1.7 deg mean head-ball angle against 3.7 +/- 1.4 deg for
    club players; maxima 10.7 vs 20.6 deg). The head is also the best-tracked region -
    large, high contrast, rarely self-occluded - so skill signal and measurability coincide,
    which is rare.

    We measure head-only components: the true head-ball angle needs ball tracking, which
    phone video does not support. Gaze is not inferred at all - elite batters hold gaze
    ahead of the ball and make two predictive saccades where club players make one, but that
    requires an eye tracker."""
    result = displacement_between(
        seq, timeline, "DS", "IMP", "nose", scale, axis=1, invert=True
    )
    if result is None:
        return None
    delta, confidence = result
    return measure(
        "batting.head_vertical_displacement_ds_imp",
        delta,
        fps=timeline.fps,
        confidence=confidence,
        units="frac_stature",
    )


def head_travel(
    seq: PoseSequence, timeline: PhaseTimeline, scale: BodyScale
) -> Optional[Measurement]:
    """Horizontal head movement through the downswing. ``headcom2007``: skilled batters
    hold the head further forward."""
    result = displacement_between(seq, timeline, "DS", "IMP", "nose", scale, axis=0)
    if result is None:
        return None
    delta, confidence = result
    return measure(
        "batting.head_horizontal_displacement_ds_imp",
        delta,
        fps=timeline.fps,
        confidence=confidence,
        units="frac_stature",
    )


def head_over_lead_knee(
    seq: PoseSequence, timeline: PhaseTimeline, scale: BodyScale, *, handedness: str
) -> Optional[Measurement]:
    """Horizontal offset of the head from the lead knee at contact. Tier B - meaningful,
    but sensitive to camera obliquity."""
    knee = to_lead_rear("lead_knee", handedness)
    head = position_at(seq, timeline, "IMP", "nose", stature_px=scale.stature_px)
    knee_pos = position_at(seq, timeline, "IMP", knee)
    if head is None or knee_pos is None:
        return None

    offset = float(head[0][0] - knee_pos[0][0]) / scale.stature_px
    confidence = min(head[1], knee_pos[1]) * scale.confidence
    return measure(
        "batting.head_over_lead_knee_imp",
        offset,
        fps=timeline.fps,
        confidence=confidence,
        units="frac_stature",
    )


# ---------------------------------------------------------------------------
# Position and base
# ---------------------------------------------------------------------------


def com_anterior_displacement(
    seq: PoseSequence, timeline: PhaseTimeline, scale: BodyScale
) -> Optional[Measurement]:
    """Forward centre-of-mass travel from its rearmost point to contact.

    Defined by ``peploe2019technique`` as COM displacement Y, minimum to impact. Uses the
    crude COM proxy from :mod:`normalisation`, which is adequate here because a constant
    offset cancels in a displacement."""
    imp = timeline.frame_of("IMP")
    start = timeline.frame_of("SS") or 0
    if imp is None:
        return None

    xs = []
    for f in range(start, min(imp + 1, seq.n_frames)):
        com = centre_of_mass(seq, f)
        if np.isfinite(com[0]):
            xs.append(com[0])
    if len(xs) < 2:
        return None

    delta = (xs[-1] - min(xs)) / scale.stature_px
    confidence = timeline.events["IMP"].confidence * scale.confidence
    return measure(
        "batting.com_anterior_displacement_min_imp",
        delta,
        fps=timeline.fps,
        confidence=confidence,
        units="frac_stature",
    )


def base_length(
    seq: PoseSequence, timeline: PhaseTimeline, scale: BodyScale
) -> Optional[Measurement]:
    """Ankle-to-ankle separation at contact, as a percentage of stature.

    ``peploe2019technique`` normalises base length the same way, so this is directly
    comparable to the published variable."""
    left = position_at(seq, timeline, "IMP", "left_ankle")
    right = position_at(seq, timeline, "IMP", "right_ankle")
    if left is None or right is None:
        return None

    separation = float(np.linalg.norm(left[0] - right[0]))
    confidence = min(left[1], right[1]) * scale.confidence
    return measure(
        "batting.base_length_pct_stature_imp",
        scale.to_stature_percent(separation),
        fps=timeline.fps,
        confidence=confidence,
        units="pct_stature",
    )


def stride_length(
    seq: PoseSequence, timeline: PhaseTimeline, scale: BodyScale, *, handedness: str
) -> Optional[Measurement]:
    """Lead ankle travel from stride start to stride end, in stature fractions.

    **Not scored as longer-is-better.** ``sarpeshkar2011`` reports that experienced batters
    deliberately take a short-to-medium stride so they can play the ball later, and
    ``pullshot_malaysia`` found no significant stride-length difference across skill levels.
    The reference band has an optimum, not a floor."""
    ankle = to_lead_rear("lead_ankle", handedness)
    result = displacement_between(seq, timeline, "SS", "SE", ankle, scale, axis=0)
    if result is None:
        return None
    delta, confidence = result
    return measure(
        "batting.stride_length_norm",
        abs(delta),
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
    """Compute every batting metric available for this clip.

    ``allowed`` restricts to keys the capture conditions support - see
    :func:`metric_tiers.applicable`. Metrics outside it are not attempted, so a low frame
    rate suppresses impact-anchored metrics rather than producing shaky ones.
    """
    candidates = {
        "batting.stride_duration_ss_se": lambda: stride_duration(timeline),
        "batting.interval_se_imp": lambda: stride_to_impact_interval(timeline),
        "batting.tempo_bc_ds": lambda: backlift_tempo(timeline),
        "batting.trigger_timing": lambda: trigger_timing(timeline),
        "batting.lead_knee_angle_imp": lambda: lead_knee_angle_at_impact(
            seq, timeline, handedness=handedness
        ),
        "batting.lead_knee_extension_se_imp": lambda: lead_knee_extension(
            seq, timeline, handedness=handedness
        ),
        "batting.head_vertical_displacement_ds_imp": lambda: head_lift(seq, timeline, scale),
        "batting.head_horizontal_displacement_ds_imp": lambda: head_travel(seq, timeline, scale),
        "batting.head_over_lead_knee_imp": lambda: head_over_lead_knee(
            seq, timeline, scale, handedness=handedness
        ),
        "batting.com_anterior_displacement_min_imp": lambda: com_anterior_displacement(
            seq, timeline, scale
        ),
        "batting.base_length_pct_stature_imp": lambda: base_length(seq, timeline, scale),
        "batting.stride_length_norm": lambda: stride_length(
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
