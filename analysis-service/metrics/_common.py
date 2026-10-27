"""Shared helpers for metric computation.

Two conventions apply to everything in :mod:`metrics`:

**Measure over a window, not a single frame.** Every event carries its own frame
uncertainty, so reading a joint angle at exactly one frame inherits that uncertainty in
full. Averaging a few frames either side costs nothing on a smooth trajectory and removes
the single-frame sensitivity.

**Return ``None`` rather than a fallback.** A metric absent from the results is visible to
the player as "not measured"; a metric computed from a missing event is indistinguishable
from a real one. There is no sensible default for a joint angle.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

import metric_tiers
import uncertainty
from normalisation import BodyScale, joint_angle
from phase_detection import PhaseTimeline
from pose_extraction import KEYPOINT_INDEX, PoseSequence
from uncertainty import Measurement

#: Frames either side of an event to average over. At 240 fps this is ~17 ms, short enough
#: that a fast joint has not moved far, long enough to suppress single-frame noise.
EVENT_HALF_WINDOW = 2


def measure(
    key: str,
    value: float,
    *,
    fps: float,
    confidence: float,
    units: str,
) -> Optional[Measurement]:
    """Build a Measurement with the registry's baseline SEM, scaled by tracking quality."""
    if value is None or not np.isfinite(value):
        return None
    spec = metric_tiers.get(key)
    sem = uncertainty.scale_sem_by_confidence(
        uncertainty.base_sem_for(spec, fps=fps, units=units), confidence
    )
    return Measurement(
        key=key, value=float(value), sem=sem, confidence=float(confidence), units=units
    )


def interval_measure(
    key: str,
    timeline: PhaseTimeline,
    start: str,
    end: str,
    *,
    units: str = "s",
) -> Optional[Measurement]:
    """Temporal metric between two events.

    Confidence is the *lower* of the two event confidences. An interval is only as
    trustworthy as its weaker endpoint, which matters because proxied events (impact,
    release) are capped well below foot contacts.
    """
    value = timeline.interval_s(start, end)
    if value is None:
        return None
    return measure(
        key,
        value,
        fps=timeline.fps,
        confidence=timeline.interval_confidence(start, end),
        units=units,
    )


def _window(timeline: PhaseTimeline, event: str, n_frames: int) -> Optional[tuple]:
    frame = timeline.frame_of(event)
    if frame is None:
        return None
    lo = max(0, frame - EVENT_HALF_WINDOW)
    hi = min(n_frames, frame + EVENT_HALF_WINDOW + 1)
    return (lo, hi) if hi > lo else None


def angle_at(
    seq: PoseSequence,
    timeline: PhaseTimeline,
    event: str,
    proximal: str,
    joint: str,
    distal: str,
) -> Optional[tuple]:
    """Mean joint angle around an event, with its tracking confidence.

    Returns ``(angle_deg, confidence)`` in the ``straight_180`` convention, or ``None``.
    """
    window = _window(timeline, event, seq.n_frames)
    if window is None:
        return None

    angles = [joint_angle(seq, proximal, joint, distal, f) for f in range(*window)]
    angles = [a for a in angles if np.isfinite(a)]
    if not angles:
        return None

    idx = [KEYPOINT_INDEX[n] for n in (proximal, joint, distal)]
    kp_conf = float(np.nanmean(seq.confidence[window[0] : window[1], idx]))
    event_conf = timeline.events[event].confidence

    return float(np.mean(angles)), min(kp_conf, event_conf)


def position_at(
    seq: PoseSequence,
    timeline: PhaseTimeline,
    event: str,
    keypoint: str,
    *,
    stature_px: Optional[float] = None,
) -> Optional[tuple]:
    """Mean position of a keypoint around an event, with confidence.

    Returns ``(xy, confidence)``.

    ``"nose"`` is routed through :func:`pose_extraction.head_point`, which tolerates a
    helmet. Reading the nose keypoint directly fails on exactly the footage this app is for -
    a helmet hides the whole face, so the model stacks every facial landmark in one spot at
    low confidence.
    """
    window = _window(timeline, event, seq.n_frames)
    if window is None:
        return None

    if keypoint == "nose":
        from pose_extraction import head_point

        points, confs = [], []
        for f in range(*window):
            result = head_point(seq, f, stature_px=stature_px)
            if result is not None:
                points.append(result[0])
                confs.append(result[1])
        if not points:
            return None
        return np.nanmean(points, axis=0), min(
            float(np.mean(confs)), timeline.events[event].confidence
        )

    idx = KEYPOINT_INDEX[keypoint]
    pts = seq.xy[window[0] : window[1], idx]
    if np.all(np.isnan(pts)):
        return None

    kp_conf = float(np.nanmean(seq.confidence[window[0] : window[1], idx]))
    event_conf = timeline.events[event].confidence
    return np.nanmean(pts, axis=0), min(kp_conf, event_conf)


def displacement_between(
    seq: PoseSequence,
    timeline: PhaseTimeline,
    start_event: str,
    end_event: str,
    keypoint: str,
    scale: BodyScale,
    *,
    axis: int,
    invert: bool = False,
) -> Optional[tuple]:
    """Displacement of a keypoint between two events, in stature fractions.

    ``invert`` negates the result, used for vertical measures so that "up" is positive -
    image ``y`` grows downward, and a head-lift metric reading negative for lifting would be
    a trap for every later reader.
    """
    a = position_at(seq, timeline, start_event, keypoint, stature_px=scale.stature_px)
    b = position_at(seq, timeline, end_event, keypoint, stature_px=scale.stature_px)
    if a is None or b is None:
        return None

    delta = float(b[0][axis] - a[0][axis]) / scale.stature_px
    if invert:
        delta = -delta
    return delta, min(a[1], b[1]) * scale.confidence


def angle_change_between(
    seq: PoseSequence,
    timeline: PhaseTimeline,
    start_event: str,
    end_event: str,
    proximal: str,
    joint: str,
    distal: str,
) -> Optional[tuple]:
    """Change in a joint angle between two events.

    The SEM for this is computed by :func:`uncertainty.difference_sem` rather than by adding
    variances, because the systematic component of pose error is similar at both events for
    a fixed camera and partly cancels in the difference.
    """
    a = angle_at(seq, timeline, start_event, proximal, joint, distal)
    b = angle_at(seq, timeline, end_event, proximal, joint, distal)
    if a is None or b is None:
        return None
    return b[0] - a[0], min(a[1], b[1])
