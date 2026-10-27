"""Anthropometric normalisation: pixels to body-segment units.

Every distance this service reports is expressed relative to the player's own body, never in
pixels and never in assumed metres. Two reasons:

* **Scale invariance.** A clip shot from 8 m and one from 12 m must yield the same stride
  measurement. Without a calibration object in frame there is no way to recover true metres
  from a monocular video, so absolute units would be a fiction dressed up with a unit label.
* **Comparability.** A 1.65 m and a 1.90 m player taking a proportionally identical stride
  should score identically. Reference values in the literature are frequently reported in
  metres without participant stature, which is exactly why several template entries in
  ``reference_templates/`` are blocked - see the pull-shot notes.

Estimated stature from pose is itself uncertain, so it is computed from the most reliably
tracked segments and at a pose where the player is upright.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pose_extraction import KEYPOINT_INDEX, PoseSequence

#: Segment lengths as a fraction of stature (Winter, standard anthropometry). Used to
#: reconstruct stature from whichever segments are visible, and to sanity-check the result.
SEGMENT_FRACTIONS = {
    "shoulder_to_hip": 0.288,
    "hip_to_knee": 0.245,
    "knee_to_ankle": 0.246,
    "shoulder_width": 0.259,
}

#: Reject a stature estimate that disagrees with the bounding-box height by more than this
#: fraction - usually a sign of bad keypoints rather than an unusual body shape.
STATURE_SANITY_TOLERANCE = 0.35


@dataclass(frozen=True)
class BodyScale:
    """Per-clip scale reference, in pixels.

    ``confidence`` reflects how many independent segments agreed. A scale derived from one
    visible segment is materially less trustworthy than one from four, and that propagates
    into every normalised metric.
    """

    stature_px: float
    shoulder_width_px: float
    confidence: float
    n_segments_used: int

    def to_stature_fraction(self, pixels: float) -> float:
        return pixels / self.stature_px

    def to_stature_percent(self, pixels: float) -> float:
        return 100.0 * pixels / self.stature_px


class ScaleError(RuntimeError):
    pass


def _segment_length(seq: PoseSequence, a: str, b: str, frame: int) -> Optional[float]:
    pa = seq.xy[frame, KEYPOINT_INDEX[a]]
    pb = seq.xy[frame, KEYPOINT_INDEX[b]]
    ca = seq.confidence[frame, KEYPOINT_INDEX[a]]
    cb = seq.confidence[frame, KEYPOINT_INDEX[b]]
    if np.isnan(pa).any() or np.isnan(pb).any() or min(ca, cb) < 0.4:
        return None
    return float(np.linalg.norm(pa - pb))


def estimate_scale(seq: PoseSequence, *, reference_frame: Optional[int] = None) -> BodyScale:
    """Estimate body scale in pixels.

    Prefers a frame where the player is upright and still - the stance phase for batting,
    the start of the run-up for bowling - because a segment measured mid-action is
    foreshortened by rotation out of the image plane, which would bias every normalised
    metric downstream in a way no later step could detect.

    Uses the median across several segment estimates rather than a single one, so a badly
    tracked ankle does not set the scale for the whole clip.
    """
    frame = reference_frame if reference_frame is not None else _default_reference_frame(seq)

    estimates: list[float] = []

    pairs = (
        ("left_shoulder", "left_hip", "shoulder_to_hip"),
        ("right_shoulder", "right_hip", "shoulder_to_hip"),
        ("left_hip", "left_knee", "hip_to_knee"),
        ("right_hip", "right_knee", "hip_to_knee"),
        ("left_knee", "left_ankle", "knee_to_ankle"),
        ("right_knee", "right_ankle", "knee_to_ankle"),
    )
    for a, b, key in pairs:
        length = _segment_length(seq, a, b, frame)
        if length:
            estimates.append(length / SEGMENT_FRACTIONS[key])

    shoulder_px = _segment_length(seq, "left_shoulder", "right_shoulder", frame)

    if not estimates:
        raise ScaleError(
            "No usable body segments in the reference frame; cannot establish scale. The "
            "clip should not have passed the capture gate."
        )

    stature_px = float(np.median(estimates))
    spread = float(np.std(estimates) / stature_px) if len(estimates) > 1 else 0.5
    confidence = max(0.0, min(1.0, 1.0 - spread))

    return BodyScale(
        stature_px=stature_px,
        # Fall back to the anthropometric ratio when the shoulder line is foreshortened,
        # which it always is on a true side-on view.
        shoulder_width_px=shoulder_px
        if shoulder_px
        else stature_px * SEGMENT_FRACTIONS["shoulder_width"],
        confidence=confidence,
        n_segments_used=len(estimates),
    )


def _default_reference_frame(seq: PoseSequence) -> int:
    """Frame with the highest mean keypoint confidence in the first quarter of the clip.

    The opening of a clip is the stance or run-up start - upright, unrotated, and the least
    foreshortened part of the action.
    """
    horizon = max(1, seq.n_frames // 4)
    means = seq.confidence[:horizon].mean(axis=1)
    return int(np.argmax(means))


def horizontal_distance(seq: PoseSequence, a: str, b: str, frame: int, scale: BodyScale) -> float:
    """Signed horizontal separation of two keypoints, in stature fractions.

    Horizontal-only, because on a side-on view the horizontal axis lies in the image plane
    and is therefore the measurable one. Resolved depth is not available from one camera.
    """
    pa = seq.xy[frame, KEYPOINT_INDEX[a]]
    pb = seq.xy[frame, KEYPOINT_INDEX[b]]
    return scale.to_stature_fraction(float(pa[0] - pb[0]))


def vertical_distance(seq: PoseSequence, a: str, b: str, frame: int, scale: BodyScale) -> float:
    """Signed vertical separation, in stature fractions.

    Image ``y`` grows downward, so this negates to make "up" positive - the convention
    metrics such as head lift are written against.
    """
    pa = seq.xy[frame, KEYPOINT_INDEX[a]]
    pb = seq.xy[frame, KEYPOINT_INDEX[b]]
    return scale.to_stature_fraction(float(-(pa[1] - pb[1])))


def joint_angle(
    seq: PoseSequence,
    proximal: str,
    joint: str,
    distal: str,
    frame: int,
) -> float:
    """Planar joint angle in degrees, ``straight = 180``.

    Matches the ``straight_180`` convention used by ``peploe2019technique`` and
    ``pullshot_malaysia``, which is the internal convention throughout this service - see
    :mod:`conventions`.

    This is a **projected** 2D angle. It equals the true anatomical angle only when the joint
    moves in the image plane, which is precisely why the capture protocol fixes camera
    position per form type and why out-of-plane angles are Tier C.
    """
    p = seq.xy[frame, KEYPOINT_INDEX[proximal]]
    j = seq.xy[frame, KEYPOINT_INDEX[joint]]
    d = seq.xy[frame, KEYPOINT_INDEX[distal]]

    v1 = p - j
    v2 = d - j
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0 or np.isnan(n1) or np.isnan(n2):
        return float("nan")

    cos = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def centre_of_mass(seq: PoseSequence, frame: int) -> np.ndarray:
    """Crude whole-body COM proxy from hip and shoulder midpoints.

    A segment-mass model would be more principled, but it needs segment orientations that
    are unreliable out of plane. This proxy is adequate for the one metric that uses it -
    *change* in anterior COM position, where a constant offset cancels.
    """
    idx = [
        KEYPOINT_INDEX[n]
        for n in ("left_hip", "right_hip", "left_shoulder", "right_shoulder")
    ]
    pts = seq.xy[frame, idx]
    weights = np.array([0.3, 0.3, 0.2, 0.2])
    if np.isnan(pts).any():
        return np.array([np.nan, np.nan])
    return np.average(pts, axis=0, weights=weights)
