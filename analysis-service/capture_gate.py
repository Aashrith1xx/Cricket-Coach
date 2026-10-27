"""Pre-analysis capture quality gate.

**Status: specified stub.** Structures and thresholds are real; the CV checks are TODO.

This runs *before* pose estimation and **rejects** non-conforming clips with a specific,
actionable reason. It does not analyse them at reduced confidence.

That is a deliberate product decision, not caution for its own sake. The OpenCap cricket
study lost 44% of trials outright even with three synchronised iPads at 240 Hz. A pipeline
that silently analyses whatever it is given will therefore spend a large fraction of its time
producing feedback that looks identical to good feedback and is wrong. Rejecting with
"your camera was at roughly 40 degrees off side-on, please reshoot square to the crease" is
more useful to a player than a confident number derived from an unmeasurable angle.

Ordering matters: the cheap metadata checks run first so an obviously unusable clip fails in
milliseconds rather than after a minute of pose extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import logging

import numpy as np

# --- Thresholds -------------------------------------------------------------

#: Hard frame-rate floor. Below this, release and impact events span too few frames to
#: locate at all (docs/ACCURACY.md s3). Clips below are not rejected outright - timing-only
#: metrics remain valid - but every impact-referenced metric is suppressed.
MIN_FPS_FOR_EVENT_METRICS = 120

#: Below this the clip is rejected entirely; even gross phase segmentation is unreliable.
MIN_FPS_ABSOLUTE = 50

#: Subject must fill at least this fraction of frame height. Small subjects mean few pixels
#: per limb, and keypoint error scales inversely with subject size in frame.
MIN_SUBJECT_HEIGHT_FRACTION = 0.35

#: Maximum tolerated deviation from the required camera view, in degrees. Angles must lie in
#: the image plane to be measurable; beyond this the sagittal metrics degrade sharply.
MAX_VIEW_ANGLE_DEVIATION_DEG = 20.0

#: Maximum tolerated background motion between frames, as a fraction of frame width.
#: Detects handheld panning, which corrupts every position-derived metric.
MAX_BACKGROUND_MOTION_FRACTION = 0.004

#: Minimum mean keypoint confidence over the frames spanning the action.
MIN_MEAN_KEYPOINT_CONFIDENCE = 0.55

#: Fraction of action-window frames that may have a missing critical keypoint.
MAX_OCCLUSION_FRACTION = 0.15

log = logging.getLogger(__name__)

MIN_CLIP_DURATION_S = 1.0
MAX_CLIP_DURATION_S = 30.0


class RejectionCode(str, Enum):
    FPS_TOO_LOW = "fps_too_low"
    SUBJECT_TOO_SMALL = "subject_too_small"
    NO_SUBJECT_DETECTED = "no_subject_detected"
    MULTIPLE_SUBJECTS = "multiple_subjects"
    CAMERA_MOTION = "camera_motion"
    WRONG_VIEW_ANGLE = "wrong_view_angle"
    POOR_TRACKING = "poor_tracking"
    OCCLUDED_ACTION = "occluded_action"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    NO_ACTION_DETECTED = "no_action_detected"
    UNREADABLE = "unreadable"


#: Player-facing remedies. Every rejection must map to something the user can actually do
#: differently next time - a bare "clip rejected" trains people to stop using the app.
REMEDIES = {
    RejectionCode.FPS_TOO_LOW: (
        "Record in slow motion. Your phone's camera settings should offer 120 or 240 fps - "
        "at normal speed the moment of release or contact happens between frames."
    ),
    RejectionCode.SUBJECT_TOO_SMALL: (
        "Move the camera closer, or zoom in, so you fill most of the frame height."
    ),
    RejectionCode.NO_SUBJECT_DETECTED: "We could not find a player in this clip.",
    RejectionCode.MULTIPLE_SUBJECTS: (
        "More than one person is in frame. Record with just the player visible, or move "
        "others out of shot."
    ),
    RejectionCode.CAMERA_MOTION: (
        "The camera moved during the clip. Rest the phone on a tripod or something solid - "
        "handheld footage cannot be measured accurately."
    ),
    RejectionCode.WRONG_VIEW_ANGLE: (
        "The camera needs to be square to the player, not at an angle. See the setup guide "
        "for where to stand."
    ),
    RejectionCode.POOR_TRACKING: (
        "We could not track the player reliably - usually low light, a busy background, or "
        "clothing close in colour to the background."
    ),
    RejectionCode.OCCLUDED_ACTION: (
        "The player was hidden - by the stumps, a net, or another person - during the part "
        "of the action we need to see."
    ),
    RejectionCode.TOO_SHORT: "The clip is too short. Include the whole action.",
    RejectionCode.TOO_LONG: "The clip is too long. Trim it to a single delivery or shot.",
    RejectionCode.NO_ACTION_DETECTED: (
        "We could not find a shot or delivery in this clip."
    ),
    RejectionCode.UNREADABLE: "We could not read this video file.",
}


@dataclass(frozen=True)
class Rejection:
    code: RejectionCode
    detail: str

    @property
    def remedy(self) -> str:
        return REMEDIES[self.code]

    def as_dict(self) -> dict:
        return {"code": self.code.value, "detail": self.detail, "remedy": self.remedy}


@dataclass
class CaptureMetadata:
    """What we know about the clip before any CV runs."""

    fps: float
    width: int
    height: int
    duration_s: float
    declared_view: str  # "side_on" | "front_on"
    device: Optional[str] = None
    rotation_deg: int = 0


@dataclass
class GateResult:
    passed: bool
    rejections: List[Rejection] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    #: Metrics suppressed by capture conditions rather than by tier, e.g. impact-referenced
    #: metrics on a 60 fps clip. Passed to the metric layer so it does not attempt them.
    suppressed_metric_keys: List[str] = field(default_factory=list)
    effective_fps: Optional[float] = None
    estimated_view_deviation_deg: Optional[float] = None
    #: Which canonical view the clip was shot from, once pose is available.
    detected_view: Optional[str] = None
    has_mesh: bool = False

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "rejections": [r.as_dict() for r in self.rejections],
            "warnings": self.warnings,
            "suppressed_metric_keys": self.suppressed_metric_keys,
            "effective_fps": self.effective_fps,
            "estimated_view_deviation_deg": self.estimated_view_deviation_deg,
            "detected_view": self.detected_view,
            "has_mesh": self.has_mesh,
        }


def check_metadata(meta: CaptureMetadata) -> GateResult:
    """Cheap checks from container metadata alone. No decoding beyond the header.

    Implemented, because it needs no CV and catches the most common failure - a clip shot at
    30 fps - before a minute of pose extraction is spent on it.
    """
    result = GateResult(passed=True, effective_fps=meta.fps)

    if meta.duration_s < MIN_CLIP_DURATION_S:
        result.rejections.append(
            Rejection(RejectionCode.TOO_SHORT, f"{meta.duration_s:.1f}s")
        )
    if meta.duration_s > MAX_CLIP_DURATION_S:
        result.rejections.append(
            Rejection(RejectionCode.TOO_LONG, f"{meta.duration_s:.1f}s")
        )
    if meta.fps < MIN_FPS_ABSOLUTE:
        result.rejections.append(
            Rejection(
                RejectionCode.FPS_TOO_LOW,
                f"{meta.fps:.0f} fps, need at least {MIN_FPS_ABSOLUTE}",
            )
        )
    elif meta.fps < MIN_FPS_FOR_EVENT_METRICS:
        result.warnings.append(
            f"Recorded at {meta.fps:.0f} fps. Timing and posture metrics still work, but "
            f"anything measured at the moment of release or contact needs "
            f"{MIN_FPS_FOR_EVENT_METRICS} fps and will be skipped."
        )

    result.passed = not result.rejections
    return result


def estimate_camera_motion(frames: List, *, sample_stride: int = 3) -> float:
    """Median background displacement per frame, as a fraction of frame width.

    Sparse optical flow on corner features. The subject is not masked out, because doing so
    needs a detector and this check runs *before* the expensive models load. Instead the
    median across all tracked corners is used: a player filling the recommended ~35-50% of
    frame height still occupies a minority of trackable corners, so the median follows the
    background. That is a weaker check than masking would give, but it runs in milliseconds
    and catches the case it exists for - someone filming handheld.
    """
    import cv2

    if len(frames) < 2:
        return 0.0

    height, width = frames[0].shape[:2]
    displacements: List[float] = []

    for i in range(0, len(frames) - sample_stride, sample_stride):
        prev = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        curr = cv2.cvtColor(frames[i + sample_stride], cv2.COLOR_BGR2GRAY)

        corners = cv2.goodFeaturesToTrack(
            prev, maxCorners=120, qualityLevel=0.05, minDistance=20
        )
        if corners is None or len(corners) < 8:
            continue

        moved, status, _ = cv2.calcOpticalFlowPyrLK(prev, curr, corners, None)
        if moved is None:
            continue

        ok = status.ravel() == 1
        if ok.sum() < 8:
            continue

        deltas = np.linalg.norm(moved[ok] - corners[ok], axis=-1).ravel()
        displacements.append(float(np.median(deltas)) / sample_stride)

    if not displacements:
        return 0.0
    return float(np.median(displacements)) / width


def check_frames(
    video_path: str,
    meta: CaptureMetadata,
    *,
    required_view: str,
    frames: Optional[List] = None,
) -> GateResult:
    """Pre-pose gate: decodes frames and runs the cheap CV checks.

    Deliberately does *not* run person detection. Detection needs the models loaded, which
    is most of the cost this gate exists to avoid; subject size, tracking quality, occlusion
    and view angle are all checked afterwards by :func:`check_pose_quality`, which gets them
    from the keypoints for free.

    Every rejection found is reported, not just the first, so a player can fix everything in
    one reshoot rather than iterating.
    """
    from pose_extraction import read_frames

    result = check_metadata(meta)
    if not result.passed:
        return result

    if frames is None:
        try:
            frames, _fps, _w, _h = read_frames(video_path)
        except (IOError, ValueError) as exc:
            result.rejections.append(Rejection(RejectionCode.UNREADABLE, str(exc)))
            result.passed = False
            return result

    motion = estimate_camera_motion(frames)
    if motion > MAX_BACKGROUND_MOTION_FRACTION:
        result.rejections.append(
            Rejection(
                RejectionCode.CAMERA_MOTION,
                f"background moved {motion * 100:.2f}% of frame width per frame, "
                f"limit {MAX_BACKGROUND_MOTION_FRACTION * 100:.2f}%",
            )
        )

    result.passed = not result.rejections
    return result


#: Fraction of the clip, from the start, treated as the set-up window for judging camera
#: angle. Must be short: a batter rotates through the shot, so anything past the set-up
#: measures the player turning rather than where the camera is.
SETUP_WINDOW_FRACTION = 0.15


def estimate_view_deviation_deg(
    seq, required_view: str, discipline: str = "batting"
) -> Optional[float]:
    """Degrees away from the required camera view, from shoulder-line foreshortening.

    A shoulder line square to the camera spans its full anthropometric width - about 0.259
    of stature. Turned side-on it collapses toward zero. The ratio of observed to expected
    width is therefore the sine of the angle away from side-on.

    **Measured over the set-up window only, not the whole clip.** A batter rotates roughly
    70 degrees between stance and follow-through, so a whole-clip median blends camera
    geometry with the player's own rotation and produces a number that describes neither.
    On a real cover drive that error read 31 degrees - an average of 90 at stance and 20
    through the swing - and rejected a clip while naming a deviation that was never true at
    any instant.

    The estimate is coarse even done properly, which is why
    ``MAX_VIEW_ANGLE_DEVIATION_DEG`` is generous: it exists to catch someone filming from in
    front or from 45 degrees, not to enforce precision.
    """
    from normalisation import SEGMENT_FRACTIONS, estimate_scale

    try:
        scale = estimate_scale(seq)
    except Exception:  # noqa: BLE001
        return None

    from pose_extraction import KEYPOINT_INDEX

    horizon = max(1, int(seq.n_frames * SETUP_WINDOW_FRACTION))
    left = seq.xy[:horizon, KEYPOINT_INDEX["left_shoulder"]]
    right = seq.xy[:horizon, KEYPOINT_INDEX["right_shoulder"]]
    conf = np.minimum(
        seq.confidence[:horizon, KEYPOINT_INDEX["left_shoulder"]],
        seq.confidence[:horizon, KEYPOINT_INDEX["right_shoulder"]],
    )

    widths = np.abs(left[:, 0] - right[:, 0])
    usable = np.isfinite(widths) & (conf > 0.4)
    if not usable.any():
        return None

    observed = float(np.median(widths[usable])) / scale.stature_px
    ratio = float(np.clip(observed / SEGMENT_FRACTIONS["shoulder_width"], 0.0, 1.0))
    shoulders_open = float(np.degrees(np.arcsin(ratio)))

    # Batting and bowling invert this, and the inversion is the whole subtlety.
    #
    # A BOWLER runs at the batsman with shoulders across the direction of travel, like any
    # runner: a camera to their side sees the shoulder line edge-on and NARROW.
    #
    # A BATTER is already rotated 90 degrees by their stance - shoulders point down the
    # pitch - so a camera to their side sees the shoulder line broadside and WIDE.
    #
    # Assuming the runner case for both is what made this reject genuinely side-on batting
    # footage as "front-on".
    if discipline == "bowling":
        angle_from_side_on = shoulders_open
    else:
        angle_from_side_on = 90.0 - shoulders_open

    return angle_from_side_on if required_view == "side_on" else 90.0 - angle_from_side_on


def classify_view(seq, discipline: str = "batting") -> tuple:
    """Identify which canonical view a clip was shot from.

    Returns ``(view, deviation_deg)`` where ``view`` is ``"side_on"``, ``"front_on"`` or
    ``"oblique"``, and the deviation is degrees from the nearer canonical position.

    A clip is only refused when it sits **between** the two - an oblique camera measures
    neither plane cleanly, so nothing it produces would be trustworthy. Both canonical views
    are valid; they simply support different metrics, since a quantity is measurable only
    when it varies in the camera's image plane. Nets footage is very often shot from in
    front, and refusing all of it was a design error rather than a safety property.
    """
    deviation = estimate_view_deviation_deg(seq, "side_on", discipline)
    if deviation is None:
        return None, None

    if deviation <= MAX_VIEW_ANGLE_DEVIATION_DEG:
        return "side_on", deviation
    if (90.0 - deviation) <= MAX_VIEW_ANGLE_DEVIATION_DEG:
        return "front_on", 90.0 - deviation
    return "oblique", min(deviation, 90.0 - deviation)


#: Edge density in the static background above which a net or fence is assumed to be between
#: camera and player. Nets are the normal place people practise, so this is about handling
#: mesh rather than excluding it.
MESH_EDGE_DENSITY = 0.085

#: How much the tracking-confidence floor is relaxed when mesh is detected. Mesh depresses
#: the pose model's confidence output because parts of the body are genuinely occluded by the
#: cords - so the drop is real, and the response is to accept the clip while widening its
#: uncertainty rather than pretending the tracking was clean.
MESH_CONFIDENCE_ALLOWANCE = 0.12

#: Multiplier applied to every measurement's standard error on a clip shot through mesh.
MESH_UNCERTAINTY_INFLATION = 1.25


#: How much of the background's edge structure must survive over the player's body before
#: the mesh is judged to be IN FRONT of them rather than behind.
MESH_IN_FRONT_RATIO = 0.55


def detect_mesh(frames: List, *, sample: int = 12, boxes=None) -> tuple:
    """Detect a net or fence **between the camera and the player**.

    Returns ``(is_mesh, edge_density)``.

    Net behind the player is harmless - it is just background - and only mesh the camera
    shoots *through* degrades tracking. Measuring edge density across the whole frame cannot
    tell those apart, and flagged clips filmed from inside the net as though they were shot
    through it.

    The discriminator is occlusion. Where the player stands in front of background netting,
    their body hides it and the edge density inside their silhouette collapses. Where the
    netting is in front, its cords still cross the body and the density barely drops. So the
    test is the ratio of edge density inside the player's box in a real frame against the same
    region of the static scene:

    * ratio near 1 - the structure survives over the body, so the mesh is in front
    * ratio well below 1 - the body blanks it out, so the net is behind

    Falls back to the whole-frame measure when no player boxes are supplied, which is only
    the pre-pose call site.
    """
    import cv2

    if len(frames) < 3:
        return False, 0.0

    step = max(1, len(frames) // sample)
    indices = list(range(0, len(frames), step))[:sample]
    static = np.median(np.stack([frames[i] for i in indices]), axis=0).astype(np.uint8)

    static_edges = cv2.Canny(cv2.cvtColor(static, cv2.COLOR_BGR2GRAY), 60, 160)
    density = float(np.count_nonzero(static_edges)) / static_edges.size

    if density <= MESH_EDGE_DENSITY:
        return False, density

    if boxes is None:
        return True, density

    # Compare the player's own region against the background IN THE SAME FRAME, rather than
    # against the temporal median. A batter barely moves, so the median still contains them
    # and cannot serve as a clean background estimate - which made the median-based version
    # flag every clip.
    ratios = []
    for i in indices:
        box = boxes[i] if i < len(boxes) else None
        if box is None or not np.isfinite(box).all():
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        pad_x, pad_y = int((x2 - x1) * 0.2), int((y2 - y1) * 0.15)
        bx1, by1 = max(0, x1 + pad_x), max(0, y1 + pad_y)
        bx2, by2 = min(frames[i].shape[1], x2 - pad_x), min(frames[i].shape[0], y2 - pad_y)
        if bx2 - bx1 < 20 or by2 - by1 < 20:
            continue

        edges = cv2.Canny(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY), 60, 160)
        body = edges[by1:by2, bx1:bx2]

        outside = edges.copy()
        outside[by1:by2, bx1:bx2] = 0
        n_outside = edges.size - body.size
        if n_outside <= 0:
            continue

        background_density = float(np.count_nonzero(outside)) / n_outside
        if background_density < 0.01:
            continue
        ratios.append((float(np.count_nonzero(body)) / body.size) / background_density)

    if not ratios:
        return True, density

    ratio = float(np.median(ratios))
    in_front = ratio >= MESH_IN_FRONT_RATIO
    log.info(
        "Mesh: frame density %.3f, structure surviving over the player %.2f -> %s",
        density, ratio, "in front" if in_front else "behind (harmless)",
    )
    return in_front, density


def estimate_subject_height_fraction(seq, frame_height: Optional[int] = None) -> Optional[float]:
    """Fraction of frame height the player occupies, from the pose rather than the bbox.

    Derived from keypoint extent so it works on replayed keypoints too, where no detection
    box was stored. Uses the 90th-percentile extent rather than the maximum, so one stray
    frame does not set the figure.
    """
    ys = seq.xy[:, :, 1]
    finite = np.isfinite(ys)
    if not finite.any():
        return None

    per_frame = []
    for i in range(seq.n_frames):
        row = ys[i][finite[i]]
        if len(row) >= 6:
            per_frame.append(float(row.max() - row.min()))
    if not per_frame:
        return None

    extent = float(np.percentile(per_frame, 90))
    if frame_height is None:
        frame_height = int(np.nanmax(ys[finite])) if finite.any() else 0
    if not frame_height:
        return None
    return extent / frame_height


def check_pose_quality(
    seq,
    *,
    required_view: str = "any",
    discipline: str,
    frame_height: Optional[int] = None,
    has_mesh: bool = False,
) -> GateResult:
    """Post-pose gate: checks that need keypoints.

    Runs after extraction because subject size, tracking quality, occlusion and view angle
    all fall out of the pose sequence at no extra cost, whereas checking them beforehand
    would mean running a detector twice.

    ``required_view`` of ``"any"`` accepts either canonical view and reports which one was
    detected, so the caller can select the matching metric set. Passing a specific view
    additionally requires that one.
    """
    from pose_extraction import CRITICAL_KEYPOINTS, KEYPOINT_INDEX

    result = GateResult(passed=True, effective_fps=seq.fps)

    confidence_floor = MIN_MEAN_KEYPOINT_CONFIDENCE - (
        MESH_CONFIDENCE_ALLOWANCE if has_mesh else 0.0
    )
    mean_conf = seq.mean_confidence()
    if mean_conf < confidence_floor:
        detail = (
            f"mean keypoint confidence {mean_conf:.2f}, need {confidence_floor:.2f}"
        )
        if has_mesh:
            detail += " (floor already relaxed for net footage)"
        result.rejections.append(Rejection(RejectionCode.POOR_TRACKING, detail))

    if has_mesh:
        result.warnings.append(
            "Filmed through a net or fence. Analysis will run, but the mesh hides parts of "
            "the body, so every measurement is reported with wider error bars than a clean "
            "view would give."
        )

    critical = CRITICAL_KEYPOINTS.get(discipline, ())
    if critical:
        idx = [KEYPOINT_INDEX[n] for n in critical]
        missing = np.isnan(seq.xy[:, idx, 0]).any(axis=1)
        occluded_fraction = float(missing.mean())
        if occluded_fraction > MAX_OCCLUSION_FRACTION:
            result.rejections.append(
                Rejection(
                    RejectionCode.OCCLUDED_ACTION,
                    f"{occluded_fraction * 100:.0f}% of frames missing a required joint",
                )
            )

    detected_view, deviation = classify_view(seq, discipline)
    result.detected_view = detected_view
    result.estimated_view_deviation_deg = deviation

    if detected_view == "oblique":
        result.rejections.append(
            Rejection(
                RejectionCode.WRONG_VIEW_ANGLE,
                f"camera sits about {deviation:.0f} degrees away from both square-on and "
                f"front-on. An angle between the two measures neither plane cleanly.",
            )
        )
    elif detected_view and required_view not in ("any", detected_view):
        result.rejections.append(
            Rejection(
                RejectionCode.WRONG_VIEW_ANGLE,
                f"detected a {detected_view.replace('_', '-')} camera, "
                f"but {required_view.replace('_', '-')} was requested",
            )
        )

    # Subject size. Keypoint error scales inversely with how many pixels the player
    # occupies, so a distant figure produces confident-looking joints with large real error -
    # exactly the failure this gate exists to catch. Enforced, not merely warned about.
    if frame_height:
        fraction = estimate_subject_height_fraction(seq, frame_height)
        if fraction is not None and fraction < MIN_SUBJECT_HEIGHT_FRACTION:
            result.rejections.append(
                Rejection(
                    RejectionCode.SUBJECT_TOO_SMALL,
                    f"player fills {fraction * 100:.0f}% of frame height, "
                    f"need {MIN_SUBJECT_HEIGHT_FRACTION * 100:.0f}%",
                )
            )

    result.passed = not result.rejections
    return result


def suppressed_by_capture(fps: float, has_front_on: bool) -> List[str]:
    """Metric keys unavailable under these capture conditions.

    Thin wrapper over :func:`metric_tiers.applicable`, kept here so the worker has a single
    call site for "what can this clip support".
    """
    import metric_tiers

    out: List[str] = []
    for spec in metric_tiers.REGISTRY.values():
        if spec.min_fps is not None and fps < spec.min_fps:
            out.append(spec.key)
        elif spec.requires_front_on and not has_front_on:
            out.append(spec.key)
    return out
