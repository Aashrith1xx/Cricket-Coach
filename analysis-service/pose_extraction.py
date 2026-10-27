"""Pose extraction: RTMDet person detection followed by RTMPose keypoints.

**Status: specified stub.** Data structures are real; model inference is TODO.

Licensing, which constrains the implementation
----------------------------------------------
RTMPose is a **top-down** model: it needs a person detector to run first. Pair it **only
with RTMDet**. Both are Apache 2.0 via OpenMMLab, so commercial use is unencumbered.

Do **not** substitute an Ultralytics YOLOv5/v8 detector. Those are AGPL-3.0, and linking one
into this service would require open-sourcing the whole application or buying a commercial
licence. This is easy to do by accident because Ultralytics is the most convenient detector
in most tutorials.

Why RTMPose rather than MediaPipe
---------------------------------
Processing is offline, so there is no latency pressure and no reason to accept MediaPipe's
lower accuracy. RTMPose-m reaches 75.8% AP on COCO at 90+ FPS on a desktop CPU, and
benchmarks above MediaPipe. On a fractional-vCPU cloud instance expect far less than 90 FPS -
plan for roughly 1-2 minutes per 3-second 120 fps clip, which is why the worker is async.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

# COCO-17 keypoint ordering, which both RTMPose and the intent classifier of
# `intent2025posture` operate on.
KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

KEYPOINT_INDEX: Dict[str, int] = {n: i for i, n in enumerate(KEYPOINT_NAMES)}

#: Keypoints without which a clip cannot be analysed at all, per discipline. Used by the
#: capture gate's occlusion check.
CRITICAL_KEYPOINTS = {
    "batting": ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "nose"),
    "bowling": ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"),
}

DEFAULT_DETECTOR = "rtmdet-nano"
DEFAULT_POSE_MODEL = "rtmpose-m"


@dataclass
class PoseSequence:
    """Per-frame keypoints for one tracked person.

    Attributes
    ----------
    xy:
        ``(n_frames, 17, 2)`` pixel coordinates.
    confidence:
        ``(n_frames, 17)`` per-keypoint scores in ``[0, 1]``. Carried all the way through
        to scoring - low-confidence frames widen intervals rather than being silently
        interpolated over and treated as certain.
    fps:
        Effective frame rate. Everything temporal depends on this being right; read it from
        the container rather than assuming.
    """

    xy: np.ndarray
    confidence: np.ndarray
    fps: float
    bbox: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.xy.ndim != 3 or self.xy.shape[1:] != (17, 2):
            raise ValueError(f"xy must be (n_frames, 17, 2), got {self.xy.shape}")
        if self.confidence.shape != self.xy.shape[:2]:
            raise ValueError(
                f"confidence must be (n_frames, 17), got {self.confidence.shape}"
            )
        if self.fps <= 0:
            raise ValueError("fps must be positive")

    @property
    def n_frames(self) -> int:
        return int(self.xy.shape[0])

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps

    def keypoint(self, name: str) -> np.ndarray:
        """``(n_frames, 2)`` trajectory for one named keypoint."""
        return self.xy[:, KEYPOINT_INDEX[name], :]

    def keypoint_confidence(self, name: str) -> np.ndarray:
        return self.confidence[:, KEYPOINT_INDEX[name]]

    def mean_confidence(self, names: Sequence[str] | None = None) -> float:
        idx = (
            [KEYPOINT_INDEX[n] for n in names]
            if names
            else list(range(len(KEYPOINT_NAMES)))
        )
        return float(np.mean(self.confidence[:, idx]))

    def window(self, start: int, end: int) -> "PoseSequence":
        return PoseSequence(
            xy=self.xy[start:end],
            confidence=self.confidence[start:end],
            fps=self.fps,
            bbox=None if self.bbox is None else self.bbox[start:end],
        )


#: Longest frame dimension retained. Raw frames dominate memory: a 3-second 1080p clip at
#: 240 fps is 720 frames of uncompressed BGR, about 4.3 GB, which OOMs any small instance.
#: Downscaling to 720 puts the same clip near 210 MB.
#:
#: Keypoint precision scales with how many pixels the subject occupies, not with the frame
#: size, so this is safe as long as the player fills the frame - which the capture protocol
#: already requires. A subject filling 60% of a 720px frame is ~430px tall, and the
#: verification photo tracked at 0.83 mean confidence at roughly 500px.
DEFAULT_MAX_DIMENSION = 720

#: Hard cap on decoded frames, independent of resolution.
DEFAULT_MAX_FRAMES = 900


def read_frames(
    video_path: str,
    *,
    max_frames: int = DEFAULT_MAX_FRAMES,
    max_dimension: Optional[int] = DEFAULT_MAX_DIMENSION,
) -> tuple[list, float, int, int]:
    """Decode a video to a list of BGR frames, downscaling to bound memory.

    Returns ``(frames, fps, width, height)`` where the dimensions are those of the frames
    actually returned, not the source file - callers normalise against them.

    The fps is read from the container rather than assumed: every temporal metric scales
    with it, and a wrong value would silently corrupt all of them while looking entirely
    plausible. Downscaling deliberately does **not** touch fps, since temporal resolution is
    the expensive thing to acquire and the cheap thing to preserve.

    Pass ``max_dimension=None`` to keep the source resolution, for offline work on a machine
    with memory to spare.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0 or not np.isfinite(fps):
            raise ValueError(
                f"Could not read a frame rate from {video_path}. Refusing to assume one - "
                f"every timing metric depends on it."
            )

        scale = 1.0
        if max_dimension and max(src_w, src_h) > max_dimension:
            scale = max_dimension / max(src_w, src_h)

        frames = []
        truncated = False
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if len(frames) >= max_frames:
                truncated = True
                break
            if scale < 1.0:
                frame = cv2.resize(
                    frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                )
            frames.append(frame)
    finally:
        cap.release()

    if not frames:
        raise IOError(f"No frames decoded from {video_path}")

    if truncated:
        log.warning(
            "Clip truncated to %d frames (%.1fs at %.0f fps). Trim it to a single action.",
            max_frames,
            max_frames / fps,
            fps,
        )

    height, width = frames[0].shape[:2]
    return frames, fps, width, height


def extract(
    video_path: str,
    *,
    backend: Optional["PoseBackend"] = None,
    detector_path: Optional[str] = None,
    pose_model_path: Optional[str] = None,
) -> PoseSequence:
    """Run detection and pose estimation over a clip.

    When ``backend`` is omitted an :class:`~pose_backends.OnnxPoseBackend` is constructed
    from the given model paths, or from ``models/rtmdet.onnx`` and ``models/rtmpose.onnx``.

    Keypoints are never fabricated: frames where the model returned nothing carry NaN with
    zero confidence, and :mod:`smoothing` decides whether the gap is bridgeable. Filling
    gaps here would hide exactly what the uncertainty model needs to see.
    """
    from pose_backends import OnnxPoseBackend

    frames, fps, _width, _height = read_frames(video_path)

    if backend is None:
        backend = OnnxPoseBackend(
            detector_path or "models/rtmdet.onnx",
            pose_model_path or "models/rtmpose.onnx",
        )

    xy, confidence, bboxes = backend.run(frames)
    return PoseSequence(xy=xy, confidence=confidence, fps=fps, bbox=bboxes)


def extract_streaming(
    video_path: str,
    backend,
    *,
    max_frames: int = DEFAULT_MAX_FRAMES,
    scale_to: Optional[float] = None,
) -> PoseSequence:
    """Estimate pose at the video's own resolution, one frame at a time.

    Buffering every frame to run pose on them forces a resolution/memory trade-off that
    costs accuracy badly: downscaling a 1916x1000 clip to 720px shrank the player from about
    600 px tall to 225 px, and since RTMPose crops the player and resizes to 192x256, that
    turned a downsample into an *upsample* of a much poorer source. Tracking confidence fell
    accordingly.

    Streaming removes the trade-off. Only one frame is resident at a time - a few megabytes -
    so pose runs at full resolution regardless of clip length, and rendering can separately
    use a cheap downscaled copy.

    ``scale_to`` rescales the returned keypoints into another frame's coordinate space, so
    they can be drawn over downscaled frames.
    """
    import cv2

    from pose_backends import N_KEYPOINTS

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or not np.isfinite(fps):
            raise ValueError(f"Could not read a frame rate from {video_path}")

        xy_rows, conf_rows, box_rows = [], [], []
        state: dict = {}
        index = 0

        while index < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            kp, kc, box = backend.process_frame(frame, index, state)
            xy_rows.append(kp)
            conf_rows.append(kc)
            box_rows.append(box if box is not None else np.full(4, np.nan, dtype=np.float32))
            index += 1
    finally:
        cap.release()

    if not xy_rows:
        raise IOError(f"No frames decoded from {video_path}")

    xy = np.stack(xy_rows).astype(np.float32)
    if scale_to and scale_to != 1.0:
        xy *= scale_to

    return PoseSequence(
        xy=xy,
        confidence=np.stack(conf_rows).astype(np.float32),
        fps=fps,
        bbox=np.stack(box_rows).astype(np.float32),
    )


def extract_from_keypoints(keypoints_path: str, fps: float) -> PoseSequence:
    """Rebuild a sequence from stored keypoints, without the original video.

    Keypoints are retained permanently while video usually is not, so this is how a stored
    clip gets re-analysed after a scoring or template change.
    """
    from pose_backends import ReplayPoseBackend

    backend = ReplayPoseBackend(keypoints_path)
    xy, confidence, bboxes = backend.run([])
    return PoseSequence(xy=xy, confidence=confidence, fps=fps, bbox=bboxes)


def infer_handedness(seq: PoseSequence, *, view: str = "side_on") -> str:
    """Estimate ``"right"`` or ``"left"`` handed stance.

    Needed because published reference values are frequently reported as "left hip" or
    "right knee" rather than "lead" or "rear" - an ambiguity that blocks a metric outright
    in ``reference_templates/pull_shot.json``.

    Method: over the most confident early frames, compare mean wrist heights. The bottom
    hand on the bat handle sits lower, and for a right-handed batter that is the right hand.
    This is a weak signal, so the caller should prefer a value declared by the player and
    use this only as a fallback - the same principle as playstyle, where a declared value
    beats an inference that is right most of the time.
    """
    horizon = max(1, seq.n_frames // 3)
    left = seq.keypoint("left_wrist")[:horizon]
    right = seq.keypoint("right_wrist")[:horizon]
    left_conf = seq.keypoint_confidence("left_wrist")[:horizon]
    right_conf = seq.keypoint_confidence("right_wrist")[:horizon]

    usable = (left_conf > 0.4) & (right_conf > 0.4) & ~np.isnan(left[:, 1]) & ~np.isnan(right[:, 1])
    if usable.sum() < 3:
        return "right"  # documented fallback: the commoner case

    # Image y grows downward, so a larger y is a lower hand.
    return "right" if np.nanmean(right[usable, 1]) > np.nanmean(left[usable, 1]) else "left"


#: Head landmarks, best first. A batter in a helmet has no visible nose, eyes or ears, so the
#: model stacks those keypoints in roughly one spot with low confidence - which is exactly
#: what a cricket clip looks like, and exactly what breaks a head metric that reads `nose`.
HEAD_KEYPOINTS = ("nose", "left_ear", "right_ear", "left_eye", "right_eye")

#: Below this, a head landmark is not trusted and the shoulder-derived fallback is used.
HEAD_CONFIDENCE_FLOOR = 0.45

#: Head centre sits roughly this far above the shoulder midpoint, as a fraction of stature.
HEAD_ABOVE_SHOULDERS_STATURE_FRAC = 0.11


def head_point(seq: "PoseSequence", frame: int, *, stature_px: Optional[float] = None):
    """Best available estimate of head position, tolerant of helmets.

    Returns ``(xy, confidence)`` or ``None``.

    Prefers whichever facial landmark the model is actually confident about, and falls back
    to a point above the shoulder midpoint when none of them is trustworthy. Reading ``nose``
    directly - as the head metrics originally did - fails precisely on the footage this app
    exists for, since every batter facing genuine pace is wearing a helmet.

    The fallback is geometric rather than a guess at facial structure: the shoulders are
    among the best-tracked joints, and the head sits a roughly fixed proportion of stature
    above their midpoint.
    """
    best_xy, best_conf = None, 0.0
    for name in HEAD_KEYPOINTS:
        idx = KEYPOINT_INDEX[name]
        point = seq.xy[frame, idx]
        conf = float(seq.confidence[frame, idx])
        if np.isnan(point).any():
            continue
        if conf > best_conf:
            best_xy, best_conf = point, conf

    if best_conf >= HEAD_CONFIDENCE_FLOOR:
        return best_xy, best_conf

    left = seq.xy[frame, KEYPOINT_INDEX["left_shoulder"]]
    right = seq.xy[frame, KEYPOINT_INDEX["right_shoulder"]]
    if np.isnan(left).any() or np.isnan(right).any():
        return (best_xy, best_conf) if best_xy is not None else None

    midpoint = (left + right) / 2.0
    offset = (stature_px or 0.0) * HEAD_ABOVE_SHOULDERS_STATURE_FRAC
    estimate = np.array([midpoint[0], midpoint[1] - offset], dtype=np.float32)

    shoulder_conf = float(
        min(
            seq.confidence[frame, KEYPOINT_INDEX["left_shoulder"]],
            seq.confidence[frame, KEYPOINT_INDEX["right_shoulder"]],
        )
    )
    # Derived rather than observed, so it never claims full shoulder confidence.
    return estimate, shoulder_conf * 0.8


def to_lead_rear(name: str, handedness: str) -> str:
    """Map an anatomical keypoint name to lead/rear semantics.

    Implemented, because it is pure logic and is the single point where a handedness bug
    would silently invert every lateralised metric.
    """
    if handedness not in ("left", "right"):
        raise ValueError(f"handedness must be 'left' or 'right', got {handedness!r}")
    lead_side = "left" if handedness == "right" else "right"
    if name.startswith("lead_"):
        return name.replace("lead_", f"{lead_side}_", 1)
    if name.startswith("rear_"):
        rear_side = "right" if lead_side == "left" else "left"
        return name.replace("rear_", f"{rear_side}_", 1)
    return name
