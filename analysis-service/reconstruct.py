"""Compose a 3D skeleton from a side-on and a front-on clip.

**This is axis composition, not triangulation, and the distinction matters.**

True 3D triangulation needs synchronised shutters and calibrated cameras - known intrinsics
and known relative pose. Two phones propped up at a net have none of that. What they do give
is two roughly orthogonal views, and each view measures two axes well:

* the **side** camera sees height and depth-along-the-pitch
* the **front** camera sees height and lateral position

So the axes can be *composed* rather than triangulated: take lateral from the front clip,
depth from the side clip, and height from both. The result is a genuine 3D skeleton in the
sense that each axis came from a camera that could actually see it, and an approximation in
the sense that it assumes the cameras are near-perpendicular and far enough away for
projection to be roughly orthographic.

Consequently this output is for **visualisation only**. It does not feed the metric layer.
Every existing metric stays measured in the single view that can see it, because the
composition's assumptions are unvalidated and a metric derived from them would carry an
error nobody has characterised - exactly the kind of confident, unfounded number the tier
system exists to prevent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from normalisation import BodyScale
from phase_detection import PhaseTimeline
from pose_extraction import KEYPOINT_INDEX, KEYPOINT_NAMES, PoseSequence

log = logging.getLogger(__name__)

#: Events tried in order as the temporal anchor between the two clips. Foot contacts first:
#: they are the most reliably detected events in the system (~24-25 ms), and a sync error
#: propagates into every frame of the reconstruction.
SYNC_EVENTS: Tuple[str, ...] = ("SE", "SS", "FFC", "BFC", "IMP", "BR", "STANCE")

#: Joints used to place the origin. The pelvis is the most stable landmark available, and
#: anchoring there stops the whole skeleton translating when a limb is mistracked.
ORIGIN_JOINTS = ("left_hip", "right_hip")


@dataclass
class Reconstruction:
    """Composed 3D skeleton, in stature units, origin at the pelvis.

    ``xyz`` is ``(n_frames, 17, 3)`` as ``(lateral, height, depth)``. Height is positive up,
    unlike image coordinates, so the viewer needs no further sign juggling.
    """

    xyz: np.ndarray
    confidence: np.ndarray
    fps: float
    sync_event: str
    sync_offset_s: float
    axes_from: Dict[str, str]

    @property
    def n_frames(self) -> int:
        return int(self.xyz.shape[0])

    def as_dict(self) -> dict:
        # Rounded hard: this is a visualisation payload, and four decimals of a
        # stature-normalised coordinate is already finer than the measurement supports.
        return {
            "n_frames": self.n_frames,
            "fps": self.fps,
            "keypoint_names": list(KEYPOINT_NAMES),
            "sync_event": self.sync_event,
            "sync_offset_s": round(self.sync_offset_s, 4),
            "axes_from": self.axes_from,
            "frames": [
                [[round(float(v), 4) for v in joint] for joint in frame]
                for frame in self.xyz
            ],
            "confidence": [
                [round(float(c), 2) for c in row] for row in self.confidence
            ],
        }


def _normalised(seq: PoseSequence, scale: BodyScale) -> np.ndarray:
    """Keypoints in stature units with the pelvis at the origin, y flipped so up is positive."""
    idx = [KEYPOINT_INDEX[n] for n in ORIGIN_JOINTS]
    origin = np.nanmean(seq.xy[:, idx, :], axis=1, keepdims=True)

    out = (seq.xy - origin) / scale.stature_px
    out[:, :, 1] *= -1.0  # image y grows downward; make up positive
    return out


def find_sync(
    side: PhaseTimeline, front: PhaseTimeline
) -> Optional[Tuple[str, float]]:
    """Temporal offset between the clips, from a shared event.

    Returns ``(event_name, offset_seconds)`` where offset is how far the front clip's
    timeline must shift to line up with the side clip's.

    Without hardware sync this is the only alignment available, and it is the weakest link in
    the reconstruction: an error here shows up as limbs that lead or lag the body. Foot
    contacts are preferred precisely because they are the events the detector locates best.
    """
    for name in SYNC_EVENTS:
        a, b = side.get(name), front.get(name)
        if a is not None and b is not None and a.time_s is not None and b.time_s is not None:
            return name, a.time_s - b.time_s
    return None


def _resample(front_xyz: np.ndarray, front_fps: float, times: np.ndarray, offset: float) -> np.ndarray:
    """Sample the front clip at the side clip's frame times, shifted by the sync offset."""
    front_times = np.arange(len(front_xyz)) / front_fps + offset
    out = np.full((len(times), front_xyz.shape[1], front_xyz.shape[2]), np.nan)

    for joint in range(front_xyz.shape[1]):
        for axis in range(front_xyz.shape[2]):
            series = front_xyz[:, joint, axis]
            valid = np.isfinite(series)
            if valid.sum() < 2:
                continue
            out[:, joint, axis] = np.interp(
                times, front_times[valid], series[valid], left=np.nan, right=np.nan
            )
    return out


def compose(
    side_seq: PoseSequence,
    side_scale: BodyScale,
    side_timeline: PhaseTimeline,
    front_seq: Optional[PoseSequence] = None,
    front_scale: Optional[BodyScale] = None,
    front_timeline: Optional[PhaseTimeline] = None,
) -> Reconstruction:
    """Build the skeleton, using the front clip when one is supplied.

    With only a side clip the lateral axis is reported as zero rather than guessed. That
    produces a flat skeleton in the viewer, which is the honest depiction: a single camera
    genuinely cannot resolve that axis, and inventing a plausible-looking spread would imply
    information nobody measured.
    """
    side_xyz_2d = _normalised(side_seq, side_scale)
    n = side_seq.n_frames

    xyz = np.zeros((n, 17, 3), dtype=np.float32)
    xyz[:, :, 1] = side_xyz_2d[:, :, 1]   # height, from the side clip
    xyz[:, :, 2] = side_xyz_2d[:, :, 0]   # depth along the pitch, from the side clip
    confidence = side_seq.confidence.copy()

    axes_from = {"lateral": "unavailable", "height": "side", "depth": "side"}
    sync_event, offset = "none", 0.0

    if front_seq is None or front_scale is None or front_timeline is None:
        return Reconstruction(xyz, confidence, side_seq.fps, sync_event, offset, axes_from)

    sync = find_sync(side_timeline, front_timeline)
    if sync is None:
        log.warning("No shared event between the clips - cannot align them, using side only")
        return Reconstruction(xyz, confidence, side_seq.fps, sync_event, offset, axes_from)

    sync_event, offset = sync
    front_xyz_2d = _normalised(front_seq, front_scale)
    times = np.arange(n) / side_seq.fps

    aligned = _resample(front_xyz_2d, front_seq.fps, times, offset)
    aligned_conf = _resample(
        front_seq.confidence[:, :, None], front_seq.fps, times, offset
    )[:, :, 0]

    xyz[:, :, 0] = np.nan_to_num(aligned[:, :, 0])          # lateral, from the front clip
    # Height is seen by both cameras. Averaging where both agree cuts the noise; where the
    # front clip has nothing, the side value stands rather than being pulled toward zero.
    both = np.isfinite(aligned[:, :, 1])
    xyz[:, :, 1] = np.where(
        both, (side_xyz_2d[:, :, 1] + np.nan_to_num(aligned[:, :, 1])) / 2.0, side_xyz_2d[:, :, 1]
    )

    confidence = np.where(
        np.isfinite(aligned_conf),
        np.minimum(side_seq.confidence, np.nan_to_num(aligned_conf)),
        side_seq.confidence * 0.7,   # lateral axis unmeasured on these frames
    )

    axes_from = {"lateral": "front", "height": "both", "depth": "side"}
    log.info("Composed 3D on %s, offset %.3fs", sync_event, offset)

    return Reconstruction(xyz, confidence, side_seq.fps, sync_event, offset, axes_from)
