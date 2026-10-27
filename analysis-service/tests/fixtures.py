"""Synthetic pose sequences with known ground-truth events.

These let the pipeline be tested without model weights or real footage. They are not a
substitute for the calibration harness in ``validation/`` - synthetic motion is clean and
noise-free, so passing here proves the *algorithms* are correct, not that they work on real
clips. Both are needed and they check different things.

Coordinates are in pixels, side-on, player facing +x, image y growing downward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from pose_extraction import KEYPOINT_INDEX, PoseSequence

#: Roughly a 1.8 m player filmed to fill the frame.
STATURE_PX = 367.0

#: Neutral standing pose. Side-on, so left and right joints nearly coincide - separated
#: slightly so laterality logic has something to work with, as it would be on a real clip
#: where the far-side limb is visible past the near one.
BASE_POSE: Dict[str, tuple] = {
    "nose": (206.0, 60.0),
    "left_eye": (208.0, 56.0),
    "right_eye": (204.0, 56.0),
    "left_ear": (202.0, 58.0),
    "right_ear": (200.0, 58.0),
    # Wide: a batter is rotated 90 degrees in their stance, so a side-on camera sees the
    # shoulder line broadside at close to full anthropometric width (~0.259 of stature).
    "left_shoulder": (250.0, 110.0),
    "right_shoulder": (156.0, 112.0),
    "left_elbow": (232.0, 160.0),
    "right_elbow": (176.0, 162.0),
    "left_wrist": (216.0, 200.0),
    "right_wrist": (212.0, 196.0),
    "left_hip": (200.0, 220.0),
    "right_hip": (196.0, 221.0),
    "left_knee": (202.0, 310.0),
    "right_knee": (196.0, 311.0),
    "left_ankle": (204.0, 400.0),
    "right_ankle": (194.0, 401.0),
}


@dataclass
class SyntheticClip:
    seq: PoseSequence
    truth: Dict[str, int]
    stature_px: float = STATURE_PX


#: Frame rate the fixture timings were authored against. Motion is defined in seconds and
#: converted to frames at whatever rate is requested, so a 30 fps clip is the *same shot
#: sampled more coarsely* rather than the same shot played eight times slower. Getting this
#: wrong makes low-frame-rate tests measure the fixture instead of the detector.
REFERENCE_FPS = 240.0


def _f(t_ref_frames: float, fps: float) -> int:
    """Convert a reference-frame index into a frame index at the requested rate."""
    return int(round(t_ref_frames / REFERENCE_FPS * fps))


def _base_array(n_frames: int) -> np.ndarray:
    xy = np.zeros((n_frames, 17, 2), dtype=np.float32)
    for name, (x, y) in BASE_POSE.items():
        xy[:, KEYPOINT_INDEX[name], 0] = x
        xy[:, KEYPOINT_INDEX[name], 1] = y
    return xy


def _ramp(xy: np.ndarray, name: str, axis: int, start: int, end: int, delta: float) -> None:
    """Smoothly move a joint by ``delta`` between two frames, holding the new value after.

    Uses a raised-cosine profile rather than a linear ramp so velocity starts and ends at
    zero. A linear ramp would produce velocity step changes that the Savitzky-Golay filter
    turns into ringing, which is an artefact of the fixture rather than of the detector.
    """
    idx = KEYPOINT_INDEX[name]
    n = end - start
    if n <= 0:
        return
    t = np.linspace(0.0, 1.0, n)
    profile = 0.5 * (1.0 - np.cos(np.pi * t))
    xy[start:end, idx, axis] += delta * profile
    xy[end:, idx, axis] += delta


def _pulse(xy: np.ndarray, name: str, axis: int, start: int, end: int, delta: float) -> None:
    """Move a joint out and back, ending where it started."""
    idx = KEYPOINT_INDEX[name]
    n = end - start
    if n <= 0:
        return
    t = np.linspace(0.0, 1.0, n)
    profile = np.sin(np.pi * t)
    xy[start:end, idx, axis] += delta * profile


def batting_clip(
    *,
    fps: float = 240.0,
    with_trigger: bool = True,
    confidence: float = 0.9,
    noise_px: float = 0.0,
    seed: int = 0,
) -> SyntheticClip:
    """A right-handed front-foot drive.

    Timeline, in frames at the default 240 fps:

    ==========  ======  ==========================================
    Event       Frame   Motion
    ==========  ======  ==========================================
    STANCE      0-40    still
    TRIGGER     45-60   small back-and-return of the lead ankle
    BC          70      hands at their lowest, backlift begins
    SS          85      lead ankle starts forward
    DS          110     hands at their highest
    SE          125     lead ankle plants
    IMP         150     peak wrist speed
    ==========  ======  ==========================================
    """
    f = lambda t: _f(t, fps)  # noqa: E731 - local alias keeps the timeline readable
    n_frames = f(240)
    xy = _base_array(n_frames)

    # Lead side for a right-handed batter is the left.
    lead_ankle, lead_wrist = "left_ankle", "left_wrist"

    if with_trigger:
        # Small, and it returns - this is what distinguishes it from the stride.
        _pulse(xy, lead_ankle, 0, f(45), f(60), -12.0)

    # Backlift: hands rise, so y decreases.
    _ramp(xy, lead_wrist, 1, f(70), f(110), -70.0)
    _ramp(xy, "left_elbow", 1, f(70), f(110), -40.0)

    # Stride: lead ankle forward, and the body follows.
    _ramp(xy, lead_ankle, 0, f(85), f(125), 70.0)
    _ramp(xy, "left_knee", 0, f(85), f(125), 45.0)
    _ramp(xy, "left_hip", 0, f(85), f(125), 22.0)
    _ramp(xy, "right_hip", 0, f(85), f(125), 20.0)
    _ramp(xy, "nose", 0, f(85), f(125), 26.0)

    # Lead knee flexes as weight transfers forward.
    _ramp(xy, "left_knee", 1, f(85), f(125), 12.0)

    # Downswing into contact. The raised-cosine ramp peaks at its midpoint, so spanning
    # 110-190 puts peak wrist speed - the impact proxy - at reference frame 150.
    _ramp(xy, lead_wrist, 1, f(110), f(190), 95.0)
    _ramp(xy, lead_wrist, 0, f(110), f(190), 45.0)
    _ramp(xy, "left_elbow", 1, f(110), f(190), 55.0)

    # Follow-through, decelerating.
    _ramp(xy, lead_wrist, 1, f(190), f(235), -45.0)

    if noise_px > 0:
        rng = np.random.default_rng(seed)
        xy = xy + rng.normal(0.0, noise_px, xy.shape).astype(np.float32)

    conf = np.full((n_frames, 17), confidence, dtype=np.float32)

    truth = {
        "STANCE": f(42),
        "BC": f(70),
        "SS": f(85),
        "DS": f(110),
        "SE": f(125),
        "IMP": f(150),
    }
    if with_trigger:
        truth["TRIGGER"] = f(52)

    return SyntheticClip(
        seq=PoseSequence(xy=xy, confidence=conf, fps=fps),
        truth=truth,
    )


def bowling_clip(
    *,
    fps: float = 240.0,
    confidence: float = 0.9,
    noise_px: float = 0.0,
    seed: int = 0,
) -> SyntheticClip:
    """A right-arm fast bowling delivery.

    ==========  ======  ==========================================
    Event       Frame   Motion
    ==========  ======  ==========================================
    RUNUP       30      body starts moving forward
    BFC         90      back (right) ankle at its lowest point
    FFC         130     front (left) ankle plants
    BR          160     peak bowling-wrist speed
    ==========  ======  ==========================================
    """
    f = lambda t: _f(t, fps)  # noqa: E731
    n_frames = f(220)
    xy = _base_array(n_frames)

    front_ankle, back_ankle, bowl_wrist = "left_ankle", "right_ankle", "right_wrist"

    # A bowler runs at the batsman with shoulders ACROSS the direction of travel, like any
    # runner, so a side-on camera sees the shoulder line edge-on and narrow. This is the
    # opposite of a batter, whose stance has already rotated them 90 degrees - which is why
    # the view estimator has to know which discipline it is looking at.
    xy[:, KEYPOINT_INDEX["left_shoulder"], 0] = 204.0
    xy[:, KEYPOINT_INDEX["right_shoulder"], 0] = 198.0
    xy[:, KEYPOINT_INDEX["left_elbow"], 0] = 212.0
    xy[:, KEYPOINT_INDEX["right_elbow"], 0] = 208.0

    # Run-up: whole body travels forward. A real bowler covers several body-lengths of
    # ground - the original 120px (0.3 stature) was far too short and made the fixture look
    # like a stationary batter to anything reading whole-body travel.
    for name in ("left_hip", "right_hip", "nose", "left_shoulder", "right_shoulder"):
        _ramp(xy, name, 0, f(30), f(150), 700.0)

    # Back foot rises through the pre-delivery stride, then lands.
    _pulse(xy, back_ankle, 1, f(40), f(90), -35.0)
    _ramp(xy, back_ankle, 0, f(30), f(90), 560.0)

    # Front foot swings through and plants.
    _pulse(xy, front_ankle, 1, f(95), f(130), -40.0)
    _ramp(xy, front_ankle, 0, f(60), f(130), 760.0)

    # Front knee braces - extends slightly from contact to release.
    _ramp(xy, "left_knee", 0, f(60), f(130), 700.0)
    _ramp(xy, "left_knee", 1, f(130), f(160), -8.0)

    # Bowling arm: over the top, fastest around release.
    # The hand must pass clearly ABOVE the head - that is what makes a delivery a delivery,
    # and the original -120 left it below the nose, so the fixture had no bowling action in
    # it at all by the one signal that distinguishes bowling from batting.
    _ramp(xy, bowl_wrist, 1, f(100), f(145), -240.0)
    _ramp(xy, bowl_wrist, 1, f(145), f(175), 270.0)
    # The wrist travels with the BODY during the run-up, plus its own arm action. Folding
    # the translation into the arm ramp made body speed dominate wrist speed and moved the
    # release proxy - peak wrist speed - away from the actual release.
    _ramp(xy, bowl_wrist, 0, f(30), f(150), 700.0)
    _ramp(xy, bowl_wrist, 0, f(120), f(175), 70.0)

    if noise_px > 0:
        rng = np.random.default_rng(seed)
        xy = xy + rng.normal(0.0, noise_px, xy.shape).astype(np.float32)

    conf = np.full((n_frames, 17), confidence, dtype=np.float32)

    return SyntheticClip(
        seq=PoseSequence(xy=xy, confidence=conf, fps=fps),
        truth={"RUNUP": f(30), "BFC": f(90), "FFC": f(130), "BR": f(160)},
    )


def occluded_clip(base: Optional[SyntheticClip] = None, *, joint: str = "left_ankle",
                  start: int = 100, end: int = 140) -> SyntheticClip:
    """A clip with one joint dropped out, for testing gap handling."""
    clip = base or batting_clip()
    conf = clip.seq.confidence.copy()
    xy = clip.seq.xy.copy()
    idx = KEYPOINT_INDEX[joint]
    conf[start:end, idx] = 0.05
    xy[start:end, idx] = np.nan
    return SyntheticClip(
        seq=PoseSequence(xy=xy, confidence=conf, fps=clip.seq.fps),
        truth=clip.truth,
    )
