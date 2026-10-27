"""Phase event detection.

Events are the backbone of this system. Temporal metrics carry the highest score weight
because event timing is the most reliable thing video gives us - roughly 24-25 ms median
absolute error, against 7.87 deg RMSE on even the best-measured angle from a far better
camera rig than ours. Every angle metric is also *anchored* to an event, so a mis-detected
event corrupts whatever depends on it.

Event names match ``peploe2019technique``'s marker-based protocol deliberately, so published
values map onto our measurements without a translation step. Where a source uses different
names - as ``pullshot_malaysia`` does - the mismatch is recorded in the template rather than
papered over.

Detection strategy
------------------
Foot-contact events (SS, SE, BFC, FFC) come from ankle kinematics and are the reliable ones:
this is the same problem gait analysis solves well. Bat and ball events (DS, IMP, BR) are
*proxied* from wrist kinematics because the bat and ball are not tracked, and they carry
markedly lower confidence as a result. That difference is propagated, not smoothed over -
an impact-anchored metric should not look as solid as a stride-timing one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pose_extraction import PoseSequence, to_lead_rear
from smoothing import condition, velocity
from uncertainty import temporal_sem

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event vocabulary
# ---------------------------------------------------------------------------

BATTING_EVENTS: Tuple[str, ...] = (
    "STANCE",
    "TRIGGER",
    "BC",   # backswing commencement
    "SS",   # stride start
    "SE",   # stride end (front foot planted)
    "DS",   # downswing commencement
    "IMP",  # bat-ball impact
    "FOLLOW_THROUGH",
)

BOWLING_EVENTS: Tuple[str, ...] = (
    "RUNUP",
    "PRE_DELIVERY_STRIDE",
    "BFC",  # back foot contact
    "FFC",  # front foot contact
    "BR",   # ball release
    "FOLLOW_THROUGH",
)

#: Events that cannot be located below the 120 fps floor. At 30 fps release spans 1-3 frames
#: and impact falls between frames entirely - these are omitted, never estimated.
HIGH_FPS_EVENTS: frozenset = frozenset({"IMP", "BR", "DS"})

# --- Detection thresholds ---------------------------------------------------
# Expressed in stature-normalised units per second, so they do not depend on camera distance
# or the player's size in frame.

#: Ankle speed above which the foot is considered moving.
FOOT_MOVING_SPEED = 0.45

#: Speed below which the whole body is considered still. Used only for walking out to the
#: onset and offset of a movement already found by displacement, never for deciding whether
#: the player is stationary - see STILLNESS_DISPLACEMENT for why.
STILLNESS_SPEED = 0.12

#: Stillness is judged by DISPLACEMENT over a window, not by instantaneous speed.
#:
#: Keypoint jitter is roughly constant per frame - about a pixel - so the apparent velocity
#: it produces scales with frame rate: 1 px of jitter at 240 fps reads as 240 px/s. A fixed
#: velocity threshold therefore becomes progressively impossible to satisfy the better your
#: capture gets, which is exactly backwards. On a real 240 fps nets clip the median hip
#: "speed" during a stationary stance was 0.31 stature/s against a 0.12 threshold, so no
#: stance was ever found and every batting clip was refused.
#:
#: Displacement over a window is immune to this: jitter averages out, real movement does not.
STILLNESS_DISPLACEMENT = 0.025

#: Window over which that displacement is measured, in seconds.
STILLNESS_WINDOW_S = 0.10

#: A trigger movement is smaller than a stride. Above this fraction of the eventual stride
#: displacement, it is the stride rather than a trigger.
TRIGGER_MAX_FRACTION_OF_STRIDE = 0.4

#: Minimum stance window, in seconds.
MIN_STANCE_DURATION_S = 0.15

#: Confidence assigned to proxied events. The bat and ball are not tracked, so DS, IMP and
#: BR are inferred from wrist kinematics; this caps how certain they may claim to be.
PROXY_EVENT_MAX_CONFIDENCE = 0.6

#: Minimum mean wrist confidence before wrist-driven events may be reported at all.
#:
#: On real cricket footage the wrists are by far the least stable joints - hands gripping a
#: bat, moving fastest, blurred, and often behind net mesh. Measured on a nets cover drive:
#: wrist confidence 0.39-0.51 with a 95th-percentile frame-to-frame jump of 0.11 stature,
#: against ~0.02 for shoulders, hips and knees. A "peak wrist speed" impact proxy computed
#: from that is detecting jitter, not contact.
#:
#: Below this floor DS, IMP and BR are reported undetected, which suppresses every
#: impact-anchored metric rather than letting them be derived from a teleporting keypoint.
#:
#: Set at 0.40 rather than 0.55 because stability is now checked DIRECTLY on the conditioned
#: trajectory. The higher floor was only ever a proxy for 'will this produce garbage events',
#: and a proxy is redundant once the thing itself is measured. Low confidence still flows
#: through to every error bar via scale_sem_by_confidence, so a marginal clip yields wide
#: intervals rather than silent precision.
MIN_WRIST_CONFIDENCE_FOR_PROXY_EVENTS = 0.40

#: Frame-to-frame movement, in stature units, above which a keypoint is judged to have
#: teleported rather than moved. Used to discount unusable wrist frames.
MAX_PLAUSIBLE_JUMP_STATURE = 0.08


def _keypoint_is_usable(seq: PoseSequence, name: str, scale_px: float) -> tuple:
    """Whether a keypoint is stable enough to drive an event. Returns ``(usable, mean_conf)``.

    Checks stability as well as confidence: a model can be moderately confident about a
    joint it is nonetheless placing somewhere different each frame, and it is the jumping
    that ruins a peak-speed event detector.
    """
    conf = float(np.nanmean(seq.keypoint_confidence(name)))

    # Judge the CONDITIONED trajectory, not the raw one. Physically impossible jumps are
    # rejected and bridged first, so what is assessed here is the signal the event detectors
    # will actually see. Testing the raw trace instead condemned every nets clip on jitter
    # that the pipeline already removes.
    traj = condition(
        seq.keypoint(name), seq.keypoint_confidence(name), seq.fps, scale_px=scale_px
    )
    xy = traj.xy
    jumps = np.linalg.norm(np.diff(xy, axis=0), axis=-1) / max(scale_px, 1.0)
    jumps = jumps[np.isfinite(jumps)]
    p95 = float(np.percentile(jumps, 95)) if len(jumps) else 0.0

    usable = conf >= MIN_WRIST_CONFIDENCE_FOR_PROXY_EVENTS and p95 <= MAX_PLAUSIBLE_JUMP_STATURE
    return usable, conf


@dataclass(frozen=True)
class PhaseEvent:
    """A detected event with explicit frame uncertainty.

    ``detected`` is False for an event the detector looked for and could not find. That is
    distinct from an event being inapplicable, and callers must not conflate them: a missing
    IMP means impact-anchored metrics are unavailable for this clip, which the player should
    be told rather than left to infer from absent numbers.
    """

    name: str
    frame: Optional[int]
    time_s: Optional[float]
    frame_sem: float
    time_sem_s: float
    confidence: float
    detected: bool = True

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "frame": self.frame,
            "time_s": self.time_s,
            "frame_sem": self.frame_sem,
            "time_sem_s": self.time_sem_s,
            "confidence": self.confidence,
            "detected": self.detected,
        }


@dataclass
class PhaseTimeline:
    events: Dict[str, PhaseEvent]
    fps: float

    def get(self, name: str) -> Optional[PhaseEvent]:
        event = self.events.get(name)
        return event if event and event.detected else None

    def frame_of(self, name: str) -> Optional[int]:
        event = self.get(name)
        return None if event is None else event.frame

    def interval_s(self, start: str, end: str) -> Optional[float]:
        a, b = self.get(start), self.get(end)
        if a is None or b is None or a.time_s is None or b.time_s is None:
            return None
        return b.time_s - a.time_s

    def interval_sem_s(self, start: str, end: str) -> Optional[float]:
        a, b = self.get(start), self.get(end)
        if a is None or b is None:
            return None
        return float(np.hypot(a.time_sem_s, b.time_sem_s))

    def interval_confidence(self, start: str, end: str) -> float:
        a, b = self.get(start), self.get(end)
        if a is None or b is None:
            return 0.0
        return min(a.confidence, b.confidence)

    def missing(self, required: Sequence[str]) -> List[str]:
        return [name for name in required if self.get(name) is None]

    def as_dict(self) -> dict:
        return {"fps": self.fps, "events": {k: v.as_dict() for k, v in self.events.items()}}


def make_event(
    name: str, frame: Optional[int], fps: float, confidence: float
) -> PhaseEvent:
    """Construct an event with uncertainty derived from frame rate.

    The uncertainty model is identical for every event regardless of how it was detected;
    detection quality is expressed through ``confidence``, not by adjusting the SEM.
    """
    if frame is None:
        return PhaseEvent(
            name=name,
            frame=None,
            time_s=None,
            frame_sem=float("inf"),
            time_sem_s=float("inf"),
            confidence=0.0,
            detected=False,
        )
    sem_s = temporal_sem(fps)
    return PhaseEvent(
        name=name,
        frame=int(frame),
        time_s=float(frame) / fps,
        frame_sem=sem_s * fps,
        time_sem_s=sem_s,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
    )


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------


def _trajectory(seq: PoseSequence, name: str, scale_px: float) -> Tuple[np.ndarray, np.ndarray]:
    """Smoothed, stature-normalised trajectory and speed for one keypoint."""
    traj = condition(
        seq.keypoint(name), seq.keypoint_confidence(name), seq.fps, scale_px=scale_px
    )
    xy = traj.xy / scale_px
    return xy, velocity(xy, seq.fps)


def _mean_confidence(seq: PoseSequence, names: Sequence[str], frame: int, half_window: int = 2) -> float:
    lo = max(0, frame - half_window)
    hi = min(seq.n_frames, frame + half_window + 1)
    vals = [seq.keypoint_confidence(n)[lo:hi] for n in names]
    if not vals:
        return 0.0
    return float(np.nanmean(np.concatenate(vals)))


def _stillness_mask(xy: np.ndarray, fps: float) -> np.ndarray:
    """Per-frame flag: has this joint stayed put over a short window around it?

    Measures positional *range* over the window rather than velocity, so per-frame keypoint
    jitter averages out instead of being multiplied by the frame rate.
    """
    half = max(1, int(STILLNESS_WINDOW_S * fps / 2))
    n = len(xy)
    mask = np.zeros(n, dtype=bool)

    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        window = xy[lo:hi]
        window = window[np.isfinite(window).all(axis=1)]
        if len(window) < 2:
            continue
        extent = float(np.max(np.ptp(window, axis=0)))
        mask[i] = extent < STILLNESS_DISPLACEMENT

    return mask


def _longest_still_window_from_mask(still: np.ndarray, fps: float) -> Optional[Tuple[int, int]]:
    """Longest contiguous run of still frames, subject to the minimum stance duration."""
    best: Optional[Tuple[int, int]] = None
    best_len = 0
    start: Optional[int] = None

    for i, flag in enumerate(still):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start > best_len:
                best, best_len = (start, i), i - start
            start = None
    if start is not None and len(still) - start > best_len:
        best = (start, len(still))

    if best is None or (best[1] - best[0]) / fps < MIN_STANCE_DURATION_S:
        return None
    return best


def _longest_still_window(speed: np.ndarray, fps: float, threshold: float) -> Optional[Tuple[int, int]]:
    """Longest run of frames below a speed threshold.

    Retained for callers that genuinely want a speed criterion. Stance detection does not -
    see :func:`_stillness_mask`.
    """
    still = speed < threshold
    best: Optional[Tuple[int, int]] = None
    best_len = 0
    start: Optional[int] = None

    for i, flag in enumerate(still):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start > best_len:
                best, best_len = (start, i), i - start
            start = None
    if start is not None and len(still) - start > best_len:
        best = (start, len(still))

    if best is None:
        return None
    if (best[1] - best[0]) / fps < MIN_STANCE_DURATION_S:
        return None
    return best


def _first_crossing_above(signal: np.ndarray, threshold: float, start: int = 0) -> Optional[int]:
    idx = np.nonzero(signal[start:] > threshold)[0]
    return int(idx[0] + start) if len(idx) else None


def _first_crossing_below(signal: np.ndarray, threshold: float, start: int = 0) -> Optional[int]:
    idx = np.nonzero(signal[start:] < threshold)[0]
    return int(idx[0] + start) if len(idx) else None


def _argmax_in(signal: np.ndarray, start: int, end: int) -> Optional[int]:
    lo, hi = max(0, start), min(len(signal), end)
    if hi <= lo:
        return None
    window = signal[lo:hi]
    if np.all(np.isnan(window)):
        return None
    return int(np.nanargmax(window) + lo)


def _argmin_in(signal: np.ndarray, start: int, end: int) -> Optional[int]:
    lo, hi = max(0, start), min(len(signal), end)
    if hi <= lo:
        return None
    window = signal[lo:hi]
    if np.all(np.isnan(window)):
        return None
    return int(np.nanargmin(window) + lo)


def _movement_bursts(
    speed: np.ndarray, threshold: float, *, min_frames: int = 2
) -> List[Tuple[int, int]]:
    """Contiguous runs where a joint is moving, as ``(start, end)`` index pairs."""
    moving = speed > threshold
    bursts: List[Tuple[int, int]] = []
    start: Optional[int] = None

    for i, flag in enumerate(moving):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_frames:
                bursts.append((start, i))
            start = None
    if start is not None and len(moving) - start >= min_frames:
        bursts.append((start, len(moving)))

    return bursts


def _net_displacement(xy: np.ndarray, start: int, end: int, axis: int = 0) -> float:
    """Absolute displacement across a burst, ignoring the path taken.

    This is what separates a trigger from a stride: a trigger goes out and comes back, so
    its net displacement is small even though its peak speed can be high. Thresholding on
    speed alone would classify a brisk trigger tap as the stride.
    """
    lo, hi = max(0, start), min(len(xy) - 1, end - 1)
    if hi <= lo:
        return 0.0
    return float(abs(xy[hi, axis] - xy[lo, axis]))


def _motion_offset(speed: np.ndarray, from_frame: Optional[int], quiet_speed: float) -> Optional[int]:
    """Walk forwards from a moving frame to where that movement stops.

    The mirror of :func:`_motion_onset`, and needed for the same reason: a threshold
    crossing reports the *end* of a movement early, because a decelerating joint drops below
    any fixed threshold before it actually stops.
    """
    if from_frame is None:
        return None
    i = int(from_frame)
    last = len(speed) - 1
    while i < last and speed[i] > quiet_speed:
        i += 1
    return i


def _motion_onset(speed: np.ndarray, from_frame: Optional[int], quiet_speed: float) -> Optional[int]:
    """Walk backwards from a moving frame to where that movement started.

    Threshold crossings systematically report an event *late*: a joint accelerating smoothly
    crosses any fixed threshold partway into the movement, not at its start. For events
    defined as an onset - the run-up beginning, the backlift beginning - that bias is tens of
    milliseconds, which matters when the whole metric is a timing.

    So the threshold is used only to find a frame that is definitely moving, and the true
    onset is found by walking back until the motion dies away.
    """
    if from_frame is None:
        return None
    i = int(from_frame)
    while i > 0 and speed[i] > quiet_speed:
        i -= 1
    return i


def _backlift_onset(
    wrist_y: np.ndarray, ds_frame: Optional[int], search_from: int, fps: float, quiet_speed: float
) -> Optional[int]:
    """Frame at which the hands begin to rise.

    Cannot be found as a maximum in wrist height: the hands sit flat at their lowest before
    the lift, so every resting frame ties and argmax picks arbitrarily among them. Instead
    locate the fastest upward movement - image y decreasing, so the most negative velocity -
    and walk back from there to the onset.
    """
    if ds_frame is None or ds_frame <= search_from:
        return None

    vel = np.gradient(wrist_y) * fps
    peak_up = _argmin_in(vel, search_from, ds_frame)
    if peak_up is None:
        return None
    return _motion_onset(np.abs(vel), peak_up, quiet_speed)


# ---------------------------------------------------------------------------
# Batting
# ---------------------------------------------------------------------------


def detect_batting(seq: PoseSequence, *, handedness: str, scale_px: float) -> PhaseTimeline:
    """Segment a batting clip.

    Order of detection is chosen so that reliable events constrain unreliable ones: the
    stance and stride come first and bound the search windows for the backlift and impact,
    rather than each event being sought independently across the whole clip.
    """
    fps = seq.fps
    events: Dict[str, PhaseEvent] = {}

    lead_ankle = to_lead_rear("lead_ankle", handedness)
    lead_wrist = to_lead_rear("lead_wrist", handedness)

    ankle_xy, ankle_speed = _trajectory(seq, lead_ankle, scale_px)
    wrist_xy, wrist_speed = _trajectory(seq, lead_wrist, scale_px)
    hip_xy, hip_speed = _trajectory(seq, "left_hip", scale_px)

    # --- SS / SE: the largest lead-ankle movement burst is the stride -------------------
    # Bursts first, then pick by net displacement. Thresholding on speed alone would call a
    # brisk trigger tap the stride, because a trigger can be fast even though it goes
    # nowhere.
    bursts = _movement_bursts(ankle_speed, FOOT_MOVING_SPEED)
    stride_burst = (
        max(bursts, key=lambda b: _net_displacement(ankle_xy, *b)) if bursts else None
    )
    # Walk out to the true onset and offset. A burst's threshold edges sit inside the real
    # movement - the foot is already going when it crosses, and still going when it drops
    # back - which shortens every measured stride duration by a consistent amount.
    ss = _motion_onset(ankle_speed, stride_burst[0], STILLNESS_SPEED) if stride_burst else None
    se = _motion_offset(ankle_speed, stride_burst[1], STILLNESS_SPEED) if stride_burst else None

    events["SS"] = make_event("SS", ss, fps, _mean_confidence(seq, [lead_ankle], ss) if ss else 0.0)
    events["SE"] = make_event("SE", se, fps, _mean_confidence(seq, [lead_ankle], se) if se else 0.0)

    # --- STANCE: the settled period before the stride -----------------------------------
    horizon = ss if ss is not None else max(1, seq.n_frames // 2)
    still = _longest_still_window_from_mask(_stillness_mask(hip_xy[:horizon], fps), fps)
    # Midpoint of the settled window, not its last frame. The stance is measured here
    # (stance width, setup posture), and the final instant before the stride is the least
    # representative point in the window - the player is already unweighting.
    stance_frame = (still[0] + still[1]) // 2 if still else None
    events["STANCE"] = make_event(
        "STANCE",
        stance_frame,
        fps,
        _mean_confidence(seq, ["left_hip", "right_hip"], stance_frame) if stance_frame else 0.0,
    )

    search_from = still[0] if still else 0

    # --- TRIGGER: an earlier burst that returns to where it started ---------------------
    trigger_frame = None
    if stride_burst is not None:
        stride_disp = _net_displacement(ankle_xy, *stride_burst)
        for burst in bursts:
            if burst[0] >= stride_burst[0]:
                break
            if _net_displacement(ankle_xy, *burst) < TRIGGER_MAX_FRACTION_OF_STRIDE * stride_disp:
                trigger_frame = _argmax_in(ankle_speed, *burst)
                break
    events["TRIGGER"] = make_event(
        "TRIGGER",
        trigger_frame,
        fps,
        _mean_confidence(seq, [lead_ankle], trigger_frame) if trigger_frame else 0.0,
    )

    # --- DS / BC: wrist vertical reversals ----------------------------------------------
    # Image y grows downward, so the top of the backlift is a MINIMUM in y. DS is found
    # first because it is an unambiguous turning point; BC is then found by walking back
    # from it, since the hands sit flat at their lowest before the lift and an argmax over
    # that flat stretch would tie arbitrarily.
    # Wrist-driven events are only reported if the wrists are actually trackable. On nets
    # footage they are routinely the worst joint in the skeleton, and a peak-speed detector
    # run over a teleporting keypoint returns a confident frame number that means nothing.
    wrist_usable, wrist_conf = _keypoint_is_usable(seq, lead_wrist, scale_px)
    if not wrist_usable:
        log.info(
            "Wrist tracking too unstable (conf %.2f) - suppressing BC, DS and IMP", wrist_conf
        )

    wrist_y = wrist_xy[:, 1]
    ds_frame = _argmin_in(wrist_y, search_from, seq.n_frames) if wrist_usable else None
    bc_frame = (
        _backlift_onset(wrist_y, ds_frame, search_from, fps, STILLNESS_SPEED)
        if wrist_usable
        else None
    )

    for name, frame in (("BC", bc_frame), ("DS", ds_frame)):
        conf = min(
            _mean_confidence(seq, [lead_wrist], frame) if frame else 0.0,
            PROXY_EVENT_MAX_CONFIDENCE,
        )
        events[name] = make_event(name, frame, fps, conf)

    # --- IMP: peak wrist speed after the downswing begins --------------------------------
    imp_frame = None
    if ds_frame is not None and wrist_usable:
        imp_frame = _argmax_in(wrist_speed, ds_frame, seq.n_frames)
    imp_conf = min(
        _mean_confidence(seq, [lead_wrist], imp_frame) if imp_frame else 0.0,
        PROXY_EVENT_MAX_CONFIDENCE,
    )
    events["IMP"] = make_event("IMP", imp_frame, fps, imp_conf)

    # --- FOLLOW_THROUGH -----------------------------------------------------------------
    ft_frame = None
    if imp_frame is not None:
        ft_frame = _first_crossing_below(wrist_speed, FOOT_MOVING_SPEED, imp_frame + 1)
    events["FOLLOW_THROUGH"] = make_event(
        "FOLLOW_THROUGH",
        ft_frame,
        fps,
        _mean_confidence(seq, [lead_wrist], ft_frame) if ft_frame else 0.0,
    )

    _suppress_high_fps_events(events, fps)
    return PhaseTimeline(events=events, fps=fps)


# ---------------------------------------------------------------------------
# Bowling
# ---------------------------------------------------------------------------


def detect_bowling(seq: PoseSequence, *, handedness: str, scale_px: float) -> PhaseTimeline:
    """Segment a bowling clip.

    Foot contacts anchor everything. Ball release is the weakest event in the system - the
    ball is not tracked, so it is proxied by peak bowling-wrist speed - and several metrics
    depend on it, which is why its confidence is capped and propagated rather than assumed.
    """
    fps = seq.fps
    events: Dict[str, PhaseEvent] = {}

    front_ankle = to_lead_rear("lead_ankle", handedness)
    back_ankle = to_lead_rear("rear_ankle", handedness)
    bowling_wrist = to_lead_rear("rear_wrist", handedness)

    front_xy, front_speed = _trajectory(seq, front_ankle, scale_px)
    back_xy, back_speed = _trajectory(seq, back_ankle, scale_px)
    wrist_xy, wrist_speed = _trajectory(seq, bowling_wrist, scale_px)
    hip_xy, hip_speed = _trajectory(seq, "left_hip", scale_px)

    # --- RUNUP: onset of sustained body movement -----------------------------------------
    runup_frame = _motion_onset(
        hip_speed, _first_crossing_above(hip_speed, FOOT_MOVING_SPEED), STILLNESS_SPEED
    )
    events["RUNUP"] = make_event(
        "RUNUP", runup_frame, fps, _mean_confidence(seq, ["left_hip"], runup_frame) if runup_frame else 0.0
    )

    # --- Flight then contact, for each foot in turn ---------------------------------------
    # Image y grows downward, so a foot at its highest is a MINIMUM in y and a planted foot
    # is a MAXIMUM. Searching for the contact directly does not work: the ground-level
    # baseline *before* the foot lifts ties with the value after it lands, and argmax
    # resolves that tie by returning the earlier frame - reporting contact before the
    # bowler has even left the ground. So each flight apex is located first and bounds the
    # search for the landing that follows it.
    search_from = runup_frame or 0

    pds = _argmin_in(back_xy[:, 1], search_from, seq.n_frames)
    bfc = _argmax_in(back_xy[:, 1], pds, seq.n_frames) if pds is not None else None

    front_apex = _argmin_in(front_xy[:, 1], bfc, seq.n_frames) if bfc is not None else None
    ffc = _argmax_in(front_xy[:, 1], front_apex, seq.n_frames) if front_apex is not None else None

    events["PRE_DELIVERY_STRIDE"] = make_event(
        "PRE_DELIVERY_STRIDE", pds, fps, _mean_confidence(seq, [back_ankle], pds) if pds else 0.0
    )
    events["BFC"] = make_event("BFC", bfc, fps, _mean_confidence(seq, [back_ankle], bfc) if bfc else 0.0)
    events["FFC"] = make_event("FFC", ffc, fps, _mean_confidence(seq, [front_ankle], ffc) if ffc else 0.0)

    # --- BR: peak bowling-wrist speed after front foot contact ----------------------------
    wrist_usable, wrist_conf = _keypoint_is_usable(seq, bowling_wrist, scale_px)
    if not wrist_usable:
        log.info(
            "Bowling wrist too unstable (conf %.2f) - suppressing ball release", wrist_conf
        )
    br = (
        _argmax_in(wrist_speed, ffc, seq.n_frames)
        if ffc is not None and wrist_usable
        else None
    )
    br_conf = min(
        _mean_confidence(seq, [bowling_wrist], br) if br else 0.0, PROXY_EVENT_MAX_CONFIDENCE
    )
    events["BR"] = make_event("BR", br, fps, br_conf)

    # --- FOLLOW_THROUGH -------------------------------------------------------------------
    ft = _first_crossing_below(hip_speed, STILLNESS_SPEED, br + 1) if br is not None else None
    events["FOLLOW_THROUGH"] = make_event(
        "FOLLOW_THROUGH", ft, fps, _mean_confidence(seq, ["left_hip"], ft) if ft else 0.0
    )

    _suppress_high_fps_events(events, fps)
    return PhaseTimeline(events=events, fps=fps)


def _suppress_high_fps_events(events: Dict[str, PhaseEvent], fps: float) -> None:
    """Drop sub-frame events below the capture floor.

    At 30 fps the arm-horizontal-to-release window spans 1-3 frames and bat-ball contact
    falls between frames entirely. Marking these undetected removes every metric anchored to
    them, which is the intended outcome - an estimate here would be indistinguishable from a
    measurement.
    """
    from capture_gate import MIN_FPS_FOR_EVENT_METRICS

    if fps >= MIN_FPS_FOR_EVENT_METRICS:
        return
    for name in HIGH_FPS_EVENTS:
        if name in events:
            events[name] = make_event(name, None, fps, 0.0)


def required_events_for_discipline(discipline: str) -> Tuple[str, ...]:
    """Events a discipline needs, before the specific shot is known.

    Auto-detection has to check the timeline is usable *before* it can say which shot it is,
    and the requirement is a property of the discipline rather than of the shot: every
    batting shot needs a stance and a stride, every delivery needs both foot contacts.
    """
    if discipline == "batting":
        return ("STANCE", "SS", "SE")
    if discipline == "bowling":
        return ("BFC", "FFC")
    raise ValueError(f"Unknown discipline {discipline!r}")


def required_events(form_type: str) -> Tuple[str, ...]:
    """Events without which a form type cannot be analysed at all.

    Deliberately short. A clip missing ``IMP`` still yields useful stride and posture
    feedback, so requiring the full set would reject clips that have something to say.
    """
    if form_type in ("cover_drive", "pull_shot"):
        return ("STANCE", "SS", "SE")
    if form_type in ("fast_arm_bowling", "spin_bowling"):
        return ("BFC", "FFC")
    raise ValueError(f"Unknown form type {form_type!r}")
