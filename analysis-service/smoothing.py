"""Trajectory conditioning: confidence weighting, gap handling, and filtering.

Angles and velocities are **never** computed from raw frame-to-frame coordinates.
Differentiation amplifies high-frequency noise, so an unfiltered keypoint trajectory that
looks acceptable as position becomes unusable as velocity - and several metrics
(sequence ordering, tempo) depend on velocity peaks.

Order of operations matters and is not interchangeable:

1. mask low-confidence samples to NaN
2. interpolate *short* gaps, reject long ones
3. filter
4. only then differentiate

Interpolating before masking would smear a confidently-wrong keypoint across its neighbours;
filtering before interpolation would propagate NaNs across the window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

#: Keypoints below this confidence are treated as missing rather than trusted.
CONFIDENCE_MASK_THRESHOLD = 0.35

#: Longest gap that may be bridged by interpolation, in seconds. Beyond this the joint was
#: genuinely unobserved - most often occlusion as the body turns - and inventing a
#: trajectory would manufacture the very measurement the uncertainty model exists to
#: qualify.
MAX_INTERPOLATED_GAP_S = 0.05

#: Savitzky-Golay window as a fraction of the clip's frame rate, i.e. of one second.
#: Cricket actions are fast, so the window must stay short or genuine peaks are flattened.
SAVGOL_WINDOW_FRACTION = 0.05
SAVGOL_POLYORDER = 3


class SmoothingError(RuntimeError):
    pass


@dataclass
class SmoothedTrajectory:
    """A conditioned trajectory plus the provenance needed for uncertainty.

    ``valid_fraction`` and ``max_gap_frames`` are carried forward into the metric layer,
    where they scale the per-metric confidence - a joint reconstructed across many small
    gaps should widen intervals, not silently pass as clean data.
    """

    xy: np.ndarray  # (n_frames, 2), NaN where unrecoverable
    valid_fraction: float
    max_gap_frames: int
    fps: float

    @property
    def has_unrecoverable_gaps(self) -> bool:
        return bool(np.isnan(self.xy).any())


def mask_low_confidence(
    xy: np.ndarray,
    confidence: np.ndarray,
    *,
    threshold: float = CONFIDENCE_MASK_THRESHOLD,
) -> np.ndarray:
    """Replace low-confidence samples with NaN.

    A low-confidence keypoint is not a slightly-noisy measurement; pose models routinely
    place an occluded joint somewhere plausible-looking but wrong. Treating it as missing is
    more honest than down-weighting it.
    """
    out = xy.astype(float, copy=True)
    out[confidence < threshold] = np.nan
    return out


def _gap_runs(mask: np.ndarray) -> list[Tuple[int, int]]:
    """Start/end index pairs of consecutive ``True`` runs."""
    runs: list[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def interpolate_short_gaps(
    xy: np.ndarray,
    fps: float,
    *,
    max_gap_s: float = MAX_INTERPOLATED_GAP_S,
) -> Tuple[np.ndarray, int]:
    """Linearly bridge gaps up to ``max_gap_s``; leave longer ones as NaN.

    Returns the filled array and the longest gap encountered, in frames, so callers can
    distinguish "clean" from "reconstructed" data.

    Leading and trailing gaps are never extrapolated - there is no evidence on one side.
    """
    max_gap_frames = int(round(max_gap_s * fps))
    out = xy.astype(float, copy=True)
    longest = 0

    for axis in range(out.shape[1]):
        series = out[:, axis]
        missing = np.isnan(series)
        if not missing.any():
            continue
        idx = np.arange(len(series))
        known = ~missing
        if known.sum() < 2:
            continue

        for start, end in _gap_runs(missing):
            longest = max(longest, end - start)
            if start == 0 or end == len(series):
                continue  # no extrapolation
            if (end - start) > max_gap_frames:
                continue  # too long to bridge honestly
            series[start:end] = np.interp(
                idx[start:end], idx[known], series[known]
            )
        out[:, axis] = series

    return out, longest


def savgol(values: np.ndarray, fps: float, *, polyorder: int = SAVGOL_POLYORDER) -> np.ndarray:
    """Savitzky-Golay filter sized relative to frame rate.

    A fixed frame-count window would behave differently at 120 fps and 240 fps, making
    metrics depend on capture settings. Sizing by duration keeps the effective cutoff
    frequency constant.
    """
    try:
        from scipy.signal import savgol_filter
    except ImportError as exc:  # pragma: no cover
        raise SmoothingError(
            "scipy is required for filtering. Do not substitute a moving average: it has a "
            "poor frequency response and will flatten the velocity peaks that sequence and "
            "tempo metrics depend on."
        ) from exc

    window = int(round(SAVGOL_WINDOW_FRACTION * fps))
    if window % 2 == 0:
        window += 1
    window = max(window, polyorder + 2)
    if window % 2 == 0:
        window += 1
    if len(values) < window:
        return values.copy()

    if values.ndim == 1:
        return savgol_filter(values, window, polyorder)
    return np.column_stack(
        [savgol_filter(values[:, i], window, polyorder) for i in range(values.shape[1])]
    )


#: Fastest a human joint plausibly travels, in stature-lengths per second.
#:
#: Elite bat-tip speed is around 27 m/s, but the hands are far closer to the axis and peak
#: nearer 10-12 m/s - about 6 stature/s for a 1.8 m player. 9 is generous headroom above
#: that, so anything exceeding it is the model relocating a keypoint, not a limb moving.
#:
#: This matters because behind net mesh the wrists jump ~0.106 stature per frame at 240 fps -
#: some 25 stature/s, four times faster than a hand can physically travel. Those samples are
#: not noisy measurements of the wrist; they are the wrist momentarily lost.
MAX_JOINT_SPEED_STATURE_PER_S = 9.0


def reject_implausible_jumps(
    xy: np.ndarray,
    fps: float,
    scale_px: float,
    *,
    max_speed: float = MAX_JOINT_SPEED_STATURE_PER_S,
) -> Tuple[np.ndarray, int]:
    """Blank samples that moved faster than a joint physically can.

    Returns ``(cleaned, n_rejected)``. Rejected samples become NaN so
    :func:`interpolate_short_gaps` can bridge them if they are brief and leave them missing
    if they are not - which is the honest outcome for a keypoint that was genuinely lost.

    Works forward from the last accepted sample rather than between consecutive frames, so a
    single wild outlier cannot drag the reference along with it.
    """
    out = xy.astype(float, copy=True)
    max_step = max_speed * scale_px / fps

    last_good: Optional[int] = None
    rejected = 0

    for i in range(len(out)):
        if np.isnan(out[i]).any():
            continue
        if last_good is None:
            last_good = i
            continue

        gap = max(i - last_good, 1)
        travelled = float(np.linalg.norm(out[i] - out[last_good]))
        if travelled > max_step * gap:
            out[i] = np.nan
            rejected += 1
        else:
            last_good = i

    return out, rejected


def condition(
    xy: np.ndarray,
    confidence: np.ndarray,
    fps: float,
    *,
    threshold: float = CONFIDENCE_MASK_THRESHOLD,
    max_gap_s: float = MAX_INTERPOLATED_GAP_S,
    scale_px: Optional[float] = None,
) -> SmoothedTrajectory:
    """Full pipeline for one keypoint trajectory.

    When ``scale_px`` is supplied, physically impossible jumps are rejected before
    interpolation. That step matters most exactly where tracking is worst - behind net mesh
    the model loses the wrists and relocates them across the frame, and those samples are
    confident enough to survive the confidence mask while being plainly wrong.
    """
    masked = mask_low_confidence(xy, confidence, threshold=threshold)
    if scale_px:
        masked, _rejected = reject_implausible_jumps(masked, fps, scale_px)
    filled, max_gap = interpolate_short_gaps(masked, fps, max_gap_s=max_gap_s)

    valid = ~np.isnan(filled).any(axis=1)
    valid_fraction = float(valid.mean()) if len(valid) else 0.0

    smoothed = filled.copy()
    if valid.all() and len(filled) > 0:
        smoothed = savgol(filled, fps)
    elif valid.any():
        # Filter only the contiguous valid span, leaving unrecoverable regions NaN rather
        # than letting the filter interpolate across them implicitly.
        first, last = int(np.argmax(valid)), len(valid) - int(np.argmax(valid[::-1]))
        segment = filled[first:last]
        if not np.isnan(segment).any() and len(segment) > 0:
            smoothed[first:last] = savgol(segment, fps)

    return SmoothedTrajectory(
        xy=smoothed,
        valid_fraction=valid_fraction,
        max_gap_frames=max_gap,
        fps=fps,
    )


def velocity(xy: np.ndarray, fps: float) -> np.ndarray:
    """Per-frame speed from an already-smoothed trajectory.

    Uses central differences, which have better error characteristics than forward
    differences at the same spacing. Callers must pass filtered input - see module docstring.
    """
    if len(xy) < 3:
        return np.zeros(len(xy))
    diffs = np.gradient(xy, axis=0) * fps
    return np.linalg.norm(diffs, axis=1)


def angular_velocity(angles_deg: np.ndarray, fps: float) -> np.ndarray:
    """Per-frame angular velocity in deg/s from a smoothed angle series."""
    if len(angles_deg) < 3:
        return np.zeros(len(angles_deg))
    return np.gradient(angles_deg) * fps
