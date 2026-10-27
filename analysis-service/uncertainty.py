"""Measurement uncertainty, confidence intervals and minimal detectable change.

Every metric this service emits carries an interval, never a bare number. Two things depend
on getting this right:

* **Scoring** - a value whose interval straddles a band boundary is reported *unclear*
  rather than pass/fail, so measurement noise is never rendered as a verdict.
* **Progress** - with ~8 deg RMSE on the best-measured angle, a 3 deg "improvement" is
  noise. Changes below the minimal detectable change are reported as "no change detected".

Error figures come from ``docs/ACCURACY.md``; the citation keys resolve in
``docs/SOURCES.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from metric_tiers import MetricSpec, Tier

#: Two-sided 95% normal quantile.
Z_95 = 1.959963984540054

# ---------------------------------------------------------------------------
# Baseline measurement error, by metric family
# ---------------------------------------------------------------------------
# These are standard errors of measurement (SEM) in the metric's own units. They are
# deliberately pessimistic relative to the published figures, because every published figure
# comes from a *better* capture setup than ours: abraham2025opencap used three synchronised
# iPads at 240 Hz, and we have one phone.

#: Sagittal-plane lower-limb angles. abraham2025opencap knee flexion RMSE 7.87 deg.
SEM_ANGLE_SAGITTAL_LOWER_DEG = 8.0

#: Trunk lateral flexion. abraham2025opencap RMSE 9.10 deg.
SEM_ANGLE_TRUNK_DEG = 9.1

#: Transverse-plane / axial rotation. abraham2025opencap shoulder axial rotation RMSE
#: 28.54 deg. Only reachable at all with the front-on camera, and still Tier B.
SEM_ANGLE_TRANSVERSE_DEG = 28.5

#: Head landmark position, as a fraction of stature. The head is the best-tracked region
#: (large, high contrast, rarely self-occluded) - mann2013head is the reason we lean on it.
SEM_HEAD_POSITION_STATURE_FRAC = 0.012

#: Generic normalised distance (stride, base length) as a fraction of stature. Wider than
#: head position because foot landmarks are noisier and more often occluded.
SEM_DISTANCE_STATURE_FRAC = 0.025

#: Event-detection error independent of frame rate. mundt2024events reports ~24-25 ms median
#: absolute error for foot contact; for a normal distribution MAE ~= 0.8 * SD.
SEM_EVENT_DETECTION_S = 0.031

#: Single-camera penalty. Applied to angle SEMs because every published RMSE we rely on came
#: from a multi-camera rig. Not a measured constant - a documented safety margin.
SINGLE_CAMERA_PENALTY = 1.25


@dataclass(frozen=True)
class Measurement:
    """A measured value with a symmetric-in-practice confidence interval.

    ``confidence`` in ``[0, 1]`` is a *tracking-quality* score derived from keypoint
    confidence over the frames the metric depends on. It scales score weight; it is not the
    same thing as the statistical interval, which is carried by ``ci_low``/``ci_high``.
    """

    key: str
    value: float
    sem: float
    confidence: float
    units: str

    @property
    def ci_low(self) -> float:
        return self.value - Z_95 * self.sem

    @property
    def ci_high(self) -> float:
        return self.value + Z_95 * self.sem

    @property
    def ci_width(self) -> float:
        return self.ci_high - self.ci_low

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "value": self.value,
            "sem": self.sem,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
            "units": self.units,
        }


def temporal_sem(fps: float, *, detection_sem_s: float = SEM_EVENT_DETECTION_S) -> float:
    """SEM in seconds for a single detected event.

    Combines frame quantisation - uniform over one frame period, SD = period / sqrt(12) -
    with frame-rate-independent detection error, added in quadrature.

    This is why the capture protocol has a 120 fps floor: at 30 fps quantisation alone
    contributes ~9.6 ms, and the release window spans only 1-3 frames, so the event cannot
    be located at all rather than merely being imprecise.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    quantisation = (1.0 / fps) / math.sqrt(12.0)
    return math.hypot(quantisation, detection_sem_s)


def interval_sem(fps: float, *, detection_sem_s: float = SEM_EVENT_DETECTION_S) -> float:
    """SEM for the interval *between* two independently detected events.

    Independent errors add in quadrature, so an interval is noisier than either endpoint by
    a factor of sqrt(2).
    """
    return temporal_sem(fps, detection_sem_s=detection_sem_s) * math.sqrt(2.0)


def difference_sem(sem_a: float, sem_b: float, *, correlation: float = 0.5) -> float:
    """SEM for the difference of two angles measured on the same clip.

    Differencing two angles (knee at FFC vs at BR, say) partially cancels the *systematic*
    component of pose error - the bias from a given camera geometry is similar at both
    events. ``correlation`` models that shared component; 0 would treat the errors as
    independent and overstate the resulting uncertainty.

    abraham2025opencap found shoulder/elbow error to be systematic (consistent
    underestimation as the bowler turns side-on) rather than purely random, which is what
    justifies a non-zero default here.
    """
    if not -1.0 <= correlation <= 1.0:
        raise ValueError("correlation must be in [-1, 1]")
    var = sem_a**2 + sem_b**2 - 2.0 * correlation * sem_a * sem_b
    return math.sqrt(max(var, 0.0))


class CategoricalMetricError(ValueError):
    """Raised when a numeric uncertainty is requested for a discrete-valued metric."""


class NotMeasurableError(ValueError):
    """Raised for quantities this system cannot measure at all, such as bat kinematics.

    Distinct from a wide error bar: there is no measurement to attach an error to.
    """


#: Units a metric may be expressed in, and the factor converting the internal
#: stature-fraction SEM into those units. Getting this wrong is not a rounding error: a
#: measurement of 21.8 percent-of-stature carrying a +/-0.025 error bar would claim a
#: precision three orders of magnitude better than reality, and every band comparison built
#: on it would return a confident verdict.
UNIT_SEM_SCALE = {
    "frac_stature": 1.0,
    "pct_stature": 100.0,
}


def base_sem_for(spec: MetricSpec, *, fps: float, units: Optional[str] = None) -> float:
    """Baseline SEM for a metric, in its own units, before tracking-quality scaling.

    ``units`` rescales the result for metrics not expressed in the internal stature-fraction
    form. Callers that omit it get the internal form, which is correct only for angles,
    times and stature fractions.

    Raises :class:`CategoricalMetricError` for classification metrics (backlift direction,
    action type). These have no standard error because they have no numeric scale - their
    uncertainty is a class posterior, handled by the classifier that produces them.
    """
    tags = spec.tags
    if spec.is_categorical:
        raise CategoricalMetricError(
            f"{spec.key!r} is discrete-valued; uncertainty is a class probability, not a SEM"
        )
    if "implement" in tags:
        raise NotMeasurableError(
            f"{spec.key!r} describes the bat, which is not a body segment and is not tracked "
            f"by pose estimation at all. There is no error model because there is no "
            f"measurement. See docs/ACCURACY.md section 7."
        )
    if "temporal" in tags:
        return interval_sem(fps) if "->" in spec.event else temporal_sem(fps)
    if "transverse" in tags:
        return SEM_ANGLE_TRANSVERSE_DEG * SINGLE_CAMERA_PENALTY
    if "trunk" in tags:
        return SEM_ANGLE_TRUNK_DEG * SINGLE_CAMERA_PENALTY
    if "angle" in tags:
        base = SEM_ANGLE_SAGITTAL_LOWER_DEG * SINGLE_CAMERA_PENALTY
        return difference_sem(base, base) if "derived" in tags else base
    scale = UNIT_SEM_SCALE.get(units, 1.0) if units else 1.0
    if "head" in tags:
        return SEM_HEAD_POSITION_STATURE_FRAC * scale
    if "distance" in tags or "position" in tags:
        return SEM_DISTANCE_STATURE_FRAC * scale
    raise ValueError(
        f"No baseline SEM rule for {spec.key!r} (tags={sorted(tags)}). "
        f"Add one here rather than letting the metric default to an optimistic value."
    )


def scale_sem_by_confidence(sem: float, confidence: float, *, floor: float = 0.2) -> float:
    """Inflate SEM when keypoint tracking was poor over the relevant frames.

    Confidence near 1 leaves the baseline unchanged; poor tracking widens the interval,
    which in turn pushes the metric toward *unclear* in scoring rather than producing a
    confident wrong verdict.
    """
    c = max(min(confidence, 1.0), floor)
    return sem / c


def mdc(sem: float, *, n: int = 1, z: float = Z_95) -> float:
    """Minimal detectable change.

    ``MDC = z * sqrt(2) * SEM / sqrt(n)`` - the sqrt(2) because a change is a difference of
    two independent measurements, and the sqrt(n) because we compare *session means* rather
    than individual clips.

    Worked consequence: for a sagittal knee angle with SEM 10 deg, a single-clip MDC is
    about 28 deg, which is uselessly large. With 5 clips per session it falls to about
    12 deg. This is the arithmetic reason the app asks for 3-5 clips per session rather
    than accepting one.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    return z * math.sqrt(2.0) * sem / math.sqrt(n)


@dataclass(frozen=True)
class SessionAggregate:
    """Mean of a metric across the clips in one session."""

    key: str
    mean: float
    sem: float
    n: int
    units: str

    @property
    def ci_low(self) -> float:
        return self.mean - Z_95 * self.sem

    @property
    def ci_high(self) -> float:
        return self.mean + Z_95 * self.sem


def aggregate_session(measurements: Sequence[Measurement]) -> Optional[SessionAggregate]:
    """Combine per-clip measurements into a session-level estimate.

    Uses the larger of the observed between-clip scatter and the propagated instrument
    error. Real between-clip variation is genuine signal about consistency, but with 3-5
    clips the observed SD is itself unstable, so the instrument floor prevents a fluke run
    of similar clips from implying more precision than the camera can support.

    ``None`` when there is nothing to aggregate.
    """
    if not measurements:
        return None

    keys = {m.key for m in measurements}
    if len(keys) != 1:
        raise ValueError(f"aggregate_session got mixed metrics: {sorted(keys)}")

    n = len(measurements)
    values = [m.value for m in measurements]
    mean = sum(values) / n

    instrument_sem = math.sqrt(sum(m.sem**2 for m in measurements)) / n

    if n >= 2:
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        observed_sem = math.sqrt(variance / n)
    else:
        observed_sem = 0.0

    return SessionAggregate(
        key=measurements[0].key,
        mean=mean,
        sem=max(instrument_sem, observed_sem),
        n=n,
        units=measurements[0].units,
    )


def change_is_detectable(
    earlier: SessionAggregate,
    later: SessionAggregate,
    *,
    z: float = Z_95,
) -> bool:
    """Whether a between-session change exceeds measurement noise.

    The caller must render a ``False`` result as "no change detected" - never as a small
    improvement or a small regression. Reporting sub-MDC movement as progress is the single
    easiest way for this app to mislead a user who is training hard and wants to see a
    number move.
    """
    if earlier.key != later.key:
        raise ValueError(f"cannot compare {earlier.key!r} with {later.key!r}")
    threshold = z * math.hypot(earlier.sem, later.sem)
    return abs(later.mean - earlier.mean) > threshold


def straddles(measurement: Measurement, low: float, high: float) -> bool:
    """Whether the interval crosses either edge of an expected band.

    ``True`` means the clip cannot distinguish in-range from out-of-range, and
    :mod:`scoring` must report *unclear* rather than pass or fail.
    """
    inside = measurement.ci_low >= low and measurement.ci_high <= high
    outside = measurement.ci_high < low or measurement.ci_low > high
    return not (inside or outside)
