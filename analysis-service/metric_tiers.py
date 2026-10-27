"""Metric tier registry and the Tier C suppression gate.

This module is the single enforcement point for the measurement limits documented in
``docs/ACCURACY.md``. Scoring, feedback generation and the render payload all consult it;
none of them keep their own copy of the rules.

The tier of a metric is a property of **how well a phone camera can measure that joint**,
not of how interesting the metric is to a coach. Several of the most coaching-relevant
variables in cricket (bat speed, shoulder counter-rotation, elbow extension) are Tier C
precisely because they matter enough that getting them wrong would be harmful.

Tiers
-----
``A``
    Scored numerically with a confidence interval. In-image-plane lower-body/trunk angles
    (~8-9 deg RMSE) and temporal metrics (~24-25 ms error).
``B``
    Scored at reduced weight and displayed only as a band ("short / in range / long").
    Wide error, or requires the optional front-on camera.
``C``
    Computed and stored, but **never displayed and never scored**. Measurement error
    exceeds the effect size being measured.
``DESCRIPTOR``
    Displayed neutrally, never scored. These are reliably measurable but are *style*, not
    correctness - grading them would encode one orthodoxy as truth. See the backlift
    finding in ``docs/ACCURACY.md`` section 6.

Adding or moving a metric requires a citation in ``docs/SOURCES.md``, not an intuition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, Iterable, Optional


class Tier(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    DESCRIPTOR = "DESCRIPTOR"


class Discipline(str, Enum):
    BATTING = "batting"
    BOWLING = "bowling"


#: Score-weight multiplier applied per tier, before per-metric confidence weighting.
#: Tier C is hard zero and is additionally blocked by :func:`assert_scorable`; the zero here
#: is a second line of defence, not the primary one.
TIER_WEIGHT_MULTIPLIER: Dict[Tier, float] = {
    Tier.A: 1.0,
    Tier.B: 0.4,
    Tier.C: 0.0,
    Tier.DESCRIPTOR: 0.0,
}


@dataclass(frozen=True)
class MetricSpec:
    """Static description of one metric.

    Attributes
    ----------
    key:
        Stable identifier, ``<discipline>.<name>``. Referenced by reference templates.
    tier:
        See module docstring.
    discipline:
        Which form type this belongs to.
    event:
        Phase event the metric is measured at, or the pair of events it spans. Must match
        an event name emitted by ``phase_detection``.
    summary:
        One line, safe to show a user (Tier A/B/DESCRIPTOR only).
    basis:
        Why this tier - cites a ``docs/SOURCES.md`` key. Never shown to users.
    discriminates_skill:
        ``False`` for variables the literature shows do *not* separate skill levels. These
        carry zero score weight even in Tier A/B; they may still be displayed for interest.
        Reference templates must declare a matching value - ``template_validator`` checks it.
    requires_front_on:
        Metric is only meaningful with the optional front-on second camera. Suppressed on a
        single side-on clip.
    min_fps:
        Minimum capture frame rate for the metric to be emitted at all. Release- and
        impact-referenced metrics are suppressed below this rather than estimated.
    """

    key: str
    tier: Tier
    discipline: Discipline
    event: str
    summary: str
    basis: str
    discriminates_skill: bool = True
    requires_front_on: bool = False
    min_fps: Optional[int] = None
    tags: FrozenSet[str] = field(default_factory=frozenset)
    #: Overrides the tag-derived view rule. Set only where the derivation is wrong.
    explicit_views: Optional[FrozenSet[str]] = None

    @property
    def is_temporal(self) -> bool:
        """Temporal metrics carry a higher base weight than angles (``docs/ACCURACY.md`` s4)."""
        return "temporal" in self.tags

    @property
    def views(self) -> FrozenSet[str]:
        """Camera views this metric can actually be measured from.

        The governing rule is that a quantity is measurable only when it varies **in the
        camera's image plane**, so which camera position you need depends on which plane the
        quantity lives in:

        * **Temporal metrics are view-independent.** A stride duration is the same however
          you film it - only the event frames matter, not the geometry. These are also the
          highest-weighted metrics in the system, which is why a front-on clip is worth
          analysing rather than refusing.
        * **Sagittal quantities** - knee flexion, stride length, forward trunk lean - need
          the camera to the side.
        * **Transverse and frontal quantities** - pelvis-thorax separation, lateral trunk
          flexion, stance width - need it in front.

        Treating front-on as a *failed* side-on was a design error: it is a different view
        with a different, smaller, but genuinely valid metric set.
        """
        if self.explicit_views is not None:
            return self.explicit_views
        if self.tags & {"temporal", "sequence"}:
            return frozenset({"side_on", "front_on"})
        if self.tags & {"transverse", "frontal"}:
            return frozenset({"front_on"})
        return frozenset({"side_on"})

    @property
    def is_categorical(self) -> bool:
        """Discrete-valued metrics: a class posterior, not a numeric confidence interval.

        Covers both classifications (backlift direction) and sequence orderings, which are
        permutations rather than quantities.
        """
        return bool(self.tags & {"classification", "sequence"})


def _spec(*args, **kwargs) -> MetricSpec:
    return MetricSpec(*args, **kwargs)


# ---------------------------------------------------------------------------
# Bowling
# ---------------------------------------------------------------------------

_BOWLING: Iterable[MetricSpec] = (
    # --- Tier A -------------------------------------------------------------
    _spec(
        key="bowling.front_knee_angle_ffc",
        tier=Tier.A,
        discipline=Discipline.BOWLING,
        event="FFC",
        summary="Front knee angle at front-foot contact",
        basis="abraham2025opencap: knee flexion RMSE 7.87 deg, sagittal plane, best-measured joint",
        tags=frozenset({"angle", "lower_body"}),
    ),
    _spec(
        key="bowling.front_knee_angle_br",
        tier=Tier.A,
        discipline=Discipline.BOWLING,
        event="BR",
        summary="Front knee angle at ball release",
        basis="elitefastbowlers2024: front knee angle at release correlates with release speed r=0.40",
        min_fps=120,
        tags=frozenset({"angle", "lower_body"}),
    ),
    _spec(
        key="bowling.front_knee_change_ffc_br",
        tier=Tier.A,
        discipline=Discipline.BOWLING,
        event="FFC->BR",
        summary="Front knee bracing through the delivery stride",
        basis="elitefastbowlers2024; differencing two angles cancels part of the systematic bias",
        min_fps=120,
        tags=frozenset({"angle", "lower_body", "derived"}),
    ),
    _spec(
        key="bowling.trunk_lateral_flexion_br",
        tier=Tier.B,
        discipline=Discipline.BOWLING,
        event="BR",
        summary="Sideways trunk lean at release",
        basis=(
            "abraham2025opencap measured trunk lateral flexion at 9.10 deg RMSE - but with "
            "THREE cameras. Lateral flexion is a frontal-plane motion, so on the single "
            "side-on camera this app specifies it points toward and away from the lens and "
            "is foreshortened. The published RMSE does not transfer to our capture "
            "geometry. Requires the optional front-on angle to be meaningful."
        ),
        requires_front_on=True,
        min_fps=120,
        tags=frozenset({"angle", "trunk", "frontal"}),
    ),
    _spec(
        key="bowling.interval_bfc_ffc",
        tier=Tier.A,
        discipline=Discipline.BOWLING,
        event="BFC->FFC",
        summary="Time from back-foot contact to front-foot contact",
        basis="mundt2024events: event timing ~24-25 ms error, the most robust signal available",
        tags=frozenset({"temporal"}),
    ),
    _spec(
        key="bowling.interval_ffc_br",
        tier=Tier.A,
        discipline=Discipline.BOWLING,
        event="FFC->BR",
        summary="Time from front-foot contact to release",
        basis="mundt2024events",
        min_fps=120,
        tags=frozenset({"temporal"}),
    ),
    _spec(
        key="bowling.runup_final_stride_tempo",
        tier=Tier.A,
        discipline=Discipline.BOWLING,
        event="RUNUP",
        summary="Rhythm of the final run-up strides",
        basis="elitefastbowlers2024: faster run-up associated with faster release; temporal metric",
        tags=frozenset({"temporal"}),
    ),
    # --- Tier B -------------------------------------------------------------
    _spec(
        key="bowling.delivery_stride_length_norm",
        tier=Tier.B,
        discipline=Discipline.BOWLING,
        event="FFC",
        summary="Delivery stride length, relative to your height",
        basis="Depth-sensitive on a monocular view; normalised to stature per accuracy strategy 3.4",
        tags=frozenset({"distance", "lower_body"}),
    ),
    _spec(
        key="bowling.trunk_forward_flexion_br",
        tier=Tier.A,
        discipline=Discipline.BOWLING,
        event="BR",
        summary="Forward trunk flexion at release",
        basis=(
            "Sagittal-plane motion, so it lies IN the image plane on the side-on camera the "
            "capture protocol specifies - the one trunk angle this geometry measures well. "
            "abraham2025opencap puts trunk angles among the better-measured variables "
            "(9.10 deg RMSE). elitefastbowlers2024 associates greater trunk flexion with "
            "faster release."
        ),
        min_fps=120,
        tags=frozenset({"angle", "trunk"}),
    ),
    _spec(
        key="bowling.head_over_front_foot_ffc",
        tier=Tier.B,
        discipline=Discipline.BOWLING,
        event="FFC",
        summary="Head position over the front foot",
        basis="Head tracks well, but the relation to foot is sensitive to camera obliquity",
        tags=frozenset({"position", "head"}),
    ),
    # --- Tier C: suppressed -------------------------------------------------
    _spec(
        key="bowling.shoulder_counter_rotation",
        tier=Tier.C,
        discipline=Discipline.BOWLING,
        event="BFC->FFC",
        summary="",
        basis=(
            "lumbar_stress: risk threshold >30 deg. abraham2025opencap: shoulder axial rotation "
            "RMSE 28.54 deg. The measurement error is larger than the threshold, so a 'safe' "
            "verdict would be unfounded and could contribute to a lumbar stress fracture."
        ),
        tags=frozenset({"angle", "transverse", "safety_critical"}),
    ),
    _spec(
        key="bowling.elbow_extension_release",
        tier=Tier.C,
        discipline=Discipline.BOWLING,
        event="ARM_HORIZONTAL->BR",
        summary="",
        basis=(
            "icc_illegal_action: legality threshold 15 deg. abraham2025opencap: elbow flexion "
            "RMSE 22.71 deg. Never comment on action legality."
        ),
        tags=frozenset({"angle", "upper_limb", "safety_critical"}),
    ),
    _spec(
        key="bowling.arm_elevation_plane",
        tier=Tier.C,
        discipline=Discipline.BOWLING,
        event="BR",
        summary="",
        basis="abraham2025opencap: arm elevation plane RMSE 28.92 deg, the worst-measured variable",
        tags=frozenset({"angle", "upper_limb"}),
    ),
    _spec(
        key="bowling.shoulder_alignment_bfc",
        tier=Tier.C,
        discipline=Discipline.BOWLING,
        event="BFC",
        summary="",
        basis="bowling_action_classification requires shoulder alignment; RMSE 28.54 deg",
        tags=frozenset({"angle", "transverse"}),
    ),
    _spec(
        key="bowling.action_type",
        tier=Tier.C,
        discipline=Discipline.BOWLING,
        event="BFC",
        summary="",
        basis=(
            "Derived from shoulder alignment and counter-rotation, both Tier C. Classifying a "
            "bowler as mixed-action implies an injury-risk claim this app must not make."
        ),
        tags=frozenset({"classification", "safety_critical"}),
    ),
)


# ---------------------------------------------------------------------------
# Batting
# ---------------------------------------------------------------------------

_BATTING: Iterable[MetricSpec] = (
    # --- Tier A -------------------------------------------------------------
    _spec(
        key="batting.lead_knee_angle_imp",
        tier=Tier.A,
        discipline=Discipline.BATTING,
        event="IMP",
        summary="Front knee angle at contact",
        basis="abraham2025opencap knee RMSE 7.87 deg; pullshot_malaysia found lower-body angles discriminate",
        min_fps=120,
        tags=frozenset({"angle", "lower_body"}),
    ),
    _spec(
        key="batting.lead_knee_extension_se_imp",
        tier=Tier.A,
        discipline=Discipline.BATTING,
        event="SE->IMP",
        summary="Front knee movement from stride end to contact",
        basis="peploe2019technique defines this variable; differencing cancels part of the bias",
        min_fps=120,
        tags=frozenset({"angle", "lower_body", "derived"}),
    ),
    _spec(
        key="batting.head_vertical_displacement_ds_imp",
        tier=Tier.A,
        discipline=Discipline.BATTING,
        event="DS->IMP",
        summary="Head lift through the downswing",
        basis=(
            "mann2013head: head-ball coupling is the strongest measurable batting discriminator "
            "(elite 2.4 vs club 3.7 deg mean head-ball angle). The head is the best-tracked "
            "region - large, high contrast, rarely self-occluded."
        ),
        min_fps=120,
        tags=frozenset({"position", "head"}),
    ),
    _spec(
        key="batting.head_horizontal_displacement_ds_imp",
        tier=Tier.A,
        discipline=Discipline.BATTING,
        event="DS->IMP",
        summary="Head travel toward the ball through the downswing",
        basis="mann2013head; headcom2007: skilled batters hold the head further forward",
        min_fps=120,
        tags=frozenset({"position", "head"}),
    ),
    _spec(
        key="batting.stride_duration_ss_se",
        tier=Tier.A,
        discipline=Discipline.BATTING,
        event="SS->SE",
        summary="Duration of the forward stride",
        basis="mundt2024events; pullshot_malaysia reports ~0.22-0.24 s stride duration",
        tags=frozenset({"temporal"}),
    ),
    _spec(
        key="batting.interval_se_imp",
        tier=Tier.A,
        discipline=Discipline.BATTING,
        event="SE->IMP",
        summary="Time from planting the front foot to contact",
        basis="frontfootdrive1999: cover drive foot planted ~0.06 s before impact",
        min_fps=120,
        tags=frozenset({"temporal"}),
    ),
    _spec(
        key="batting.tempo_bc_ds",
        tier=Tier.A,
        discipline=Discipline.BATTING,
        event="BC->DS",
        summary="Backlift to downswing tempo",
        basis="peploe2019technique event definitions; temporal metrics are the most robust signal",
        tags=frozenset({"temporal"}),
    ),
    _spec(
        key="batting.trigger_timing",
        tier=Tier.A,
        discipline=Discipline.BATTING,
        event="TRIGGER",
        summary="Timing of your trigger movement",
        basis="headcom2007; trigger_formats. Timing is temporal and robust; presence is style",
        tags=frozenset({"temporal"}),
    ),
    _spec(
        key="batting.com_anterior_displacement_min_imp",
        tier=Tier.A,
        discipline=Discipline.BATTING,
        event="IMP",
        summary="Weight transfer toward the ball",
        basis="peploe2019technique defines COM displacement Y (min to IMP); headcom2007",
        min_fps=120,
        tags=frozenset({"position"}),
    ),
    _spec(
        key="batting.base_length_pct_stature_imp",
        tier=Tier.A,
        discipline=Discipline.BATTING,
        event="IMP",
        summary="Width of your base at contact, relative to your height",
        basis="peploe2019technique normalises base length to stature - already scale-invariant",
        min_fps=120,
        tags=frozenset({"distance", "lower_body"}),
    ),
    # --- Tier B -------------------------------------------------------------
    _spec(
        key="batting.stride_length_norm",
        tier=Tier.B,
        discipline=Discipline.BATTING,
        event="SE",
        summary="Stride length, relative to your height",
        basis=(
            "sarpeshkar2011: experienced batters take a SHORT-to-medium stride to play the ball "
            "later. pullshot_malaysia found no significant difference across age groups. Banded "
            "with an optimum range - never 'longer is better'."
        ),
        tags=frozenset({"distance", "lower_body"}),
    ),
    _spec(
        key="batting.head_over_lead_knee_imp",
        tier=Tier.B,
        discipline=Discipline.BATTING,
        event="IMP",
        summary="Head position over the front knee at contact",
        basis="headcom2007; sensitive to camera obliquity, hence Tier B not A",
        min_fps=120,
        tags=frozenset({"position", "head"}),
    ),
    _spec(
        key="batting.x_factor_ds",
        tier=Tier.B,
        discipline=Discipline.BATTING,
        event="DS",
        summary="Separation between hips and shoulders at the start of the downswing",
        basis=(
            "peploe2019technique: strongest single predictor of bat speed (28.0% of variance). "
            "But it is a transverse-plane measure - abraham2025opencap shoulder axial rotation "
            "RMSE 28.54 deg - so it is suppressed entirely without the front-on camera."
        ),
        requires_front_on=True,
        tags=frozenset({"angle", "transverse"}),
    ),
    _spec(
        key="batting.lower_trunk_sequence_order",
        tier=Tier.B,
        discipline=Discipline.BATTING,
        event="DS->IMP",
        summary="Order in which your lower body and trunk accelerate",
        basis=(
            "Ordering of peak angular velocities is more robust than magnitude, but arm and "
            "wrist peaks are too poorly tracked to include - only the lower-body to trunk "
            "portion survives."
        ),
        min_fps=120,
        tags=frozenset({"sequence"}),
    ),
    # --- Tier C: suppressed -------------------------------------------------
    _spec(
        key="batting.lead_elbow_extension_ds_imp",
        tier=Tier.C,
        discipline=Discipline.BATTING,
        event="DS->IMP",
        summary="",
        basis="peploe2019technique 2nd bat-speed predictor; abraham2025opencap elbow RMSE 22.71 deg",
        tags=frozenset({"angle", "upper_limb"}),
    ),
    _spec(
        key="batting.wrist_uncocking_ds_imp",
        tier=Tier.C,
        discipline=Discipline.BATTING,
        event="DS->IMP",
        summary="",
        basis=(
            "peploe2019technique 3rd bat-speed predictor. More distal and more occluded than the "
            "elbow - hidden behind the hands and bat through the swing."
        ),
        tags=frozenset({"angle", "upper_limb"}),
    ),
    _spec(
        key="batting.bat_angle_imp",
        tier=Tier.C,
        discipline=Discipline.BATTING,
        event="IMP",
        summary="",
        basis=(
            "The bat is not a body segment - pose models do not track it. pullshot_malaysia also "
            "found no significant difference in bat angle across skill levels."
        ),
        discriminates_skill=False,
        tags=frozenset({"implement"}),
    ),
    _spec(
        key="batting.bat_speed_max",
        tier=Tier.C,
        discipline=Discipline.BATTING,
        event="DS->IMP",
        summary="",
        basis=(
            "batsensor2025: even a bat-mounted IMU shows 1.4-3.9% bias vs mocap with error "
            "growing at higher speeds. Not measurable from monocular video at any useful "
            "accuracy. Also not a skill discriminator in controlled shots."
        ),
        discriminates_skill=False,
        tags=frozenset({"implement", "sensor_only"}),
    ),
    _spec(
        key="batting.impact_location",
        tier=Tier.C,
        discipline=Discipline.BATTING,
        event="IMP",
        summary="",
        basis=(
            "peploe2019technique: impact location + bat speed explain 68% of ball launch speed, "
            "so it matters - but it requires bat-face tracking that monocular video cannot do."
        ),
        tags=frozenset({"implement"}),
    ),
    _spec(
        key="batting.arm_wrist_sequence_order",
        tier=Tier.C,
        discipline=Discipline.BATTING,
        event="DS->IMP",
        summary="",
        basis="Arm and wrist angular-velocity peaks are too poorly tracked to order reliably",
        tags=frozenset({"sequence"}),
    ),
    # --- Descriptors: displayed, never scored -------------------------------
    _spec(
        key="batting.backlift_direction",
        tier=Tier.DESCRIPTOR,
        discipline=Discipline.BATTING,
        event="BC",
        summary="Backlift direction",
        basis=(
            "noorbhai_lbbt: 77% of successful batsmen and 90% of IPL batsmen use a LATERAL "
            "backlift while 83% of coaches teach STRAIGHT. Scoring this would penalise the "
            "technique most elite players actually use."
        ),
        discriminates_skill=False,
        requires_front_on=True,
        tags=frozenset({"classification", "style"}),
    ),
    _spec(
        key="batting.trigger_present",
        tier=Tier.DESCRIPTOR,
        discipline=Discipline.BATTING,
        event="TRIGGER",
        summary="Trigger movement",
        basis=(
            "trigger_formats: prevalence is format-dependent (Test 56.1%, ODI 47.4%, T20I 73.3%, "
            "IPL 76%). Absence is not a fault."
        ),
        discriminates_skill=False,
        tags=frozenset({"classification", "style"}),
    ),
    _spec(
        key="batting.stance_width",
        tier=Tier.DESCRIPTOR,
        discipline=Discipline.BATTING,
        event="STANCE",
        summary="Stance width",
        basis="battingsetup_review: stance is a set-up preference, not a correctness axis",
        discriminates_skill=False,
        tags=frozenset({"classification", "style", "frontal"}),
    ),
)


REGISTRY: Dict[str, MetricSpec] = {s.key: s for s in (*_BOWLING, *_BATTING)}

#: Hard suppression list. Nothing here may reach a user, in any surface, ever.
TIER_C_SUPPRESSED: FrozenSet[str] = frozenset(
    k for k, s in REGISTRY.items() if s.tier is Tier.C
)

#: Metrics that imply an injury-risk or legality claim. Suppressed for safety, not only
#: accuracy - kept as a named set so the reason survives future refactors.
SAFETY_CRITICAL: FrozenSet[str] = frozenset(
    k for k, s in REGISTRY.items() if "safety_critical" in s.tags
)


class SuppressedMetricError(RuntimeError):
    """Raised when suppressed metric data is about to reach a user-facing surface."""


def get(key: str) -> MetricSpec:
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"Unknown metric {key!r}. Add a MetricSpec with a docs/SOURCES.md citation "
            f"before referencing it in a reference template."
        ) from None


def is_displayable(key: str) -> bool:
    """Tier C is never shown. Everything else may be, subject to its own confidence."""
    return get(key).tier is not Tier.C


def is_scorable(key: str) -> bool:
    """Tier C, descriptors, and non-discriminating variables carry no score weight."""
    spec = get(key)
    if spec.tier in (Tier.C, Tier.DESCRIPTOR):
        return False
    return spec.discriminates_skill


def assert_scorable(key: str) -> None:
    """Guard for :mod:`scoring`. Fails loudly rather than silently contributing zero."""
    if not is_scorable(key):
        raise SuppressedMetricError(
            f"{key!r} is not scorable (tier={get(key).tier.value}, "
            f"discriminates_skill={get(key).discriminates_skill}). See docs/ACCURACY.md."
        )


def assert_displayable(key: str) -> None:
    """Guard for :mod:`feedback` and :mod:`render_output`."""
    if not is_displayable(key):
        raise SuppressedMetricError(
            f"{key!r} is Tier C and must never reach a user. Reason: {get(key).basis}"
        )


def filter_for_display(keys: Iterable[str]) -> list[str]:
    return [k for k in keys if is_displayable(k)]


def applicable(
    discipline: Discipline,
    *,
    fps: float,
    view: str = "side_on",
    has_front_on: bool = False,
) -> list[MetricSpec]:
    """Metrics computable for this discipline under the given capture conditions.

    ``view`` is the camera position the clip was shot from. ``has_front_on`` additionally
    admits front-on metrics when a *second* camera was used alongside a side-on primary.

    Metrics whose ``min_fps`` is unmet are **omitted, not estimated** - below 120 fps the
    release and impact windows span too few frames to locate (``docs/ACCURACY.md`` s3).
    """
    available = {view} | ({"front_on"} if has_front_on else set())

    out = []
    for spec in REGISTRY.values():
        if spec.discipline is not discipline:
            continue
        if spec.min_fps is not None and fps < spec.min_fps:
            continue
        if not (spec.views & available):
            continue
        out.append(spec)
    return out


def unavailable_in_view(discipline: Discipline, view: str) -> list[MetricSpec]:
    """Metrics this view cannot measure, so the UI can say what is missing and why."""
    return [
        spec
        for spec in REGISTRY.values()
        if spec.discipline is discipline
        and spec.tier is not Tier.C
        and view not in spec.views
    ]
