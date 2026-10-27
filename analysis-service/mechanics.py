"""Mechanical rules: feedback that needs no reference population.

The app cannot yet say "your front knee is more flexed than typical", because that needs
cohort data nobody has gathered. It can already say "your front knee collapsed instead of
bracing", because that is a claim about the *direction* a variable moved, not about where it
sits in a distribution.

That distinction is what makes this module possible, and the research is lopsided in exactly
the helpful way: normative ranges for cricket are scarce, while established *directions* are
not. ``elitefastbowlers2024`` gives a correlation between front-knee extension and ball speed
without publishing a target angle. ``mann2013head`` gives elite-versus-club head-ball coupling
without a threshold. A correlation is useless as a band and perfectly good as a direction.

Three rule shapes, in descending order of how well-founded they are:

``ORDERING``
    Sequence facts that follow from mechanics rather than statistics - the front foot lands
    before contact, the hips fire before the shoulders. Violations are errors at any skill
    level, so these need no evidence beyond physics.
``DIRECTION``
    A variable whose sign or trend is known to matter, cited to a source. Fires only when the
    observed value is far enough from neutral to exceed its own measurement error.
``CONSISTENCY``
    Repeatability across clips in a session. Left to :mod:`progressStats` on the server,
    which already has the session aggregation.

Every rule states what was seen, so the player can check it against their own video. Rules
never produce a score - they are observations, and the composite stays null until real
reference ranges exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence

import metric_tiers
from phase_detection import PhaseTimeline
from uncertainty import Z_95, Measurement


class RuleKind(str, Enum):
    ORDERING = "ordering"
    DIRECTION = "direction"


class Severity(str, Enum):
    GOOD = "good"
    WATCH = "watch"
    FAULT = "fault"


@dataclass(frozen=True)
class Finding:
    """One mechanical observation about a clip."""

    rule: str
    kind: RuleKind
    severity: Severity
    text: str
    #: What was measured, in the player's own numbers. Lets them verify against the video.
    evidence: str
    source: str
    #: What to do about it. Present only on faults - a remedy attached to something you did
    #: well is noise, and a remedy for something unmeasured would be invented advice.
    remedy: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "rule": self.rule,
            "kind": self.kind.value,
            "severity": self.severity.value,
            "text": self.text,
            "evidence": self.evidence,
            "source": self.source,
            "remedy": self.remedy,
        }


@dataclass(frozen=True)
class DirectionRule:
    """A metric whose direction is known to matter, even where its range is not.

    ``desirable`` is the sign that is better: ``-1`` when lower is better (head lift),
    ``+1`` when higher is better (knee extension through release). Metrics with an optimum
    band rather than a direction - stride length most importantly - are deliberately absent,
    because for those neither sign is a fault and saying otherwise would be inventing a
    coaching opinion.
    """

    metric_key: str
    desirable: int
    good_text: str
    fault_text: str
    #: A cue addressing the fault directly. Legitimate because it only says "do less of the
    #: thing we measured you doing" - it asserts no norm and prescribes no target value.
    remedy: str
    source: str
    #: Multiples of the measurement's own standard error before the rule fires. A finding
    #: inside the error bar would be commenting on noise.
    min_effect_sems: float = 1.5


#: Rules, each tied to a source in docs/SOURCES.md.
#:
#: Deliberately short. Every entry here is a claim the app makes about a player's technique,
#: so the bar is a citation rather than a plausible-sounding coaching cue.
DIRECTION_RULES: Sequence[DirectionRule] = (
    DirectionRule(
        metric_key="batting.head_vertical_displacement_ds_imp",
        desirable=-1,
        good_text="Your head stayed steady through the downswing.",
        fault_text="Your head lifted as you came down into the shot.",
        remedy=(
            "Pick a spot on the pitch and keep your eyes level over it as you swing. Lifting "
            "the head early pulls your eyes off the ball at the moment you most need them."
        ),
        source="mann2013head",
    ),
    DirectionRule(
        metric_key="bowling.front_knee_change_ffc_br",
        desirable=+1,
        good_text="Your front leg braced and drove through the delivery.",
        fault_text="Your front knee collapsed through the delivery instead of bracing.",
        remedy=(
            "Work on landing with a firmer front leg so it resists rather than folds. A "
            "braced front leg is what the rest of the body rotates over."
        ),
        source="elitefastbowlers2024",
    ),
    DirectionRule(
        metric_key="batting.com_anterior_displacement_min_imp",
        desirable=+1,
        good_text="Your weight moved forward into the shot.",
        fault_text="Your weight stayed back rather than moving into the shot.",
        remedy=(
            "Feel your weight travelling onto the front foot as you play, rather than the "
            "hands going out to the ball while the body stays where it started."
        ),
        source="headcom2007",
    ),
)


def _measurement_map(measurements: Sequence[Measurement]) -> Dict[str, Measurement]:
    return {m.key: m for m in measurements}


def check_directions(measurements: Sequence[Measurement]) -> List[Finding]:
    """Apply the direction rules to one clip's measurements."""
    found: List[Finding] = []
    by_key = _measurement_map(measurements)

    for rule in DIRECTION_RULES:
        m = by_key.get(rule.metric_key)
        if m is None:
            continue

        # The whole interval must sit clear of zero. A value inside its own error bar is
        # indistinguishable from no movement, and a rule that fires there is commenting on
        # measurement noise dressed up as technique.
        threshold = rule.min_effect_sems * m.sem
        if abs(m.value) < threshold:
            continue

        aligned = (m.value > 0) == (rule.desirable > 0)
        spec = metric_tiers.get(rule.metric_key)

        found.append(
            Finding(
                rule=rule.metric_key,
                kind=RuleKind.DIRECTION,
                severity=Severity.GOOD if aligned else Severity.FAULT,
                text=rule.good_text if aligned else rule.fault_text,
                evidence=f"{spec.summary}: {m.value:+.3f} {m.units} (±{Z_95 * m.sem:.3f})",
                source=rule.source,
                remedy=None if aligned else rule.remedy,
            )
        )

    return found


def check_ordering(timeline: PhaseTimeline, discipline: str) -> List[Finding]:
    """Apply sequence rules.

    These are the best-founded feedback the app produces: they follow from mechanics rather
    than from any population, so they hold for a beginner and an international alike.

    Only fires when both events are detected and their separation exceeds the combined
    timing uncertainty - two events 5 ms apart at 240 fps are not reliably ordered.
    """
    found: List[Finding] = []

    checks = (
        [
            ("SS", "SE", "stride", "Your stride completed before contact, as it should.",
             "Your front foot was still moving at contact - you're playing the ball while "
             "still striding, which costs you a stable base.",
             "Get the front foot down earlier so it is planted and settled before you play. "
             "You cannot swing from a base that is still moving."),
            ("SE", "IMP", "plant_before_contact", "You planted your front foot before contact.",
             "You made contact before your front foot had planted.",
             "Start the stride sooner. Playing before the foot lands means the shot is made "
             "entirely with the hands."),
            ("BC", "DS", "backlift", "Your backlift completed before the downswing began.",
             "Your downswing started before the backlift had finished.",
             "Let the bat reach the top before starting down. Rushing the changeover shortens "
             "the swing and costs you time to read the ball."),
        ]
        if discipline == "batting"
        else [
            ("BFC", "FFC", "delivery_stride", "Your back foot landed before your front foot.",
             "Your front foot appears to land before your back foot, which is not a normal "
             "delivery stride - worth checking the clip.",
             "Check the clip against this: if it looks wrong, the foot events were probably "
             "mis-detected rather than your action being unusual."),
            ("FFC", "BR", "release_after_plant", "You released after your front foot planted.",
             "The release looks to come before front-foot contact, which is unusual - worth "
             "checking the clip.",
             "Release is inferred from peak wrist speed rather than seen, so on net footage "
             "this often means the wrists were mistracked. Check the overlay."),
        ]
    )

    for first, second, name, good, fault, remedy in checks:
        a, b = timeline.get(first), timeline.get(second)
        if a is None or b is None or a.time_s is None or b.time_s is None:
            continue

        gap = b.time_s - a.time_s
        uncertainty_s = Z_95 * (a.time_sem_s**2 + b.time_sem_s**2) ** 0.5
        if abs(gap) < uncertainty_s:
            continue  # too close to order reliably

        ordered = gap > 0
        found.append(
            Finding(
                rule=f"order.{name}",
                kind=RuleKind.ORDERING,
                severity=Severity.GOOD if ordered else Severity.FAULT,
                text=good if ordered else fault,
                evidence=f"{first} to {second}: {gap * 1000:+.0f} ms (±{uncertainty_s * 1000:.0f})",
                source="mechanical",
                remedy=None if ordered else remedy,
            )
        )

    return found


def analyse(
    measurements: Sequence[Measurement], timeline: PhaseTimeline, discipline: str
) -> List[Finding]:
    """All mechanical findings for one clip, faults first."""
    findings = check_ordering(timeline, discipline) + check_directions(measurements)
    order = {Severity.FAULT: 0, Severity.WATCH: 1, Severity.GOOD: 2}
    return sorted(findings, key=lambda f: order[f.severity])


@dataclass
class Summary:
    """Conclusion, weak points, remedies, and what could not be assessed."""

    headline: str
    strengths: List[Finding]
    faults: List[Finding]
    not_assessed: List[str]

    def as_dict(self) -> dict:
        return {
            "headline": self.headline,
            "strengths": [f.as_dict() for f in self.strengths],
            "faults": [f.as_dict() for f in self.faults],
            "remedies": [f.remedy for f in self.faults if f.remedy],
            "not_assessed": self.not_assessed,
        }


def _why_not_assessed(rule: DirectionRule, measurements: Sequence[Measurement]) -> str:
    spec = metric_tiers.get(rule.metric_key)
    by_key = _measurement_map(measurements)
    m = by_key.get(rule.metric_key)
    if m is None:
        return f"{spec.summary} - not measured in this clip"
    return f"{spec.summary} - the movement was smaller than the measurement error"


def summarise(
    findings: Sequence[Finding],
    measurements: Sequence[Measurement],
    discipline: str,
) -> Summary:
    """Turn findings into a conclusion, weak points and what went unchecked.

    Listing what could NOT be assessed matters as much as the findings. A clean result from
    two rules is not a clean technique - it is two rules that happened to be checkable. Left
    unsaid, a short list of green ticks reads as a clean bill of health it has not earned.
    """
    strengths = [f for f in findings if f.severity is Severity.GOOD]
    faults = [f for f in findings if f.severity is Severity.FAULT]

    checked = {f.rule for f in findings}
    not_assessed = [
        _why_not_assessed(rule, measurements)
        for rule in DIRECTION_RULES
        if rule.metric_key.startswith(discipline) and rule.metric_key not in checked
    ]

    if faults:
        headline = (
            f"{len(faults)} thing{'' if len(faults) == 1 else 's'} to work on"
            + (f", and {len(strengths)} you did well" if strengths else "")
            + "."
        )
    elif strengths:
        headline = (
            f"Nothing we could check came out wrong - {len(strengths)} "
            f"{'point' if len(strengths) == 1 else 'points'} looked sound. "
            f"That is not the same as a clean technique: see what we could not assess below."
        )
    else:
        headline = "This clip did not support any of the mechanical checks."

    return Summary(headline, strengths, faults, not_assessed)
