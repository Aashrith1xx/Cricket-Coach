"""Score a shot out of 100, grouped into aspects.

Everything else in this project refuses to produce a score, because scoring a joint angle
needs a reference population nobody has gathered. So it is worth being exact about what this
score *is*, because the number looks identical either way and the difference is everything.

**This grades the mechanical checks, not your technique against other players.**

It answers "of the things that can be verified from this clip, how many held up" - where
each check is either a sequence fact settled by mechanics, or a direction with a cited
source. It does not answer "how does this compare to a good player", and it never will until
:mod:`cohort.build` has real reference ranges to work from.

Two consequences follow, and both are load-bearing:

**Unassessable aspects score nothing.** They are not passes and are not failures - they are
excluded from the mean and reported separately. Counting an unmeasurable aspect as a pass is
how a thin analysis comes to look like a thorough one.

**Coverage is reported alongside the score, always.** 100 out of 100 from three of nine
aspects is not a good shot; it is three checks that happened to be possible. A caller showing
the score without the coverage is misrepresenting it, which is why
:meth:`AspectReport.headline` refuses to phrase one without the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from mechanics import Finding, Severity


@dataclass(frozen=True)
class Aspect:
    """One facet of form, and the checks that speak to it."""

    key: str
    label: str
    blurb: str
    #: Rule ids contributing to this aspect - metric keys for direction rules, ``order.*``
    #: for sequence rules.
    rules: frozenset


BATTING_ASPECTS: Sequence[Aspect] = (
    Aspect(
        "sequence", "Sequence",
        "Whether the parts of the shot happened in the right order.",
        frozenset({"order.stride", "order.plant_before_contact", "order.backlift"}),
    ),
    Aspect(
        "head", "Head position",
        "Whether your head stayed steady through the shot - the strongest measurable "
        "difference between elite and club batters.",
        frozenset({"batting.head_vertical_displacement_ds_imp"}),
    ),
    Aspect(
        "balance", "Weight transfer",
        "Whether your weight moved into the shot rather than staying back.",
        frozenset({"batting.com_anterior_displacement_min_imp"}),
    ),
)

BOWLING_ASPECTS: Sequence[Aspect] = (
    Aspect(
        "sequence", "Sequence",
        "Whether the delivery stride happened in the right order.",
        frozenset({"order.delivery_stride", "order.release_after_plant"}),
    ),
    Aspect(
        "front_leg", "Front leg",
        "Whether the front leg braced through the delivery rather than folding.",
        frozenset({"bowling.front_knee_change_ffc_br"}),
    ),
)


@dataclass
class AspectScore:
    key: str
    label: str
    blurb: str
    #: ``None`` when nothing in this aspect could be checked. Not a zero.
    score: Optional[float]
    passed: int
    failed: int

    @property
    def assessed(self) -> bool:
        return self.score is not None

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "blurb": self.blurb,
            "score": None if self.score is None else round(self.score),
            "passed": self.passed,
            "failed": self.failed,
            "assessed": self.assessed,
        }


@dataclass
class AspectReport:
    aspects: List[AspectScore]
    overall: Optional[float]
    checks_passed: int
    checks_total: int

    @property
    def assessed(self) -> List[AspectScore]:
        return [a for a in self.aspects if a.assessed]

    @property
    def coverage(self) -> float:
        return len(self.assessed) / len(self.aspects) if self.aspects else 0.0

    @property
    def headline(self) -> str:
        """Score and coverage together. Never one without the other - see module docstring."""
        if self.overall is None:
            return "Not enough of this clip could be checked to score it."
        return (
            f"{round(self.overall)} out of 100 on the "
            f"{len(self.assessed)} of {len(self.aspects)} aspects this clip could support."
        )

    def as_dict(self) -> dict:
        return {
            "overall": None if self.overall is None else round(self.overall),
            "headline": self.headline,
            "coverage": round(self.coverage, 2),
            "assessed_count": len(self.assessed),
            "aspect_count": len(self.aspects),
            "checks_passed": self.checks_passed,
            "checks_total": self.checks_total,
            "aspects": [a.as_dict() for a in self.aspects],
        }


def score(findings: Sequence[Finding], discipline: str) -> AspectReport:
    """Group findings into aspects and grade each on what it could verify."""
    definitions = BATTING_ASPECTS if discipline == "batting" else BOWLING_ASPECTS
    by_rule = {f.rule: f for f in findings}

    scored: List[AspectScore] = []
    total_passed = total_checks = 0

    for aspect in definitions:
        relevant = [by_rule[r] for r in aspect.rules if r in by_rule]
        passed = sum(1 for f in relevant if f.severity is Severity.GOOD)
        failed = sum(1 for f in relevant if f.severity is Severity.FAULT)

        total_passed += passed
        total_checks += passed + failed

        scored.append(
            AspectScore(
                key=aspect.key,
                label=aspect.label,
                blurb=aspect.blurb,
                # None, not zero: nothing was checkable here.
                score=(100.0 * passed / (passed + failed)) if (passed + failed) else None,
                passed=passed,
                failed=failed,
            )
        )

    assessed = [a for a in scored if a.assessed]
    # Weighted by how many checks each aspect actually ran, so an aspect resting on one
    # check does not carry the same weight as one resting on three.
    if assessed:
        weight = sum(a.passed + a.failed for a in assessed)
        overall = sum(a.score * (a.passed + a.failed) for a in assessed) / weight
    else:
        overall = None

    return AspectReport(scored, overall, total_passed, total_checks)
