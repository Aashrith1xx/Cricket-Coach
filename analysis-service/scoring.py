"""Per-metric scoring and composite score assembly.

Implements steps 7-9 of the grading pipeline in ``README.md`` section 4.

Two properties matter more than the arithmetic:

1. **Noise is never rendered as a verdict.** When a measurement's confidence interval
   straddles a band boundary, the metric is ``UNCLEAR`` - not a marginal pass or a marginal
   fail - and its weight is cut.
2. **A composite score is never fabricated.** If too little reliable weight survives, the
   result carries ``composite=None`` and the UI shows individual observations only. An
   overall grade assembled from three noisy metrics would look authoritative while meaning
   nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence

import metric_tiers
from conventions import range_to_internal
from metric_tiers import Tier
from uncertainty import Measurement, straddles

# ---------------------------------------------------------------------------
# Weighting
# ---------------------------------------------------------------------------

#: Temporal metrics outweigh angles. mundt2024events puts event timing at ~24-25 ms error,
#: while abraham2025opencap puts the best-measured angle at 7.87 deg RMSE from a far better
#: rig than ours. This inverts the intuitive "angles first" design and is the single most
#: consequential weighting decision in the system (docs/ACCURACY.md s4).
BASE_WEIGHT_TEMPORAL = 1.0
BASE_WEIGHT_OTHER = 0.6

#: Weight retained by a metric whose interval straddles a band edge.
UNCLEAR_WEIGHT_FACTOR = 0.25

#: Minimum total surviving weight before a composite score may be shown at all.
#: Roughly "at least three solid Tier A metrics, or equivalent".
MIN_COMPOSITE_WEIGHT = 2.5


class Verdict(str, Enum):
    IN_RANGE = "in_range"
    OUT_OF_RANGE = "out_of_range"
    UNCLEAR = "unclear"
    NOT_SCORED = "not_scored"


@dataclass(frozen=True)
class ExpectedBand:
    """One metric's expected range, already resolved from the reference template."""

    low: float
    high: float
    convention: Optional[str]
    source: str
    population: str
    confidence: str
    style_variant: Optional[str] = None

    @property
    def width(self) -> float:
        return self.high - self.low


@dataclass(frozen=True)
class MetricScore:
    key: str
    verdict: Verdict
    score: Optional[float]
    weight: float
    measurement: Measurement
    band: Optional[ExpectedBand]
    tier: Tier
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "verdict": self.verdict.value,
            "score": self.score,
            "weight": self.weight,
            "tier": self.tier.value,
            "measurement": self.measurement.as_dict(),
            "band": None
            if self.band is None
            else {
                "low": self.band.low,
                "high": self.band.high,
                "source": self.band.source,
                "population": self.band.population,
                "confidence": self.band.confidence,
                "style_variant": self.band.style_variant,
            },
            "note": self.note,
        }


@dataclass(frozen=True)
class CompositeResult:
    """``composite is None`` means: show observations, show no overall grade."""

    composite: Optional[float]
    composite_ci_low: Optional[float]
    composite_ci_high: Optional[float]
    total_weight: float
    scores: List[MetricScore]
    suppressed_reason: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "composite": self.composite,
            "composite_ci_low": self.composite_ci_low,
            "composite_ci_high": self.composite_ci_high,
            "total_weight": self.total_weight,
            "suppressed_reason": self.suppressed_reason,
            "scores": [s.as_dict() for s in self.scores],
        }


def _band_to_internal(band: ExpectedBand) -> tuple[float, float]:
    """Convert a template band into the internal angle convention.

    Non-angle metrics (times, normalised distances) carry ``convention: null`` and pass
    through untouched.
    """
    if band.convention is None:
        return band.low, band.high
    return range_to_internal(band.low, band.high, band.convention)


def _distance_score(value: float, low: float, high: float) -> float:
    """Score in ``[0, 1]`` for a value outside its band.

    Falls off with distance from the nearest edge, scaled by band width, so a metric with a
    wide acceptable range is not punished as harshly for the same absolute deviation. One
    full band-width outside scores 0.
    """
    width = high - low
    if width <= 0:
        return 0.0
    if value < low:
        excess = low - value
    elif value > high:
        excess = value - high
    else:
        return 1.0
    return max(0.0, 1.0 - excess / width)


def score_metric(
    measurement: Measurement,
    band: Optional[ExpectedBand],
) -> MetricScore:
    """Score one metric against its expected band.

    Tier C and descriptors are refused outright rather than scored to zero, so that a bug
    routing suppressed data here surfaces as ``NOT_SCORED`` instead of quietly dragging a
    composite down.
    """
    spec = metric_tiers.get(measurement.key)

    if not metric_tiers.is_scorable(measurement.key):
        reason = (
            "tier C - never scored or displayed"
            if spec.tier is Tier.C
            else "descriptor - style, not correctness"
            if spec.tier is Tier.DESCRIPTOR
            else "does not discriminate skill level"
        )
        return MetricScore(
            key=measurement.key,
            verdict=Verdict.NOT_SCORED,
            score=None,
            weight=0.0,
            measurement=measurement,
            band=band,
            tier=spec.tier,
            note=reason,
        )

    if band is None:
        return MetricScore(
            key=measurement.key,
            verdict=Verdict.NOT_SCORED,
            score=None,
            weight=0.0,
            measurement=measurement,
            band=None,
            tier=spec.tier,
            note="no sourced reference range for this form type and style",
        )

    low, high = _band_to_internal(band)

    base = BASE_WEIGHT_TEMPORAL if spec.is_temporal else BASE_WEIGHT_OTHER
    weight = (
        base
        * metric_tiers.TIER_WEIGHT_MULTIPLIER[spec.tier]
        * max(min(measurement.confidence, 1.0), 0.0)
    )

    if straddles(measurement, low, high):
        return MetricScore(
            key=measurement.key,
            verdict=Verdict.UNCLEAR,
            score=None,
            weight=weight * UNCLEAR_WEIGHT_FACTOR,
            measurement=measurement,
            band=band,
            tier=spec.tier,
            note=(
                "measurement interval crosses the edge of the expected range - this clip "
                "cannot tell in-range from out-of-range"
            ),
        )

    if low <= measurement.value <= high:
        return MetricScore(
            key=measurement.key,
            verdict=Verdict.IN_RANGE,
            score=1.0,
            weight=weight,
            measurement=measurement,
            band=band,
            tier=spec.tier,
        )

    return MetricScore(
        key=measurement.key,
        verdict=Verdict.OUT_OF_RANGE,
        score=_distance_score(measurement.value, low, high),
        weight=weight,
        measurement=measurement,
        band=band,
        tier=spec.tier,
    )


def score_all(
    measurements: Sequence[Measurement],
    bands: Mapping[str, ExpectedBand],
) -> List[MetricScore]:
    return [score_metric(m, bands.get(m.key)) for m in measurements]


def composite(
    scores: Sequence[MetricScore],
    *,
    min_weight: float = MIN_COMPOSITE_WEIGHT,
) -> CompositeResult:
    """Weighted mean of the scored metrics, or ``None`` if evidence is too thin.

    ``UNCLEAR`` metrics contribute weight but no score, so they widen the composite interval
    without pulling the point estimate in either direction - which is the honest treatment
    of "we could not tell".
    """
    scored = [s for s in scores if s.score is not None and s.weight > 0]
    unclear = [s for s in scores if s.verdict is Verdict.UNCLEAR]

    scored_weight = sum(s.weight for s in scored)
    total_weight = scored_weight + sum(s.weight for s in unclear)

    if scored_weight <= 0 or total_weight < min_weight:
        return CompositeResult(
            composite=None,
            composite_ci_low=None,
            composite_ci_high=None,
            total_weight=total_weight,
            scores=list(scores),
            suppressed_reason=(
                "Not enough reliably measured metrics in this clip to justify an overall "
                "score. Individual observations are shown instead."
            ),
        )

    point = sum(s.score * s.weight for s in scored) / scored_weight

    # Unclear metrics widen the interval in proportion to how much of the total weight they
    # represent: the more of the picture we could not resolve, the less precise the grade.
    unresolved_fraction = 1.0 - (scored_weight / total_weight) if total_weight else 0.0
    spread = 0.5 * unresolved_fraction

    return CompositeResult(
        composite=point,
        composite_ci_low=max(0.0, point - spread),
        composite_ci_high=min(1.0, point + spread),
        total_weight=total_weight,
        scores=list(scores),
    )


def group_by_verdict(scores: Sequence[MetricScore]) -> Dict[Verdict, List[MetricScore]]:
    out: Dict[Verdict, List[MetricScore]] = {v: [] for v in Verdict}
    for s in scores:
        out[s.verdict].append(s)
    return out
