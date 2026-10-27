"""Playstyle resolution: declared style selects the reference, classifier only corroborates.

This module encodes the central decision of the playstyle design, so it is worth restating
the reasoning rather than leaving it in a doc.

Posture-based intent classification works: ``intent2025posture`` reports 77% accuracy, 0.83
AUC and 0.77 F1 with a 1D CNN over plain joint coordinates, falling to 74% on a held-out
batter. That is genuinely useful. It is also **not good enough to gate grading on**:

1. At ~1-in-4 error, routing the reference template by classifier output would silently give
   a quarter of users the wrong band - producing feedback that is confidently wrong with no
   signal to the user that anything went astray.
2. It classifies per-shot *intent*, not player *style*. Style is the distribution of intents
   across many deliveries.
3. Its training labels were outcome-contaminated: aggressive/defensive was proxied by runs
   scored (>=3 vs <=1), so a well-executed attacking shot that happens to be fielded well is
   labelled defensive. The published accuracy is against noisy ground truth.
4. It was trained on broadcast footage of elite players, not amateur phone clips, so
   real-world accuracy will be lower still.

So the player declares their style, and the classifier is used for something it *is* good
enough for: noticing when observed intent diverges from declared style, and saying so. An
intent-execution mismatch is useful coaching feedback in its own right.

**Style conditions the expected range, never the tier.** Measurement error is a property of
the camera and the joint, not of playing style.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence

from scoring import ExpectedBand


class Style(str, Enum):
    ATTACKING = "attacking"
    BALANCED = "balanced"
    DEFENSIVE = "defensive"


class Intent(str, Enum):
    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"


#: Minimum clips before an observed style distribution is worth mentioning at all. Below
#: this, sampling noise dominates and a "divergence" is meaningless.
MIN_CLIPS_FOR_OBSERVED_STYLE = 8

#: Minimum mean classifier confidence for a clip's intent to count toward the distribution.
MIN_INTENT_CONFIDENCE = 0.6

#: How far the observed distribution must sit from the declared style before we say
#: anything. Set well above the classifier's own error rate so a routine run of defensive
#: shots does not trigger a message.
DIVERGENCE_THRESHOLD = 0.35


@dataclass(frozen=True)
class IntentEstimate:
    """One clip's classifier output. Advisory only."""

    intent: Intent
    confidence: float

    @property
    def counts(self) -> bool:
        return self.confidence >= MIN_INTENT_CONFIDENCE


@dataclass(frozen=True)
class ObservedStyle:
    aggressive_fraction: float
    n_clips_used: int
    n_clips_total: int

    @property
    def sufficient(self) -> bool:
        return self.n_clips_used >= MIN_CLIPS_FOR_OBSERVED_STYLE


def summarise_intents(estimates: Sequence[IntentEstimate]) -> ObservedStyle:
    """Aggregate per-clip intents into an observed distribution.

    Low-confidence estimates are dropped rather than down-weighted: a classifier that is
    unsure about a shot contributes nothing useful, and including it just adds noise to a
    figure already built on weak labels.
    """
    usable = [e for e in estimates if e.counts]
    if not usable:
        return ObservedStyle(0.0, 0, len(estimates))
    counts = Counter(e.intent for e in usable)
    aggressive = counts[Intent.AGGRESSIVE] / len(usable)
    return ObservedStyle(aggressive, len(usable), len(estimates))


#: Where each declared style would be expected to sit on the aggressive fraction.
_STYLE_CENTRE = {
    Style.ATTACKING: 0.75,
    Style.BALANCED: 0.5,
    Style.DEFENSIVE: 0.25,
}


def divergence_note(declared: Style, observed: ObservedStyle) -> Optional[str]:
    """A neutral observation when observed intent diverges from declared style.

    Returns ``None`` when there is nothing worth saying - which is most of the time, and
    should be. This never changes which reference is used; it is a remark, not a correction.
    """
    if not observed.sufficient:
        return None

    expected = _STYLE_CENTRE[declared]
    delta = observed.aggressive_fraction - expected
    if abs(delta) < DIVERGENCE_THRESHOLD:
        return None

    direction = "more attacking" if delta > 0 else "more defensive"
    pct = round(observed.aggressive_fraction * 100)
    return (
        f"Across your last {observed.n_clips_used} clips, your movement pattern reads "
        f"{direction} than the {declared.value} profile you selected "
        f"({pct}% of shots read as attacking). That is not a fault - but if it is not what "
        f"you intended, it is worth a look. You can change your profile any time."
    )


def resolve_bands(
    template: Mapping,
    declared: Style,
) -> Dict[str, ExpectedBand]:
    """Build the metric-to-band map for a clip.

    Resolution order per metric, per the plan:

    1. a ``style_variants`` entry for the declared style that carries its own ``source``
    2. otherwise the style-agnostic ``metrics`` entry
    3. otherwise no band, and the metric goes unscored

    A style variant is **never synthesised**. If no sourced variant exists for a style, the
    style-agnostic band applies - which is the honest answer, since no study located
    establishes which technique variants map to which style.
    """
    bands: Dict[str, ExpectedBand] = {}
    variants = (template.get("style_variants") or {}).get(declared.value, {})

    for key, body in template.get("metrics", {}).items():
        chosen = None
        variant = variants.get(key)
        if variant and variant.get("source") and variant.get("band_derivation") != "todo":
            chosen = (variant, declared.value)
        elif body.get("source") and body.get("band_derivation") != "todo":
            chosen = (body, None)

        if chosen is None:
            continue

        entry, variant_name = chosen
        bands[key] = ExpectedBand(
            low=entry["expected_low"],
            high=entry["expected_high"],
            convention=entry.get("convention"),
            source=entry["source"],
            population=entry["population"],
            confidence=entry["confidence"],
            style_variant=variant_name,
        )

    return bands


def population_caveat(bands: Mapping[str, ExpectedBand], player_group: str) -> List[str]:
    """Warnings where a band's reference population does not match the player.

    Elite adult male ranges must not be applied silently to a junior. The app still scores -
    refusing would leave juniors with nothing - but it must say what it is comparing against.
    """
    notes: List[str] = []
    seen: set[str] = set()
    for band in bands.values():
        if band.population in ("unknown", player_group) or band.population in seen:
            continue
        seen.add(band.population)
        notes.append(
            f"Some ranges here come from {band.population.replace('_', ' ')} players, "
            f"which may not describe a {player_group.replace('_', ' ')} player well."
        )
    return notes
