"""Coaching feedback generation, with hard guards on what may be said.

Step 10 of the pipeline in ``README.md`` section 4. The generator is deliberately
constrained rather than expressive:

* **No Tier C metric may be mentioned**, in any phrasing.
* **No injury, safety or legality claim**, ever - not even a reassuring one. The app cannot
  measure shoulder counter-rotation or elbow extension to the precision those judgements
  need, so both "your action looks risky" and "your action looks safe" are unfounded.
* **No bare numbers for Tier B**, which is banded by definition.
* ``UNCLEAR`` metrics are reported as *not measurable from this clip*, with a capture fix -
  never as a marginal pass or fail.
* **Descriptors are phrased neutrally.** A lateral backlift is not a fault; saying so in
  passing would reintroduce the orthodoxy the tiering exists to avoid.

The guards are enforced by :func:`_assert_safe` over the assembled text, so a future
template that smuggles in a forbidden phrase fails loudly in tests rather than shipping.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence

import metric_tiers
from metric_tiers import Tier
from scoring import CompositeResult, MetricScore, Verdict

# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

#: Substrings that must never appear in generated feedback. Matched case-insensitively on
#: word boundaries. Covers injury/legality claims and the Tier C vocabulary.
FORBIDDEN_PATTERNS: tuple[str, ...] = (
    r"\binjur\w*",
    r"\bsafe\b",
    r"\bunsafe\b",
    r"\brisk\w*",
    r"\bstress fracture\b",
    r"\blumbar\b",
    r"\bspine\b",
    r"\bchuck\w*",
    r"\bthrow(ing)?\b",
    r"\billegal\b",
    r"\blegal\w*",
    r"\bmixed action\b",
    r"\bcounter[- ]rotation\b",
    r"\belbow extension\b",
    r"\bbat speed\b",
    r"\bdegrees? of elbow\b",
    r"\bdiagnos\w*",
)

_FORBIDDEN_RE = re.compile("|".join(FORBIDDEN_PATTERNS), re.IGNORECASE)


class FeedbackGuardError(RuntimeError):
    """Raised when generated text violates a hard content rule."""


def _assert_safe(text: str) -> str:
    match = _FORBIDDEN_RE.search(text)
    if match:
        raise FeedbackGuardError(
            f"Generated feedback contains forbidden phrase {match.group(0)!r}. "
            f"See docs/ACCURACY.md - this app must not make injury, legality or "
            f"Tier C claims. Text was: {text!r}"
        )
    return text


# ---------------------------------------------------------------------------
# Phrasing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeedbackItem:
    """One line of feedback, tagged so the UI can style it appropriately."""

    kind: str  # "strength" | "work_on" | "unmeasured" | "descriptor" | "meta"
    metric_key: Optional[str]
    text: str

    def as_dict(self) -> dict:
        return {"kind": self.kind, "metric_key": self.metric_key, "text": self.text}


@dataclass(frozen=True)
class FeedbackBundle:
    headline: str
    items: List[FeedbackItem]

    def as_dict(self) -> dict:
        return {"headline": self.headline, "items": [i.as_dict() for i in self.items]}


def _band_direction(score: MetricScore) -> str:
    """"below"/"above" the expected range, from the value alone."""
    assert score.band is not None
    return "below" if score.measurement.value < score.band.low else "above"


def _tier_b_band_word(score: MetricScore) -> str:
    """Banded wording for Tier B, which must never carry a precise number."""
    return {"below": "on the short side", "above": "on the long side"}[
        _band_direction(score)
    ]


def _phrase_in_range(score: MetricScore, spec) -> str:
    return f"{spec.summary} is where you want it."


def _phrase_out_of_range(score: MetricScore, spec) -> str:
    if spec.tier is Tier.B:
        return f"{spec.summary} looks {_tier_b_band_word(score)} for this shot."
    direction = _band_direction(score)
    return f"{spec.summary} is sitting {direction} the range for this shot."


def _phrase_unclear(score: MetricScore, spec) -> str:
    return (
        f"{spec.summary} could not be measured reliably from this clip. "
        f"A steadier camera at the recommended angle, or a higher frame rate, would resolve it."
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _headline(result: CompositeResult) -> str:
    if result.composite is None:
        return "Not enough of this clip was measurable to give an overall score."
    pct = round(result.composite * 100)
    if pct >= 80:
        return f"Solid overall - around {pct} out of 100 on what we could measure."
    if pct >= 55:
        return f"Reasonable overall - around {pct} out of 100, with clear things to work on."
    return f"Around {pct} out of 100 on what we could measure. Plenty to build on."


def _descriptor_line(key: str, value: str) -> str:
    """Neutral phrasing for style descriptors - stated, never judged."""
    spec = metric_tiers.get(key)
    return f"{spec.summary}: {value}. This is a style choice, not a fault."


def generate(
    result: CompositeResult,
    *,
    descriptors: Optional[Mapping[str, str]] = None,
    style_note: Optional[str] = None,
    max_work_on: int = 3,
) -> FeedbackBundle:
    """Assemble feedback from scored metrics.

    Parameters
    ----------
    result:
        Output of :func:`scoring.composite`.
    descriptors:
        Style markers (backlift direction, trigger movement, stance width) as
        ``{metric_key: display_value}``. Rendered neutrally.
    style_note:
        Optional intent/execution mismatch line from :mod:`style_resolution`, e.g. observed
        intent diverging from declared style. Advisory only.
    max_work_on:
        Cap on corrective items. Long fault lists are demotivating and rarely actionable;
        the lowest-scoring metrics are kept.
    """
    items: List[FeedbackItem] = []

    displayable = [s for s in result.scores if metric_tiers.is_displayable(s.key)]

    strengths = [s for s in displayable if s.verdict is Verdict.IN_RANGE]
    work_on = sorted(
        (s for s in displayable if s.verdict is Verdict.OUT_OF_RANGE),
        key=lambda s: s.score if s.score is not None else 1.0,
    )[:max_work_on]
    unclear = [s for s in displayable if s.verdict is Verdict.UNCLEAR]

    for s in strengths:
        spec = metric_tiers.get(s.key)
        items.append(
            FeedbackItem("strength", s.key, _assert_safe(_phrase_in_range(s, spec)))
        )

    for s in work_on:
        spec = metric_tiers.get(s.key)
        items.append(
            FeedbackItem("work_on", s.key, _assert_safe(_phrase_out_of_range(s, spec)))
        )

    for s in unclear:
        spec = metric_tiers.get(s.key)
        items.append(
            FeedbackItem("unmeasured", s.key, _assert_safe(_phrase_unclear(s, spec)))
        )

    for key, value in (descriptors or {}).items():
        metric_tiers.assert_displayable(key)
        items.append(
            FeedbackItem("descriptor", key, _assert_safe(_descriptor_line(key, value)))
        )

    if style_note:
        items.append(FeedbackItem("meta", None, _assert_safe(style_note)))

    if result.suppressed_reason:
        items.append(FeedbackItem("meta", None, result.suppressed_reason))

    return FeedbackBundle(headline=_assert_safe(_headline(result)), items=items)


def redact_for_client(metric_keys: Iterable[str]) -> List[str]:
    """Final gate before serialisation. Drops anything Tier C.

    Called by :mod:`render_output` as defence in depth - scoring and feedback already
    exclude Tier C, so anything caught here indicates a bug upstream.
    """
    return metric_tiers.filter_for_display(metric_keys)
