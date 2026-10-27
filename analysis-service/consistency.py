"""Repeatability across the clips in a session.

The third kind of feedback, and the only one that needs neither a reference population nor a
published direction: **how alike were your repetitions?**

That is a genuine form quality, not a consolation prize for lacking reference data. A player
who produces the same shot five times has something a player who produces five different
shots does not, whatever the absolute numbers say. And it is measurable from the player's own
clips alone, which is why it works today while population comparison does not.

The statistical trap, and the reason this module is careful
-----------------------------------------------------------
Observed spread across clips is **real variation plus measurement error**, added in
quadrature. On mesh footage the measurement error alone can exceed the variation being
looked for, so raw scatter would report a player as inconsistent when the camera was the
inconsistent party.

So every figure here subtracts the instrument: ``true_sd = sqrt(observed_sd^2 - sem^2)``,
and where the observed spread does not clear the measurement error, the metric reports
*unresolvable* rather than a number. Same discipline as :mod:`cohort.build`, applied to one
player instead of a population.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import metric_tiers

#: Clips needed before repeatability is worth reporting. Below this the spread estimate is
#: itself so unstable that it says more about sampling than about the player.
MIN_CLIPS = 3

#: Observed spread must exceed measurement error by this factor before the variation is
#: treated as real rather than instrumental.
MIN_SIGNAL_TO_NOISE = 1.2

#: Coefficient of variation bands, as fractions. Working conventions for turning a number
#: into a word, not published thresholds - which is why they only ever pick an adjective and
#: never feed a score.
CV_TIGHT = 0.10
CV_LOOSE = 0.25


@dataclass
class MetricConsistency:
    metric_key: str
    summary: str
    units: str
    n: int
    mean: float
    observed_sd: float
    measurement_sem: float
    true_sd: Optional[float]

    @property
    def resolvable(self) -> bool:
        """False when the spread is indistinguishable from measurement noise."""
        return self.true_sd is not None

    @property
    def coefficient_of_variation(self) -> Optional[float]:
        if not self.resolvable or abs(self.mean) < 1e-9:
            return None
        return self.true_sd / abs(self.mean)

    @property
    def verdict(self) -> str:
        if not self.resolvable:
            # Spread below the instrument's own error is not a failure to measure - it means
            # the player repeated themselves more tightly than the camera can resolve, which
            # is good news phrased honestly rather than an absent result.
            if self.observed_sd < self.measurement_sem:
                return "as tight as we can measure"
            return "not resolvable"
        cv = self.coefficient_of_variation
        if cv is None:
            return "not resolvable"
        if cv <= CV_TIGHT:
            return "very repeatable"
        if cv <= CV_LOOSE:
            return "fairly repeatable"
        return "varies a lot"

    def as_dict(self) -> dict:
        return {
            "metric_key": self.metric_key,
            "summary": self.summary,
            "units": self.units,
            "n": self.n,
            "mean": round(self.mean, 3),
            "observed_sd": round(self.observed_sd, 3),
            "measurement_sem": round(self.measurement_sem, 3),
            "true_sd": None if self.true_sd is None else round(self.true_sd, 3),
            "cv": None if self.coefficient_of_variation is None
                  else round(self.coefficient_of_variation, 3),
            "verdict": self.verdict,
            "resolvable": self.resolvable,
        }


@dataclass
class SessionConsistency:
    n_clips: int
    metrics: List[MetricConsistency]

    @property
    def resolvable(self) -> List[MetricConsistency]:
        return [m for m in self.metrics if m.resolvable]

    @property
    def headline(self) -> str:
        if self.n_clips < MIN_CLIPS:
            return (
                f"{self.n_clips} clip{'' if self.n_clips == 1 else 's'} - "
                f"record at least {MIN_CLIPS} of the same shot to see how repeatable you are."
            )
        usable = self.resolvable
        if not usable:
            return (
                f"Across {self.n_clips} clips, none of the measurements varied by more than "
                f"the measurement error - so nothing can be said about repeatability from "
                f"this footage."
            )
        tight = sum(1 for m in usable if m.verdict == "very repeatable")
        loose = sum(1 for m in usable if m.verdict == "varies a lot")
        if loose == 0:
            return f"Across {self.n_clips} clips you repeated yourself well on all {len(usable)} measurements."
        return (
            f"Across {self.n_clips} clips, {tight} of {len(usable)} measurements were tight "
            f"and {loose} varied noticeably."
        )

    def as_dict(self) -> dict:
        return {
            "n_clips": self.n_clips,
            "headline": self.headline,
            "resolvable_count": len(self.resolvable),
            "metrics": [m.as_dict() for m in self.metrics],
        }


def analyse(clips: Sequence[Sequence[dict]]) -> SessionConsistency:
    """Repeatability across clips.

    ``clips`` is one list of metric dicts per clip, each carrying ``key``, ``value``,
    ``sem`` and ``units`` - the shape the demo already stores for saved analyses.

    Only metrics present in **every** clip are compared. A metric measured in three clips of
    five would otherwise have its spread computed over a different, self-selected subset -
    and the clips where a metric failed to measure are rarely a random sample.
    """
    if len(clips) < MIN_CLIPS:
        return SessionConsistency(len(clips), [])

    per_metric: Dict[str, List[dict]] = {}
    for clip in clips:
        for m in clip:
            per_metric.setdefault(m["key"], []).append(m)

    out: List[MetricConsistency] = []
    for key, entries in sorted(per_metric.items()):
        if len(entries) != len(clips):
            continue
        try:
            spec = metric_tiers.get(key)
        except KeyError:
            continue
        if not metric_tiers.is_scorable(key):
            continue

        values = [float(e["value"]) for e in entries]
        sems = [float(e.get("sem", 0.0)) for e in entries]

        mean = statistics.fmean(values)
        observed_sd = statistics.stdev(values)
        sem = statistics.fmean(sems) if any(sems) else 0.0

        resolvable = sem <= 0 or observed_sd / sem >= MIN_SIGNAL_TO_NOISE
        true_sd = math.sqrt(max(observed_sd**2 - sem**2, 0.0)) if resolvable else None

        out.append(
            MetricConsistency(
                metric_key=key,
                summary=spec.summary or key,
                units=entries[0].get("units", ""),
                n=len(entries),
                mean=mean,
                observed_sd=observed_sd,
                measurement_sem=sem,
                true_sd=true_sd,
            )
        )

    return SessionConsistency(len(clips), out)
