"""Calibration harness: measure this system's real accuracy against reference labels.

**Status: specified stub.** Statistics are real; clip ingestion is TODO.

See ``README.md`` in this directory for what a pass looks like and how to source reference
data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Z_95 = 1.959963984540054


@dataclass
class AgreementStats:
    """Bland-Altman style agreement between our value and a reference."""

    metric_key: str
    n: int
    bias: float
    sd_of_differences: float
    rmse: float
    assumed_sem: float

    @property
    def limits_of_agreement(self) -> Tuple[float, float]:
        return (
            self.bias - Z_95 * self.sd_of_differences,
            self.bias + Z_95 * self.sd_of_differences,
        )

    @property
    def passes(self) -> bool:
        """RMSE no worse than the SEM assumed in :mod:`uncertainty`, and bias plausibly zero."""
        if self.n < 2:
            return False
        se_bias = self.sd_of_differences / math.sqrt(self.n)
        bias_ok = abs(self.bias) <= Z_95 * se_bias
        return bias_ok and self.rmse <= self.assumed_sem


@dataclass
class CoverageStats:
    """Whether stated confidence intervals actually contain the reference value.

    Overconfidence is the dangerous direction: it turns measurement noise into verdicts,
    which is precisely what the ``UNCLEAR`` mechanism in :mod:`scoring` exists to prevent.
    """

    metric_key: str
    n: int
    covered: int

    @property
    def coverage(self) -> float:
        return self.covered / self.n if self.n else 0.0

    @property
    def verdict(self) -> str:
        c = self.coverage
        if c < 0.90:
            return "overconfident - intervals too narrow"
        if c > 0.98:
            return "underconfident - intervals too wide to be useful"
        return "calibrated"


@dataclass
class EventStats:
    event_name: str
    n: int
    mean_abs_frame_error: float
    p90_frame_error: float
    detection_rate: float


@dataclass
class HarnessReport:
    agreement: List[AgreementStats] = field(default_factory=list)
    coverage: List[CoverageStats] = field(default_factory=list)
    events: List[EventStats] = field(default_factory=list)
    gate_rejection_rate: Optional[float] = None
    gate_rejection_reasons: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def failing_metrics(self) -> List[str]:
        """Metrics that must be demoted a tier, not merely given a wider error bar."""
        failed = {a.metric_key for a in self.agreement if not a.passes}
        failed |= {c.metric_key for c in self.coverage if c.verdict.startswith("overconfident")}
        return sorted(failed)


def agreement(
    metric_key: str,
    ours: Sequence[float],
    reference: Sequence[float],
    *,
    assumed_sem: float,
) -> AgreementStats:
    """Bias, scatter and RMSE against reference values. Implemented - pure statistics."""
    if len(ours) != len(reference):
        raise ValueError("ours and reference must be the same length")
    n = len(ours)
    if n == 0:
        raise ValueError("no paired observations")

    diffs = [o - r for o, r in zip(ours, reference)]
    bias = sum(diffs) / n
    if n > 1:
        var = sum((d - bias) ** 2 for d in diffs) / (n - 1)
        sd = math.sqrt(var)
    else:
        sd = 0.0
    rmse = math.sqrt(sum(d * d for d in diffs) / n)

    return AgreementStats(
        metric_key=metric_key,
        n=n,
        bias=bias,
        sd_of_differences=sd,
        rmse=rmse,
        assumed_sem=assumed_sem,
    )


def coverage(
    metric_key: str,
    intervals: Sequence[Tuple[float, float]],
    reference: Sequence[float],
) -> CoverageStats:
    """Fraction of reference values falling inside our stated intervals. Implemented."""
    if len(intervals) != len(reference):
        raise ValueError("intervals and reference must be the same length")
    covered = sum(1 for (lo, hi), r in zip(intervals, reference) if lo <= r <= hi)
    return CoverageStats(metric_key=metric_key, n=len(reference), covered=covered)


def run(clips_dir: str, labels_path: str) -> HarnessReport:
    """Run the full harness over a labelled clip set.

    TODO:

    * Load reference labels: per-clip event frames and per-metric reference values.
    * Run the real pipeline over each clip - the same code path production uses, not a
      reimplementation, or the harness validates something users never touch.
    * Record gate rejections separately. A rejected clip is not a measurement failure, but
      the rejection *rate* is itself a result worth reporting.
    * Emit a report keyed by metric, so ``failing_metrics`` can drive tier demotions.
    """
    raise NotImplementedError("validation.harness.run is not implemented")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cricket form analysis calibration harness")
    parser.add_argument("--clips", required=True, help="directory of labelled clips")
    parser.add_argument("--labels", required=True, help="reference labels JSON")
    parser.add_argument("--out", default="report.json")
    args = parser.parse_args()

    report = run(args.clips, args.labels)
    print(f"Failing metrics: {report.failing_metrics or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
