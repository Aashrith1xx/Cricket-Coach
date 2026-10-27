"""Derive reference ranges from a folder of clips.

This is how a reference template stops being empty. Point it at clips of players whose
technique you want to treat as the benchmark, and it runs each one through the *same* gates
and metrics as a user's upload, aggregates whatever survives, and writes candidate bands into
the template with their provenance attached.

    python -m cohort.build --clips ./elite_drives --form-type cover_drive
    python -m cohort.build --clips ./elite_drives --form-type cover_drive --write

It is deliberately hard to get a band out of this. Four things will stop one being emitted,
and each corresponds to a way that cohort-derived ranges normally go wrong:

**Too few clips.** A band from three players describes those three players.

**Heterogeneous capture.** If the cohort was shot at mixed frame rates or camera angles, the
spread encodes the camera rather than the technique, and grading a user against it would
compare them to an artifact. This is the failure the whole tier system exists to prevent, and
it is trivially easy to inflict on yourself here.

**Measurement noise dominating.** Observed spread is real player variation *and* measurement
error added together. If the observed SD is no larger than the known measurement SEM, the
variation cannot be resolved at all, and a band drawn from it would be noise with a number
on it.

**Implausible spread.** A standard deviation wider than the plausible physical range of the
quantity means something failed rather than that players differ.

Nothing here invents a number. Every band that comes out carries its sample size, the capture
conditions it came from, and the estimated true spread separate from the measurement error.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import capture_gate  # noqa: E402
import metric_tiers  # noqa: E402
import normalisation  # noqa: E402
import phase_detection  # noqa: E402
import pose_extraction  # noqa: E402
import template_validator  # noqa: E402
import uncertainty  # noqa: E402
from capture_gate import CaptureMetadata  # noqa: E402
from metrics import batting as batting_metrics  # noqa: E402
from metrics import bowling as bowling_metrics  # noqa: E402
from uncertainty import Measurement  # noqa: E402

log = logging.getLogger("cohort")

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}

#: Below this many contributing clips, no band is emitted. Eight is not a statistical
#: threshold so much as a floor below which the mean is obviously unstable.
MIN_CLIPS_FOR_BAND = 8

#: Frame rates within the cohort must not differ by more than this factor. A cohort mixing
#: 60 and 240 fps has systematically different event-timing precision between its members.
MAX_FPS_RATIO = 2.0

#: Spread of estimated camera view angle across the cohort, in degrees.
MAX_VIEW_ANGLE_SPREAD_DEG = 15.0

#: Observed SD must exceed measurement SEM by this factor before the variation is treated as
#: real. At 1.0 the two are indistinguishable.
MIN_SIGNAL_TO_NOISE = 1.3

#: Beyond this many SDs from the cohort mean, a clip is flagged for inspection rather than
#: silently averaged in.
OUTLIER_SD = 3.0

#: Physically plausible spread per unit, above which a large SD means a processing failure
#: rather than genuine variation between players.
IMPLAUSIBLE_SD = {
    "deg": 45.0,
    "s": 1.0,
    "frac_stature": 0.5,
    "pct_stature": 50.0,
    "ratio": 3.0,
}


@dataclass
class ClipOutcome:
    path: str
    passed: bool
    reasons: List[str] = field(default_factory=list)
    measurements: List[Measurement] = field(default_factory=list)
    fps: Optional[float] = None
    view_deviation_deg: Optional[float] = None
    tracking_confidence: Optional[float] = None


@dataclass
class CandidateBand:
    metric_key: str
    units: str
    n: int
    mean: float
    observed_sd: float
    mean_measurement_sem: float
    true_sd_estimate: float
    low: float
    high: float
    outlier_clips: List[str] = field(default_factory=list)

    @property
    def signal_to_noise(self) -> float:
        return self.observed_sd / self.mean_measurement_sem if self.mean_measurement_sem else 0.0


@dataclass
class CohortReport:
    form_type: str
    discipline: str
    outcomes: List[ClipOutcome]
    bands: Dict[str, CandidateBand]
    rejected_metrics: Dict[str, str]
    cohort_warnings: List[str]

    @property
    def passed(self) -> List[ClipOutcome]:
        return [o for o in self.outcomes if o.passed]

    @property
    def failed(self) -> List[ClipOutcome]:
        return [o for o in self.outcomes if not o.passed]


# ---------------------------------------------------------------------------
# Per-clip analysis
# ---------------------------------------------------------------------------


def analyse_clip(
    path: Path, form_type: str, discipline: str, handedness: str, backend
) -> ClipOutcome:
    """Run one clip through the same gates and metrics a user upload gets."""
    outcome = ClipOutcome(path=str(path), passed=False)

    try:
        frames, fps, width, height = pose_extraction.read_frames(str(path))
    except (IOError, ValueError) as exc:
        outcome.reasons.append(f"unreadable: {exc}")
        return outcome

    outcome.fps = fps
    meta = CaptureMetadata(
        fps=fps,
        width=width,
        height=height,
        duration_s=len(frames) / fps,
        declared_view="side_on",
    )

    meta_gate = capture_gate.check_metadata(meta)
    if not meta_gate.passed:
        outcome.reasons.extend(r.detail for r in meta_gate.rejections)
        return outcome

    motion = capture_gate.estimate_camera_motion(frames)
    if motion > capture_gate.MAX_BACKGROUND_MOTION_FRACTION:
        outcome.reasons.append(f"camera motion {motion * 100:.2f}% of frame width per frame")
        return outcome

    xy, conf, boxes = backend.run(frames)
    seq = pose_extraction.PoseSequence(xy=xy, confidence=conf, fps=fps, bbox=boxes)
    outcome.tracking_confidence = float(seq.mean_confidence())

    pose_gate = capture_gate.check_pose_quality(
        seq, required_view="side_on", discipline=discipline, frame_height=height
    )
    outcome.view_deviation_deg = pose_gate.estimated_view_deviation_deg
    if not pose_gate.passed:
        outcome.reasons.extend(r.detail for r in pose_gate.rejections)
        return outcome

    if handedness == "auto":
        handedness = pose_extraction.infer_handedness(seq)

    scale = normalisation.estimate_scale(seq)
    detect = (
        phase_detection.detect_batting if discipline == "batting"
        else phase_detection.detect_bowling
    )
    timeline = detect(seq, handedness=handedness, scale_px=scale.stature_px)

    missing = timeline.missing(phase_detection.required_events(form_type))
    if missing:
        outcome.reasons.append(f"could not locate {', '.join(missing)}")
        return outcome

    module = batting_metrics if discipline == "batting" else bowling_metrics
    allowed = {
        s.key
        for s in metric_tiers.applicable(metric_tiers.Discipline(discipline), fps=fps)
    }
    outcome.measurements = module.compute_all(
        seq, timeline, scale, handedness=handedness, allowed=allowed
    )
    outcome.passed = True
    return outcome


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def check_homogeneity(outcomes: Sequence[ClipOutcome]) -> List[str]:
    """Warn where the cohort's capture conditions differ enough to pollute the bands."""
    warnings: List[str] = []
    passed = [o for o in outcomes if o.passed]
    if len(passed) < 2:
        return warnings

    rates = [o.fps for o in passed if o.fps]
    if rates and max(rates) / min(rates) > MAX_FPS_RATIO:
        warnings.append(
            f"Frame rates range {min(rates):.0f}-{max(rates):.0f} fps. Event-timing precision "
            f"differs systematically across the cohort, so timing bands will be part camera, "
            f"part technique. Prefer a cohort shot at one frame rate."
        )

    views = [o.view_deviation_deg for o in passed if o.view_deviation_deg is not None]
    if len(views) >= 2 and (max(views) - min(views)) > MAX_VIEW_ANGLE_SPREAD_DEG:
        warnings.append(
            f"Camera angle varies by {max(views) - min(views):.0f} degrees across the cohort. "
            f"Angles are measured in the image plane, so this inflates the spread of every "
            f"angle metric with pure geometry."
        )

    confidences = [o.tracking_confidence for o in passed if o.tracking_confidence]
    if confidences and min(confidences) < 0.65:
        warnings.append(
            f"Weakest clip tracked at {min(confidences):.2f} mean confidence. Consider dropping "
            f"the worst-tracked clips before deriving bands from them."
        )

    return warnings


def aggregate(
    outcomes: Sequence[ClipOutcome],
) -> tuple[Dict[str, CandidateBand], Dict[str, str]]:
    """Turn per-clip measurements into candidate bands, with reasons for every refusal."""
    by_metric: Dict[str, List[tuple]] = {}
    for outcome in outcomes:
        if not outcome.passed:
            continue
        for m in outcome.measurements:
            by_metric.setdefault(m.key, []).append((m, outcome.path))

    bands: Dict[str, CandidateBand] = {}
    rejected: Dict[str, str] = {}

    for key, entries in sorted(by_metric.items()):
        if not metric_tiers.is_scorable(key):
            rejected[key] = "not scorable (Tier C, descriptor, or non-discriminating)"
            continue

        n = len(entries)
        if n < MIN_CLIPS_FOR_BAND:
            rejected[key] = f"only {n} clips yielded it, need {MIN_CLIPS_FOR_BAND}"
            continue

        values = [m.value for m, _ in entries]
        mean = statistics.fmean(values)
        observed_sd = statistics.stdev(values)
        mean_sem = statistics.fmean([m.sem for m, _ in entries])
        units = entries[0][0].units

        implausible = IMPLAUSIBLE_SD.get(units)
        if implausible is not None and observed_sd > implausible:
            rejected[key] = (
                f"spread ({observed_sd:.2f} {units}) is wider than physically plausible - "
                f"suggests a processing failure rather than variation between players"
            )
            continue

        if mean_sem > 0 and observed_sd / mean_sem < MIN_SIGNAL_TO_NOISE:
            rejected[key] = (
                f"observed spread ({observed_sd:.3f}) is not meaningfully larger than the "
                f"measurement error ({mean_sem:.3f}) - real variation between these players "
                f"cannot be resolved from this footage"
            )
            continue

        # Observed variance is real variation plus measurement error. Subtracting the latter
        # gives a band describing how players actually differ, not how noisy the camera is.
        true_var = max(observed_sd**2 - mean_sem**2, 0.0)
        true_sd = math.sqrt(true_var)

        outliers = [
            path for (m, path) in entries if abs(m.value - mean) > OUTLIER_SD * observed_sd
        ]

        bands[key] = CandidateBand(
            metric_key=key,
            units=units,
            n=n,
            mean=mean,
            observed_sd=observed_sd,
            mean_measurement_sem=mean_sem,
            true_sd_estimate=true_sd,
            low=mean - true_sd,
            high=mean + true_sd,
            outlier_clips=outliers,
        )

    return bands, rejected


# ---------------------------------------------------------------------------
# Template writing
# ---------------------------------------------------------------------------


def to_template_entries(report: CohortReport, cohort_id: str) -> Dict[str, dict]:
    """Render bands as reference-template metric entries."""
    entries: Dict[str, dict] = {}
    for key, band in report.bands.items():
        spec = metric_tiers.get(key)
        entries[key] = {
            "tier": spec.tier.value,
            "event": spec.event,
            "units": band.units,
            "convention": "straight_180" if band.units == "deg" else None,
            "expected_low": round(band.low, 4),
            "expected_high": round(band.high, 4),
            "band_derivation": "observed_cohort",
            "source": cohort_id,
            "population": "unknown",
            "confidence": "low",
            "discriminates_skill": spec.discriminates_skill,
            "cohort": {
                "n_clips": band.n,
                "mean": round(band.mean, 4),
                "observed_sd": round(band.observed_sd, 4),
                "measurement_sem": round(band.mean_measurement_sem, 4),
                "true_sd_estimate": round(band.true_sd_estimate, 4),
                "signal_to_noise": round(band.signal_to_noise, 2),
                "derived_on": date.today().isoformat(),
            },
            "notes": (
                f"Derived from {band.n} clips, not from published literature. The band is "
                f"mean +/- one estimated true SD, with measurement error removed from the "
                f"observed spread. Population is 'unknown' until you record who these players "
                f"actually were - do not leave it unknown if you know."
            ),
        }
    return entries


def write_into_template(report: CohortReport, cohort_id: str, template_path: Path) -> None:
    """Merge candidate bands into a template file, replacing only `todo` entries.

    Existing sourced bands are never overwritten: a published range from a controlled study
    beats one derived from footage of unknown provenance, and silently replacing it would
    lose that.
    """
    data = json.loads(template_path.read_text(encoding="utf-8"))
    entries = to_template_entries(report, cohort_id)

    replaced, skipped = [], []
    for key, entry in entries.items():
        existing = data["metrics"].get(key)
        if existing and existing.get("band_derivation") not in (None, "todo"):
            skipped.append(key)
            continue
        data["metrics"][key] = entry
        replaced.append(key)

    major, minor, patch = data["version"].split(".")
    data["version"] = f"{major}.{int(minor) + 1}.0"

    template_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    print(f"\n  wrote {len(replaced)} bands into {template_path.name}")
    for key in replaced:
        print(f"    + {key}")
    for key in skipped:
        print(f"    . {key} (kept existing sourced band)")
    print(f"  template version -> {data['version']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(clips_dir: Path, form_type: str, handedness: str) -> CohortReport:
    from pose_backends import OnnxPoseBackend

    templates = template_validator.load_all()
    if form_type not in templates:
        raise SystemExit(f"Unknown form type {form_type!r}. Known: {sorted(templates)}")
    discipline = templates[form_type]["discipline"]

    paths = sorted(p for p in clips_dir.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
    if not paths:
        raise SystemExit(f"No video files in {clips_dir}")

    models = Path(__file__).resolve().parents[1] / "models"
    backend = OnnxPoseBackend(str(models / "rtmdet.onnx"), str(models / "rtmpose.onnx"))

    outcomes = []
    for i, path in enumerate(paths, 1):
        print(f"  [{i}/{len(paths)}] {path.name} ... ", end="", flush=True)
        outcome = analyse_clip(path, form_type, discipline, handedness, backend)
        print("ok" if outcome.passed else f"rejected: {outcome.reasons[0] if outcome.reasons else '?'}")
        outcomes.append(outcome)

    bands, rejected = aggregate(outcomes)
    return CohortReport(
        form_type=form_type,
        discipline=discipline,
        outcomes=outcomes,
        bands=bands,
        rejected_metrics=rejected,
        cohort_warnings=check_homogeneity(outcomes),
    )


def print_report(report: CohortReport) -> None:
    n_pass, n_total = len(report.passed), len(report.outcomes)
    print(f"\n{'=' * 76}")
    print(f"  {report.form_type}: {n_pass}/{n_total} clips usable")
    print(f"{'=' * 76}")

    if report.failed:
        print("\n  Rejected clips")
        for outcome in report.failed:
            print(f"    {Path(outcome.path).name}")
            for reason in outcome.reasons:
                print(f"        {reason}")

    for warning in report.cohort_warnings:
        print(f"\n  ! {warning}")

    if report.bands:
        print(f"\n  Candidate bands ({len(report.bands)})")
        print(f"    {'metric':<44} {'n':>3}  {'band':>22}  {'S/N':>5}")
        for key, band in report.bands.items():
            span = f"{band.low:.3f} to {band.high:.3f}"
            print(f"    {key:<44} {band.n:>3}  {span:>22}  {band.signal_to_noise:>5.1f}")
            for path in band.outlier_clips:
                print(f"        outlier, worth checking: {Path(path).name}")
    else:
        print("\n  No bands could be derived.")

    if report.rejected_metrics:
        print(f"\n  Metrics with no band ({len(report.rejected_metrics)})")
        for key, reason in report.rejected_metrics.items():
            print(f"    {key}\n        {reason}")

    if n_pass < MIN_CLIPS_FOR_BAND:
        print(
            f"\n  Only {n_pass} clips passed. Nothing will be emitted below "
            f"{MIN_CLIPS_FOR_BAND}; fix the rejections above or gather more footage."
        )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Derive reference ranges from a folder of clips."
    )
    parser.add_argument("--clips", required=True, type=Path)
    parser.add_argument("--form-type", required=True)
    parser.add_argument("--handedness", default="auto", choices=["auto", "left", "right"])
    parser.add_argument("--write", action="store_true", help="write bands into the template")
    parser.add_argument("--cohort-id", default=None, help="citation key for docs/SOURCES.md")
    parser.add_argument("--json", type=Path, help="also write the full report as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    report = run(args.clips, args.form_type, args.handedness)
    print_report(report)

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "form_type": report.form_type,
                    "clips": [o.__dict__ | {"measurements": len(o.measurements)} for o in report.outcomes],
                    "bands": {k: v.__dict__ for k, v in report.bands.items()},
                    "rejected_metrics": report.rejected_metrics,
                    "warnings": report.cohort_warnings,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"  full report -> {args.json}")

    if args.write:
        if not report.bands:
            print("  nothing to write.")
            return 1
        cohort_id = args.cohort_id or f"cohort_{date.today():%Y%m}_{report.form_type}"
        template = (
            Path(__file__).resolve().parents[1]
            / "reference_templates"
            / f"{report.form_type}.json"
        )
        write_into_template(report, cohort_id, template)
        print(
            f"\n  Now add a '{cohort_id}' entry to docs/SOURCES.md describing who these "
            f"players were and where the footage came from, then re-run "
            f"template_validator.py."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
