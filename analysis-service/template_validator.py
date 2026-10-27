"""Reference-template validation.

Runs at service boot. A template that fails validation **stops the service** rather than
being skipped, because the failure mode it guards against - scoring against unsourced or
convention-ambiguous numbers - produces confident, wrong feedback that looks identical to
correct feedback.

Checks, beyond what ``schema.json`` expresses structurally:

* every metric key exists in :mod:`metric_tiers`
* declared ``tier`` and ``discriminates_skill`` match the registry, so a template cannot
  quietly promote a Tier C metric into scoring
* ``convention`` is present and appropriate for the units - angles must declare one, and
  non-angle metrics must explicitly declare ``null``
* ``source`` is non-empty unless the metric is an explicit ``todo``
* ``expected_low <= expected_high`` *after* conversion to the internal convention, which is
  where an inverted band would otherwise hide
* ``event`` appears in the template's ``phase_events``
* ``derived_conversion`` bands carry a ``derivation`` note
* Tier C metrics are absent entirely - they are never scored, so a range for one is a
  category error
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import metric_tiers
from conventions import AngleConvention, range_to_internal
from metric_tiers import Tier

TEMPLATE_DIR = Path(__file__).parent / "reference_templates"

ANGLE_UNITS = {"deg"}
NON_ANGLE_UNITS = {"s", "frac_stature", "pct_stature", "ratio", "count", "category"}


class TemplateError(ValueError):
    """Raised when a reference template is unusable."""


@dataclass
class ValidationReport:
    template: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if self.errors:
            joined = "\n  - ".join(self.errors)
            raise TemplateError(f"{self.template} failed validation:\n  - {joined}")


def _validate_metric(
    key: str,
    body: dict,
    *,
    phase_events: set[str],
    discipline: str,
    where: str,
    report: ValidationReport,
) -> None:
    prefix = f"{where}[{key}]"

    try:
        spec = metric_tiers.get(key)
    except KeyError as exc:
        report.errors.append(f"{prefix}: {exc}")
        return

    if spec.discipline.value != discipline:
        report.errors.append(
            f"{prefix}: discipline mismatch - metric belongs to {spec.discipline.value}, "
            f"template declares {discipline}"
        )

    if spec.tier is Tier.C:
        report.errors.append(
            f"{prefix}: Tier C metrics are never scored, so a reference range for one is a "
            f"category error. Reason this metric is Tier C: {spec.basis}"
        )
        return

    declared_tier = body.get("tier")
    if declared_tier != spec.tier.value:
        report.errors.append(
            f"{prefix}: declared tier {declared_tier!r} but registry says "
            f"{spec.tier.value!r}. Templates must not override the registry."
        )

    declared_disc = body.get("discriminates_skill")
    if declared_disc != spec.discriminates_skill:
        report.errors.append(
            f"{prefix}: declared discriminates_skill={declared_disc!r} but registry says "
            f"{spec.discriminates_skill!r}"
        )

    event = body.get("event")
    if event:
        endpoints = event.split("->")
        for ep in endpoints:
            if ep not in phase_events:
                report.errors.append(
                    f"{prefix}: event {ep!r} is not in phase_events {sorted(phase_events)}"
                )
    if event != spec.event:
        report.warnings.append(
            f"{prefix}: event {event!r} differs from registry event {spec.event!r}"
        )

    units = body.get("units")
    convention = body.get("convention", "__missing__")

    if convention == "__missing__":
        report.errors.append(
            f"{prefix}: 'convention' is mandatory. Use null for non-angle metrics - the "
            f"omission must be an explicit choice."
        )
        # Normalise so the band checks below can still run rather than raising on the
        # sentinel. The error above already fails the template.
        convention = None
    elif units in ANGLE_UNITS:
        if convention is None:
            report.errors.append(
                f"{prefix}: angle metric must declare a convention "
                f"(straight_180 or flexion_from_0); these are inverses."
            )
        else:
            try:
                AngleConvention(convention)
            except ValueError:
                report.errors.append(f"{prefix}: unknown convention {convention!r}")
    elif units in NON_ANGLE_UNITS and convention is not None:
        report.errors.append(
            f"{prefix}: non-angle metric (units={units!r}) must have convention null, "
            f"got {convention!r}"
        )

    derivation_kind = body.get("band_derivation")
    low = body.get("expected_low")
    high = body.get("expected_high")
    source = body.get("source", "")

    if derivation_kind == "todo":
        if low is not None or high is not None:
            report.errors.append(
                f"{prefix}: with band_derivation 'todo', expected_low and expected_high "
                f"must be null. Do not invent a placeholder range."
            )
        report.warnings.append(f"{prefix}: unscored (band_derivation=todo)")
        return

    if not source:
        report.errors.append(
            f"{prefix}: 'source' is required unless band_derivation is 'todo'. "
            f"Add a citation key to docs/SOURCES.md."
        )

    if derivation_kind == "derived_conversion" and not body.get("derivation"):
        report.errors.append(
            f"{prefix}: band_derivation 'derived_conversion' requires a 'derivation' note "
            f"stating every assumption made."
        )

    if derivation_kind == "observed_cohort":
        cohort = body.get("cohort")
        if not isinstance(cohort, dict):
            report.errors.append(
                f"{prefix}: band_derivation 'observed_cohort' requires a 'cohort' block "
                f"recording sample size and spread. Without it the band is indistinguishable "
                f"from a published one, which it is not."
            )
        else:
            missing = [
                k
                for k in ("n_clips", "mean", "observed_sd", "measurement_sem", "true_sd_estimate")
                if k not in cohort
            ]
            if missing:
                report.errors.append(
                    f"{prefix}: cohort block missing {', '.join(missing)}"
                )
            else:
                sem = cohort["measurement_sem"]
                observed = cohort["observed_sd"]
                if sem and observed and observed / sem < 1.3:
                    report.errors.append(
                        f"{prefix}: observed spread ({observed}) barely exceeds measurement "
                        f"error ({sem}), so this band mostly describes camera noise rather "
                        f"than how players differ."
                    )
                if cohort["n_clips"] < 8:
                    report.warnings.append(
                        f"{prefix}: derived from only {cohort['n_clips']} clips"
                    )
        if body.get("population") == "unknown":
            report.warnings.append(
                f"{prefix}: population is 'unknown' - record who these players were, or the "
                f"UI cannot warn a junior that they are being compared against adults"
            )

    if low is None or high is None:
        report.errors.append(
            f"{prefix}: expected_low and expected_high must both be set unless "
            f"band_derivation is 'todo'"
        )
        return

    # Check ordering as authored, BEFORE any convention conversion. range_to_internal
    # deliberately re-sorts its output, which would otherwise silently repair an inverted
    # band on angle metrics while catching it on every other unit.
    if low > high:
        report.errors.append(
            f"{prefix}: expected_low ({low}) > expected_high ({high}) as authored"
        )

    if units in ANGLE_UNITS and convention:
        lo_i, hi_i = range_to_internal(low, high, convention)
    else:
        lo_i, hi_i = (low, high) if low <= high else (high, low)

    if lo_i >= hi_i:
        report.errors.append(
            f"{prefix}: band is empty or inverted after conversion to the internal "
            f"convention: [{lo_i}, {hi_i}]"
        )

    if body.get("confidence") == "low" and not body.get("notes"):
        report.warnings.append(
            f"{prefix}: confidence 'low' without a note explaining why"
        )


def validate_template(data: dict, *, name: str) -> ValidationReport:
    report = ValidationReport(template=name)

    for required in ("form_type", "discipline", "version", "phase_events", "metrics"):
        if required not in data:
            report.errors.append(f"missing required key {required!r}")
    if report.errors:
        return report

    discipline = data["discipline"]
    phase_events = set(data["phase_events"])

    if data["form_type"] != name:
        report.errors.append(
            f"form_type {data['form_type']!r} does not match filename stem {name!r}"
        )

    for key, body in data["metrics"].items():
        _validate_metric(
            key,
            body,
            phase_events=phase_events,
            discipline=discipline,
            where="metrics",
            report=report,
        )

    for style, overrides in (data.get("style_variants") or {}).items():
        for key, body in overrides.items():
            _validate_metric(
                key,
                body,
                phase_events=phase_events,
                discipline=discipline,
                where=f"style_variants.{style}",
                report=report,
            )

    scorable = [
        k
        for k, b in data["metrics"].items()
        if b.get("band_derivation") != "todo"
        and k in metric_tiers.REGISTRY
        and metric_tiers.is_scorable(k)
    ]
    if not scorable:
        report.warnings.append(
            "no scorable metrics - this form type will always return a null composite score"
        )

    return report


def load_template(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    report = validate_template(data, name=path.stem)
    report.raise_if_failed()
    return data


def load_all(directory: Optional[Path] = None) -> Dict[str, dict]:
    """Load and validate every template. Raises on the first invalid one.

    Called at service boot so a bad template fails fast and loudly.
    """
    directory = directory or TEMPLATE_DIR
    out: Dict[str, dict] = {}
    for path in sorted(directory.glob("*.json")):
        if path.name == "schema.json":
            continue
        out[path.stem] = load_template(path)
    return out


def main() -> int:
    """CLI: ``python template_validator.py`` - validate everything, print a report."""
    directory = TEMPLATE_DIR
    exit_code = 0
    for path in sorted(directory.glob("*.json")):
        if path.name == "schema.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        report = validate_template(data, name=path.stem)
        status = "OK  " if report.ok else "FAIL"
        print(f"[{status}] {path.name}")
        for err in report.errors:
            print(f"    ERROR   {err}")
        for warn in report.warnings:
            print(f"    warning {warn}")
        if not report.ok:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
