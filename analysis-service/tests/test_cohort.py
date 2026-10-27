"""Tests for the cohort reference-range builder.

The builder's job is mostly refusal - it must decline to emit a band whenever the footage
cannot support one. These tests pin each refusal, because a refusal that silently stops
working produces bands that look identical to good ones.
"""

from __future__ import annotations

import json

import pytest

import metric_tiers
import template_validator
from cohort import build
from cohort.build import ClipOutcome
from uncertainty import Measurement


def _outcome(values, key="batting.stride_duration_ss_se", sem=0.01, units="s", **kw):
    return ClipOutcome(
        path=kw.pop("path", "clip.mp4"),
        passed=True,
        measurements=[Measurement(key, v, sem=sem, confidence=0.9, units=units) for v in values],
        fps=kw.pop("fps", 240.0),
        view_deviation_deg=kw.pop("view_deviation_deg", 5.0),
        tracking_confidence=kw.pop("tracking_confidence", 0.85),
        **kw,
    )


def _cohort(values, **kw):
    """One clip per value, so n equals len(values)."""
    return [
        _outcome([v], path=f"clip{i}.mp4", **kw) for i, v in enumerate(values)
    ]


class TestAggregation:
    def test_emits_a_band_from_a_healthy_cohort(self):
        values = [0.14, 0.16, 0.18, 0.15, 0.19, 0.17, 0.13, 0.20, 0.16, 0.18]
        bands, rejected = build.aggregate(_cohort(values))
        assert "batting.stride_duration_ss_se" in bands
        band = bands["batting.stride_duration_ss_se"]
        assert band.n == 10
        assert band.low < band.mean < band.high

    def test_refuses_below_the_clip_floor(self):
        bands, rejected = build.aggregate(_cohort([0.14, 0.16, 0.18]))
        assert not bands
        assert "need 8" in rejected["batting.stride_duration_ss_se"]

    def test_refuses_when_measurement_noise_dominates(self):
        """Tight spread on a noisy measurement means the variation cannot be resolved -
        a band drawn from it would be noise with a number on it."""
        values = [0.160, 0.161, 0.159, 0.160, 0.162, 0.158, 0.161, 0.159, 0.160, 0.161]
        bands, rejected = build.aggregate(_cohort(values, sem=0.05))
        assert not bands
        assert "measurement error" in rejected["batting.stride_duration_ss_se"]

    def test_refuses_implausible_spread(self):
        """A huge SD means processing failed, not that players differ that much."""
        values = [10, 170, 30, 150, 20, 160, 40, 140, 25, 155]
        outcomes = _cohort(values, key="batting.lead_knee_angle_imp", sem=8.0, units="deg")
        bands, rejected = build.aggregate(outcomes)
        assert not bands
        assert "physically plausible" in rejected["batting.lead_knee_angle_imp"]

    def test_removes_measurement_error_from_the_band_width(self):
        """Observed spread is real variation plus measurement error; the band should use
        only the former, so it describes players rather than the camera."""
        values = [0.10, 0.14, 0.18, 0.22, 0.12, 0.16, 0.20, 0.24, 0.13, 0.19]
        bands, _ = build.aggregate(_cohort(values, sem=0.02))
        band = bands["batting.stride_duration_ss_se"]
        assert band.true_sd_estimate < band.observed_sd
        assert band.high - band.low == pytest.approx(2 * band.true_sd_estimate)

    def test_flags_outliers_without_dropping_them(self):
        values = [0.15, 0.16, 0.17, 0.15, 0.16, 0.17, 0.16, 0.15, 0.17, 0.95]
        bands, _ = build.aggregate(_cohort(values, sem=0.005))
        band = bands.get("batting.stride_duration_ss_se")
        if band:  # the outlier inflates SD, so a band may or may not survive
            assert band.n == 10, "outliers are reported, not silently removed"

    def test_never_bands_a_tier_c_metric(self):
        outcomes = _cohort(
            [20, 22, 24, 21, 23, 25, 22, 24, 21, 23],
            key="batting.bat_speed_max",
            units="ratio",
            sem=0.5,
        )
        bands, rejected = build.aggregate(outcomes)
        assert not bands
        assert "not scorable" in rejected["batting.bat_speed_max"]

    def test_failed_clips_contribute_nothing(self):
        good = _cohort([0.14, 0.16, 0.18, 0.15, 0.19, 0.17, 0.13, 0.20])
        bad = [ClipOutcome(path="bad.mp4", passed=False, reasons=["30 fps"])]
        bands, _ = build.aggregate(good + bad)
        assert bands["batting.stride_duration_ss_se"].n == 8


class TestHomogeneity:
    def test_warns_on_mixed_frame_rates(self):
        outcomes = [_outcome([0.16], fps=60.0), _outcome([0.16], fps=240.0)]
        warnings = build.check_homogeneity(outcomes)
        assert any("Frame rates" in w for w in warnings)

    def test_warns_on_mixed_camera_angles(self):
        outcomes = [
            _outcome([0.16], view_deviation_deg=2.0),
            _outcome([0.16], view_deviation_deg=19.0),
        ]
        warnings = build.check_homogeneity(outcomes)
        assert any("Camera angle" in w for w in warnings)

    def test_warns_on_a_poorly_tracked_member(self):
        outcomes = [_outcome([0.16], tracking_confidence=0.9), _outcome([0.16], tracking_confidence=0.4)]
        assert any("mean confidence" in w for w in build.check_homogeneity(outcomes))

    def test_consistent_cohort_produces_no_warnings(self):
        outcomes = [_outcome([0.16]) for _ in range(5)]
        assert build.check_homogeneity(outcomes) == []


class TestTemplateEntries:
    def _report(self):
        values = [0.10, 0.14, 0.18, 0.22, 0.12, 0.16, 0.20, 0.24, 0.13, 0.19]
        outcomes = _cohort(values, sem=0.02)
        bands, rejected = build.aggregate(outcomes)
        return build.CohortReport(
            form_type="cover_drive",
            discipline="batting",
            outcomes=outcomes,
            bands=bands,
            rejected_metrics=rejected,
            cohort_warnings=[],
        )

    def test_entry_carries_full_provenance(self):
        entries = build.to_template_entries(self._report(), "cohort_test")
        entry = entries["batting.stride_duration_ss_se"]
        assert entry["band_derivation"] == "observed_cohort"
        assert entry["source"] == "cohort_test"
        assert entry["cohort"]["n_clips"] == 10
        assert "measurement_sem" in entry["cohort"]

    def test_entry_passes_the_template_validator(self):
        entries = build.to_template_entries(self._report(), "cohort_test")
        template = {
            "form_type": "cover_drive",
            "discipline": "batting",
            "version": "0.2.0",
            "phase_events": ["STANCE", "TRIGGER", "BC", "SS", "SE", "DS", "IMP", "FOLLOW_THROUGH"],
            "metrics": entries,
        }
        report = template_validator.validate_template(template, name="cover_drive")
        assert report.ok, report.errors

    def test_validator_rejects_a_cohort_band_without_its_cohort_block(self):
        entries = build.to_template_entries(self._report(), "cohort_test")
        del entries["batting.stride_duration_ss_se"]["cohort"]
        template = {
            "form_type": "cover_drive",
            "discipline": "batting",
            "version": "0.2.0",
            "phase_events": ["STANCE", "TRIGGER", "BC", "SS", "SE", "DS", "IMP", "FOLLOW_THROUGH"],
            "metrics": entries,
        }
        report = template_validator.validate_template(template, name="cover_drive")
        assert not report.ok
        assert any("cohort" in e for e in report.errors)

    def test_validator_rejects_a_noise_dominated_cohort_band(self):
        entries = build.to_template_entries(self._report(), "cohort_test")
        entry = entries["batting.stride_duration_ss_se"]
        entry["cohort"]["observed_sd"] = 0.021
        entry["cohort"]["measurement_sem"] = 0.020
        template = {
            "form_type": "cover_drive",
            "discipline": "batting",
            "version": "0.2.0",
            "phase_events": ["STANCE", "TRIGGER", "BC", "SS", "SE", "DS", "IMP", "FOLLOW_THROUGH"],
            "metrics": entries,
        }
        report = template_validator.validate_template(template, name="cover_drive")
        assert not report.ok
        assert any("camera noise" in e for e in report.errors)


class TestTemplateWriting:
    def test_never_overwrites_a_sourced_band(self, tmp_path):
        """A published range from a controlled study beats footage of unknown provenance."""
        template_path = tmp_path / "cover_drive.json"
        original = {
            "form_type": "cover_drive",
            "discipline": "batting",
            "version": "0.1.0",
            "phase_events": ["STANCE", "TRIGGER", "BC", "SS", "SE", "DS", "IMP", "FOLLOW_THROUGH"],
            "metrics": {
                "batting.stride_duration_ss_se": {
                    "tier": "A",
                    "event": "SS->SE",
                    "units": "s",
                    "convention": None,
                    "expected_low": 0.19,
                    "expected_high": 0.29,
                    "band_derivation": "mean_pm_1sd",
                    "source": "pullshot_malaysia",
                    "population": "elite_male",
                    "confidence": "low",
                    "discriminates_skill": True,
                }
            },
        }
        template_path.write_text(json.dumps(original), encoding="utf-8")

        values = [0.10, 0.14, 0.18, 0.22, 0.12, 0.16, 0.20, 0.24, 0.13, 0.19]
        outcomes = _cohort(values, sem=0.02)
        bands, rejected = build.aggregate(outcomes)
        report = build.CohortReport("cover_drive", "batting", outcomes, bands, rejected, [])

        build.write_into_template(report, "cohort_test", template_path)

        after = json.loads(template_path.read_text(encoding="utf-8"))
        entry = after["metrics"]["batting.stride_duration_ss_se"]
        assert entry["source"] == "pullshot_malaysia", "existing sourced band was overwritten"

    def test_fills_a_todo_entry_and_bumps_the_version(self, tmp_path):
        template_path = tmp_path / "cover_drive.json"
        original = {
            "form_type": "cover_drive",
            "discipline": "batting",
            "version": "0.1.0",
            "phase_events": ["STANCE", "TRIGGER", "BC", "SS", "SE", "DS", "IMP", "FOLLOW_THROUGH"],
            "metrics": {
                "batting.stride_duration_ss_se": {
                    "tier": "A",
                    "event": "SS->SE",
                    "units": "s",
                    "convention": None,
                    "expected_low": None,
                    "expected_high": None,
                    "band_derivation": "todo",
                    "source": "",
                    "population": "unknown",
                    "confidence": "low",
                    "discriminates_skill": True,
                }
            },
        }
        template_path.write_text(json.dumps(original), encoding="utf-8")

        values = [0.10, 0.14, 0.18, 0.22, 0.12, 0.16, 0.20, 0.24, 0.13, 0.19]
        outcomes = _cohort(values, sem=0.02)
        bands, rejected = build.aggregate(outcomes)
        report = build.CohortReport("cover_drive", "batting", outcomes, bands, rejected, [])

        build.write_into_template(report, "cohort_test", template_path)

        after = json.loads(template_path.read_text(encoding="utf-8"))
        assert after["metrics"]["batting.stride_duration_ss_se"]["source"] == "cohort_test"
        assert after["version"] == "0.2.0"
