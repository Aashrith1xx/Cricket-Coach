"""Tests for the safety and accuracy guards.

These are the tests that matter most in this codebase. The implemented logic is not
complicated; what it is doing is *refusing* to say things, and a refusal that silently stops
working looks exactly like everything being fine. Each test below pins one refusal.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import conventions
import feedback
import metric_tiers
import scoring
import template_validator
import uncertainty
from conventions import AngleConvention
from metric_tiers import Discipline, SuppressedMetricError, Tier
from scoring import ExpectedBand, Verdict
from uncertainty import Measurement

TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "reference_templates"


# ---------------------------------------------------------------------------
# Tier suppression
# ---------------------------------------------------------------------------


class TestTierSuppression:
    def test_elbow_extension_is_tier_c(self):
        """ICC's threshold is 15 deg; measured elbow error is 22.7 deg."""
        assert metric_tiers.get("bowling.elbow_extension_release").tier is Tier.C
        assert not metric_tiers.is_displayable("bowling.elbow_extension_release")
        assert not metric_tiers.is_scorable("bowling.elbow_extension_release")

    def test_counter_rotation_is_tier_c(self):
        """Risk threshold 30 deg is smaller than the 28.5 deg measurement error."""
        assert metric_tiers.get("bowling.shoulder_counter_rotation").tier is Tier.C
        assert "bowling.shoulder_counter_rotation" in metric_tiers.SAFETY_CRITICAL

    def test_bat_speed_is_tier_c_and_non_discriminating(self):
        spec = metric_tiers.get("batting.bat_speed_max")
        assert spec.tier is Tier.C
        assert spec.discriminates_skill is False

    def test_assert_displayable_raises_for_tier_c(self):
        with pytest.raises(SuppressedMetricError):
            metric_tiers.assert_displayable("batting.bat_speed_max")

    def test_assert_scorable_raises_for_descriptor(self):
        with pytest.raises(SuppressedMetricError):
            metric_tiers.assert_scorable("batting.backlift_direction")

    def test_filter_for_display_drops_tier_c_only(self):
        keys = [
            "batting.lead_knee_angle_imp",
            "batting.bat_speed_max",
            "batting.backlift_direction",
        ]
        out = metric_tiers.filter_for_display(keys)
        assert "batting.bat_speed_max" not in out
        assert "batting.backlift_direction" in out, "descriptors are shown, just not scored"

    def test_every_tier_c_metric_has_a_basis(self):
        for key in metric_tiers.TIER_C_SUPPRESSED:
            assert metric_tiers.get(key).basis.strip(), f"{key} lacks a documented reason"

    def test_tier_c_metrics_have_no_user_facing_summary(self):
        for key in metric_tiers.TIER_C_SUPPRESSED:
            assert metric_tiers.get(key).summary == "", (
                f"{key} has a user-facing summary but must never be shown"
            )


# ---------------------------------------------------------------------------
# Capture conditions gate which metrics exist at all
# ---------------------------------------------------------------------------


class TestApplicability:
    def test_low_fps_suppresses_impact_referenced_metrics(self):
        """At 30 fps, bat-ball contact is a sub-frame event and cannot be located."""
        keys = {s.key for s in metric_tiers.applicable(Discipline.BATTING, fps=30)}
        assert "batting.lead_knee_angle_imp" not in keys
        assert "batting.stride_duration_ss_se" in keys, "pure timing survives low fps"

    def test_120fps_unlocks_impact_metrics(self):
        keys = {s.key for s in metric_tiers.applicable(Discipline.BATTING, fps=120)}
        assert "batting.lead_knee_angle_imp" in keys

    def test_x_factor_requires_front_on_camera(self):
        side_on = {s.key for s in metric_tiers.applicable(Discipline.BATTING, fps=240)}
        both = {
            s.key
            for s in metric_tiers.applicable(
                Discipline.BATTING, fps=240, has_front_on=True
            )
        }
        assert "batting.x_factor_ds" not in side_on
        assert "batting.x_factor_ds" in both


# ---------------------------------------------------------------------------
# Angle conventions
# ---------------------------------------------------------------------------


class TestConventions:
    def test_flexion_from_zero_inverts(self):
        assert conventions.to_internal(30.0, AngleConvention.FLEXION_FROM_0) == 150.0

    def test_straight_180_is_identity(self):
        assert conventions.to_internal(150.0, AngleConvention.STRAIGHT_180) == 150.0

    def test_range_conversion_reorders_bounds(self):
        """The inverting conversion swaps low and high - forgetting this yields a band
        that silently passes every value it should fail."""
        low, high = conventions.range_to_internal(
            0.0, 30.0, AngleConvention.FLEXION_FROM_0
        )
        assert (low, high) == (150.0, 180.0)
        assert low < high

    def test_round_trip(self):
        for conv in AngleConvention:
            v = conventions.to_internal(42.0, conv)
            back = conventions.to_internal(v, conv)
            assert math.isclose(back, 42.0)


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


class TestUncertainty:
    def test_temporal_sem_improves_with_fps(self):
        assert uncertainty.temporal_sem(240) < uncertainty.temporal_sem(30)

    def test_interval_sem_exceeds_single_event_sem(self):
        assert uncertainty.interval_sem(120) > uncertainty.temporal_sem(120)

    def test_difference_partially_cancels_systematic_error(self):
        """Correlated errors cancel; treating them as independent overstates uncertainty."""
        independent = uncertainty.difference_sem(10.0, 10.0, correlation=0.0)
        correlated = uncertainty.difference_sem(10.0, 10.0, correlation=0.5)
        assert correlated < independent

    def test_mdc_shrinks_with_session_size(self):
        """The arithmetic reason the app asks for 3-5 clips rather than one."""
        single = uncertainty.mdc(10.0, n=1)
        session = uncertainty.mdc(10.0, n=5)
        assert single > session
        assert single == pytest.approx(27.7, abs=0.5)

    def test_low_confidence_widens_interval(self):
        good = uncertainty.scale_sem_by_confidence(8.0, 0.95)
        poor = uncertainty.scale_sem_by_confidence(8.0, 0.3)
        assert poor > good

    def test_straddles_detects_boundary_crossing(self):
        m = Measurement("batting.lead_knee_angle_imp", 150.0, sem=8.0, confidence=1.0, units="deg")
        assert uncertainty.straddles(m, 145.0, 175.0), "CI crosses the lower edge"

        tight = Measurement(
            "batting.lead_knee_angle_imp", 160.0, sem=1.0, confidence=1.0, units="deg"
        )
        assert not uncertainty.straddles(tight, 145.0, 175.0)

    def test_session_aggregate_uses_larger_of_scatter_and_instrument_error(self):
        ms = [
            Measurement("batting.lead_knee_angle_imp", v, sem=8.0, confidence=1.0, units="deg")
            for v in (150.0, 152.0, 148.0)
        ]
        agg = uncertainty.aggregate_session(ms)
        assert agg is not None
        assert agg.n == 3
        assert agg.mean == pytest.approx(150.0)
        assert agg.sem >= 8.0 / 3, "instrument floor prevents overstated precision"

    def test_aggregate_rejects_mixed_metrics(self):
        with pytest.raises(ValueError):
            uncertainty.aggregate_session(
                [
                    Measurement("batting.lead_knee_angle_imp", 1.0, 1.0, 1.0, "deg"),
                    Measurement("batting.stride_length_norm", 1.0, 1.0, 1.0, "frac_stature"),
                ]
            )

    def test_small_change_is_not_detectable(self):
        earlier = uncertainty.SessionAggregate("batting.lead_knee_angle_imp", 150.0, 4.0, 5, "deg")
        later = uncertainty.SessionAggregate("batting.lead_knee_angle_imp", 153.0, 4.0, 5, "deg")
        assert not uncertainty.change_is_detectable(earlier, later), (
            "a 3 deg move on an 8 deg RMSE metric is noise, not progress"
        )

    def test_large_change_is_detectable(self):
        earlier = uncertainty.SessionAggregate("batting.lead_knee_angle_imp", 150.0, 2.0, 5, "deg")
        later = uncertainty.SessionAggregate("batting.lead_knee_angle_imp", 172.0, 2.0, 5, "deg")
        assert uncertainty.change_is_detectable(earlier, later)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _band(low, high, convention=None, **kw):
    return ExpectedBand(
        low=low,
        high=high,
        convention=convention,
        source=kw.get("source", "test_source"),
        population=kw.get("population", "elite_male"),
        confidence=kw.get("confidence", "medium"),
    )


class TestScoring:
    def test_confident_in_range_scores_full(self):
        m = Measurement("batting.lead_knee_angle_imp", 160.0, sem=1.0, confidence=1.0, units="deg")
        s = scoring.score_metric(m, _band(145.0, 175.0, "straight_180"))
        assert s.verdict is Verdict.IN_RANGE
        assert s.score == 1.0

    def test_straddling_interval_is_unclear_not_a_verdict(self):
        m = Measurement("batting.lead_knee_angle_imp", 146.0, sem=8.0, confidence=1.0, units="deg")
        s = scoring.score_metric(m, _band(145.0, 175.0, "straight_180"))
        assert s.verdict is Verdict.UNCLEAR
        assert s.score is None
        assert s.weight > 0, "unclear still carries weight, it just has no score"

    def test_clearly_out_of_range_scores_by_distance(self):
        m = Measurement("batting.lead_knee_angle_imp", 100.0, sem=1.0, confidence=1.0, units="deg")
        s = scoring.score_metric(m, _band(145.0, 175.0, "straight_180"))
        assert s.verdict is Verdict.OUT_OF_RANGE
        assert 0.0 <= s.score < 1.0

    def test_tier_c_is_refused_not_scored_zero(self):
        m = Measurement("batting.bat_speed_max", 25.0, sem=1.0, confidence=1.0, units="ratio")
        s = scoring.score_metric(m, _band(20.0, 30.0))
        assert s.verdict is Verdict.NOT_SCORED
        assert s.weight == 0.0

    def test_missing_band_yields_not_scored(self):
        m = Measurement("batting.lead_knee_angle_imp", 160.0, sem=1.0, confidence=1.0, units="deg")
        s = scoring.score_metric(m, None)
        assert s.verdict is Verdict.NOT_SCORED
        assert "no sourced reference range" in s.note

    def test_temporal_metrics_outweigh_angles(self):
        """Inverts the intuitive design; follows from the error figures."""
        temporal = scoring.score_metric(
            Measurement("batting.stride_duration_ss_se", 0.24, 0.01, 1.0, "s"),
            _band(0.19, 0.29),
        )
        angle = scoring.score_metric(
            Measurement("batting.lead_knee_angle_imp", 160.0, 1.0, 1.0, "deg"),
            _band(145.0, 175.0, "straight_180"),
        )
        assert temporal.weight > angle.weight

    def test_tier_b_weighted_below_tier_a(self):
        a = scoring.score_metric(
            Measurement("batting.lead_knee_angle_imp", 160.0, 1.0, 1.0, "deg"),
            _band(145.0, 175.0, "straight_180"),
        )
        b = scoring.score_metric(
            Measurement("batting.stride_length_norm", 0.38, 0.005, 1.0, "frac_stature"),
            _band(0.32, 0.44),
        )
        assert b.weight < a.weight

    def test_inverted_convention_band_is_handled(self):
        """A 160 deg knee is well extended. Under flexion_from_0 the same band is [5, 35]."""
        m = Measurement("batting.lead_knee_angle_imp", 160.0, sem=1.0, confidence=1.0, units="deg")
        straight = scoring.score_metric(m, _band(145.0, 175.0, "straight_180"))
        flexion = scoring.score_metric(m, _band(5.0, 35.0, "flexion_from_0"))
        assert straight.verdict is flexion.verdict is Verdict.IN_RANGE

    def test_composite_suppressed_when_evidence_is_thin(self):
        m = Measurement("batting.lead_knee_angle_imp", 160.0, sem=1.0, confidence=1.0, units="deg")
        result = scoring.composite([scoring.score_metric(m, _band(145.0, 175.0, "straight_180"))])
        assert result.composite is None
        assert result.suppressed_reason is not None

    def test_composite_computed_with_enough_weight(self):
        scores = [
            scoring.score_metric(
                Measurement(k, v, sem=s, confidence=1.0, units=u), _band(lo, hi, c)
            )
            for k, v, s, u, lo, hi, c in [
                ("batting.stride_duration_ss_se", 0.24, 0.005, "s", 0.19, 0.29, None),
                ("batting.interval_se_imp", 0.06, 0.003, "s", 0.03, 0.12, None),
                ("batting.tempo_bc_ds", 0.30, 0.005, "s", 0.20, 0.40, None),
                ("batting.lead_knee_angle_imp", 160.0, 1.0, "deg", 145.0, 175.0, "straight_180"),
            ]
        ]
        result = scoring.composite(scores)
        assert result.composite == pytest.approx(1.0)
        assert result.total_weight >= scoring.MIN_COMPOSITE_WEIGHT

    def test_unclear_metrics_widen_the_composite_interval(self):
        base = [
            scoring.score_metric(
                Measurement(k, v, sem=0.005, confidence=1.0, units="s"), _band(lo, hi)
            )
            for k, v, lo, hi in [
                ("batting.stride_duration_ss_se", 0.24, 0.19, 0.29),
                ("batting.interval_se_imp", 0.06, 0.03, 0.12),
                ("batting.tempo_bc_ds", 0.30, 0.20, 0.40),
            ]
        ]
        clean = scoring.composite(base)

        muddied = base + [
            scoring.score_metric(
                Measurement("batting.lead_knee_angle_imp", 146.0, 8.0, 1.0, "deg"),
                _band(145.0, 175.0, "straight_180"),
            )
        ]
        murky = scoring.composite(muddied)
        clean_width = clean.composite_ci_high - clean.composite_ci_low
        murky_width = murky.composite_ci_high - murky.composite_ci_low
        assert murky_width > clean_width


# ---------------------------------------------------------------------------
# Feedback guards
# ---------------------------------------------------------------------------


class TestFeedbackGuards:
    @pytest.mark.parametrize(
        "text",
        [
            "Your action looks safe.",
            "This reduces injury risk.",
            "Your elbow extension is within the legal limit.",
            "Increase your bat speed.",
            "This looks like a mixed action.",
            "There is a risk to your lumbar spine.",
        ],
    )
    def test_forbidden_phrases_are_rejected(self, text):
        with pytest.raises(feedback.FeedbackGuardError):
            feedback._assert_safe(text)

    def test_ordinary_coaching_text_passes(self):
        feedback._assert_safe("Front knee angle at contact is where you want it.")

    def test_descriptors_are_phrased_neutrally(self):
        line = feedback._descriptor_line("batting.backlift_direction", "lateral")
        assert "style choice" in line
        assert "fault" in line and "not a fault" in line

    def test_generate_excludes_tier_c_scores(self):
        scores = [
            scoring.score_metric(
                Measurement("batting.bat_speed_max", 25.0, 1.0, 1.0, "ratio"),
                _band(20.0, 30.0),
            ),
            scoring.score_metric(
                Measurement("batting.lead_knee_angle_imp", 160.0, 1.0, 1.0, "deg"),
                _band(145.0, 175.0, "straight_180"),
            ),
        ]
        bundle = feedback.generate(scoring.composite(scores))
        assert all(i.metric_key != "batting.bat_speed_max" for i in bundle.items)

    def test_unclear_metric_produces_capture_advice_not_a_verdict(self):
        s = scoring.score_metric(
            Measurement("batting.lead_knee_angle_imp", 146.0, 8.0, 1.0, "deg"),
            _band(145.0, 175.0, "straight_180"),
        )
        bundle = feedback.generate(scoring.composite([s]))
        unmeasured = [i for i in bundle.items if i.kind == "unmeasured"]
        assert unmeasured
        assert "could not be measured reliably" in unmeasured[0].text

    def test_suppressed_composite_produces_honest_headline(self):
        s = scoring.score_metric(
            Measurement("batting.lead_knee_angle_imp", 160.0, 1.0, 1.0, "deg"),
            _band(145.0, 175.0, "straight_180"),
        )
        bundle = feedback.generate(scoring.composite([s]))
        assert "Not enough" in bundle.headline

    def test_descriptor_passed_to_generate_is_rendered(self):
        s = scoring.score_metric(
            Measurement("batting.lead_knee_angle_imp", 160.0, 1.0, 1.0, "deg"),
            _band(145.0, 175.0, "straight_180"),
        )
        bundle = feedback.generate(
            scoring.composite([s]),
            descriptors={"batting.backlift_direction": "lateral"},
        )
        assert any(i.kind == "descriptor" for i in bundle.items)

    def test_tier_c_descriptor_is_refused(self):
        s = scoring.composite([])
        with pytest.raises(SuppressedMetricError):
            feedback.generate(s, descriptors={"batting.bat_speed_max": "fast"})


# ---------------------------------------------------------------------------
# Template validation
# ---------------------------------------------------------------------------


BASE_TEMPLATE = {
    "form_type": "cover_drive",
    "discipline": "batting",
    "version": "0.1.0",
    "phase_events": ["STANCE", "TRIGGER", "BC", "SS", "SE", "DS", "IMP", "FOLLOW_THROUGH"],
    "metrics": {},
}


def _template(metrics: dict) -> dict:
    return {**BASE_TEMPLATE, "metrics": metrics}


GOOD_METRIC = {
    "tier": "A",
    "event": "IMP",
    "units": "deg",
    "convention": "straight_180",
    "expected_low": 145.0,
    "expected_high": 175.0,
    "band_derivation": "mean_pm_1sd",
    "source": "pullshot_malaysia",
    "population": "elite_male",
    "confidence": "medium",
    "discriminates_skill": True,
}


class TestTemplateValidator:
    def test_shipped_templates_all_validate(self):
        loaded = template_validator.load_all(TEMPLATE_DIR)
        assert set(loaded) == {
            "cover_drive",
            "pull_shot",
            "fast_arm_bowling",
            "spin_bowling",
        }

    def test_good_metric_passes(self):
        report = template_validator.validate_template(
            _template({"batting.lead_knee_angle_imp": GOOD_METRIC}), name="cover_drive"
        )
        assert report.ok, report.errors

    def test_missing_source_is_rejected(self):
        bad = {**GOOD_METRIC, "source": ""}
        report = template_validator.validate_template(
            _template({"batting.lead_knee_angle_imp": bad}), name="cover_drive"
        )
        assert not report.ok
        assert any("source" in e for e in report.errors)

    def test_angle_without_convention_is_rejected(self):
        bad = {**GOOD_METRIC, "convention": None}
        report = template_validator.validate_template(
            _template({"batting.lead_knee_angle_imp": bad}), name="cover_drive"
        )
        assert not report.ok
        assert any("convention" in e for e in report.errors)

    def test_omitted_convention_is_rejected(self):
        bad = {k: v for k, v in GOOD_METRIC.items() if k != "convention"}
        report = template_validator.validate_template(
            _template({"batting.lead_knee_angle_imp": bad}), name="cover_drive"
        )
        assert not report.ok
        assert any("mandatory" in e for e in report.errors)

    def test_tier_c_metric_in_template_is_rejected(self):
        bad = {
            **GOOD_METRIC,
            "tier": "C",
            "event": "DS->IMP",
            "units": "ratio",
            "convention": None,
            "discriminates_skill": False,
        }
        report = template_validator.validate_template(
            _template({"batting.bat_speed_max": bad}), name="cover_drive"
        )
        assert not report.ok
        assert any("Tier C" in e for e in report.errors)

    def test_tier_mismatch_with_registry_is_rejected(self):
        bad = {**GOOD_METRIC, "tier": "B"}
        report = template_validator.validate_template(
            _template({"batting.lead_knee_angle_imp": bad}), name="cover_drive"
        )
        assert not report.ok
        assert any("declared tier" in e for e in report.errors)

    def test_unknown_metric_key_is_rejected(self):
        report = template_validator.validate_template(
            _template({"batting.invented_metric": GOOD_METRIC}), name="cover_drive"
        )
        assert not report.ok

    def test_event_not_in_phase_events_is_rejected(self):
        bad = {**GOOD_METRIC, "event": "NONSENSE"}
        report = template_validator.validate_template(
            _template({"batting.lead_knee_angle_imp": bad}), name="cover_drive"
        )
        assert not report.ok
        assert any("phase_events" in e for e in report.errors)

    def test_todo_with_a_range_is_rejected(self):
        """Guards against 'I'll just put a plausible number here for now'."""
        bad = {**GOOD_METRIC, "band_derivation": "todo", "source": ""}
        report = template_validator.validate_template(
            _template({"batting.lead_knee_angle_imp": bad}), name="cover_drive"
        )
        assert not report.ok
        assert any("must be null" in e for e in report.errors)

    def test_derived_conversion_requires_a_derivation_note(self):
        bad = {**GOOD_METRIC, "band_derivation": "derived_conversion"}
        report = template_validator.validate_template(
            _template({"batting.lead_knee_angle_imp": bad}), name="cover_drive"
        )
        assert not report.ok
        assert any("derivation" in e for e in report.errors)

    def test_inverted_band_is_rejected(self):
        bad = {**GOOD_METRIC, "expected_low": 175.0, "expected_high": 145.0}
        report = template_validator.validate_template(
            _template({"batting.lead_knee_angle_imp": bad}), name="cover_drive"
        )
        assert not report.ok

    def test_wrong_discipline_is_rejected(self):
        report = template_validator.validate_template(
            _template({"bowling.front_knee_angle_ffc": {**GOOD_METRIC, "event": "IMP"}}),
            name="cover_drive",
        )
        assert not report.ok
        assert any("discipline" in e.lower() for e in report.errors)

    def test_non_angle_with_convention_is_rejected(self):
        bad = {
            **GOOD_METRIC,
            "event": "SS->SE",
            "units": "s",
            "convention": "straight_180",
            "expected_low": 0.19,
            "expected_high": 0.29,
        }
        report = template_validator.validate_template(
            _template({"batting.stride_duration_ss_se": bad}), name="cover_drive"
        )
        assert not report.ok
        assert any("convention null" in e for e in report.errors)

    def test_load_template_raises_on_invalid(self, tmp_path):
        bad = _template({"batting.lead_knee_angle_imp": {**GOOD_METRIC, "source": ""}})
        path = tmp_path / "cover_drive.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(template_validator.TemplateError):
            template_validator.load_template(path)


# ---------------------------------------------------------------------------
# Cross-cutting invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_no_template_scores_a_suppressed_metric(self):
        for name, data in template_validator.load_all(TEMPLATE_DIR).items():
            for key, body in data["metrics"].items():
                if body["band_derivation"] == "todo":
                    continue
                assert metric_tiers.is_scorable(key), (
                    f"{name} gives a scoreable band to non-scorable metric {key}"
                )

    def test_every_scorable_metric_has_a_baseline_sem_rule(self):
        """A scorable metric with no SEM rule would default to optimistic uncertainty.

        This is the invariant that matters: anything that can reach a score must have a
        documented error model.
        """
        for spec in metric_tiers.REGISTRY.values():
            if not metric_tiers.is_scorable(spec.key) or spec.is_categorical:
                continue
            assert uncertainty.base_sem_for(spec, fps=120) > 0

    def test_categorical_metrics_refuse_to_produce_a_sem(self):
        spec = metric_tiers.get("batting.backlift_direction")
        assert spec.is_categorical
        with pytest.raises(uncertainty.CategoricalMetricError):
            uncertainty.base_sem_for(spec, fps=120)

    def test_bat_metrics_refuse_to_produce_a_sem(self):
        """No error model, because there is no measurement - not merely a wide one."""
        with pytest.raises(uncertainty.NotMeasurableError):
            uncertainty.base_sem_for(metric_tiers.get("batting.bat_speed_max"), fps=240)

    def test_safety_critical_metrics_are_all_tier_c(self):
        for key in metric_tiers.SAFETY_CRITICAL:
            assert metric_tiers.get(key).tier is Tier.C
