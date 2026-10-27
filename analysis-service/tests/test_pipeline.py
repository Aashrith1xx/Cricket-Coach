"""Pipeline tests: pose conditioning, phase detection, metrics, and a full worker run.

These use synthetic fixtures with known ground-truth events. Synthetic motion is clean and
noise-free, so passing here proves the *algorithms* are correct - not that they work on real
footage. That is what ``validation/`` is for, and the two check different things.
"""

from __future__ import annotations

import numpy as np
import pytest

import capture_gate
import metric_tiers
import normalisation
import phase_detection
import render_output
import retention
import scoring
import smoothing
import worker
from capture_gate import CaptureMetadata, RejectionCode
from metrics import batting as batting_metrics
from metrics import bowling as bowling_metrics
from pose_extraction import PoseSequence, to_lead_rear
from style_resolution import Style
from tests import fixtures

#: Tolerance for event detection, in frames. At 240 fps this is 25 ms, matching the
#: ~24-25 ms video event-detection error reported in the literature. Tightening it beyond
#: that would be testing the fixture rather than the detector.
EVENT_TOLERANCE_FRAMES = 6


def _analyse(clip):
    scale = normalisation.estimate_scale(clip.seq)
    return scale


# ---------------------------------------------------------------------------
# Scale estimation
# ---------------------------------------------------------------------------


class TestScaleEstimation:
    def test_recovers_fixture_stature(self):
        clip = fixtures.batting_clip()
        scale = normalisation.estimate_scale(clip.seq)
        assert scale.stature_px == pytest.approx(clip.stature_px, rel=0.05)

    def test_uses_multiple_segments(self):
        clip = fixtures.batting_clip()
        scale = normalisation.estimate_scale(clip.seq)
        assert scale.n_segments_used >= 4
        assert 0.0 <= scale.confidence <= 1.0

    def test_scale_is_distance_invariant(self):
        """Normalised metrics must not change when the camera moves closer."""
        clip = fixtures.batting_clip()
        zoomed = PoseSequence(
            xy=clip.seq.xy * 1.7, confidence=clip.seq.confidence, fps=clip.seq.fps
        )
        a = normalisation.estimate_scale(clip.seq)
        b = normalisation.estimate_scale(zoomed)
        assert b.stature_px == pytest.approx(a.stature_px * 1.7, rel=0.02)

    def test_raises_when_no_segments_visible(self):
        clip = fixtures.batting_clip()
        blank = PoseSequence(
            xy=np.full_like(clip.seq.xy, np.nan),
            confidence=np.zeros_like(clip.seq.confidence),
            fps=clip.seq.fps,
        )
        with pytest.raises(normalisation.ScaleError):
            normalisation.estimate_scale(blank)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


class TestSmoothing:
    def test_low_confidence_samples_become_nan(self):
        xy = np.ones((10, 2), dtype=float)
        conf = np.ones(10)
        conf[3:5] = 0.1
        masked = smoothing.mask_low_confidence(xy, conf)
        assert np.isnan(masked[3:5]).all()
        assert not np.isnan(masked[0]).any()

    def test_short_gaps_are_bridged(self):
        xy = np.column_stack([np.arange(60.0), np.arange(60.0)])
        xy[30:32] = np.nan
        filled, longest = smoothing.interpolate_short_gaps(xy, fps=240.0)
        assert not np.isnan(filled[30:32]).any()
        assert longest == 2

    def test_long_gaps_are_left_unrecoverable(self):
        """A joint occluded for a long stretch was genuinely unobserved. Inventing a
        trajectory would manufacture the measurement the uncertainty model exists to
        qualify."""
        xy = np.column_stack([np.arange(120.0), np.arange(120.0)])
        xy[30:100] = np.nan
        filled, longest = smoothing.interpolate_short_gaps(xy, fps=240.0)
        assert np.isnan(filled[30:100]).any()
        assert longest == 70

    def test_leading_gaps_are_never_extrapolated(self):
        xy = np.column_stack([np.arange(60.0), np.arange(60.0)])
        xy[:5] = np.nan
        filled, _ = smoothing.interpolate_short_gaps(xy, fps=240.0)
        assert np.isnan(filled[:5]).all()

    def test_condition_reports_provenance(self):
        clip = fixtures.occluded_clip()
        traj = smoothing.condition(
            clip.seq.keypoint("left_ankle"),
            clip.seq.keypoint_confidence("left_ankle"),
            clip.seq.fps,
        )
        assert traj.valid_fraction < 1.0
        assert traj.max_gap_frames > 0
        assert traj.has_unrecoverable_gaps


# ---------------------------------------------------------------------------
# Phase detection
# ---------------------------------------------------------------------------


class TestBattingPhaseDetection:
    def _timeline(self, clip):
        scale = normalisation.estimate_scale(clip.seq)
        return phase_detection.detect_batting(
            clip.seq, handedness="right", scale_px=scale.stature_px
        )

    @pytest.mark.parametrize("event", ["STANCE", "BC", "SS", "DS", "SE", "IMP"])
    def test_events_within_tolerance(self, event):
        clip = fixtures.batting_clip()
        timeline = self._timeline(clip)
        detected = timeline.frame_of(event)
        assert detected is not None, f"{event} not detected"
        assert abs(detected - clip.truth[event]) <= EVENT_TOLERANCE_FRAMES

    def test_event_ordering_is_physical(self):
        clip = fixtures.batting_clip()
        timeline = self._timeline(clip)
        order = ["STANCE", "BC", "SS", "DS", "IMP"]
        frames = [timeline.frame_of(e) for e in order]
        assert all(f is not None for f in frames)
        assert frames == sorted(frames), f"events out of order: {dict(zip(order, frames))}"

    def test_trigger_detected_when_present(self):
        clip = fixtures.batting_clip(with_trigger=True)
        timeline = self._timeline(clip)
        assert timeline.frame_of("TRIGGER") is not None

    def test_trigger_absent_when_not_performed(self):
        """Absence must not be invented. Trigger prevalence ranges 47-76% by format, so a
        false positive would report a style the player does not have."""
        clip = fixtures.batting_clip(with_trigger=False)
        timeline = self._timeline(clip)
        assert timeline.frame_of("TRIGGER") is None

    def test_trigger_is_not_mistaken_for_the_stride(self):
        with_trigger = self._timeline(fixtures.batting_clip(with_trigger=True))
        without = self._timeline(fixtures.batting_clip(with_trigger=False))
        assert abs(with_trigger.frame_of("SS") - without.frame_of("SS")) <= 3

    def test_proxied_events_carry_lower_confidence(self):
        """IMP and DS are inferred from wrist kinematics, not observed. They must not look
        as certain as a foot contact."""
        timeline = self._timeline(fixtures.batting_clip())
        foot = timeline.events["SE"].confidence
        proxy = timeline.events["IMP"].confidence
        assert proxy < foot
        assert proxy <= phase_detection.PROXY_EVENT_MAX_CONFIDENCE

    def test_survives_keypoint_noise(self):
        clip = fixtures.batting_clip(noise_px=1.5, seed=3)
        timeline = self._timeline(clip)
        for event in ("SS", "SE", "IMP"):
            detected = timeline.frame_of(event)
            assert detected is not None
            assert abs(detected - clip.truth[event]) <= EVENT_TOLERANCE_FRAMES * 2


class TestBowlingPhaseDetection:
    def _timeline(self, clip):
        scale = normalisation.estimate_scale(clip.seq)
        return phase_detection.detect_bowling(
            clip.seq, handedness="right", scale_px=scale.stature_px
        )

    @pytest.mark.parametrize("event", ["RUNUP", "BFC", "FFC", "BR"])
    def test_events_within_tolerance(self, event):
        clip = fixtures.bowling_clip()
        timeline = self._timeline(clip)
        detected = timeline.frame_of(event)
        assert detected is not None, f"{event} not detected"
        assert abs(detected - clip.truth[event]) <= EVENT_TOLERANCE_FRAMES

    def test_back_foot_lands_before_front_foot(self):
        timeline = self._timeline(fixtures.bowling_clip())
        assert timeline.frame_of("BFC") < timeline.frame_of("FFC")

    def test_contact_is_not_reported_before_the_flight(self):
        """Regression: searching for a foot-contact maximum directly ties with the
        ground-level baseline before the foot lifts, and argmax resolves the tie by
        returning the earlier frame - reporting contact before the bowler left the ground."""
        clip = fixtures.bowling_clip()
        timeline = self._timeline(clip)
        assert timeline.frame_of("BFC") > timeline.frame_of("PRE_DELIVERY_STRIDE")
        assert timeline.frame_of("BFC") > 60

    def test_release_carries_capped_confidence(self):
        timeline = self._timeline(fixtures.bowling_clip())
        assert timeline.events["BR"].confidence <= phase_detection.PROXY_EVENT_MAX_CONFIDENCE


class TestLowFrameRateSuppression:
    def test_sub_frame_events_are_dropped_below_the_floor(self):
        """At 30 fps, impact falls between frames. Suppressing it is correct; estimating it
        would be indistinguishable from measuring it."""
        clip = fixtures.batting_clip(fps=30.0)
        scale = normalisation.estimate_scale(clip.seq)
        timeline = phase_detection.detect_batting(
            clip.seq, handedness="right", scale_px=scale.stature_px
        )
        assert timeline.frame_of("IMP") is None
        assert timeline.frame_of("DS") is None
        assert timeline.frame_of("SS") is not None, "foot events survive low frame rates"

    def test_metrics_requiring_impact_are_not_offered(self):
        allowed = {s.key for s in metric_tiers.applicable(metric_tiers.Discipline.BATTING, fps=30)}
        assert "batting.lead_knee_angle_imp" not in allowed
        assert "batting.stride_duration_ss_se" in allowed


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestBattingMetrics:
    def _setup(self, clip=None):
        clip = clip or fixtures.batting_clip()
        scale = normalisation.estimate_scale(clip.seq)
        timeline = phase_detection.detect_batting(
            clip.seq, handedness="right", scale_px=scale.stature_px
        )
        return clip, scale, timeline

    def test_computes_a_full_set(self):
        clip, scale, timeline = self._setup()
        allowed = {s.key for s in metric_tiers.applicable(metric_tiers.Discipline.BATTING, fps=240)}
        results = batting_metrics.compute_all(
            clip.seq, timeline, scale, handedness="right", allowed=allowed
        )
        assert len(results) >= 10
        assert all(np.isfinite(m.value) for m in results)
        assert all(m.sem > 0 for m in results)

    def test_no_tier_c_metric_is_ever_produced(self):
        clip, scale, timeline = self._setup()
        results = batting_metrics.compute_all(clip.seq, timeline, scale, handedness="right")
        for m in results:
            assert metric_tiers.get(m.key).tier is not metric_tiers.Tier.C

    def test_stride_duration_matches_fixture(self):
        clip, _, timeline = self._setup()
        m = batting_metrics.stride_duration(timeline)
        expected = (clip.truth["SE"] - clip.truth["SS"]) / clip.seq.fps
        assert m is not None
        assert m.value == pytest.approx(expected, abs=0.03)

    def test_percent_stature_sem_is_in_percent(self):
        """Regression: a 21.8% measurement once carried a +/-0.025 error bar, claiming a
        precision three orders of magnitude better than reality."""
        clip, scale, timeline = self._setup()
        m = batting_metrics.base_length(clip.seq, timeline, scale)
        assert m is not None
        assert m.units == "pct_stature"
        assert m.sem > 1.0, f"SEM {m.sem} is implausibly tight for a percentage"

    def test_missing_events_yield_none_not_a_fallback(self):
        clip, scale, _ = self._setup()
        empty = phase_detection.PhaseTimeline(events={}, fps=clip.seq.fps)
        assert batting_metrics.stride_duration(empty) is None
        assert batting_metrics.lead_knee_angle_at_impact(clip.seq, empty, handedness="right") is None

    def test_knee_angle_is_in_straight_180_convention(self):
        clip, scale, timeline = self._setup()
        m = batting_metrics.lead_knee_angle_at_impact(clip.seq, timeline, handedness="right")
        assert m is not None
        assert 90.0 < m.value <= 180.0, "a standing leg should read near-straight"

    def test_poor_tracking_widens_the_interval(self):
        good = fixtures.batting_clip(confidence=0.95)
        poor = fixtures.batting_clip(confidence=0.45)
        widths = []
        for clip in (good, poor):
            _, scale, timeline = self._setup(clip)
            m = batting_metrics.stride_duration(timeline)
            widths.append(m.ci_width)
        assert widths[1] > widths[0]


class TestBowlingMetrics:
    def _setup(self):
        clip = fixtures.bowling_clip()
        scale = normalisation.estimate_scale(clip.seq)
        timeline = phase_detection.detect_bowling(
            clip.seq, handedness="right", scale_px=scale.stature_px
        )
        return clip, scale, timeline

    def test_computes_a_full_set(self):
        clip, scale, timeline = self._setup()
        allowed = {s.key for s in metric_tiers.applicable(metric_tiers.Discipline.BOWLING, fps=240)}
        results = bowling_metrics.compute_all(
            clip.seq, timeline, scale, handedness="right", allowed=allowed
        )
        assert len(results) >= 7

    def test_no_counter_rotation_or_elbow_metric_exists(self):
        """These are refused for safety, not merely unimplemented. The risk thresholds are
        smaller than the measurement error."""
        assert not hasattr(bowling_metrics, "shoulder_counter_rotation")
        assert not hasattr(bowling_metrics, "elbow_extension")
        assert not hasattr(bowling_metrics, "action_type")

    def test_lateral_flexion_needs_the_front_on_camera(self):
        """Frontal-plane motion is foreshortened on the side-on view the protocol specifies."""
        side_on = {s.key for s in metric_tiers.applicable(metric_tiers.Discipline.BOWLING, fps=240)}
        assert "bowling.trunk_lateral_flexion_br" not in side_on
        assert "bowling.trunk_forward_flexion_br" in side_on

    def test_delivery_interval_matches_fixture(self):
        clip, _, timeline = self._setup()
        m = bowling_metrics.back_to_front_foot_interval(timeline)
        expected = (clip.truth["FFC"] - clip.truth["BFC"]) / clip.seq.fps
        assert m is not None
        assert m.value == pytest.approx(expected, abs=0.03)


# ---------------------------------------------------------------------------
# Capture gate
# ---------------------------------------------------------------------------


class TestCaptureGate:
    def _meta(self, **kw):
        base = dict(fps=240.0, width=1080, height=1920, duration_s=2.0, declared_view="side_on")
        base.update(kw)
        return CaptureMetadata(**base)

    def test_accepts_good_metadata(self):
        assert capture_gate.check_metadata(self._meta()).passed

    def test_rejects_unusably_low_frame_rate(self):
        result = capture_gate.check_metadata(self._meta(fps=24.0))
        assert not result.passed
        assert result.rejections[0].code is RejectionCode.FPS_TOO_LOW

    def test_warns_but_accepts_60fps(self):
        """60 fps still supports timing and posture work; only impact-anchored metrics go."""
        result = capture_gate.check_metadata(self._meta(fps=60.0))
        assert result.passed
        assert any("120" in w for w in result.warnings)

    def test_rejects_clips_that_are_too_short_or_long(self):
        assert not capture_gate.check_metadata(self._meta(duration_s=0.4)).passed
        assert not capture_gate.check_metadata(self._meta(duration_s=90.0)).passed

    def test_every_rejection_code_has_a_remedy(self):
        """A bare rejection teaches players to stop using the app."""
        for code in RejectionCode:
            assert capture_gate.REMEDIES[code].strip()

    def test_pose_quality_rejects_poor_tracking(self):
        clip = fixtures.batting_clip(confidence=0.2)
        result = capture_gate.check_pose_quality(
            clip.seq, required_view="side_on", discipline="batting"
        )
        assert not result.passed
        assert any(r.code is RejectionCode.POOR_TRACKING for r in result.rejections)

    def test_pose_quality_rejects_heavy_occlusion(self):
        clip = fixtures.occluded_clip(start=20, end=200)
        result = capture_gate.check_pose_quality(
            clip.seq, required_view="side_on", discipline="batting"
        )
        assert not result.passed
        assert any(r.code is RejectionCode.OCCLUDED_ACTION for r in result.rejections)

    def test_side_on_fixture_passes_the_view_check(self):
        clip = fixtures.batting_clip()
        result = capture_gate.check_pose_quality(
            clip.seq, required_view="side_on", discipline="batting"
        )
        assert result.passed, [r.detail for r in result.rejections]

    def test_view_angle_is_measured_at_setup_not_averaged_over_the_shot(self):
        """Regression: a batter rotates ~70 degrees through a shot, so a whole-clip median
        blended camera geometry with the player's own rotation. On a real cover drive that
        reported 31 degrees - the average of 90 at stance and 20 through the swing - and
        rejected the clip while naming a deviation true at no instant."""
        from pose_extraction import KEYPOINT_INDEX

        clip = fixtures.batting_clip()
        xy = clip.seq.xy.copy()
        setup = int(clip.seq.n_frames * capture_gate.SETUP_WINDOW_FRACTION)

        # Square-on at setup: shoulders nearly coincident, as a side-on camera sees them.
        xy[:setup, KEYPOINT_INDEX["left_shoulder"], 0] = 250.0
        xy[:setup, KEYPOINT_INDEX["right_shoulder"], 0] = 156.0
        # Then the player rotates through the shot, foreshortening the shoulder line.
        xy[setup:, KEYPOINT_INDEX["left_shoulder"], 0] = 205.0
        xy[setup:, KEYPOINT_INDEX["right_shoulder"], 0] = 200.0

        seq = PoseSequence(xy=xy, confidence=clip.seq.confidence, fps=clip.seq.fps)
        deviation = capture_gate.estimate_view_deviation_deg(seq, "side_on")

        assert deviation is not None
        assert deviation < capture_gate.MAX_VIEW_ANGLE_DEVIATION_DEG, (
            f"rotation during the shot leaked into the camera-angle estimate ({deviation:.0f} deg)"
        )

    def test_stance_survives_keypoint_jitter_at_high_frame_rate(self):
        """Regression: stillness was judged by instantaneous velocity, but per-frame jitter
        is roughly constant while the velocity it implies scales with frame rate - 1px of
        jitter at 240fps reads as 240px/s. A real nets clip showed a median hip 'speed' of
        0.31 stature/s while standing still, against a 0.12 threshold, so no stance was ever
        found and every batting clip was refused. Stillness is now displacement over a
        window, which jitter averages out of."""
        clip = fixtures.batting_clip(fps=240.0, noise_px=1.2, seed=11)
        scale = normalisation.estimate_scale(clip.seq)
        timeline = phase_detection.detect_batting(
            clip.seq, handedness="right", scale_px=scale.stature_px
        )
        assert timeline.frame_of("STANCE") is not None, (
            "stance lost to keypoint jitter - the velocity-threshold bug is back"
        )

    def test_stillness_threshold_is_frame_rate_independent(self):
        """The same physical stillness must read as still at any capture rate."""
        for fps in (120.0, 240.0):
            clip = fixtures.batting_clip(fps=fps, noise_px=1.0, seed=5)
            scale = normalisation.estimate_scale(clip.seq)
            timeline = phase_detection.detect_batting(
                clip.seq, handedness="right", scale_px=scale.stature_px
            )
            assert timeline.frame_of("STANCE") is not None, f"no stance at {fps} fps"

    def test_subject_too_small_is_rejected_not_merely_warned(self):
        """Keypoint error scales inversely with subject pixels, so a distant figure yields
        confident-looking joints with large real error."""
        clip = fixtures.batting_clip()
        result = capture_gate.check_pose_quality(
            clip.seq, required_view="side_on", discipline="batting", frame_height=4000
        )
        assert not result.passed
        assert any(r.code is RejectionCode.SUBJECT_TOO_SMALL for r in result.rejections)

    def test_subject_size_check_skipped_without_frame_height(self):
        clip = fixtures.batting_clip()
        result = capture_gate.check_pose_quality(
            clip.seq, required_view="side_on", discipline="batting"
        )
        assert all(r.code is not RejectionCode.SUBJECT_TOO_SMALL for r in result.rejections)

    def test_front_on_is_a_valid_view_not_a_failure(self):
        """Regression: front-on footage was refused as a failed side-on. It is a different
        view with a different, smaller, valid metric set - and nets footage is very often
        shot from in front."""
        from pose_extraction import KEYPOINT_INDEX

        clip = fixtures.batting_clip()
        xy = clip.seq.xy.copy()
        # Narrow shoulder line on a BATTER means the camera is down the pitch, not to the side.
        xy[:, KEYPOINT_INDEX["left_shoulder"], 0] = 203.0
        xy[:, KEYPOINT_INDEX["right_shoulder"], 0] = 200.0
        front_on = PoseSequence(xy=xy, confidence=clip.seq.confidence, fps=clip.seq.fps)

        view, deviation = capture_gate.classify_view(front_on)
        assert view == "front_on", f"got {view} at {deviation}"

        result = capture_gate.check_pose_quality(
            front_on, required_view="any", discipline="batting"
        )
        assert result.passed, [r.detail for r in result.rejections]
        assert result.detected_view == "front_on"

    def test_oblique_camera_is_still_refused(self):
        """Between the two canonical views, neither plane is measured cleanly."""
        from pose_extraction import KEYPOINT_INDEX

        clip = fixtures.batting_clip()
        xy = clip.seq.xy.copy()
        # Shoulder line about half its full width: roughly 45 degrees.
        xy[:, KEYPOINT_INDEX["left_shoulder"], 0] = 226.0
        xy[:, KEYPOINT_INDEX["right_shoulder"], 0] = 192.0
        oblique = PoseSequence(xy=xy, confidence=clip.seq.confidence, fps=clip.seq.fps)

        view, _ = capture_gate.classify_view(oblique)
        assert view == "oblique"
        result = capture_gate.check_pose_quality(
            oblique, required_view="any", discipline="batting"
        )
        assert not result.passed

    def test_temporal_metrics_survive_a_front_on_camera(self):
        """Timing is view-independent, and carries the most score weight."""
        keys = {
            s.key
            for s in metric_tiers.applicable(
                metric_tiers.Discipline.BATTING, fps=240, view="front_on"
            )
        }
        assert "batting.stride_duration_ss_se" in keys
        assert "batting.tempo_bc_ds" in keys
        assert "batting.lead_knee_angle_imp" not in keys, "sagittal angle needs side-on"

    def test_mesh_is_detected_and_relaxes_the_tracking_floor(self):
        """Nets are where people practise. Mesh genuinely lowers tracking, so the clip is
        accepted with wider error bars rather than refused."""
        import numpy as _np

        plain = [_np.full((300, 300, 3), 90, _np.uint8) for _ in range(10)]
        assert not capture_gate.detect_mesh(plain)[0]

        netted = []
        for _ in range(10):
            frame = _np.full((300, 300, 3), 90, _np.uint8)
            frame[::6, :] = 20
            frame[:, ::6] = 20
            netted.append(frame)
        is_mesh, density = capture_gate.detect_mesh(netted)
        assert is_mesh, f"net pattern not detected (density {density:.3f})"

        clip = fixtures.batting_clip(confidence=0.47)
        strict = capture_gate.check_pose_quality(
            clip.seq, required_view="any", discipline="batting", has_mesh=False
        )
        relaxed = capture_gate.check_pose_quality(
            clip.seq, required_view="any", discipline="batting", has_mesh=True
        )
        assert not strict.passed
        assert relaxed.passed
        assert any("net or fence" in w for w in relaxed.warnings)

    def test_front_on_view_is_rejected_when_side_on_required(self):
        """A wide shoulder line means the camera is in front, not to the side."""
        clip = fixtures.batting_clip()
        xy = clip.seq.xy.copy()
        from pose_extraction import KEYPOINT_INDEX

        xy[:, KEYPOINT_INDEX["left_shoulder"], 0] = 203.0
        xy[:, KEYPOINT_INDEX["right_shoulder"], 0] = 200.0
        wide = PoseSequence(xy=xy, confidence=clip.seq.confidence, fps=clip.seq.fps)

        result = capture_gate.check_pose_quality(
            wide, required_view="side_on", discipline="batting"
        )
        assert not result.passed
        assert any(r.code is RejectionCode.WRONG_VIEW_ANGLE for r in result.rejections)


# ---------------------------------------------------------------------------
# Render payload
# ---------------------------------------------------------------------------


class TestRenderOutput:
    def _payload(self):
        clip = fixtures.batting_clip()
        scale = normalisation.estimate_scale(clip.seq)
        timeline = phase_detection.detect_batting(
            clip.seq, handedness="right", scale_px=scale.stature_px
        )
        measurements = batting_metrics.compute_all(
            clip.seq, timeline, scale, handedness="right"
        )
        scores = scoring.score_all(measurements, {})
        return render_output.build(
            clip_id="c1", seq=clip.seq, timeline=timeline, scores=scores
        )

    def test_payload_is_serialisable(self):
        import json

        payload = self._payload()
        parsed = json.loads(payload.to_json())
        assert parsed["schema_version"] == render_output.RENDER_SCHEMA_VERSION
        assert len(parsed["frames"]) == payload.n_frames

    def test_coordinates_are_normalised(self):
        payload = self._payload()
        values = [v for f in payload.frames for v in f.xy if v is not None]
        assert all(-0.5 <= v <= 1.5 for v in values)

    def test_unobserved_joints_serialise_as_null(self):
        """Zero would place the joint in the frame corner and draw a limb to it."""
        clip = fixtures.occluded_clip(start=100, end=140)
        scale = normalisation.estimate_scale(clip.seq)
        timeline = phase_detection.detect_batting(
            clip.seq, handedness="right", scale_px=scale.stature_px
        )
        payload = render_output.build(
            clip_id="c1", seq=clip.seq, timeline=timeline, scores=[]
        )
        assert any(v is None for f in payload.frames for v in f.xy)

    def test_only_detected_events_are_emitted(self):
        payload = self._payload()
        assert all(e.frame is not None for e in payload.events)

    def test_tier_b_annotations_carry_no_precise_value(self):
        payload = self._payload()
        for annotation in payload.metrics:
            if annotation.tier == "B":
                assert annotation.value is None


# ---------------------------------------------------------------------------
# Retention policy
# ---------------------------------------------------------------------------


class TestRetention:
    def test_consent_governs_target_state(self):
        assert retention.target_state(True) is retention.RetentionState.PROXY_ONLY
        assert retention.target_state(False) is retention.RetentionState.KEYPOINTS_ONLY

    def test_nothing_is_deleted_without_storage(self):
        outcome = retention.apply(clip_id="c1", video_path="x.mp4", consent_for_training=True)
        assert outcome.bytes_freed == 0
        assert "Nothing was deleted" in outcome.note

    def test_original_is_kept_when_keypoints_are_unconfirmed(self):
        """Deleting video before the keypoints are safe would destroy the only durable
        artefact."""

        class Storage(retention.RetentionStorage):
            def keypoints_persisted(self, clip_id):
                return False

        outcome = retention.apply(
            clip_id="c1", video_path="x.mp4", consent_for_training=False, storage=Storage()
        )
        assert outcome.state is retention.RetentionState.ORIGINAL
        assert outcome.bytes_freed == 0

    def test_storage_projection_shows_the_saving(self):
        projection = retention.estimate_storage_bytes(n_clips=1000)
        assert projection["total_mb"] < projection["naive_total_mb"] / 5


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def _job(clip, **kw):
    params = dict(
        clip_id="c1",
        session_id="s1",
        user_id="u1",
        form_type="cover_drive",
        video_path="",
        metadata=CaptureMetadata(
            fps=clip.seq.fps, width=1080, height=1920, duration_s=2.0, declared_view="side_on"
        ),
        declared_style=Style.BALANCED,
        handedness="right",
        pose_sequence=clip.seq,
        skip_frame_gate=True,
    )
    params.update(kw)
    return worker.JobRequest(**params)


class TestEndToEnd:
    def test_batting_clip_completes(self):
        result = worker.process(_job(fixtures.batting_clip()))
        assert result.status is worker.JobStatus.DONE, result.error
        assert result.error is None

    def test_composite_is_none_without_sourced_bands(self):
        """The templates ship mostly unscored on purpose. The correct behaviour is to
        withhold an overall grade, not to invent one from two metrics."""
        result = worker.process(_job(fixtures.batting_clip()))
        assert result.composite["composite"] is None
        assert result.composite["suppressed_reason"]

    def test_feedback_is_generated_and_honest(self):
        result = worker.process(_job(fixtures.batting_clip()))
        assert "Not enough" in result.feedback["headline"]

    def test_no_tier_c_metric_reaches_the_result(self):
        result = worker.process(_job(fixtures.batting_clip()))
        for score in result.composite["scores"]:
            assert metric_tiers.get(score["key"]).tier is not metric_tiers.Tier.C

    def test_phase_events_are_reported(self):
        result = worker.process(_job(fixtures.batting_clip()))
        assert result.phase_events["events"]["SS"]["detected"]

    def test_bowling_clip_completes(self):
        clip = fixtures.bowling_clip()
        result = worker.process(_job(clip, form_type="fast_arm_bowling"))
        assert result.status is worker.JobStatus.DONE, result.error

    def test_unknown_form_type_fails_cleanly(self):
        result = worker.process(_job(fixtures.batting_clip(), form_type="nonsense"))
        assert result.status is worker.JobStatus.FAILED
        assert "template" in result.error.lower()

    def test_untrackable_clip_is_rejected_with_a_remedy(self):
        result = worker.process(_job(fixtures.batting_clip(confidence=0.15)))
        assert result.status is worker.JobStatus.REJECTED
        assert result.gate.rejections
        assert all(r.remedy for r in result.gate.rejections)

    def test_template_version_is_recorded(self):
        """Historical scores must stay interpretable after a template changes."""
        result = worker.process(_job(fixtures.batting_clip()))
        assert result.template_id == "cover_drive"
        assert result.template_version


class TestMeshInFrontVsBehind:
    """Net behind the player is harmless; only mesh the camera shoots through matters.

    Regression: edge density measured across the whole frame cannot tell them apart, so a
    clip filmed from INSIDE the net - net visible all around, nothing between camera and
    player - was flagged as shot through mesh.
    """

    def _scene(self, *, net_in_front: bool):
        import numpy as _np

        frames, boxes = [], []
        for t in range(12):
            frame = _np.full((400, 400, 3), 150, _np.uint8)
            # Background netting, everywhere.
            frame[::8, :] = 30
            frame[:, ::8] = 30

            x = 150 + t  # player drifts so the temporal median sees through them
            body = (slice(120, 340), slice(x, x + 90))
            frame[body] = 235          # solid body occludes the net behind it

            if net_in_front:           # cords drawn back OVER the body
                frame[120:340:8, x : x + 90] = 30
                frame[120:340, x : x + 90 : 8] = 30

            frames.append(frame)
            boxes.append(_np.array([x, 120, x + 90, 340], dtype=float))
        return frames, _np.stack(boxes)

    def test_mesh_in_front_is_detected(self):
        frames, boxes = self._scene(net_in_front=True)
        is_mesh, density = capture_gate.detect_mesh(frames, boxes=boxes)
        assert is_mesh, f"cords over the body should read as in front (density {density:.3f})"

    def test_net_behind_the_player_is_not_flagged(self):
        frames, boxes = self._scene(net_in_front=False)
        is_mesh, _ = capture_gate.detect_mesh(frames, boxes=boxes)
        assert not is_mesh, "background netting is harmless and must not be flagged"

    def test_falls_back_to_whole_frame_without_boxes(self):
        frames, _ = self._scene(net_in_front=True)
        is_mesh, _ = capture_gate.detect_mesh(frames)
        assert is_mesh


class TestMechanicalRules:
    """Feedback that needs no reference population - only direction and ordering."""

    def _clip(self):
        clip = fixtures.batting_clip()
        scale = normalisation.estimate_scale(clip.seq)
        timeline = phase_detection.detect_batting(
            clip.seq, handedness="right", scale_px=scale.stature_px
        )
        measurements = batting_metrics.compute_all(
            clip.seq, timeline, scale, handedness="right"
        )
        return measurements, timeline

    def test_produces_findings_without_any_reference_band(self):
        """The whole point: this works while the scored ranges are still empty."""
        import mechanics

        measurements, timeline = self._clip()
        findings = mechanics.analyse(measurements, timeline, "batting")
        assert findings

    def test_ordering_rules_fire_on_a_well_formed_shot(self):
        import mechanics

        _, timeline = self._clip()
        found = mechanics.check_ordering(timeline, "batting")
        assert found
        assert all(f.kind is mechanics.RuleKind.ORDERING for f in found)
        assert all(f.severity is mechanics.Severity.GOOD for f in found), (
            "a clean synthetic drive should not trip a sequence fault"
        )

    def test_direction_rule_ignores_movement_inside_the_error_bar(self):
        """A value smaller than its own uncertainty is noise, not technique."""
        import mechanics
        from uncertainty import Measurement

        tiny = Measurement(
            "batting.head_vertical_displacement_ds_imp",
            value=0.001, sem=0.02, confidence=0.9, units="frac_stature",
        )
        assert mechanics.check_directions([tiny]) == []

    def test_direction_rule_fires_on_a_clear_fault(self):
        import mechanics
        from uncertainty import Measurement

        lifted = Measurement(
            "batting.head_vertical_displacement_ds_imp",
            value=0.09, sem=0.01, confidence=0.9, units="frac_stature",
        )
        found = mechanics.check_directions([lifted])
        assert len(found) == 1
        assert found[0].severity is mechanics.Severity.FAULT
        assert "lifted" in found[0].text

    def test_direction_rule_credits_the_good_direction(self):
        import mechanics
        from uncertainty import Measurement

        steady = Measurement(
            "batting.head_vertical_displacement_ds_imp",
            value=-0.09, sem=0.01, confidence=0.9, units="frac_stature",
        )
        found = mechanics.check_directions([steady])
        assert found[0].severity is mechanics.Severity.GOOD

    def test_every_finding_survives_the_feedback_guard(self):
        """Rules are user-facing text and must obey the same injury/legality bans."""
        import feedback
        import mechanics

        measurements, timeline = self._clip()
        for f in mechanics.analyse(measurements, timeline, "batting"):
            feedback._assert_safe(f.text)

    def test_no_rule_targets_a_tier_c_metric(self):
        import mechanics

        for rule in mechanics.DIRECTION_RULES:
            assert metric_tiers.get(rule.metric_key).tier is not metric_tiers.Tier.C

    def test_optimum_band_metrics_have_no_direction_rule(self):
        """Stride length has an optimum, not a direction - experienced batters deliberately
        take a SHORT stride to play the ball later, so neither sign is a fault."""
        import mechanics

        keys = {r.metric_key for r in mechanics.DIRECTION_RULES}
        assert "batting.stride_length_norm" not in keys

    def test_every_finding_carries_its_evidence(self):
        import mechanics

        measurements, timeline = self._clip()
        for f in mechanics.analyse(measurements, timeline, "batting"):
            assert f.evidence.strip(), f"{f.rule} gives no numbers to check against the video"


class TestAspectRating:
    """The score grades the checks, not the player against a population."""

    def _findings(self, discipline="batting"):
        import mechanics

        clip = fixtures.batting_clip() if discipline == "batting" else fixtures.bowling_clip()
        scale = normalisation.estimate_scale(clip.seq)
        detect = (
            phase_detection.detect_batting if discipline == "batting"
            else phase_detection.detect_bowling
        )
        timeline = detect(clip.seq, handedness="right", scale_px=scale.stature_px)
        module = batting_metrics if discipline == "batting" else bowling_metrics
        measurements = module.compute_all(clip.seq, timeline, scale, handedness="right")
        return mechanics.analyse(measurements, timeline, discipline)

    def test_produces_a_score(self):
        import aspects

        report = aspects.score(self._findings(), "batting")
        assert report.overall is not None
        assert 0 <= report.overall <= 100

    def test_unassessable_aspects_score_none_not_zero(self):
        """Counting an unmeasurable aspect as a failure would understate; as a pass would
        overstate. It is neither."""
        import aspects

        report = aspects.score([], "batting")
        assert report.overall is None
        assert all(a.score is None for a in report.aspects)
        assert all(not a.assessed for a in report.aspects)

    def test_headline_always_quotes_coverage(self):
        """A score without its coverage misrepresents a thin analysis as a thorough one."""
        import aspects

        report = aspects.score(self._findings(), "batting")
        assert "of" in report.headline and "aspects" in report.headline

    def test_coverage_reflects_what_was_checkable(self):
        import aspects

        full = aspects.score(self._findings(), "batting")
        empty = aspects.score([], "batting")
        assert full.coverage > empty.coverage
        assert empty.coverage == 0.0

    def test_faults_lower_the_score(self):
        import aspects
        import mechanics

        good = mechanics.Finding(
            "order.stride", mechanics.RuleKind.ORDERING, mechanics.Severity.GOOD,
            "fine", "evidence", "mechanical",
        )
        bad = mechanics.Finding(
            "order.backlift", mechanics.RuleKind.ORDERING, mechanics.Severity.FAULT,
            "not fine", "evidence", "mechanical",
        )
        assert aspects.score([good], "batting").overall == 100.0
        assert aspects.score([good, bad], "batting").overall == 50.0

    def test_bowling_has_its_own_aspects(self):
        import aspects

        keys = {a.key for a in aspects.BOWLING_ASPECTS}
        assert "front_leg" in keys
        assert keys != {a.key for a in aspects.BATTING_ASPECTS}


class TestShotClassifier:
    """Identifies the shot from kinematics, and declines when the features disagree."""

    def test_front_foot_vertical_bat_reads_as_a_drive(self):
        import shot_classifier as sc

        f = sc.ShotFeatures(stride_forward=0.18, vertical_ratio=2.4, peak_hand_speed=3.0)
        result = sc.classify_batting(f)
        assert result.form_type == "cover_drive"
        assert result.confidence > 0.5
        assert result.reasoning

    def test_back_foot_horizontal_bat_reads_as_a_pull(self):
        import shot_classifier as sc

        f = sc.ShotFeatures(stride_forward=0.01, vertical_ratio=0.4, peak_hand_speed=4.0)
        result = sc.classify_batting(f)
        assert result.form_type == "pull_shot"

    def test_declines_when_the_axes_disagree(self):
        """A front-foot horizontal-bat shot is not one of our templates. Guessing between
        them would be worse than saying so."""
        import shot_classifier as sc

        f = sc.ShotFeatures(stride_forward=0.20, vertical_ratio=0.3)
        result = sc.classify_batting(f)
        assert result.form_type is None
        assert result.confidence < sc.MIN_CONFIDENCE + 0.06

    def test_declines_with_no_features_at_all(self):
        import shot_classifier as sc

        result = sc.classify_batting(sc.ShotFeatures())
        assert result.form_type is None
        assert result.confidence == 0.0

    def test_reports_the_numbers_behind_the_call(self):
        """A decision the player cannot check is a decision they have to take on trust."""
        import shot_classifier as sc

        f = sc.ShotFeatures(stride_forward=0.18, vertical_ratio=2.4)
        assert sc.classify_batting(f).features.as_dict()["stride_forward"] == 0.18


class TestConsistency:
    """Repeatability across a session - the one form quality needing no reference data."""

    def _clips(self, values, sem=0.005):
        return [
            [{"key": "batting.stride_duration_ss_se", "value": v, "sem": sem, "units": "s"}]
            for v in values
        ]

    def test_needs_a_minimum_number_of_clips(self):
        import consistency

        result = consistency.analyse(self._clips([0.20, 0.21]))
        assert result.metrics == []
        assert "record at least" in result.headline.lower()

    def test_tight_repetition_reads_as_repeatable(self):
        import consistency

        # sem well below the spread, so the variation is genuinely resolvable.
        result = consistency.analyse(
            self._clips([0.200, 0.203, 0.198, 0.201, 0.199], sem=0.0005)
        )
        assert result.metrics[0].verdict == "very repeatable"

    def test_spread_tighter_than_the_instrument_is_reported_as_such(self):
        """Not a failure: the player repeated themselves more tightly than the camera can
        resolve, which is worth saying rather than dropping."""
        import consistency

        result = consistency.analyse(
            self._clips([0.200, 0.201, 0.199, 0.200, 0.201], sem=0.02)
        )
        assert result.metrics[0].verdict == "as tight as we can measure"

    def test_scattered_repetition_is_flagged(self):
        import consistency

        result = consistency.analyse(self._clips([0.12, 0.26, 0.17, 0.31, 0.14]))
        assert result.metrics[0].verdict == "varies a lot"

    def test_measurement_error_is_subtracted_from_the_spread(self):
        """Otherwise the app reports the player as inconsistent when the camera was."""
        import consistency

        result = consistency.analyse(self._clips([0.19, 0.21, 0.20, 0.22, 0.18], sem=0.004))
        m = result.metrics[0]
        assert m.true_sd < m.observed_sd

    def test_spread_inside_measurement_error_is_unresolvable(self):
        import consistency

        # Spread comparable to - not far below - the measurement error: genuinely
        # indistinguishable from instrument noise, so neither a verdict nor a number.
        result = consistency.analyse(self._clips([0.19, 0.22, 0.18, 0.23, 0.20], sem=0.019))
        m = result.metrics[0]
        assert not m.resolvable
        assert m.verdict == "not resolvable"

    def test_metrics_missing_from_some_clips_are_skipped(self):
        """A metric measured in three clips of five would have its spread computed over a
        self-selected subset - and clips where a metric failed are not a random sample."""
        import consistency

        clips = self._clips([0.20, 0.21, 0.19, 0.20, 0.22])
        clips[0].append({"key": "batting.tempo_bc_ds", "value": 0.4, "sem": 0.01, "units": "s"})
        result = consistency.analyse(clips)
        assert all(m.metric_key != "batting.tempo_bc_ds" for m in result.metrics)


class TestShotClassifierFacing:
    """Regression: 'forward' depends on which way the batter faces, not on +x."""

    def test_facing_is_derived_not_assumed(self):
        import shot_classifier as sc
        from pose_extraction import KEYPOINT_INDEX

        clip = fixtures.batting_clip()
        scale = normalisation.estimate_scale(clip.seq)
        timeline = phase_detection.detect_batting(
            clip.seq, handedness="right", scale_px=scale.stature_px
        )
        forward = sc.extract_features(
            clip.seq, timeline, scale, handedness="right"
        ).stride_forward

        # Mirror the whole clip: same shot, camera on the other side.
        mirrored = PoseSequence(
            xy=clip.seq.xy * np.array([-1.0, 1.0], dtype=np.float32),
            confidence=clip.seq.confidence,
            fps=clip.seq.fps,
        )
        m_scale = normalisation.estimate_scale(mirrored)
        m_timeline = phase_detection.detect_batting(
            mirrored, handedness="right", scale_px=m_scale.stature_px
        )
        mirrored_forward = sc.extract_features(
            mirrored, m_timeline, m_scale, handedness="right"
        ).stride_forward

        if forward is not None and mirrored_forward is not None:
            assert (forward > 0) == (mirrored_forward > 0), (
                "mirroring the camera flipped front-foot into back-foot"
            )


class TestDisciplineDetection:
    """Regression: a long flick clip was classified as spin bowling."""

    def test_batting_and_bowling_are_separated(self):
        import shot_classifier as sc

        for name, mk in (("batting", fixtures.batting_clip), ("bowling", fixtures.bowling_clip)):
            clip = mk()
            discipline, _, _ = sc.detect_discipline(
                clip.seq, normalisation.estimate_scale(clip.seq)
            )
            assert discipline == name

    def test_a_batter_who_walks_is_still_a_batter(self):
        """Travel alone cannot call bowling: a batter walking between deliveries, or a
        tracker switching to someone else in the nets, covers just as much ground as a
        run-up. This is what made a flick read as spin bowling."""
        import shot_classifier as sc
        from pose_extraction import KEYPOINT_INDEX

        clip = fixtures.batting_clip()
        xy = clip.seq.xy.copy()
        for name in ("left_hip", "right_hip", "nose", "left_shoulder", "right_shoulder"):
            xy[:, KEYPOINT_INDEX[name], 0] += np.linspace(0, 900, len(xy))
        walking = PoseSequence(xy=xy, confidence=clip.seq.confidence, fps=clip.seq.fps)

        discipline, _, why = sc.detect_discipline(
            walking, normalisation.estimate_scale(walking)
        )
        assert discipline == "batting", why
        assert "over the head" in why

    def test_bowling_needs_the_arm_over_the_head(self):
        """The one signal every delivery has and no batting shot does."""
        import shot_classifier as sc
        from pose_extraction import KEYPOINT_INDEX

        clip = fixtures.bowling_clip()
        xy = clip.seq.xy.copy()
        # Pin both wrists at chest height: travel remains, arm action gone.
        for side in ("left_wrist", "right_wrist"):
            xy[:, KEYPOINT_INDEX[side], 1] = xy[:, KEYPOINT_INDEX["left_shoulder"], 1] + 40
        armless = PoseSequence(xy=xy, confidence=clip.seq.confidence, fps=clip.seq.fps)

        discipline, _, _ = sc.detect_discipline(
            armless, normalisation.estimate_scale(armless)
        )
        assert discipline == "batting"
