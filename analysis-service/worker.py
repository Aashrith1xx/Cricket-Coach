"""Background job runner: the full analysis pipeline for one clip.

**Status: orchestration is real; the steps it calls are stubs.**

Analysis is asynchronous, for two reasons that are both about Render's constraints rather
than elegance: a clip takes an estimated 1-2 minutes on a fractional-vCPU instance, and free
tier services spin down after 15 minutes idle, so anything holding an HTTP connection open
would lose in-flight work.

The pipeline mirrors ``README.md`` section 4. Ordering is not arbitrary - the gate runs
before pose extraction so an unusable clip fails in milliseconds rather than after a minute
of inference.
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import capture_gate
import feedback
import metric_tiers
import normalisation
import phase_detection
import pose_extraction
import render_output
import retention
import scoring
import style_resolution
import template_validator
from capture_gate import CaptureMetadata, GateResult
from style_resolution import Style

log = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class JobRequest:
    clip_id: str
    session_id: str
    user_id: str
    form_type: str
    video_path: str
    metadata: CaptureMetadata
    declared_style: Style
    handedness: Optional[str] = None
    player_group: str = "unknown"
    consent_for_training: bool = False
    front_on_video_path: Optional[str] = None
    #: Pre-extracted pose, bypassing video decode. Used when re-analysing a stored clip from
    #: retained keypoints after a scoring or template change, and by the test suite.
    pose_sequence: Optional[object] = None
    #: Skip the pre-pose frame gate. Only valid when ``pose_sequence`` is supplied, since
    #: there is no video to check. The post-pose gate still runs.
    skip_frame_gate: bool = False


@dataclass
class JobResult:
    clip_id: str
    status: JobStatus
    gate: Optional[GateResult] = None
    composite: Optional[dict] = None
    feedback: Optional[dict] = None
    phase_events: Optional[dict] = None
    render_data_url: Optional[str] = None
    template_id: Optional[str] = None
    template_version: Optional[str] = None
    style_variant: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "clip_id": self.clip_id,
            "status": self.status.value,
            "gate": self.gate.as_dict() if self.gate else None,
            "composite": self.composite,
            "feedback": self.feedback,
            "phase_events": self.phase_events,
            "render_data_url": self.render_data_url,
            "template_id": self.template_id,
            "template_version": self.template_version,
            "style_variant": self.style_variant,
            "warnings": self.warnings,
            "error": self.error,
        }


#: Templates are validated once at boot. A failure here stops the service rather than
#: degrading to unsourced scoring.
_TEMPLATES: Dict[str, dict] = {}


def load_templates() -> Dict[str, dict]:
    global _TEMPLATES
    if not _TEMPLATES:
        _TEMPLATES = template_validator.load_all()
        log.info("Loaded %d reference templates", len(_TEMPLATES))
    return _TEMPLATES


def process(job: JobRequest) -> JobResult:
    """Run the full pipeline for one clip.

    Failure policy: a rejected clip is a *successful* run with an actionable reason, and is
    reported as ``REJECTED``. Only unexpected errors become ``FAILED``. The distinction
    matters because a rejection should be shown to the player as guidance, whereas a failure
    should page whoever maintains this.
    """
    result = JobResult(clip_id=job.clip_id, status=JobStatus.PROCESSING)

    try:
        templates = load_templates()
        template = templates.get(job.form_type)
        if template is None:
            result.status = JobStatus.FAILED
            result.error = f"No reference template for form type {job.form_type!r}"
            return result

        result.template_id = job.form_type
        result.template_version = template["version"]

        # 1. Cheap metadata gate --------------------------------------------
        meta_gate = capture_gate.check_metadata(job.metadata)
        result.gate = meta_gate
        result.warnings.extend(meta_gate.warnings)
        if not meta_gate.passed:
            result.status = JobStatus.REJECTED
            return result

        # 2. Full gate, decoding frames -------------------------------------
        required_view = "side_on"
        if not job.skip_frame_gate:
            full_gate = capture_gate.check_frames(
                job.video_path, job.metadata, required_view=required_view
            )
            result.gate = full_gate
            result.warnings.extend(full_gate.warnings)
            if not full_gate.passed:
                result.status = JobStatus.REJECTED
                return result

        has_front_on = job.front_on_video_path is not None
        fps = result.gate.effective_fps if result.gate else job.metadata.fps
        fps = fps or job.metadata.fps

        # 3. Pose -----------------------------------------------------------
        seq = job.pose_sequence or pose_extraction.extract(job.video_path)
        handedness = job.handedness or pose_extraction.infer_handedness(seq)
        discipline = template["discipline"]

        # 3b. Post-pose gate: subject size, tracking, occlusion, view angle --
        pose_gate = capture_gate.check_pose_quality(
            seq, required_view=required_view, discipline=discipline
        )
        result.warnings.extend(pose_gate.warnings)
        if not pose_gate.passed:
            result.gate = pose_gate
            result.status = JobStatus.REJECTED
            return result

        # 4. Scale ----------------------------------------------------------
        scale = normalisation.estimate_scale(seq)

        # 5. Events ---------------------------------------------------------
        if discipline == "batting":
            timeline = phase_detection.detect_batting(
                seq, handedness=handedness, scale_px=scale.stature_px
            )
        else:
            timeline = phase_detection.detect_bowling(
                seq, handedness=handedness, scale_px=scale.stature_px
            )

        missing = timeline.missing(phase_detection.required_events(job.form_type))
        if missing:
            result.status = JobStatus.REJECTED
            result.gate = GateResult(
                passed=False,
                rejections=[
                    capture_gate.Rejection(
                        capture_gate.RejectionCode.NO_ACTION_DETECTED,
                        f"could not locate {', '.join(missing)}",
                    )
                ],
            )
            return result

        result.phase_events = timeline.as_dict()

        # 6. Metrics --------------------------------------------------------
        measurements = _compute_metrics(
            discipline,
            seq=seq,
            timeline=timeline,
            scale=scale,
            handedness=handedness,
            fps=fps,
            has_front_on=has_front_on,
        )

        # 7. Style ----------------------------------------------------------
        bands = style_resolution.resolve_bands(template, job.declared_style)
        result.style_variant = next(
            (b.style_variant for b in bands.values() if b.style_variant), None
        )
        result.warnings.extend(
            style_resolution.population_caveat(bands, job.player_group)
        )

        # 8-9. Score --------------------------------------------------------
        scores = scoring.score_all(measurements, bands)
        composite = scoring.composite(scores)
        result.composite = composite.as_dict()

        # 10. Feedback ------------------------------------------------------
        descriptors = _detect_style_markers(discipline, seq, timeline, handedness, has_front_on)
        bundle = feedback.generate(composite, descriptors=descriptors)
        result.feedback = bundle.as_dict()

        # 11. Render payload ------------------------------------------------
        result.render_data_url = render_output.publish(
            clip_id=job.clip_id,
            seq=seq,
            timeline=timeline,
            scores=scores,
        )

        # 12. Retention -----------------------------------------------------
        retention.apply(
            clip_id=job.clip_id,
            video_path=job.video_path,
            consent_for_training=job.consent_for_training,
        )

        result.status = JobStatus.DONE
        return result

    except NotImplementedError as exc:
        result.status = JobStatus.FAILED
        result.error = f"Pipeline step not implemented: {exc}"
        return result
    except Exception as exc:  # noqa: BLE001 - worker boundary
        log.exception("Analysis failed for clip %s", job.clip_id)
        result.status = JobStatus.FAILED
        result.error = f"{type(exc).__name__}: {exc}"
        result.warnings.append(traceback.format_exc(limit=3))
        return result


def _compute_metrics(
    discipline: str,
    *,
    seq,
    timeline,
    scale,
    handedness: str,
    fps: float,
    has_front_on: bool,
):
    """Compute every metric applicable under these capture conditions.

    Only metrics returned by :func:`metric_tiers.applicable` are attempted. That function
    already excludes anything whose ``min_fps`` is unmet or which needs the front-on camera,
    so capture conditions gate the metric set in one place rather than being re-checked
    inside each metric function.

    A metric returning ``None`` is skipped, never defaulted - there is no sensible default
    for a joint angle, and a defaulted one is indistinguishable from a measured one.
    """
    from metrics import batting as batting_metrics
    from metrics import bowling as bowling_metrics

    allowed = {
        spec.key
        for spec in metric_tiers.applicable(
            metric_tiers.Discipline(discipline), fps=fps, has_front_on=has_front_on
        )
    }
    module = batting_metrics if discipline == "batting" else bowling_metrics
    measurements = module.compute_all(
        seq, timeline, scale, handedness=handedness, allowed=allowed
    )
    log.info(
        "%d of %d applicable metrics computed at %.0f fps",
        len(measurements),
        len(allowed),
        fps,
    )
    return measurements


def _detect_style_markers(
    discipline: str, seq, timeline, handedness: str, has_front_on: bool
) -> Dict[str, str]:
    """Style descriptors, or empty when unavailable.

    Returns ``{}`` rather than raising - markers are supplementary, and their absence should
    not fail an otherwise good analysis. Backlift direction in particular is unavailable
    without a front-on view and reports indeterminate rather than guessing, since a wrong
    classification would be shown to the player as a fact about their technique.
    """
    if discipline != "batting":
        return {}

    import style_markers

    markers: Dict[str, str] = {}

    bc, ds = timeline.frame_of("BC"), timeline.frame_of("DS")
    if bc is not None and ds is not None:
        direction, confidence = style_markers.detect_backlift_direction(
            seq, bc, ds, handedness=handedness, has_front_on=has_front_on
        )
        markers["batting.backlift_direction"] = style_markers.DISPLAY[direction]

    stance, ss = timeline.frame_of("STANCE"), timeline.frame_of("SS")
    if stance is not None and ss is not None:
        trigger_present = timeline.frame_of("TRIGGER") is not None
        trigger_type, _ = style_markers.classify_trigger(
            seq, stance, ss, handedness=handedness, detected=trigger_present
        )
        markers["batting.trigger_present"] = style_markers.DISPLAY[trigger_type]

    if stance is not None:
        try:
            width, _ = style_markers.detect_stance_width(
                seq, stance, stature_px=_stature_of(seq)
            )
            markers["batting.stance_width"] = style_markers.DISPLAY[width]
        except Exception:  # noqa: BLE001 - descriptors are supplementary
            log.debug("stance width unavailable", exc_info=True)

    return markers


def _stature_of(seq) -> float:
    return normalisation.estimate_scale(seq).stature_px
