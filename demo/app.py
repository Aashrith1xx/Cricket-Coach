"""Local demo: upload a clip, run the real analysis pipeline, see what comes out.

This is not the production service - it is a single-process Flask app that calls exactly the
same pipeline modules, so what you see here is what the real system would produce.

Run it:

    cd demo
    ../analysis-service/.venv/Scripts/python.exe app.py

then open http://localhost:5000.

**What it will and will not tell you.** It reports what it *measured* - detected phase
events, joint angles, timings, distances - each with a confidence interval. It does not give
you a score, because the reference templates have almost no sourced expected ranges yet
(see ``docs/ACCURACY.md`` and ``reference_templates/``). Measuring and grading are different
problems, and only the first one is solved.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import traceback
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Optional

# The analysis service is a sibling directory, not an installed package.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "analysis-service"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from flask import Flask, jsonify, render_template, request, send_from_directory  # noqa: E402

import aspects  # noqa: E402
import capture_gate  # noqa: E402
import consistency  # noqa: E402
import shot_classifier  # noqa: E402
import feedback  # noqa: E402
import mechanics  # noqa: E402
import metric_tiers  # noqa: E402
import normalisation  # noqa: E402
import phase_detection  # noqa: E402
import pose_extraction  # noqa: E402
import scoring  # noqa: E402
import reconstruct  # noqa: E402
import style_resolution  # noqa: E402
import template_validator  # noqa: E402
from capture_gate import CaptureMetadata  # noqa: E402
from metrics import batting as batting_metrics  # noqa: E402
from metrics import bowling as bowling_metrics  # noqa: E402
from pose_backends import OnnxPoseBackend  # noqa: E402
from style_resolution import Style  # noqa: E402

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("demo")

app = Flask(__name__)

#: Upload cap. Kept modest on a small instance: the request body is buffered before decode,
#: so this is memory spent before any of the analysis budget is touched.
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", 200))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

OUTPUT_DIR = Path(tempfile.gettempdir()) / "cricket_demo"
OUTPUT_DIR.mkdir(exist_ok=True)

#: Rendered overlays live on an ephemeral disk that nothing else prunes, so old ones are
#: swept on each request. Long enough that a user can still replay their result, short
#: enough that a free-tier disk does not fill.
RENDER_TTL_SECONDS = 3600

#: Saved analyses. Kept as JSON next to the renders so a result can be reopened later.
#: Deliberately excludes the uploaded video, which is deleted the moment analysis finishes.
SAVED_DIR = OUTPUT_DIR / "saved"
SAVED_DIR.mkdir(exist_ok=True)


def save_report(report: dict, form_type: str) -> str:
    """Persist a finished analysis and return its id."""
    import datetime as _dt
    import json as _json

    analysis_id = uuid.uuid4().hex[:12]
    record = {
        "id": analysis_id,
        "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "form_type": form_type,
        "report": report,
    }
    (SAVED_DIR / f"{analysis_id}.json").write_text(
        _json.dumps(record, default=str), encoding="utf-8"
    )
    return analysis_id


def load_saved(analysis_id: str) -> Optional[dict]:
    import json as _json

    path = SAVED_DIR / f"{analysis_id}.json"
    if not path.exists():
        return None
    return _json.loads(path.read_text(encoding="utf-8"))


def list_saved() -> list:
    """Saved analyses, newest first, with just enough to render a list."""
    import json as _json

    out = []
    for path in SAVED_DIR.glob("*.json"):
        try:
            rec = _json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        r = rec.get("report", {})
        out.append({
            "id": rec["id"],
            "saved_at": rec["saved_at"],
            "form_label": r.get("form_label", rec.get("form_type", "")),
            "overall": (r.get("rating") or {}).get("overall"),
            "headline": (r.get("rating") or {}).get("headline", ""),
            "metrics": f"{r.get('metrics_measured', 0)} of {r.get('metrics_total', 0)}",
            "rejected": bool(r.get("rejected")),
        })
    return sorted(out, key=lambda x: x["saved_at"], reverse=True)


def sweep_old_renders() -> None:
    """Delete rendered overlays older than the TTL. Best effort - never fails a request."""
    import time

    cutoff = time.time() - RENDER_TTL_SECONDS
    for path in OUTPUT_DIR.glob("*"):
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass

MODELS = ROOT.parent / "analysis-service" / "models"

FORM_TYPES = {
    "cover_drive": ("Cover drive", "batting"),
    "pull_shot": ("Pull shot", "batting"),
    "fast_arm_bowling": ("Fast bowling", "bowling"),
    "spin_bowling": ("Spin bowling", "bowling"),
}

_backend = None


def backend() -> OnnxPoseBackend:
    """Load the models once. Cold start is several seconds; per-clip cost should not pay it."""
    global _backend
    if _backend is None:
        det = MODELS / "rtmdet.onnx"
        pose = MODELS / "rtmpose.onnx"
        if not det.exists() or not pose.exists():
            raise FileNotFoundError(
                f"Model weights missing from {MODELS}. See README.md - they are Apache 2.0 "
                f"downloads from OpenMMLab, not committed to the repo."
            )
        _backend = OnnxPoseBackend(str(det), str(pose), detect_every=5)
    return _backend


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
]

LOW_CONFIDENCE = 0.5


def annotate(frames, seq, timeline, out_path: Path, fps: float) -> None:
    """Write a skeleton-overlay video.

    Low-confidence joints are drawn hollow and their limbs dashed, so you can see where the
    analysis was unsure rather than being shown a uniformly crisp skeleton over guessed
    joints. Event frames are labelled as they pass.
    """
    height, width = frames[0].shape[:2]
    # VP8/WebM rather than mp4: this OpenCV build has no OpenH264, so its only working mp4
    # codec is mp4v (MPEG-4 Part 2), which browsers will not play. VP8 is supported natively
    # by every current browser and writes fine here.
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"VP80"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open a video writer for {out_path}")

    event_by_frame = {
        e.frame: e.name for e in timeline.events.values() if e.detected and e.frame is not None
    }

    try:
        for i, frame in enumerate(frames):
            canvas = frame.copy()
            xy, conf = seq.xy[i], seq.confidence[i]

            for a, b in SKELETON:
                if np.isnan(xy[a]).any() or np.isnan(xy[b]).any():
                    continue
                weak = min(conf[a], conf[b]) < LOW_CONFIDENCE
                colour = (120, 120, 120) if weak else (60, 220, 60)
                pa, pb = tuple(xy[a].astype(int)), tuple(xy[b].astype(int))
                cv2.line(canvas, pa, pb, colour, 1 if weak else 2, cv2.LINE_AA)

            for k in range(len(xy)):
                if np.isnan(xy[k]).any():
                    continue
                weak = conf[k] < LOW_CONFIDENCE
                centre = tuple(xy[k].astype(int))
                cv2.circle(canvas, centre, 3, (40, 40, 220), -1 if not weak else 1, cv2.LINE_AA)

            label = event_by_frame.get(i)
            if label:
                cv2.rectangle(canvas, (0, 0), (width, 46), (20, 20, 20), -1)
                cv2.putText(
                    canvas, label, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (80, 240, 255), 2, cv2.LINE_AA,
                )

            cv2.putText(
                canvas, f"frame {i}", (14, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (230, 230, 230), 1, cv2.LINE_AA,
            )
            writer.write(canvas)
    finally:
        writer.release()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def analyse(
    video_path: str,
    form_type: str,
    declared_style: Style,
    handedness: str,
    declared_view: str = "auto",
    front_video_path: Optional[str] = None,
) -> dict:
    """Run the real pipeline and shape the result for the page."""
    templates = template_validator.load_all()
    auto = form_type == "auto"
    if not auto:
        label, discipline = FORM_TYPES[form_type]
        template = templates[form_type]
    else:
        # Placeholders until pose tells us what this is. Detection needs the skeleton, and
        # the skeleton needs no template, so this ordering works - it just means the report
        # cannot name the shot until after pose has run.
        label, discipline, template = "Working it out…", None, None

    # Frames buffered for the overlay only - downscaled, since rendering needs no detail.
    frames, fps, width, height = pose_extraction.read_frames(video_path)
    duration = len(frames) / fps

    meta = CaptureMetadata(
        fps=fps, width=width, height=height, duration_s=duration, declared_view="side_on"
    )

    report: dict = {
        "form_label": label,
        "discipline": discipline,
        "auto_detect": auto,
        "fps": round(fps, 1),
        "frames": len(frames),
        "duration": round(duration, 2),
        "resolution": f"{width}x{height}",
    }

    # --- Gate 1: metadata -------------------------------------------------
    meta_gate = capture_gate.check_metadata(meta)
    report["warnings"] = list(meta_gate.warnings)
    if not meta_gate.passed:
        report["rejected"] = [r.__dict__ | {"remedy": r.remedy} for r in meta_gate.rejections]
        return report

    # --- Gate 2: camera motion --------------------------------------------
    motion = capture_gate.estimate_camera_motion(frames)
    report["camera_motion_pct"] = round(motion * 100, 3)
    if motion > capture_gate.MAX_BACKGROUND_MOTION_FRACTION:
        report["rejected"] = [
            {
                "code": "camera_motion",
                "detail": f"background moved {motion * 100:.2f}% of frame width per frame",
                "remedy": capture_gate.REMEDIES[capture_gate.RejectionCode.CAMERA_MOTION],
            }
        ]
        return report

    # --- Pose --------------------------------------------------------------
    # Pose runs on a SEPARATE streaming pass at the video's own resolution. Estimating from
    # the downscaled render frames shrank the player badly and cost real tracking accuracy.
    seq = pose_extraction.extract_streaming(video_path, backend())

    import cv2 as _cv2
    _probe = _cv2.VideoCapture(video_path)
    src_w = int(_probe.get(_cv2.CAP_PROP_FRAME_WIDTH)) or width
    _probe.release()
    render_scale = width / src_w if src_w else 1.0
    report["source_resolution"] = f"{src_w}px wide"
    report["mean_confidence"] = round(float(seq.mean_confidence()), 3)

    if handedness == "auto":
        handedness = pose_extraction.infer_handedness(seq)
    report["handedness"] = handedness

    # --- Gate 3: pose quality ---------------------------------------------
    # "any" accepts either canonical view; the detected one then selects the metric set.
    scaled_boxes = seq.bbox * render_scale if seq.bbox is not None else None
    has_mesh, edge_density = capture_gate.detect_mesh(frames, boxes=scaled_boxes)
    report["has_mesh"] = has_mesh
    report["mesh_edge_density"] = round(edge_density, 4)

    pose_gate = capture_gate.check_pose_quality(
        seq,
        required_view="any",
        discipline=discipline,
        frame_height=height,
        has_mesh=has_mesh,
    )
    report["warnings"].extend(pose_gate.warnings)
    # A DECLARED view always wins. The shoulder-width estimator assumes a person facing
    # their direction of travel, but a batter is already rotated 90 degrees in their stance,
    # so a camera to their side sees the shoulder line broadside and WIDE - which the
    # estimator then reads as front-on. The player knows where they stood; we are guessing.
    # Same principle as playstyle: a declared value beats an inference that is often wrong.
    detected = pose_gate.detected_view
    view = declared_view if declared_view in ("side_on", "front_on") else (detected or "side_on")
    report["detected_view"] = view
    report["view_was_declared"] = declared_view in ("side_on", "front_on")
    report["view_estimator_said"] = detected
    report["view_deviation_deg"] = (
        round(pose_gate.estimated_view_deviation_deg, 1)
        if pose_gate.estimated_view_deviation_deg is not None
        else None
    )
    # A declared view overrides any view-angle complaint - the user is the authority on
    # where their own camera was.
    blocking = [
        r
        for r in pose_gate.rejections
        if not (report["view_was_declared"] and r.code is capture_gate.RejectionCode.WRONG_VIEW_ANGLE)
    ]
    if blocking:
        report["rejected"] = [
            {"code": r.code.value, "detail": r.detail, "remedy": r.remedy} for r in blocking
        ]
        return report

    # --- Scale and events --------------------------------------------------
    scale = normalisation.estimate_scale(seq)
    report["stature_px"] = round(scale.stature_px, 1)

    if auto:
        discipline, disc_conf, disc_why = shot_classifier.detect_discipline(seq, scale)
        report["discipline"] = discipline
        report["detected_discipline"] = {
            "discipline": discipline, "confidence": round(disc_conf, 2), "reasoning": disc_why,
        }

    detect = (
        phase_detection.detect_batting if discipline == "batting"
        else phase_detection.detect_bowling
    )
    timeline = detect(seq, handedness=handedness, scale_px=scale.stature_px)

    report["events"] = [
        {
            "name": e.name,
            "frame": e.frame,
            "time": round(e.time_s, 3) if e.time_s is not None else None,
            "confidence": round(e.confidence, 2),
            "uncertainty_ms": round(e.time_sem_s * 1000, 1) if e.detected else None,
            "detected": e.detected,
        }
        for e in timeline.events.values()
    ]

    missing = timeline.missing(
        phase_detection.required_events_for_discipline(discipline)
        if auto
        else phase_detection.required_events(form_type)
    )
    if missing:
        report["rejected"] = [
            {
                "code": "no_action_detected",
                "detail": f"could not locate {', '.join(missing)}",
                "remedy": capture_gate.REMEDIES[capture_gate.RejectionCode.NO_ACTION_DETECTED],
            }
        ]
        report["render"] = write_render(
            frames, _scaled_sequence(seq, render_scale), timeline, fps
        )
        return report

    # --- Metrics -----------------------------------------------------------
    detected_shot = shot_classifier.classify(
        seq, timeline, scale, discipline=discipline, handedness=handedness, view=view
    )
    report["detected_shot"] = detected_shot.as_dict()

    if auto:
        if detected_shot.form_type:
            form_type = detected_shot.form_type
        else:
            # Features could not separate the shots. Fall back to the discipline's default
            # template rather than refusing outright - the metrics and mechanical checks do
            # not depend on which shot it is, only the reference bands would, and those are
            # empty anyway.
            form_type = "cover_drive" if discipline == "batting" else "fast_arm_bowling"
            report["auto_fell_back"] = True
        label = FORM_TYPES[form_type][0]
        template = templates[form_type]
        report["form_label"] = label + (
            "" if detected_shot.form_type else " (best guess)"
        )

    module = batting_metrics if discipline == "batting" else bowling_metrics
    allowed = {
        s.key
        for s in metric_tiers.applicable(
            metric_tiers.Discipline(discipline), fps=fps, view=view
        )
    }
    measurements = module.compute_all(
        seq, timeline, scale, handedness=handedness, allowed=allowed
    )

    # Mesh occludes parts of the body, so the tracking really is worse. Accept the clip but
    # widen every interval rather than pretending the view was clean.
    if has_mesh:
        measurements = [
            replace(m, sem=m.sem * capture_gate.MESH_UNCERTAINTY_INFLATION)
            for m in measurements
        ]

    report["unavailable_in_view"] = [
        s.summary
        for s in metric_tiers.unavailable_in_view(metric_tiers.Discipline(discipline), view)
        if s.summary
    ]

    bands = style_resolution.resolve_bands(template, declared_style)
    scores = scoring.score_all(measurements, bands)
    composite = scoring.composite(scores)

    report["metrics"] = _full_metric_list(scores, discipline, view, fps)
    report["metrics_measured"] = sum(1 for m in report["metrics"] if m["measured"])
    report["metrics_total"] = len(report["metrics"])
    report["composite"] = composite.composite
    report["suppressed_reason"] = composite.suppressed_reason
    report["render"] = write_render(
        frames, _scaled_sequence(seq, render_scale), timeline, fps
    )

    # --- Optional front clip: extra metrics, and the lateral axis for the 3D view -------
    front_bits = None
    if front_video_path:
        try:
            front_bits = _analyse_front(front_video_path, discipline, handedness, fps_hint=fps)
        except Exception as exc:  # noqa: BLE001 - the side analysis must survive this
            log.warning("Front clip failed: %s", exc)
            report["front_error"] = str(exc)

    if front_bits:
        report["front_ok"] = True
        report["front_metrics"] = len(front_bits["measurements"])
        extra = scoring.score_all(front_bits["measurements"], bands)
        report["metrics"] = _full_metric_list(scores + extra, discipline, "both", fps)
        report["metrics_measured"] = sum(1 for m in report["metrics"] if m["measured"])
        report["metrics_total"] = len(report["metrics"])

    recon = reconstruct.compose(
        seq, scale, timeline,
        front_bits["seq"] if front_bits else None,
        front_bits["scale"] if front_bits else None,
        front_bits["timeline"] if front_bits else None,
    )
    # Mechanical findings: feedback that needs no reference population, only direction and
    # ordering. Routed through the same guard as every other user-facing string, so a rule
    # cannot smuggle in an injury claim or a Tier C metric.
    findings = mechanics.analyse(measurements, timeline, discipline)
    for f in findings:
        feedback._assert_safe(f.text)
    report["findings"] = [f.as_dict() for f in findings]
    summary = mechanics.summarise(findings, measurements, discipline)
    feedback._assert_safe(summary.headline)
    for f in summary.faults:
        if f.remedy:
            feedback._assert_safe(f.remedy)
    report["summary"] = summary.as_dict()
    report["rating"] = aspects.score(findings, discipline).as_dict()

    report["reconstruction"] = recon.as_dict()
    report["skeleton_edges"] = [list(e) for e in SKELETON]
    return report


def _analyse_front(path: str, discipline: str, handedness: str, *, fps_hint: float) -> dict:
    """Run the front clip through pose and events, for its own metrics and the lateral axis."""
    seq = pose_extraction.extract_streaming(path, backend())
    scale = normalisation.estimate_scale(seq)
    detect = (
        phase_detection.detect_batting if discipline == "batting"
        else phase_detection.detect_bowling
    )
    timeline = detect(seq, handedness=handedness, scale_px=scale.stature_px)

    detected_shot = shot_classifier.classify(
        seq, timeline, scale, discipline=discipline, handedness=handedness, view=view
    )
    report["detected_shot"] = detected_shot.as_dict()

    if auto:
        if detected_shot.form_type:
            form_type = detected_shot.form_type
        else:
            # Features could not separate the shots. Fall back to the discipline's default
            # template rather than refusing outright - the metrics and mechanical checks do
            # not depend on which shot it is, only the reference bands would, and those are
            # empty anyway.
            form_type = "cover_drive" if discipline == "batting" else "fast_arm_bowling"
            report["auto_fell_back"] = True
        label = FORM_TYPES[form_type][0]
        template = templates[form_type]
        report["form_label"] = label + (
            "" if detected_shot.form_type else " (best guess)"
        )

    module = batting_metrics if discipline == "batting" else bowling_metrics
    allowed = {
        s.key
        for s in metric_tiers.applicable(
            metric_tiers.Discipline(discipline), fps=seq.fps, view="front_on"
        )
    }
    return {
        "seq": seq,
        "scale": scale,
        "timeline": timeline,
        "measurements": module.compute_all(
            seq, timeline, scale, handedness=handedness, allowed=allowed
        ),
    }


def _metric_row(s) -> dict:
    """Shape one score for the page, clamping intervals to the physically possible.

    The confidence interval is a normal approximation, so on a joint angle near full
    extension its upper bound can run past 180 degrees - a knee that reads "142 to 207
    degrees" is arithmetically defensible and anatomically nonsense, and nonsense on screen
    costs more trust than the extra width buys in honesty. Angles are bounded to [0, 180]
    in the ``straight_180`` convention this service uses internally.
    """
    lo, hi = s.measurement.ci_low, s.measurement.ci_high
    if s.measurement.units == "deg":
        lo, hi = max(0.0, lo), min(180.0, hi)

    return {
        "key": s.key,
        "summary": metric_tiers.get(s.key).summary,
        "tier": s.tier.value,
        "value": round(s.measurement.value + 0.0, 3),  # +0.0 normalises negative zero
        "ci_low": round(lo + 0.0, 3),
        "ci_high": round(hi + 0.0, 3),
        "units": s.measurement.units,
        "confidence": round(s.measurement.confidence, 2),
        "verdict": s.verdict.value,
        "has_band": s.band is not None,
        "band_low": round(s.band.low, 3) if s.band else None,
        "band_high": round(s.band.high, 3) if s.band else None,
        "source": s.band.source if s.band else None,
    }


def _scaled_sequence(seq, scale: float):
    """Keypoints rescaled into the downscaled render frames' coordinate space."""
    if scale == 1.0:
        return seq
    return pose_extraction.PoseSequence(
        xy=seq.xy * scale, confidence=seq.confidence, fps=seq.fps, bbox=seq.bbox
    )


def _why_unmeasured(spec, view: str, fps: float) -> str:
    """Plain-language reason a metric produced nothing, so the gap is explained not hidden."""
    if view not in spec.views:
        other = "front-on" if view == "side_on" else "side-on"
        return f"needs a {other} camera - this angle can't see it"
    if spec.min_fps and fps < spec.min_fps:
        return f"needs {spec.min_fps}fps or higher to locate the moment of contact"
    return "the phase it's measured at wasn't found in this clip"


def _full_metric_list(scores, discipline: str, view: str, fps: float) -> list:
    """Every metric for this discipline, measured or not.

    Showing only what succeeded makes a thin result look complete and hides *why* the rest
    are absent - which is usually something the player can fix by moving the camera.
    """
    measured = {s.key: s for s in scores}
    rows = []

    for spec in metric_tiers.REGISTRY.values():
        if spec.discipline.value != discipline or spec.tier is metric_tiers.Tier.C:
            continue
        if spec.key in measured:
            row = _metric_row(measured[spec.key])
            row["measured"] = True
            rows.append(row)
        else:
            rows.append(
                {
                    "key": spec.key,
                    "summary": spec.summary,
                    "tier": spec.tier.value,
                    "measured": False,
                    "reason": _why_unmeasured(spec, view, fps),
                    "value": None,
                    "ci_low": None,
                    "ci_high": None,
                    "units": "",
                    "verdict": "not_measured",
                    "has_band": False,
                    "band_low": None,
                    "band_high": None,
                    "source": None,
                }
            )

    rows.sort(key=lambda r: (not r["measured"], r["tier"], r["key"]))
    return rows


def write_render(frames, seq, timeline, fps: float) -> str:
    name = f"{uuid.uuid4().hex}.webm"
    # Slow-motion clips are captured at 120-240 fps; played back at that rate the action is
    # over before you can see it. Cap playback so the overlay is actually watchable.
    annotate(frames, seq, timeline, OUTPUT_DIR / name, min(fps, 30.0))
    return name


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    sweep_old_renders()
    return render_template(
        "index.html", form_types=FORM_TYPES, max_upload_mb=MAX_UPLOAD_MB
    )


@app.errorhandler(413)
def too_large(_err):
    return (
        render_template(
            "result.html",
            r={
                "fatal": f"That file is over the {MAX_UPLOAD_MB} MB limit.",
                "trace": (
                    "Trim the clip to a single shot or delivery. A one-to-two second clip is "
                    "all the analysis uses, and shorter clips are processed far faster."
                ),
            },
        ),
        413,
    )


@app.route("/analyse", methods=["POST"])
def analyse_route():
    upload = request.files.get("clip")
    if upload is None or not upload.filename:
        return jsonify({"error": "No clip uploaded"}), 400

    form_type = request.form.get("form_type", "cover_drive")
    if form_type != "auto" and form_type not in FORM_TYPES:
        return jsonify({"error": "Unknown form type"}), 400

    style = Style(request.form.get("style", "balanced"))
    handedness = request.form.get("handedness", "auto")
    declared_view = request.form.get("view", "auto")

    front_upload = request.files.get("front_clip")
    front_path = None
    if front_upload and front_upload.filename:
        front_path = OUTPUT_DIR / f"front_{uuid.uuid4().hex}{Path(front_upload.filename).suffix or '.mp4'}"
        front_upload.save(str(front_path))

    suffix = Path(upload.filename).suffix or ".mp4"
    temp_path = OUTPUT_DIR / f"upload_{uuid.uuid4().hex}{suffix}"
    upload.save(str(temp_path))

    try:
        report = analyse(
            str(temp_path), form_type, style, handedness, declared_view,
            front_video_path=str(front_path) if front_path else None,
        )
        # Save under the DETECTED type, so session consistency groups correctly.
        saved_type = report.get("detected_shot", {}).get("form_type") or form_type
        report["analysis_id"] = save_report(report, saved_type if saved_type != "auto" else "unknown")
        return render_template("result.html", r=report)
    except Exception as exc:  # noqa: BLE001 - demo boundary
        log.exception("Analysis failed")
        return render_template(
            "result.html",
            r={"fatal": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()},
        )
    finally:
        temp_path.unlink(missing_ok=True)
        if front_path:
            front_path.unlink(missing_ok=True)


@app.route("/sample")
def sample():
    """Run a built-in synthetic clip through the real pipeline.

    Exists because you need cricket footage shot to the capture protocol before this app can
    show you anything, and that is a lot to ask before you know whether it is worth it. The
    motion here is generated, not filmed - the skeleton is exact and perfectly tracked, which
    is precisely what real footage never is - but every stage after pose estimation is the
    same code path your own clip takes.
    """
    sys.path.insert(0, str(ROOT.parent / "analysis-service" / "tests"))
    import fixtures

    discipline = request.args.get("discipline", "batting")
    clip = (
        fixtures.batting_clip(fps=120.0)
        if discipline == "batting"
        else fixtures.bowling_clip(fps=120.0)
    )
    form_type = "cover_drive" if discipline == "batting" else "fast_arm_bowling"
    seq = clip.seq

    scale = normalisation.estimate_scale(seq)
    detect = (
        phase_detection.detect_batting if discipline == "batting"
        else phase_detection.detect_bowling
    )
    timeline = detect(seq, handedness="right", scale_px=scale.stature_px)

    detected_shot = shot_classifier.classify(
        seq, timeline, scale, discipline=discipline, handedness=handedness, view=view
    )
    report["detected_shot"] = detected_shot.as_dict()

    if auto:
        if detected_shot.form_type:
            form_type = detected_shot.form_type
        else:
            # Features could not separate the shots. Fall back to the discipline's default
            # template rather than refusing outright - the metrics and mechanical checks do
            # not depend on which shot it is, only the reference bands would, and those are
            # empty anyway.
            form_type = "cover_drive" if discipline == "batting" else "fast_arm_bowling"
            report["auto_fell_back"] = True
        label = FORM_TYPES[form_type][0]
        template = templates[form_type]
        report["form_label"] = label + (
            "" if detected_shot.form_type else " (best guess)"
        )

    module = batting_metrics if discipline == "batting" else bowling_metrics
    allowed = {
        s.key for s in metric_tiers.applicable(metric_tiers.Discipline(discipline), fps=seq.fps)
    }
    measurements = module.compute_all(
        seq, timeline, scale, handedness="right", allowed=allowed
    )

    template = template_validator.load_all()[form_type]
    bands = style_resolution.resolve_bands(template, Style.BALANCED)
    scores = scoring.score_all(measurements, bands)
    composite = scoring.composite(scores)

    # Blank frames sized around the figure, so the overlay has something to draw on.
    finite = seq.xy[np.isfinite(seq.xy).all(axis=-1)]
    width = int(np.nanmax(finite[:, 0]) * 1.5) + 60
    height = int(np.nanmax(finite[:, 1]) * 1.2) + 60
    frames = [np.full((height, width, 3), 24, np.uint8) for _ in range(seq.n_frames)]

    report = {
        "form_label": ("Cover drive" if discipline == "batting" else "Fast bowling")
        + " — built-in sample",
        "is_sample": True,
        "fps": seq.fps,
        "frames": seq.n_frames,
        "duration": round(seq.duration_s, 2),
        "mean_confidence": round(float(seq.mean_confidence()), 3),
        "handedness": "right",
        "stature_px": round(scale.stature_px, 1),
        "warnings": [],
        "events": [
            {
                "name": e.name,
                "frame": e.frame,
                "time": round(e.time_s, 3) if e.time_s is not None else None,
                "confidence": round(e.confidence, 2),
                "uncertainty_ms": round(e.time_sem_s * 1000, 1) if e.detected else None,
                "detected": e.detected,
            }
            for e in timeline.events.values()
        ],
        "metrics": _full_metric_list(scores, discipline, "side_on", seq.fps),
        "metrics_measured": sum(
            1 for m in _full_metric_list(scores, discipline, "side_on", seq.fps) if m["measured"]
        ),
        "metrics_total": len(_full_metric_list(scores, discipline, "side_on", seq.fps)),
        "composite": composite.composite,
        "suppressed_reason": composite.suppressed_reason,
        "render": write_render(frames, seq, timeline, 30.0),
    }

    findings = mechanics.analyse(measurements, timeline, discipline)
    report["findings"] = [f.as_dict() for f in findings]
    report["summary"] = mechanics.summarise(findings, measurements, discipline).as_dict()
    report["rating"] = aspects.score(findings, discipline).as_dict()

    recon = reconstruct.compose(seq, scale, timeline)
    report["reconstruction"] = recon.as_dict()
    report["skeleton_edges"] = [list(e) for e in SKELETON]
    return render_template("result.html", r=report)


@app.route("/session/<form_type>")
def session_view(form_type: str):
    """Repeatability across every saved clip of one shot type."""
    import json as _json

    clips, ids = [], []
    for item in list_saved():
        record = load_saved(item["id"])
        if record is None or record.get("form_type") != form_type:
            continue
        metrics = [
            {"key": m["key"], "value": m["value"], "sem": m.get("sem", 0.0),
             "units": m.get("units", "")}
            for m in (record["report"].get("metrics") or [])
            if m.get("measured") and m.get("value") is not None
        ]
        if metrics:
            clips.append(metrics)
            ids.append(item["id"])

    label = FORM_TYPES.get(form_type, (form_type, ""))[0]
    return render_template(
        "session.html",
        form_type=form_type,
        label=label,
        result=consistency.analyse(clips).as_dict(),
        min_clips=consistency.MIN_CLIPS,
    )


@app.route("/saved")
def saved_index():
    return render_template("saved.html", items=list_saved())


@app.route("/saved/<analysis_id>")
def saved_view(analysis_id: str):
    record = load_saved(analysis_id)
    if record is None:
        return render_template(
            "result.html",
            r={"fatal": "That saved analysis no longer exists.",
               "trace": "Saved results live on temporary storage and are cleared periodically."},
        ), 404
    return render_template("result.html", r=record["report"], saved_at=record["saved_at"])


@app.route("/media/<name>")
def media(name: str):
    return send_from_directory(OUTPUT_DIR, name)


@app.route("/health")
def health():
    """Liveness for the platform health check.

    Reports unhealthy when templates fail to validate or the models are missing, because a
    service that accepts uploads and then fails on every one is worse than one that is
    plainly down. Deliberately does not load the models - that would make the first health
    check pay the cold-start cost and time out.
    """
    try:
        templates = template_validator.load_all()
        models_present = (MODELS / "rtmpose.onnx").exists() and (MODELS / "rtmdet.onnx").exists()
        payload = {
            "status": "ok" if models_present else "unhealthy",
            "templates": sorted(templates),
            "models_present": models_present,
            "models_loaded": _backend is not None,
            "max_upload_mb": MAX_UPLOAD_MB,
        }
        if not models_present:
            payload["error"] = (
                "Model weights missing. Run scripts/fetch_models.py during the build."
            )
        return jsonify(payload), 200 if models_present else 503
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "unhealthy", "error": str(exc)}), 503


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5057))
    print(f"\n  Cricket form analysis demo\n  http://localhost:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
