"""Client render payload: the JSON contract between this service and the mobile app.

**Status: contract is real; upload is TODO.**

This is the single definition of the render payload. ``SkeletonRenderer.tsx`` and
``TrajectoryOverlay.tsx`` are written against it, so both sides agree before either is
built.

Per-frame keypoints are large, so the payload goes to Appwrite Storage and only a URL is
stored on the ``analysisResults`` document.

Tier C is filtered here as defence in depth. Scoring and feedback already exclude it, so
anything caught at this layer indicates an upstream bug - hence the assertion rather than a
silent drop.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

import metric_tiers
from phase_detection import PhaseTimeline
from pose_extraction import KEYPOINT_NAMES, PoseSequence
from scoring import MetricScore

#: Bump when the payload shape changes so the client can refuse an incompatible version
#: rather than rendering it wrongly.
RENDER_SCHEMA_VERSION = "1.0.0"

#: Skeleton edges, as index pairs into COCO-17.
SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
)

#: Keypoints below this confidence render dimmed, so the player can see where the analysis
#: was uncertain rather than being shown a clean skeleton over guessed joints.
LOW_CONFIDENCE_RENDER_THRESHOLD = 0.5


@dataclass
class FramePose:
    frame: int
    time_s: float
    #: Flattened ``[x0, y0, x1, y1, ...]`` in normalised coordinates - smaller on the wire
    #: than nested arrays and directly usable by the renderer. ``None`` marks an unobserved
    #: joint, so the renderer breaks the limb rather than drawing it to the frame corner.
    xy: List[Optional[float]]
    confidence: List[float]


@dataclass
class EventMarker:
    name: str
    frame: Optional[int]
    time_s: Optional[float]
    confidence: float
    #: Uncertainty in frames, rendered as a band on the scrubber rather than a hard line.
    #: A player should see that "impact" is an estimate, not a certainty.
    frame_sem: float


@dataclass
class MetricAnnotation:
    key: str
    summary: str
    verdict: str
    tier: str
    #: Present only for Tier A. Tier B is banded and must never show a precise number.
    value: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    band_low: Optional[float]
    band_high: Optional[float]
    units: str
    event: str
    source: Optional[str]
    population: Optional[str]
    evidence_confidence: Optional[str]


@dataclass
class RenderPayload:
    schema_version: str
    clip_id: str
    fps: float
    n_frames: int
    keypoint_names: List[str]
    skeleton_edges: List[List[int]]
    frames: List[FramePose]
    events: List[EventMarker]
    metrics: List[MetricAnnotation]
    low_confidence_threshold: float = LOW_CONFIDENCE_RENDER_THRESHOLD
    reference_frames: List[FramePose] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


def build(
    *,
    clip_id: str,
    seq: PoseSequence,
    timeline: PhaseTimeline,
    scores: Sequence[MetricScore],
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
) -> RenderPayload:
    """Assemble the payload.

    Coordinates are normalised to ``[0, 1]`` so the renderer is resolution-independent.
    Where frame dimensions are not supplied they are inferred from the keypoint extent with
    a margin - adequate for replayed keypoints, where the original video is gone.

    NaN keypoints are serialised as ``null`` rather than zero. A zero would place the joint
    in the corner of the frame and draw a skeleton limb to it; ``null`` lets the renderer
    break the line, which is what an unobserved joint should look like.
    """
    import math

    if frame_width is None or frame_height is None:
        finite = seq.xy[np.isfinite(seq.xy).all(axis=-1)]
        if len(finite):
            frame_width = frame_width or int(np.nanmax(finite[:, 0]) * 1.2) or 1
            frame_height = frame_height or int(np.nanmax(finite[:, 1]) * 1.2) or 1
        else:
            frame_width = frame_width or 1
            frame_height = frame_height or 1

    frames: List[FramePose] = []
    for i in range(seq.n_frames):
        flat: List[Optional[float]] = []
        for k in range(len(KEYPOINT_NAMES)):
            x, y = seq.xy[i, k]
            if math.isnan(x) or math.isnan(y):
                flat.extend([None, None])
            else:
                flat.extend([float(x) / frame_width, float(y) / frame_height])
        frames.append(
            FramePose(
                frame=i,
                time_s=i / seq.fps,
                xy=flat,
                confidence=[float(c) for c in seq.confidence[i]],
            )
        )

    events = [
        EventMarker(
            name=event.name,
            frame=event.frame,
            time_s=event.time_s,
            confidence=event.confidence,
            # Infinite SEM on an undetected event would not survive JSON; zero is correct
            # here because the renderer draws nothing for an undetected event anyway.
            frame_sem=event.frame_sem if event.detected else 0.0,
        )
        for event in timeline.events.values()
        if event.detected
    ]

    return RenderPayload(
        schema_version=RENDER_SCHEMA_VERSION,
        clip_id=clip_id,
        fps=seq.fps,
        n_frames=seq.n_frames,
        keypoint_names=list(KEYPOINT_NAMES),
        skeleton_edges=[list(e) for e in SKELETON_EDGES],
        frames=frames,
        events=events,
        metrics=_annotations(scores),
    )


def _annotations(scores: Sequence[MetricScore]) -> List[MetricAnnotation]:
    """Convert scores to client annotations, enforcing tier display rules.

    Implemented, because this is where the display rules actually bite.
    """
    out: List[MetricAnnotation] = []
    for score in scores:
        spec = metric_tiers.get(score.key)
        if spec.tier is metric_tiers.Tier.C:
            continue
        metric_tiers.assert_displayable(score.key)

        show_value = spec.tier is metric_tiers.Tier.A
        out.append(
            MetricAnnotation(
                key=score.key,
                summary=spec.summary,
                verdict=score.verdict.value,
                tier=spec.tier.value,
                value=score.measurement.value if show_value else None,
                ci_low=score.measurement.ci_low if show_value else None,
                ci_high=score.measurement.ci_high if show_value else None,
                band_low=score.band.low if score.band else None,
                band_high=score.band.high if score.band else None,
                units=score.measurement.units,
                event=spec.event,
                source=score.band.source if score.band else None,
                population=score.band.population if score.band else None,
                evidence_confidence=score.band.confidence if score.band else None,
            )
        )
    return out


def publish(
    *,
    clip_id: str,
    seq: PoseSequence,
    timeline: PhaseTimeline,
    scores: Sequence[MetricScore],
    uploader: Optional[Callable[[str, bytes], str]] = None,
) -> Optional[str]:
    """Build, compress and upload the payload; return its URL.

    ``uploader`` takes ``(object_name, gzipped_bytes)`` and returns a URL. Injected rather
    than importing an Appwrite client here, so the payload format stays testable and this
    module has no storage dependency.

    Gzipped because per-frame keypoints compress extremely well - typically to under a tenth
    of their JSON size - and bandwidth is metered on both Render and Appwrite.

    Returns ``None`` when no uploader is configured. The analysis result is still valid
    without a render payload; the client simply cannot draw the skeleton overlay.
    """
    import gzip

    payload = build(clip_id=clip_id, seq=seq, timeline=timeline, scores=scores)
    if uploader is None:
        return None

    blob = gzip.compress(payload.to_json().encode("utf-8"))
    return uploader(f"render/{clip_id}.json.gz", blob)
