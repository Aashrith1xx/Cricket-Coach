"""Per-clip shot-intent classification. Advisory output only.

**Status: specified stub.**

Reproduces the approach of ``intent2025posture``: a small 1D CNN over per-frame joint
coordinates, classifying a shot as aggressive or defensive intent. Reported performance is
77% accuracy / 0.83 AUC / 0.77 F1 (leave-one-out over 11 matches), 74% / 0.76 / 0.73 on a
held-out batter.

**This module must never select a reference template.** Its output goes to
:mod:`style_resolution` as corroboration for the player's *declared* style. See that
module's docstring for the four reasons - the short version is that at ~1-in-4 error, gating
grading on it would silently mis-grade a quarter of users.

Training data note: the published labels were weakly supervised by runs scored (>=3
aggressive, <=1 defensive), which conflates intent with outcome. If we ever train our own
model on collected clips, that flaw should not be reproduced - a coach-labelled intent set,
even a small one, would be better ground truth than a large runs-derived one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from pose_extraction import PoseSequence
from style_resolution import Intent, IntentEstimate

#: Joints used as model input, matching intent2025posture. Note the absence of any bat
#: information - the classifier works from body posture alone, which is what makes it
#: applicable here at all.
FEATURE_JOINTS: tuple[str, ...] = (
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "nose",
)

#: Frames the model expects, resampled from the detected shot window. Fixed length so clips
#: at different frame rates and durations present identically.
SEQUENCE_LENGTH = 64

MODEL_PATH = "models/intent_1dcnn.onnx"


@dataclass
class IntentModelInfo:
    """Provenance, surfaced in the UI so the estimate is never mistaken for ground truth."""

    name: str = "intent-1dcnn"
    version: str = "0.0.0-unimplemented"
    reported_accuracy: float = 0.77
    reported_auc: float = 0.83
    trained_on: str = "broadcast footage, elite players (intent2025posture)"
    caveat: str = (
        "Estimates the intent of a single shot, not a player's style. Trained on elite "
        "broadcast footage with labels derived from runs scored, so accuracy on amateur "
        "phone clips will be lower than the published figures."
    )


def build_features(seq: PoseSequence, start: int, end: int) -> np.ndarray:
    """Assemble the model input tensor for one shot window.

    TODO. Planned steps:

    1. Slice the pose sequence to ``[start, end)`` - the shot window from phase detection.
    2. Translate so the pelvis midpoint is the origin, and scale by stature, so the model
       sees posture rather than the player's position and size in frame. This mirrors the
       normalisation applied elsewhere in the service.
    3. Resample to ``SEQUENCE_LENGTH`` frames by linear interpolation along time.
    4. Stack to ``(SEQUENCE_LENGTH, len(FEATURE_JOINTS) * 2)``.

    Confidence values are deliberately *not* included as input features. Doing so would let
    the model learn to associate poor tracking with a class, which would make it fail in a
    correlated way exactly where the rest of the pipeline is already weakest.
    """
    raise NotImplementedError("style_classifier.build_features is not implemented")


def classify(seq: PoseSequence, start: int, end: int) -> IntentEstimate:
    """Estimate intent for one shot.

    TODO. Load the ONNX model once at worker start, not per clip.

    The returned confidence must be a calibrated probability, not a raw softmax score.
    :mod:`style_resolution` drops estimates below ``MIN_INTENT_CONFIDENCE``, and that filter
    is meaningless if the score is not calibrated - an uncalibrated network will happily
    report 0.95 on inputs it has never seen anything like.
    """
    raise NotImplementedError("style_classifier.classify is not implemented")


def classify_session(
    sequences: Sequence[tuple[PoseSequence, int, int]],
) -> list[IntentEstimate]:
    return [classify(seq, s, e) for seq, s, e in sequences]
