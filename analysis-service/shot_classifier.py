"""Identify which shot or delivery a clip contains, from its kinematics.

Why not the published approach
------------------------------
The best reported cricket shot classifiers reach 89-93% across ten shot types
(``cricshotclassify``), but they are appearance-based CNN-GRU hybrids - VGG16 or
EfficientNetV2 over raw frames - trained on the CricShot10 dataset. Reproducing that needs
that dataset and the compute to train on it, neither of which this project has. Claiming
their accuracy for an untrained reimplementation would be borrowing a number that was never
earned here.

So this classifies from the kinematics the pipeline **already computes**: where the feet go,
which way the bat travels, how fast the hands move. Those features are the actual physical
definitions of the shots - a pull is a back-foot horizontal-bat stroke, and that is a
statement about geometry rather than about pixels - so a feature-based classifier is
defensible even without training data. It is also inspectable: every decision comes with the
numbers behind it, which a fine-tuned VGG16 could not offer.

The cost is accuracy. This will not match 93%, and its output is therefore a **suggestion**,
pre-selected but overridable, exactly as with playstyle and camera view. The player knows
which shot they played; the classifier is guessing from geometry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from normalisation import BodyScale, centre_of_mass
from phase_detection import PhaseTimeline
from pose_extraction import KEYPOINT_INDEX, PoseSequence, to_lead_rear
from smoothing import condition

log = logging.getLogger(__name__)

#: Lead-ankle travel toward the bowler, in stature units, above which the shot is played off
#: the front foot. Below it the weight stayed back.
FRONT_FOOT_STRIDE = 0.06

#: Ratio of vertical to horizontal hand travel through the downswing. A vertical-bat shot -
#: drive, defence - swings down the line of the ball; a horizontal-bat shot - pull, cut -
#: swings across it. This is the defining geometric difference between the families.
VERTICAL_BAT_RATIO = 1.0

#: Peak hand speed, stature per second, separating an attacking stroke from a defensive one.
ATTACKING_HAND_SPEED = 2.5

#: Below this the classifier declines to name a shot rather than guessing between two that
#: its features cannot separate.
#:
#: Must exceed 0.5 for a two-way choice: at exactly 0.5 the features are split evenly, which
#: is precisely the case where naming a shot is least justified. A threshold below 0.5 lets a
#: dead tie through as a confident answer.
MIN_CONFIDENCE = 0.6


@dataclass
class ShotFeatures:
    """The kinematic quantities the decision rests on. Surfaced so a call can be checked."""

    stride_forward: Optional[float] = None
    weight_shift: Optional[float] = None
    vertical_ratio: Optional[float] = None
    peak_hand_speed: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "stride_forward": _round(self.stride_forward),
            "weight_shift": _round(self.weight_shift),
            "vertical_ratio": _round(self.vertical_ratio),
            "peak_hand_speed": _round(self.peak_hand_speed),
        }


def _round(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(float(v), 3)


@dataclass
class Classification:
    form_type: Optional[str]
    label: str
    confidence: float
    reasoning: List[str] = field(default_factory=list)
    features: ShotFeatures = field(default_factory=ShotFeatures)
    alternatives: List[Tuple[str, float]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "form_type": self.form_type,
            "label": self.label,
            "confidence": round(self.confidence, 2),
            "reasoning": self.reasoning,
            "features": self.features.as_dict(),
            "alternatives": [
                {"form_type": k, "confidence": round(c, 2)} for k, c in self.alternatives
            ],
        }


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _facing_sign(
    seq: PoseSequence, timeline: PhaseTimeline, handedness: str
) -> Optional[int]:
    """+1 if the batter faces increasing x, -1 if decreasing, ``None`` if undeterminable.

    Taken from the stance: the lead foot is the one nearer the bowler, so the sign of
    (lead ankle - rear ankle) along x is the direction the player is playing toward.
    """
    frame = timeline.frame_of("STANCE") or 0
    lead = seq.xy[frame, KEYPOINT_INDEX[to_lead_rear("lead_ankle", handedness)]]
    rear = seq.xy[frame, KEYPOINT_INDEX[to_lead_rear("rear_ankle", handedness)]]
    if np.isnan(lead).any() or np.isnan(rear).any():
        return None

    separation = float(lead[0] - rear[0])
    # Feet nearly level: the stance does not resolve the axis, and guessing its sign would
    # flip the front-foot test at random.
    if abs(separation) < 1e-3:
        return None
    return 1 if separation > 0 else -1


def extract_features(
    seq: PoseSequence,
    timeline: PhaseTimeline,
    scale: BodyScale,
    *,
    handedness: str,
    view: str = "side_on",
) -> ShotFeatures:
    """Measure the quantities that physically define the shot families.

    **Front-foot versus back-foot cannot be measured from a front-on camera.** The stride
    runs toward or away from the lens, so it is foreshortened to near nothing - the same
    in-image-plane constraint that governs every other measurement here. Those features are
    therefore left unmeasured rather than read off a foreshortened axis, which is what made a
    front-on cover drive classify as a pull shot: the stride looked like it went nowhere,
    because from that angle it does.
    """
    front_on = view == "front_on"
    features = ShotFeatures()
    lead_ankle = to_lead_rear("lead_ankle", handedness)
    lead_wrist = to_lead_rear("lead_wrist", handedness)

    # Which way is "forward"? A batter can face either way depending on which side the
    # camera is, so raw +x is meaningless. The lead foot sits nearer the bowler than the rear
    # foot, and that difference defines the axis. Assuming +x made every front-foot shot read
    # as back-foot whenever the camera was on the other side - the same class of error as
    # assuming a fixed shoulder-line orientation.
    facing = _facing_sign(seq, timeline, handedness)

    ss, se = timeline.frame_of("SS"), timeline.frame_of("SE")
    if ss is not None and se is not None and facing is not None and not front_on:
        ankle = seq.keypoint(lead_ankle)
        if np.isfinite(ankle[ss]).all() and np.isfinite(ankle[se]).all():
            features.stride_forward = (
                facing * float(ankle[se, 0] - ankle[ss, 0]) / scale.stature_px
            )

    stance, imp = timeline.frame_of("STANCE"), timeline.frame_of("IMP")
    if stance is not None and imp is not None and facing is not None and not front_on:
        a, b = centre_of_mass(seq, stance), centre_of_mass(seq, imp)
        if np.isfinite(a).all() and np.isfinite(b).all():
            features.weight_shift = facing * float(b[0] - a[0]) / scale.stature_px

    ds = timeline.frame_of("DS")
    if ds is not None and imp is not None and imp > ds:
        wrist = condition(
            seq.keypoint(lead_wrist),
            seq.keypoint_confidence(lead_wrist),
            seq.fps,
            scale_px=scale.stature_px,
        ).xy
        span = wrist[ds:imp + 1]
        if len(span) >= 2 and np.isfinite(span).all():
            horizontal = float(np.ptp(span[:, 0]))
            vertical = float(np.ptp(span[:, 1]))
            if horizontal > 1e-6:
                features.vertical_ratio = vertical / horizontal

            steps = np.linalg.norm(np.diff(span, axis=0), axis=1)
            features.peak_hand_speed = (
                float(np.percentile(steps, 90)) * seq.fps / scale.stature_px
            )

    return features


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_batting(features: ShotFeatures) -> Classification:
    """Place a batting clip among the shot templates that exist.

    A two-axis split, because those two axes are what actually define the shot families:
    which foot the weight is on, and whether the bat swings down the line of the ball or
    across it. Everything else is refinement.
    """
    reasoning: List[str] = []
    scores: Dict[str, float] = {"cover_drive": 0.0, "pull_shot": 0.0}

    # Axis 1 - front foot or back foot.
    if features.stride_forward is not None:
        if features.stride_forward > FRONT_FOOT_STRIDE:
            scores["cover_drive"] += 0.5
            reasoning.append(
                f"Front foot moved {features.stride_forward:.2f} of your height toward the "
                f"bowler, so this is a front-foot shot."
            )
        else:
            scores["pull_shot"] += 0.5
            reasoning.append(
                f"Front foot barely advanced ({features.stride_forward:+.2f}), so the weight "
                f"stayed back."
            )

    # Axis 2 - vertical bat or horizontal bat.
    if features.vertical_ratio is not None:
        if features.vertical_ratio > VERTICAL_BAT_RATIO:
            scores["cover_drive"] += 0.5
            reasoning.append(
                f"The hands travelled mostly downward (ratio {features.vertical_ratio:.1f}), "
                f"which is a vertical-bat shot."
            )
        else:
            scores["pull_shot"] += 0.5
            reasoning.append(
                f"The hands travelled mostly across the body (ratio "
                f"{features.vertical_ratio:.1f}), which is a horizontal-bat shot."
            )

    return _finalise(scores, reasoning, features, {
        "cover_drive": "Cover drive", "pull_shot": "Pull shot",
    })


def classify_bowling(features: ShotFeatures, timeline: PhaseTimeline) -> Classification:
    """Separate pace from spin.

    The honest discriminator available from a single clip is run-up speed: a fast bowler
    covers ground between back-foot and front-foot contact that a spinner does not. Grip and
    wrist position would be far better and are not measurable here - the hands are the least
    reliably tracked part of the skeleton, and the ball is not tracked at all.
    """
    reasoning: List[str] = []
    scores: Dict[str, float] = {"fast_arm_bowling": 0.0, "spin_bowling": 0.0}

    delivery = timeline.interval_s("BFC", "FFC")
    if delivery is not None:
        # A quick delivery stride means pace. The threshold is a working assumption rather
        # than a sourced value, which is one reason this stays a suggestion.
        if delivery < 0.20:
            scores["fast_arm_bowling"] += 1.0
            reasoning.append(f"Delivery stride took {delivery * 1000:.0f} ms - quick, so pace.")
        else:
            scores["spin_bowling"] += 1.0
            reasoning.append(
                f"Delivery stride took {delivery * 1000:.0f} ms - slower, so more likely spin."
            )

    return _finalise(scores, reasoning, features, {
        "fast_arm_bowling": "Fast bowling", "spin_bowling": "Spin bowling",
    })


def _finalise(
    scores: Dict[str, float],
    reasoning: List[str],
    features: ShotFeatures,
    labels: Dict[str, str],
) -> Classification:
    total = sum(scores.values())
    if total <= 0:
        return Classification(
            None, "Couldn't tell", 0.0,
            ["Not enough of the shot was measurable to identify it."], features,
        )

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best, best_score = ranked[0]
    confidence = best_score / total

    if confidence < MIN_CONFIDENCE:
        return Classification(
            None, "Couldn't tell", confidence,
            reasoning + ["The features point both ways, so this is not a call worth making."],
            features, ranked[1:],
        )

    return Classification(best, labels[best], confidence, reasoning, features, ranked[1:])


def classify(
    seq: PoseSequence,
    timeline: PhaseTimeline,
    scale: BodyScale,
    *,
    discipline: str,
    handedness: str,
    view: str = "side_on",
) -> Classification:
    features = extract_features(
        seq, timeline, scale, handedness=handedness, view=view
    )
    if discipline == "batting":
        return classify_batting(features)
    return classify_bowling(features, timeline)


# ---------------------------------------------------------------------------
# Discipline: batting or bowling
# ---------------------------------------------------------------------------

#: Whole-body travel across the clip, in stature units, above which the player is running.
#: A bowler covers ground; a batter stays roughly where they started. This is by far the
#: largest kinematic difference between the two, which is why it can be read before any
#: discipline-specific event detector has run - and it has to be, because choosing that
#: detector is the decision being made.
RUNNING_TRAVEL = 0.9


def detect_discipline(seq: PoseSequence, scale: BodyScale) -> Tuple[str, float, str]:
    """Decide whether a clip shows batting or bowling, before any events are known.

    Returns ``(discipline, confidence, reasoning)``.

    Requires **two** signals to call bowling, because travel alone is not enough: a batter
    who walks between deliveries, or a tracker that switches to someone else in the nets,
    covers just as much ground as a run-up. A long clip of a flick was classified as spin
    bowling for exactly that reason.

    The second signal is the bowling arm passing **above the head**. Every delivery has it
    and no batting shot does, so requiring both makes walking harmless. Where they disagree
    the answer is batting - it is the commoner case, and the player can override.
    """
    hips = [KEYPOINT_INDEX["left_hip"], KEYPOINT_INDEX["right_hip"]]
    centre = np.nanmean(seq.xy[:, hips, 0], axis=1)
    centre = centre[np.isfinite(centre)]
    if len(centre) < 5:
        return "batting", 0.0, "Not enough tracking to tell; assuming batting."

    travel = float(np.nanmax(centre) - np.nanmin(centre)) / scale.stature_px
    travels = travel >= RUNNING_TRAVEL

    # Arm over the head. Image y grows downward, so "above" means a smaller y than the head.
    head_y = seq.xy[:, KEYPOINT_INDEX["nose"], 1]
    arm_over = False
    for side in ("left_wrist", "right_wrist"):
        wrist_y = seq.xy[:, KEYPOINT_INDEX[side], 1]
        conf = seq.confidence[:, KEYPOINT_INDEX[side]]
        usable = np.isfinite(wrist_y) & np.isfinite(head_y) & (conf > 0.4)
        if usable.sum() < 3:
            continue
        # Clearly over the top, not merely level with the head.
        if np.any(wrist_y[usable] < head_y[usable] - 0.08 * scale.stature_px):
            arm_over = True
            break

    if travels and arm_over:
        return "bowling", min(1.0, travel / (RUNNING_TRAVEL * 2)), (
            f"The player travelled {travel:.1f} body-lengths and the arm passed over the "
            f"head, so this is a run-up and delivery."
        )

    if travels and not arm_over:
        return "batting", 0.55, (
            f"There was {travel:.1f} body-lengths of movement, but the arm never went over "
            f"the head - so this looks like a batter moving about rather than a delivery."
        )

    return "batting", min(1.0, 1.0 - travel / RUNNING_TRAVEL), (
        f"The player stayed roughly in place ({travel:.1f} body-lengths) and the arm never "
        f"went overhead, so this is a shot."
    )
