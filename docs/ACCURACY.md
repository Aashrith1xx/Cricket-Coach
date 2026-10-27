# Measurement accuracy: what this app can and cannot grade

This document is the evidence behind `analysis-service/metric_tiers.py`. If you want to move
a metric between tiers, the bar is a citation, not an intuition.

---

## 1. The governing constraint

Markerless pose estimation error is **not uniform across joints**. It is small for
in-image-plane lower-body and trunk angles, and large for transverse-plane rotation and
distal upper-limb segments. Any design that treats all joint angles as equally trustworthy
will produce confident, wrong feedback.

Abraham, Feros & Fox (2025), cricket bowling, three synchronised iPads at 240 Hz, OpenCap vs
marker-based mocap:

| Variable | RMSE | Verdict here |
| --- | --- | --- |
| Knee flexion | 7.87° ± 2.10° | gradeable |
| Trunk lateral flexion | 9.10° ± 2.38° | gradeable |
| Elbow flexion | 22.71° ± 7.31° | not gradeable |
| Shoulder axial rotation | 28.54° ± 7.86° | not gradeable |
| Arm elevation plane | 28.92° ± 5.32° | not gradeable |
| Average, all joints | 17.61° ± 7.72° | — |

Additional findings from the same study:

- **56% trial success rate.** 44% of trials failed outright. This is why `capture_gate.py`
  rejects rather than degrades.
- **Systematic underestimation** of shoulder and elbow angles as the bowler turns side-on —
  the error is directional, not just noisy, so averaging more clips does not remove it.
- **Reversed ankle dorsiflexion/plantarflexion** identification at high movement speed.
- Ball-to-ball variability was large even where trial means looked reasonable, so the system
  was unreliable for detecting subtle technique variation — which is precisely what a form
  coach cares about.

Our situation is **worse than this study's** in every respect: one camera instead of three,
consumer phone optics, uncontrolled lighting and background, and an unconstrained camera
position unless the capture protocol is enforced.

## 2. Monocular limits

Depth from a single camera is ill-posed: multiple distinct 3D poses project to the same 2D
image. The practical rule from the literature is that **angles of interest must lie in the
image plane**. This is why the capture protocol specifies camera position per form type, and
why it is enforced rather than suggested — it is the highest-leverage accuracy component in
the entire system, ahead of model choice.

Consequence: a sagittal-plane knee angle filmed side-on is measurable. The same joint filmed
from 45° off-axis is not, and no amount of smoothing recovers it.

## 3. Frame rate

| Frame rate | Frame interval | Consequence |
| --- | --- | --- |
| 30 fps | 33 ms | Arm-horizontal-to-release spans 1–3 frames. Bat–ball contact invisible. Unusable. |
| 60 fps | 17 ms | Still inadequate for release and impact events. |
| 120 fps | 8 ms | Minimum viable floor. |
| 240 fps | 4 ms | Preferred; matches the OpenCap study's capture rate. |
| 250 Hz | 4 ms | Research standard for cricket biomechanics. |

Below 120 fps, release- and impact-referenced metrics are **suppressed, not estimated**.

## 4. What is reliable: temporal metrics

Event timing is the most robust signal available from video:

- Video-based gait event detection: median absolute error **~24–25 ms** (foot-down / foot-up),
  90th percentile 79 ms.
- Pose-based athletics event detection: **92.3% / 89.0%** precision/recall at ±1 frame at
  200 Hz.
- OpenPose vs mocap toe-off: mean absolute difference up to 3 frames (30 ms).

This is why `scoring.py` gives temporal metrics a higher base weight than angles. It inverts
the intuitive design, and it follows directly from the error figures.

---

## 5. Tier assignments

### Tier A — scored numerically, with confidence intervals

In-image-plane lower-body/trunk angles, and all temporal metrics.

| Metric | Basis |
| --- | --- |
| Front/lead knee angle at key events | 7.87° RMSE, sagittal plane |
| Knee angle change between events | Differences cancel part of the systematic bias |
| Trunk **forward** flexion | Sagittal plane, so in the image plane on the specified side-on camera. Trunk angles measured at 9.10° RMSE |
| Phase event timings and intervals | ~24–25 ms error |
| Tempo ratios between phases | Ratio of two timings; scale-free |
| Head displacement through downswing→impact | Head is the best-tracked region: large, high-contrast, rarely self-occluded |
| Trigger movement presence/timing/magnitude | Temporal + gross displacement |
| Base length at impact as % stature | Already normalised in the source literature |

### Tier B — banded categories only, never a precise number

Wide error, or requires a second camera angle.

| Metric | Basis |
| --- | --- |
| Stride length (normalised to stature) | Depth-sensitive; also *not* "longer is better" — see §6 |
| Head position over front knee | Meaningful, but sensitive to camera obliquity |
| Pelvis–thorax separation (X-factor) | Transverse plane. **Suppressed entirely on a single side-on clip**; available only with the optional front-on camera |
| Lower-body → trunk sequence ordering | Ordering is more robust than magnitude, but still noisy |
| Trunk **lateral** flexion | Frontal-plane motion, so on a side-on camera it points toward and away from the lens and is foreshortened. Requires the front-on angle |

**Correction, made during implementation.** Trunk lateral flexion was initially placed in
Tier A and forward flexion in Tier B, reading the 9.10° RMSE figure straight off the OpenCap
study. That was wrong for *our* camera: the study used three cameras, and which trunk angle
is measurable depends on where the camera stands. On the single side-on view this app
specifies, the sagittal (forward) angle is in the image plane and the frontal (lateral) one
is not. The two have been swapped. A published RMSE is a property of a capture geometry, not
of a joint.

### Tier C — captured, never displayed, never scored

Error exceeds the effect size being measured.

| Metric | Basis |
| --- | --- |
| Elbow extension | 22.71° RMSE vs ICC's 15° threshold |
| Shoulder axial rotation | 28.54° RMSE |
| Arm elevation plane | 28.92° RMSE |
| Shoulder counter-rotation | Derived from shoulder axial rotation; the 30° risk threshold is *smaller than the measurement error* |
| Bowling action type classification | Depends on shoulder alignment |
| Wrist uncocking | More distal and more occluded than the elbow; hidden behind hands and bat |
| Bat angle, bat speed, impact location | The bat is not a body segment — pose models do not track it at all (§7) |

Tier C metrics are still computed and stored, so that if capture hardware improves the data
is not lost. They are excluded from scores, feedback text and progress charts by a hard
suppression list, not by convention.

---

## 6. Findings that change what "good" means

These matter as much as the error figures, because they determine whether a *correctly
measured* metric is being graded in the right direction.

**Backlift direction.** 77% of successful batsmen and 90% of IPL batsmen use a *lateral*
backlift (toward second slip, face toward off-side); 83% of coaches teach the *straight*
backlift. Over 70% of *uncoached* players naturally adopt lateral; over 70% of *coached*
players adopt straight. A template encoding "straight = correct" would penalise the technique
most elite players use. → **Classified and reported, never scored.**

**Trigger movement is format-dependent.** Prevalence: Test 56.1%, ODI 47.4%, T20I 73.3%,
IPL 76%. → Detect and report; absence is not a fault.

**Longer stride is not better.** No significant stride-length difference across senior / U19 /
U16 (0.77 / 0.85 / 0.80 m), and experienced batters deliberately take a short-to-medium
stride *so they can play the ball later*. → Banded with an optimum range, never "more is
better".

**Some variables do not discriminate skill at all.** No significant senior/U19/U16 difference
in bat angle, shoulder or elbow angles at any phase of the pull shot; no skilled vs
less-skilled difference in shoulder angle, bat angle or bat speed in the delivery-methods
study. → Marked `discriminates_skill: false`; zero score weight. Grading them adds noise, not
signal.

**Where the pull shot *did* discriminate:** right knee extension at follow-through (senior
139.97 ± 12.81° vs U16 120.77 ± 5.81°, p<.03) and left hip angle at impact (vs U19
134.21 ± 12.71°, p<.04) — both sagittal-plane lower-body angles, squarely in the measurable
tier. Caveat: small single-squad sample, authors' own angular reliability 3.1–10.7°, so it is
tagged **low confidence**.

**Head–ball coupling is the strongest measurable batting discriminator.** Elite vs club mean
head–ball angle 2.4 ± 1.7° vs 3.7 ± 1.4° (p=.003); maximum 10.7 ± 8.7° vs 20.6 ± 8.7°
(p<.0001). Skill signal and measurability coincide here, which is rare. Caveat: true
head–ball angle needs ball tracking, which is unreliable on phone video, so we measure the
head-only components — rotation continuity through contact, head lift at impact, and head
position over the front knee.

**Gaze is not measurable.** Elite batters hold gaze ~−3.4° ahead of the ball and make two
predictive saccades where club batters make one, but this requires an eye tracker. Out of
scope entirely — do not attempt to infer it from head pose.

---

## 7. Bat speed

Reference values, for context only:

| Source | Population | Bat speed |
| --- | --- | --- |
| Bat-sensor validation (Cricket NSW) | 9 professional female, 4 shot types | 52.8–87.9 km/h (14.7–24.4 m/s) |
| Peploe et al. 2019 | international→club male, range hitting | max 27.0 ± 2.3 m/s (~97 km/h) |

**It cannot be measured here.** Every validated method is something else: marker-based mocap,
a bat-mounted IMU, or multi-camera markerless. Even the dedicated bat-mounted sensor shows
1.0–2.5 km/h bias (1.4–3.9%) and 2.7–4.4 km/h precision against mocap, with proportional
error *increasing* at higher bat speeds. The bat is not a body segment, so pose models do not
track it at all; it would need a separate object tracker fighting motion blur, foreshortening
through the swing arc, and occlusion by the body.

**"Ideal bat speed" is also the wrong frame.** Bat speed does not discriminate skill in
controlled shots, and impact location plus bat speed together explain 68% of ball launch
speed variation — a slower centred strike beats a faster edge.

The honest future path is not better vision: it is optionally ingesting data from a bat
sensor the player already owns (str8bat, StanceBeam), which would make bat speed a Tier A
*sensor* metric rather than a vision metric. Not built.

---

## 8. Power-hitting mechanics

Peploe et al. (2019) stepwise regression on maximum bat speed:

| Model | Cumulative variance explained |
| --- | --- |
| X-factor DS (pelvis–thorax transverse separation at downswing start) | 28.0% (SEE 1.65 m/s) |
| + lead elbow extension (DS→IMP) | 58.6% |
| + wrist uncocking (DS→IMP) | 77.7% (SEE 0.92 m/s) |

All three are transverse-plane or distal upper-limb variables — the worst-measured
categories. **Power-hitting mechanics are therefore not gradeable from a single camera.**
This is the batting analogue of the bowling-legality problem, and the main reason the batting
metric set is smaller than a coaching manual would suggest.

---

## 9. Playstyle classification

Posture-based aggressive/defensive intent classification from joint coordinates:

| Model / split | Accuracy | AUC-ROC | F1 |
| --- | --- | --- | --- |
| 1D CNN, leave-one-out (11 matches) | 77% | 0.83 | 0.77 |
| 2s-AGCN, leave-one-out | 75% | 0.85 | 0.75 |
| 1D CNN, held-out single batter | 74% | 0.76 | 0.73 |

Three reasons this is a **cross-check, not a gate**:

1. It classifies per-shot *intent*, not player *style*. Style is the distribution of intents
   across many deliveries and must be aggregated over a history.
2. Labels were outcome-contaminated — aggressive/defensive was proxied by runs scored (≥3 vs
   ≤1), so a well-executed attacking shot that is fielded well is labelled defensive. The
   published accuracy is against noisy ground truth.
3. At 74–77%, roughly one clip in four is misclassified. If style silently selected the
   grading reference, a quarter of users would get wrong feedback with no signal it happened.
   It was also trained on broadcast footage of elite players, not amateur phone clips, so
   real-world accuracy will be lower.

Hence: the player **declares** style; the classifier corroborates and surfaces divergence as
an observation.

**No study establishes which technique variants map to which style.** Lateral backlift being
90% in IPL vs 77% among successful batsmen generally is suggestive of a format/aggression
association, not evidence of a definitional link. Style-variant ranges therefore ship as TODO
stubs pending real sources.
