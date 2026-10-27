# Cricket Form Analysis

Upload a clip of a batting shot or a bowling delivery, get form analysis with constructive
feedback, and track improvement across sessions.

**Status: pipeline implemented end to end, unscored pending reference data.**

The full analysis path runs: video decode → capture gate → pose → smoothing → scale →
phase detection → metrics → style resolution → scoring → feedback → render payload →
retention. 137 Python tests and 16 Node tests pass.

Two things are outstanding, and they are different in kind:

1. **Model weights.** `pose_backends.OnnxPoseBackend` is written but needs the RTMDet and
   RTMPose ONNX exports in `analysis-service/models/`. Everything downstream is exercised
   through `ReplayPoseBackend` and `ArrayPoseBackend`, so the pipeline is testable today.
2. **Reference ranges.** Only 2 of 12 batting metrics and 0 of 9 bowling metrics have
   sourced expected bands, so **the app currently returns no overall score** — by design.
   See §7. This is the honest state, not an incomplete one: fabricated bands would produce
   feedback indistinguishable from correct feedback.

Measured event-detection accuracy on synthetic fixtures is 17–25 ms worst case, comparable
to the ~24–25 ms reported for video-based event detection in the literature. That validates
the algorithms, not the system — see `analysis-service/validation/`.

---

## 1. Read this before changing any metric

The accuracy ceiling of this product is set by **capture conditions and by which joint is
being measured** — not by the choice of pose model. Designing as though every joint angle is
equally trustworthy is the main way this product fails.

Abraham, Feros & Fox (2025) validated OpenCap markerless motion capture against marker-based
mocap **specifically on cricket bowling**, using three synchronised iPads at 240 Hz:

| Variable | RMSE |
| --- | --- |
| Knee flexion | 7.87° ± 2.10° |
| Trunk lateral flexion | 9.10° ± 2.38° |
| Elbow flexion | 22.71° ± 7.31° |
| Shoulder axial rotation | 28.54° ± 7.86° |
| Arm elevation plane | 28.92° ± 5.32° |
| *Average across all joints* | *17.61° ± 7.72°* |

Only **56%** of OpenCap trials succeeded. Shoulder and elbow were systematically
underestimated by occlusion as bowlers turn side-on. The authors concluded OpenCap is
"presently inappropriate" for bowling legality testing.

A single phone camera performs **worse** than that three-camera 240 Hz setup.

### Consequences that are not negotiable

1. **Bowling action legality is out of scope.** The ICC threshold is 15° of elbow extension;
   measured elbow error is 22.7°. The app must never comment on whether an action is legal.
2. **Mixed-action injury screening is out of scope.** The risk threshold is shoulder
   counter-rotation >30°, derived from shoulder axial rotation whose error is 28.5° — larger
   than the threshold itself. Telling a bowler their action is "safe" on that basis could
   contribute to a lumbar stress fracture. This is a safety refusal, not a scoping preference.
3. **Bat speed and power-hitting mechanics are out of scope.** The three strongest predictors
   of bat speed (pelvis–thorax separation, lead elbow extension, wrist uncocking) are all in
   the worst-measured categories. See `docs/ACCURACY.md`.
4. **Lower-body and trunk metrics in the image plane are defensible** (~8–9° RMSE).
5. **Temporal metrics are the most robust signal available** — video foot-contact detection
   shows ~24–25 ms median absolute error. Timing therefore carries the most grading weight,
   which inverts the naive "angles first" design.

If you are about to re-enable a Tier C metric because it "seems like it should work", read
`docs/ACCURACY.md` first. The tier list in `analysis-service/metric_tiers.py` is enforced in
code, not by convention.

---

## 2. Architecture

```
cricket-form-app/
├── mobile/            React Native (Expo) — guided capture, results, progress
├── server/            Node/Express + Clerk + Appwrite — API, storage, progress stats
└── analysis-service/  Python + RTMDet/RTMPose — CPU-only, async job worker
```

Request flow:

```
mobile ──upload──► server ──enqueue──► analysis-service
                     │                       │
                     │                  gate → pose → smooth → events
                     │                       → metrics → style → score
                     │                       → feedback → render data
                     │                       → retention
                     ◄────── results ────────┘
```

Analysis is **asynchronous**. A clip takes an estimated 1–2 minutes on a fractional-vCPU
instance, and Render free-tier services spin down after 15 minutes idle, so a blocking
request-response design would lose in-flight jobs.

### Pose stack and a licensing trap

- **Detector:** RTMDet (Apache 2.0). **Pose:** RTMPose-m (Apache 2.0, 75.8% AP COCO).
- RTMPose is top-down and *requires* a person detector. Pair it **only with RTMDet**.
  Do **not** substitute an Ultralytics YOLOv5/v8 detector — those are AGPL-3.0 and would
  require open-sourcing this application or buying a commercial licence.

---

## 3. Capture protocol

Capture quality dominates everything downstream, because angles must lie **in the camera's
image plane** to be measurable at all. The app enforces this rather than advising it.

| | Bowling | Batting (front-foot shots) |
| --- | --- | --- |
| Camera position | Side-on, perpendicular to the crease | Side-on to the batter |
| Distance | ~10–15 m | ~6–10 m |
| Height | ~1.5 m | ~1.5 m |
| Mount | Tripod or steady rest — handheld is rejected | Same |
| Frame rate | 120 fps minimum, 240 fps preferred | 120 fps minimum, 240 fps preferred |
| Optional second angle | — | Front-on, unlocks Tier B pelvis–thorax metrics |

**Why 120 fps is a hard floor:** at 30 fps (33 ms/frame) the entire arm-horizontal-to-release
window is 1–3 frames, and bat–ball contact is a sub-frame event. Below 120 fps, release- and
impact-referenced metrics are suppressed rather than estimated.

Clips failing the gate (`analysis-service/capture_gate.py`) are **rejected with a specific
reason**, never analysed at reduced confidence. The OpenCap study lost 44% of trials;
silently analysing a bad clip is exactly how a user receives confidently wrong feedback.

---

## 4. How grading works

1. **Gate** — reject non-conforming clips with an actionable reason.
2. **Pose** — RTMDet → RTMPose, per-keypoint confidence retained.
3. **Smooth** — confidence weighting, gap interpolation, Savitzky–Golay before any
   differentiation. Never compute angles from raw frame-to-frame coordinates.
4. **Normalise** — all distances in body-segment units, so metrics are invariant to camera
   distance and player height.
5. **Events** — detect phase events, each with frame-uncertainty derived from fps.
6. **Metrics** — each emits `{value, ci_low, ci_high, tier, convention, confidence}`.
7. **Style** — the player's *declared* style selects the template variant. The pose-based
   intent classifier only cross-checks; it never silently reroutes grading (see §5).
8. **Score** — value normalised into the template's angle convention, then:
   - CI entirely inside the expected band → full marks
   - CI entirely outside → scored by distance, scaled by band width
   - CI **straddling** a boundary → reported *unclear*, not pass/fail, at reduced weight
9. **Weight** — `base_weight × confidence × discriminates_skill`. Temporal metrics outweigh
   angles. Tier B is banded and reduced. **Tier C is hard-zero.**
10. **Composite** — weighted mean with propagated CI. If surviving weight falls below
    threshold, **no composite score is shown at all** rather than fabricating one.
11. **Feedback** — templated text; forbidden from emitting bare numbers, Tier C metrics, or
    injury/legality claims.
12. **Progress** — session aggregation and minimal-detectable-change gating (§6).

---

## 5. Playstyle

Grading everyone against one orthodoxy is wrong, and cricket has documented cases where the
textbook contradicts elite practice — most sharply the backlift, where 77% of successful
batsmen and 90% of IPL batsmen use a *lateral* backlift while 83% of coaches teach *straight*.

So references are conditioned on style, on two independent axes:

- **Intent axis** (attacking / balanced / defensive): **declared by the player**, and used to
  select template variants.
- **Technical style markers** (backlift direction, trigger movement, stance width):
  **detected and reported, never scored**. They are descriptors, not faults.

A pose-based classifier estimates per-shot intent (77% accuracy, 0.83 AUC in the literature),
but at ~1-in-4 error it is **not safe as an automatic gate**. It runs as a cross-check only:
where observed intent diverges from declared style, that is surfaced as an observation
("these clips read more defensive than the attacking profile you selected"), which is useful
coaching feedback in its own right.

**Style conditions the expected range, never the tier.** Measurement error is a property of
the camera and the joint, not of playing style.

---

## 6. Progress tracking

With ~8° RMSE on the best-measured angle, a 3° "improvement" is noise. Therefore:

- **Minimal detectable change (MDC)** is computed per metric; changes below it display as
  "no change detected", never as improvement or regression.
- Aggregation is **per session, not per clip** — 3–5 clips per session, using the session
  mean, shrinking standard error by √n.
- Trends use **rolling windows across sessions** with confidence bands, not consecutive-point
  deltas.
- Only Tier A metrics appear on progress charts. Tier B appears as banded categories.
  Tier C never appears.
- Like-for-like only: same form type, capture conditions within tolerance.

---

## 7. Reference templates

`analysis-service/reference_templates/` holds one file per form type. Every metric must
carry `tier`, `event`, `convention`, `source`, `population`, `confidence` and
`discriminates_skill`. **A metric missing `source` or `convention` cannot be scored** —
enforced by `template_validator.py` at boot, not by convention.

`convention` is mandatory because the literature uses mutually inverse conventions
(`straight_180` vs `flexion_from_0`). Mixing them silently flips a grade from "well braced"
to "collapsed".

`discriminates_skill: false` marks variables shown *not* to separate skill levels — bat
angle, bat speed, shoulder angle. They may be displayed for interest but carry zero score
weight.

Only `cover_drive.json` is populated. The others are structurally valid TODO stubs.
**Do not invent ranges to fill them** — fabricated reference values would defeat the entire
accuracy goal. Populating them requires sourced biomechanics or coaching data.

---

## 8. Local development

Everything runs locally at no cost, and pose extraction is faster on a dev machine than on a
small cloud instance.

### Try it

```bash
cd demo && ../analysis-service/.venv/Scripts/python.exe app.py
```

Open <http://localhost:5057>. Upload a clip, or click a **built-in sample** to see the output
format without needing footage.

Model weights are not committed (~58 MB). Fetch them into `analysis-service/models/` — both
Apache 2.0 from OpenMMLab:

```bash
curl -sL -o /tmp/d.zip "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.zip" && curl -sL -o /tmp/p.zip "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
```

Then extract each archive's `end2end.onnx` to `models/rtmdet.onnx` and `models/rtmpose.onnx`.

Throughput is roughly **0.13 s per frame** on CPU — a 150-frame clip takes about 20 seconds.

### Development

Python 3.11 is required — 3.7 cannot install the current scientific stack.

```bash
cd analysis-service && py -3.11 -m venv .venv && ./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

```bash
cd analysis-service && ./.venv/Scripts/python.exe -m pytest tests/ -q
```

To run the pipeline without model weights, drive it through `ArrayPoseBackend` or
`ReplayPoseBackend` — see `tests/test_pipeline.py::TestEndToEnd` for a worked example.

```bash
cd server && npm install && npm run dev
```

```bash
cd mobile && npm install && npx expo start
```

---

## 9. Deployment and cost

All software is free: RTMPose, RTMDet and MMPose are Apache 2.0 with commercial use allowed;
there is no paid AI API in this design because pose estimation runs on our own CPU.

Hosting is not free:

| Service | Free tier | Realistic need |
| --- | --- | --- |
| Render — analysis service | 750 instance-hrs/mo/workspace, spins down after 15 min idle | ~2 GB instance; Starter's 512 MB likely insufficient for RTMDet + RTMPose + OpenCV |
| Render — Node API | same pool | Starter |
| Appwrite Cloud | 2 GB storage, 5 GB bandwidth | Pro (from $25/mo) once clips accumulate |
| Clerk | 50K monthly retained users | free unless MFA / custom branding needed |

**Storage is the dominant long-term cost**, because clips are retained as future training
data. At ~15 MB per clip and 5 clips/session, the 2 GB free tier holds ~25 sessions.
Mitigation is implemented in `retention.py`: keypoints are ~1000× smaller than video
(~150 KB vs ~15 MB) and are what model training actually needs, so keypoints are retained
permanently while original video is kept only for consenting users and transcoded to a
compressed proxy after analysis.

App store fees ($99/yr Apple, $25 one-off Google) are avoidable by shipping the MVP as a
mobile web upload page.

---

## 10. Out of scope

Bowling action legality (ICC 15°); mixed-action or injury-risk screening; any medical claim;
bat speed and power-hitting mechanics; gaze/eye tracking (needs an eye tracker); 3D avatar
reconstruction; bat–ball contact-point judgement; ML-based form classification — revisit only
once consented labelled data accumulates, and note the rules-based scorer produces *weak*
labels that need coach spot-checks before they can be trusted for training.

## 11. Documentation

- `docs/ACCURACY.md` — per-metric measurement limits and the evidence behind each tier
- `docs/SOURCES.md` — full citation list
- `analysis-service/validation/README.md` — calibration harness
