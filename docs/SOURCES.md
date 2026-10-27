# Sources

Every reference range in `analysis-service/reference_templates/` must cite an entry here by
its `source` key. `template_validator.py` rejects templates whose `source` is empty.

---

## Measurement accuracy

**`abraham2025opencap`** — Abraham A, Feros SA, Fox AS (2025). *Exploring the accuracy of
OpenCap for three-dimensional analysis of cricket bowling.* International Journal of Sports
Science & Coaching.
https://journals.sagepub.com/doi/10.1177/17479541251348081
The decisive study for this project. Per-joint RMSE against marker-based mocap using three
iPads at 240 Hz; 56% trial success; concluded unsuitable for bowling legality testing.

**`kanko2023markerless`** — Markerless motion capture estimates of lower-extremity kinematics
and kinetics are comparable to marker-based across 8 movements.
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9980110/
Ankle/knee correlation ≥0.877, RMSD ≤5.9°; hip considerably worse (6.7–15.9°), especially in
rapid movement.

**`mundt2024events`** — Mundt M et al. (2024). *Automating video-based two-dimensional motion
analysis in sport? Implications for gait event detection, pose estimation, and performance
parameter analysis.* Scandinavian Journal of Medicine & Science in Sports.
https://doi.org/10.1111/sms.14693
Basis for temporal metrics outweighing angles.

**`monocular2025survey`** — *A Survey of the State of the Art in Monocular 3D Human Pose
Estimation: Methods, Benchmarks, and Challenges.* Sensors 25(8):2409.
https://www.mdpi.com/1424-8220/25/8/2409
Depth ambiguity; the in-image-plane requirement that drives the capture protocol.

**`posemodels2026`** — Pose-tracking stack comparison (OpenPose, MediaPipe, RTMPose,
ViTPose).
https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/openpose-mediapipe-rtmpose-pose-tracking
RTMPose-m: 75.8% AP COCO, 90+ FPS on CPU; benchmarks above MediaPipe on accuracy.

---

## Batting

**`peploe2019technique`** — Peploe C, McErlain-Naylor SA, Harland AR, King MA (2019).
*Relationships between technique and bat speed, post-impact ball speed, and carry distance
during a range hitting task in cricket.* Human Movement Science.
https://www.stuartmcnaylor.com/publication/CB3tech/Peploe_et_al_2019.pdf
Population: 20 batters, international → club, male. Range-hitting task.
- Max bat speed 27.0 ± 2.3 m/s; at impact 26.8 ± 2.3 m/s
- X-factor DS explains 28.0% of bat speed variance (SEE 1.65 m/s); + lead elbow extension
  58.6%; + wrist uncocking 77.7% (SEE 0.92 m/s)
- Impact location (lateral + longitudinal) + bat speed at impact explain 68% of ball launch
  speed variance
- Event definitions used by our phase detector: BC, SS, SE, DS, IMP
- Angle convention: **straight = 180°, flexed < 180°** (`straight_180`)

**`mann2013head`** — Mann DL, Spratford W, Abernethy B (2013). *The head tracks and gaze
predicts: how the world's best batters hit a ball.* PLoS ONE.
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3596397/
Elite vs club: mean head–ball angle 2.4 ± 1.7° vs 3.7 ± 1.4° (p=.003); max 10.7 ± 8.7° vs
20.6 ± 8.7° (p<.0001). Gaze −3.4 ± 1.5° vs 0.9 ± 1.4° ahead of ball. Elite make two
predictive saccades vs club batters' one. Gaze components require an eye tracker and are out
of scope.

**`noorbhai_lbbt`** — Noorbhai H, Noakes T. Lateral batting backlift technique research.
https://www.researchgate.net/publication/331834979
https://www.academia.edu/37841701/
77% of successful batsmen and 90% of IPL batsmen use lateral backlift; 83% of coaches teach
straight; >70% of uncoached players adopt lateral, >70% of coached adopt straight.
**Basis for treating backlift as a classified descriptor, never a graded fault.**

**`pullshot_malaysia`** — *Kinematics of the pull shot of the Malaysian national cricket
batsmen.* International Journal of Physiotherapy.
https://ijphy.com/index.php/journal/article/download/240/227
**Low confidence** — small single-squad sample; authors' stated angular reliability 3.1–10.7°.
- Phases: backlift of bat, bat–ball impact, follow-through
- Stride length at backlift: senior 0.77 ± .13 m, U19 0.85 ± .11, U16 0.80 ± .11 (n.s.)
- Stride duration at backlift ~0.22–0.24 s
- Right knee angle at follow-through: senior 139.97 ± 12.81° vs U16 120.77 ± 5.81° (p<.03)
- Left hip angle at impact: vs U19 134.21 ± 12.71° (p<.04)
- **No** significant difference in bat angle, shoulder or elbow angles at any phase
- Angle convention: **180° = full extension, 0° = full flexion** (`straight_180`)

**`headcom2007`** — *The position of the head and centre of mass during the front foot
off-drive in skilled and less-skilled cricket batsmen.* Sports Biomechanics 6(3).
https://www.tandfonline.com/doi/abs/10.1080/14763140701491310
Skilled batsmen hold the head further forward of the centre base point, with COM further
forward at predicted contact. 70% of skilled vs 20% of less-skilled used a preparatory foot
trigger movement.

**`frontfootdrive1999`** — *Kinematics and kinetics of the drive off the front foot in cricket
batting.* Journal of Sports Sciences.
https://pubmed.ncbi.nlm.nih.gov/10189076/
Front-foot stride ≈0.72 m; cover drive ≈0.68 m, planted ~0.06 s before impact.

**`sarpeshkar2011`** — Sarpeshkar V, Mann DL (2011). *Biomechanics and visual-motor control:
how it has, is, and will be used to reveal the secrets of hitting a cricket ball.* Sports
Biomechanics 10(4).
https://pubmed.ncbi.nlm.nih.gov/22303783/
Experienced batters take a **short-to-medium stride so they can play the ball later** — the
basis for banding stride length rather than rewarding length.

**`trigger_formats`** — *Video analysis of trigger movement by the batsman.*
https://www.physicaleducationjournals.com/archives/2026/vol8issue4/PartA/8-4-4-226.pdf
Trigger movement prevalence: Test 56.09%, ODI 47.36%, T20I 73.33%, IPL 76%.
Basis for treating trigger movement as format-dependent style, not a fault.

**`battingsetup_review`** — *A Systematic Review of Batting Set-up Mechanics in Cricket*
(2024). 11 studies, 1980–2024, covering grip, stance, initial movements, backlift.
https://www.researchgate.net/publication/387748671

---

## Playstyle and bat speed

**`intent2025posture`** — *Posture-Driven Action Intent Inference for Playing Style and
Fatigue Assessment.* arXiv:2507.11642.
https://arxiv.org/abs/2507.11642
1D CNN 77% accuracy / 0.83 AUC / 0.77 F1 leave-one-out over 11 matches; 74% / 0.76 / 0.73 on
a held-out batter. Features: shoulder, elbow, wrist, hip, knee, ankle, heel coordinates.
Dataset 2,612 shots from broadcast footage. Labels weakly supervised by runs scored (≥3 =
aggressive, ≤1 = defensive) — outcome-contaminated.

**`batsensor2025`** — *Accuracy of a bat-mounted sensor for the measurement of bat speed among
elite female cricket players.* Sports Biomechanics.
https://www.tandfonline.com/doi/full/10.1080/14763141.2025.2549136
9 professional female players, 40 swings, 4 shot types. Bat speed 52.8–87.9 km/h. Sensor vs
mocap bias: drive −1.0 km/h (1.4%), slog-sweep 2.0 (2.7%), pull 2.0 (2.8%), cut 2.5 (3.9%).
Precision 2.7–4.4 km/h. Proportional error increases with bat speed except for the pull.

**`powerhitting_sexdiff`** — *Comparing power hitting kinematics between skilled male and
female cricket batters.* Journal of Sports Sciences.
https://www.tandfonline.com/doi/full/10.1080/02640414.2021.1934289
Relevant to `population` tagging — elite male ranges must not be applied to other groups
without saying so.

---

## Bowling

**`icc_illegal_action`** — ICC. *Illegal Bowling Actions.*
https://www.icc-cricket.com/about/cricket/rules-and-regulations/illegal-bowling-actions
15° elbow extension threshold between arm reaching horizontal and ball release. Testing is
done at accredited centres with marker-based systems. **Out of scope for this app.**

**`lumbar_stress`** — *Lumbar Stress Fractures in Cricket Players.* Physiopedia.
https://www.physio-pedia.com/Lumbar_Stress_Fractures_in_Cricket_Players
Shoulder counter-rotation >30° defines mixed action and is associated with lumbar stress
fracture; more recent work suggests >25°, with knee flexion <30° at front-foot contact
elevating lumbar loading. **Screening for this is out of scope** — the risk threshold is
smaller than the measurement error.

**`elitefastbowlers2024`** — *Anthropometric and biomechanical factors in elite male and
female fast bowlers.* Journal of Biomechanics 170:112157.
https://www.sciencedirect.com/science/article/pii/S0021929024002355
Front knee angle at release correlates with ball release speed (r = 0.40, p < 0.001). Faster
release also associated with faster run-up, reduced back-foot contact time, braced front
knee, greater trunk flexion, delayed onset of shoulder counter-rotation, later peak
hip–shoulder separation.

**`bowling_action_classification`** — Shoulder alignment at back-foot contact: side-on <190°,
semi-open mean ~227.6°, front-on 246–262°. Recorded for completeness; **not used**, since it
depends on shoulder alignment measurement whose error is 28.5°.

---

## Stack and licensing

- MMPose / RTMPose licence (Apache 2.0):
  https://github.com/open-mmlab/mmpose/blob/main/LICENSE
- Render pricing: https://render.com/pricing
- Appwrite pricing: https://appwrite.io/pricing
- Clerk pricing: https://clerk.com/pricing
