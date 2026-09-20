# Deepfake Detection System — Design

**Date:** 2026-09-20
**Status:** Design approved in principle; P0 scope pending review
**Author:** rohit.kohli@scoreme.in with Claude

---

## 1. Problem

ScoreMe's video-KYC (v-CIP) onboarding flow can be defeated by a real-time face
swap injected through a virtual camera. This is not hypothetical — it has been
demonstrated and measured on our own infrastructure.

### 1.1 Measured evidence from `fraud_gff/deepfake_detection`

From 442 recorded capture sessions and 24 cached Reality Defender responses:

| Layer | Result against our live swap | Source |
|---|---|---|
| Face match (ArcFace), attacker vs victim ID | **0.0359 → rejected** | `scripts/measure_attack.py` |
| Face match, same attacker **with swap applied** | **0.8573 → accepted** | same |
| Liveness (MediaPipe challenge–response) | **passes** | by design; a live human drives the puppet |
| ScoreMe liveness scan | **5 sessions: `swapped=true` → `approved=true`** | `captures/*/results.json` |
| Reality Defender, still frame | MANIPULATED 96% | `cache/*/result.json` |

**The decisive fact:** face match and liveness are both defeated. Five sessions
with an injected swap were approved. The only layer that engaged at all was
frame-level deepfake detection, and it engaged weakly.

### 1.2 What the Reality Defender telemetry shows

Analysis of the 24 cached RD responses with per-model breakdowns:

| model | unique values / 24 | stdev | separation vs RD's own verdict |
|---|---|---|---|
| **rd-pine-img** | 13 | 0.402 | **0.992** |
| rd-full-pine-img | 20 | 0.358 | 0.966 |
| rd-cedar-img | 6 | 0.430 | 0.794 |
| rd-full-oak-img | 13 | 0.371 | 0.756 |
| rd-context-img | 4 | 0.160 | 0.752 |
| rd-oak-img | 10 | 0.442 | 0.706 |
| rd-full-cedar-img | 5 | 0.427 | 0.676 |
| rd-elm-img | 7 | 0.283 | 0.538 |
| **rd-full-elm-img** | **2** | **0.042** | 0.529 |

**Caveat:** n=24, and separation is measured against RD's *own aggregate*, not
independent ground truth. A member that drives the aggregate scores high here
by construction. This shows which members dominate the output, not which are
accurate.

With that caveat, three findings stand:

1. **The ensemble is reproducible from one member.** `rd-pine-img` alone
   reconstructs the aggregate at 0.992 separation. A ten-model ensemble whose
   output is predictable from one model is a one-model detector. When that one
   model drifts against a new generator, the whole product drifts with it.
2. **At least one member is inert.** `rd-full-elm-img` emits two distinct values
   across 24 samples (stdev 0.042). `rd-elm-img` at 0.538 is a coin flip.
3. **Outputs are saturated, not calibrated.** cedar and full-cedar produce 5–6
   distinct values across 24 samples, overwhelmingly 0.01 or 0.99. A detector
   that only says "definitely" cannot support a risk-based operating point and
   cannot be fused with other evidence.

Additionally: **no sample had more than 6 of 10 members concur** (median 5/10),
while the aggregate tracks the maximum. Near-max aggregation means the system's
false-positive rate approximates the *union* of its members' false-positive
rates — recall-tuned, at the cost of precision on genuine applicants.

---

## 2. First principles: what a deepfake physically is

A real frame ends a physical chain: photons → lens → Bayer sensor → demosaic →
ISP (denoise, sharpen, white balance) → codec. Each stage leaves a statistical
fingerprint: PRNU sensor noise, CFA periodicity, ISP residue, codec structure.

A synthesized face is a neural upsampling stack, then **composited** into that
frame through a mask.

Therefore a deepfaked frame is a **spatially heterogeneous statistical
composite**: the face region has one physical origin, the remainder has another,
joined at a seam. Every durable detection signal follows from this:

| # | Signal | Physical basis | Survives unseen generators? |
|---|---|---|---|
| 1 | Upsampling fingerprint | transposed-conv periodicity | partly |
| 2 | **Absent camera noise** | no PRNU in face region; residual decorrelated from frame | **yes — absence of physics** |
| 3 | **Blending boundary** | every composite has a seam | **yes — by construction** |
| 4 | Illumination / geometry | approximate relighting; corneal reflections must agree | weakly |
| 5 | rPPG pulse coherence | real skin pulses; face-vs-neck coherence breaks | video only |
| 6 | Temporal incoherence | per-frame generation → identity flicker, landmark jitter | video only |
| 7 | Inner-vs-outer face identity | swap replaces inner face only | yes, swap-specific |
| 8 | Provenance | C2PA, EXIF, codec consistency | a prior, not a detector |

Signals **2, 3 and 7** carry the most weight, because they do not require having
seen the generator before.

---

## 3. Second-order analysis: why passive vendors lose

Three structural failures, all visible in our own data:

**(a) Distribution shift is the default state.** Published cross-dataset results
show detectors at ~0.99 in-dataset AUC falling to roughly 0.60–0.75 on unseen
generators. The 4-of-10 RD dissent is this failure mode caught in the act: half
the ensemble had never seen a real-time DeepFaceLive swap with GFPGAN
restoration. That was not a detection; it was a favourable coin flip.

**(b) Supervised binary training learns shortcuts, not causes.** Training on
FF++ teaches FF++'s compression and blending idiosyncrasies. Change the blend
and the knowledge evaporates.

**(c) A finished JPEG cannot be interrogated.** Against interactive fraud,
forfeiting interactivity forfeits the largest available asymmetry.

**Architectural consequence:** any fixed classifier wins for ~3–6 months, then
loses. Therefore the *expensive, stable* layer must be fusion, calibration,
policy and serving; the *perishable* layer — individual detectors — must be
hot-swappable, with a red-team loop that measures decay. **This is an MLOps
problem wearing a model problem's clothes.**

**Third-order:** a deployed fixed ensemble invites adversarial optimisation
against it. Mitigations that do not rely on secrecy: randomised detector subsets
per request, detectors on deliberately destroyed representations
(noise-residual only, heavy downsample) that imperceptible perturbation cannot
survive, input purification, and probing-pattern monitoring.

---

## 4. Design principles

1. **Likelihood ratios, not scores.** The common currency across all detectors
   and modalities.
2. **Refuse to answer below a quality floor.** Abstention is a first-class
   output.
3. **Ensemble diversity must be in the physics, not the architecture.**
4. **Admission to the portfolio is earned** by measured marginal lift under
   leave-one-generator-out, never by existing.
5. **Detectors are perishable and hot-swappable; the core is stable.**
6. **Measure decay continuously**; do not assume durability.
7. **Correctness first.** Latency is an optimisation applied after correctness
   is established, not a design driver.

---

## 5. Core architecture

### 5.1 The abstraction

Image, video, live stream and audio differ along only three axes:

| | observations | time budget | interrogable? |
|---|---|---|---|
| Image upload | 1 | seconds | no |
| Video file | N sampled | seconds–minutes | no |
| Live stream | unbounded | session-bound | **yes** |
| Audio | N windows | seconds | maybe |

The evidence is the same kind of thing in all four. Hence:

```
Sample      = ordered [Observation] + Context
Observation = (t, payload, roi, quality)
Detector    : Observation[] → Evidence
Evidence    = { llr, uncertainty, artifacts, provenance }
Fusion      : Evidence[] → posterior
Policy      : posterior × Context → decision
```

### 5.2 The likelihood-ratio currency

```
llr = log[ P(observation | fake) / P(observation | real) ]
```

Chosen over score-averaging because:

1. **LRs add.** Independent evidence accumulates by summation — the only
   coherent way to combine a frequency detector with rPPG with audio with a
   C2PA manifest.
2. **It extends over time for free.** A live session accumulates evidence until
   the posterior crosses a decision boundary — a sequential probability ratio
   test. Fast decisions on obvious cases, patient ones on ambiguous cases, from
   one mechanism.
3. **It can express ignorance.** LR ≈ 0 means "this observation carries no
   information." A 0–1 score cannot distinguish that from genuine uncertainty.
4. **It is auditable.** "Detector B contributed +2.1 nats because face-region
   noise residual was decorrelated from the frame" is a defensible reason code.
   "The model said 0.96" is not.

### 5.3 Quality gating

Every detector declares a quality floor: minimum inter-ocular pixels, maximum
blur, pose envelope, exposure range. Below it the detector is **not consulted**
and returns LR = 0. System output becomes `INSUFFICIENT_EVIDENCE → re-capture`
rather than a confident guess on a 40-pixel blurry face.

Vendor APIs cannot do this — they are contractually obliged to return a number
for every call. On-prem, we can demand a better frame instead.

### 5.4 Adding a modality

A new detector registers, declares its modality and quality floor, and emits
LRs. Fusion, policy and serving are unchanged. This is the hot-swappability
that section 3 established as mandatory.

---

## 6. Detector portfolio

**Governing principle:** each slot must fail for a *different physical reason*.
Five spatial CNNs trained on FF++ are one detector wearing five hats — their
errors correlate, so the ensemble reports false confidence exactly when it is
wrong.

| Slot | Model | Physics read | Weakness |
|---|---|---|---|
| **A. Blending boundary** | SBI (Self-Blended Images) | the composite seam | weak on fully-synthetic faces |
| **B. Noise residual** | SRM / Noiseprint-style | absence of camera PRNU | destroyed by heavy compression |
| **C. Upsampling fingerprint** | NPR | transposed-conv periodicity | post-hoc resampling blurs it |
| **D. Frequency** | SPSL / F3Net | phase-spectrum anomalies | codec-dependent |
| **E. Spatial appearance** | EfficientNet-B4 / UCF | learned texture artifacts | worst cross-generator drop |
| **F. Open-set generalization** | CLIP ViT linear probe | semantic manifold deviation | expensive |
| **G. Identity consistency** | ICT (inner vs outer face) | swap replaces inner face only | needs resolution |
| **H. Provenance** | C2PA + EXIF / quantization tables | container & compression history | usually absent |
| **I. Temporal** *(video)* | identity-flicker, LipForensics, TALL | per-frame generation jitter | needs ≥2 s |
| **J. Physiology** *(video)* | rPPG (POS / CHROM) | pulse coherence face-vs-neck | fails in low light |
| **K. Audio** | AASIST → wav2vec2-AASIST | vocoder artifacts | telephony degradation |
| **L. AV sync** | SyncNet-style lip-audio correspondence | cross-modal coupling | catches what neither alone catches |
| **M. Active challenge** *(live)* | flash reflectance + injection attestation | real-time optical physics | live only |

**Expected load-bearing slots for our attack:** A, B and G. SBI trains on real
images only and never learns a specific generator; B detects absence of physics
rather than presence of an artifact. The spatial CNNs that dominate published
benchmarks are expected to contribute least.

**Deferred optimisation — tier cascade.** T0 (quality gate, NPR, provenance),
T1 (SBI, SRM, frequency, spatial, rPPG), T2 (CLIP, ICT, AV-sync) consulted only
when accumulated LR has not crossed a boundary. v1 runs the full portfolio on
every sample and records per-detector latency as data. The cascade is switched
on later, if and when latency matters.

---

## 7. Fusion, calibration and policy

1. **Per-detector calibration.** Raw score → LR via isotonic or Platt scaling on
   held-out data, **conditioned on quality band and compression level**. A
   detector's reliability at 512 px uncompressed is not its reliability at 96 px
   after messaging-app recompression.
2. **Correlation-aware combination.** Naive summation assumes independence,
   which is false (SBI and SRM both degrade under compression). Fit a low-rank
   correlation structure, or stack over LRs with **disagreement spread as an
   explicit feature**. Disagreement is signal: a 6/4 split means off-distribution
   and should escalate. RD discards this information.
3. **Explicit OOD head.** Separate from "is it fake": "is this sample like
   anything I was calibrated on?" (Mahalanobis or energy score). High OOD →
   suppress confidence, route to review. This is the honest response to
   distribution shift — detect that you cannot generalize rather than pretend to.
4. **Four outputs, not two:** REAL / FAKE / INSUFFICIENT_EVIDENCE /
   OUT_OF_DISTRIBUTION.

### 7.1 Policy tied to fraud economics

```
E[loss | decision] = P(fake|evidence)·L_fraud + P(real|evidence)·C_friction
```

Thresholds derive from transaction economics. A ₹50k personal loan and a ₹5cr
facility do not deserve the same threshold; one threshold for both is mispriced
by construction. Bands: **AUTO-PASS / STEP-UP / MANUAL REVIEW / BLOCK**, with
boundaries as versioned, auditable configuration.

This is an advantage unrelated to model quality: a vendor returning a score has
no knowledge of what the transaction is worth.

**Status:** fraud-loss and friction figures not yet supplied. Policy layer ships
with configurable placeholders; calibration deferred to ScoreMe.

### 7.2 Audit record

Every decision emits an immutable record: input hash, model versions,
per-detector LRs, quality metrics, OOD score, policy version, threshold,
decision, localization heatmap. Serves RBI defensibility and doubles as
training data for the next cycle.

---

## 8. Benchmark harness

Given correctness-first priority, this is the foundation, not infrastructure.
**We cannot claim "better" without an instrument capable of proving us worse.**

### 8.1 Protocol

**Leave-one-generator-out, video-level, cross-compression.** Train on a
generator subset, test on a generator never seen. This is the only number that
predicts field performance, because the field always brings an unseen generator.

### 8.2 Evaluation-hygiene guards

Five failures that make published deepfake numbers untrustworthy. Each is an
explicit guard:

1. **Identity leakage** — same person in train and test teaches faces, not
   forgery. *Guard:* identity-disjoint splits, verified by ArcFace clustering
   across the split boundary.
2. **Frame-level metrics reported as video-level** — 10,000 frames from 100
   videos is 100 independent samples. *Guard:* video-level aggregation; CIs
   bootstrapped over videos.
3. **Compression matching** — train c23 / test c23 flatters everything.
   *Guard:* evaluate across c0/c23/c40; report the **worst** cell.
4. **Preprocessing leakage** — different face detectors on real vs fake is a
   giveaway. *Guard:* one detector, one alignment, applied blind to label.
5. **Threshold selected on test set.** *Guard:* operating points frozen on
   validation, applied untouched.

### 8.3 Metrics

- **TPR@FPR=0.1% and 1%** (the fraud-relevant numbers)
- LOGO AUC
- ECE (calibration)
- **Robustness surface:** JPEG quality sweep, resize, blur, noise,
  **screenshot-of-screen, print-recapture**
- p95 latency per detector (recorded, not optimised, in v1)

### 8.4 Head-to-head vs Reality Defender

The 24 cached results are free and already labelled by our records. Beyond that,
RD free tier is 50 scans/month — error bars too wide for a defensible claim. A
publishable comparison requires either one month of paid tier or explicitly
reported wide confidence intervals.

---

## 9. Live layer

### 9.1 Attack taxonomy

| | Attack | Defeated by |
|---|---|---|
| 1 | Virtual camera (OBS / v4l2loopback ← DeepFaceLive) — **our demonstrated attack** | capture-path attestation |
| 2 | Hardware injection (HDMI→UVC capture card) | timing forensics + platform attestation |
| 3 | Stream MITM, frames swapped server-side | signed capture, TLS pinning |
| 4 | Emulator / rooted device | Play Integrity / App Attest |
| 5 | Replay: screen or print to a real camera | PAD + optical challenge |

### 9.2 D1 — Capture-path attestation

Highest-value layer, contains no ML. Device labels are trivially renamed —
ignore them. The real controls:

- **Camera control probe.** Issue `applyConstraints()` (exposure, zoom, torch,
  resolution) and verify pixels respond with correct latency and correct
  *physical* response. A virtual camera fails to implement it, lies about it, or
  produces a response inconsistent with optics.
- **Frame-timing forensics.** Real sensors have characteristic inter-frame
  jitter from the sensor clock and exposure loop. A GPU-fed virtual camera is
  implausibly regular or carries the swap pipeline's processing variance.
  Measure the distribution, not the mean.
- **Platform attestation.** Play Integrity, App Attest, hardware-backed
  keystore. For mobile v-CIP this is the single strongest available control.
  **Implication: mobile-app v-CIP is structurally more defensible than browser
  v-CIP** — an architecture decision, not a detection one.

### 9.3 D2 — Active optical challenge

Display a freshly-random colour sequence; it illuminates the face. Verify:

1. Face reflectance tracks the emitted sequence — timing and chromaticity.
2. **Face and background respond coherently to the same light source.** A swap
   replaces only the face, so the two decouple under changing illumination.
3. Sequence is random per session — replay impossible by construction.

Defeating this requires real-time relighting conditioned on an unpredictable
signal visible only at challenge time; far beyond commodity swap tooling.

**Hard constraints:** requires the screen close enough to illuminate; degrades
in bright ambient light; **flash rate and intensity must be bounded for
photosensitive-epilepsy safety** — a design requirement, not a footnote.

### 9.4 D3 — Pose-conditioned degradation

Swap training data is overwhelmingly frontal, so swap quality collapses at
profile while a real face's detector score is pose-invariant. Do not merely
check that the subject turned — measure **the detector score as a function of
yaw**. The derivative is the signal. Nearly free, reusing existing challenges.

### 9.5 D4 — Sequential accumulation, done correctly

**Frame-to-frame errors are strongly correlated** (same pipeline, same identity,
same lighting). Naively summing LRs across 900 frames yields a posterior of
~1.0 regardless of truth — the textbook way to build a system that is
confidently wrong.

**Mandatory:** estimate effective sample size from temporal autocorrelation and
discount accordingly; bootstrap over segments, not frames. Expect 900 frames to
be worth roughly 10–30 independent observations.

### 9.6 D5 — rPPG

Pulse coherence, face region vs neck.

### 9.7 D6 — Population-level intelligence

Per-session detection is eventually beatable; cross-session signals are not.
Same device fingerprint across multiple applicant identities, velocity
anomalies, recurring synthetic identities. Organised fraud is visible at
population scale even when each session looks clean. Not optional for
industry-grade fraud prevention.

### 9.8 Latency decoupling

The challenge must be *issued* live; the analysis need not be. Record the
session with its challenge response and decide asynchronously if analysis is
slow. Decouples correctness from latency.

---

## 10. Packaging, operations, red-team loop

**On-prem packaging.** Docker-compose single-box; Helm for k8s. Models ship as a
versioned, **signed** OCI artifact separate from code, so detector swaps need no
service redeploy. Zero required egress — air-gap installable. Both a compliance
win and a differentiator against a US SaaS.

**Model lifecycle.** Registry with semantic versions. **Shadow mode** for every
new detector (runs, logs, does not vote) until it shows marginal lift on the
benchmark; then canary; then promote. Every decision record pins exact model
versions, so an audit years later can reconstruct the verdict.

**Monitoring — watch decay, not uptime.** Score-distribution drift, OOD rate,
abstention rate, **inter-detector agreement entropy** (rising disagreement =
the field has changed), manual-review outcomes fed back as labels.

**Red-team loop — what makes this durable.** Any fixed detector decays in 3–6
months, so decay is measured rather than assumed:

- The existing `face_swap` / DeepFaceLive rig plus each newly released generator
  mints fresh attacks on a standing cadence.
- Each new generator runs against the **frozen production ensemble before
  retraining**, producing the metric that matters: **time-to-detect a novel
  generator**.
- Falling below a floor triggers retraining.

That is a contractable SLA no passive-classifier vendor can offer.

**Security of the detector itself.** Adversarial perturbation, model extraction
via systematic probing, resource exhaustion via crafted media. Mitigations:
input purification, randomised detector subsets, probing rate limits, hard
decode limits.

**Compliance.** Face data is sensitive personal data under India's DPDP Act.
On-prem helps substantially; plus retention limits, purpose-specific consent,
and a model card per detector.

---

## 11. Licensing posture

**Decision (2026-09-20):** build first using research-licensed assets for
development and benchmarking; revisit commercial licensing based on market
traction; train owned models before commercial deployment.

**Phase gate (binding):** research-licensed datasets and non-commercial weights
are permitted through build, benchmark and internal demo. They **must be
swapped out before any paying-customer deployment.** This is a release gate, not
a gate on current work.

**Cheap insurance, adopted now:**

- **Asset manifest.** One record per dataset and weight file: source, license,
  commercial-use verdict, evidence URL, date checked. Minutes now; weeks to
  reconstruct later.
- **Architecture keeps the swap cheap.** Detectors sit behind the LR interface,
  so replacing a weight file is a config change, not a rewrite.

**Known exposure in the current stack** (independent of this project, worth
checking regardless): `fraud_gff/deepfake_detection` uses InsightFace
`buffalo_l` for ArcFace matching. InsightFace pretrained models have
historically been academic-research-only. SCRFD/RetinaFace and the GFPGAN
dependency chain warrant the same check. Permissive substitutes exist for
detection (MediaPipe Apache-2.0, OpenCV Zoo YuNet).

**The eventual clean path, for reference.** Methods and algorithms are not
copyrightable — SBI, Face X-Ray, NPR, F3Net, SRM, AASIST, POS/CHROM are free to
reimplement. Encumbrance attaches to specific datasets and specific weight
files. Reimplementing from paper and training on owned data yields weights that
are genuinely ours, warranties included. **SBI makes this unusually cheap: it
needs no fake data at all, only real faces we have consent for.** Layers needing
no licensed training data: active challenge, capture attestation, timing
forensics, provenance, classical rPPG, quality gating, noise-residual, SBI —
which is largely the same set as the strongest layers.

---

## 12. Decomposition

| Phase | Scope |
|---|---|
| **P0** | **Evidence core + benchmark harness.** Sample abstraction, ingest adapters (image, video), face normalize + quality gate, detector plugin interface, LOGO benchmark harness with the five hygiene guards, three baseline detectors (**slot C / NPR**, **slot A / SBI**, **slot E / EfficientNet-B4** — chosen as three *distinct* physics: upsampling fingerprint, blending seam, learned appearance, so the harness is exercised on genuinely uncorrelated evidence from day one), head-to-head vs the 24 cached RD results and our own swap corpus, asset manifest. |
| P1 | Image engine: full ensemble, calibration, fusion, OOD head, explainability. |
| P2 | Video-file engine: temporal extractors, sampling strategy, correlation-aware aggregation. |
| P3 | Live-stream engine: active challenge, capture-path attestation, sequential testing, pose-conditioned degradation. |
| P4 | Audio engine: synthetic speech + AV sync. |
| P5 | Productization: on-prem packaging, audit trail, model registry and hot-swap, monitoring, red-team loop, population-level intelligence. |

**P0 is non-negotiably first.** Without the measuring instrument, "better than
Reality Defender" is a claim rather than a fact — repeating precisely the
mistake that let a 96% score with 40% internal dissent look like a win.

### 12.1 P0 acceptance criteria

1. A reproducible LOGO benchmark run producing TPR@FPR=1%, LOGO AUC and ECE for
   every registered detector, with all five hygiene guards active and verified.
2. Identity-disjointness of splits verified by ArcFace clustering, with the
   check reported as a number, not asserted.
3. Video-level metrics with CIs bootstrapped over videos.
4. A head-to-head table against the 24 cached RD results and our 442-session
   swap corpus, with confidence intervals honestly reported.
5. Per-detector p95 latency recorded on the target hardware (i5-1235U).
6. Asset manifest covering every dataset and weight file in use.
7. Quality gate demonstrably abstaining on degraded input rather than guessing.

---

## 13. Open questions

| # | Question | Status |
|---|---|---|
| 1 | Can onboarding operations absorb `INSUFFICIENT_EVIDENCE → re-capture`? | Proceeding with configurable bands; config change, not redesign, if not |
| 2 | Fraud-loss and friction figures for policy calibration | Placeholders; ScoreMe to supply |
| 3 | Paid RD tier for a statistically meaningful head-to-head | Budget decision, deferred to P0 execution |
| 4 | Browser vs mobile-app v-CIP | Raised in §9.2; mobile is structurally more defensible |
| 5 | Public dataset EULA lead time (FF++, Celeb-DF, DFDC, DF40) | Request early — days to weeks; gates P0 benchmark breadth |

---

## 14. The bet, in one line

We win not by having a better classifier, but by owning **calibration,
abstention, capture-path physics, and a measured decay loop** — the four things
a score-returning SaaS structurally cannot sell.
