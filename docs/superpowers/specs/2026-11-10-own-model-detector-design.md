# An Own-Model Image Deepfake Detector — Design

**Date:** 2026-11-10
**Status:** Approved in conversation; implementation plan pending
**Author:** rohit.kohli@scoreme.in with Claude
**Parent spec:** `2026-09-20-deepfake-detection-design.md` (the authority; this
document adds one detector slot and the corpus that can measure it)

---

## 1. Problem

The engine is deployed and returns nothing.

As of 2026-11-09 the service runs on an EC2 instance, answers HTTP 200, accepts
files through the inbox, and writes audit records. Every one of those records
reads `insufficient_evidence` with `llr_total: 0.0`, because all three
detectors are gated out at startup:

```
WARN: detector blend_seam is gated out of deciding: measured AUC 0.289 is below the 0.75 floor
WARN: detector effnet_b4 is gated out of deciding: measured AUC never measured is below the 0.75 floor
WARN: detector npr is gated out of deciding: measured AUC never measured is below the 0.75 floor
WARN: discarded the calibration curve for npr: it has not been measured above the evidence floor
```

This is the gate working correctly, not a bug. Slot A measured *inverted*
(0.289) and slot C has never been measured on a corpus capable of measuring
anything. The only corpus that ever produced a number — DF40 — was ruled unable
to support or refute a detector: a Lab-colour shortcut separates its halves at
0.843, its 1601 fakes carry a Kish effective sample size of **2.4**, and a model
fitted on *shuffled* labels scores 0.229–0.780, which contains every number this
project has ever reported on it.

So the deliverable is not plumbing. **It is one detector with a defensible
number, measured on a corpus that can carry one.**

The product target is Reality Defender's contract: upload an image, receive a
confidence score. The constraint is that no third-party detection API may serve
it — the detector must be ours.

## 2. Scope

**In:**

1. **Slot D — `gend_clip`**: a CLIP ViT-L/14 encoder with LayerNorm-only tuning,
   an L2-normalised class token, and a linear head. Implements the existing
   `Detector` protocol; registered in `default_registry`.
2. **A paired real/fake corpus** built from `vkyc-server/captures`, where every
   fake is generated from a specific real frame and re-encoded identically.
3. **Two or more generators**, so leave-one-generator-out can fold.
4. **A measurement** of slot D on that corpus, with a permutation null beside
   every interval, written into `bench/evidence_card.json`.
5. **A calibration** per quality band, so a passing AUC actually opens the gate.
6. **An image upload endpoint** returning the per-model score list the product
   needs, served from the already-deployed service.
7. **A `commercial_use: true` entry in `assets/manifest.yaml`** for every asset
   slot D touches, each citing a licence text read at its origin. The software
   is to be commercialised, so this is an acceptance criterion, not hygiene:
   the asset gate fails closed and unregistered means non-commercial.

**Out** — each needs its own cycle:

- Video. The pipeline samples frames already; a video verdict is a fusion
  question, not a detector question, and it doubles the validation surface.
- Audio and text modalities. Reality Defender spans them; this does not.
- Retiring slots A, C and E. They stay registered and keep abstaining. Their
  raw scores are still recorded, and ensemble *disagreement* is a product
  feature, not noise — see §5.3.
- The §7.1 decision economics (AUTO-PASS / STEP-UP / BLOCK bands). Loss and
  friction figures are still unsupplied; bands fitted to placeholder economics
  are numbers nobody can defend.
- Any claim of parity with a commercial vendor. See §4.3.

## 3. What is being matched

Reality Defender, stripped of marketing, is an **ensemble of per-modality models
run in parallel**, whose confidence score reflects cross-model consensus rather
than one classifier's output. They retain models that disagree on purpose: an
attacker who evades one model's pattern meets a different pattern in the next.

The output contract is visible in `fraud_gff/deepfake_detection`, which already
adapts it:

| Field | Meaning |
|---|---|
| `verdict` | AUTHENTIC / MANIPULATED / UNKNOWN |
| `score` | overall confidence, 0–1 |
| `models[]` | per-model `{name, verdict, score, heatmap}` |
| `dissenting_models` | those disagreeing with the overall verdict |

That maps onto this repo's `AuditRecord` almost field for field: `evidence[]`
already carries per-detector `{detector, llr, raw_score, reason, abstained}`,
and `fuse` already treats hard disagreement as `OUT_OF_DISTRIBUTION` rather than
averaging it away. **The contract needs adapting, not inventing.**

Published numbers, for calibration of expectations only: ROC-AUC 0.89, accuracy
0.85, MCC 0.646 on VendorBench-100 — second of 36 systems.

## 4. Approach

### 4.1 Decision: build on GenD

The instruction was to use open source where it reaches vendor accuracy, and
build only what does not exist. For image deepfake detection it reaches it:

| | Reality Defender | GenD (WACV 2026) |
|---|---|---|
| Image AUC | 0.89 (VendorBench-100, n=100) | **0.912–0.916** mean AUROC, 14 datasets |
| Celeb-DF-v2 | — | 0.960 |
| DFDC | — | 0.871 |
| Licence | commercial, per-scan quota | **MIT** |

GenD's method is three changes to a frozen pretrained encoder: tune only the
LayerNorm parameters (**~91K of 304M, 0.03%**), L2-normalise the class token,
and fit a linear classifier. Backbone options are CLIP ViT-L/14, Perception
Encoder, and DINOv3; they score within 0.4pp of each other.

**We take the method, not the checkpoint.** GenD's code is MIT, but its
released weights are trained on FaceForensics++, whose terms are non-commercial
research only — a model fitted on that data is derived data under those terms.
This is the same bar that already disqualified `sbi_effnetb4_weights` and
`ffpp_effnetb4_weights` in this repo's manifest, and it applies identically
here regardless of the MIT tag on the code. So: no fine-tuning from
`yermandy/GenD_CLIP_L_14`, and no initialising from it. The LayerNorm
parameters and the linear head are fitted here, from scratch, on commercially
cleared data.

That is affordable precisely because the fitted part is ~91K parameters. If the
method required a full fine-tune, losing the checkpoint would end the project;
losing it costs a training run instead.

**CLIP ViT-L/14 is the backbone to use**, for a licence reason rather than an
accuracy one: OpenAI's CLIP weights are MIT, which permits commercial use with
attribution as the only real obligation. DINOv3's terms are Meta's and need
reading before anything ships; the 0.4pp it might buy does not justify opening
that question. This repo has already ruled that `NonCommercial`-licensed
weights (SBI, DeepfakeBench, InsightFace pretrained) can never be deployed, and
that ruling stands.

### 4.2 Why LN-tuning is the right size for this project

Two reasons, and the second is the one that survives commercialisation.

**Reason one: it resists overfitting on a narrow corpus.** The commercially
cleared supply is FairFace (97,698 reals) and SFHQ (~118k synthetic) — ample in
count, but narrow in *kind*: Flickr photographs and StyleGAN2 samples, against a
product that must work on v-CIP webcam captures. A 304M-parameter fine-tune on
material that narrow learns the corpus. Fitting 91K over a frozen encoder keeps
the general-purpose representation that CLIP already has and fits only the
read-out, which is the property that makes GenD generalise across 14 datasets it
never trained on. Domain shift is addressed by augmentation matched to capture
conditions (§9.2), never by training on the captures themselves — §6.4 keeps
them out of the training path for a reason no measurement can overturn.

**Reason two: losing GenD's checkpoint is survivable.** Because FF++ taints
their weights (§4.1), everything must be fitted here from scratch. With a full
fine-tune that would mean reproducing a 115k-frame training run with no
commercially usable equivalent of FF++ to run it on. With LN-tuning it means
fitting 91K parameters over a frozen MIT-licensed encoder on CC BY / CC0 data —
which this project can actually do. The method choice and the licence
constraint point the same way.

### 4.3 What "at par" may and may not be claimed

VendorBench-100 is **100 images** — 79 fake, 21 real — hand-selected for
difficulty. An AUC over 21 reals has error bars wide enough to cover most of
the leaderboard, and this project has already been destroyed once by trusting a
corpus whose effective sample size was 2.4.

**Therefore: no number produced by this work will be reported as parity with
Reality Defender.** The target is GenD's *cross-dataset* behaviour reproduced on
our own held-out generators. Beating a vendor on a 100-image benchmark is cheap
and means nothing; generalising to a generator never seen in training is the
property the product actually needs.

## 5. Architecture

### 5.1 Slot D

A new `dfd/detectors/gend.py` implementing the existing `Detector` protocol:
`name`, `slot`, `version`, `modalities`, `min_quality_band`, and a call that
returns a raw score or `abstain(...)`.

- `slot = "D"` — a distinct physics from A (blend seam), C (upsampling
  fingerprint) and E (learned appearance on EfficientNet). Slot-diversity of the
  default registry is a tested invariant, so the slot letter is load-bearing.
- `min_quality_band` — to be **measured**, not assumed. Slot C's `"reject"`
  floor was admitted only after measurement showed blur *is* the artefact there.
  Slot D gets the same treatment: the floor is whatever the band-wise numbers
  justify.
- Weights load through `loading.load_model`'s secure path (`weights_only=True`,
  plain `state_dict`, explicit `model_factory`). A full pickle is arbitrary code
  execution from a weights file and is not acceptable in a deployed service.

The encoder is 304M parameters and will not be committed. It is fetched and
hash-pinned the way `ops/fetch-assets.sh` already pins YuNet and SFace, so the
manifest's licence claim attaches to specific bytes rather than to a path.

### 5.2 Nothing else changes

`fuse`, `Calibrator`, `decide`, `build_audit_record`, the policy, the service,
the inbox and the dashboard are untouched. Slot D produces a raw score; the
existing calibration turns it into an LLR; the existing fusion sums LLRs; the
existing gate decides whether it is allowed to count. **If slot D's measured
AUC clears 0.75, the gate opens by itself.** That is the whole mechanism, and
it already works — it has simply never had a detector worth admitting.

### 5.3 The product surface

An upload endpoint on the existing service returning the RD-shaped contract:
overall verdict and score, plus the per-detector list. Slots A, C and E stay in
that list as abstentions with their raw scores recorded.

This is deliberate. A single number is weaker than a number plus the four
opinions behind it, and the ensemble's disagreement is the honest part of the
answer. `fuse` already escalates hard disagreement to `OUT_OF_DISTRIBUTION`,
which is the correct output for an input unlike anything measured — and far more
defensible than a confident score on an unfamiliar generator.

## 6. The corpus

This section is the one most likely to sink the project, and commercialisation
rewrites it. The earlier draft of this design proposed training on
`vkyc-server/captures` with fakes generated by inswapper_128. **Both halves of
that are blocked for a commercial product**, for reasons the manifest already
records against `npr_capture_head`:

1. **The swapper.** InsightFace's distributed models, inswapper_128 among them,
   are non-commercial research only; that model was additionally withdrawn from
   public distribution. A head fitted on its output is derived data. Settled
   only by re-generating with a licensed or permissively licensed generator.
2. **The captures.** The owner attests rights to the corpus, but no legal review
   has addressed whether v-CIP recordings may train an unrelated commercial
   product, and biometric consent is a separate matter from rights in the
   recording. The manifest's own words: this one **"is not settled by any
   amount of re-generation."**

So the training corpus changes supply.

### 6.1 Reals: FairFace, not the captures

**97,698 real faces, CC BY 4.0, commercial use permitted, no EULA.** The
manifest calls it the only face supply this project has found that is both
commercially usable and reachable without an EULA, a PI, or an academic
address. Attribution is a licence *condition*: `training/export_fairface.py`
already writes the dataset, licence and URL into every exported session, and
that must not be stripped from anything that ships.

It is also 190x the unique material in the capture corpus (509 frames), which
matters independently of licensing.

### 6.2 Fakes: self-blending is the training family, and it needs no generator

**Training needs no face-swap generator at all.** Self-blended images blend a
real frame with a photometrically and geometrically jittered copy of *itself*
under a landmark-derived mask. The result carries the one artefact every face
swap shares — a composite seam — while identity, camera and lighting stay
unchanged, so the detector learns the seam rather than a generator's signature.
`corpora/sbi.py` already implements this and already says exactly that.

The published evidence that this is sufficient: SBI trained on **real images
only** reaches **93.18% AUC on Celeb-DF**, a face-swap corpus from which it
never saw a single swap; the frequency-enhanced variant FSBI reaches 95.49%.
Both are cross-dataset, which is the generalisation property the product needs.

This is also the licence-clean path by construction: it consumes only real
faces and no generator weight file, so it inherits FairFace's CC BY 4.0 and
nothing else.

| Family | Source | Licence | Role |
|---|---|---|---|
| Blending seam | `corpora/sbi.py` self-blends of FairFace | CC BY 4.0 (inherited) | **the training family** |
| Entire-face synthesis | SFHQ part 3, ~118k StyleGAN2 faces | CC0 / MIT | LOGO fold + saturation control |
| Real face swaps | existing inswapper_128 swaps of the captures | non-commercial | **evaluation only** (§6.4) |

### 6.2.1 Why this project's first self-blend attempt failed

Slot A (`blend_seam`) *is* the SBI approach, fitted on FairFace self-blends, and
it measured 0.289 — inverted. That looks like a refutation of everything above,
and is not, for two reasons worth stating because they shape slot D:

1. **The corpus could not measure it.** 0.289 sits inside DF40's permutation
   null (0.229–0.780). It is not evidence of inversion or of anything else.
2. **The read-out was wrong, not the training signal.** `blend_seam` puts a
   small hand-crafted feature set over the seam. SBI's published results come
   from a deep encoder (Xception/EfficientNet) *learning* the seam. The repo's
   own note records that refitting the same features reached 0.800 in-domain,
   so the features are not blind — the capacity to read them was missing.

**Slot D is the combination this project has never tried: the SBI training
signal read by a deep encoder.** Self-blends supply the artefact; CLIP ViT-L/14
with LN-tuning supplies the read-out that SBI's own results depend on. That is
the specific bet of this design, and §6.2.1 is why it is not a repeat of slot A.

### 6.3 Pairing, and why it survives the supply change

GenD's ablation: training on **paired** real/fake from the same source is worth
**4.7pp mean AUROC** (90.0 vs 85.3), because unpaired training lets the network
learn which *source* an image came from instead of what was done to it.

Self-blending is inherently paired — the fake is made from the real it is paired
to. The diffusion-edit family must be generated the same way: edit a specific
FairFace crop, keep the pair. SFHQ cannot be paired at all (its faces
correspond to no real person), which is a second reason it is a fold and a
control rather than a training family.

Both halves then pass through an **identical re-encode**. Non-negotiable: one
JPEG header byte labels 99.2% of the capture corpus, and matching native
resolution and container on DF40 *raised* the colour shortcut from 0.800 to
0.904 rather than removing it. If reals keep their original encode and fakes are
written by a different encoder, the detector learns the encoder.

### 6.4 The captures become evaluation, and even that is conditional

`vkyc-server/captures` remains valuable as the **in-domain evaluation set** —
it is the only material that looks like live v-CIP traffic, and a detector
trained on FairFace faces a real domain shift towards it.

But evaluation of a commercial product on real customers' biometric captures is
the same unresolved question as training on them, only smaller. It is recorded
here, kept out of the training path entirely, and **flagged for legal review as
a release blocker** (§9.1). Nothing in this design resolves it, and no
measurement can.

The existing inswapper_128 swaps of those captures stay in this role too, and
only this role. They are the single most valuable thing the project owns for
answering *does the seam transfer to a real swap* — the question self-blend
training cannot answer about itself. Weights fitted on self-blends and merely
scored against those swaps contain nothing derived from inswapper, so the
shipped artefact stays clean; what is encumbered is the evaluation set, which
is not distributed. **No swapper is built and none is licensed for training.**

## 7. Validation

A number is admissible only with all of the following:

1. **A permutation null.** Fit on shuffled labels, report alongside. If the real
   number sits inside the null, it is not evidence — that is how 0.316 was
   retracted at p=0.33. `bench.metrics.permutation_null` exists.
2. **Intervals resampled over families**, not rows. Row-wise bootstrap was 26×
   too narrow on DF40's fake side.
3. **Effective sample size reported** (Kish). 509 uniques is the ceiling and the
   real figure will be lower once identities collapse.
4. **A leave-one-generator-out fold**, reported per fold, with the *best* fold
   explicitly distrusted.
5. **A shortcut control.** Fit a model on colour statistics alone, or on the
   JPEG header, and report what it achieves. If that reaches the detector's AUC,
   the detector has measured nothing. The repo has a control that survives this
   on the capture corpus already.

Success is slot D clearing **AUC 0.75** under those conditions. That number is
the existing `auc_floor`, not a new goalpost.

## 8. Deployment

Inference runs CPU-only on the existing instance (8 vCPU, 15 GB, no GPU).
CLIP ViT-L/14 forward cost on that box is **unmeasured and must be benchmarked
before any latency claim**. For scale: the dima806 ViT already profiled here is
ViT-Base/16 (768-wide, 12 layers, ~86M parameters, ~17.6 GFLOPs at 224px), and
CLIP ViT-L/14 is 1024-wide over 24 layers at ~304M parameters and ~81 GFLOPs —
roughly 3.5x the parameters and 4.5x the compute, before any quantisation.
Dynamic INT8 gave 1.52x on the Base model here and is the first lever if
latency is unacceptable.

Training is a separate question. 91K trainable parameters still backprops
through a frozen 304M encoder; a GPU box for the training run, with CPU-only
inference afterwards, is the expected shape. The instance stays CPU-only.

The service already binds `127.0.0.1` with no authentication and is reached
through an SSH tunnel. Exposing the upload endpoint publicly requires an
authenticating proxy first; that is a deployment task, not a detector task.

## 9. Release blockers and failure modes

### 9.1 Commercial release blockers

These are not engineering tasks and will not be closed by any amount of code.
They are listed because the software is to be commercialised and each one can
stop a release after the detector works.

| Blocker | Status | Who closes it |
|---|---|---|
| Biometric-consent basis for using v-CIP captures (even for evaluation) | **open** — owner attestation covers rights in the recording, not biometric processing | Legal review. DPDP Act / GDPR Art. 9 treat biometric data as sensitive |
| Face-swap generator licence | **closed by design** — training needs no generator (§6.2); swaps are evaluation-only and not distributed (§6.4) | — |
| FairFace attribution travelling with shipped weights | **condition, not blocker** | Engineering: already implemented, must not be stripped |
| CLIP ViT-L/14 attribution (MIT) | **condition, not blocker** | Engineering: copyright notice in distribution |

The FairFace entry carries a caveat that applies to this whole product and is
worth restating plainly: **CC BY 4.0 settles copyright and nothing else.** Those
are photographs of people who licensed an image, not people who consented to
biometric processing of their faces. For a product in the identity-verification
space that is a data-protection question, and no dataset licence answers it.

I am not qualified to clear any of the above, and this design does not pretend
to. What it does is keep them visible and keep the training path free of the
one blocker that regeneration cannot fix.

### 9.2 Technical failure modes

- **Slot D measures below 0.75 on our corpus.** The gate stays shut and the
  honest product is a human-review flagger, not an auto-decider. The service
  already abstains correctly, so this failure ships safely.
- **FairFace-to-v-CIP domain shift is too large.** Trained on Flickr
  photographs, deployed on webcam captures under v-CIP lighting and
  compression. This is a *predictable* risk, not a hypothetical, and the
  in-domain evaluation set (§6.4) is what exposes it. If it is severe, the
  answer is domain augmentation matched to capture conditions — not training on
  the captures, which §6 rules out.
- **The two trainable families are too similar.** Self-blending and diffusion
  inpainting could share more artefacts than expected; LOGO would fold high and
  generalisation would be unproven. The DF40 mistake repeated.
- **SFHQ saturates and flatters the mean.** Already measured as too easy.
  Reported as its own fold, never averaged into a headline number.
- **CPU inference too slow.** Quantise (1.52x measured on the Base model here),
  or accept batch rather than interactive scoring.
- **The seam does not transfer to real swaps.** This is the central risk, and
  the one the design is built to expose rather than hide: self-blend training
  assumes the composite boundary is the invariant across swap methods. SBI's
  93.18% on Celeb-DF says it generally is; if the held-out inswapper swaps
  (§6.4) say otherwise on v-CIP-like captures, that is the finding, and it is
  reported rather than averaged away.

## 10. Record

Every measurement lands in `bench/evidence_card.json` with its caveat, in the
form the existing card already uses — including, where it applies, the sentence
that a corpus can neither support nor refute a detector. A card that records
only good news is not a report card.
