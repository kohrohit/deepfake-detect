# A Benchmark Harness That Can Falsify "Better Than Them" — Design

**Date:** 2026-11-10
**Status:** Approved in conversation; implementation plan pending
**Author:** rohit.kohli@scoreme.in with Claude
**Parent spec:** `2026-09-20-deepfake-detection-design.md` (the authority)
**Sibling spec:** `2026-11-10-own-model-detector-design.md` (slot D, the detector this
harness exists to judge)

---

## 1. Problem

The instruction was: build a system better than Reality Defender and DuckDuckGoose.

That sentence has no truth value today, and neither does its denial. There is no corpus on
which this project and those two vendors have both been measured, so "better" is not a claim
that is currently wrong — it is a claim that cannot be checked. This document is about making
it checkable, which has to happen before any work aimed at making it true.

**What the two vendors actually publish.**

| | Reality Defender | DuckDuckGoose |
|---|---|---|
| Modalities | audio, video, image, text | video, audio, image |
| Deployment | SaaS, REST API, private cloud, on-prem | cloud, on-prem, hybrid, API + SDK |
| Headline figures | MCC 0.646, ROC-AUC 0.89 | video 99% acc / 0–5% FAR / 0.8% FRR; image 91% acc; audio 99% acc / <0.5% FAR |
| Benchmark behind them | VendorBench-100 | **none cited** |
| Uncertainty reported | none | none |

VendorBench-100 is **100 still images, 79 fake and 21 real**, hand-selected for difficulty.
An AUC over 21 negatives has error bars wide enough to cover most of a 36-system leaderboard.
This project has already been destroyed once by trusting a corpus whose effective sample size
turned out to be 2.4, and the lesson generalises: a leaderboard position on 100 images is not
evidence about anything. DuckDuckGoose's figures are worse off still — they name no benchmark
at all, so there is nothing to critique and nothing to reproduce.

**What the literature says about numbers like those.** `Deepfake-Eval-2024` (arXiv 2503.02857)
collected deepfakes in the wild and re-measured open-source detectors on them: AUC fell by
**45% for image models, 48% for audio, 50% for video** against the lab benchmarks those models
were published on. NIST's GenAI: Deepfakes 2026 programme reports the same 45–50% operational
collapse. So a vendor's 99% is not merely unverified; the published evidence predicts that a
number like it, obtained the way vendors obtain it, will roughly halve on contact with real
traffic.

**Therefore the first deliverable is not a better detector. It is the instrument that can tell
whether any detector here is better than anything.** Build the scale before claiming the weight.

## 2. Scope

This request decomposes into seven subsystems. Spec'ing them as one document would produce
something nobody can execute, so the decomposition is recorded here and each piece gets its own
spec → plan → build cycle.

| # | Subsystem | State |
|---|---|---|
| 1 | Image detection | specified and planned (slot D) |
| 2 | Video detection | frames are sampled; no temporal model, no fusion |
| 3 | Audio / voice clone | nothing exists |
| 4 | Injection and liveness (virtual camera, replay, spoofed feed) | nothing exists; **not an artefact-detection problem** |
| 5 | Explainability (heatmaps, architecture attribution, compliance summary) | nothing exists; this is DuckDuckGoose's primary differentiator |
| 6 | **Benchmark harness** | **this document** |
| 7 | Platform (auth, SDK, on-prem packaging, quotas, console) | service and dashboard exist; no authentication |

**Text detection is dropped and should stay dropped.** Reality Defender gates it to Enterprise.
AI-text detection does not survive contact with evidence, and shipping it would contradict the
measurement discipline that is this project's actual advantage over both vendors.

**In scope for this document:**

1. A loader for Deepfake-Eval-2024's **image split** into `bench.runner` records.
2. A score-file protocol, so third-party baselines are measured without entering this repo.
3. An evaluation that reports AUC with **domain-grouped** intervals, Kish effective sample
   size, abstention by class, and the two shortcut controls.
4. A reproduction of the paper's published NPR number, as the harness's own correctness check.
5. The corpus registered in `assets/manifest.yaml` and the results written into
   `bench/evidence_card.json`.

**Out of scope** — each needs its own cycle:

- The video and audio splits. Different cost, different detectors, and this project has
  neither.
- Running Reality Defender's or DuckDuckGoose's API on the corpus. Decided against in
  conversation: ToS benchmarking-and-publication clauses are common, and the comparison is not
  needed for an internal instrument.
- A NIST GenAI submission. The route is open (registration at `ai-challenges.nist.gov/genai`,
  participants must own their algorithms, non-US participants face additional vetting) and it
  is the only genuinely independent arbiter — but it is bound to NIST's timeline and is a
  separate decision.
- Publication of any comparison. See §9.
- Any new detector. The harness judges what exists.

## 3. The corpus

**Deepfake-Eval-2024, image split**, from the paper and the dataset card:

| Property | Value |
|---|---|
| Images | **1,975** |
| Real / fake | **767 / 1,208** |
| Web domains of origin | **88** |
| Mode resolution | 1024×1024 |
| Per-item generator labels | **none** |
| Licence | CC-BY-SA-4.0 |
| Access | HuggingFace gate: company or institutional email, plus evidence of deepfake-detection work. Opens to anyone accepting the terms in **May 2027** |

767 negatives against VendorBench-100's 21 is the entire reason to prefer it. It is still one
corpus and it is still 2024 media, which §9.2 treats as a live limitation rather than a
footnote.

**Published baselines on this split**, which are the numbers this harness must sit beside:

| Model | AUC | Accuracy |
|---|---|---|
| UFD | 0.56 | 0.63 |
| DistilDIRE | 0.52 | 0.61 |
| **NPR** | **0.53** | 0.47 |
| commercial models (aggregate) | 0.90 | 0.82 |

Open source is at chance. Commercial systems reach 0.90. This repo's evidence floor is 0.75 and
sits between them, which is the first honest statement of what "better" would have to mean.

**The paper reports no confidence intervals**, for any model, on any split. The statistics in §5
are therefore not a reproduction of the state of the art; they are an addition to it.

### 3.1 Licence posture

CC-BY-SA-4.0 is **ShareAlike**, and that matters more here than the attribution clause.

- As an **evaluation** corpus that is never redistributed and that nothing is fitted on, it is
  clean, and it is the identical posture §6.4 of the sibling spec takes toward the
  `inswapper_128` swaps: weights that merely *score against* a corpus contain nothing derived
  from it.
- If anything is ever **fitted** on these images, ShareAlike becomes a live question about the
  resulting weights, and the answer is not obviously favourable for a commercial product.

**Ruling: nothing in this project trains on Deepfake-Eval-2024, now or later.** It is an exam,
and an exam that has been studied for is not an exam. The shortcut controls in §5 are the sole
exception, because they are fitted *on* the corpus specifically to expose it — they are
diagnostics of the corpus, never shipped artefacts, and §5 requires them to carry a permutation
null for exactly that reason.

## 4. Architecture

Four components, each with one job and a seam the others do not reach through.

### 4.1 `corpora/dfeval.py` — the loader

Reads the gated download into `bench.runner` records, with:

- `generator: None` on every fake. The corpus has no technique labels and inventing one would
  fabricate a LOGO fold (§5).
- `domain` — the originating web domain, which is the resampling family.
- `compression: "unknown"`. The precedent is `corpora.df40` and `corpora.sfhq`: an invented
  `c23` passes `check_compression_coverage` by lying to it.
- `face_detector` / `align` recorded, because `check_uniform_preprocessing` reads them and we
  control both.

**Deduplication by content hash happens before anything else.** This repo has already read 58
distinct images as a pool of 1,088, and the in-the-wild corpus is assembled from social media,
where one image circulates across many domains. A duplicate spanning the real and fake halves
is a labelling contradiction; duplicates within one half inflate every denominator. Both are
counted and reported, never silently dropped.

### 4.2 `bench/scorefile.py` — the baseline protocol

A baseline never enters this repository. It runs in its own environment and emits
`sample_id,score` as CSV; this module ingests that.

The reason is dependency containment, and it is not stylistic: `tests/test_ci_gates.py` enforces
exact `==` pins across two CI legs, and UFD, DistilDIRE and the DeepfakeBench family each carry
their own trees. The second reason is licence containment — several obtainable baselines are
NonCommercial, and under this protocol their weights never sit in the tree of a product that
ships.

The ingest validates and refuses rather than coerces: ids must exist in the corpus, no id twice,
scores finite. It reports **coverage** as a first-class field, because a baseline that scored
60% of the corpus and one that scored all of it do not have comparable AUCs, and nothing in an
AUC says which you are looking at.

### 4.3 `bench/eval_dfeval.py` — the orchestrator

Our detectors run **in-process** through `default_registry` and the existing scoring path, so
the benchmark exercises the code the service actually runs. Baselines arrive as score files.
Both go through the same statistics. Writes `bench/dfeval_report.json` and `.md`.

### 4.4 `bench/metrics.kish_ess`

`(Σn)² / Σn²` over group sizes. Reported beside every n, because a reader handed 1,975 rows
will otherwise read them as 1,975 independent observations. Shared with the slot D plan, which
needs the same function; whichever lands first writes it.

## 5. The statistics, adapted to a corpus with no technique labels

Three of the sibling spec's five admissibility conditions transfer unchanged. Two do not, and
applying them mechanically where they do not fit would be the same error as applying them
nowhere.

1. **Grouped bootstrap — transfers.** `bench.metrics.bootstrap_ci_by_group` resampling the 88
   domains, never rows. Row-wise resampling was 26× too narrow on DF40.
2. **Kish effective sample size — transfers.** Over domain counts, per class.
3. **Shortcut controls — transfer, and matter more here than anywhere.** A model on Lab colour
   means alone, and a model on the JPEG luma quantisation table alone. Lab colour separated
   DF40's halves at 0.843; one JPEG header byte labels 99.2% of the capture corpus. An
   in-the-wild corpus assembled from different platforms is *more* exposed to both, not less,
   because its halves arrive down different upload pipelines. If either control approaches the
   commercial models' 0.90, every number on this corpus is suspect and the report says so
   before it says anything else.
4. **Leave-one-generator-out — impossible, and recorded as impossible.** There are no per-item
   generator labels, so `bench.protocol.logo_splits` raises `UnsplittableCorpusError` and the
   report carries "this corpus cannot fold" in the place where a LOGO table would go. This is
   the single largest thing Deepfake-Eval-2024 cannot do, and it is exactly the property the
   sibling spec calls the only number that predicts field performance. The two corpora are
   complementary for this reason: the FairFace swap corpus folds and is synthetic; this one is
   real and cannot fold.
5. **Permutation null — attaches to the controls, not to the detectors.** `permutation_null`
   refits on shuffled labels; its docstring is explicit that permuting a fixed model's scores
   tests a different and much weaker hypothesis. Our detectors arrive pre-fitted and nothing is
   fitted on this corpus, so there is no fit variance to null out, and the honest uncertainty
   statement for them is the grouped bootstrap alone. The shortcut controls *are* fitted here,
   and each carries a null.

Stating condition 5 this way is a deliberate narrowing of the sibling spec's blanket rule. A
permutation null demanded where nothing is fitted would be theatre, and a project that performs
statistics it does not need will eventually skip ones it does.

## 6. The face-subset problem, stated before it is measured

Deepfake-Eval-2024 is in-the-wild AI media. Not all of it is facial — the paper's labelling
codebook identifies fakes by stylistic artefacts, anatomical implausibilities and sociocultural
implausibilities, with face swapping and morphing as one category among them.

**Every detector in this repository is a face detector.** Slots A, C, D and E all require a
face ROI; the pipeline abstains with `no_face` otherwise. So a large share of this corpus will
be scored by nothing we own.

That is not a defect to be engineered around in this cycle. It is a measurement, and it has to
be reported as one:

- The headline is AUC **over the face-bearing subset**, with that subset's size and composition
  stated beside it.
- `abstention_by_class` is reported for every detector. If abstention correlates with the label,
  every metric above it was computed on a label-selected subsample — the distortion already
  measured on slot A, which abstained on 74.3% of `swap_lowres_paste` fakes against 56.9% of
  reals while that fold carried its best AUC.
- A single corpus-wide number is never reported, because for a face detector on a mixed corpus
  there is no honest one.

The gap this exposes — that the product detects face manipulation and the threat landscape is
wider — is a finding for the roadmap, not a failure of the harness.

## 7. The harness's own correctness check

**NPR is in the paper at AUC 0.53 on this split, and NPR is slot C of this repository.**

So the corpus ships with an oracle. Run our slot C over it; if the loader, the preprocessing
and the scoring path are right, the result should land near 0.53 — inside the grouped interval,
and certainly not at 0.9 or at 0.2.

This is a **gate, not a diagnostic**: if slot C does not reproduce, no other number from this
harness is reported, because a harness that cannot reproduce a published result on a published
corpus has not been shown to measure anything. The same discipline as verifying the CLIP tower
against OpenAI's own module rather than trusting key coverage to prove the weights landed in
the right tensors.

Two caveats that must travel with the comparison. Our NPR is a reimplementation whose constants
are recorded as unverified against Tan et al., and the paper's NPR was run on whole images while
ours scores a face ROI at native resolution. So the target is the right *neighbourhood*, not a
decimal match, and the gate's tolerance is set accordingly and justified in the plan rather
than tuned until it passes.

## 8. Refusals

Each of these has a wrong version that this project has already shipped once somewhere.

| Condition | Behaviour |
|---|---|
| Corpus absent | A message naming the HuggingFace gate and what it requires. Not a traceback, and not an empty corpus that reads as "measured nothing". |
| Unknown `sample_id` in a score file | Refuse, naming the ids. A baseline scoring a corpus it was not given is a configuration error, not a low coverage number. |
| Duplicate `sample_id` in a score file | Refuse. One row silently overwriting another changes the denominator invisibly. |
| Coverage below 100% | Reported prominently beside that baseline's every number, never silently compared. There is no threshold below which a baseline is dropped: dropping one is a judgement the reader should make with the coverage figure in hand, not one the harness makes on their behalf. |
| A subset with one class | `auc: None`. Never a fabricated 1.0. |
| Real/fake duplicate across halves | Refuse the corpus and name the hash. That is a labelling contradiction, not a curiosity. |

## 9. Limitations and release posture

### 9.1 What this instrument cannot do

**It cannot establish that this system is better than Reality Defender or DuckDuckGoose.** It
was scoped as internal evidence, so it compares against published open-source baselines on a
public corpus. Neither vendor has been measured on it by us. The asymmetry is real and
favourable — our numbers will be checkable and theirs are not — but asymmetry is not a
comparison, and a sales claim built on this would be the same move both vendors are already
making.

Two routes make the claim sayable, and both are decisions rather than code: a head-to-head
through the vendors' APIs after a ToS review, or a NIST GenAI submission. Until one of them
happens, the honest internal statement is "we are at X on a public in-the-wild corpus where
open source is at 0.53 and commercial systems are reported at 0.90", and nothing stronger.

### 9.2 Technical limitations

- **One corpus, and it is 2024 media.** Generators have moved. A good number here is evidence
  about 2024 deepfakes circulating on social platforms, not about what arrives at a v-CIP
  session in 2026.
- **No LOGO.** See §5 condition 4. This corpus cannot answer the generalisation question.
- **Our detectors abstain on much of it.** See §6.
- **The baselines are reproductions.** We run UFD and DistilDIRE from their own releases in
  their own environments; a discrepancy against the paper's numbers could be theirs, ours, or
  the environment's, and the report must not resolve that ambiguity by assumption.
- **ShareAlike constrains the future, not the present.** See §3.1.

### 9.3 Owner actions

Neither is engineering and neither can be closed by any amount of code.

| Action | Cost | Unblocks |
|---|---|---|
| Request Deepfake-Eval-2024 access on the HuggingFace gate, with `@scoreme.in` email and a link evidencing deepfake-detection work | one form, unknown turnaround | the entire document |
| Confirm that NonCommercial baselines may be run as internal evaluation, under the eval-only posture already taken on the capture swaps | a ruling | the baseline comparison; without it the harness reports our detectors alone |

## 10. Record

Every number lands in `bench/dfeval_report.json`, and the corpus-level findings land in
`bench/evidence_card.json` under a new `deepfake_eval_2024` id registered in
`assets/manifest.yaml` — including, if the shortcut controls fire or slot C fails to reproduce,
the sentence that this corpus can neither support nor refute a detector. The card already
carries that sentence for DF40. A card that records only good news is not a report card.
