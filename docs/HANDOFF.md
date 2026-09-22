# Handoff — P0 Evidence Core and Benchmark Harness

**As of 2026-09-21.** Branch `p0-evidence-core`. The original **22-of-22-task plan** below is
finished; a second, 6-task plan (composition root — §2, §3) landed on top of it the same day and is
also finished, and a whole-branch review's single fix wave landed on top of both (§3's
"deliberately NOT fixed" block records what it chose to leave). **565 tests green**, ruff clean,
`mypy --strict` clean, coverage **95.12%** local against a ≥85% gate — measured 2026-09-21 after
that wave, see "Verified by hand" under §2. The pinned `requirements-floor.txt` venv was **not**
rebuilt for this last measurement; its previous run reported 553 tests at 95.08%, and only CI's
floor leg will confirm the new count there. The
22-task plan's commits are pushed; the composition-root plan's Task 6 commit (this documentation)
is **not yet pushed** as of this line — deliberately, per that plan's own Task 6 ruling: the branch
carries an open PR, so pushing is left to whoever reviews the commit. **Neither plan's branch has
been merged.**

(The "95%" a previous version of this line claimed is not what the tool reported at the time.
Measured 2026-09-21 (before the composition-root plan): 92.7% locally, 93.4% in the pinned floor
venv — the two environments' coverage tooling counted 868 and 858 statements respectively. The
composition-root plan's own measurement, above, is a later and different number — more statements
now exist. Both sets of numbers clear the gate with room; the local/floor discrepancy is unexplained
in both and nobody has looked into it.)

Read this, then `docs/superpowers/ledger/2026-09-20-p0-execution-ledger.md` — it carries all 111
rulings made during the build, each with what it costs if wrong. The spec
(`docs/superpowers/specs/2026-09-20-deepfake-detection-design.md`) remains the authority, except as
corrected below (§1's dated correction block, and §3).

(**"106" was wrong, corrected 2026-09-21.** Counted as `grep -c '^Ruling'` on the ledger — lines
opening a ruling, 101 of them `Ruling:` and 10 `Ruling on ...` / `Ruling (...)` / `Ruling for ...`
— which gives **111**. Three further rulings are written inline at the end of a finding paragraph
rather than at the start of a line, so `grep -c 'Ruling'` gives 114; the line count is the number
quoted here. In a document whose bar is measured output over paraphrase, a checkable number that is
wrong undercuts every uncheckable claim beside it.)

```bash
cd /home/rohit/Desktop/agents/deepfake && git checkout p0-evidence-core
python3 -m pytest -q          # 565 passed
```

**Corrected 2026-09-21.** A previous version of this handoff said the suite ends in "1 warning"
that was deliberate and must not be suppressed. That is no longer true, and the reasoning behind it
was wrong. The warning was torch's `FutureWarning` about `torch.load`'s `weights_only` **default**
changing — and that default has since flipped to `True` (torch 2.6), which silently turned the
gated `allow_unsafe_load=True` branch into a no-op re-run of the `weights_only=True` attempt that
had already failed one branch above. The documented escape hatch did not work on modern torch.
`weights_only=False` is now explicit, so the warning no longer fires on any torch version. The
evidence that the unsafe path is unsafe was never that warning; it is the unconditional
`logger.warning` immediately above the call, which is untouched.

### Environment drift — what happened, and what now guards it

Every dependency used to be a floor with no ceiling. Local resolved numpy 1.26.4 / opencv 4.10 /
torch 2.4.1; CI resolved numpy 2.2.6 / opencv 5.0.0 / torch 2.14.0. The first CI run failed on
`mypy --strict` for that reason alone, and because Types runs before Tests, **the suite had never
executed under the CI resolution at all** — which is how the `torch.load` defect stayed hidden
behind it.

Now: `requirements-dev.txt` is **exactly `==`-pinned** to the set all four gates were verified
against, and `pyproject.toml` carries consumer-facing ranges with ceilings at the next major.
The floors were raised too — `numpy>=1.24`, `opencv>=4.8`, `pillow>=10.0`, `scikit-learn>=1.3`,
`torch>=2.2` had never been run by anyone, and are now the oldest versions actually exercised.

**Do not over-read the ceilings.** They would not have caught the failure that prompted them: torch
flipped the `weights_only` default in **2.6, a minor release**, which `torch<3` admits. The exact
pins are the guard, and only for direct dependencies — transitive ones still float. Three tests in
`tests/test_ci_gates.py` enforce that pins stay pinned, that ranges stay bounded, and that every pin
sits inside its range (otherwise the `pip install -e .` CI runs after the pinned install would
quietly re-resolve it).

**Now guarded (2026-09-21).** The floors used to be exercised only because two machines happened
to sit at opposite ends of the ranges, and CI ran the upper end alone. `requirements-floor.txt`
pins the declared floors exactly, and `ci.yml` runs all four gates twice — `matrix: deps: [dev,
floor]`, `fail-fast: false`, so a red upper leg cannot cancel the lower one. Verified before
pushing: a clean venv built from `requirements-floor.txt` with no local site-packages passes ruff,
`mypy --strict` (21 files), 502 tests, coverage 93%, and the asset gate; `pip install -e .` after it
re-resolves nothing. Four tests in `tests/test_ci_gates.py` hold it there — the floor file stays
`==`-pinned, each pin *is* the declared pyproject floor rather than merely satisfying it, the dev
tooling is identical across both legs (so a red floor leg is unambiguous), at least one runtime pin
actually differs (so the leg is not a silent duplicate), and CI keeps both legs. All six mutations
of those were run and failed for the right reason.

**Still only two points.** Both endpoints are gates now; the interior of every range is untested,
and transitive dependencies still float in both files. The opencv floor reads `>=4.10.0.84`, not
`>=4.10`, because the pin must equal the declared floor exactly.

To reproduce what CI sees: build a venv from `requirements-dev.txt` (or `requirements-floor.txt`
for the other leg) with no local site-packages and run all four gates there.

### Three project facts corrected, 2026-09-21 — read before trusting the spec's framing

The project owner corrected these mid-session, while Task 6 of the composition-root plan was in
flight. None of the three is reflected in the spec
(`docs/superpowers/specs/2026-09-20-deepfake-detection-design.md`), and the spec is **deliberately
left unedited** — it is the binding authority every review in this plan and the previous one judged
against, so rewriting its framing now would retroactively invalidate those reviews. This block is
where the correction lives until re-framing the spec becomes its own cycle.

1. **This product is not for ScoreMe.** The spec frames the entire problem around ScoreMe's v-CIP
   fraud (its opening paragraph, the 442-session capture table, and measurement (a) below), and its
   open questions table says outright: "Fraud-loss and friction figures for policy calibration —
   Placeholders; **ScoreMe to supply**" (spec line 696; also §9's `E[loss | decision]` framing at
   line 407-419). That sponsor framing is now stale. The engine itself (`src/dfd/`) is
   sponsor-agnostic — nothing in `Policy`, `fuse`, or `decide` names ScoreMe — so nothing in
   `src/dfd/` needed to change. What is stale is the *problem statement*: whose fraud-loss and
   friction numbers calibrate `Policy`, and who the P0 acceptance criteria are ultimately for.
2. **Dataset EULA requests will be sent by `kohrohit@gmail.com`.** Say the consequence honestly
   rather than assuming it away: FF++, Celeb-DF and DFDC agreements generally expect an
   institutional signatory and an institutional email address, so a request from a personal Gmail
   address may be refused outright or simply go unanswered. Separately, and regardless of who signs:
   all three are research-only licences (spec line 614, "research-licensed datasets and
   non-commercial weights"; `assets/manifest.yaml` marks the equivalent weight entries
   `"research-only — VERIFY before any commercial release"`), which matters once this is a
   commercial product rather than an internal ScoreMe tool. Nothing in this build works around that
   — `assert_all_assets_registered` (`src/dfd/asset_scan.py`, wired into CI as the asset registration
   gate) already fails closed on anything not registered as commercially cleared in
   `assets/manifest.yaml`, so an uncleared dataset or weight file cannot silently enter a release.
3. **Hardware is CPU-only for now; a GPU may come later.** Consequence: training EfficientNet-B4 or
   SBI from scratch is not feasible on CPU in any reasonable time, so the first realistic detector to
   actually train (rather than run pretrained-and-abstaining, as today) is a handcrafted-feature
   approach — NPR-style upsampling-fingerprint features and DCT/SRM residuals feeding a light
   classifier — which trains on CPU in minutes. This does not touch P0 inference: inference was
   always specified as CPU-bound (acceptance criterion 5 pins per-detector p95 latency on the target
   hardware, an i5-1235U — spec line 670), so no acceptance criterion changes. It bears on which
   detector is realistic to *train* next, not on how `decide()` runs today.

---

## 0. Resume here (last touched 2026-09-22, after the detector was fitted and measured)

`main` is at `50ab737` — the merge of `feat/sbi-corpus-and-blend-detector` (the SBI corpus builder,
the CPU-only blend-seam detector in slot A, and its fitter). Both CI legs were green on that merge.
Local `p0-evidence-core` and `feat/sbi-corpus-and-blend-detector` still exist, merged and harmless.

### The corpus is 58 distinct images, not 442 sessions. Measured 2026-09-22, by hashing it.

Every previous block in this file, this project's spec, and `corpora/captures.py`'s own module
docstring describe "the 442-session v-CIP capture corpus". That count is real but it is a count of
**session folders**, and nobody had ever checked what is inside them. Hashing every frame file:

```
442 session folders, 1088 frame_NN.jpg files
   -> 58 DISTINCT images
979 of those 1088 files are byte-identical to assets/attack/victim_id.jpg,
   a demo asset, spread over 368 of the 442 sessions
368 sessions consist ENTIRELY of that one image
 74 sessions contain any non-asset frame; they hold 57 distinct images
   26 distinct images in the 7 swapped sessions   (the positives)
   31 distinct images everywhere else             (the negatives)
```

Run the real face pool over it and the number that actually reaches training is smaller still:

```
build_face_pool(435 genuine sessions) -> 19 crops, skipped {duplicate: 715, no_face: 2}
                                          from 15 distinct sessions
build_face_pool(7 swapped sessions)   -> 12 crops, skipped nothing   (evaluation only)
```

**19 genuine face crops.** Not 435 sessions, not 870 frames — nineteen distinct faces.

**What this would have done, left alone.** `training/fit_blend.py` builds its SBI corpus from that
pool and splits it with `split_by_subject`, whose subject is the capture SESSION id. Before the fix
below, 368 sessions each contributed the same image under a different session id, so that split
would have placed one identical picture on **both sides of the holdout** several hundred times over.
The fitter would have run clean, written a model file, and printed a held-out AUC that measured
memorisation of a single demo asset and reported it as generalisation — the project's recurring
defect (§6), this time at the corpus level rather than in a test. It was never caught because every
guard in the chain counted sessions, and the sessions were genuinely there.

**Fixed 2026-09-22.** `build_face_pool` now deduplicates crops by content hash across the whole pool
and counts every drop under the new `DUPLICATE` skip reason, which already flows into
`fit_blend`'s JSON report via `skipped`. The hash is over the ALIGNED CROP, not the source frame,
because the crop is what reaches training. Test:
`tests/corpora/test_face_pool.py::test_byte_identical_frames_across_sessions_yield_one_crop`,
watched failing (3 crops where 1 is correct) before the guard existed.

**What this does NOT fix.** Deduplication removes only byte-identical crops. Two different frames of
the same person in the same session remain near-duplicates and still split apart, which is the
identity-disjointness gap already recorded below — now with a much shorter corpus to hide in. And
nothing here makes 19 negatives and 26 positives a training set. §0a's line "7 positives is not a
training set" was correct and understated: **the negatives are the binding constraint.** Generating
real faces, or licensing them, is not one option among several — it is the whole remaining path.

### Criterion 4 cannot be built from the RD cache. Measured 2026-09-22.

Every previous block calls the 24 cached Reality Defender results "free, already labelled" and makes
the RD adapter the next engineering step. They are free; they are not joinable. The cache key is
`sha256_file(path)` of the submitted image (`core/cache.py:34` in the source project), so the join is
computable — and it mostly fails:

```
24 cached RD results
10 of their source images still exist anywhere on this machine; 14 are gone
 8 of those 10 are assets/demo/applicant_NN.jpg or assets/attack/*.jpg, not captures
 2 are actual capture frames
```

So the head-to-head this criterion asks for has **n = 2**:

| capture session | our label | RD verdict | RD score |
|---|---|---|---|
| `20260831-142708-227903` | swapped **and approved** — one of the five missed attacks | MANIPULATED | 0.98 |
| `20260831-143114-545797` | not swapped | AUTHENTIC | 0.24 |

RD got both right. Two samples is an anecdote, not a benchmark, and no adapter, table or metric
should be built to dress it as one. **Criterion 4 is not blocked on engineering — it is blocked on
data that no longer exists.** Closing it honestly means either re-submitting known-label captures to
RD (spends quota, and `cache/quota.json` is the budget) or recording the criterion as unmeetable
from the cache and saying why. That is a decision for the owner, not a task to pick up.

### The negatives are solved. FairFace, 2026-09-22.

The measurement above said the binding constraint is real faces, not fakes. That constraint is now
lifted, without an EULA, a PI, or an academic address — the three walls `docs/EULA-ACCESS.md`
documents.

**FairFace** (`github.com/joojs/fairface`, mirrored ungated as `HuggingFaceM4/FairFace`): 97,698
real faces, **CC BY 4.0**, already cropped and aligned to **224x224** — which is exactly
`face_pool.DEFAULT_CROP_SIZE` and exactly the resolution `dfd.detectors.blend`'s seam features
assume. Downloaded (2.5 GB) to `/home/rohit/Desktop/agents/datasets/fairface`, outside the repo, as
the capture corpus is. CC BY 4.0 permits **commercial** use with attribution, so unlike FF++ and
Celeb-DF this does not poison a later commercial turn. Registered as `fairface_corpus` in
`assets/manifest.yaml`.

`training/export_fairface.py` writes it into the capture layout, so every already-tested stage —
`load_capture_sessions`, `build_face_pool` (real YuNet landmarks, dedup, ROI clamping),
`build_sbi_corpus` — is reused rather than duplicated. JPEG bytes are copied **verbatim**: the
parquet holds the original files, and re-encoding would stack a second generation of JPEG
quantisation on every real face while its pseudo-fake is blended from decoded pixels — a
corpus-wide shortcut a seam detector would learn in preference to the seam.

Measured end to end on 2,000 exported faces:

```
build_face_pool(2000 FairFace sessions) -> 1995 crops, skipped {no_face: 5}   [25 s, CPU]
                          vs the captures ->   19 crops
```

20,000 are exported at `/home/rohit/Desktop/agents/datasets/fairface_sessions`. The full 97,698 are
one command away.

**Two things this surfaced that were latent bugs, not FairFace quirks:**

- **`build_face_pool` had no ROI clamp and crashed on 49% of real face crops.** YuNet returns a box
  hanging off the frame edge on 146 of the first 300 FairFace images (a tightly-cropped face fills
  the frame, so the box overshoots). `measure_quality` slices `frame[y:y + h, x:x + w]` unclamped,
  a negative origin slices from the FAR end of the array, and `cv2.cvtColor` raises on the empty
  crop. `dfd.pipeline._clamp_roi` and `dfd.faces.align` each already solved this privately;
  `build_face_pool` was the third site and had nothing. The rule now lives once, as
  `dfd.faces.clamp_roi`, and pipeline delegates to it. This is §6's "correct about the case it was
  shown, blind one level down" again. It never fired on the captures because those faces sit well
  inside the frame — it would have fired the first time anyone pointed this at a public dataset.
- **No FairFace crop bands `high`, and the reason is not the one it looks like.** `dfd.quality`'s
  docstring already asks for this ("Thresholds here are starting values ... so they can be set from
  data rather than from intuition"), so the distribution was recorded rather than guessed. Over
  2,990 FairFace crops and the deduped capture pool:

  | corpus | n | iod p50 | blur p50 | forced `reject` by blur | by iod | bands |
  |---|---|---|---|---|---|---|
  | FairFace crops | 2990 | 79.2 | 29.6 | **1070** | 111 | medium 1689, reject 1132, low 169, high 0 |
  | captures, genuine | 19 | 90.4 | 86.5 | 5 | 0 | medium 12, reject 5, low 1, high 1 |
  | captures, **swapped** | 12 | 110.0 | **14.7** | **11** | 0 | reject 11, medium 1 |

  Interocular distance does cap the `high` band — `MIN_IOD_HIGH = 96.0` against a FairFace p95 of
  88.6 makes `high` all but unreachable on a 224x224 pre-aligned crop, where iod is a different
  quantity than on a full camera frame. But it is **not** what drives the `reject` band: blur is.
  1,070 crops fall below `MIN_BLUR_LOW = 20.0` against 111 below `MIN_IOD_REJECT`. Laplacian
  variance on a downscaled, re-compressed 224x224 thumbnail is simply a smaller number than on a
  full frame.

- **The serious one: this project's only real fraud almost all bands `reject`, on blur.** 11 of the
  12 swapped crops have `blur_var < 20`; their median is 14.7 against 86.5 for genuine crops from
  the same corpus. Their interocular distances are fine (89-167). **A detector with any quality
  floor above `reject` would never be consulted on the five missed attacks the whole project exists
  to catch** — `decide` would abstain for quality reasons and never score them.

  The mechanism is not a threshold being slightly off. A swapped face is generated at a fixed,
  usually lower resolution and upsampled into the frame, which destroys exactly the high-frequency
  detail `_laplacian_var` measures. **So the quality gate reads the artifact as an absence of
  evidence.** Blur is confounded with the thing being detected, and the abstention mechanism is
  anti-correlated with the signal — it is most likely to refuse to look precisely when there is
  something to see. `corpora/sbi.py`'s `RESCALE_RANGE = (0.70, 1.00)` encodes the same physical
  fact deliberately, as a *feature*; `dfd.quality` encodes it accidentally, as a *disqualification*.

  **n = 12. Do not treat this as established** — it is one corpus, one swap tool, and a sample far
  too small to set a threshold from. It is, however, the sharpest prediction available about what
  the pipeline will do the first day it is calibrated, and it is cheap to test properly.

  **Not fixed here, deliberately.** Re-tuning a threshold to make a number look better, before the
  number exists, is how you get a detector that agrees with you. The right fix is probably not a
  new constant at all: it is separating "the image carries too little information to judge" from
  "the image carries a low-frequency signature", which are the same measurement today.

### What else open source could supply, and what it could not (2026-09-22)

Swept for anything reachable without a form, a PI or an academic address. Two things came back, and
only one of them is worth much.

**Taken — `df40_eval_subset`.** A repackaging of the DF40 test split on HuggingFace
(`pujanpaudel/deepfake_face_classification`), ungated: 3,212 test and 3,212 val images, balanced
real/fake, 256-1024px. DF40 itself spans 40 techniques — 10 face-swap, 13 reenactment, 12
entire-face-synthesis, 5 editing — which is why it was fetched, because leave-one-generator-out
needs several generators to hold out. **It does not deliver that.** Verified by listing both
archives: the repackaging is a flat `fake/` vs `real/` split with **no per-technique label**.
Filenames fall into families that may track technique, but using a guessed grouping as a generator
axis is precisely how this project's recurring defect (§6) starts, so it is not used as one.

What it is: an ungated multi-technique **binary sanity benchmark**, far better than anything the
project had. What it is not: a LOGO benchmark. Its `real` half is also drawn from upstream
forensics corpora, so those frames are encumbered too — they are not free real faces.

Its HuggingFace page declares `apache-2.0`. That is not authoritative: DF40 restricts itself to
CC BY-NC-4.0, and a derivative cannot grant rights the upstream withholds. Registered as
NonCommercial. **A permissive tag on a repackaged dataset is not evidence of anything** — this is
the second time today a headline licence claim failed on contact with its source.

**Not taken, and why:**

- **OpenFake** (`ComplexDataLab/OpenFake`, CC BY-NC-4.0, ungated) — **3.4 TB**, and it is
  text-to-image political imagery, not face swaps. Wrong content at an impossible size.
- **SFHQ** (MIT, ~425k synthetic faces) — the licence is ideal and the GitHub repo is MIT, but the
  images themselves are distributed through Kaggle, which needs an account. The HuggingFace mirror
  `bitmind/SyntheticFacesHQ` declares **no licence at all**, so it cannot be treated as MIT on the
  strength of sharing a name. Needs a Kaggle login — see the owner-action list.
- **DeepfakeBench / SBI pretrained weights** — both obtainable, both NonCommercial (verified above).
  Worth having as research-track baselines, but each needs its author's framework wired in to run,
  which is a task, not a download.

### The detector is fitted, and measured on an unseen corpus it is worse than a coin. 2026-09-22.

Everything above this line was about supply. This is the first section in this document that reports
what the detector **does**, because until today there were no weights to ask.

**Fitted.** `training/fit_blend.py` over 10,000 FairFace sessions: 9,974 aligned crops (23 no face,
3 duplicate), self-blended by `corpora/sbi.py` into 19,948 samples, split 13,964 train / 5,984 test
over 9,974 subjects — subject-disjoint, never row-disjoint, because a crop and the pseudo-fake made
from it share a face. **Held-out AUC 0.870.** That is the number to be careful with: the fakes it
was tested on are self-blends of the same FairFace photographs the model trained on, so 0.870 says
the seam features separate *a FairFace face from a warped copy of itself*. It is an in-family number.

**Measured on DF40.** `python3 -m bench.eval_df40` over the ungated DF40 test split — 3,207 records
from 3,212 files (2 no face, 3 duplicate crops), a corpus this model has never seen and whose fakes
were made by techniques it has never seen. `bench/df40_report.md` is committed. The result:

| | blend_seam on DF40 |
|---|---|
| AUC | **0.289** |
| TPR@FPR=1% | **0.000** |
| ECE | 0.485 |
| abstained | 15.4% (495 records, all below the quality floor) |
| n | 3,207 |

**0.289 is not "bad", it is inverted.** A coin is 0.500. This model ranks DF40's *real* frames as
more fake than its fakes: mean score 0.148 on reals against 0.027 on fakes. Nothing here is
salvageable by flipping the sign — a sign chosen because it helps on the test set is the test set
fitting the model.

**The obvious explanation is wrong, which matters.** DF40's labels are not balanced across
resolution (fakes run 559 at 256px / 1,000 at 512px / 42 at 1,024px; reals 277 / 1,329 / 0), so
resolution is the first thing to suspect. It does not explain it. Sliced by source resolution:

| bucket | n scored | AUC |
|---|---|---|
| 256px | 350 | 0.494 — chance |
| 512px | 2,320 | **0.232** — inverted |
| 1,024px | 42 | all fake, unmeasurable |

Controlling for resolution leaves the inversion exactly where it was. So this is not a resampling
artifact; it is the features. The seam features read high-frequency energy in concentric annuli, and
on DF40 the thing with the most high-frequency structure is a *real* compressed video frame from a
forensics corpus, not a synthesised face — DF40's entire-face-synthesis and reenactment fakes are
smooth. The detector is reading compression and texture, calling that a seam, and the corpus it was
fitted on could never have told it otherwise: FairFace photographs are clean Flickr stills, so
"clean means real" was never contradicted in training.

**What this measurement does and does not license you to say:**

- It *is* honest cross-corpus evidence, over an unseen multi-technique corpus. That is strictly more
  than this project had yesterday, when the pipeline abstained and there was nothing to measure.
- It is *not* a leave-one-generator-out result, and it is guard-waived — two of the five spec §8.2
  guards cannot pass on this data (see the waiver block at the top of the report). Do not quote the
  number without the waiver.
- It *is* a refutation of one specific claim: that a seam detector fitted on licence-clean real
  faces alone transfers to fakes in the wild. Measured, it does not. The SBI paper's own result
  stands on FF++ frames as the real half; swapping in Flickr portraits is not the same experiment,
  and this is what that substitution costs.

**The quality floor also matters more than expected.** 495 of 3,207 records (15.4%) abstained at
`below_quality_floor`, 494 of them banded `reject` — overwhelmingly the 256px images, of which only
350 of 836 were scored at all. On a corpus where a sixth of the evidence is refused, an operating
point set from the scored sixth-fewer is not the operating point the field will see.

### The licence questions the manifest said to VERIFY are now verified

Checked at source 2026-09-22, replacing two "VERIFY before any commercial release" placeholders:

| asset | verified licence | commercial |
|---|---|---|
| SBI pretrained (`mapooon/SelfBlendedImages`) | "freely available for research purpose. For commercial use: A license agreement is required" — **and** trained on FF-raw/FF-c23, so FF++ derived data regardless | **no**, twice over |
| DeepfakeBench weights (`SCLBD/DeepfakeBench`) | CC BY-NC-4.0; its own table marks FF++ "Rights Cleared: NO" | **no** |
| FairFace | CC BY 4.0, attribution required | **yes** |

So the two tracks must never mix, and the manifest's `commercial_use` flag is what keeps them
apart: **research baselines** (SBI, DeepfakeBench — an immediate cross-dataset comparison under the
live research-only ruling) and **the shippable path** (self-blends over FairFace, fitted here,
`license: owned`). A benchmark number from the first can never become a weight file in the second.

**Also worth knowing, not yet taken:** SFHQ (`SelfishGene/SFHQ-dataset`) is ~425,000 synthetic
faces under **MIT**, with no depicted real person at all — which sidesteps the biometric-consent
caveat below entirely. Part 3 (118,358 images, pure StyleGAN2 sampling) is the cleanest: parts 1, 2
and 4 derive from other datasets or from Stable Diffusion, whose own terms would need reading. Not
downloaded. It is the right control set for asking whether a seam detector fitted on real faces
also fires on synthetic ones.

**The caveat that licence fields do not answer.** CC BY 4.0 settles copyright and nothing else.
FairFace is photographs of real people who licensed an *image*, not people who consented to
biometric processing of their *face*. For a product in the identity-verification space that is a
data-protection question — GDPR Art. 9 and India's DPDP Act both treat biometric data as sensitive
— and it sits beside, not inside, the licence. Recorded in the manifest entry; it is the same
class of open question as the owner attestations, arrived at from a different direction.

**Next, in order:**

1. **Separate "too little information" from "low-frequency signature" in `dfd.quality`.** This now
   blocks everything downstream. The distribution is recorded above; the finding that matters is
   that 11 of 12 swapped crops band `reject` on blur, so a quality floor above `reject` would
   abstain on the fraud this project exists to catch. Detector floors and per-band calibration both
   read the band. Highest leverage task on this list, and it is a design question, not a constant.
2. **Rule on what the captures are now FOR.** With FairFace supplying negatives, the 26 positive
   images in the 7 swapped sessions are this project's only real fraud and should be spent as a
   held-out evaluation set, never as train. The EULA route (§0a step 1) now buys *evaluation*
   breadth and the several generators LOGO needs — not training data, which is solved.
2. **Find an academic signatory** — unchanged from §0a, and more load-bearing than it was.
3. **Criterion 2 — an embedder.** Unchanged, and now cheap: 58 images is a trivial embedding job,
   and it would settle the near-duplicate question deduplication cannot reach.
4. **Do not run the fitter for a number yet.** It remains the owner's reserved decision (§0a item 4),
   and the measurement above is why: with 19 genuine crops, whatever AUC it prints will be an
   artefact of the split, not a detector claim.

---

## 0a. Previous resume block (2026-09-21, after PR #2 merged and the three questions were answered)

**Both PRs are merged and the workspaces are gone.** `main` is at `c058934`. PR #2 (docs only —
the final review's rulings appended to the committed ledger) was squash-merged, its branch deleted
on both ends, and `.superpowers/sdd/` deleted per §7 now that its stated precondition holds. Local
`p0-evidence-core` still exists, merged and harmless.

**The two questions that blocked all data work are answered.** The owner ruled, 2026-09-21:

1. **Research / internal only — not a commercial product (for now).** So the FF++, Celeb-DF and
   DFDC EULAs are worth pursuing, and §4's critical path is live rather than moot. §1's correction
   block still stands on the two things this does *not* change: the requests go from
   `kohrohit@gmail.com`, a personal address these agreements generally do not expect, and
   `assert_all_assets_registered` still fails closed on anything `assets/manifest.yaml` does not
   mark commercially cleared — so a later commercial turn cannot silently inherit research-only
   weights. This ruling makes the EULA wait the longest pole again; start it before any engineering.
2. **The labelled data is available, with rights — and it is on this machine.** Verified by running
   the loaders against it, not by asking:

   ```
   D=/home/rohit/Desktop/agents/fraud_gff/deepfake_detection
   load_capture_sessions("$D/captures")  -> 442 sessions, 0 skipped
   load_rd_cache("$D/cache")             -> 24 results, 10 models each
   ```

   442 sessions (299M, frames present as `frame_NN.jpg`), 7 `swapped`, **5 `swapped AND approved`**
   — the five that are the actual fraud, ids `20260826-221956-387743`, `20260827-104716-349039`,
   `20260829-010524-969870`, `20260831-142514-700890`, `20260831-142708-227903`, all `LIVE` to the
   scan, all 5 frames. 258 approved / 184 not; scan verdicts 162 LIVE / 144 NOT LIVE / 136 absent;
   frame counts 1–5 (163 sessions have 3). RD cache: 17 MANIPULATED / 7 AUTHENTIC. Every number
   §1 argues from reproduces exactly. **The corpus lives outside the repo** and both loaders take a
   root path, so nothing is wired to that path yet — that is a deliberate choice to preserve, not an
   omission to fix.

**Re-confirmed by the owner 2026-09-22: this product is not ScoreMe's.** That settles the framing
question §1's correction block raised — and sharpens a different one. The 442 sessions are v-CIP
captures; the product they would now train a detector for is not the one they were recorded for.
`assets/manifest.yaml`'s `blend_seam_weights` entry says `license: owned`, and that claim was audited
and holds **for the third-party question only** — no licensed dataset, weight file or encumbered code
reaches those coefficients. It does not establish the right to fit and ship a model on the captures
themselves. That rests on two owner attestations, now recorded verbatim in the manifest's
`owner_attestation` field: that the corpus is available with rights (2026-09-21), and that the
product is not ScoreMe's (2026-09-22). Neither has been independently verified and no one with legal
standing has looked at it. **This is the one open question on the licence-clean path that engineering
cannot close**, and it is worth closing before a model trained on that corpus is distributed rather
than after.

**The consequence nobody had written down: 7 positives is not a training set.** The CPU-feasible
first detector (NPR-style upsampling fingerprints, DCT/SRM residuals, light classifier — §1's
correction block 3) cannot be *trained* on 5–7 swapped sessions however licence-clean they are. They
are an evaluation set, and a precious one. Generating a swap corpus is therefore required on the
research path too, not only on the commercial one §1 described — the ruling above changes who may
supply the *real* faces, not whether swaps must be made. Treat the 442 as test, never as train.

**Next, in order:**

1. **Find an academic signatory** — the owner's ruling, 2026-09-21, after §4 showed only DFDC is
   reachable alone. The working packet is `docs/EULA-ACCESS.md`: what each agreement requires, an
   outreach note to a prospective PI, and form text ready to paste. It also records the constraint
   that decides what these datasets are *for* — a model trained on FF++ or Celeb-DF is "derived
   data" under their terms and can never ship commercially, so the research route buys a
   **benchmark**, not training data. Settle the FF++ clause 6 / employer question in §2 of that file
   before anything is submitted.
2. **Criterion 4 — the RD adapter.** The 24 cached results are free, already labelled, and the
   benchmark runner still does not call `decide()`, which is exactly why criteria 4 and 11 are open.
3. **Criterion 2 — an embedder.** `check_identity_disjoint` now *refuses* ids with no embedding
   rather than skipping them; a partial-embedding pipeline must omit unembeddable ids explicitly.
   The same gap also sits under `training/fit_blend.py::split_by_subject`: its "subject" is
   `corpora.sbi`'s `subject_id`, which is set to the capture SESSION id, not a person identity, so
   that split is session-disjoint rather than identity-disjoint and a person enrolled in more than
   one session can land on both sides of it. The absent embedder is therefore not only a benchmark
   gap — it is a correctness gap for the blend-seam fitter too.
4. **The blend-seam detector now exists and is wired in.** `corpora/sbi.py` (the self-blend corpus
   builder), `src/dfd/detectors/blend.py` (`seam_features`, `BlendDetector`, slot A) and
   `training/fit_blend.py` (the fitter) are implemented and tested. `default_registry()` now
   registers three detectors — `blend_seam`, `npr`, `effnet_b4` — instead of two, and
   `assets/manifest.yaml` carries `blend_seam_weights` as an owned asset at the fixed path
   `assets/models/blend_seam.npz`, licensed `owned` (no third-party dataset or model contributed)
   and `commercial_use: true`, with its provenance — fitted in this repo by `training/fit_blend.py`
   from the project's own capture corpus — recorded in that entry's `source` field.
   **The fitter has not been run against the real capture corpus.** No model
   file exists on this machine, so there is no accuracy number — not measured, not estimated, not
   implied — and `bench/blend_seam_report.json` does not exist. The only measurement that exists is
   on synthetic fixtures in the test suite, and it is explicitly not a detector claim: it documents
   that pure-noise features passed the old `> 0.5` bar on 19 of 40 seeds, which is why that test's
   bar is now 0.9. Running the fitter for real, against
   `/home/rohit/Desktop/agents/fraud_gff/deepfake_detection`, is the next step and is a decision the
   project owner has reserved — it produces the first model file and the first honest accuracy
   number together.

**Do not read "it runs" as "it detects".** Every real input still abstains, correctly — see §3. This
now applies to `blend_seam` too: even once a model file exists at `assets/models/blend_seam.npz`,
`dfd score` will keep returning `insufficient_evidence` on every real input, because
`Calibrator.to_evidence` still returns `uncalibrated_for_band` for every band — no calibration curve
has been fitted yet. A third detector in the registry does not mean the pipeline decides anything.

---

## 0b. Previous resume block (2026-09-21, after PR #1 merged)

**PR #1 IS MERGED.** `main` is at merge commit `f6ddeaf`; the composition-root plan is complete and
in. 565 tests green on merged `main`, coverage 95.12%, ruff and `mypy --strict` clean, both CI legs
(`gates (dev)` and `gates (floor)`) green on `f52ac9a`. Everything below this block is the layered
history of earlier sessions — read it for *why*, not for current state.

```bash
git clone git@github.com:kohrohit/deepfake-detect.git && cd deepfake-detect   # main has it all
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .
python3 -m pytest -q                                      # 565 passed
python3 -m dfd score <any jpg or png>                      # insufficient_evidence, exit 0
```

**What exists now that did not:** `decide()` in `src/dfd/pipeline.py` — the composition root — plus
`normalize()`, a `Policy` object applied by `fuse` and recorded by the audit record, `stage_reasons`
on `AuditRecord` (schema 2), and the `dfd score` CLI. `fuse`, `Calibrator.to_evidence`,
`build_audit_record`, `load_image`/`load_video` and `detect_faces` have non-test callers for the
first time, and the decode-bomb limits are finally exercised through a real caller.

**Three things to do next, in order:**

1. **Merge PR #2** — https://github.com/kohrohit/deepfake-detect/pull/2, docs only. It appends the
   final review's rulings to the committed ledger. §7 below says to delete the `.superpowers/sdd/`
   workspaces once the PR merges, on the stated rationale that the ledger holds every ruling — which
   was not true until PR #2, because the fix wave never touched the ledger. Merge it, then the
   workspaces are safe to delete.
2. **Decide: commercial or research?** This settles the entire data strategy. Research → the
   FF++/Celeb-DF/DFDC EULAs are worth pursuing (with the personal-address friction noted in §1's
   correction block). Commercial → those datasets are off the table regardless of EULA approval, and
   the plan becomes licence-clean real faces plus self-generated swaps. The asset-scan gate already
   enforces this.
3. **Confirm the data.** Are the 442-session capture corpus and the 24 cached Reality Defender
   results still available, with rights? They are the only labelled data here, and every measurement
   in §1 came from them. If yes, the CPU-feasible first detector (NPR-style upsampling fingerprints
   and DCT/SRM residuals with a light classifier — minutes to train on CPU) can be built against
   them now. If no, the first job is generating a swap corpus, and §1 and the spec both overstate
   what evidence is reachable.

**Do not read "it runs" as "it detects".** Every real input abstains, correctly — see §3.

---

## 0c. Previous resume block (2026-09-21, before the merge)

**Update, later the same day (composition-root plan, Task 6 of 6): the "Nothing is uncommitted or
unpushed" line below is no longer true.** The composition-root plan (§2, §3, §5) is complete —
`decide()` and `dfd score` now exist — and this documentation commit sits on top of the PR branch
**committed but deliberately not pushed** (that plan's own Task 6 ruling: pushing to a branch
carrying an open PR is left to whoever reviews the commit). "502 tests" and "§5.3" a few lines down
are this section's own history, from before the composition-root plan ran; they are now 553 tests
and struck respectively (§3, §5 above). The rest of this section is left as the record of what the
*previous* session did.

**PR #1 is open and green:** https://github.com/kohrohit/deepfake-detect/pull/1 — 84+ commits,
`MERGEABLE`, all four CI gates passing. Nothing is uncommitted or unpushed.

On a fresh machine:

```bash
git clone git@github.com:kohrohit/deepfake-detect.git && cd deepfake-detect
git checkout p0-evidence-core
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .   # now exactly pinned = CI's env
python3 -m pytest -q                                      # 502 passed
```

`gh` note: this repo is `kohrohit/*`, and `gh` may be active as a different account
(`gh auth switch -h github.com -u kohrohit`). `gh pr edit` fails here with a Projects-classic
GraphQL deprecation; use `gh api -X PATCH repos/kohrohit/deepfake-detect/pulls/1 -F body=@file`.

**What this session did, beyond opening the PR:** CI failed on its first run and exposed two real
defects that local could not see (opencv≥5 type stubs; torch≥2.6 flipping the `torch.load`
`weights_only` default, which had silently disabled the `allow_unsafe_load` branch). Both fixed.
Dependencies are now pinned, with three new drift gates in `tests/test_ci_gates.py`. See §1's
correction block and the environment-drift section below.

**Then, this session:** closed that last gap — `requirements-floor.txt` plus the two-leg CI matrix
described above, with four new drift tests and all four gates verified green in a clean floor venv.
502 tests.

**Next, in order:** merge the PR (or get it reviewed); start the dataset EULAs — §4, still the
longest pole and still not started; then the composition root in §5.3.

---

## 1. The three measurements that justify the design

Unchanged from the previous handoff, and still the reason this system is built this way.

**(a) Your own fraud was approved five times.** Of 442 v-CIP sessions, five are `swapped=true` AND
`approved=true`. Face match was defeated (0.036 rejected → 0.857 accepted with the swap). Liveness
passed by design: a live human drives the puppet.

**(b) Reality Defender's ensemble is one model wearing ten hats.** `rd-pine-img` alone reconstructs
the aggregate at 0.992 separation across 24 cached responses. The aggregate tracks the maximum, so
its false-positive rate approximates the *union* of its members' FPRs.

**(c) An Apache-2.0 open detector scores 0.011 AUC on your fraud.** `dima806` rates the five
approved swaps as *more real* than genuine applicants (mean P(fake) 0.002 vs 0.005). **There is no
off-the-shelf shortcut.**

---

## 2. What is true now

Everything in `src/dfd/` (engine), `bench/` (instrument) and `corpora/` (loaders) is implemented,
reviewed, and green. The additions this session:

| Module | What it does |
|---|---|
| `bench/protocol.py` | LOGO splits, identity-disjoint **by construction**, video-whole |
| `bench/robustness.py` | perturbation surface incl. screenshot and print re-capture |
| `bench/adversarial.py` | white-box PGD baseline |
| `bench/runner.py` + `report.py` | the benchmark run and its markdown report |
| `corpora/` | RD cache + 442-session capture loaders |
| `src/dfd/audit.py` + `errors.py` | immutable, tamper-evident decision record (spec §7.2) |
| `src/dfd/limits.py` | decode-bomb defence, enforced **before** allocation |
| `src/dfd/asset_scan.py` | release gate that cannot pass without examining files |
| CI | ruff, `mypy --strict`, coverage ≥85%, asset gate — all cleared **and** enabled |
| `src/dfd/policy.py` | `Policy` frozen dataclass + `DEFAULT_POLICY`; `fuse` takes `policy=` |
| `src/dfd/pipeline.py` | `normalize()` + `decide(path, ...) -> AuditRecord` — the composition root |
| `src/dfd/cli.py`, `__main__.py`, `[project.scripts] dfd` | `dfd score <path>` runs from a shell |

### Verified by hand, 2026-09-21: `dfd score` end to end

All four gates green in both environments (local venv and the pinned `requirements-floor.txt`
venv): `ruff check src bench corpora`, `mypy --config-file mypy.ini`, and
`pytest -q --cov=src/dfd --cov-fail-under=85` — **553 tests passed** in both, coverage 94.94% local
/ 95.08% floor (both comfortably clear the 85% gate; the two environments still count a slightly
different statement total, 1067 vs 1057, the same unexplained-but-harmless discrepancy noted above
for the previous session). **Re-measured locally after the review's fix wave, 2026-09-21: 565 tests
passed, coverage 95.12% over 1085 statements, ruff clean, `mypy --strict` clean over 25 files.** The
floor venv was not rebuilt for that re-measurement. `mypy --strict`: "Success: no issues found in 25 source files" in both.

This machine (not CI) has `assets/models/face_detection_yunet_2023mar.onnx` on disk — it is
gitignored and absent in CI. That changes the *face* stage's reason but not the outcome: no detector
weights exist anywhere (also gitignored, also absent everywhere), so both detectors abstain
regardless, and the verdict is `insufficient_evidence` either way.

Ran exactly this, on this machine:

```
$ python3 -c "
import cv2, numpy as np
rng = np.random.default_rng(0)
cv2.imwrite('/tmp/dfd-demo.png', rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
"
$ python3 -m dfd score /tmp/dfd-demo.png | python3 -m json.tool ; echo "exit=${PIPESTATUS[0]}"
```

stderr (the CLI's human-readable summary line), then stdout piped through `json.tool`, verbatim:

```
verdict=insufficient_evidence llr=0.00 band=unmeasured contributing=0/2 faces=no_face digest=0326b982
{
    "created_at": "2026-09-21T10:21:49.606609+00:00",
    "evidence": [
        {
            "abstained": true,
            "detector": "effnet_b4",
            "llr": 0.0,
            "raw_score": null,
            "reason": "weights_absent",
            "version": "0.1.0"
        },
        {
            "abstained": true,
            "detector": "npr",
            "llr": 0.0,
            "raw_score": null,
            "reason": "weights_absent",
            "version": "0.1.0"
        }
    ],
    "input_sha256": "414543ee013c9b57f0f79ebfc8c3bb2d4b302ea62bd0c9d19c5e05ba5c53ca93",
    "llr_total": 0.0,
    "model_versions": {
        "effnet_b4": "0.1.0",
        "npr": "0.1.0"
    },
    "ood_score": 0.0,
    "policy_version": "p0-default-v0",
    "posterior": 0.5,
    "quality_band": "unmeasured",
    "sample_id": "dfd-demo",
    "schema_version": "2",
    "stage_reasons": {
        "faces": "no_face",
        "frames_with_face": "0/1",
        "max_faces_in_frame": "0"
    },
    "threshold": 1.0,
    "verdict": "insufficient_evidence"
}
exit=0
```

**One line of stderr in that transcript is a property of this machine, 2026-09-21.** `main` now
configures logging (`WARNING` by default, `-v` INFO, `-vv` DEBUG, formatted, on stderr), so on a
checkout WITHOUT the YuNet weights — CI, and most machines — the same command prints
`WARNING dfd.faces: Face detector weights absent at ...` above the summary line. stdout is
unaffected at any verbosity. Design spec §6 describes this; it previously claimed stderr carries
one line, which was only ever true where the weights happened to exist.

`faces=no_face` here (not `weights_absent`) is because YuNet's `.onnx` **is** present on this
machine — it ran, and found no face in random noise, which is the correct outcome for that input on
this machine's weight state. A CI checkout, with no YuNet weights, would instead report
`faces=weights_absent` for the same reason the two detectors do. Either way, both detectors abstain
(`weights_absent`, always, everywhere — no detector weights are vendored anywhere), `llr_total` stays
0.0, and the verdict is `insufficient_evidence` — as expected, since nothing exists yet that could
produce any other verdict. `input_sha256` matches `sha256sum /tmp/dfd-demo.png` exactly (checked).

---

## 3. What is NOT true — read this before claiming anything

Four acceptance criteria are **unmet**, now disclosed in the plan's Known-gaps block:

- **Criterion 2 (identity leakage).** No ArcFace embedder exists. `identity_report` is hardcoded
  `None`. This is the largest gap; do not close P0 without it.
- **Criterion 4 (head-to-head vs RD).** The loaders exist and **nothing consumes them**. No adapter
  joins them to `run_benchmark`; no RD table is rendered. **And one cannot usefully be written** —
  measured 2026-09-22 (§0): 14 of the 24 cached results' source images no longer exist on this
  machine and 8 more are demo assets, leaving n=2 real capture frames. This criterion is blocked on
  vanished data, not on the adapter.
- **Criterion 8 (adversarial).** `adversarial_tpr` has no caller — P0 detectors abstain without
  weights, so there is nothing to attack yet.
- **Criterion 11 (demographic parity).** The guard is built and **never invoked**; `ParityReport`
  never reaches `RunRecord`.

Five more limitations, each recorded with its severity:

- **The CI asset gate provides no protection in CI.** Weight files are gitignored, so it runs with
  `allow_empty=True`. It protects only where run with assets present. **A green CI run is not
  evidence that criterion 6 holds.**
- **The benchmark's quality measurements are computed from fabricated landmarks, not detected
  ones (found 2026-09-21, in fix round 1 of this task).** `bench/runner.py:111` builds every
  `Observation` via `lm = np.array([[w * 0.35, h * 0.4], [w * 0.65, h * 0.4]])` — two eye positions
  fixed at 35%/65% of frame width and 40% of frame height, on every record, regardless of where (or
  whether) a face is actually in the frame — then passes `lm` and a whole-frame `roi = (0, 0, w, h)`
  to `measure_quality`. No detector produced these points; `bench/runner.py` never calls
  `detect_faces` or `normalize()` (see above — it bypasses ingest entirely). Consequence: every
  quality number the benchmark reports — interocular distance, and everything `measure_quality`
  derives from it, hence every quality band — is computed against these two fixed, fabricated
  points rather than a real face's real landmarks. This is exactly what the next item, quality
  banding's blindness to the robustness surface, is measured against: that finding was never
  checked against a real interocular distance to begin with.
- **Quality banding is blind to every perturbation in the robustness surface.** Blur halves
  high-frequency energy and bands identically; so do both re-capture paths. The abstention
  mechanism will never route a recaptured sample to manual review on quality grounds. Note the
  coupling: the robustness sweep reuses the clean quality object *because* banding is blind, so
  fixing one invalidates the other's construction.
- **LOGO is computed but is not a trained protocol.** Each fold's train side is unused. It becomes
  load-bearing when calibration lands, and is where the operating threshold must be frozen.
- **`DfdError` is not yet the root of everything.** 19 bare `ValueError`/`RuntimeError` sites remain
  in `fusion.py`, `calibration.py`, `detectors/`, `ingest/`. The `errors.py` docstring says so.

### Five more, found by the whole-branch review and deliberately NOT fixed (2026-09-21)

These are recorded, not repaired. Each is a design decision belonging to a later milestone, and the
fix wave that found them judged that changing them with no fitted calibrator and no detector weights
anywhere would be changing behaviour nobody can measure. They are also in the plan's known-gaps
block (gaps 7-11). The first one is the serious one.

- **The band used for calibration is not the band the detector was scored on.** `pipeline.py` takes
  `_worst_band` over **all** observations; each detector separately drops observations below **its
  own** floor via `filter_by_quality_floor`. Those are different sets, so on any mixed-quality
  sample the record's `quality_band` names a regime the detector never saw. Demonstrated by
  execution on this machine, 2026-09-21: a 32-frame clip of 31 `high` frames plus one flat `reject`
  frame, a detector with floor `low`, a calibrator fitted on `high`. The detector scores the 31 high
  frames and returns raw 0.3135; `decide` calibrates that on `reject` and the evidence row comes
  back `{"llr": 0.0, "abstained": true, "reason": "uncalibrated_for_band"}`. The identical clip with
  the one flat frame removed returns `{"llr": -0.713, "abstained": false, "reason": "ok"}` from the
  same raw score. **One bad frame in thirty-two destroys the whole sample's evidence.** And the cost
  is not what the design spec claimed: it said "systematically pessimistic calibration ... the safe
  direction", but the realistic outcome is NO calibration at all, because no `reject`-band curve
  will ever be fitted — detectors abstain in that band by design. The spec's §4.2 cost statement has
  been corrected; the code's band semantics have not been touched, and must be re-argued when
  calibration lands. **This is invisible today only because nothing is calibrated at all. It is live
  the first day a fitted calibrator exists.**
- **`Detector.modalities` is declared by every detector and consumed by nothing.** `decide` scores
  every registered detector regardless of `sample.modality`.
- **`normalize` never supplies yaw or pitch**, so `Quality.yaw_deg` and `Quality.pitch_deg` are
  `0.0` in every record ever produced and `quality.MAX_YAW_HIGH` is permanently inert — a profile
  face bands exactly like a frontal one.
- **Per-detector latency goes only to a DEBUG log.** `decide` times each `score` call and logs it;
  nothing returns it, so acceptance criterion 5's p95 latency has no path out of `decide`.
- **`src/dfd/ingest/base.py`'s `IngestAdapter` Protocol is dead code.** Nothing implements or
  references it, and its `(path, context)` signature matches neither real adapter (`load_image` takes
  `limits`; `load_video` takes `max_frames`, `seed` and `limits`). It reports 0% coverage.

**The composition root now exists (2026-09-21).** `src/dfd/pipeline.py` has `normalize()` (faces +
quality onto observations, with ROI clamping) and `decide(path, ...) -> AuditRecord`, which wires
ingest → faces → quality → detectors → calibration → fusion → audit into one runnable path. `dfd
score <path>` (`src/dfd/cli.py`, `src/dfd/__main__.py`, `[project.scripts] dfd = "dfd.cli:main"`) runs
it from a shell. `fuse`, `Calibrator.to_evidence`, `build_audit_record`, `load_image`/`load_video` and
`detect_faces` now all have a non-test caller — `decide` is the first. Task 21's decode-bomb defences
are exercised through `decide`, so they now have a caller outside their own unit tests too.

What is still **not** true: the benchmark runner (`bench/runner.py`) still builds `Observation`s
directly from dicts and still bypasses ingest entirely — it does not call `decide` or `normalize` —
so criteria 4 and 11 remain unmet exactly as before; wiring a composition root did not wire the
benchmark to it. And because no detector weights are vendored on any machine that matters (gitignored
locally, absent in CI), every real call to `decide` abstains on both detectors and returns
`insufficient_evidence` — the path runs end-to-end, but it cannot yet produce any other verdict.

---

## 4. The critical path, which is still not engineering

**Rewritten 2026-09-21, against the primary sources rather than from memory.** The research-only
ruling (§0) made this path live, so the three EULAs were actually looked up. Two of the three cannot
be applied for honestly from where this project stands, and that is a harder blocker than the
"a personal address may be refused" wording this section and §1 previously carried.

### What each of the three actually requires

**FaceForensics++** — request via a Google form linked from `github.com/ondyari/FaceForensics`;
on approval they send a download script. The form's required fields are: Email, Name,
**Lab/Department/Affiliation**, **Principal Investigator/Advisor's Name**, **Principal
Investigator/Advisor's Email**, Research purpose/project description, and a terms checkbox. An
unaffiliated individual has no lab, no PI and no advisor — there is nothing truthful to type in
three required fields.

The terms of use (`kaldir.vc.in.tum.de/faceforensics_tos.pdf`) are short, and two clauses are
load-bearing here, quoted verbatim:

> 1. Researcher shall use the Database only for non-commercial research and educational purposes.

> 6. If Researcher is employed by a for-profit, commercial entity, Researcher's employer shall also
> be bound by these terms and conditions, and Researcher hereby represents that he or she is fully
> authorized to enter into this agreement on behalf of such employer.

**Clause 6 is the one to read twice, and it is not solved by using a personal address.** It binds on
*employment*, not on which mailbox the form was submitted from. If the signer is employed by a
for-profit entity, that employer is bound and the signer is representing they may bind it.
Given §1's correction that this product is not for ScoreMe, submitting this form from a personal
Gmail while employed by ScoreMe would make a representation about ScoreMe that nobody has
authorised — the opposite of the independence the personal-address route was meant to establish.
Sort that out before submitting, not after.

**Celeb-DF (v1/v2)** — a Google form linked from `github.com/yuezunli/celeb-deepfakeforensics`
(Tencent mirror also offered; questions to `deepfakeforensics@gmail.com`). Required fields: Name,
Affiliation/Organization, City, Country, Purpose Description, Version, and a typed signature. Its
terms state the dataset "is for non-commercial research purposes only", that "you and your
affiliated institution must agree not to reproduce, duplicate, copy, sell, trade, resell or exploit
any portion of the videos or derived data" — and, decisively, that the download link goes to
**"your ACADEMIC email address"**. `kohrohit@gmail.com` is not one. This is a stated requirement,
not a soft preference, so this request is expected to fail as things stand.

**DFDC** — the only one of the three with no PI field and no academic-address requirement. Meta's
page (`ai.meta.com/datasets/dfdc/`) routes to `dfdc.ai` and the prerequisites are an AWS account, an
IAM user, and the AWS account ID; the dataset is also reachable through the Kaggle competition by
accepting its rules. **Caveat, stated honestly: the non-commercial restriction on DFDC is the one
claim in this section taken from secondary sources rather than read in the agreement itself** —
`dfdc.ai` returned no readable terms to this session. Read the actual licence at download time
before relying on it, and record what it says here.

### The consequence

**Only DFDC is realistically reachable today.** FF++ needs an affiliation and a PI; Celeb-DF needs
an academic address; both are non-commercial-only in terms that bind an employer if there is one.
Three ways forward were put to the owner; **they ruled option 2, find an academic signatory**
(2026-09-21). `docs/EULA-ACCESS.md` carries the packet for it. The other two are kept here because
option 2 can fail, and because option 3 is required regardless:

1. **Pursue DFDC alone** and accept a single-source benchmark — which undercuts the
   leave-one-generator-out protocol the build already implements, since LOGO needs several
   generators to hold out.
2. **Find an academic affiliation** — a collaborating lab or supervisor willing to be the named PI
   and signatory. This unlocks all three and is the only route to the benchmark the spec assumes.
3. **Drop the research datasets** and build on self-generated swaps over licence-clean real faces —
   the same plan the *commercial* answer would have forced. Slower to credibility, but it owes
   nobody an EULA and survives a later commercial turn without re-licensing.

Note that option 3 is required *anyway* for training data (§0: 7 positives cannot train anything),
so the question is only whether the research datasets are worth chasing as an **evaluation**
benchmark on top of it.

### Then, still

**Usable weights.** Measurement (c) shows public detectors do not transfer. Realistic routes: train
SBI ourselves (needs only *real* faces, so licence-clean), or rent GPU to fine-tune on the
442-session corpus — noting §0's warning that those 442 are the evaluation set and training on them
spends the only labelled fraud this project has.

---

## 5. Recommended next steps, in order

1. **Open the PR.** The branch is review-clean and pushed.
2. **Start the EULA requests** — in parallel with everything else, or the wait becomes sequential.
3. **An embedder for criterion 2.** Note `check_identity_disjoint` now *refuses* ids with no
   embedding rather than skipping them — a partial-embedding pipeline must omit unembeddable ids
   explicitly, which is the point.
4. ~~**The RD adapter for criterion 4** — the 24 cached results are free and already labelled.~~
   **Struck 2026-09-22.** They are free and labelled; they are not joinable. Only 2 of the 24 are
   capture frames — see §0, which replaces this step with a ruling the owner has to make.

("A composition root" was step 3 here; it is done — see §3 above — and struck from this list
2026-09-21. The benchmark runner still does not call it, which is why criteria 4 and 11 are still
open and still numbered above as the next two steps.)

**Reordered 2026-09-22, after the DF40 measurement.** The list above was written when nothing had
been measured. It now has a step 0 in front of it, because AUC 0.289 changes which problem is
binding:

0. **Fix the training distribution before fitting anything else.** The failure measured in §0 is not
   a shortage of fakes — it is that the model's *reals* (clean Flickr portraits) and the field's
   reals (compressed video frames) are different populations, so "clean" became the model's proxy
   for "real". Two experiments settle it, both code-only and both cheap:
   - Fit the same seam model on self-blends of DF40's own **real** half, evaluate on its fake half,
     splitting so no source frame appears on both sides. If the AUC recovers, the defect is domain
     shift and the fix is real-face supply that matches the field, not more fakes. Note the
     resulting weights would be CC BY-NC (DF40 reals are encumbered) — a **diagnostic artifact that
     must never be written to `assets/models/blend_seam.npz`**, which the manifest registers as
     commercially usable.
   - Degrade FairFace to match: re-encode each crop through the compression and resolution ladder
     the capture path actually produces, then refit. Licence-clean, and it tests the same
     hypothesis from the other side.

   Only after one of these moves the number is there any point spending a EULA on more fakes.

---

## 6. The recurring defect, and the method that caught it

**Roughly thirty tests across this plan shipped green while unable to fail for the right reason.**
Variants, all found here: bound assertions a stub satisfies; open intervals; `pytest.raises` with no
discriminating `match=`; a boolean asserted one direction; a loop or `all(...)` over a collection
empty under the mutation; a fixture below a quality floor so every metric was `nan`; a
correctly-wired value no test could detect regressing.

Two patterns worth carrying forward:

- **A fix correct about the case it was shown, blind one level down.** Five instances. An `except`
  narrowed for parsing but not construction; asset matching fixed for stems and broken for duplicate
  claims; a leaf-value check applied to one field and not its sibling; `"finite and positive"`
  implemented as `"positive"`, where the dropped word was the one that mattered against an adversary.
- **When you narrow a requirement, state which part you dropped and why it is safe.** If you cannot
  write that sentence, the narrowing is not safe.

**The method:** break the implementation, watch the test fail, restore, watch it pass. A test nobody
watched fail is a hope, not a guard. Require that proof in every dispatch, and verify each staged
plan section by *running* it before handing it to an implementer — four of my own reference
implementations failed their own tests.

---

## 7. Workspace

**Deleted 2026-09-21**, once its stated precondition held. `.superpowers/sdd/` held two workspaces
— `2026-09-20-p0-evidence-core-and-benchmark/` and `2026-09-21-composition-root/`, 123 gitignored
files, 2.9M of per-task briefs and reports. They were retained past the process default because the
committed ledger did not yet carry the final review's rulings; PR #2 fixed that, and they were
removed immediately after it merged. The ledger has every ruling; the reports had the working, and
that working is now gone. Nothing tracked was touched.
