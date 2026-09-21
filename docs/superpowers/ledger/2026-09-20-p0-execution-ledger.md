# SDD ledger — plan: docs/superpowers/plans/2026-09-20-p0-evidence-core-and-benchmark.md

Branch: p0-evidence-core (off main @ 82b0521)
Spec: docs/superpowers/specs/2026-09-20-deepfake-detection-design.md

## Pre-flight conflict scan

### Interface pairs (producer -> consumer)
| Producer | Consumer | Interface | Finding |
|---|---|---|---|
| T1 types | T3,T5,T6,T7,T8,T9,T10,T17 | Quality/Observation/RawScore/Evidence/QUALITY_BANDS | clean |
| T3 meets_floor(band,floor) | T6,T7,T8 | signature | clean |
| T4 detect_faces/align | T5 ingest | declared consumed | **CONFLICT 2** |
| T6 RawScore | T9 Calibrator.to_evidence | RawScore -> Evidence | clean |
| T6 Registry (base.py) | T17 runner tests | re-export via registry.py | clean |
| T9 Evidence | T10 fuse | Evidence[] -> FusedResult | clean |
| T11 tpr_at_fpr | T15 adversarial_tpr, T17 runner | signature | clean |
| T12 GuardViolation | T13 protocol | declared consumed | **CONFLICT 3** |
| T12 check_*/IdentityReport | T17 runner | all imported | clean |
| T14 robustness | (nobody) | never wired | **CONFLICT 4** |
| T16 datasets | T17 runner | declared, unused | clean (loaders standalone by design) |

### Task self-consistency (own tests vs own code)
| Task | Finding |
|---|---|
| T1 | clean |
| T2 | clean |
| T3 | clean — flat image blur<20 -> "reject" != "high", test holds |
| T4 | clean |
| T5 | clean (code); interfaces line wrong, see CONFLICT 2 |
| T6 | **CONFLICT 5** — select_subset seed-collision flake risk |
| T7 | clean |
| T8 | clean |
| T9 | clean |
| T10 | **CONFLICT 1** — specified test cannot pass |
| T11 | clean — tie-averaged ranks give exactly 0.5 |
| T12 | clean; uniform-preprocessing check is global not per-label (stricter than described, tests hold) |
| T13 | clean |
| T14 | clean |
| T15 | clean — TinyModel weight cat (2,192) matches fc shape |
| T16 | clean |
| T17 | **CONFLICT 6** — ECE computed on raw scores, not probabilities |

## Pre-flight rulings

Ruling: CONFLICT 1 (T10) — test_disagreement_is_recorded_as_a_feature asserts
OUT_OF_DISTRIBUTION for evidence +3.0/-3.0, but disagreement=3.0 < the
DISAGREEMENT_OOD=4.0 constant the same task specifies, so the verdict is
INSUFFICIENT_EVIDENCE and the test FAILS as written. Spec §7 requires
disagreement to escalate, and two detectors at full opposite confidence is
exactly the 6/4-split case §1.2 says RD discards. Lowering the constant to 3.0
is the smaller change and matches the spec's intent. Decision: set
DISAGREEMENT_OOD = 3.0, keep the test as written. — Cost if wrong: OOD fires
slightly more eagerly, raising manual-review load; one constant to retune.

Ruling: CONFLICT 2 (T5) — the Interfaces block claims T5 consumes detect_faces,
align and measure_quality, but the specified code sets roi=None, quality=None
and calls none of them. The code is correct: ingest should decode, not analyse,
so face detection stays swappable per spec §5.4. Decision: code stands, the
Interfaces line is wrong; implementer follows the code. — Cost if wrong:
Observations reach detectors with quality=None and every detector abstains with
"quality_not_measured"; T17's runner already attaches quality itself, so P0 is
unaffected and wiring it into ingest is a P1 change.

Ruling: CONFLICT 3 (T13) — Interfaces claims T13 consumes GuardViolation; the
code raises KeyError for a missing generator key instead. KeyError is the right
error for a malformed record (a programming error) rather than a hygiene
violation (an evaluation-design error), and the task's own test asserts
KeyError. Decision: code stands, Interfaces line is wrong. — Cost if wrong:
callers catching only GuardViolation miss malformed records; loud failure, not
silent.

Ruling: CONFLICT 4 (T14) — robustness_sweep is built and tested but never called
by the runner, so acceptance criterion 9 (screenshot/print re-capture measured)
is NOT met by the 17 tasks as written. Decision: do not expand P0 scope mid-
flight; add Task 18 after Task 17 to wire the sweep into the runner, and record
criterion 9 as unmet until then. — Cost if wrong: P0 closes believing it
measured physical re-capture when it did not — which is why this is a task, not
a note.

Ruling: CONFLICT 5 (T6) — select_subset test asserts different seeds give
different subsets; choosing 3 of 5 can collide by chance and flake. Decision:
implementer must pick seeds verified to differ, and assert on a specific
expected subset rather than mere inequality. — Cost if wrong: an intermittently
red test that erodes trust in the suite.

Ruling: CONFLICT 6 (T17) — ECE is computed over raw detector scores, but ECE is
only meaningful over calibrated probabilities; a raw score is not one. Decision:
keep the column (it is still a monotone miscalibration signal) but the runner
must label it `ece_raw` in DetectorResult and the report must footnote that true
ECE arrives when fused LLRs are scored in P1. — Cost if wrong: a misread number
in the first report; the footnote is the mitigation.

## Task log
Plan amended: Task 18 added (CONFLICT 4 ruling materialised) — commit 117c3da.
Task 1: dispatched (haiku, transcription-tier). BASE recorded 82b0521.
NOTE: controller committed 117c3da (docs only) while Task 1 implementer was
live. Review package for Task 1 must use base 117c3da, NOT 82b0521, or the
doc commit pollutes the review diff. Do not commit to the branch during an
implementer run again.

Ruling: my `git add -A` at the Task 18 docs commit swept the live implementer's
pyproject.toml, __init__.py and test_types.py into a docs commit — the exact
hazard logged one step earlier. Branch was unpushed, so I rewrote both commits
to separate docs from implementation. Decision: history rewritten, not left
misattributed. — Cost if wrong: none; branch was local-only and the rewrite is
verified by `git show --stat` plus a green suite. New standing rule: controller
never runs `git add -A` while an implementer is live; stage explicit paths only.

Ruling: `python` is not on PATH on this machine, only `python3`. Every plan
task specifies `python -m pytest`. Decision: implementers run `python3 -m
pytest`; the plan text stays as written rather than being rewritten 18 times.
— Cost if wrong: an implementer reports a spurious command-not-found; cheap.

Task 1: complete (commits 1ffa265..4595299, 6/6 tests green) — review pending.

Task 1 review: spec ✅; quality approved with 2 Important, 3 Minor.

Ruling: Important-1 (QUALITY_BANDS has zero test coverage) — the reviewer is
right and the gap is plan-mandated: the brief's own Step 1 test code omits it.
The tuple's worst-to-best ORDER is semantic — Task 3's meets_floor() calls
QUALITY_BANDS.index() on it, and 17 tasks depend on that comparison. A silent
reorder or typo would currently pass the whole suite while inverting every
quality gate in the system. Decision: enter fix loop, add a test pinning the
exact tuple and its ordering. — Cost if wrong: one extra trivial test.

Ruling: Important-2 (report cites commit 7c3ceec and 117c3da; diff shows only
4595299) — NOT an implementer defect. This is the visible scar of my own
history rewrite: the implementer's report was written before I re-split the
commits, so its SHAs are stale by my action, not its error. Decision: park, no
fix dispatched; the report file stays as written since rewriting a subagent's
report to match my cleanup would be falsifying the audit trail. — Cost if
wrong: a future reader sees two commit SHAs that no longer resolve; this ledger
entry is the explanation.

Task 1 minors (deferred, for final review triage):
 - test_quality_band_is_explicit_not_derived uses self-consistent values, so it
   does not actually prove band is non-derived (plan-inherited).
 - 4 of 6 tests are field round-trip checks guarding name/arity drift, not
   semantics (largely unavoidable for pure value dataclasses).
 - test_sample_is_immutable catches bare Exception, not FrozenInstanceError.
Task 1: fix round 1/5 dispatched — FIX_BASE 4595299.
Task 1: fix round 1/5 (1 addressed pending re-review, 0 open; commits 4595299..ec4cec1, 7/7 green)
Task 1: re-review clean (1 addressed, 0 open, no new breakage).
Task 1: complete (commits 1ffa265..ec4cec1, review clean, 7/7 green, 1 parked, 3 minors deferred)
Task 2: dispatched (haiku). BASE ec4cec1.

Task 2 review: spec ✅; quality approved; 0 Critical/Important, 2 Minor. No fix loop.
Fail-closed verified by control-flow reading: manifest.get() -> None short-circuits
to the raise before touching rec.commercial_use; empty manifest rejects everything;
malformed record raises loud KeyError; error message names the failing ids.

Ruling: the reviewer noted inside its edge-case analysis that assert_release_clean
judges only the ids it is HANDED, so an empty asset_ids list returns cleanly. That
is not a minor — it is the same class of defect as pre-flight CONFLICT 4 (a
component built correct in isolation and wired to nothing). With no enumerator, a
forgotten weight file ships unchecked, which is exactly the false confidence the
gate exists to prevent, and spec §12.1 criterion 6 is unverifiable. Decision:
added Task 19 (discover_assets + assert_all_assets_registered) rather than
recording it as polish. — Cost if wrong: one small module and 5 tests that turn
out to be redundant if release tooling later enumerates assets itself.

Task 2 minors (deferred, for final review triage):
 - a None element in asset_ids makes sorted(bad) raise TypeError rather than
   NonCommercialAsset; still blocks the release, but with a confusing error.
 - no type coercion on YAML values feeding AssetRecord; an unquoted date_checked
   would parse as a date object, not str (authoring-hygiene risk for future entries).
Task 2: complete (commits ec4cec1..1b481f8, review clean, 11/11 green)
Plan amended: Task 19 added — commit aae1c80. Plan now 19 tasks.
Task 3: dispatched (haiku). BASE aae1c80.

Task 3 review: spec ✅; quality approved; 2 Important, 2 Minor.

Ruling: Important-1 (report claims "reject never satisfies any floor"; actually
meets_floor("reject","reject") is True). Reviewer independently confirmed after
I verified it by running the shipped code. Root cause is a TEST gap, not a code
gap: test_meets_floor_uses_band_ordering never exercises floor="reject", so the
overclaim went unchallenged. Decision: fix round — pin the true behaviour in a
test and have the implementer append a correction to its own report. The report
is a verification artifact; leaving a false claim in it is materially different
from the stale SHAs I parked in Task 1, which were my own doing and harmless.
— Cost if wrong: one extra test and a report paragraph.

Ruling: Important-2 (floor is typed bare `str`, so nothing prevents a future
detector declaring min_quality_band="reject" and silently consuming samples the
gate exists to exclude). Reviewer judged it latent, not live, and explicitly
advised NOT fixing in Task 3 as scope creep — I agree; meets_floor's reflexivity
is mathematically correct for an "at least as good as" relation and special-
casing it would be worse. Decision: do not touch Task 3; carry the constraint
into Task 6, which defines the detector config, typing min_quality_band as
Literal["low","medium","high"]. — Cost if wrong: if Task 6 forgets, a future
detector can opt into consuming unusable frames; recorded here so the final
review catches it.

Ruling: Minor (test_flat_image_is_not_banded_high asserts only inequality with
"high", so it would still pass if _band always returned "reject"). That is a
test that cannot fail for the right reason. Promoting it into the same fix round
rather than deferring — a flat image has ~0 Laplacian variance and should pin as
"reject" exactly. — Cost if wrong: negligible.

Task 3 minors (deferred): grayscale 2-D crop path is correct but untested.
Task 3: fix round 1/5 dispatched — FIX_BASE dc547a0.
Task 3: fix round 1/5 (3 addressed, 0 open; commits cc5b467..c7cb866, 17/17 green)
Task 3: complete (commits aae1c80..c7cb866, review clean, 1 minor deferred)

Plan amended twice (user steer: "i want production level code"):
 - 3575fe8 Global Constraints raised to production engineering standards.
 - cc5b467 Tasks 20-22 added.
Ruling: Task 20 restores a DROPPED SPEC REQUIREMENT. Spec §7.2 mandates an
immutable per-decision audit record; the 19-task plan had no task for it. That
was my omission when writing the plan, not a scope change requested now. Without
it the system can decide but cannot account for a decision — unshippable in BFSI.
— Cost if wrong: none; it is a spec requirement either way.
Ruling: retrofitting production standards onto Tasks 1-3 is deferred to Task 22,
whose ruff/mypy gate sweeps the whole tree at once, rather than reopening each
completed task. — Cost if wrong: Task 22 becomes a larger fix wave than a normal
task; visible and bounded.
Task 4: dispatched (haiku). BASE c7cb866.

Task 4 review: spec ✅; quality NOT APPROVED. 1 Critical, 4 Important, 6 Minor.

Ruling: Critical-1 + Important-2 (detect_faces has NO return annotation; report
falsely claims "type hints on all public callables"). Both real. The polymorphic
return (bare list vs tuple, keyed on with_reason) is internally consistent today
— reviewer traced both paths — but is undescribed by types, so nothing protects
a future caller. Decision: fix via @overload pair, which documents the shape
statically. Also a straight violation of the raised global constraint. — Cost if
wrong: none, this is strictly additive typing.

Ruling: Important-3 (align tests assert only .shape/.dtype; reviewer verified a
stub returning np.zeros passes both). Same defect class as Task 3's weak
assertion. A test that passes against a function which never reads its input is
not a test. Decision: fix — assert on actual pixel content. — Cost if wrong: none.

Ruling: Important-4 (report claims the manifest is "validated by existing
manifest tests"; false — test_manifest.py uses an inline fixture and NO test in
the repo loads the real assets/manifest.yaml). The false claim must be corrected,
but it also exposes a real coverage hole: the shipped manifest is unverified.
Decision: fix the claim AND add a test that loads the real file. — Cost if wrong:
one cheap test that Task 19 would partly duplicate.

Ruling: Important-5 (frozen=True does not deep-freeze; FaceBox.landmarks is a
mutable ndarray, so box.landmarks[0,0]=999 silently corrupts evidence). Real and
systemic — types.py Observation.payload and Context.meta share it. Decision: fix
at the origin here with setflags(write=False), one line, and log the systemic
case for a later sweep rather than reopening Tasks 1-3 now. — Cost if wrong: a
caller wanting to mutate landmarks must .copy() first; correct for evidence.

Ruling: Minor-6 PROMOTED to Important. faces.py documents landmark order as
"right eye, left eye, ..." while quality.py documents the same slots as
"[[lx,ly],[rx,ry]]" — left first. Harmless today only because inter-ocular
distance is symmetric. Two modules disagreeing about which physical eye is at
index 0 is a landmine for anything needing eye identity (roll correction, gaze),
and neither claim is verifiable from the repo. Decision: fix — make both
docstrings state the order is UNVERIFIED against real YuNet output and flag it as
a P1 verification item. — Cost if wrong: none; honesty about what we have not
checked is free.

Ruling: Minors 7, 8, 9 are constraint violations, not polish. No module logger
(I explicitly required one in the dispatch), score_threshold hardcoded with no
named constant, and a docstring promising "never an exception" that is false for
a corrupt-but-present model file. Decision: all three fixed. On the corrupt-file
path specifically: do NOT catch cv2.error and report it as weights_absent —
silently treating a corrupt model as a missing one hides a real deployment
failure. Correct the docstring and let it raise. — Cost if wrong: a corrupt model
file crashes loudly at startup instead of degrading silently, which is intended.

Task 4: fix round 1/5 dispatched — FIX_BASE d31e494.
Task 4: fix round 1/5 (commits d31e494..425356f). Controller verification found
FIX-4 (read-only landmarks) NOT actually enforced: my round-1 instruction scoped
setflags to inside detect_faces, a path that never runs in CI without weights,
and the covering test was tautological (it called setflags itself before
asserting). Root cause was my instruction, not the implementer.
Ruling: the invariant belongs in the TYPE (FaceBox.__post_init__), not on one
call path, so it holds however a box is constructed. A FaceBox is evidence; if
landmarks can be mutated post-detection an audit record can disagree with what
the detector saw and nothing reveals it. — Cost if wrong: callers needing a
mutable landmark array must .copy() first.
Task 4: fix round 2/5 (commits 425356f..3696f86). Controller re-verified by
execution: direct-construction mutation now raises; only one setflags remains;
logging is lazy. Implementer confirmed the rewritten test FAILS with
__post_init__ disabled — it fails for the right reason.
Also verified by execution: an align() stub returning np.zeros now FAILS both
align tests, where the pre-fix versions passed it.
Task 4: re-review clean — all 10 findings ADDRESSED, no new breakage. Reviewer
verified the @overload pair with real mypy reveal_type calls (not cosmetic) and
confirmed no cv2.error catch was smuggled in on the corrupt-model path.
Task 4: complete (commits c7cb866..3696f86, 23/23 green, 2 rounds, 2 minors deferred)

Task 4 minors (deferred, for final review triage):
 - test_align_clamps_box_to_frame_bounds uses an all-white 100x100 frame, so
   numpy's own slice truncation yields the same region as the clamp; the test
   cannot distinguish "clamped correctly" from "clamping removed". Pre-existing,
   outside the fix round's scope. Low risk (numpy truncates safely anyway) but
   it is another test that cannot fail for the right reason.
 - mypy --strict flags bare np.ndarray annotations missing generic parameters on
   5 lines in faces.py, and the same pattern exists in types.py/quality.py.
   Pre-existing and repo-wide. Task 22 WILL surface these when it turns mypy on
   — expect a fix wave there, it is not a surprise.
Task 5: dispatched (haiku). BASE 3696f86.

Task 5 review: spec ✅; quality NOT APPROVED. 5 Important, 6 Minor.

Ruling: MY sample_indices dedup concern was WRONG and the reviewer disproved it.
I flagged sorted(set(...)) as silently lossy (two bins jittering onto one index
returning fewer than k frames). The reviewer proved otherwise both algebraically
and by brute force over total<=3000, k<=500: when total>k the linspace edges are
strictly increasing, so bins never overlap and set() never removes anything.
Decision: no fix; record that the invariant is real but untested, and require a
test at the tight boundary (total=k+1) so a future change to the binning formula
cannot silently reintroduce the loss. — Cost if wrong: none; I was the one who
was wrong and verification settled it.

Ruling: Important-5 (silent empty result) is the most serious finding and I am
treating it as the priority fix. cv2's CAP_PROP_FRAME_COUNT returns 0 or negative
for some codecs and VFR containers even when frames decode fine. total<=0 then
takes the total<=k branch, range(-1) is empty with no error, and load_video
returns a Sample with ZERO observations for a video that may contain real
evidence — no exception, and (per finding 1) no log line either. In a fraud
pipeline a video silently evaluated as "no frames" is an approved fraud.
Decision: fall back to sequential reading when metadata is unusable rather than
trusting it, warn, and raise only if nothing actually decodes. — Cost if wrong:
unstratified sampling on metadata-broken files, which is strictly better than
discarding the evidence.

Ruling: Important 1-4 (no logger, no docstrings on two raising public functions,
undocumented 25.0 fps fallback, cap.release() not exception-safe) are all direct
violations of the raised global constraints, not style. All fixed. The report
also marked logging "✅ not required by brief" — global constraints bind
independently of the brief, so that claim is doubly wrong.

Ruling: Minor-10 PROMOTED. No test asserts roi is None and quality is None — the
architectural invariant that IS this task (ingest decodes, does not analyse). A
regression wiring in detect_faces would pass all 7 tests. That is precisely the
coupling my pre-flight CONFLICT-2 ruling exists to prevent, so it must be pinned
by a test, not by my memory. — Cost if wrong: none.

Task 5 minors (deferred): sample_id = path.stem has no cross-directory
uniqueness guarantee (brief-inherited); bare ValueError should migrate to the
DfdError hierarchy when Task 20 creates it.
Task 5: fix round 1/5 dispatched — FIX_BASE 8346580.
Task 5 review addendum: reviewer extended its brute force to 1,139,700 combos
(total<=2000, k<=200, multiple seeds) plus an edge-monotonicity scan to
total=5000/k=300. Zero collisions, zero non-monotonic edges. My dedup concern is
conclusively disproved, not merely unsupported. Verdicts unchanged.
Task 5: fix round 1/5 (9 addressed, 0 open; commits 8346580..c86167f, 33/33 green)
Task 5: complete (commits 3696f86..c86167f, review clean, 1 round, 3 minors deferred)
Re-review confirmed the stratified path is byte-for-byte unchanged and the
membership test collapses to the pre-fix `i in wanted`, so normal videos are
unaffected. It also noted the fix closed a SECOND latent hole I had not asked
for: total>0 but every cap.read() fails now raises instead of silently returning
an empty Sample.

Task 5 minors (deferred, for final review triage):
 - DEFAULT_MAX_FRAMES=32 comment restates the name; no provenance for why 32.
 - test_ingest_decodes_but_does_not_analyse would vacuously pass on an empty
   observation list; safe today only because Finding 1's guard makes empty
   impossible. Not intrinsically self-sufficient.
 - fallback path reads to EOF rather than breaking at max_frames (performance
   only; matches the original pattern).
Task 6: dispatched (haiku). BASE c86167f.

Task 6 review: spec ✅; quality NOT APPROVED. 2 Critical, 3 Important, 3 Minor.
All three functional defects reproduced by execution, not inferred.

Ruling: Critical-1 (registry identity drift). SyntheticDetector is a non-frozen
dataclass and the Detector protocol declares `name` as a mutable attribute, so
`d.name = "b"` after registration leaves the dict keyed "a" pointing at an object
that calls itself "b" — reachable through code that typechecks cleanly. This
also violates the raised global constraint (frozen unless mutation required; none
is). Decision: freeze the concrete detector, make protocol identity attributes
read-only properties, use frozenset for modalities (a mutable set inside a frozen
dataclass is the same deep-freeze hole found in FaceBox), and have Registry
snapshot the name at registration rather than trusting a live attribute. — Cost
if wrong: detectors become immutable value objects, which is what they should be.

Ruling: Critical-2 (mixed-batch abstention reason is order-dependent). The reason
is chosen from obs[0] alone, so the SAME situation reports
quality_not_measured or below_quality_floor purely by list order. An analyst
auditing an abstention gets an arbitrary root cause. Decision: make the rule
principled — if ANY observation had quality measured and failed the floor, the
reason is below_quality_floor; quality_not_measured only when NO observation
carried quality at all. Add the missing mixed-case test. — Cost if wrong: a
slightly more conservative diagnosis, which is the right bias for an audit trail.

Ruling: Important-3 (XOR cancellation) — I raised this and the reviewer confirmed
it empirically: one observation scores 0.6123, the same observation twice scores
exactly 0.0. XOR is self-inverse, so a duplicated frame silently vanishes. The
detector declares Modality.VIDEO, and duplicate or byte-identical frames are
exactly what a static scene or a frozen injected stream looks like — the case
that matters most. Every existing test calls score() with a single-element list,
so 42 green tests never touched it. Decision: replace XOR with addition modulo
2**64, which keeps order-independence but makes duplicates accumulate instead of
cancel. — Cost if wrong: the synthetic scores change value; nothing depends on
their specific values, only on determinism.

Ruling: Important 4-5 and Minors 6-7 (bare magic numbers digest()[:8] and
10_000 with no provenance; report falsely claims "frozen=True where applicable"
when the only dataclass is not frozen; select_subset has no Raises section;
empty-observation list reuses NO_QUALITY). All fixed; the frozen claim gets an
appended correction rather than an edit.

Task 6: fix round 1/5 dispatched — FIX_BASE 7ae7a67.
Task 6: fix round 1/5 (7 addressed, 0 open; commits 7ae7a67..32f2c1a, 48/48 green)
Task 6: complete (commits c86167f..32f2c1a, review clean, 1 round, 2 minors deferred)
Re-review verified protocol satisfiability with a standalone mypy run (a frozen
dataclass field DOES satisfy a read-only @property protocol), so Tasks 7-8 are
not blocked. It also judged the addition-mod-2^64 collision surface: the old XOR
bug was a systematic 100% collapse on any duplicate; the new scheme's collision
is ordinary birthday-bound (~2^-64), immaterial for a deterministic test double.

Task 6 minors (deferred, for final review triage):
 - two near-duplicate tests of the frozen-mutation fact could be consolidated.
 - the registry name snapshot guards the KEY only. Nothing stops a future
   non-frozen Detector implementer from reporting a different name from .score()
   at call time than the key it was registered under. Out of scope for Task 6,
   but a real gap once Tasks 7-8 add hand-written detectors.
Task 7: dispatched (haiku). BASE 32f2c1a. FIRST REAL DETECTOR.

Task 7 review: spec ✅; quality NOT APPROVED. 3 Critical, 6 Important, 5 Minor.

Ruling: Critical-1 is the most serious finding of the build so far. 59 green
tests, and the first real detector's actual scoring path has 0% coverage — every
test either calls npr_feature standalone or short-circuits at weights_absent.
Two tests are NAMED for quality filtering and exit before reaching that loop.
My own "hermetic: no test requires a weight file" constraint caused this: the
implementer read it as "never exercise the weights path". Decision: clarify that
a test may CREATE a tiny dummy torch model in tmp_path — that is still hermetic
(no network, no pre-existing artifact) — and require the real path be exercised.
— Cost if wrong: none; this closes a hole that 59 green tests were hiding.

Ruling: Critical-2 (torch.load without weights_only) is a genuine supply-chain
hole under spec §3A, which assumes a supply-chain-capable adversary. Default
torch.load unpickles and can execute arbitrary code from a tampered .pt. The
weights are registered in the manifest by URL with no checksum. Decision: secure
by default — weights_only=True, plus an explicit allow_unsafe_load flag that must
be set deliberately and logs a WARNING. A full-module pickle is then REJECTED
with an actionable error rather than silently loaded. — Cost if wrong: the
published research weights may ship as a full pickle and need an operator
conversion step. That is the correct trade for BFSI; a silent unsafe default is
not.

Ruling: Important-6 (probs.mean() dilutes a strong single-frame detection — one
frame at 0.95 and four at 0.05 averages to 0.23, exactly what a partial-duration
swap produces). The reviewer is right, but cross-observation aggregation is
Task 10's (fusion) responsibility, not a detector's. Decision: keep mean as the
scalar, but ALSO record max and the per-observation scores in artifacts so no
information is destroyed before fusion can use it, and document the limitation
explicitly. — Cost if wrong: fusion later ignores the artifacts; nothing is lost.

Ruling: my odd-dimension concern was WRONG. The reviewer traced the arithmetic
and verified numerically on 63x65 and on an odd-cropped upsample: no pixel
misalignment, residual still exactly 0.0. Decision: no code change, add the
missing coverage only. — Cost if wrong: none; I was wrong and execution settled it.

Noted uncredited: the implementer silently fixed the Task-6 order-dependent
abstention bug that was still present in the brief's draft code. Good catch.

Task 7: fix round 1/5 dispatched — FIX_BASE 0276f1f.
Task 7: fix round 2/5 (9 addressed; commits 0276f1f..ab02bd5, 70/70 green).
Re-review confirmed load_state_dict uses strict=True (no silent partial load of
mismatched weights -> no confident nonsense), the cache lock spans check AND
insert, and CASE 1 (model_factory) is checked before allow_unsafe_load so a
caller supplying both still lands on the SECURE path.

But the re-review raised two NEW items that join the open list:

Ruling: NEW BREAKAGE introduced by round 2's own fix. The cache key is
(resolved_path, mtime_ns, size) and does NOT include model_factory identity or
allow_unsafe_load. Two detectors sharing a weights_path but built with different
architectures would silently share ONE cached model — the second gets the
first's object. Not exploitable in the current suite only because each fixture
uses a distinct tmp_path. This is a real design gap opened by the very field I
added to close the security hole. Decision: fix — include factory identity and
the load mode in the key. — Cost if wrong: marginally lower cache hit rate,
which is the correct direction for a correctness-vs-speed trade.

Ruling: EIGHTH test that cannot fail for the right reason.
test_detector_cache_invalidation_on_file_replacement asserts only
`not r2.abstained`, despite deliberately filling the replacement model's weights
with 2.0. A stale cached model also returns a valid non-abstained score, so the
test passes against the exact pre-fix bug it was written to catch. Decision: fix
— assert the SCORE changed. — Cost if wrong: none.

Noted: reviewer identified the suite's single warning as torch.load without
weights_only in the gated unsafe branch, exercised once by the gate test. That
is expected and correct — the warning is evidence the unsafe path is unsafe.
Task 7: fix round 3/5 dispatched — FIX_BASE ab02bd5.
Task 7: fix round 3/5 (commits ab02bd5..480189d, 70/70 green).
Controller verification: the cache-key fix WORKS — two factories on one shared
weights file now yield distinct models (0.3227 vs 0.6773). Implementation correct.
BUT the two tests round 3 required were NOT added: NPR test count unchanged at 22,
and no test is named for factory collision or factory/file mismatch. The fix is
correct and UNGUARDED — a refactor could silently revert it.
Ruling: escalate per the skill's rounds 4-5 rule — fresh implementer, model tier
up (sonnet, from haiku). The remaining work is small and purely additive (two
tests), but three rounds of the same agent have now produced correct code with
missing guards twice, which is the signal the rule exists for. — Cost if wrong:
one more dispatch on a pricier tier for a two-test change.
Task 7: fix round 4/5 dispatched — FRESH implementer, sonnet. FIX_BASE 480189d.
Task 7: fix round 4/5 (3 addressed, 0 open; commits 480189d..5c1cf55, 72/72 green).
Re-review confirmed both new tests fail for the right reason: the collision test
uses a NEGATED-logits factory so the two models differ systematically rather than
by luck (softmax([a,b]) != softmax([-a,-b])), and the mismatch test matches on
"mutually exclusive", a phrase absent from torch's raw error.
Task 7: complete (commits 32f2c1a..5c1cf55, review clean, 4 rounds, 24 NPR tests)
Escalation to a fresh implementer on a higher tier worked: the same instruction
had gone unimplemented for one round under the original agent.

Ruling: Task 8 must NOT copy Task 7's loader. npr.py now carries ~150 lines of
security-critical machinery — weights_only=True gating, model_factory, the
allow_unsafe_load escape hatch, the (path, mtime, size, factory_id, mode) cache
key, the threading lock spanning check-and-insert, strict=True, and output-shape
validation. Task 8 (EfficientNet-B4, slots A and E) needs all of it. Duplicating
it means a future security fix must land twice, and the second copy is the one
that gets forgotten — precisely how the round-2 "theatre" bug would recur in a
file nobody re-reviewed. Decision: Task 8 first EXTRACTS this into
src/dfd/detectors/loading.py, refactors npr.py onto it (its 24 tests must stay
green, unmodified, proving behaviour is preserved), then builds effnet.py on the
shared module. — Cost if wrong: one refactor commit; the alternative is two
divergent copies of the supply-chain control.
Task 8: dispatched (sonnet — refactor + new detector, not transcription). BASE 5c1cf55.

Task 8 review: spec ✅; quality APPROVED with 1 Important, 3 Minor.
Refactor verified a GENUINE verbatim extraction — cache key, lock scope, branch
order and both error strings unchanged character-for-character vs pre-refactor
npr.py. tests/test_npr.py byte-identical across the range, 24/24 green. mypy clean
on all three detector files. Slot A/E distinction proven behaviourally (two
oppositely-signed weight files, scores differ), not merely by type check.

Ruling: Important-1 — the quality-filter/abstention block is now TRIPLICATED
across base.py, npr.py and effnet.py. This is the exact anti-pattern Phase 1
existed to kill, applied to the loader but not to this logic. It matters more
here than anywhere: this specific block has ALREADY been buggy twice (the
order-dependent obs[0] rule, fixed separately in base.py and then again in
npr.py). A fourth detector copies it a fourth time, and slots B/D/F are coming.
Decision: fix — extract filter_by_quality_floor() into base.py and route all
three detectors through it. The NPR and EffNet suites must pass UNMODIFIED as
the proof of behaviour preservation, exactly as Phase 1 used test_npr.py.
— Cost if wrong: one more shared function to keep stable; the alternative is
three copies of logic with a demonstrated history of divergence.

Task 8 minors (deferred, for final review triage):
 - factory_id is derived from __module__ + __qualname__ TEXT, not the class it
   builds. A generic factory-generator (make_factory(cls) -> factory) would give
   two architectures the identical qualname "make_factory.<locals>.factory".
   Not reachable today (distinct weight files differentiate), and inherited
   unchanged from pre-refactor npr.py — but the blast radius grew now that one
   module-level cache serves multiple detector classes.
 - report claimed ruff clean but did not cover loading.py, which carries 3 F541
   (f-strings without placeholders) moved verbatim from npr.py. Task 22 will
   surface them.
 - test_detector_is_frozen catches bare Exception, not FrozenInstanceError.
Task 8: fix round 1/5 dispatched — FIX_BASE fdabb2d.
Task 8: fix round 1/5 (3 addressed, 0 open; commits fdabb2d..7df02a0, 85/85 green)
Task 8: complete (commits 5c1cf55..7df02a0, review clean, 1 round, 3 minors deferred)
Re-review traced all four input shapes through the extracted helper and confirmed
the accumulating-flag rule is genuinely order-independent, that the new test
asserts BOTH orderings (a single-ordering test would pass against the broken
obs[0] rule by coincidence), and that no existing assertion was weakened to make
the refactor pass.

Ruling for Task 9: the §1.3 measurement changes what calibration is FOR. An
Apache-2.0 detector scored 0.011 AUC on our fraud — systematically inverted, not
random. A calibrator fitted on real labelled data would have caught exactly that:
either learning the inversion and correcting it, or reporting near-zero
information. That is the argument for the LLR currency over raw scores, and it is
now empirical rather than theoretical. The implementer must be told, because it
motivates why "uncalibrated band -> llr 0.0" is a feature and not a cop-out.
Task 9: dispatched (haiku). BASE 7df02a0.

Task 9 review: spec ✅; quality NOT APPROVED. 2 Critical, 6 Important, 4 Minor.
The implementation is CORRECT — reviewer found no broken code path. The defect is
entirely in the tests, on exactly the two behaviours I named as make-or-break.

Ruling: Critical-1. Every test builds balanced n-real/n-fake bands, so
labels.mean() == 0.5 and prior_logodds == log(1) == 0.0 in every single case.
Deleting the prior subtraction entirely (llr = post_logodds) would leave all 6
tests green. The prior subtraction is what makes this a LIKELIHOOD RATIO rather
than a posterior — without it the training set's base rate is smuggled into every
production decision, and a balanced training set's base rate is nothing like a
real onboarding flow's fraud rate. Zero discriminating coverage on the keystone.
Decision: fix with an imbalanced-band test asserting the llr differs from the raw
posterior log-odds by exactly log(prior/(1-prior)). — Cost if wrong: none.

Ruling: Critical-2. The clip test asserts only math.isfinite(llr).
LogisticRegression.decision_function is linear with L2-bounded coefficients on
finite inputs — it CANNOT produce inf or NaN for a score in [0,1]. The assertion
is mathematically vacuous; deleting the clip would not fail it. The clip is the
defence against RD's saturation failure (§1.2: members emitting only 0.01/0.99),
where one overconfident detector dominates the fused posterior regardless of what
every other detector says. Decision: fix with a small constructor-supplied
max_abs_llr that forces the bound, asserting the value is pinned. — Cost if wrong: none.

That is tests #10, #11 and #12 in this codebase that could not fail for the right
reason. Notably the implementer DID perform a guard proof — on per-band
independence, which passed — and the two untested behaviours were the ones the
guard proof did not cover. Watching one test fail does not license the others.

Ruling: Important-8 (no binary label validation) is more than hygiene. A {-1,+1}
real/fake encoding — common in this literature — passes the n_classes>=2 guard,
then labels.mean() yields a "prior" outside (0,1), which silently falls into the
`else: prior_logodds = 0.0` branch. That is a WRONG result, not an absent one,
reached through a plausible caller mistake. Decision: validate and raise.

Recorded gap (not Task 9's contract): spec §7 requires calibration conditioned on
quality band AND COMPRESSION LEVEL. The brief and implementation condition on band
only. Real gap against the full architecture; must be closed before the harness
reports cross-compression numbers.
Task 9: fix round 1/5 dispatched — FIX_BASE 3e29352.
Task 9: fix round 2/5 (correction relocated; commits b7ab54c..8e428c5, 96/96 green)
Task 9: complete (commits 7df02a0..8e428c5, review clean, 2 rounds)
Controller verified by execution: breaking BOTH the prior subtraction and the clip
simultaneously fails exactly the two new tests and nothing else. {-1,+1} labels
now raise a named ValueError. Review history removed from production source;
correction appended to the report with originals intact.
Ruling: process history does not belong in production source. A reader of
calibration.py needs the mechanism, not a narrative of which review round found
what. The WHY of the prior subtraction stays in the docstring (it explains the
code); the review-round narrative goes to the report. — Cost if wrong: none.
Task 10: dispatched (haiku). BASE 8e428c5. Carries pre-flight ruling CONFLICT 1
(DISAGREEMENT_OOD must be 3.0, not the plan's 4.0, or the specified test cannot pass).

Task 10 review: spec ✅; quality NOT APPROVED. 4 Critical, 3 Important, 3 Minor.

Ruling: Critical-1 is MY ERROR, in the plan, and the reviewer is right.
I specified sqrt(ess/n_frames) as the correlated-frame discount. That is wrong.
Log-likelihood ratios ADD for independent evidence, so N observations worth ESS
independent ones carry ESS*l, making the correct factor LINEAR: ess/n. Square-root
scaling belongs to standard errors and z-scores, not additive evidence.
Verified by arithmetic at the pathological case the feature exists for (900
identical frames, per-frame llr 1.0, ESS 1):
  naive          900.0  -> posterior ~1.0, confidently wrong
  sqrt (my plan)  30.0  -> SATURATES the +/-20 cap -> posterior ~1.0, STILL WRONG
  linear           1.0  -> exactly one frame's worth, correct
So the shipped formula fails at precisely the scenario it was written to prevent.
Decision: change to linear ess/n_frames and fix the plan text. — Cost if wrong:
the discount is more aggressive; video evidence accumulates more slowly, which is
the correct bias for a fraud control.

Ruling: Critical-2 and -3 are real defects in effective_sample_size.
(2) No ceiling at n. For rho -> -1 (clamped -0.999) ESS = n*1999, which under the
    discount AMPLIFIES evidence ~44.7x instead of discounting it. A detector whose
    per-frame confidence legitimately oscillates (flicker, interlacing, alternating
    lighting) triggers this. Clamp ESS to [1, n].
(3) n=2 is deterministically broken: for any two distinct values rho is exactly
    -0.5 by construction, so ESS is always 6.0 regardless of the data - 3x the
    sample size. The n<2 guard does not catch n==2.
Decision: fix both, with tests.

Ruling: Critical-4 - FOUR more vacuous tests, bringing the codebase total to 16.
 - both abstention tests use an abstained fixture with llr=0.0, so deleting the
   abstention FILTER entirely changes nothing (adding 0.0 to a sum is a no-op).
   The filter has no regression protection at all.
 - the empty-evidence test passes even if the early-return block is deleted; the
   arithmetic falls through to the same verdict.
 - the ESS test asserts only `many < one*900` and `<= 20`, both of which hold with
   the discount completely REMOVED. The implementer's own guard proof computed the
   right number (0.1491) and never encoded it as an assertion.
Decision: all four must pin exact expected values, and the abstention fixtures must
carry a NONZERO llr.

Recorded for spec owners (Important-7, not fixed unilaterally): with
DISAGREEMENT_OOD at 3.0, nine detectors at +3 and one at -3 gives disagreement 3.0
and routes to OUT_OF_DISTRIBUTION despite total +24. A single miscalibrated or
adversarial detector can force manual review over a high-confidence fraud call.
Matches spec §7.2 intent but is an operational risk worth an explicit decision.
Task 10: fix round 1/5 dispatched — FIX_BASE 0445994.
Task 10: fix rounds 1-4 (commits 0445994..dbec0cc, 113/113 green)
Task 10: complete (commits 8e428c5..dbec0cc, review clean, 4 rounds)

All four Task 10 defects originated in MY plan or MY fix instructions:
 R1 wrong exponent      — specified sqrt(ess/n); LLRs are additive so the
                          discount is linear. sqrt left 30.0 at the pathological
                          case, saturating the cap and reproducing the exact
                          confidently-wrong failure it existed to prevent.
 R2 undefined contract  — both my instruction and the review assumed the caller
                          sums per-frame LLRs. Detectors aggregate internally via
                          probs.mean(), so Evidence.llr is whole-sample. The
                          discount double-counted, keeping 2% of evidence at
                          ESS=20. Fixed by DEFINING the contract, not a third exponent.
 R3 cap-saturated test  — I wrote ess=20/n=900 expecting 20.0, which equals
                          MAX_TOTAL_LLR. Correct (20.0), sqrt (134->capped 20.0)
                          and deleted (900->capped 20.0) are indistinguishable.
 R4 same trap again     — the clamp test written to fix R3 saturated identically
                          (900 and 5000 both cap to 20.0).

GENERAL RULE, now recorded in the task report:
  When a computation is clamped, a test asserting a value AT the clamp cannot
  distinguish anything below it. Choose fixtures where BOTH the correct and the
  broken result sit strictly below the bound, and add a self-guard asserting the
  expected value is below it so a later edit cannot silently re-saturate.
  Exception: a test whose PROPERTY is saturation (test_total_llr_is_capped) is
  legitimate at the bound.

Round 4 escalated to a fresh implementer (sonnet) per the rounds-4-5 rule, because
the same trap had been hit twice consecutively. It fixed it and audited every
other assertion in the file for the same pattern — none found.
Task 11: dispatched (haiku). BASE dbec0cc. Benchmark metrics.

Task 11 review: spec ✅; quality APPROVED. 0 Critical, 4 Important, 4 Minor.
Reviewer verified the pinned constants INDEPENDENTLY rather than re-deriving from
the code: AUC 2/3 by exact Mann-Whitney pairwise count, ECE 0.9 and 0.99 by hand
bin arithmetic, TPR 0.82 by a hand-rolled quantile computed outside bench.metrics.
All three legitimate. Also traced a partial-tie AUC case by hand (0.875, matches).

Settled my tie-handling concern: tpr_at_fpr's error under saturated/tied scores is
CONSERVATIVE, never optimistic — achieved FPR can fall short of requested but never
exceed it, so it cannot overstate our TPR in the head-to-head. It can however report
TPR=0 for an FPR budget smaller than the smallest tie-block fraction, which looks
misleadingly BAD rather than good. Acceptable direction; worth documenting.

Ruling: Important-2 is the one that matters most and I am treating it as the
priority. bootstrap_ci_by_group skips degenerate resamples (all-one-class) and
warns ONLY when every draw is dropped. Reviewer demonstrated 4.7% silently dropped
with 3 fraud videos out of 100. Those dropped draws are precisely the fragile ones
you want represented when positive groups are scarce, so skipping them narrows the
interval optimistically — the SAME "fabricates precision" failure this function
exists to prevent, in milder form, and in exactly the low-fraud regime this
benchmark is built for. Decision: count the drops, warn above a threshold, and
surface the rate so a reader can judge the interval. — Cost if wrong: one extra
number in the output.

Ruling: Important-1 (ece silently drops out-of-range probabilities and still
divides by the full N, understating ECE — demonstrated 0.675 instead of 0.9).
A calibration metric that silently makes a badly-calibrated model look better is
the wrong direction of error for this system. Decision: validate and raise.
Task 11: fix round 1/5 dispatched — FIX_BASE 265f13b.
Task 11: fix rounds 1-3 (commits 265f13b..05dc176, 137/137 green)
Task 11: complete (commits dbec0cc..05dc176, review clean, 3 rounds, 24 metric tests)
Re-review confirmed all 9 new ValueError contract tests use match=, so none can
pass on an unrelated ValueError, and the drop-rate test is seed-robust by
construction (natural ~4.7% degeneracy always exceeds the 1% threshold) rather
than by a lucky constant.

KEY MEASUREMENT retained from this task, worth carrying into the final report:
group-wise bootstrap CI width 0.751 vs row-wise 0.063 — 11.9x wider. That
quantifies how much precision frame-level resampling fabricates, and it is the
evidence behind spec §8.2 guard 2.

Lesson recorded: a threshold chosen without computing the base rate of the thing
it monitors will usually sit on the wrong side of it. The 5% degenerate-draw
warning sat above the 4.7% natural rate of its own motivating case, so two of
four seeds produced no message at all.
Task 12: dispatched (haiku). BASE 05dc176. The six hygiene guards — includes
acceptance criterion 2 (identity leakage), the largest currently-unmet criterion.
Task 12: fix rounds 1-2 (commits 0b009cc..fb020ee+, 154/154 green)
Task 12: complete — six hygiene guards. Acceptance criterion 2 now MET:
check_identity_disjoint returns a measured max-cosine number on the passing path.

Ruling: the 1e-3 parity floor SUPPRESSED the disparity it measures. Flooring only
the denominator pulls the ratio below truth for any nonzero lo < 1e-3: lo=0.0002,
hi=0.0008 computed 0.8 against a true 4.0x, passing any ceiling >= 1.0. A real 4x
demographic FPR gap would have run green in CI. Decision: remove the floor,
special-case lo==0 only. — Cost if wrong: an infinite ratio must be rendered
legibly rather than arithmetically.

Ruling: check_video_level was scheduled to ship VACUOUS, and that was MY plan
defect. Task 17's brief passed sample_ids as both arguments; the guard's only
failure condition is groups[i] != sample_ids[i], so it could never fire. My
pre-flight scan FOUND this and talked itself out of it ("correct for image
records"). That reasoning was wrong: an image being its own source must be
RECORDED explicitly, not aliased, or the guard dies silently the moment video
records arrive. Decision: plan corrected to read a real source_id field (commit
3666858), and the guard now REFUSES identical sequences. — Cost if wrong: callers
must populate source_id; that is the point.

LESSON: a guard that a plausible call can silently neuter is worse than no guard,
because it sits in the codebase as evidence of diligence.

=== SESSION BOUNDARY — context exhausted at Task 12 of 22 ===
RESUME AT: Task 13 (leave-one-generator-out splits). BASE = HEAD of p0-evidence-core.
Briefs for Tasks 13-22 are already staged in this workspace.

=== SESSION 2 — resumed at Task 13. BASE = 88fb27d, 154 tests green, branch pushed. ===

Pre-flight scan of Tasks 13-22 (rows I checked, not a verdict):

| Pair / task | produces -> consumes | finding |
|---|---|---|
| T13 -> T17 | `Split`, `logo_splits` -> runner | T17 never calls logo_splits; LOGO folds are not yet wired into the runner. Recorded as a gap, not fixed here — see ruling 5. |
| T12 -> T17 | `check_video_level(sample_ids, groups)` | T17 reads `r["source_id"]` (corrected last session) but its fixture never set the key -> guaranteed KeyError on first run. FIXED. |
| T12 -> T17 | `check_compression_coverage` | consistent, fixture carries `compression`. |
| T13 internal | its tests vs its code | identity test VACUOUS against its own code: fixture gave every fake a unique subject, so the leak the code produces was untestable. FIXED. |
| T17 internal | `_observation` vs the guard it feeds | aliased `source_id=record["sample_id"]`, the exact thing the corrected comment 10 lines above forbids. FIXED. |
| T17 internal | `dataset_hash` vs record schema | omitted `source_id`; two corpora differing only in source grouping hashed identically, so the Task 20 audit record could not tell them apart. FIXED. |
| T17 internal | known-gaps item 4 vs its own code | still described the vacuous call as "correct for image records". FIXED. |
| T16 -> T13 | `load_rd_cache`/`load_capture_sessions` -> records | T16 produces RDResult/CaptureSession, not split records. No adapter task exists. Recorded, see ruling 5. |
| T14, T15, T18-T22 | pairwise file/interface overlap | no contradictions found. |

Ruling: LOGO splits partition SUBJECTS, not records, and drop the fakes whose
generator wants one side while their subject sits on the other. The plan's
version partitioned only reals and assigned fakes by generator alone. Measured
on a fixture where five subjects are each faked by 2-3 generators: identity
leaked in 3/3 splits, 4 subjects per split — spec 8.2 guard 1 violated by
construction, while the task's own test passed because its fixture gave every
fake a unique subject. Spec 8.2 guard 1 (identity disjointness) binds over
"use every available fake", so the conflict resolves toward dropping. Drops are
counted in `Split.dropped_for_identity` and asserted non-empty, because on a
corpus where every subject is faked by every generator roughly half the fakes
fall out of each fold and a reader who cannot see that number will over-read
the fold. — Cost if wrong: folds are smaller than the corpus suggests, and the
drop count must be reported alongside every LOGO number.

Ruling: `logo_splits` REFUSES a corpus it cannot measure rather than emitting a
degenerate split. The plan's version, given one real subject, produced a split
with zero reals in test — an unmeasurable FPR presented as a fold. Same for a
single-generator corpus (zero train fakes) and a fake with `generator=None`
(silently joined the training side of every split, the one place it can never
be measured). All three now raise with a message naming the missing side.
— Cost if wrong: small or single-generator corpora need an explicit exemption
rather than working by accident.

Ruling: the seed test must prove the seed is CONSUMED, not merely that the
function is deterministic. The plan compared seed=5 against seed=5, which an
implementation ignoring `seed` entirely also passes — the same open-interval
class of vacuity the ledger already records. Added a test asserting the
partition differs across seeds. — Cost if wrong: none; it is strictly stronger.

Ruling: Task 17's `source_id` contract is repaired at the plan level now rather
than when Task 17 runs. It read `r["source_id"]` while its fixture never set the
key, so the runner would have KeyError'd on first execution; `_observation`
still aliased `sample_id`; and `dataset_hash` omitted the field so the Task 20
audit record could not distinguish two corpora differing only in source
grouping. Fixing at dispatch time would have cost a fix round. — Cost if wrong:
Task 17's fixture shape changes, and its implementer must follow the brief over
any memory of the older schema.

Ruling: the two gaps the scan found are RECORDED, not fixed. (a) No task wires
`logo_splits` into `run_benchmark` — Task 17 evaluates one flat record list, so
the LOGO number spec 8.1 calls "the only number that predicts field
performance" is computed by nothing. (b) No adapter turns Task 16's RDResult /
CaptureSession into split records. Both are new work, not defects in an
existing task, and inventing tasks mid-execution is how plans silently double.
They surface to the user at the end. — Cost if wrong: P0 ships a protocol
module and a runner that never meet, which is exactly the failure mode the
final review must catch.

Plan corrections committed as c114822.
Task 13: dispatched (sonnet). BASE c114822.

Pre-flight scan of Task 14 (done while Task 13 was in flight; plan edit deferred
until Task 13's implementer has committed, to avoid interleaving on the branch).

Measured against the planned code, Laplacian mean as the high-frequency-energy
proxy, on a STRUCTURED image (gradient + edges + 4px scanlines) and on the
plan's own white-noise fixture:

  perturbation           structured        white noise (plan's fixture)
  clean                    100.59                181.20
  jpeg                      97.77  (0.97x)       199.05  (1.10x RAISES)
  resize                    25.13  (0.25x)        16.39  (0.09x)
  blur                      13.93  (0.14x)        11.41  (0.06x)
  noise                    106.29  (1.06x)       181.80  (1.00x)
  screenshot_recapture      39.38  (0.39x)        34.16  (0.19x)
  print_recapture           35.08  (0.35x)        31.52  (0.17x)

GOOD NEWS: the module's central claim is TRUE and measurable. Both re-capture
paths destroy ~2/3 of high-frequency energy, which is exactly the evidence NPR
and most frequency-domain detectors depend on. Nothing in the task tests it.

Finding 14-A (Important): spec 8.3 asks for a "JPEG quality sweep". The planned
`robustness_sweep` emits exactly ONE jpeg point, at the default quality=50. One
point is not a curve and cannot show where a detector falls off.

Finding 14-B (Important): the fixture is uniform white noise, on which JPEG
RAISES high-frequency energy (1.10x) rather than lowering it. Any mechanism
assertion written against that fixture is measuring an artefact of the fixture.
Structured images are also the only ones on which "print re-capture" means
anything.

Finding 14-C (Important): nothing asserts the sweep is DETERMINISTIC. It is
(verified identical across two calls), but two perturbations draw from RNGs and
a later edit reaching for fresh entropy would silently destroy reproducibility
of the whole benchmark. For a harness whose value IS reproducibility this must
be pinned.

Finding 14-D (Important): `test_unknown_perturbation_raises` uses
`pytest.raises(KeyError)` with no `match=` — the exact vacuity class already
recorded twice in this ledger. Passes on any KeyError from anywhere.

Finding 14-E (Minor): `test_every_perturbation_preserves_shape_and_dtype` is the
shape/dtype-only class; a stub returning `img + 1` passes it and the
"actually changes" test together. Mechanism assertions replace it.

Finding 14-F (Minor): `_screenshot_recapture(img, seed=0)` never uses `seed`;
its moire and glare are deterministic by construction. Dead parameter.

Ruling (to apply when Task 14 is dispatched): jpeg becomes a real sweep over
several qualities; the fixture becomes structured; determinism, per-perturbation
mechanism, and `match=` assertions are added; the dead seed parameter goes. A
blanket "all perturbations reduce HF energy" assertion would be WRONG — `noise`
correctly raises it (1.06x) — so the mechanism test must be per-perturbation.
— Cost if wrong: the robustness surface reports a curve where the plan reported
a point, which is more output to render in Task 17.

NOTE: the live ledger is gitignored. Its durable copy is
docs/superpowers/ledger/2026-09-20-p0-execution-ledger.md (committed 88fb27d)
and is now STALE. Sync it before the session ends.

Pre-flight scan of Task 15 (white-box PGD baseline). Plan edit deferred to
dispatch time, same reason as Task 14.

Finding 15-A (CRITICAL): the task's ONLY end-to-end assertion is vacuous in two
independent ways at once, and it is the assertion that carries acceptance
criterion 8 — "what makes the state-sponsored threat model real rather than
decorative". Measured against the planned code and the planned fixture:

  clean TPR@FPR=0.1                      = 1.0
  attacked TPR, real PGD  (eps=0.3)      = 1.0
  attacked TPR, NO-OP attack (eps=0.3)   = 1.0
  the assertion `attacked <= clean`      : PASSES in all three cases

  (1) Bound saturation, the class this ledger already records twice: with a
      no-op attack, attacked == clean, and `<=` still holds. An unimplemented
      pgd_attack passes.
  (2) The fixture cannot be attacked anyway. Positives sit at mean pixel 0.80,
      negatives at 0.20, eps=0.3. The attack moves positives the full 0.3 to
      0.50 — attacked-positive score 0.5000 vs clean-negative score 0.3543 —
      so the ranking never flips and TPR stays 1.0 with a PERFECTLY WORKING
      attack. The number the task exists to produce cannot move.

Finding 15-B (Important): `seed` is dead code and its test cannot fail. The
planned PGD has no random start, so it is fully deterministic regardless of
seed: verified seed=3 and seed=999 produce identical tensors. `torch.manual_seed`
is called and nothing reads from the RNG. Separately, PGD without a random start
is BIM, not PGD — the random start is what the name denotes and it makes the
attack strictly stronger, which is what a *baseline* wants.

Ruling: the fixture becomes negatives 0.45 / positives 0.55 with eps=0.2, and
the assertion becomes exact rather than an inequality. Verified on that fixture:
clean = 1.0, real PGD = 0.0, no-op attack = 1.0. The test now separates a
working attack from a broken one, which the planned one never could. A margin
assertion (`attacked < clean - delta`) would also work, but an exact collapse to
0.0 against an exact clean 1.0 is legible in a way a margin is not, and this
number ends up in the report. — Cost if wrong: the fixture is tuned so the
attack succeeds completely, so it proves the attack CAN collapse a detector,
not how much eps a realistic detector survives. That is the right thing for a
unit test; the eps sweep against real weights is Task 18's job.

Ruling: `pgd_attack` gains a real random start inside the eps-ball, drawn from
an explicit `torch.Generator` seeded by `seed`, rather than calling the global
`torch.manual_seed`. Verified: seed=3 twice is identical, seed=3 vs seed=999
differs, the eps-ball bound still holds exactly (0.100000 <= 0.1) and output
stays in [0,1]. A seeded local generator also avoids perturbing global torch RNG
state for every other test in the suite, which the planned `torch.manual_seed`
call does. — Cost if wrong: adversarial numbers shift slightly between seeds,
so the report must name the seed it used.

Pre-flight scan of Task 20 (immutable audit record — the restored spec 7.2
requirement). Plan edit deferred to dispatch time. All three verified by running
the planned code.

Finding 20-A (Important): `test_record_is_immutable` uses
`pytest.raises(Exception)` — the broadest catch there is. An AttributeError from
a typo in the test satisfies it. Must be `dataclasses.FrozenInstanceError`.

Finding 20-B (Important): `test_record_never_contains_image_bytes` is vacuous.
It searches the JSON for the raw string `\x89PNG` — seven literal characters,
not PNG magic bytes — which no implementation ever inserts. Verified: a record
with a whole 64x64x3 image smuggled through `model_versions` serialises without
that substring, so the test PASSES on a record carrying image data. The cause is
`json.dumps(..., default=str)`, which silently stringifies anything
non-serialisable rather than refusing it. For a record whose digest is the
tamper-evidence, silently absorbing an unexpected type is the wrong failure
direction. Ruling: drop `default=str`, let serialisation raise, and test that a
non-JSON value is REFUSED rather than absorbed.

Finding 20-C (Important): `record_digest` excludes `created_at`, so backdating a
decision is invisible to the tamper-evidence. Verified: moving created_at from
2026-09-20 to 1999-01-01 leaves the digest identical. The exclusion exists only
so that `test_digest_is_stable_for_identical_records` can call `_record()` twice
and get matching digests — the guarantee was weakened to fit the test, and the
neighbouring test is then named `test_digest_changes_when_any_field_changes`
while checking one field. For an audit record defended to a regulator, the
timestamp is among the most attack-relevant fields there is. Ruling: `created_at`
becomes an injectable parameter defaulting to now, so two genuinely identical
records are identical, and the digest covers it. — Cost if wrong: callers that
want a fresh timestamp keep the default and nothing changes for them.

Finding 20-D (Important): `frozen=True` is shallow, so the "immutable" record is
not. Verified: `r.model_versions["npr"] = "tampered"` raises nothing and CHANGES
the digest — a record can be altered after the fact and re-digested to match.
`test_record_is_immutable` only tries to rebind `.verdict` and never looks at the
mutable containers. Ruling: freeze the containers (MappingProxyType for the dict,
tuple for evidence rows) and assert mutation raises for each.

These three are the same shape: the record is called immutable and
tamper-evident, and neither property is true or tested. Spec 7.2 is the one
requirement whose whole purpose is surviving hostile scrutiny.

Task 14 ruling REFINED before dispatch, after measuring threshold stability.
High-frequency energy is the right mechanism probe for four of the five
perturbations but the WRONG one for JPEG: blocking artefacts ADD edges at block
boundaries, so HF ratio across q=90..10 is 0.97, 0.98, 0.97, 0.95, 1.00 — flat
and non-monotonic. Mean absolute distortion from clean IS monotonic over the
same sweep (14.77, 14.82, 15.24, 17.32, 19.08), so the plan's original
`test_jpeg_quality_is_monotonic_in_degradation` was right and stays; it becomes
an all-adjacent-pairs assertion over the sweep rather than a two-point check.
Had I applied the blanket "assert each perturbation lowers HF energy" ruling as
first written, the JPEG case would have been wrong.

Thresholds pinned from measured ratios over 5 jittered structured images
(min/max across seeds, so the margins are real, not one lucky draw):
  resize               0.231-0.233   -> assert < 0.5
  blur                 0.130-0.131   -> assert < 0.3
  noise                1.018-1.022   -> assert > 1.0  (noise ADDS high frequency)
  screenshot_recapture 0.360-0.364   -> assert < 0.6
  print_recapture      0.307-0.309   -> assert < 0.6
Task 13: implemented b9e55d6 (58 new tests, 212 total green, ruff+mypy clean).
Task 13: review found 4 Important, 0 Critical. Three of the four are defects I
wrote into the brief; the reviewer verified each by probe rather than by reading.

Ruling: review finding 1 (the vanishing-subject repair is one-directional) is
LOAD-BEARING and enters the fix round. The implementer correctly found that a
test-slated subject with no real record and no fake by the held-out generator
places nothing and vanishes; it moved such subjects to train. The exact mirror
— a train-slated subject with no real whose fakes are ALL by the held-out
generator — is unhandled, and those are precisely the held-out fakes TPR is
computed from. Reviewer demonstrated a corpus with a valid zero-drop split for
both folds that `logo_splits` REFUSES at all 8 seeds. The symmetric move is
Pareto-improving by the same argument as the existing one. — Cost if wrong:
folds place more fakes than a conservative reading would, so the drop counts
fall; they remain reported.

Ruling: `Split` gains the per-fold subject partition (`train_subjects`,
`test_subjects` as frozensets). This is scope I am adding beyond the brief, and
I am adding it because the reviewer identified it as the root of the whole
episode: the brief's test could observe a subject's side ONLY by looking for it
in `s.test`, so a subject that placed zero records read as a train subject, and
the implementer changed production placement to satisfy an oracle that was
simply blind. With the partition exposed, the test asserts against the intended
side directly, and the drop counts become interpretable to the Task 17 reporter.
— Cost if wrong: two more fields on a dataclass consumers may ignore.

Ruling: review findings 2 and 3 are plan-mandated — I wrote both tests — and
both are defects, so both enter the fix round. The plan's authorship does not
excuse them. (2) `test_no_source_video_straddles_the_split` cannot fail: my
`rec()` helper defaults `source_id` to `sample_id`, so no two fixture records
ever share a source and the assertion restates sample-level disjointness. The
guard-2 dimension the spec actually describes — several frames sharing one
source video — is exercised nowhere on the passing path. My own fixture commits
the aliasing the module forbids. (3) `match="train fakes"` discriminates
nothing: `_require_measurable` embeds the full counts dict in the message and
that dict always contains the literal `'train fakes'`, so the regex matches any
failure of that function. This is a new variant worth recording — a `match=`
that is present, looks specific, and matches unconditionally because the message
interpolates a structure containing every key.

Ruling: two findings the reviewer classed Minor are promoted into the fix round
because they contradict the module's stated contract rather than merely
polishing it: a record with `label=2` is silently treated as a fake (it is not
caught by `_validate`, which only tests `label == 1`, nor counted by
`_require_measurable`, which counts only 0 and 1), and a REAL carrying a
generator string passes validation and can then trigger a confusing
source-straddle rejection for an unrelated reason. A module whose job is
refusing corpora it cannot measure should not have a silent third label.
— Cost if wrong: two more rejections callers must satisfy.

Task 13: remaining minors DEFERRED to final review — duplicate
`match="missing required"` in the two missing-key tests (neither verifies the
specific key is named); determinism test compares only `test_ids()`, not train
or dropped; `frozen=True` on a dataclass holding the caller's mutable dicts buys
nothing; `cut = max(1, len//2)` gives the test side the larger half on odd
subject counts, unstated; `Split` imported only to be noqa'd.
Task 13: fix round 1/5 dispatched — FIX_BASE b9e55d6.
Task 14: corrected section staged and VERIFIED before dispatch (scratchpad):
extracted its own code and test blocks, ran them — 12 passed, matching the
section's stated expectation. Then ran the Step 5 proof-by-deletion myself:
neutering both re-capture functions to no-ops fails exactly the two mechanism
tests (screenshot_recapture and print_recapture at the 0.6 limit) and restoring
returns 12/12. So the proof I am asking the implementer to perform is known
achievable, and the reference implementation does not fail its own tests — the
failure mode Task 13 hit.
Task 15: corrected section staged and VERIFIED before dispatch (scratchpad, with
the real bench/metrics.py): 11 tests pass. Ran the Step 5 proof-by-deletion
myself — making pgd_attack return its input unchanged fails
test_attack_collapses_tpr_to_zero with "1.0 where 0.0 expected", and restoring
returns 11/11. That is precisely what the planned `assert attacked <= clean`
could never do, since a no-op attack satisfies it.
Task 13: fix round 1 complete (1a3032b, 92 protocol tests, 246 total green);
implementer added an UNREQUESTED real-subject rebalance because the symmetric
move alone still failed 3/8 seeds — flagged to the re-reviewer as unreviewed
production logic.
Task 13: fix round 1/5 (6 addressed, 0 open; commits b9e55d6..1a3032b)
Task 13: complete (commits c114822..1a3032b, review clean, 92 protocol tests,
246 total green)

Re-review independently REPLAYED every claim rather than accepting the report:
ran the reviewer's 4-record corpus against the shipped module (8 seeds, both
folds, zero drops); fuzzed 898 accepted corpora to confirm the exposed partition
matches actual placement (0 mismatches); replayed both move-deletions and
matched the reported failure counts exactly (6/8 and 7/8 seeds); and built a
rebalance-free copy to confirm the unrequested rebalance is genuinely
load-bearing (seeds 1,2,3 raise without it). Compared with/without across 6000
random corpora: 154 rescued, 0 newly refused, 0 with increased drops — strictly
Pareto-improving on that sample.

KEY MEASUREMENT worth carrying forward: identity-disjoint LOGO folds DROP a
large fraction of fakes on corpora where subjects are faked by several
generators — measured at 6-8 of 12 fakes per fold on the task fixture. Any LOGO
number must be reported with its drop count or it will be over-read.

Task 13: minors DEFERRED to final review (from both review rounds):
 - tests/bench/test_protocol.py:222-226 docstring MISATTRIBUTES its coverage:
   it now also guards the real-subject rebalance, but names only the symmetric
   move and a 'no test fakes' failure, whereas deleting the rebalance fails it
   with 'no test reals'/'no train reals'. A future reader deleting the rebalance
   is actively misdirected. Cheapest real fix in this list.
 - bench/protocol.py:132-146 the real-subject rebalance has no test naming it
   and no proof-by-deletion in the report (it IS covered implicitly).
 - tests/bench/test_protocol.py:193 match="label" and :205 match="generator"
   are loose; "generator" also appears in the unattributed-fake and
   source-straddle messages, so it does not discriminate which rejection fired.
 - bench/protocol.py:139-146 always moving the alphabetically-first real subject
   makes the rescued partition seed-independent in the collapsed case, reducing
   seed-to-seed variation on small corpora.
 - duplicate match="missing required" in the two missing-key tests.
 - determinism test compares only test_ids(), not train or dropped.
 - frozen=True on a dataclass holding the caller's mutable dicts buys nothing.
 - cut = max(1, len//2) gives the test side the larger half on odd counts.
 - Split imported in the test only to be noqa'd.
 - bench/guards.py carries 8 PRE-EXISTING F541 ruff findings (f-string with no
   placeholders combined with %-formatting, e.g. line 80). Not from this task,
   but they will fail the Task 22 ruff gate — fix there or before.
Task 20: corrected section staged and VERIFIED before dispatch — and it FAILED
its own tests on the first run, 15 of 25. Root cause: `dataclasses.asdict()`
deep-copies every field value and a `MappingProxyType` cannot be deep-copied
("TypeError: cannot pickle 'mappingproxy' object"). So the deep-freeze that
Finding 20-D requires is incompatible with the obvious serialisation route.
Fixed by walking `dataclasses.fields()` explicitly instead of `asdict()`, with
the reason recorded in the docstring so nobody reinstates it. 25/25 after.

This is the second reference implementation I have written that failed its own
tests (Task 13 was the first, caught by its implementer). Verifying each staged
section by running it is now standard for the rest of this plan, not optional.

Both Step 5 proofs verified achievable: dropping `created_at` from the digest
fails EXACTLY the created_at parametrisation while the other 11 field cases
still pass (so the test discriminates rather than collapsing), and replacing
`_freeze(dict(...))` with `dict(...)` fails the in-place mutation test.
Restoring returns 25/25.
Task 14: implemented 6ef3196 (12 new tests, 258 total green, ruff clean).

Ruling: Task 14's implementer flagged that `mypy --strict` fails on
bench/robustness.py and left it, following house style. I VERIFIED the claim
rather than accepting it — bare `np.ndarray` annotations fail --strict
repo-wide: bench/robustness.py 10 errors, bench/metrics.py 7,
src/dfd/calibration.py 8. bench/protocol.py passes only because it happens to
annotate `list[dict[str, Any]]` and never a bare array. There is NO [tool.mypy]
section in pyproject.toml, so nothing has ever enforced this. The implementer
made the right call: inventing a local fix would have diverged one file from
the other five. — Cost if wrong: nothing now, but see the gap below.

GAP for Task 22 (CI gates), recorded because its brief does not currently
account for it: "mypy --strict" cannot be switched on as a gate without first
either parameterising every array annotation across bench/ and src/dfd/
(npt.NDArray[np.uint8] and friends) or configuring mypy to permit bare
generics. That is real work in at least four modules and is not in Task 22's
scope as written. Whoever runs Task 22 must size it before promising the gate.
Task 14: review found 1 CRITICAL, 1 Important. The Critical is MY defect and it
is the sharpest lesson of this session so far.

`_screenshot_recapture` crashes on ANY non-square image. `yy = np.arange(h)[:, None]`
has shape (h,1), which numpy left-pads to (1,h,1) against an (h,w,3) array — it
broadcasts only when h == w. Verified on the shipped module:
  (128,128,3)  all six perturbations OK
  (100,150,3)  screenshot_recapture ValueError
  (720,1280,3) screenshot_recapture ValueError
  robustness_sweep on a 720x1280 frame — ValueError
So one of the two capabilities this task exists to deliver (acceptance
criterion 9) cannot run on a realistic video frame, and all 12 tests pass.

The bug was in the ORIGINAL plan and SURVIVED my correction, because my
structured fixture is 128x128 — square, like the white-noise fixture it
replaced. I rewrote this task specifically to kill the untested-dimension
defect class, wrote the class into the reviewer's briefing myself, verified the
section by running it, ran its proof-by-deletion — and every one of those
checks used a single square shape. Verifying a section by running it is
necessary and is NOT sufficient: it proves the code passes the fixtures I
thought of, which is precisely the blind spot the class describes.

Ruling: image shape becomes a PARAMETRISED DIMENSION across this file's tests,
not one extra case bolted on. Square, portrait, landscape, and odd-sized, over
every perturbation. A single non-square test would fix this instance and leave
the class alive. — Cost if wrong: the test file grows by a parametrize
decorator.

Ruling: the Important finding stands — one-sided HF bounds
(`hf(out) < limit * hf(clean)`) are also satisfied by a `return
np.zeros_like(img)` stub, which has ~zero high-frequency energy and also passes
the shape, dtype and "actually changes" tests. Companion assertion measured
rather than guessed: correlation with the clean image discriminates cleanly —
jpeg 0.998, noise 0.997, print_recapture 0.877-0.882, blur 0.801-0.804,
resize 0.785-0.788, all across three shapes, against EXACTLY 0.0 for a
zeros_like stub. Threshold 0.5 carries huge margin. — Cost if wrong: one more
assertion per perturbation.
Task 14: fix round 1/5 dispatched — FIX_BASE 6ef3196 (re-review will diff from
dcbe6c8, excluding the two plan-correction commits made while the review ran).

Pre-flight scan of Tasks 16-22 (pattern sweep for the tracked defect classes,
then a full read of the one that lit up). Hits: T16 0, T17 1, T18 0, T19 1,
T21 5, T22 0.

Finding 21-A (CRITICAL): Task 21's resource limits DO NOT DEFEND AGAINST THE
ATTACK THEY NAME, and the task contradicts itself in its own text. Its opening
says "Limits must be enforced BEFORE allocation, not after" and its module
docstring says "Checks run on metadata, before allocation" — but the wiring
instruction reads: "in src/dfd/ingest/image.py, call check_file_size(path)
before cv2.imread, and check_frame_dims(rgb.shape[1], rgb.shape[0]) AFTER
decode." `rgb` only exists once cv2.imread has already allocated. The check runs
after the harm.

The file-size check does not cover for it, because bypassing file size is the
definition of a decompression bomb. Measured: a 12000x12000 uniform PNG is
161,331 bytes on disk — 0.15 MB — and decodes to 0.40 GB. It sails through the
256 MB default. The task's own headline example, 50000x50000, is roughly 2.8 MB
on disk and 7.5 GB decoded.

Ruling: dimensions must be read from the image HEADER before any decode.
Pillow is already available in this environment and does exactly this — verified
`Image.open(path).size` returns (12000, 12000) in 0.007s without decoding.
check_frame_dims then runs on those header dims, and only a frame that passes is
decoded. Pillow also carries its own MAX_IMAGE_PIXELS bomb guard, which should
be set from our Limits rather than left at the library default so there is one
source of truth. — Cost if wrong: Pillow becomes a hard dependency of the ingest
path, alongside OpenCV which is already there.

Finding 21-B (Important): `test_limits_are_a_frozen_value_object` uses
`pytest.raises(Exception)` — the same broadest-possible catch already ruled on
in Task 20. Must be `dataclasses.FrozenInstanceError`.

Finding 21-C (Important): three of the remaining four raises assert only the
BASE class. `test_zero_or_negative_dimensions_are_rejected` and
`test_missing_file_raises_a_typed_error` both accept any `DfdError`, and
`ResourceLimitExceeded` IS a `DfdError` — so an implementation that raised
"resource limit exceeded" for a missing file or a zero dimension passes. Those
are InvalidInput conditions and the tests must say so, with `match=`.
`test_decode_bomb_dimensions_are_rejected` has no `match=` at all.

Finding 21-D (Important): `test_defaults_are_documented_constants` asserts only
`> 0` for all three constants, so `max_pixels = 1` passes a test whose name
claims it documents them. Assert the actual values, so changing a limit is a
deliberate edit to a test rather than a silent drift.

Finding 21-E (Important): untested dimension — every bomb case is SQUARE
(50000x50000). A degenerate strip such as 1 x 10**10 or 10**10 x 1 has the same
pixel count and is the shape an attacker reaches for when a naive check tests
each side against a max dimension instead of the product. Parametrise the shape.
(This is the same class that just shipped a Critical in Task 14, from a fixture
that was square in every test.)

Finding 21-F (Important): nothing tests that the limits are WIRED IN. Task 21
modifies ingest/image.py and ingest/video.py, but every test calls
check_file_size / check_frame_dims directly, so both could be enforced nowhere
and the suite stays green — the "guard that ships vacuous" failure this plan
already hit once with check_video_level. Needs a test that drives a bomb through
the real `load_image` and asserts it raises before allocating.
Task 14: fix round 1/5 (3 addressed, 0 open; commits 6ef3196..82a4185)
Task 14: complete (commits f04041e..82a4185, review clean, 58 robustness tests,
304 total green)

Re-review verified the RED count arithmetically against the parametrisation
actually in the diff (7 shape-parametrised tests x 3 non-square shapes = 21,
square exempt since h==w) rather than trusting the number, and confirmed the
odd shape (97,131) genuinely exercises _resize's truncation path. It also chased
the nan question properly: `nan > 0.5` is a clean False so the assertion raises
a normal AssertionError, and running the real suite under
`-W error::RuntimeWarning` showed zero warnings — np.corrcoef only emits
"invalid value encountered in divide" for a constant array, which exists solely
in the reverted mutation.

Task 14: minor DEFERRED — tests/bench/test_robustness.py:89 `_correlation`
docstring still says "exactly 0.0 for a zeros_like stub"; the measured value is
nan. The implementer reported the discrepancy honestly in its report but did not
update the docstring. Cosmetic; both compare False.

Task 21: corrected section staged and verified — 25/25, but only after a second
failure of my own: my `context` fixture invented `Context(source=..., captured_at=...)`
and the real dataclass takes subject_id/generator/compression/label/meta. Two of
25 errored. That is the third staged section of mine to fail its own tests
(Task 13's placement bug, Task 20's mappingproxy/asdict clash, now this).
Pattern worth naming: every one was an assumption about code I had not opened —
I keep writing against a remembered interface instead of reading the real one.
Task 21: correction committed 9221ec4, brief regenerated, both Step 5 proofs
verified achievable (ordering: moving the checks after a decode fails the
"cv2.imread was called" assertion; wiring: deleting the call from load_image
fails the loader test). 25/25 restored.

GAP for Task 22, second one: pyproject.toml declares NO dependencies at all —
no [project.dependencies] key exists. numpy, opencv, torch and now Pillow are
all imported and none is declared. CI cannot install this project from
pyproject as it stands, so the Task 22 gate work includes writing the
dependency list, not just wiring the gates. Combined with the mypy gap already
recorded, Task 22 is larger than its brief implies and should be re-scoped
before dispatch.

Pre-flight scan of Task 17 (benchmark runner) — the most consequential findings
in this plan so far.

Finding 17-A (CRITICAL, structural): THE HARNESS NEVER COMPUTES THE LOGO NUMBER.
Task 17 contains no mention of `logo_splits`, `protocol`, `Split` or `held_out`
— grepped, zero hits. `run_benchmark` takes ONE flat record list and computes a
single `auc` over all of it. That is in-dataset AUC, which the design spec
describes in Task 13's own words as measuring MEMORISATION, and spec 8.1 calls
leave-one-generator-out "the only number that predicts field performance".
So Task 13 builds the splitter, and nothing in the plan consumes it: the
benchmark measures the number the spec calls meaningless and does not measure
the one it calls essential. I flagged this in the session's first pre-flight as
a gap and deferred it as "new work"; having now read Task 17 properly, that was
too generous. It is a defect in the plan's core deliverable, not a missing
enhancement.

Finding 17-B (Important, precise): the runner guards a property and then
violates it eleven lines later. The comment at the top of `run_benchmark` reads
"`groups` MUST identify the SOURCE VIDEO, never the sample id" and passes
`source_id` to check_video_level — then line 266 sets
`groups = np.array([r["sample_id"] for r in records])`, and THAT is what
`bootstrap_ci_by_group` resamples over. Scope of the harm, stated accurately:
`check_video_level` enforces one sample per source, so while guards are on the
two are 1:1 and the numbers agree. But `enforce_guards=False` is an explicitly
supported, explicitly tested option, and on that path a multi-frame corpus
bootstraps over frames. Task 11 measured what that costs: group CI 0.751 vs row
CI 0.063, 11.9x narrower. The waiver path is exactly where an honest interval
matters most.

Finding 17-C (Important): `assert p95_latency_ms >= 0.0` — satisfied by a stub
that never measures anything and returns 0.0. Bound-saturation class.

Finding 17-D (Important): `test_abstention_rate_is_reported` has the docstring
"A detector that abstains on everything must be visible as such", registers a
detector with min_quality_band="high" to force exactly that, and then asserts
only `0.0 <= rate <= 1.0`. The open-interval class, with the intended assertion
written out in prose directly above the one that does not test it. A detector
abstaining on everything should assert rate == 1.0.

Finding 17-E (Important): `test_guards_run_by_default_and_fail_the_run` uses
`pytest.raises(GuardViolation)` with no match=. Six guards can raise it; the
test intends compression coverage specifically and would pass if an unrelated
guard fired for an unrelated reason.

Finding 17-F (Minor): `test_dataset_hash_is_stable_and_content_sensitive`
mutates only `sample_id`. The hash now covers six fields; parametrise, as Task
20's digest test now does.

Finding 17-G (Minor): every fixture image is 64x64 — square and uniform. That
is the shape assumption that just produced a Critical in Task 14.

Ruling deferred on 17-A until Tasks 15-16 land, because the fix has real scope
and I want it shaped once rather than twice. The direction: `run_benchmark`
should evaluate per LOGO fold and report the held-out-generator number as the
headline, with any whole-corpus AUC labelled in the output as in-dataset
memorisation rather than presented beside it as an equal. Recording now so it
cannot be lost if this session ends.

Pre-flight scan of Task 16 (corpus loaders). Its pattern sweep came back clean;
reading it found two real defects, one of them environmental and invisible to
any pattern.

Finding 16-A (CRITICAL, plan-mandated): Task 16 creates a top-level package
named `datasets/` at the repo root. HuggingFace `datasets` 3.0.1 IS INSTALLED in
this environment (alongside transformers 4.41.2 and huggingface_hub 0.36.2,
which this project already uses for the dima806 ViT). pytest is configured with
`pythonpath = ["src", "."]`, so the repo root is prepended to sys.path and our
package WINS. Proven by construction:

  import datasets -> <repo>/datasets/__init__.py
  has HF load_dataset? False
  from datasets import load_dataset -> ImportError: cannot import name
      'load_dataset' from 'datasets'

So creating this package silently breaks every HuggingFace dataset import in the
project. That matters beyond tidiness: FF++, Celeb-DF and DFDC — the corpora on
the critical path — are routinely loaded through HF `datasets`, and Task 22's CI
would inherit the breakage. This is the same shadowing failure the project
already hit once, when a stray tests/bench/__init__.py shadowed the real bench/
package; that one was caught by an import error, this one would be caught by
someone's HF loader failing much later with a confusing message.

Ruling: the package is named `corpora/`, not `datasets/`. Module paths become
corpora/rd_cache.py and corpora/captures.py, tests tests/corpora/. — Cost if
wrong: one directory name differs from the plan's text, and Task 17's imports
must follow.

Finding 16-B (Important): `aggregate_is_max_like` is only ever asserted True
(line 56 of the task, `assert aggregate_is_max_like(results, tolerance=0.2) is
True`, the sole call site in the tests). A constant `return True` implementation
passes the suite. This function encodes one of the three headline measurements
in the handoff — that Reality Defender's aggregate tracks the MAXIMUM of its
members, which is why its false-positive rate approximates the union of theirs —
so a function that cannot distinguish max-like from mean-like is worse than
absent: it launders the claim. Needs a negative case built from an ensemble
whose aggregate tracks the mean, asserted False.

Finding 16-C (Minor): `load_capture_sessions` silently skips malformed sessions
(`test_malformed_session_is_skipped_not_fatal`). Across 442 sessions of which
exactly 5 are the fraud that matters, a silently dropped session could be one of
the 5 and nobody would know. The loader should return or log a skip count.

Finding 16-D (Minor): every fixture session writes `frame_count: 1`. Untested
dimension, same class as Task 14's square-image fixture.
Task 15: implemented 48a6830 + 130d91f (12 tests, 316 total green).
Task 15: review found 0 Critical, 2 Important — both tests that cannot fail for
the reason they name, both demonstrated by mutation rather than argued.

The reviewer mutation-tested the attack's numerics before looking at anything
else, which is the right order for a routine whose correctness is invisible from
a green suite: gradient direction (descent mutant fails 2 tests), projection
against the ORIGINAL x rather than the iterate (mutant drifts to 0.08 with
eps=0.03), grad.sign() load-bearing (mutant fails 3), local Generator genuinely
isolating global RNG, and alpha fidelity. All correct. It also noticed
unprompted that using torch.autograd.grad rather than loss.backward() leaves
model .grad buffers untouched, so repeated attacks cannot corrupt a caller's
training state.

Ruling: review finding 1 enters the fix round. `test_negatives_are_left_clean`
asserts the CALLER's tensor is unmutated, but `adversarial_tpr` does
`adv = x.clone()` and writes only into `adv`, so x is never mutated no matter
which rows are attacked. Reviewer replaced the function with one attacking the
ENTIRE batch and all 12 tests still passed — the collapse test included, because
attacking negatives lifts them 0.45->0.65 while positives fall to 0.35, so TPR
is still 0.0. Positives-only is the spec-load-bearing choice (attacking
negatives moves the threshold and understates the detector, which is what the
3A.4 demotion rides on) and it was entirely unguarded. The implementer's report
claimed the opposite — that the test "would catch adversarial_tpr accidentally
attacking the whole batch". It does not. Decision: assert on what the model is
actually SCORED on, via a spy module, not on the caller's tensor; keep the old
test as a no-in-place-mutation guard under an honest name. — Cost if wrong: one
more test double in the file.

Ruling: review finding 2 enters the fix round. The in-loop [0,1] clamp is
untested — deleting it leaves all 12 green. `test_pgd_output_stays_in_valid_pixel_range`
uses x=0.99 with y=1, and ascending the loss on class 1 drives pixels DOWNWARD
for this model, so the iterate walks away from 1.0 and only the initialisation
clamp is exercised. Decision: flip the label so the attack pushes INTO the
bound (x=0.99 with y=0; reviewer verified the shipped code passes and the
clamp-removed mutant fails at max=1.09), and mirror it at x=0.01 with y=1.
— Cost if wrong: none, strictly stronger.

Resolved the reviewer's warning myself, as the controller holds the cross-task
view it lacks: `adversarial_tpr` having no caller outside its own tests is a
KNOWN, DELIBERATE deferral, not an oversight. The plan's own known-gaps list
records it — the runner wires `adversarial_tpr_at_1pct=None` because attacking
requires a differentiable model with real weights, and every P0 detector
abstains without weights. There is nothing to attack yet. It is wired when real
weights land, and the handoff's critical path (dataset EULAs, then training or
fine-tuning) is what unblocks that. NOT a Task 15 finding.

Task 15: minors DEFERRED — `test_zero_epsilon_is_the_identity` passes with the
eps==0 early return deleted (behaviourally identical, so not a correctness
risk); the random start's ball containment is unpinned and only observable at
steps=0, since the first projection repairs it; `torch.rand` uses the default
dtype so a float16/float64 input comes back in a different dtype than it was
handed (pass dtype=x.dtype); the `alpha = max(eps/4, 1e-4)` default is a real
design choice undocumented in its docstring. The reviewer also checked the
clamp ORDER and correctly found it is not a live risk — for x in [0,1] the two
clamps commute, so a swapped-order mutant is behaviourally identical rather
than merely undetected. Recorded because it was asked and answered.
Task 15: fix round 1/5 dispatched — FIX_BASE 130d91f.
Task 15: fix round 1/5 (2 addressed, 0 open; commits 130d91f..e67422f)
Task 15: complete (commits 9221ec4..e67422f, review clean, 14 adversarial
tests, 318 total green)

Both findings were fixed by changing TESTS ONLY — bench/adversarial.py was not
modified, because the implementation was already correct and the tests simply
failed to pin it. The re-reviewer confirmed that rather than accepting it:
checked the diff touches only the test file, then traced the new spy test
against the production code to confirm `seen[-1]` is the final whole-batch
SCORING pass and not one of pgd_attack's ten internal forward calls — the
specific doubt I asked it to chase, since a spy capturing an intermediate pass
would assert the wrong tensor and still look like it worked. It also re-derived
the gradient direction from the model's logits (logit0 = -mean(x)+0.5,
logit1 = mean(x)-0.5) to confirm y=0 drives pixels UP into the 1.0 bound and
y=1 drives them DOWN into 0.0, which is why the original fixture never
exercised the clamp.

Task 15: minor DEFERRED — a PytestDeprecationWarning about
asyncio_default_fixture_loop_scope appears when running a single test file
directly (pytest-asyncio plugin config). Pre-existing infra noise, absent from
full-suite runs. Belongs with the Task 22 CI work.
Task 16: dispatched (sonnet). BASE e67422f.

Ruling on Finding 17-A, the deferred one: Task 17 now computes LOGO. Committed
d49de40. run_benchmark evaluates per held-out generator, reports the WORST fold
as headline, and the report labels the whole-corpus table "memorisation, not
field performance". Scoring runs once per detector and folds slice it, since a
record's score does not depend on its fold. — Cost if wrong: RunRecord carries
one more field and the report has one more table; the in-dataset number is
still there, just demoted.

Recorded in the plan as a known gap rather than papered over: this is NOT a
trained LOGO protocol. P0 detectors are not trained, so each fold's TRAIN side
is unused. It becomes load-bearing when calibration lands, and is where the
operating threshold must be frozen to make spec 8.2 guard 5 real instead of the
string check on config.threshold_source that it is today.

THE FIXTURE FINDING, which I did not predict and only found by running it:
Task 17's 64x64 fixture images measure quality band "reject", below
SyntheticDetector's "low" floor, so EVERY detector abstained on EVERY record.
AUC, CI, TPR and ECE were nan throughout, meaning the runner's entire metric
path was never exercised by any test. Two of its tests passed only because of
that (`p95_latency_ms >= 0.0` and `0.0 <= abstention_rate <= 1.0`), and
test_run_is_reproducible_given_a_seed compared nan == nan — so Task 17 would
have FAILED its own test the moment it was dispatched. Fixture is now >=128px
short side, non-square, varying.

Measured band by size, for anyone choosing fixture dimensions later:
  64x64 reject | 48x64 reject | 64x88 reject | 128x128 low | 112x144 low
  160x224 medium

Proof-by-deletion: removing the LOGO wiring fails 5 tests. It initially failed
only 3 — two tests iterated `rec.logo_results.values()` and an empty dict never
enters the loop, so they passed vacuously under the very mutation they existed
to catch. Both now assert the fold count first. That is the same empty-iteration
vacuity in a new costume, caught only because I ran the mutation rather than
assuming the tests covered it.
Task 16: implemented 9af0af4 (16 tests, 334 total green, ruff clean).
Task 16: review found 0 Critical, 2 Important — both plan-mandated, both mine.

Ruling: review finding 1 enters the fix round. Both loaders catch only
`(json.JSONDecodeError, OSError)`, so a file that is VALID JSON of the wrong
shape sails past the guard and then crashes on use. Verified — six distinct
malformations each kill the entire load rather than skipping one session:
  captures, a JSON list          AttributeError: 'list' object has no attribute 'get'
  captures, a JSON string        AttributeError: 'str' object has no attribute 'get'
  captures, frame_count "many"   ValueError: invalid literal for int()
  rd_cache, a JSON list          AttributeError: 'list' object has no attribute 'get'
  rd_cache, score "high"         ValueError: could not convert string to float
  rd_cache, model without score  KeyError: 'score'
This directly contradicts the comment I wrote three lines above it — "Never
silent: a dropped session may be one of the five that are the actual fraud" —
because the failure is not silent, it is fatal: one malformed session out of 442
takes the whole corpus down, including the five that matter. Decision: validate
the parsed object is a dict and wrap the per-field coercions, so a
malformed-but-valid-JSON file is skipped and logged exactly like a decode
failure, with a test per loader. — Cost if wrong: the loaders tolerate more
garbage, and the skip log is where you look to find out.

Ruling: review finding 2 enters the fix round. `"1" in caplog.text` is not
merely weak — it cannot fail. Verified the JSONDecodeError message is
'Expecting property name enclosed in double quotes: line 1 column 2 (char 1)',
which contains "1" three times on its own, so the assertion passes on the error
text regardless of whether the skip COUNT is right or the counting logic exists
at all. The implementer flagged it as weak and left it because the brief
specified it verbatim; the brief was mine and it was wrong. Decision: assert the
literal count phrase the format string actually produces ("skipped 1"), or read
caplog.records structurally. — Cost if wrong: none.

Task 16: minors DEFERRED — test_tolerance_is_load_bearing never probes the exact
mean(diffs) == tolerance boundary, so a `<` vs `<=` inversion survives; no
empty-directory test for captures specifically (rd_cache covers the mechanism);
test_loads_results_and_skips_quota_file's name overstates what it verifies,
since the skip is an artifact of the */result.json glob rather than any
quota-aware code path.
Task 16: fix round 1/5 dispatched — FIX_BASE 9af0af4.

Pre-flight scan of Task 18, corrected and committed 8b6ae6a. Three defects,
plus one process bug of my own.

18-A: the fixture omits `source_id`, which the corrected run_benchmark now
reads. Task 18 would have KeyError'd on dispatch.
18-B: 64x64 images again — quality band "reject", so every detector abstains
and the robustness surface this task exists to produce is nan in every cell
while its tests pass on key presence alone. Same root cause as Task 17's, which
I only found by running it; here I found it by knowing to look.
18-C: single generator, leaving LOGO undefined for its corpus.
18-D, mine: correcting Task 14 to emit a real JPEG quality curve replaced the
single "jpeg" sweep key with jpeg_q90..q10, so Task 18's
`set(PERTURBATIONS).issubset(set(got))` became FALSE — PERTURBATIONS keeps a
bare "jpeg" the sweep no longer emits. A correction in one task silently broke
a downstream task's assertion, which is exactly what the cross-task scan at the
start of this session was supposed to catch and did not, because I made the
change after the scan and never re-ran it.

LESSON, recorded because it will recur: correcting a task's INTERFACE obliges a
re-scan of every task downstream of it. I have now corrected Tasks 13, 14, 15,
16, 17, 18, 20 and 21; the only reason 18-D surfaced is that I happened to read
Task 18 before dispatching it.

PROCESS BUG, mine: my splices introduced duplicate `### Task` headers for 17
and 18, because `sed -n '/^### Task N/,/^### Task N+1/p'` includes the trailing
header and I inserted that before the real one. Fixed both; the splice now
strips trailing headers and asserts uniqueness afterwards. Worth noting that
this was invisible to every test — only a structural check on the plan found it.

Verified Task 18 by applying its described edits to the already-verified Task 17
runner and running the whole bench suite: 35 passed (29 + 6), and the
robustness surface comes back finite in every cell:
  clean 0.000 | blur 0.067 | jpeg_q10 0.133 | jpeg_q30 0.067 | jpeg_q50 0.133
  jpeg_q70 0.000 | jpeg_q90 0.000 | noise 0.000 | print_recapture 0.200
  resize 0.067 | screenshot_recapture 0.000
(The synthetic detector is a payload hash, so these are near-random by design —
what is being verified is that the pipeline produces real numbers, not that the
stand-in detector is any good.)

Re-scan of Tasks 19 and 22 against every interface I have changed this session
(the discipline Task 18 taught me). Task 19: no references to any changed
interface — clean. Task 22: two real problems.

Finding 22-A (Important): Task 22's mypy gate FAILS ON DAY ONE, and its own
tests cannot tell. Ran mypy under the exact config the task ships
(strict = True, warn_unreachable = True, files = src/dfd): 22 errors in 7 files.
  19x Missing type parameters for generic type "ndarray"  [type-arg]
   2x Unused "type: ignore" comment                        [unused-ignore]
   1x Returning Any from a function declared to return ndarray [no-any-return]
  faces.py 5 | types.py 4 | quality.py 3 | detectors/npr.py 3 |
  calibration.py 3 | fusion.py 2 | detectors/effnet.py 2

The task's own test is `assert "strict = True" in (ROOT/"mypy.ini").read_text()`
— it asserts the CONFIG FILE CONTAINS A STRING, not that mypy passes. So Task 22
would go green locally while CI goes red on the first push. That is a new
variant of the tracked class worth naming: a gate test that asserts the gate is
CONFIGURED rather than that it PASSES. The same shape applies to the ruff test
(`"E722" in cfg`) and the asset-gate test (`"assert_all_assets_registered" in ci`).

Note this also narrows my earlier recorded gap: `files = src/dfd` means bench/
and corpora/ are NOT type-checked, so bench/metrics.py's and robustness.py's
bare-ndarray errors do not block the gate. Only src/dfd's 22 do. That is a
smaller job than I first recorded — parameterising ~19 annotations across 7
files — but it is real work that Task 22's brief does not mention.

Ruling: Task 22 must (a) fix the 22 errors so the gate passes, and (b) replace
its config-text assertions with tests that RUN the gates and assert they exit
zero. A gate asserted by the presence of a string in a config file is exactly
the "enforced by intention" failure the task's own rationale warns about.
— Cost if wrong: Task 22 is larger than planned; the alternative is shipping a
CI file that has never been observed to pass.

Finding 22-B (Minor): requirements-dev.txt IS created by the task (its Step 3
prose, line 151) though it is missing from the task's Files header. Its list —
pytest, pytest-cov, ruff, mypy, numpy, opencv-python-headless, scikit-learn,
pyyaml, torch — now needs Pillow, which Task 21's header-probe defence depends
on. pyproject.toml still declares no [project.dependencies] at all; that is
separate and still open.
Task 16: fix round 1/5 (2 addressed, 1 NEW Important in the fix diff; commits
9af0af4..47ad28a)

Ruling: the new finding enters fix round 2 rather than being parked. The
widened `except` now wraps RECORD CONSTRUCTION, not just parsing and shape
validation, and TypeError/ValueError/KeyError/AttributeError are precisely what
a coding mistake in that construction would raise — a typo'd keyword, a missing
field, `m["nam"]` for `m["name"]`. Such a bug fires on every entry and is
swallowed identically to malformed input. It is materially worse in
rd_cache.py, which logs NOTHING on skip: a construction bug there would silently
drop all 24 cached results with zero signal, where before this fix it would have
crashed loudly. That is a REGRESSION in failure visibility introduced by a fix
for failure visibility.

This also runs straight into the plan's own Global Constraint — "a swallowed
error in a fraud detector is an approved fraud" — which Task 22 encodes as a
ruff ban on blind excepts. Decision: narrow the try to parse-and-validate, and
construct the record outside it, so malformed input is skipped while a defect in
our own code still raises. And give rd_cache the same per-skip logging captures
has; its silence is what makes the same latent bug invisible there and merely
misreported here. — Cost if wrong: two more lines of structure in each loader.

The reviewer was right to call this latent rather than live (it inspected both
constructors and found no current typo), and right to raise it anyway: the
property wanted is structural, not "true by inspection today".
Task 16: fix round 2/5 dispatched — FIX_BASE 47ad28a.

Pre-flight scan of Task 19 (asset enumeration — "make the release gate
non-vacuous").

Finding 19-A (CRITICAL): TASK 19 REPRODUCES, ONE LEVEL UP, THE EXACT VACUITY IT
EXISTS TO ELIMINATE — and it is worst in CI, the only place it runs
automatically.

Its own rationale: "`assert_release_clean` can only judge assets it is handed.
Passing it an empty list returns cleanly — a vacuous pass." Its fix supplies the
list from the filesystem. But `assert_all_assets_registered` is

    assert_release_clean(load_manifest(manifest_path), discover_assets(root))

and `assert_release_clean` iterates `asset_ids`, so an EMPTY discovery still
returns cleanly. The vacuity moved from "a human forgot to pass the list" to
"the scan found nothing", which is strictly harder to notice.

And on a fresh checkout the scan finds nothing, always. Verified: of everything
under assets/, git tracks exactly ONE file — assets/manifest.yaml. The weights
are untracked, and .gitignore carries *.onnx, *.pth and models/. So in CI:

    discover_assets(root)                  -> []
    assert_release_clean(manifest, [])     -> no bad ids, returns cleanly
    the gate                               -> PASSES, unconditionally, forever

Task 22 wires this into CI as the enforcement of spec 12.1 criterion 6. As
written, that criterion would be certified green by a check that has never
examined a single file.

Ruling: an empty scan must be a loud, explicit condition, not a pass.
`assert_all_assets_registered` gains `allow_empty: bool = False` and raises when
it discovers nothing, with a message naming the root it searched and the
suffixes it looked for. CI must then either provide the assets or opt into
emptiness deliberately, which is a decision someone makes rather than a silence
nobody notices. Its test must assert the raise, not merely that
`discover_assets` returns [] — the current
test_empty_tree_is_not_treated_as_success_by_accident checks the scanner and
never calls the gate, so it tests the one function that was never the problem.
— Cost if wrong: CI needs an explicit flag, and someone has to decide what the
gate means in an environment with no weights. That decision is the point.

Finding 19-B (Important): `assert_release_clean` reports an UNREGISTERED asset
through `NonCommercialAsset` with the message "assets not cleared for commercial
release". Those are different failures — "I have never heard of this file" is not
"this file's licence forbids commercial use" — and the message actively
misdescribes the first. A reader debugging a red gate is told a licensing story
about a file whose only sin is being absent from the manifest. Decision: keep
one exception type if the hierarchy is not worth expanding, but the message must
distinguish the two lists.

Finding 19-C (Minor): `pytest.raises(NonCommercialAsset) as exc` has no
`match=`, though the following line asserts "sneaky_weights" in the message, so
it is not vacuous — just less direct than `match=` would be.

Finding 19-D (Minor): ASSET_SUFFIXES covers .onnx .pt .pth .safetensors .tflite
.bin .npz, but .gitignore lists only *.onnx and *.pth (plus models/). A
committed .safetensors or .bin would be discovered by the scan and is not
ignored by git — worth a note, since the dima806 weights on this machine are
.safetensors and sit untracked only because assets/models/dima806/ was never
added.
Task 16: fix round 2/5 (1 addressed, 0 open; commits 47ad28a..b9f1136)
Task 16: complete (commits e67422f..b9f1136, review clean, 339 total green)

Re-review traced EVERY untrusted field access to its side of the try boundary
by name in both loaders, rather than accepting "narrowed" as a claim, and
confirmed none was left split across it. It also checked the injected-defect
proof was genuinely construction-side rather than validation-side (both
injections add an unexpected keyword to the dataclass call AFTER the except
block), and noted the rd_cache trace is discriminating: the malformed entry
still logs a skip while the injected defect on the GOOD entry propagates
uncaught. That is the structural property, demonstrated rather than asserted.

Task 16: minor DEFERRED — rd_cache's except tuple omits AttributeError while
captures' includes it. Pre-existing asymmetry, not introduced by either fix
round; likely unreachable given the isinstance(m, dict) guard, but the
asymmetry itself is worth removing for the reader.
Task 19: correction committed 7a32698. Verified 13 passed; Step 5 proof
confirmed — restoring the plan's original one-line body fails exactly the two
empty-scan tests.
Task 17: dispatched (sonnet). BASE 7a32698.
Task 22: correction committed 50f2ef5. Every remaining task section (13-22) has
now been corrected and verified before dispatch.

Ruling on 22-A: the gate tests now RUN ruff and mypy as subprocesses and assert
exit zero, and the task clears both gates as part of its own work. Measured
before deciding to run them in-suite: ruff 0.04s, mypy ~1s warm / ~41s cold.
Turning a gate on without clearing it ships a red pipeline; clearing it without
turning it on ships a standard nobody enforces. — Cost if wrong: the suite gains
~1s warm, and Task 22 is meaningfully larger than its brief implied.

Correcting my own earlier count: the ruff backlog is 11 F541 findings, not 8 —
8 in bench/guards.py plus 3 in src/dfd/detectors/loading.py, which I had not
looked at when I first recorded it. All auto-fixable.

Ruling on 22-B: requirements-dev.txt gains Pillow (Task 21's decode-bomb defence
imports it, so CI would have installed a tree that cannot import dfd.limits),
and pyproject.toml gains [project.dependencies], which it had never had at all
despite the package importing numpy, opencv, Pillow and torch. torch and
scikit-learn stay OUT of the runtime set deliberately: the detectors needing
them abstain cleanly when absent, so a caller wanting only the NPR physics
detector and the evidence core should not be forced to install a GPU stack.
They stay in requirements-dev.txt, which is what CI installs. — Cost if wrong:
a caller who wants the torch detectors must install torch explicitly, which the
abstention message should say.
Task 17: implemented 8c339f9 (29 tests, 368 total green).
Task 17: review found 0 Critical, 4 Important, 14 Minor. ALL FOUR Importants are
defects in my corrected section, not the implementer's execution.

The reviewer verified the computation before judging it: traced that fold
slicing aligns (position maps sample_id to its index in the same list
observations was built from), confirmed abstentions are filtered before metrics
with groups sliced in lockstep, and ran an all-abstaining detector end to end to
confirm the degenerate path prints n/a rather than a flattering number.

Ruling: finding 2 is the most important and enters the fix round first. NO TEST
CAN FAIL IF `groups` REGRESSES from source_id to sample_id — the single wiring
decision I corrected this task for, and the one with a measured cost (11.9x too
narrow). The fixture gives every record its own source_id, so the two groupings
are the same partition. Reviewer confirmed empirically: bootstrap_ci_by_group
with groups=source_id and groups=sample_id returns BIT-IDENTICAL bounds
(0.20697727272727273, 0.5342249855407751), and all 21 runner tests stay green
under the regression. check_video_level does not cover it — different arguments,
and skipped entirely under enforce_guards=False, which is the path the runner's
own comment calls out as where an honest interval matters most. I fixed the
wiring and left it unguarded, which is the same shape of error as the guards
this plan keeps finding. — Cost if wrong: one test needs a multi-sample-source
corpus, necessarily with guards off.

Ruling: finding 1 enters. `dataset_hash` omits `face_detector` and `align`,
though check_uniform_preprocessing treats varying them as grounds to ABORT the
run and they sit on the same record dicts. Two runs over one corpus aligned
yunet/v1 vs retinaface/v2 hash identically and produce different metrics,
breaking the audit tie acceptance criterion 10 exists to create. Worse, my
test_dataset_hash_is_sensitive_to_every_identifying_field asserts a COMPLETENESS
property while enumerating exactly the six fields the implementation already
covers — a test that cannot discover the omission it is named for. The reviewer
also checked whether a per-sample content hash exists to fold in and found none,
so pixel content is genuinely unavailable — correctly not raised.

Ruling: finding 3 enters. LOGO folds pass `latencies=[]` and `abstentions=0`
literally, so every fold reports abstention_rate=0.0 and p95_latency_ms=0.0
regardless of what happened. Verified: with an all-abstaining detector both
folds returned auc=nan alongside abst_rate=0.0, which is false for every row.
"0% abstained" beside an unmeasurable AUC invites precisely the wrong reading.
Decision: compute fold abstentions from the slice, and report nan for a latency
not measured per fold so the unmeasured field is visibly unmeasured.

Ruling: finding 4 enters. Per-fold `dropped_for_identity` is discarded. Task 13
surfaces it specifically so identity-conflicted fakes are "dropped and reported
rather than silently leaked", and the runner reads only test_ids(). The fixture
drops nothing, so no test notices — but on a real corpus where a subject is
faked by several generators, drops are routine (measured at 6-8 of 12 on Task
13's own fixture), and the headline LOGO AUC would be computed over a silently
reduced test set. A reader sees n=23 and cannot tell whether 2 or 20 records
were removed. This is the SAME defect I flagged at the start of this session —
a number that must be reported alongside LOGO or the fold gets over-read — and
I then failed to wire it through.

Promoted from Minor into the fix round, because each is one line and each is
fail-closed posture the module already claims: report.py raises KeyError where
worst_logo_auc defends against the identical case (a detector absent from a
fold renders a bold worst-AUC then crashes building the same row); duplicate
sample_ids silently collapse in the `position` dict so a fold scores one row
twice and omits another with no error; and no report test asserts the seed's
VALUE, so a renderer emitting seed 0 ships green against criterion 10's own
artifact.
Task 17: fix round 1/5 dispatched — FIX_BASE 8c339f9.
Task 17: fix round 1/5 (7 addressed, 0 open; commits 8c339f9..99feddd)
Task 17: complete (commits 7a32698..99feddd, review clean, 39 runner/report
tests, 378 total green)

The re-review did the thing that matters most: it separated "evidence offered"
from "test can actually fail". Findings 1, 5 and part of 4 carried genuine
revert-and-rerun RED output; findings 2, 3, 4-runner and 6 were backed only by
write-then-implement, which is weaker for a fix to EXISTING wrong behaviour. So
it checked each remaining assertion analytically against the pre-fix `-` lines
in the diff and established each is unsatisfiable against the old code — the
pre-fix key tuple omitted the mutated field, pre-fix `_detector_result(..., [], 0)`
yields 0.0, pre-fix RunRecord had no logo_dropped attribute at all, pre-fix
there was no raise to catch. Every test can fail, even where the offered
evidence did not itself show it.

It also independently reproduced Finding 1's numbers rather than trusting them:
source-grouped CI (0.22, 0.5801875) width 0.360 vs row-grouped
(0.20697727272727273, 0.5342249855407751) width 0.327, with the runner's
reported CI equal to the source value. And it verified the new test is
self-proving — asserting BOTH `== expected_by_source` and `!= expected_by_row`,
so it cannot pass on a fixture where the two coincide. That is the guard I had
left untestable, now testable and proven.

Task 17: minors DEFERRED to final review — report.py renders "dropped 0" via
`logo_dropped.get(g, 0)` for hand-built records (same fabricated-zero species
as finding 3, unreachable from run_benchmark, which always populates both);
the duplicate-sample_id check sits inside _logo_results so a duplicated corpus
pays for all detector scoring before raising; the Finding-1 paired-source
fixture makes logo_splits refuse the corpus, so the paired-source path has no
LOGO coverage; plus the seven deferred from round 1 (renderer-side demotion
floor, worst_logo_auc on an unmeasurable fold, latency magnitude bound, ece
range abort, unexercised identity_report branch, one-directional
guards_enforced assertion, "**0.510**" pinning).
Task 18: dispatched (sonnet). BASE 1868322.
Task 18: implemented d213299 (6 new tests, 384 total green).

Implementer raised a real concern: the sweep reuses the CLEAN observation's
`quality` object rather than re-measuring quality on the perturbed pixels, so a
perturbation that degraded quality below a detector's floor would not produce
the abstention it should. I measured the effect before ruling, across 11 sweep
variants at three fixture sizes:

  128x160 structured   every variant bands "low", same as clean
  240x320 sharp        every variant bands "high", same as clean
  360x480 sharp        every variant bands "high", same as clean

So the current behaviour is exactly equivalent — no fix round needed, and the
reuse should be documented as deliberate with the condition under which it would
start to matter.

But the measurement surfaces something larger, and it is NOT Task 18's to fix:
QUALITY BANDING IS BLIND TO EVERY PERTURBATION IN THE ROBUSTNESS SURFACE.
Blur halves high-frequency energy (0.13x, measured in Task 14) and bands
identically. Screenshot re-capture destroys ~64% of it and bands identically.
Print re-capture likewise. So the quality floor offers NO protection against the
two cheapest laundering steps in the threat model, and the abstention mechanism
will never route a recaptured sample to manual review on quality grounds.

That is a finding about src/dfd/quality.py (Task 3, long complete), not about
the task in flight. It matters because spec 3A's posture assumes low-quality or
degraded input is caught and routed, and here it simply is not. Recorded for the
final review and the handoff rather than acted on mid-task — changing the
quality thresholds now would invalidate every abstention-related test across
eight completed tasks.
Task 18: review found 0 Critical, 2 Important — both untested wiring, both
consequences of a brief (mine) that asked only for runner tests.

The reviewer confirmed the wiring itself is right before finding the gaps: ROI
reuse is valid because every perturbation returns an array of the same (h,w) as
its input (it checked _resize downscales then upscales back), Observation reuse
replaces only the payload across all five fields, LOGO folds correctly default
to {} rather than fabricating an empty-as-measured surface, and the source_id
bootstrap contract added last round is undisturbed.

Ruling: finding 1 enters the fix round. LOGO folds correctly receive {} — and
NOTHING ASSERTS IT. That is the identical pattern to Task 17's finding 1 one
round earlier: a value wired correctly with no test able to detect it
regressing. Having just paid a fix round for exactly this in the same file, it
would be absurd to park it. One line.

Ruling: finding 2 enters. The report's robustness table — the any_rob gate,
pooled column names, per-cell lookup — ships with zero automated coverage, and
the implementer's manual render is not a regression guard. The rendered report
is the artifact acceptance criterion 9 is actually about; "the brief only asked
for runner tests" is a defect in my brief, not a defence.

Promoted from Minor: the clean-quality reuse gets a comment at the call site. I
measured it as behaviourally equivalent across 11 variants at three sizes, but
that context lives in a report nobody will read again, and a future reader
cannot tell a considered simplification from an oversight.

Task 18: minors DEFERRED — robustness_sweep is recomputed once per detector
though perturbed images are detector-independent, O(detectors) redundant work
(plan-mandated by my reference implementation; worth optimising once the
registry grows); and test_the_whole_jpeg_quality_curve_is_measured is subsumed
by the exact-set check two tests earlier, harmless duplication with no
independent failure mode.
Task 18: fix round 1/5 dispatched — FIX_BASE d213299.
Task 18: fix round 1/5 (3 addressed, 0 open; commits d213299..dbf0057)
Task 18: complete (commits 1868322..dbf0057, review clean, 387 total green)

Re-review checked the two things most likely to be wrong in a fix of this shape.
The closed-gate test asserts ABSENCE, which is only meaningful if the string it
keys on is unique to the thing being gated — it verified "Robustness" appears
exactly once in render_markdown and "screenshot_recapture" only inside the gated
block, so absence is a genuine proxy rather than a coincidental miss. And the
LOGO assertion asserts rec.logo_results is truthy BEFORE looping, so it cannot
pass vacuously over an empty dict — the failure mode that has now appeared three
times in this plan. It also confirmed the populated-case test slices the report
from the "## Robustness" heading before matching table rows, avoiding the
in-dataset table's identical-looking header.
Task 19: dispatched (sonnet). BASE dbf0057.
Task 19: implemented 0d158ac (9 tests, 396 total green).
Task 19: review found 1 CRITICAL, 1 Important. The Critical is mine, and running
it against the real repo showed it is worse than the reviewer could see from the
diff alone.

Reviewer's finding: discover_assets returns filename STEMS via a set, so
model.onnx and model.pt collapse to one id — if the survivor is registered and
cleared, the other file ships invisible to the gate.

What I found running it against this repo's actual assets:

  discover_assets('.')  -> ['face_detection_yunet_2023mar', 'model']
  manifest ids          -> ['ffpp_effnetb4_weights', 'npr_weights',
                            'sbi_effnetb4_weights', 'yunet_face_detector']
  the gate              -> NonCommercialAsset: unregistered assets (absent from
                           the manifest): face_detection_yunet_2023mar, model

THE STEMS DO NOT MATCH THE MANIFEST IDS AT ALL. The YuNet detector IS registered
— as `yunet_face_detector` — and the gate calls it unregistered, because the id
is derived from the filename. And the dima806 ViT ships as `model.safetensors`,
whose stem is `model`: the reviewer's collision case, already present in the
repo, on the most collision-prone name there is.

So the gate does not merely have a bypass. It does not work on this repository's
real assets in either direction: it fails a registered asset and would pass an
unregistered one that shared a stem with a cleared file.

Ruling: the manifest must declare WHICH FILES each id covers, and the scan must
match by PATH, not by stem. AssetRecord gains `files: tuple[str, ...]` of
repo-relative paths; discover_assets returns repo-relative paths;
assert_all_assets_registered requires every discovered path to be claimed by some
manifest entry, and that entry's commercial_use decides clearance. This is the
only design in which "a manifest covering every weight file in use" (spec 12.1
criterion 6) is a checkable statement rather than a naming coincidence — and
vendor-named files like `model.safetensors` make the coincidence impossible to
rely on. — Cost if wrong: Task 2's manifest schema gains a field and
assets/manifest.yaml must name real paths, which is work someone has to do once
and which is itself the criterion being claimed.

Ruling: the Important (rglob follows symlinks, admitting mislabeled ids or a
cycle hang) enters too. Cheap, and this module's whole posture is fail-closed.

Resolved the reviewer's warning myself: Task 22's CI step calls
`assert_all_assets_registered('.', 'assets/manifest.yaml')` and nothing calls
`assert_release_clean(manifest, discover_assets(root))` directly, so the
original vacuity is not reachable from CI. Confirmed in the Task 22 brief.
Task 19: fix round 1/5 dispatched — FIX_BASE 0d158ac.

CROSS-TASK ITEM to settle after Task 20 lands: Task 19 now has TWO bare exceptions — `AssetScanEmpty` and `DuplicateAssetClaim`. Original note follows.
CROSS-TASK ITEM to settle after Task 20 lands: Task 19's `AssetScanEmpty` is
currently a bare `Exception` with a docstring saying it joins the `DfdError`
hierarchy once Task 20 introduces it. Task 20 creates src/dfd/errors.py with
DfdError / InvalidInput / ResourceLimitExceeded. So after Task 20 is complete,
AssetScanEmpty must be rebased onto DfdError — otherwise the plan ships one
exception that sits outside the "one catchable root for everything this package
raises" guarantee that errors.py exists to provide, and a caller doing
`except DfdError` misses it. Task 21 already consumes errors.py, so the ordering
works; this is the one loose thread. I will carry it into Task 20's dispatch
rather than leaving it to the final review.
Task 19: fix round 1/5 (2 addressed, 1 NEW CRITICAL in the fix diff; commits
0d158ac..fbd5c88)

The re-review answered exactly the question I sent it to answer. I had verified
the POSITIVE path myself (both real assets discovered, claimed, cleared, gate
passes) and told it not to re-derive that, because for a gate the negative path
is the half that matters. It found the negative path broken in a new way.

Ruling: the new Critical enters fix round 2. `_resolve_ids` builds its
path->asset_id map by iterating manifest entries and overwriting, so a path
claimed by TWO entries resolves to whichever was declared LAST in the YAML.
Verified on the shipped code with a file claimed by both a research-only entry
and a cleared one:

  uncleared declared FIRST, cleared second  -> GATE PASSES
  cleared declared FIRST, uncleared second  -> NonCommercialAsset

So the gate's verdict on a research-only weight file depends on the order two
entries happen to appear in a YAML file. That is precisely the
"passes when it should fail" licensing exposure this task exists to prevent, and
it is NEW — the old stem-based scheme had no notion of claims to collide, so the
design change introduced it. No test covered two entries claiming one path.

Decision: a path claimed by more than one entry is a MANIFEST ERROR and must
raise, naming the path and the competing ids — not resolved by picking the most
restrictive entry. Ambiguous provenance is not something to silently reconcile:
if two entries claim one file, a human recorded the licence twice and at least
one record is wrong, and that is worth knowing before a release rather than
after. — Cost if wrong: a manifest with a deliberate duplicate claim must be
deduplicated before the gate will run.
Task 19: fix round 2/5 dispatched — FIX_BASE fbd5c88.
Task 19: fix round 2/5 (1 addressed, 0 open; commits fbd5c88..05a25ac)
Task 19: complete (commits dbf0057..05a25ac, review clean, 17 asset_scan tests,
404 total green)

Re-review settled the identical-clearance question STRUCTURALLY rather than by
test: there is no commercial_use comparison anywhere in _build_claims, so the
unconditional len(ids) > 1 check cannot depend on whether the two entries agree.
It also confirmed the new regression test goes through the public
assert_all_assets_registered rather than reimplementing _build_claims — a test
that reimplements the logic it guards agrees with a buggy implementation.

Task 19: minors DEFERRED — the AssetScanEmpty check runs BEFORE the manifest is
loaded, so a manifest with duplicate claims AND an empty scan reports only the
emptiness. Both refuse, so it is not a licensing hole, but the message would not
reveal that the manifest is also malformed, and a reader would fix the wrong
thing first. Pre-existing ordering from round 1, not introduced by round 2. Also
DuplicateAssetClaim lacks the docstring note about joining DfdError that
AssetScanEmpty carries.
Task 20: dispatched (sonnet). BASE 05a25ac. Carries the DfdError rebase for BOTH
of Task 19's bare exceptions, so the plan does not ship exceptions outside the
"one catchable root" guarantee errors.py exists to provide.
Task 20: first dispatch died on an API rate limit, not a code failure. Verified
the working tree was clean, HEAD still at 05a25ac, and none of errors.py,
audit.py, tests/test_audit.py or the report file existed — the agent stopped
before writing anything. Nothing to recover or unwind. Re-dispatched unchanged.
Task 20: implemented b5b87d9 (25 audit tests + 2 hierarchy tests, 431 total).
Task 20: review found 0 Critical, 4 Important, 11 Minor. Every Important was
verified live by the reviewer rather than argued.

Ruling: finding 1 enters. The evidence rows' freezing is UNTESTED — the
reviewer built a one-token mutant (`tuple(dict({...}))` instead of
MappingProxyType) and ran the file's own 25 tests against it: 25 passed. Under
that mutant `record.evidence[0]["llr"] = 9.9` raises nothing and changes the
digest. That is brief defect #1 verbatim — "a record can be altered after the
fact and re-digested to match" — guarded for model_versions and left unguarded
for the per-detector LLRs, which are the rows a regulator would most want
tamper-evident.

Ruling: finding 2 enters, and it is MY error. I instructed the rebase for the
two exceptions in asset_scan.py and never checked whether others existed.
NonCommercialAsset in manifest.py is raised out of the SAME call —
assert_all_assets_registered — and is still a bare Exception. Verified:

  AssetScanEmpty         DfdError subclass: True
  DuplicateAssetClaim    DfdError subclass: True
  NonCommercialAsset     DfdError subclass: False

So `except DfdError` around the asset gate catches the two housekeeping
failures and misses the one that fires on an actual licensing violation. The
new hierarchy test parametrises over exactly the two classes I named, so my
own instruction defined the blind spot the test then inherited.

Ruling: finding 3 enters. `_freeze` is applied to ONE of thirteen fields, so
immutability rests on the others happening to hold scalars. Reviewer verified
live: `quality_band=["high"]` then `.append("TAMPERED")` succeeds and changes
the digest; `sample_id` only has to be truthy, which a list is. Plan-mandated —
my implementation is identical.

Ruling: finding 4 enters. `json.dumps` defaults to `allow_nan=True`, so a NaN
llr_total or infinite posterior is silently emitted as `NaN` / `Infinity` —
tokens no conforming JSON parser accepts, in an artifact whose whole purpose is
to be re-read by someone else's tooling. Same failure DIRECTION the brief
rejects for `default=str`, and NaN metrics are a documented occurrence in this
codebase (every abstaining detector produces them).

Promoted from Minor: (5) the PII refusal is deferred to serialisation, so
`build_audit_record` CONSTRUCTS SUCCESSFULLY holding 4008 raw image bytes and
only refuses at to_json — a record object carrying PII exists, and anything
that logs or reprs it before serialising leaks it. That is the wrong boundary
for a DPDP-Act-relevant discipline. (6) `created_at` gets no format validation
at all while `input_sha256` gets a regex, and `created_at=""` silently becomes
now() — the field the brief itself calls "among the most attack-relevant there
is".

Ruling on the reviewer's warning: 19 bare `raise ValueError`/`RuntimeError`
sites remain across fusion, calibration, detectors and ingest, so
"one catchable root for every error this package raises" is FALSE TODAY
regardless of the three exception classes. Migrating them touches eight
completed tasks and risks their tests for what is, today, a documentation
claim. Decision: make the errors.py docstring accurate now rather than ship a
false guarantee, and record the migration as a P0-closing gap — the same
posture taken with the quality.py banding finding. — Cost if wrong: a caller
must catch ValueError alongside DfdError until the migration lands, and the
docstring will say so.
Task 20: fix round 1/5 dispatched — FIX_BASE b5b87d9.
Task 20: fix round 1/5 (6 addressed, 1 partial; commits b5b87d9..9d8254b)

Six of seven closed cleanly and the happy path was verified unharmed — the
re-reviewer confirmed raw_score=None, zero and negative numeric values all
still build, so the new validation does not over-reject. Finding 4 was checked
specifically for silent substitution and is a genuine refusal, not a coercion
to null or 0.0.

Ruling: finding 5 is PARTIAL and enters round 2. `_validate_serialisable` is
called on `model_versions` and never on `evidence`. Reproduced myself on the
shipped code, via the field the fix did not touch:

  Evidence(..., reason=b"\x89PNG" + 4000 nulls)
  build_audit_record(...)  -> SUCCEEDS, record holds 4004 raw bytes
  r.to_json()              -> TypeError, only now

That is the identical build-succeeds/serialise-fails defect the finding was
written against, relocated one field sideways. `Evidence` is a plain dataclass
with no runtime type enforcement, and the row-building loop copies `detector`,
`detector_version` and `reason` straight into the row unvalidated.

Worth naming as a pattern, because it is the third time this session: a fix
that is correct on the case it was SHOWN and blind to the same defect one level
down. Task 16's except-narrowing fixed parsing and left construction; Task 19's
path-matching fixed stems and introduced duplicate claims; now a leaf-value
check fixed one field and skipped its sibling. I have started asking reviewers
about partial fixes explicitly, which is what caught this one.
Task 20: fix round 2/5 dispatched — FIX_BASE 9d8254b.
Task 20: fix round 2/5 (1 addressed, 0 open; commits 9d8254b..0dd8bcf)
Task 20: complete (commits 05a25ac..0dd8bcf, review clean, 27 audit tests,
442 total green)

Re-review checked the over-rejection risk specifically, which is the failure
mode of a round that adds validation: confirmed raw_score=None is exempted,
that `uncertainty` and `artifacts` are read by neither the validator nor the
row builder so a value there cannot reject an otherwise-valid Evidence, and
that the only production construction site (calibration.py) always passes types
matching RawScore's own fields — so no legitimate evidence row can trip it. It
also spot-checked the implementer's sweep rather than accepting the summary.

`abstained` is deliberately left unvalidated and the reviewer accepted the
reasoning: `bool(e.abstained)` never raises and never preserves original
content, the same shape as `verdict`'s `str(...)` coercion. That is a real
distinction — coercion that cannot carry a payload through needs no guard.

Task 20: minor DEFERRED — build_audit_record now iterates `evidence` twice
(validation pass, then row building), so a single-use iterator would validate
clean and yield no rows. Not a live risk: the parameter is typed
Sequence[Evidence], which contractually requires re-iterability, and there are
no non-test call sites yet. Worth a note if that ever loosens.
Task 21: dispatched (sonnet). BASE 0dd8bcf.
Task 21: implemented 784535e (25 tests, 467 total green).
Task 21: review found 0 Critical, 4 Important. The image path is sound — I
verified a real 12000x12000 bomb (161,331 bytes on disk, 0.40 GB decoded) is
refused in 9ms with cv2.imread never called. The VIDEO path is where it falls
down, which is exactly why I sent the reviewer at the paths I had not
exercised.

Ruling: finding 1 enters. The video decode loop has NO dimension gate at all,
and the report justified the omission by claiming OpenCV cannot expose frame
dimensions without decoding. That claim is FALSE and both the reviewer and I
tested it:

  cap = cv2.VideoCapture(path)   # no read() yet
  CAP_PROP_FRAME_WIDTH  = 320.0
  CAP_PROP_FRAME_HEIGHT = 240.0

They come from the container header at open time, exactly like
CAP_PROP_FRAME_COUNT which this same function already trusts on the next line.
So the header-before-decode discipline the whole task rests on IS available for
video and was skipped on a false premise. A container declaring 20000x20000
frames commits ~1.2 GB per cap.read() with nothing checking.

Ruling: finding 2 enters. Neither video guard has a test — deleting both
check_file_size and the max_frames clamp from load_video leaves the entire
suite green. The Step 5 wiring proof was performed for load_image only. This is
verbatim "a guard wired correctly that no test could detect regressing", which
the brief names as this task's specific hazard.

Ruling: finding 3 enters. Nothing can detect probe_image_dims regressing into
an actual decode: the no-decode test monkeypatches cv2.imread, but
probe_image_dims uses PILLOW. An implementation calling im.load() inside the
with-block would pass every test while allocating 144 MB on the bomb fixture —
too small to trip a timeout, too invisible to fail anything. That is the
load-bearing property of the module, unguarded.

Ruling: finding 4 enters, and it is the one with a false claim attached.
`max_duration_s` is pinned by a test and read by NO code path — verified, it
appears once in src/, at its own definition. Worse, the decode loop is
`while True: cap.read()` with no break on max_frames, so the clamp bounds
RETAINED OBSERVATIONS, not decodes. The added docstring says the clamp means
"a caller cannot reintroduce unbounded frame extraction". It cannot do that.
256 MB of well-compressed H.264 is hours of footage and millions of decodes.
A security control's documentation asserting a bound the code does not provide
is worse than silence.
Task 21: fix round 1/5 dispatched — FIX_BASE 784535e.
Task 21: fix round 1 verified by me before re-review — dimension gate refuses
(ResourceLimitExceeded, 320x240 vs cap 1000), duration gate refuses
("declares 6.0s, exceeds limit 1.0s"), loop bounded, normal video still loads.
475 passing.

NUANCE worth carrying: for max_frames=3 on a 60-frame video the loop made 56
cap.read() calls, not 3. Deterministic spread sampling cannot reach frame 55
without decoding to it, so the early break only helps when wanted frames
cluster early. On a long video the DURATION GATE is what actually bounds decode
work — it is load-bearing, not belt-and-braces. Correct behaviour, but it means
raising max_duration_s without re-examining the loop would reopen the exposure.
Task 21: fix round 1/5 (4 addressed, 1 NEW Important in the fix; 784535e..8970f95)

Ruling: the new finding enters round 2. `fps = cap.get(CAP_PROP_FPS) or DEFAULT_FPS`
substitutes only when fps is exactly 0.0. Negative and NaN fps are both TRUTHY,
so both survive the `or` — negative gives a negative duration and NaN gives NaN,
and `negative > limit` and `nan > limit` are both False, so the gate is silently
SKIPPED rather than raised or crashed. fps is adversary-controlled container
header metadata, the same class of value this whole task treats as untrusted.
Every duration test uses fps=30.0, so nothing covers it. Not a full regression —
the dimension, file-size and max_frames gates still hold — but it is a live
bypass of the specific control finding 4 asked for.

Re-review also confirmed the "collateral mock fix" was legitimate: a codec that
fails to report frame count while still reporting real dimensions is the case
the test's own docstring describes, and its actual assertions are unchanged.
Task 21: fix round 2/5 dispatched — FIX_BASE 8970f95.
Task 21: fix round 2/5 (0 addressed, 1 still open; commits 8970f95..aeb29fa)

Ruling: the fps domain check is INCOMPLETE and goes to round 3. The fix reduced
"finite and greater than zero" to just "greater than zero". `float('inf') > 0`
is True, so an infinite fps survives the guard, `total / inf` collapses to 0.0,
and the gate is silently skipped — the same failure mode as the negative and NaN
cases it just closed, reached through a third adversary-controlled value.

Negative and NaN ARE fixed and the mutation proof for them is sound. But the
report's justification never mentions infinity, which tells me the gap was not
considered rather than considered and accepted. Tests cover 100.0, -30.0 and
nan; neither inf nor a tiny positive like 1e-9 is pinned. (1e-9 is handled
correctly by the current code — finite, positive, duration blows up, gate fires
— so infinity is the only broken edge.)

Worth recording as the lesson rather than just the bug: I asked for "finite and
greater than zero" and got half of it, and the half that was dropped is the half
that only matters against an adversary. This is the fourth partial fix in this
plan, and the third where the missing half was the security-relevant one.
Task 21: fix round 3/5 dispatched — FIX_BASE aeb29fa.
Task 21: fix round 3/5 (1 addressed, 0 open; commits aeb29fa..59a4029)
Task 21: complete (commits 0dd8bcf..59a4029, review clean, 480 total green)

Verified all seven fps cases myself: -30.0, nan, inf, -inf and 0.0 all fall back
to DEFAULT_FPS (2.4s), 1e-9 yields 6e10s, and an honest 10.0 yields its true
6.0s — so the fallback is pinned as not-always-taken. Re-review confirmed the
test parametrisation pins the same set, and that reverting to bare
`raw_fps > 0` fails ONLY the inf case, which is the correct isolation signature.

Task 21: minor DEFERRED, found by spot-checking the implementer's sweep claim:
`int(cap.get(...))` on CAP_PROP_FRAME_WIDTH/HEIGHT/FRAME_COUNT would raise
untyped ValueError/OverflowError if OpenCV ever reported nan/inf for those
props, rather than a typed DfdError. Pre-existing since round 1, not touched by
any fix. "Immune by construction" is right about the failure MODE — no silent
wrong answer is possible there — but it trades silent corruption for an untyped
crash out of a fail-closed control, which is not the same as immune.
Task 22: dispatched (sonnet). BASE 59a4029. LAST TASK.
Task 22: implemented 549055a. Verified independently: ruff clean on
src/bench/corpora, `mypy --strict` clean on 21 source files, 491 passed,
coverage 93% against an 85% threshold. Both gates cleared AND enabled.

Ruling on deviation 3, which is the significant one: the CI asset-scan step
passes `allow_empty=True`. The implementer verified against a real fresh
checkout that the brief's literal snippet raises AssetScanEmpty on EVERY CI
push, because weight files are gitignored. That is exactly the decision my
Task 19 ruling said someone would have to make deliberately, and it is made
visibly, in a commented CI step, rather than by silence.

But the honest consequence must be recorded and NOT left for a green badge to
imply: with allow_empty=True, THE ASSET GATE PROVIDES NO PROTECTION IN CI. It
protects only where it runs with assets present — a local release check, or a
CI job that first fetches weights. So spec 12.1 criterion 6 is enforced by
whoever runs the gate in an environment that has the assets, not by the
pipeline. A passing CI run is not evidence that criterion 6 holds. That
limitation belongs in the handoff, because the whole point of Task 19 was to
stop this gate from certifying something it never examined — and the remaining
gap is now visible and named rather than silent.

Deviations 1 and 2 accepted: ruff ignores N818/N812 (naming-style rules with a
125-call-site blast radius and no safety content), and the CI lint step is
scoped to `src bench corpora` rather than `.` (tests/ carries 38 findings that
were never in this task's baseline). Both are documented in the report.
Task 22: complete (commits 59a4029..549055a, review APPROVED, 0 Critical,
0 Important, 491 total green). ALL 22 TASKS COMPLETE.

Reviewer walked EVERY non-annotation edit in the 58KB diff as a table and
confirmed none can change a computed result — no cast(), no inserted asarray,
no coercion added where a value previously passed through, no default changed
from None. Verified by EXECUTION that the deliberate torch.load FutureWarning
still fires at loading.py:187. Zero `# type: ignore` tree-wide: both stale ones
removed, none added, so the annotations were made true rather than silenced.
Also confirmed the gate tests fail rather than pass when their tool is absent —
closing the missing-tool costume of the vacuity class.

Task 22: minors DEFERRED to final review — effnet.py:79 and quality.py:44 cast
only in one branch, so a float payload would diverge between branches (latent,
off-contract); npr.py:76,79 add two full-array copies per frame (~1.2MB) in the
hot path the benchmark measures p95 latency on, fixable with copy=False;
guards.py:117 zip(strict=True) widens check_video_level's contract to ValueError
without a Raises: note; guards.py:79 threshold now renders 0.8 not 0.800000 in
an operator-facing message; loading.py:191 `from None` drops the chained reason
the safe load failed, in the supply-chain path; THE COVERAGE GATE IS THE ONE
GATE WITH NO GATE TEST (impractical to self-invoke without recursion, so
red-in-CI-only); and tests/test_ci_gates.py:5 imports pytest unused.

=== FINAL WHOLE-BRANCH REVIEW: Ready with follow-ups. 6 must-fix. ===
Most serious, verified by the reviewer's own execution:
  check_identity_disjoint(['a','b','c'], ['a','b','c'], {}, threshold=0.6)
  -> IdentityReport(n_train=3, n_test=3, max_similarity=0.0, violations=0)
Three identical identities on both sides, certified disjoint, because no
embeddings were supplied — guards.py:72 `continue`s past any missing id. Same
vacuity class Task 19 spent two rounds removing from the asset gate, left in
the guard carrying criterion 2, which the plan calls its most important gap.
A future task wiring this into the runner before an embedder exists would
produce a green, numbered, apparently-verified identity report over ZERO
comparisons.

Also found: 2 of 6 guards never called (identity, demographic parity) so
criterion 1's "all five hygiene guards active" is FALSE; criteria 4 and 11
unmet and UNDISCLOSED while the Self-Review claims coverage; runner.py:290
catches protocol's ValueError wholesale, so a malformed corpus and an
unsplittable one both render as "Not computed for this corpus"; pyproject
declares neither torch nor sklearn though three modules import them at top
level; mypy and coverage both scope to src/dfd only, so bench/ (1335 lines,
the actual P0 deliverable) is ungated and "mypy clean on 21 files / 93%
coverage" reads as tree-wide when it is not.

Ledger synced by me (my own bookkeeping file): 1828 -> current.
Final fix wave dispatched — one subagent, five code/doc items.

=== FIX-WAVE RE-REVIEW: all 5 addressed, no new breakage. 5 residuals. ===
Re-review verified each of protocol.py's SEVEN raise sites lands on the correct
side of the new split (5 propagate as corpus defects, 2 degrade as unsplittable)
and confirmed _validate runs first, so a malformed corpus can never reach a
degradable site. It also reproduced the old bootstrap fixture and found 20
straddling sources out of 20 — so that fixture was genuinely invalid and the
edit is a correction, with production right. And it confirmed the identity
refusal is correctly scoped: zero production call sites, no test relied on the
skip, and a future ArcFace caller omits unembeddable ids explicitly rather than
having them silently dropped.

Ruling (adjudicated, not fixed — there is no second fix wave):
1. torch>=2.2 is now unconditional install_requires, so `pip install dfd` pulls
   ~800MB-2GB even for a caller touching only dfd.quality. ACCEPT: the imports
   are genuinely top-level, so the declaration is honest; an optional extra plus
   lazy import is P1 work. — Cost if wrong: a heavier install than some callers
   need, visible and fixable.
2. The corrected bootstrap fixture collapses to one generator, so the test takes
   the degrade-and-warn path — as it did before, via the swallowed straddle.
   ACCEPT: no assertion weakened, auc_ci is all it asserts.
3. _logo_results runs after all detector scoring, so a malformed corpus now
   aborts only after paying full scoring cost. ACCEPT: same species as the
   already-deferred duplicate-sample_id placement; hoist validation to the front
   of run_benchmark in P1.
4. A corpus legitimately sharing one source_id between a real video and its
   derived fake is classed malformed and now aborts loudly. ACCEPT: it is
   protocol.py's pre-existing rule; the fix only made it audible. NOTE IT FOR
   THE FIRST REAL CORPUS ADAPTER — this is the most likely of the five to bite.
5. The missing-embedding refusal uses GuardViolation, which reads "hygiene
   violated" rather than "input incomplete". ACCEPT: documented in both the
   function docstring and the report; no caller distinguishes.

WORKSPACE NOT DELETED, deliberately, against the skill's default. The committed
ledger carries every ruling, but the per-task reports are gitignored and would
be destroyed — and the branch is not merged, so nobody has read them yet.
Deleting the reasoning before the decision it informs is the wrong order.

---

# Ledger continued — plan: docs/superpowers/plans/2026-09-21-composition-root.md

Spec: docs/superpowers/specs/2026-09-21-composition-root-design.md (also binding; the earlier
spec above remains binding for everything it already covers)
Branch: p0-evidence-core (same branch, continued). BASE at start of this plan: 928ad63

## Task 1: Policy becomes an object fusion applies (thresholds become auditable configuration)

Ruling: `fake_threshold`, `real_threshold` and `disagreement_ood` move off `fusion.py`'s module
constants into a frozen `Policy` dataclass (`DEFAULT_POLICY`), and `fuse()` takes `policy=` so the
exact object that decided a verdict is the one object that reaches `build_audit_record` — a
record's stated threshold is provably the one applied, not a constant that could have drifted
between decision and audit. Fix round 1 required after review: the plan's own Step 3 snippet
deleted the constants' rationale comments (1.0 nat ≈ 73% posterior; the REAL-side symmetry; the
§7.2 "6-of-10 split vendors average away" framing for disagreement) rather than relocating them.
Spec §7.1 requires these thresholds be "versioned, auditable configuration"; configuration whose
derivation is undocumented is not auditable, so the plan was wrong here and the spec's method won.
Restored into `Policy`'s per-field docstrings. Cost if wrong: a caller could construct
`Policy(fake_threshold=X)` and have the audit record's stated threshold silently fail to match what
`fuse()` actually used — the record's "provably applied" guarantee would be false while still
looking true.

## Task 2: AuditRecord gains stage_reasons; AUDIT_SCHEMA_VERSION bumps "1" -> "2"

Ruling: `AuditRecord.stage_reasons: Mapping[str, str]` is added, frozen and digest-covered, so a
record can name which stage (faces, quality, calibration) could not measure, instead of only
carrying a verdict with no explanation for an abstention. `AUDIT_SCHEMA_VERSION` moves to `"2"`
because this changes the wire format of an audit record — an existing consumer must not be able to
parse a schema-2 record as if it were schema-1 and silently drop the new field without noticing the
shape changed underneath it. Cost if wrong (bump omitted): a schema-1-aware parser deserialises a
schema-2 record successfully, silently discards `stage_reasons`, and never learns the record's
shape changed — the same silent-drift failure class the predecessor plan's whole environment-drift
section exists to eliminate, reintroduced on the data side instead of the dependency side.

## Task 3: normalize() clamps face ROIs to non-negative origins before cropping

Ruling: `_clamp_roi`'s `max(0, ...)` on the box origin is kept, and given its own dedicated test and
mutation, after the task reviewer showed the clamp could be deleted with all 11 of the task's
original tests still green — the missing coverage was the plan's gap (its own Step 1 test file
never constructed a face box with a negative origin), not the implementer's. Without the clamp, a
face box straddling the frame edge (a detector reporting a negative x/y) silently crops the wrong
region: `reasons["faces"]` still reads `"ok"`, the record reports success, and every downstream
detector scores a crop offset from the face it claims to observe — verified by execution, not
assertion: `(50,20,3)` vs `(50,256,3)`, different pixels, same "ok" reason. Cost if wrong: a wrong
answer that is indistinguishable from a right one at the audit-record boundary — exactly the
~30-unfalsifiable-tests failure class §6 of the handoff exists to name.

## Task 4a: decide()'s ValueError -> InvalidInput translation is scoped to the ingest adapter call only

Ruling: the `except ValueError` around `load_image`/`load_video` in `decide()` wraps only that
adapter call, not the surrounding ingest block. Surfaced by a disclosed implementer deviation: the
plan's own undecodable-file fixture (`b"not an image"`) is rejected by Pillow's header probe
*before* the adapter's `ValueError` branch is ever reached, so it could not have exercised this
translation regardless of scope — the implementer swapped in a fixture truncated mid-`IDAT` chunk
that genuinely reaches it, and the reviewer independently reproduced both the original fixture's
miss and the replacement's hit. A `try/except` scoped around the whole ingest block instead of just
the adapter call would also relabel unrelated `ValueError`s from other ingest code as
"invalid input," hiding where they actually came from. Cost if wrong (scope too wide): a genuine bug
elsewhere in ingest reports as "bad input" in the audit record and gets debugged in the wrong place,
or never gets debugged at all, because the record's own error type misattributes the fault.

## Task 4b: ood_score carries FusedResult.disagreement, not an out-of-distribution score

Ruling (a documented compromise, not a defect): P0 has no Mahalanobis- or energy-based OOD head.
`disagreement` — `min(positive evidence, negative evidence)` — is the only OOD-shaped quantity
`fuse()` already computes, so `decide()` reports it under the `ood_score` field rather than adding a
fourth field nobody populates, or blocking the composition root on an OOD head that does not exist.
Documented in `decide()`'s own docstring and named again in this plan's known-gaps block, not left
for a caller to discover only by reading source. Cost if wrong (a caller trusts the field name over
the docstring): a threshold built on `ood_score` for OOD rejection is actually thresholding detector
disagreement — a plausible-looking number with the wrong semantics, silently wrong until someone
reads past the field name to the docstring.

## Task 5: dfd score exit codes — 0 for every verdict (including abstention), 2 for DfdError, 1 (uncaught) for anything else

Ruling: `main()` catches `DfdError` specifically and exits 2; it does not catch bare `Exception`.
Verified by mutation: deleting the `try`/`except` entirely lets an `InvalidInput` traceback
propagate uncaught instead of becoming an exit code, proving the `except` clause — not Python's
default — is what produces 2. Exit 0 covers `insufficient_evidence` deliberately: an abstaining
verdict is the pipeline succeeding at its job (correctly declining to decide), not the CLI failing,
so a caller scripting on exit code must not read 0 as "a real verdict was reached." Exit 1 for
anything unexpected has no code of its own — it is Python's ordinary behaviour when
`sys.exit(main())` (in `__main__.py`) is reached via an uncaught exception, which already satisfies
"1 for unexpected failure" without a matching `except` clause to maintain. Cost if wrong (0 conflated
with a real verdict): a caller scripting `if exit_code == 0: act_on_verdict()` would act on
`insufficient_evidence` as though it were a decision, in a system whose entire premise is that
abstention must route to manual review rather than be treated as an answer.

## Task 6: commit without pushing; record three corrected project facts without touching the spec

Ruling: Task 6 commits `docs/HANDOFF.md` and this ledger but does not run `git push` — the branch
carries open PR #1, so pushing is an outward-facing action the controller performs after reviewing
the commit, not something a dispatched task does unsupervised. Cost if wrong (pushed anyway): a diff
nobody reviewed lands on a branch with an open, green, `MERGEABLE` PR, discovered only after the
fact.

Ruling: the user corrected three project facts mid-execution — this product is NOT for ScoreMe;
dataset EULA requests will be sent by `kohrohit@gmail.com`; hardware is CPU-only for now, a GPU may
come later. These are recorded in `docs/HANDOFF.md` as a dated correction block. The spec
(`docs/superpowers/specs/2026-09-20-deepfake-detection-design.md`) is deliberately NOT edited: it is
the binding authority every review in this plan and its predecessor judged against, and rewriting
its ScoreMe framing mid-plan would retroactively invalidate those reviews. Re-framing §1/§7.1/§9/§11/
§12 for an independent, non-ScoreMe product is a substantial change that deserves its own cycle, not
a drive-by edit at the tail of an unrelated plan. Cost if wrong: the spec keeps a stale sponsor
framing — including a "ScoreMe to supply" placeholder that ScoreMe will never supply — until that
cycle runs, with the handoff contradicting it in one clearly-dated, clearly-labelled place in the
interim.

---

## Final whole-branch review and its single fix wave (2026-09-21)

These decisions postdate the Task 6 entries above. They were made after the composition-root plan's
six tasks were complete, during the whole-branch review and the one fix wave it triggered.

Ruling: the final whole-branch review was scoped to `3e35eb6..165c19e` — this plan's spec, plan and
task commits — rather than to `merge-base main..HEAD`, which would have been 101 commits. The
earlier P0 work on this branch already had its own whole-branch review in a previous session
(verdict "Ready with follow-ups, 6 must-fix", all six fixed and recorded above), and re-reviewing it
would have diluted attention across code unchanged since. Integration risk stayed covered because
this plan's diffs into the previously-reviewed modules (`fusion.py`, `audit.py`,
`detectors/registry.py`) all sit inside the chosen range. Cost if wrong: a defect introduced by the
earlier P0 work, and untouched by this plan, got no second look.

**What that review found.** 0 Critical, 5 Important. The two that mattered: the video path through
`decide()` had never executed — it was the only uncovered line in `pipeline.py`, and `load_video`'s
`max_frames`/`seed` were passed positionally as two `int`s, so a swap was invisible to
`mypy --strict` and to every test; and the band used for calibration is not the band the detector was
scored on, demonstrated by execution (31 `high` frames plus one `reject`, detector floor `low`,
calibrator fitted on `high` → the detector scores the 31, `decide` calibrates on `reject`, evidence
collapses to `uncalibrated_for_band` with llr 0.0).

Ruling F1: fix Important 1 (video path, keyword arguments, real video test), 3 (`default_registry`'s
composition asserted by nothing — deleting a `register` call left all 553 tests green), 4 (one
raising detector aborted the whole decision, contradicting parent-spec principle 8), 5 (the CLI
configured no logging, so its "stderr carries one line" contract was false in the default checkout),
and the cheap deferred items. Each closes a hole the suite could not see.

Ruling F2: do NOT change calibration-band semantics in a fix wave. Correct the spec's cost statement
and record the defect instead. Changing which band a detector calibrates on alters decision
semantics and belongs to the calibration milestone with its own design pass. The design spec §4.2
previously claimed the cost was "systematically pessimistic calibration ... the safe direction";
that is wrong and has been replaced — the real outcome is *no* calibration at all, because no
`reject`-band curve will ever be fitted, since detectors abstain there by design. Cost if wrong: the
defect stays latent until a calibrator is first fitted, which is the next milestone.

Ruling F3: defer the remaining minors into the known-gaps block rather than the fix wave — the dead
`IngestAdapter` Protocol matching neither adapter, `model_versions` vs `raw.version` as two sources
for one version string, `Detector.modalities` declared by every detector and consumed by nothing,
`normalize` never supplying yaw/pitch (so `Quality.yaw_deg`/`pitch_deg` are `0.0` in every record and
`quality.MAX_YAW_HIGH` is permanently inert), and per-detector latency captured only to a DEBUG log,
leaving acceptance criterion 5 no path out of `decide`. Cost if wrong: they persist as documented
gaps rather than silent ones.

Ruling: accept the fix wave's deferral of a characterisation test pinning today's
`uncalibrated_for_band` behaviour. A green test asserting that reason would read as a specification
rather than a symptom, forcing the next engineer to delete a *passing* test in order to fix a
documented defect — inverting the usual signal. The re-reviewer agreed and raised a real
counter-point, recorded here: a docs-only record can drift silently if `_worst_band` or
`filter_by_quality_floor` change shape, where a characterisation test would fail loudly and force
the docs to be revisited at that moment. Cost if wrong: the gap's documentation goes stale without
anything going red.

**Outcome.** 565 tests (up from 553), coverage 95.12%, `pipeline.py` at 100%, ruff and
`mypy --strict` clean, both CI legs green on `f52ac9a`. The suite was re-run with the YuNet `.onnx`
moved aside: same 565 passing, so no test depends on weights that exist on one machine and not in
CI. 13 mutations were run across the wave; one first-draft test survived its mutation and was
strengthened to assert the literal `"detector_error"`. The scoped re-review re-ran those mutations
itself in a scratch copy rather than trusting the report, and verified the `except Exception` wraps
only `detector.score` — calibration and record-building stay unshielded.
