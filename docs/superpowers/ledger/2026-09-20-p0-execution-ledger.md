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
