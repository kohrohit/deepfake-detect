# Composition Root — Design

**Date:** 2026-09-21
**Status:** Approved in conversation; implementation plan pending
**Author:** rohit.kohli@scoreme.in with Claude
**Parent spec:** `2026-09-20-deepfake-detection-design.md` (the authority; this
document refines §5.1, §7.1 and §7.2 for one runnable path)

---

## 1. Problem

Every part of the engine exists and is tested. Nothing connects them.

`fuse`, `Calibrator.to_evidence`, `build_audit_record`, `load_image`,
`load_video` and `detect_faces` have **no non-test callers**. There is no
`__main__`, no CLI, no `[project.scripts]`. The benchmark runner builds
`Observation`s directly from dicts — fabricating landmarks at 35% and 65% of
frame width and an ROI covering the whole frame — so it bypasses ingest
entirely. Task 21's decode-bomb defences are therefore exercised by their own
unit tests and by nothing else.

The consequence is not stylistic. A system whose stages have never run in
sequence has never demonstrated that its stages *fit*: that quality measured by
one module is the band the next module calibrates on, that the threshold the
record reports is the threshold fusion applied, that a refused input fails at
the boundary rather than three stages in. Each of those is an assumption today.

**This design wires one path — file to audit record — and nothing else.**

## 2. Scope

**In:**

1. `normalize` — faces and quality onto ingested observations, which no module
   currently does outside the runner's fabrication.
2. `decide(path, ...) -> AuditRecord` — the composition root.
3. A `Policy` object, so the threshold in the record is the threshold applied.
4. `dfd score <path>` — a thin CLI over `decide`.
5. `stage_reasons` on `AuditRecord`, so an abstention states which stage caused
   it.

**Out** — each needs its own cycle, and saying so here is what keeps this one
finishable:

| Deferred | Why not here |
|---|---|
| Calibrator persistence | `decide` accepts fitted calibrators; nothing serialises them. No weights exist yet, so persistence would be a format designed against zero users. |
| Policy economics (§7.1 AUTO-PASS / STEP-UP / MANUAL / BLOCK) | Spec assigns to P1; the loss and friction figures are still "ScoreMe to supply". |
| RD adapter (criterion 4) | Benchmark-side wiring; reuses this ingest boundary once it exists. |
| Parity into `RunRecord` (criterion 11) | Same. |
| ArcFace embedder (criterion 2) | New mechanism, not wiring. |
| Batch mode, audit-trail directory | P5 packaging. |

## 3. What this path actually does today

It abstains three times over, for three different reasons, and that is the
correct behaviour:

| Stage | Reason it cannot conclude | Recorded as |
|---|---|---|
| normalize | YuNet ONNX is gitignored and absent | `faces: weights_absent` |
| score | detector weights absent | per-detector `weights_absent` |
| calibrate | no fitted curve for the band | `uncalibrated_for_band` |

**The first output of this work is an `INSUFFICIENT_EVIDENCE` record, not a
verdict.** Anyone reviewing this branch should expect that. The value is that
the record names all three causes, so the path is an audit trail rather than a
demonstration. A test (§7.1) proves the path is not *structurally* incapable of
a verdict, which is the only thing that separates "correctly abstaining" from
"broken in a way abstention hides".

## 4. Architecture

```
path ──hash──► ingest ──► normalize ──► score ──► calibrate ──► fuse ──► record
               limits     faces +       registry  per band      policy
               header-    quality                               ▲
               first                                            │
                                    same Policy object ─────────┘
```

**New modules**

- `src/dfd/policy.py` — frozen `Policy(fake_threshold, real_threshold,
  disagreement_ood, version)` and `DEFAULT_POLICY` (version `p0-default-v0`).
  The three numbers move here from `fusion.py`.
- `src/dfd/pipeline.py` — `normalize()` and `decide()`.
- `src/dfd/cli.py` and `src/dfd/__main__.py`; `[project.scripts]` gains
  `dfd = "dfd.cli:main"`.
- `default_registry()` — registers the NPR and EfficientNet detectors at their
  default weight paths. Both abstain when weights are absent, which is today.

**Modified**

- `fusion.fuse` gains `policy: Policy = DEFAULT_POLICY`. Defaults reproduce
  current behaviour exactly, so existing callers and tests are untouched.
  `fusion` re-exports `FAKE_THRESHOLD`, `REAL_THRESHOLD` and `DISAGREEMENT_OOD`
  from `policy`, and a test asserts each re-export equals its `Policy` field —
  two homes for one number is how they drift.
- `audit.AuditRecord` gains `stage_reasons`; `AUDIT_SCHEMA_VERSION` `"1"` → `"2"`.

### 4.1 `decide`

```python
def decide(
    path: str | Path,
    *,
    registry: Registry,
    calibrators: Mapping[str, Calibrator] | None = None,
    policy: Policy = DEFAULT_POLICY,
    context: Context | None = None,
    limits: Limits = DEFAULT_LIMITS,
    detect: FaceDetectFn = detect_faces,
    face_model: str | Path = faces.DEFAULT_MODEL,
    max_frames: int = DEFAULT_MAX_FRAMES,
    seed: int = 0,
    created_at: str | None = None,
) -> AuditRecord
```

`detect` and `created_at` are injectable for the same reason `build_audit_record`
already injects `created_at`: a path whose every dependency is hard-wired cannot
be tested for the case that matters (§7.1). `calibrators=None` means every
detector calibrates through an unfitted `Calibrator`, which returns
`llr = 0.0, abstained=True, reason="uncalibrated_for_band"` — the honest answer,
already implemented.

Stages, in order:

1. **Hash** — streamed sha256 of the file, computed before decode.
2. **Ingest** — `load_image` or `load_video`, chosen by an explicit extension
   map. An unknown extension raises `InvalidInput` naming it; it does not guess.
   Both adapters set `sample_id` to the file stem, and that is what reaches the
   record — two files with the same stem in different directories produce
   records with the same `sample_id` but different `input_sha256`.
3. **Normalize** — per observation: `detect(frame, face_model, with_reason=True)`,
   take the largest box by area, `measure_quality(frame, roi, box.landmarks[:2])`,
   rebuild the `Observation` with `roi` and `quality` set. Records one stage
   reason for the sample: `ok`, `no_face`, or `weights_absent`.
4. **Score** — each registered detector over `sample.observations`, wall-clock
   latency captured per detector.
5. **Calibrate** — `to_evidence(raw, band)` where `band` is the **worst measured
   band** across observations — worst by position in `types.QUALITY_BANDS`,
   which is ordered `reject < low < medium < high` — or `unmeasured` when no
   observation carried quality at all.
6. **Fuse** — `fuse(evidence, n_frames=1, policy=policy)`. `n_frames=1` because
   every detector in this repo aggregates internally; passing the frame count
   here would trigger the ESS discount against already-aggregated evidence,
   which `fuse`'s contract documents as misuse.
7. **Record** — `build_audit_record(...)` with the same `policy`.

### 4.2 Decisions, with what each costs if wrong

**Worst band, not mean or first.** Calibration conditions on the regime that
held for the whole sample; the mean of `high` and `reject` is a band the sample
never occupied.

**Cost if wrong — corrected 2026-09-21.** This paragraph previously read:
"systematically pessimistic calibration on mixed-quality video, which suppresses
evidence rather than inventing it — the safe direction." **That cost statement
is wrong,** and it is wrong in the direction that matters: it describes a cost
that is bounded and self-correcting, when the real one is neither.

The band `decide` calibrates on is not the band the detector was scored on.
`_worst_band` runs over **all** observations, while each detector internally
drops observations below **its own** floor via `filter_by_quality_floor`. Those
are different sets, so on any mixed-quality sample the calibration band names a
regime the detector never saw. And the realistic outcome is not pessimistic
calibration but **no calibration at all**: no `reject`-band curve will ever be
fitted, because detectors abstain in that band by design, so there is nothing
to fit from. `Calibrator.to_evidence` then returns `uncalibrated_for_band` with
llr 0.0 — the detector's evidence is discarded entirely, not merely discounted.

Demonstrated by execution (2026-09-21): a 32-frame clip of 31 `high` frames and
one flat `reject` frame, a detector with floor `low`, and a calibrator fitted on
`high`. The detector scores the 31 high frames and returns raw 0.3135;
`decide` calibrates that score on `reject` and emits
`{"llr": 0.0, "abstained": true, "reason": "uncalibrated_for_band"}`. The same
clip with the single flat frame removed emits `{"llr": -0.713, "abstained":
false, "reason": "ok"}` from the identical raw score. **One bad frame in
thirty-two destroys the whole sample's evidence** — and destroys it silently,
since `uncalibrated_for_band` is indistinguishable in the record from the
uncalibrated state every sample is in today.

The choice is deliberately left unchanged here: band semantics are a design
decision for the calibration milestone, and changing them in a wave with no
fitted calibrator anywhere would be changing behaviour nobody can yet measure.
**It must be re-argued when calibration lands**, against at least these
alternatives: calibrate each detector on the worst band among the observations
*that detector actually scored*; or carry the band per evidence row rather than
per record. Until then this is a known gap, recorded in the plan's known-gaps
block and in `docs/HANDOFF.md` §3.

**Largest face box.** v-CIP is single-subject. Cost if wrong: on a
multi-face frame the wrong subject is measured. The face count is recorded so
the situation is visible rather than silent.

**`ood_score` carries `fused.disagreement`.** P0 has no Mahalanobis or energy
OOD head; disagreement is the only OOD-shaped quantity that exists. The field
name overstates what it holds, so the docstring and the spec say so here. Cost
if wrong: a reader believes an OOD head ran. This is the weakest point in the
design and should be revisited when P1 adds the real head.

**A refused input raises and emits no record.** Spec §7.2 says every *decision*
emits a record; a decode-bomb refusal is not a decision. Cost if wrong: refusals
leave no trail — acceptable while the CLI logs them, revisit if refusals ever
need to be auditable.

**`decide` translates `load_image`'s documented `ValueError` into
`InvalidInput`.** That one bare raise is one of the 19 `errors.py` names, fixed
where this work already stands, with `__cause__` preserved. It is deliberately
not a blanket `except ValueError`, which would swallow genuine bugs — the
distinction that a previous review round in this repo already had to make once.

## 5. `stage_reasons` on the audit record

`AuditRecord` has no free-form field, and `build_audit_record` **drops
`Evidence.artifacts`** when it builds its rows, so a reason attached there does
not survive into the record. Without a new field, a record can say
`INSUFFICIENT_EVIDENCE` while being unable to say whether the face detector had
no weights or the frame had no face — two situations with completely different
remedies.

```python
stage_reasons: Mapping[str, str]   # e.g. {"faces": "weights_absent"}
```

It follows `model_versions` exactly: `_freeze`d to a `MappingProxyType`,
validated by `_validate_serialisable`, serialised by `to_json`, and therefore
covered by `record_digest` — the new field inherits immutability and
tamper-evidence rather than inventing either. `AUDIT_SCHEMA_VERSION` goes to
`"2"` because a consumer parsing a `"1"` record will not find this key.

Rejected: a sentinel band alone (loses the distinction the field exists to
carry) and a synthetic evidence row for the face stage (puts a non-detector in
the detector list, corrupting `n_contributing` and every downstream count).

## 6. CLI

```
dfd score PATH [--subject-id ID] [--max-frames N] [--seed N]
               [--face-model PATH] [-v|-vv] [--pretty]
```

- **stdout carries the record JSON and nothing else**, so it pipes into `jq`.
  `--pretty` reformats that JSON for reading and is display-only: the digest is
  always taken over the canonical `to_json()` output, never over the
  pretty-printed bytes, or the same record would digest two ways. This holds at
  every verbosity: log records go to stderr, never stdout.
- **stderr carries the one-line human summary, plus any log record at or above
  the configured level.** The summary is always the last line:
  `verdict=insufficient_evidence llr=0.00 band=unmeasured contributing=0/2 faces=weights_absent digest=a3f1…`

  `main` calls `logging.basicConfig(level=..., stream=sys.stderr,
  format="%(levelname)s %(name)s: %(message)s")`. Default is **WARNING**, `-v`
  is INFO, `-vv` is DEBUG. So the default checkout — no YuNet weights — emits
  two stderr lines, not one:

  ```
  WARNING dfd.faces: Face detector weights absent at assets/models/face_detection_yunet_2023mar.onnx
  verdict=insufficient_evidence llr=0.00 band=unmeasured contributing=0/2 faces=weights_absent digest=a3f1…
  ```

  **Corrected 2026-09-21.** This section previously claimed stderr carries one
  line. That was only ever true on a machine that happened to have the YuNet
  weights on disk — the minority case, and the opposite of CI. Worse, the
  warning arrived through Python's `lastResort` handler with no level and no
  logger name, because nothing configured logging; and every `logger.info` /
  `logger.debug` diagnostic in `decide` was unreachable from the only entry
  point that exists. Configuring logging in `main` is what makes the described
  contract true rather than accidental. `-vv` also admits third-party library
  loggers (PIL, torch) — verbose output is a debugging aid, not an interface.

- **`--max-frames` is validated at the parser and must be at least 1.** `0`
  otherwise reached `load_video`, which raised its zero-frame `ValueError`,
  which `_ingest` relabelled `could not decode <file>` — a mistyped flag
  reported as a corrupt video.

**Exit codes**

| Code | Meaning |
|---|---|
| `0` | a decision was produced — **any** verdict, `INSUFFICIENT_EVIDENCE` included |
| `2` | input refused (`DfdError`) — message on stderr, no traceback; also an `argparse` usage error, which exits 2 by its own convention |
| `1` | unexpected failure |

The verdict is deliberately **not** encoded in the exit status. `if dfd score f`
would otherwise read a `REAL` verdict as failure, and `set -e` would abort a
script on a correct answer. Callers branch on the JSON.

## 7. Testing

TDD throughout, and every test proved by breaking the implementation, watching
it fail, restoring, watching it pass — the method that caught roughly thirty
unfalsifiable tests earlier in this plan. A test nobody watched fail is a hope.

### 7.1 The load-bearing test

**The path can produce a real verdict.** With a `SyntheticDetector`, a fitted
`Calibrator` and an injected `detect` returning a usable face, `decide` returns
`FAKE` with `llr > policy.fake_threshold`. Every other test in this file passes
on a pipeline that always abstains; this is the only one that does not. It is
the reason `detect` is injectable.

### 7.2 The rest

| Test | Mutation that must make it fail |
|---|---|
| End-to-end on a real PNG → record names all three abstention causes | drop the `stage_reasons` wiring |
| Recorded threshold is the applied one | restore the module constant inside `fuse` |
| Decode-bomb refused **through `decide`**, CLI exits 2 | remove the pre-decode header check |
| Worst-band selection (`["high","low"]` → `low`) | select best band instead |
| `input_sha256` equals an independently computed hash | hash the wrong bytes |
| Digest covers `stage_reasons`; the mapping is frozen | omit the field from `to_json`; drop the `_freeze` |
| stdout parses as exactly one JSON object | print the human summary to stdout |
| Unknown extension → `InvalidInput` naming it | fall through to `load_image` |

Gates unchanged: ruff, `mypy --strict`, coverage ≥85%, both CI legs (`dev` and
`floor`).

## 8. Acceptance

1. `dfd score <a real jpeg>` emits a schema-2 audit record on stdout, exits 0,
   and the record names the face stage, each detector, and the calibration
   state as separate reasons.
2. `decide` is the only caller path needed to exercise ingest, limits, faces,
   quality, calibration, fusion and audit together.
3. `record.threshold` and `record.policy_version` describe the `Policy` that
   produced the verdict, demonstrated by a test that moves both.
4. A decode-bomb input exits 2 with a message and no traceback.
5. The verdict-capable test (§7.1) passes, so abstention is a measurement and
   not an artefact of the wiring.
6. Existing suites stay green with `fuse`'s new parameter defaulted.
