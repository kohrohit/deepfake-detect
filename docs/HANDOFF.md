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

## 0. Resume here (last touched 2026-09-21, after the merge)

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

## 0b. Previous resume block (2026-09-21, before the merge)

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
  joins them to `run_benchmark`; no RD table is rendered.
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

1. **Dataset EULAs — FF++, Celeb-DF, DFDC. Still not started.** Days to weeks of external lead time.
   Nothing in the build shortens this, and every benchmark number waits on it.
2. **Usable weights.** Measurement (c) shows public detectors do not transfer. Realistic routes:
   train SBI ourselves (needs only *real* faces, so licence-clean), or rent GPU to fine-tune on the
   442-session corpus.

---

## 5. Recommended next steps, in order

1. **Open the PR.** The branch is review-clean and pushed.
2. **Start the EULA requests** — in parallel with everything else, or the wait becomes sequential.
3. **An embedder for criterion 2.** Note `check_identity_disjoint` now *refuses* ids with no
   embedding rather than skipping them — a partial-embedding pipeline must omit unembeddable ids
   explicitly, which is the point.
4. **The RD adapter for criterion 4** — the 24 cached results are free and already labelled.

("A composition root" was step 3 here; it is done — see §3 above — and struck from this list
2026-09-21. The benchmark runner still does not call it, which is why criteria 4 and 11 are still
open and still numbered above as the next two steps.)

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

`.superpowers/sdd/2026-09-20-p0-evidence-core-and-benchmark/` is retained deliberately (the process
default is to delete it). It holds the per-task briefs and reports, which are gitignored and would
be lost. The committed ledger has every ruling; the reports have the working. Delete it once the PR
is merged.
