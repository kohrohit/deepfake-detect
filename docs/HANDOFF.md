# Handoff — P0 Evidence Core and Benchmark Harness

**As of 2026-09-21.** Branch `p0-evidence-core`, **22 of 22 tasks complete**, 498 tests green,
ruff clean, `mypy --strict` clean, coverage 95%. Pushed. **The plan is finished; the branch has
not been merged.**

Read this, then `docs/superpowers/ledger/2026-09-20-p0-execution-ledger.md` — it carries all 106
rulings made during the build, each with what it costs if wrong. The spec
(`docs/superpowers/specs/2026-09-20-deepfake-detection-design.md`) remains the authority.

```bash
cd /home/rohit/Desktop/agents/deepfake && git checkout p0-evidence-core
python3 -m pytest -q          # 498 passed
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

**Still not guarded:** nothing exercises the declared floors automatically. Both ends are tested
today only because two machines happen to sit at opposite ends; CI runs the upper end alone. A
second CI leg installing the floor versions would turn that from coincidence into a gate.

To reproduce what CI sees: build a venv from `requirements-dev.txt` with no local site-packages and
run all four gates there.

---

## 0. Resume here (last touched 2026-09-21, end of session)

**PR #1 is open and green:** https://github.com/kohrohit/deepfake-detect/pull/1 — 84+ commits,
`MERGEABLE`, all four CI gates passing. Nothing is uncommitted or unpushed.

On a fresh machine:

```bash
git clone git@github.com:kohrohit/deepfake-detect.git && cd deepfake-detect
git checkout p0-evidence-core
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .   # now exactly pinned = CI's env
python3 -m pytest -q                                      # 498 passed
```

`gh` note: this repo is `kohrohit/*`, and `gh` may be active as a different account
(`gh auth switch -h github.com -u kohrohit`). `gh pr edit` fails here with a Projects-classic
GraphQL deprecation; use `gh api -X PATCH repos/kohrohit/deepfake-detect/pulls/1 -F body=@file`.

**What this session did, beyond opening the PR:** CI failed on its first run and exposed two real
defects that local could not see (opencv≥5 type stubs; torch≥2.6 flipping the `torch.load`
`weights_only` default, which had silently disabled the `allow_unsafe_load` branch). Both fixed.
Dependencies are now pinned, with three new drift gates in `tests/test_ci_gates.py`. See §1's
correction block and the environment-drift section below.

**Next, in order:** merge the PR (or get it reviewed); start the dataset EULAs — §4, still the
longest pole and still not started; then the composition root in §5.3. One deliberate gap left
open: nothing exercises the declared dependency floors automatically — a second CI matrix leg on
the floor versions would fix that, roughly ten lines of `ci.yml`.

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

Four more limitations, each recorded with its severity:

- **The CI asset gate provides no protection in CI.** Weight files are gitignored, so it runs with
  `allow_empty=True`. It protects only where run with assets present. **A green CI run is not
  evidence that criterion 6 holds.**
- **Quality banding is blind to every perturbation in the robustness surface.** Blur halves
  high-frequency energy and bands identically; so do both re-capture paths. The abstention
  mechanism will never route a recaptured sample to manual review on quality grounds. Note the
  coupling: the robustness sweep reuses the clean quality object *because* banding is blind, so
  fixing one invalidates the other's construction.
- **LOGO is computed but is not a trained protocol.** Each fold's train side is unused. It becomes
  load-bearing when calibration lands, and is where the operating threshold must be frozen.
- **`DfdError` is not yet the root of everything.** 19 bare `ValueError`/`RuntimeError` sites remain
  in `fusion.py`, `calibration.py`, `detectors/`, `ingest/`. The `errors.py` docstring says so.

**There is no composition root.** No `__main__`, no CLI, no `[project.scripts]`. `fuse`,
`Calibrator.to_evidence`, `build_audit_record`, `load_image`/`load_video` and `detect_faces` have no
non-test callers. The runner builds `Observation`s directly from dicts, bypassing ingest — so Task
21's decode-bomb defences are exercised by their unit tests and by nothing else.

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
3. **A composition root**: wire ingest → detectors → calibration → fusion → audit into one runnable
   path. Most of the unmet criteria are wiring, not new mechanism.
4. **An embedder for criterion 2.** Note `check_identity_disjoint` now *refuses* ids with no
   embedding rather than skipping them — a partial-embedding pipeline must omit unembeddable ids
   explicitly, which is the point.
5. **The RD adapter for criterion 4** — the 24 cached results are free and already labelled.

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
