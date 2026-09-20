# Handoff — P0 Evidence Core and Benchmark Harness

**As of 2026-09-20.** Branch `p0-evidence-core`, 12 of 22 tasks complete, 154 tests green.

Read this, then `docs/superpowers/specs/2026-09-20-deepfake-detection-design.md` (the authority) and
`docs/superpowers/plans/2026-09-20-p0-evidence-core-and-benchmark.md` (the plan).
Every ruling made on the user's behalf is in `.superpowers/sdd/2026-09-20-p0-evidence-core-and-benchmark/progress.md`.

---

## 1. The three measurements that justify the whole design

These are receipts, taken on this machine against real data. They are why this system is built the
way it is, and they should survive into any pitch or review.

**(a) Your own fraud was approved five times.** From 442 v-CIP capture sessions: five have
`swapped=true` AND `approved=true`. Face match was defeated (attacker→victim 0.036 rejected; with
swap applied 0.857 accepted). Liveness passed, by design — a live human drives the puppet.

**(b) Reality Defender's ensemble is one model wearing ten hats.** Across 24 cached responses,
`rd-pine-img` alone reconstructs the aggregate at 0.992 separation. One member (`rd-full-elm-img`)
emits two distinct values across 24 samples. No sample ever had more than 6 of 10 members concur,
yet the aggregate tracks the maximum — so its false-positive rate approximates the *union* of its
members' FPRs.

**(c) An Apache-2.0 open detector scores 0.011 AUC on your fraud.**
`dima806/deepfake_vs_real_image_detection`, 21k downloads, run against the 24 frames from those five
approved sessions plus 150 genuine frames: mean P(fake) **0.002 on fraud vs 0.005 on genuine**. It
rates your fraud as *more real* than genuine applicants. 0% caught at every threshold. The label
direction was verified against the StyleGAN/FFHQ answer key, not assumed.

**The conclusion (c) forces: there is no off-the-shelf shortcut.** Shipping a pretrained public
detector produces a system that detects nothing while appearing operational — worse than none,
because it manufactures confidence.

---

## 2. What is built and verified

| Module | Status | Verified by execution |
|---|---|---|
| `src/dfd/types.py` | done | 8 core types; `QUALITY_BANDS` ordering pinned |
| `src/dfd/quality.py` | done | banding + `meets_floor`; `"reject"` unusable as a floor (mypy rejects it) |
| `src/dfd/faces.py` | done | YuNet (MIT) loads; weights on disk; landmark order documented UNVERIFIED |
| `src/dfd/ingest/` | done | deterministic frame sampling; survives unusable frame-count metadata |
| `src/dfd/manifest.py` | done | fail-closed; **observed blocking `npr_weights`** |
| `src/dfd/detectors/` | done | NPR physics: **0.0 on upsampled, 0.25 on natural**; EffNet slots A/E |
| `src/dfd/detectors/loading.py` | done | `weights_only=True` default; `model_factory` secure path; cache keyed on load config |
| `src/dfd/calibration.py` | done | **flips an inverted detector**: raw 0.95 → llr −4.40 |
| `src/dfd/fusion.py` | done | 900 identical frames → 1 observation's worth; abstentions contribute exactly 0 |
| `bench/metrics.py` | done | **group CI 0.751 vs row CI 0.063 — 11.9× wider** |
| `bench/guards.py` | done | six guards, all raise; identity report returns a measured number |

**Weights on disk** (gitignored, `assets/models/`): YuNet face detector (MIT, working);
`dima806` ViT (Apache-2.0, loads, but see measurement (c) — useless against face swaps).

---

## 3. What remains

Tasks 13–22, briefs already staged in the SDD workspace:

13 LOGO splits · 14 robustness surface (incl. screenshot/print recapture) · 15 **white-box PGD
baseline** · 16 corpus loaders · 17 benchmark runner · 18 wire robustness into the runner ·
19 asset enumeration · 20 **audit record (dropped spec requirement §7.2)** · 21 resource limits ·
22 CI gates (ruff, mypy --strict, coverage, asset gate)

Tasks 15, 17 and 20 are the ones where being wrong is expensive: 15 is what makes the
state-sponsored threat model measured rather than decorative; 17 produces the actual head-to-head
number; 20 is the RBI-defensible artifact without which the system can decide but not account for a
decision.

---

## 4. Rulings that a reviewer should sanity-check

Full list with costs-if-wrong is in the ledger. The ones most worth a second opinion:

1. **ESS discount is linear `ess/n`, not `sqrt`** — LLRs add for independent evidence. Under `sqrt`,
   900 identical frames still yielded 30.0, saturating the cap and reproducing the confidently-wrong
   failure the discount exists to prevent. *(This was my error, found in review.)*
2. **The ESS discount applies only to per-frame evidence lists.** Detectors aggregate internally via
   `probs.mean()`, so `Evidence.llr` is already whole-sample; discounting on top double-counts.
   Misuse now warns.
3. **`DISAGREEMENT_OOD = 3.0`, not the plan's 4.0** — the plan's own test was unreachable at 4.0.
   Operational risk recorded: 9 detectors at +3 and 1 at −3 routes to OOD despite total +24, so one
   miscalibrated or adversarial detector can force manual review over a high-confidence fraud call.
4. **Task 20 restores a dropped spec requirement**, not an enhancement. §7.2 mandates an immutable
   audit record; the original plan had no task for it.
5. **Licensing posture**: research-licensed assets permitted through build/benchmark/internal demo,
   must be swapped before any paying deployment. `npr_weights` and both EffNet weight sets are
   registered `commercial_use: false` and the gate blocks them.
6. **Calibration conditions on quality band only.** Spec §7 requires band AND compression level.
   Real gap; must close before the harness reports cross-compression numbers.

---

## 5. The recurring defect, and how to keep catching it

**Twenty-plus tests across eleven tasks shipped unable to fail for the right reason.** Every one
passed in a green suite. The variants seen:

- assertions on `.shape`/`.dtype` only — passed against a stub doing no work
- a test that performed the protection itself before asserting it
- **assertions at a clamp or bound** — correct, buggy and deleted implementations all saturate to the
  same value (hit twice, in consecutive rounds)
- **open intervals** (`0.5 < x < 1.0`) — a constant 0.75 stub passes
- `pytest.raises(ValueError)` with no `match=` — passes on any ValueError
- an entire untested *dimension*: every test passed a single-element list, hiding a bug where
  duplicate observations cancelled to zero

**The method that works:** break the implementation, watch the test fail, restore, watch it pass.
A test nobody watched fail is a hope, not a guard. Require that proof in every dispatch.

---

## 6. The critical path, which is not engineering

Two items gate a working detector and neither is shortened by more code:

1. **Dataset EULAs** — FF++, Celeb-DF, DFDC. Days to weeks of external lead time. **Not started.**
2. **Usable weights** — measurement (c) shows public Apache-2.0 detectors do not transfer to live
   face swaps. Realistic routes: train SBI ourselves (needs only *real* faces, so it is licence-clean
   and the consent problem is tractable), or rent GPU to fine-tune on the 442-session corpus.

Start the EULA requests in parallel with the remaining build, or the wait becomes sequential.

---

## 7. Resume instructions

```bash
cd /home/rohit/Desktop/agents/deepfake
git checkout p0-evidence-core
python3 -m pytest -q          # expect 154 passed, 1 warning
```

The 1 warning is expected and must not be suppressed: it is `torch.load` without `weights_only=True`
in the deliberately-gated unsafe branch, exercised once by a test. The warning is evidence the unsafe
path is unsafe.

Then read the ledger, and dispatch Task 13 using its staged brief. Nothing is half-finished: every
task is either complete with its review clean, or not started.
