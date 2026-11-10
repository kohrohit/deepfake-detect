# Third-party baselines: run elsewhere, ingested here

This directory holds instructions, never code. A baseline detector (UFD, DistilDIRE, and
eventually the DeepfakeBench family) runs in **its own environment**, scores the
Deepfake-Eval-2024 image split, and emits a `sample_id,score` CSV. `bench/scorefile.py` ingests
that CSV. Nothing upstream of the CSV ever enters this repository.

## Why these live outside the package

1. **Dependency containment.** `tests/test_ci_gates.py` enforces exact `==` pins on both
   `requirements-dev.txt` and `requirements-floor.txt` across two CI legs
   (`test_every_dev_requirement_is_exactly_pinned`, `tests/test_ci_gates.py:127`). UFD,
   DistilDIRE and the DeepfakeBench family each carry their own dependency tree — their own
   pinned, often conflicting, torch/numpy/diffusion-library versions. Installing even one of
   them into this repo's environment would make that pinning unenforceable the moment the next
   baseline is added the same way.
2. **Licence containment.** `assets/manifest.yaml` fails closed: an unregistered asset is
   treated as non-commercial (see its header comment, and the `ffpp_effnetb4_weights` /
   `sbi_effnetb4_weights` entries for what a NonCommercial weight looks like once verified).
   DistilDIRE's own README states its licence as Creative Commons
   Attribution-NonCommercial 4.0 (verified at source this session — see
   `baselines/distildire/run.md`). A weight file that never enters this tree cannot be the file
   that blocks a commercial release, so this protocol measures every baseline by its output,
   never by vendoring it.

## The contract (`bench/scorefile.py`)

- **Header**: exactly `sample_id,score`. Anything else is refused, not coerced
  (`EXPECTED_HEADER` in `bench/scorefile.py`).
- **One row per image the baseline actually scored.** Partial coverage is permitted —
  `ScoreFile.coverage` is reported, never thresholded, and there is deliberately no floor below
  which a baseline is dropped from the report.
- **`sample_id` exactly as `corpora/dfeval.py` emits it**: `f"dfeval/{item_id}"`
  (`corpora/dfeval.py`, `load_dfeval_images`). `item_id` comes from the gated download's own
  manifest; a score file that reinvents this format a different way will be refused as scoring
  unknown ids, naming them.
- **Any monotone score scale is fine.** `bench.metrics.auc` is rank-based, so raw logits are
  exactly as usable as calibrated probabilities. Precisely *because* of that, no calibration
  (`ece`) is ever computed from an ingested score file — only AUC-family statistics are valid on
  a `ScoreFile`.
- **A duplicate `sample_id`, or an id outside the known corpus, is refused outright** — not
  silently overwritten or dropped — because either would change a denominator invisibly.

## The licence posture

Baselines are run as **internal evaluation only**. Nothing they produce is published, and no
weights they use ever enter this repository. This rests on an owner ruling recorded as open in
`docs/superpowers/specs/2026-11-10-benchmark-harness-design.md` §9.3: "Confirm that NonCommercial
baselines may be run as internal evaluation, under the eval-only posture already taken on the
capture swaps" is listed there as an unresolved owner action, not a settled fact. **If that ruling
goes the other way, the harness still runs and reports this project's own detectors alone** — the
baseline comparison is additive evidence, not a dependency of the harness.

## The honesty clause

Our reproduction of a published baseline is **our** reproduction, run by us, in an environment we
assembled, against a corpus we loaded. A discrepancy against the paper's published number could be
theirs (a different preprocessing or split), ours (a bug in how we wired the baseline or mapped
`sample_id`s), or the environment's (a dependency version the paper's authors didn't pin). The
report that comes out of this harness must state the discrepancy and must **not** resolve that
ambiguity by assumption — not "our number is right because the paper is old" and not "the paper's
number is right because it's published."

## Baselines documented here

| Baseline | `run.md` | Paper's published AUC on Deepfake-Eval-2024's image split |
|---|---|---|
| UFD (Universal Fake Image Detector) | `baselines/ufd/run.md` | 0.56 |
| DistilDIRE | `baselines/distildire/run.md` | 0.52 |

NPR is also reported in that paper, at AUC 0.53, but needs no `run.md` here: it is already slot C
of this repository (`dfd/detectors/npr.py`), reproduced in-process through `default_registry`
rather than ingested as a score file. See
`docs/superpowers/specs/2026-11-10-benchmark-harness-design.md` §7 for why that reproduction is
treated as the harness's own correctness gate.

The 0.56 / 0.52 / 0.53 figures above are recorded as read from
`docs/superpowers/specs/2026-11-10-benchmark-harness-design.md` §3 (itself sourced from "the paper
and the dataset card" for Deepfake-Eval-2024), not independently re-derived from the paper PDF in
this session. Treat them as the figure a reproduction should be checked against, per the honesty
clause above — not as a number this session re-verified against the primary source.
