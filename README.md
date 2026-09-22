# dfd — a deepfake-detection system that reports what it can prove

A CPU-only pipeline for deciding whether a face in an image or video is
synthetic, plus an always-on service around it: watch a folder, score what
lands in it, write an immutable audit record, serve the results.

## The state of it, in one table

Every detector reachable from this machine, measured on a corpus it did not
train on:

| detector | physics | weights | measured AUC | decides? |
|---|---|---|---|---|
| `blend_seam` | blending boundary (slot A) | fitted here, from FairFace self-blends | **0.289** — inverted | no |
| `dima806_vit` | learned appearance | on disk, Apache-2.0 | **0.521** — chance | not wired |
| `npr` | upsampling fingerprint (slot C) | absent | — | no |
| `effnet_b4` | learned appearance (slot E) | absent (obtainable weights are NonCommercial) | — | no |

**Nothing here can currently tell a deepfake from a real face.** The system
is built so that this is a fact you cannot miss rather than one you have to
go looking for: every verdict it issues today is `insufficient_evidence`, the
dashboard says why in a banner, and a test asserts it.

That is the product. A detector that does not work is normal; a system that
does not know it is the failure this repository exists to avoid.

## Run it

```bash
python3 -m pip install -e .
./ops/install.sh                    # systemd user unit, no root
```

Dashboard on <http://127.0.0.1:8077>, inbox at `~/.local/share/dfd/inbox` —
drop a file in and it is scored within a second or two. Full operator
documentation: **[docs/SERVICE.md](docs/SERVICE.md)**.

One file, without the service:

```bash
dfd score suspect.jpg              # the audit record on stdout
```

## The evidence gate

`bench/evidence_card.json` records each detector's measured AUC and the
corpus it was measured on. At startup the service keeps a calibration curve
only for detectors that clear a floor (0.75) **on a corpus they did not train
on**. Everything below it still runs and still has its raw score written into
the audit record — that data is worth accumulating — but contributes `llr
0.0` and cannot move a verdict.

Unmeasured fails closed. In-dataset AUC does not count. The card may raise
the floor, never lower it. A missing card is an error, not an empty gate.

`tests/service/test_evidence.py::test_no_detector_currently_clears_the_floor`
pins the present state: the day something is measured above the floor, that
test goes red and a person has to delete it deliberately. That commit is when
this system starts issuing real verdicts.

## Layout

| path | what it is |
|---|---|
| `src/dfd/` | the decision library — ingest, faces, quality, detectors, calibration, fusion, policy, audit |
| `src/dfd/service/` | the always-on service — store, worker, HTTP API, dashboard, evidence gate |
| `bench/` | the benchmark harness, LOGO protocol, guards, and the measured reports |
| `corpora/` | corpus loaders — captures, FairFace, DF40, self-blending |
| `training/` | fitters (not installed): `fit_blend`, `fit_calibration`, `export_fairface` |
| `ops/` | systemd unit and install scripts |
| `assets/manifest.yaml` | every dataset and weight file, with its licence. Unregistered means non-commercial |
| `docs/HANDOFF.md` | the full history: what was measured, what was refuted, what is blocked |

## What would make it work

Measured, not guessed — see `docs/HANDOFF.md §0` for the controls behind each:

1. **A swap-only evaluation subset.** Slot A finds composite boundaries;
   three quarters of DF40 is synthesis and reenactment, which have none. This
   needs per-technique labels the ungated repackaging does not carry — the
   full DF40 (a form) or FF++ (an academic signatory).
2. **A second slot with different physics** for everything that is not a
   swap. Slot C is declared and weightless; fitting it needs generated fakes.
3. **Licence-clean fake supply.** SFHQ (~425k synthetic faces, MIT upstream)
   needs a Kaggle account. Everything else found is NonCommercial.

More real faces will not help. That experiment was run: refitting on the
evaluation corpus's own reals moves the AUC from 0.289 to 0.344, which is
still the wrong side of chance.

## Gates

`ruff`, `mypy --strict`, 752 tests, 85% coverage floor, an asset-registration
gate, and dependency pins tested at both ends of every declared range. CI
runs the whole set twice — once against the newest pinned versions, once
against the oldest the package claims to support.
