# Benchmark Harness (Deepfake-Eval-2024, images) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure this repo's detectors on Deepfake-Eval-2024's image split beside the published
open-source baselines, with intervals, effective sample size and shortcut controls the paper does
not provide — so that "better than them" becomes a question with an answer.

**Architecture:** A loader turns the gated download into `bench.runner` records carrying the
originating web domain as the resampling family and one honest `dfeval_unknown_mixture` generator
id. Our detectors run in-process through the existing registry; third-party baselines never enter
this repo and arrive as `sample_id,score` CSVs. A separate controls module fits colour-only and
JPEG-table-only models on the corpus, each with a permutation null, to establish whether the
corpus can measure anything at all. Slot C's reproduction of the paper's published NPR number
gates every other figure the harness emits.

**Tech Stack:** Python 3.10+, numpy, opencv, scikit-learn, torch — all already pinned. **No new
runtime dependency.**

**Spec:** `docs/superpowers/specs/2026-11-10-benchmark-harness-design.md`

## Global Constraints

- **Nothing in this project trains on Deepfake-Eval-2024, now or later** (spec §3.1). It is an
  exam, and an exam that has been studied for is not an exam. The shortcut controls are the sole
  exception; they are diagnostics of the corpus, never shipped artefacts, and each carries a
  permutation null for that reason.
- **CC-BY-SA-4.0.** Evaluation-only, never redistributed. The corpus images never enter this
  repository or any artefact it ships.
- **No third-party baseline enters this repo.** Baselines run in their own environments and are
  ingested as score files. `tests/test_ci_gates.py` enforces exact `==` pins across two CI legs;
  several obtainable baselines are additionally NonCommercial.
- **No new runtime dependency.**
- **Every fake carries exactly one generator id, `dfeval_unknown_mixture`.** `generator: None`
  raises a plain `ValueError` in `bench.protocol._validate` and must not be used.
- **LOGO is refused and recorded as refused.** No fold is fabricated.
- **The permutation null attaches to the controls, not to the detectors** — nothing is fitted on
  this corpus, so there is no fit variance to null out.
- **No single corpus-wide AUC is reported for a face detector.** The headline is the face-bearing
  subset, with `abstention_by_class` beside it.
- **Guards are waived explicitly, in a named report block**, never quietly satisfied — the
  pattern `bench/eval_df40.py` already establishes.
- **Repo gates, every commit:** `ruff check .`, `mypy --strict src/`, `python3 -m pytest -q`,
  coverage ≥ 85%, asset gate. Baseline: 928 tests, 92.8% coverage.
- **Tasks 1–6 need no corpus access.** Only Task 7 does. Everything is built and tested against
  synthetic fixtures first, so the HuggingFace gate blocks one task rather than the plan.

---

## File Structure

**Created:**

| Path | Responsibility |
|---|---|
| `corpora/dfeval.py` | Gated download → `bench.runner` records. Owns the on-disk layout, the domain field, dedup, and the face/no-face determination. |
| `bench/scorefile.py` | Ingest `sample_id,score` CSVs from baselines run elsewhere. Validation and coverage accounting. Knows nothing about models. |
| `bench/dfeval_controls.py` | Colour-only and JPEG-table-only models fitted on the corpus, each with a permutation null. Answers "can this corpus measure anything". |
| `bench/eval_dfeval.py` | Orchestrator: the oracle gate, our detectors, the baselines, the statistics, the waiver and scope blocks, the report. |
| `baselines/README.md`, `baselines/<name>/run.md` | How each third-party baseline is run, outside this repo, to produce its score file. Documentation, not code. |
| tests mirroring each module | One test file per unit. |

**Modified:**

| Path | Change |
|---|---|
| `bench/metrics.py` | `kish_ess`. |
| `bench/runner.py:217-224` | `_observation` honours a record's own `roi` and `landmarks` when present. |
| `assets/manifest.yaml` | `deepfake_eval_2024`. |
| `bench/evidence_card.json` | The corpus entry and its caveat. |
| `docs/EULA-ACCESS.md` | The HuggingFace gate and what it requires. |
| `docs/HANDOFF.md` | A §0 resume block. |

---

### Task 1: `kish_ess`

Reported beside every n. A reader handed 1,975 rows will otherwise read them as 1,975
independent observations, and on this corpus they are 88 domains' worth.

**Files:**
- Modify: `bench/metrics.py`
- Test: `tests/bench/test_metrics.py`

**Interfaces:**
- Produces: `def kish_ess(counts: Sequence[int] | np.ndarray) -> float` — `(Σn)² / Σn²`.

**Note:** the slot D plan (`2026-11-10-slot-d-gend-clip.md`, Task 7) specifies the identical
function. Whichever plan executes first writes it; the second finds it present, runs its tests
against the existing implementation, and skips to its next task rather than writing a second copy.

- [ ] **Step 1: Write the failing tests**

```python
def test_kish_ess_of_equal_groups_is_the_group_count():
    assert kish_ess([10, 10, 10, 10]) == pytest.approx(4.0)


def test_kish_ess_collapses_when_one_group_dominates():
    # DF40's measured shape: 1601 fakes in 45 filename families, 999 in the
    # largest. The answer was 2.4, and a corpus worth 2.4 fakes cannot carry
    # an interval however many rows it has.
    assert kish_ess([999] + [13] * 44) < 3.0


def test_kish_ess_of_one_group_is_one():
    assert kish_ess([500]) == pytest.approx(1.0)


def test_kish_ess_refuses_an_empty_or_all_zero_input():
    # Returning 0.0 would read as a measured answer rather than as nothing
    # to measure, and would then be divided by somewhere downstream.
    with pytest.raises(ValueError, match="at least one"):
        kish_ess([])
    with pytest.raises(ValueError, match="at least one"):
        kish_ess([0, 0])
```

- [ ] **Step 2: Run them and watch them fail**

Run: `python3 -m pytest tests/bench/test_metrics.py -q -k kish`
Expected: `ImportError: cannot import name 'kish_ess' from 'bench.metrics'`.

- [ ] **Step 3: Implement**

```python
def kish_ess(counts: Sequence[int] | np.ndarray) -> float:
    """Kish's effective sample size over group sizes: (sum n)^2 / sum n^2.

    Equals the group count when groups are equal and collapses toward 1 when
    one group dominates. Measured on DF40: 1601 fakes in 45 filename families
    with 999 in the largest gives 2.4, which is why no interval over that
    corpus meant what it appeared to. Report it beside every n, or the reader
    is invited to read rows as independent observations.

    This is a DIFFERENT quantity from `dfd.fusion.effective_sample_size`,
    which estimates ESS from lag-1 autocorrelation of a frame series. That
    one asks "how independent are these consecutive frames"; this one asks
    "how concentrated is this corpus into a few families". Both are called
    ESS in the literature and they are not interchangeable.

    Raises:
        ValueError: on an empty or all-zero input.
    """
    n = np.asarray(counts, dtype=float)
    if n.size == 0 or not n.sum():
        raise ValueError("kish_ess needs at least one non-empty group")
    return float(n.sum() ** 2 / np.square(n).sum())
```

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/bench/test_metrics.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add bench/metrics.py tests/bench/test_metrics.py
git commit -m "feat: effective sample size, so 1,975 rows stop reading as 1,975 observations"
```

---

### Task 2: The corpus loader

**The on-disk layout is unknown and must not be guessed.** The HuggingFace dataset card does not
publish it and the viewer is behind the gate. So the layout knowledge lives in **one function**,
`_manifest_rows`, and every test below drives the loader through a synthetic fixture in the shape
that function returns. When the real download arrives (Task 7), confirming the layout is a change
to that one function and nothing else.

**Files:**
- Create: `corpora/dfeval.py`
- Create: `tests/corpora/test_dfeval.py`
- Modify: `bench/runner.py:217-224`
- Modify: `tests/bench/test_runner.py`

**Interfaces:**
- Consumes: `dfd.faces.{detect_faces, FaceBox, clamp_roi}`, `dfd.quality.measure_quality`.
- Produces:
  - `DFEVAL_GENERATOR: str = "dfeval_unknown_mixture"`
  - `COMPRESSION: str = "unknown"`, `FACE_DETECTOR: str = "yunet_2023mar"`,
    `ALIGN: str = "native_roi"`
  - skip reasons `UNREADABLE`, `DUPLICATE`, `UNLABELLED` — counted in the returned tally
  - `CONTRADICTORY_DUPLICATE: str` — **not** a skip reason. It is the message prefix of a
    `ValueError` that refuses the whole corpus, because identical pixels carrying opposite
    labels is a contradiction no per-row tally can express.
  - `class DFEvalRow` — `path: Path`, `label: int`, `domain: str`, `item_id: str`
  - `def _manifest_rows(root: Path) -> list[DFEvalRow]` — **the only function that knows the
    on-disk layout.**
  - `def load_dfeval_images(root, *, limit=None, seed=0, detect=detect_faces) -> tuple[list[dict], dict[str, int]]`
    — `(records, skipped)`. Each record carries `sample_id`, `source_id`, `subject_id`,
    `generator`, `label`, `compression`, `face_detector`, `align`, `image`, `domain`,
    `has_face`, `roi`, `landmarks`.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import numpy as np
import pytest

from corpora.dfeval import (
    CONTRADICTORY_DUPLICATE,
    DFEVAL_GENERATOR,
    DUPLICATE,
    load_dfeval_images,
)


def test_records_satisfy_bench_protocol_validation(dfeval_root, stub_detect):
    # _validate is strict about three things this loader can get wrong:
    # labels outside {0,1}, an unattributed fake, and a real carrying a
    # generator. Catching them here beats catching them inside a fold.
    from bench.protocol import _validate

    records, _ = load_dfeval_images(dfeval_root, detect=stub_detect)
    _validate(records)


def test_every_fake_carries_the_single_unknown_mixture_generator(
        dfeval_root, stub_detect):
    records, _ = load_dfeval_images(dfeval_root, detect=stub_detect)
    fakes = [r for r in records if r["label"] == 1]
    assert fakes
    assert {r["generator"] for r in fakes} == {DFEVAL_GENERATOR}
    assert all(r["generator"] is None for r in records if r["label"] == 0)


def test_logo_refuses_this_corpus_gracefully(dfeval_root, stub_detect):
    # The whole point of one honest generator id rather than None: holding
    # out the only generator leaves no training fakes, so logo_splits raises
    # UnsplittableCorpusError -- which a caller may catch and record -- and
    # NOT the plain ValueError that means a malformed corpus.
    from bench.protocol import UnsplittableCorpusError, logo_splits

    records, _ = load_dfeval_images(dfeval_root, detect=stub_detect)
    with pytest.raises(UnsplittableCorpusError):
        logo_splits(records)


def test_the_domain_is_carried_as_the_resampling_family(dfeval_root, stub_detect):
    records, _ = load_dfeval_images(dfeval_root, detect=stub_detect)
    assert all(r["domain"] for r in records)
    assert len({r["domain"] for r in records}) > 1


def test_identical_pixels_are_deduplicated_and_counted(dfeval_root_with_dupe,
                                                       stub_detect):
    # This repo has already read 58 distinct images as a pool of 1,088. An
    # in-the-wild corpus assembled from social media is the place that
    # happens again: one image circulates across many domains.
    records, skipped = load_dfeval_images(dfeval_root_with_dupe,
                                          detect=stub_detect)
    assert skipped[DUPLICATE] == 1
    arrays = [r["image"].tobytes() for r in records]
    assert len(set(arrays)) == len(arrays)


def test_a_duplicate_spanning_both_labels_is_refused_not_dropped(
        dfeval_root_with_contradiction, stub_detect):
    # The same pixels labelled real in one place and fake in another is a
    # labelling contradiction. Dropping one silently picks a winner; the
    # corpus has to be refused and the hash named.
    with pytest.raises(ValueError, match="contradictory"):
        load_dfeval_images(dfeval_root_with_contradiction, detect=stub_detect)


def test_images_with_no_face_are_KEPT_and_flagged(dfeval_root_mixed, stub_detect_sometimes):
    # Dropping them would hide the gap §6 of the spec exists to measure:
    # this corpus is in-the-wild AI media and not all of it is facial, while
    # every detector here needs a face. A loader that filtered them would
    # make the abstention accounting impossible and quietly flatter us.
    records, skipped = load_dfeval_images(dfeval_root_mixed,
                                          detect=stub_detect_sometimes)
    assert any(not r["has_face"] for r in records)
    assert all(r["roi"] is None for r in records if not r["has_face"])
    assert "no_face" not in skipped


def test_a_faced_record_carries_a_real_roi_and_real_landmarks(dfeval_root,
                                                              stub_detect):
    records, _ = load_dfeval_images(dfeval_root, detect=stub_detect)
    faced = [r for r in records if r["has_face"]]
    assert faced
    for r in faced:
        x, y, w, h = r["roi"]
        assert w >= 2 and h >= 2
        assert np.asarray(r["landmarks"]).shape[1] == 2


def test_preprocessing_fields_are_uniform(dfeval_root, stub_detect):
    # check_uniform_preprocessing fails a corpus whose face_detector or
    # align varies, and it is right to: applying different preprocessing by
    # label is itself a giveaway the model will happily learn.
    from bench.guards import check_uniform_preprocessing

    records, _ = load_dfeval_images(dfeval_root, detect=stub_detect)
    check_uniform_preprocessing(records)


def test_compression_is_recorded_unknown_not_invented(dfeval_root, stub_detect):
    # corpora.df40 and corpora.sfhq set the precedent: an invented c23 label
    # passes check_compression_coverage by lying to it.
    records, _ = load_dfeval_images(dfeval_root, detect=stub_detect)
    assert {r["compression"] for r in records} == {"unknown"}


def test_limit_is_applied_deterministically(dfeval_root, stub_detect):
    a, _ = load_dfeval_images(dfeval_root, limit=4, seed=0, detect=stub_detect)
    b, _ = load_dfeval_images(dfeval_root, limit=4, seed=0, detect=stub_detect)
    assert [r["sample_id"] for r in a] == [r["sample_id"] for r in b]
    assert len(a) <= 4


def test_a_missing_root_names_the_gate_rather_than_raising_a_traceback(tmp_path):
    with pytest.raises(FileNotFoundError, match="huggingface.co/datasets"):
        load_dfeval_images(tmp_path / "absent")
```

Fixtures go in the **top-level `tests/conftest.py`**, not `tests/corpora/`: Task 5's tests live
under `tests/bench/` and need the same `dfeval_root` and `stub_detect`, and a fixture copied into
a second conftest is a fixture that drifts. `dfeval_root` writes a handful of JPEGs with a drawn
face-like blob plus the metadata file `_manifest_rows` reads, across at least two domains and both
labels; `dfeval_root_with_dupe` repeats one image's exact bytes under a second name;
`dfeval_root_with_contradiction` repeats it under the opposite label; `dfeval_root_mixed` includes
images the stub detector finds no face in. `stub_detect` returns one fixed box;
`stub_detect_sometimes` returns none for images whose mean exceeds a threshold. Follow the shape
already used in `tests/corpora/test_sfhq.py` rather than inventing a fourth fixture style.

- [ ] **Step 2: Run them and watch them fail**

Run: `python3 -m pytest tests/corpora/test_dfeval.py -q`
Expected: `ModuleNotFoundError: No module named 'corpora.dfeval'`.

- [ ] **Step 3: Write the loader**

Module docstring must carry: what the corpus is (1,975 images, 767 real / 1,208 fake, 88 domains);
its licence and the training prohibition; that `_manifest_rows` is the single point of layout
knowledge and was written before the gated download arrived; and that no-face images are kept
deliberately.

```python
#: The single generator id every fake carries. One value, so a LOGO split is
#: refused by `bench.protocol` rather than fabricated -- the same device
#: `corpora.df40` uses, and for the same reason. The name says outright that
#: the techniques are unknown, because they are: Deepfake-Eval-2024 carries
#: no per-item generator labels. Inventing SEVERAL ids would be the lie;
#: inventing one that admits ignorance is what makes the refusal work.
DFEVAL_GENERATOR = "dfeval_unknown_mixture"

#: Recorded rather than guessed. The images arrive from 88 web domains at
#: whatever quality each platform re-encoded them to, and `corpora.df40` and
#: `corpora.sfhq` both set the precedent: an invented "c23" passes
#: `check_compression_coverage` by lying to it.
COMPRESSION = "unknown"

FACE_DETECTOR = "yunet_2023mar"
#: Not an alignment at all -- the ROI is passed through at its native
#: resolution and each detector crops as it sees fit. Recorded under the key
#: `check_uniform_preprocessing` reads so that the value is uniform and
#: honest rather than absent.
ALIGN = "native_roi"
```

`load_dfeval_images` must, in order:

1. refuse a missing root with a message naming
   `https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024` and the gate's requirements;
2. read `_manifest_rows(root)`;
3. decode each image, hashing the **decoded array bytes** (not the file bytes — a platform
   re-encode of identical pixels is still the same image, and hashing the file would miss it;
   near-duplicates are not caught by either, which the docstring must say);
4. on a repeated hash: if the two rows carry the **same** label, count `DUPLICATE` and skip; if
   they carry **different** labels, raise `ValueError` naming the hash and both item ids;
5. detect faces, take the largest box — **the same rule `training.fit_vcip.select_face` uses**,
   because a different face rule between fitter and evaluator is the defect that made every slot C
   number stale once — and record `roi`, `landmarks`, `has_face`;
6. keep every decoded image whether or not it has a face;
7. return records plus the skip tally.

- [ ] **Step 4: Teach `bench/runner.py` to honour a record's own ROI**

`_observation` currently hardcodes `roi=(0, 0, w, h)` and **synthesises** landmarks at
`[[w*0.35, h*0.4], [w*0.65, h*0.4]]`. Both are wrong for this corpus: a whole-image ROI makes a
face detector score background, and synthetic landmarks give a fabricated inter-ocular distance,
which feeds `measure_quality` and therefore the quality-floor filter every detector applies.

```python
def _observation(record: dict) -> Observation:
    """Build the Observation a detector sees.

    A record MAY carry its own `roi` and `landmarks`, and when it does they
    are used. Corpora that localise faces at load time (`corpora.dfeval`)
    know where the face is and at what scale; the whole-image fallback below
    exists for corpora whose records are already face crops, where the crop
    IS the ROI. Synthesising landmarks for an image that has real ones
    fabricates the inter-ocular distance `measure_quality` bands on, and the
    band decides which detectors are allowed to score at all.
    """
    img = record["image"]
    h, w = img.shape[:2]
    roi = record.get("roi") or (0, 0, w, h)
    lm = record.get("landmarks")
    if lm is None:
        lm = np.array([[w * 0.35, h * 0.4], [w * 0.65, h * 0.4]])
    q = measure_quality(img, roi, np.asarray(lm))
    return Observation(t=0.0, payload=img, roi=roi, quality=q,
                       source_id=record["source_id"])
```

Add two tests to `tests/bench/test_runner.py`:

```python
def test_observation_uses_a_records_own_roi_and_landmarks_when_present():
    img = np.full((100, 100, 3), 120, np.uint8)
    rec = {"image": img, "source_id": "s", "roi": (10, 10, 40, 40),
           "landmarks": np.array([[20.0, 25.0], [45.0, 25.0]])}
    obs = _observation(rec)
    assert obs.roi == (10, 10, 40, 40)


def test_observation_falls_back_to_the_whole_frame_for_crop_corpora():
    # Unchanged behaviour for every existing corpus, whose records ARE crops.
    img = np.full((64, 64, 3), 120, np.uint8)
    obs = _observation({"image": img, "source_id": "s"})
    assert obs.roi == (0, 0, 64, 64)
```

- [ ] **Step 5: Run the tests**

Run: `python3 -m pytest tests/corpora/test_dfeval.py tests/bench/test_runner.py -q`
Expected: all pass, including every pre-existing runner test — the fallback must be byte-for-byte
the old behaviour.

- [ ] **Step 6: Run the gates**

Run: `ruff check . && mypy --strict src/ && python3 -m pytest -q`

- [ ] **Step 7: Commit**

```bash
git add corpora/dfeval.py tests/corpora/test_dfeval.py tests/corpora/conftest.py \
        bench/runner.py tests/bench/test_runner.py
git commit -m "feat: a loader for the in-the-wild corpus, and a runner that stops inventing landmarks"
```

---

### Task 3: The score-file protocol

A baseline never enters this repository. It runs in its own environment and emits
`sample_id,score`; this module ingests that and nothing else.

**Files:**
- Create: `bench/scorefile.py`
- Create: `tests/bench/test_scorefile.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class ScoreFile` — `name: str`, `scores: dict[str, float]`,
    `covered: int`, `total: int`, `coverage: float`, `missing: tuple[str, ...]`
  - `def load_score_file(path: str | Path, known_ids: Collection[str], *, name: str | None = None) -> ScoreFile`
  - `def aligned(sf: ScoreFile, records: Sequence[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
    — `(scores, labels, groups)` over the covered subset only, in record order.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import numpy as np
import pytest

from bench.scorefile import aligned, load_score_file


def _write(tmp_path, text, name="ufd.csv"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_loads_scores_and_reports_coverage(tmp_path):
    p = _write(tmp_path, "sample_id,score\na,0.1\nb,0.9\n")
    sf = load_score_file(p, known_ids={"a", "b", "c", "d"})
    assert sf.scores == {"a": 0.1, "b": 0.9}
    assert (sf.covered, sf.total) == (2, 4)
    assert sf.coverage == pytest.approx(0.5)
    assert sf.missing == ("c", "d")


def test_the_name_defaults_to_the_file_stem(tmp_path):
    sf = load_score_file(_write(tmp_path, "sample_id,score\na,0.1\n"),
                         known_ids={"a"})
    assert sf.name == "ufd"


def test_an_unknown_sample_id_is_refused_and_named(tmp_path):
    # A baseline scoring a corpus it was not given is a configuration error,
    # not a low coverage number, and the two must not look alike.
    p = _write(tmp_path, "sample_id,score\na,0.1\nzzz,0.9\n")
    with pytest.raises(ValueError, match="zzz"):
        load_score_file(p, known_ids={"a"})


def test_a_duplicate_sample_id_is_refused(tmp_path):
    # One row silently overwriting another changes the denominator
    # invisibly: coverage still reads 100% and one image was scored twice.
    p = _write(tmp_path, "sample_id,score\na,0.1\na,0.9\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_score_file(p, known_ids={"a"})


def test_a_non_finite_score_is_refused(tmp_path):
    p = _write(tmp_path, "sample_id,score\na,nan\n")
    with pytest.raises(ValueError, match="not finite"):
        load_score_file(p, known_ids={"a"})


def test_scores_outside_zero_one_are_accepted(tmp_path):
    # Deliberate. AUC is rank-based, so a baseline emitting logits is as
    # usable as one emitting probabilities, and rejecting logits would
    # silently exclude the baselines most likely to be honest about what
    # their model outputs. Calibration is NOT claimed for these and the
    # report must not print an ECE for them.
    sf = load_score_file(_write(tmp_path, "sample_id,score\na,-3.2\nb,7.9\n"),
                         known_ids={"a", "b"})
    assert sf.scores["a"] == pytest.approx(-3.2)


def test_an_empty_file_is_refused_rather_than_read_as_total_abstention(tmp_path):
    p = _write(tmp_path, "sample_id,score\n")
    with pytest.raises(ValueError, match="no rows"):
        load_score_file(p, known_ids={"a"})


def test_a_missing_header_is_refused(tmp_path):
    p = _write(tmp_path, "a,0.1\nb,0.9\n")
    with pytest.raises(ValueError, match="header"):
        load_score_file(p, known_ids={"a", "b"})


def test_aligned_returns_the_covered_subset_in_record_order(tmp_path):
    records = [
        {"sample_id": "a", "label": 0, "domain": "x.com"},
        {"sample_id": "b", "label": 1, "domain": "y.com"},
        {"sample_id": "c", "label": 1, "domain": "x.com"},
    ]
    sf = load_score_file(_write(tmp_path, "sample_id,score\nc,0.7\na,0.2\n"),
                         known_ids={"a", "b", "c"})
    scores, labels, groups = aligned(sf, records)
    assert list(scores) == [0.2, 0.7]
    assert list(labels) == [0, 1]
    assert list(groups) == ["x.com", "x.com"]
```

- [ ] **Step 2: Run them and watch them fail**

Run: `python3 -m pytest tests/bench/test_scorefile.py -q`
Expected: `ModuleNotFoundError: No module named 'bench.scorefile'`.

- [ ] **Step 3: Implement**

Module docstring states the containment argument: no third-party dependency enters a repo whose
two requirements files are exactly `==`-pinned and gate-enforced, and no NonCommercial weight file
sits in the tree of a product that ships. It also states that coverage is reported rather than
thresholded — there is no number below which a baseline is dropped, because dropping one is a
judgement the reader makes with the figure in hand, not one the harness makes for them.

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/bench/test_scorefile.py -q`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add bench/scorefile.py tests/bench/test_scorefile.py
git commit -m "feat: baselines are measured without being installed"
```

---

### Task 4: The shortcut controls

The question these answer is not "how good is our detector" but "can this corpus measure a
detector at all". On an in-the-wild corpus whose halves arrive down different platform upload
pipelines, that question is sharper than anywhere this project has asked it: Lab colour alone
separated DF40 at 0.843, and one JPEG header byte labels 99.2% of the capture corpus.

**Files:**
- Create: `bench/dfeval_controls.py`
- Create: `tests/bench/test_dfeval_controls.py`

**Interfaces:**
- Consumes: `bench.metrics.{auc, bootstrap_ci_by_group, permutation_null, permutation_p}`,
  `bench.scorefile` is **not** used here.
- Produces:
  - `def colour_features(images: Sequence[np.ndarray]) -> np.ndarray` — `(N, 6)`: per-channel
    Lab means and standard deviations.
  - `def jpeg_table_features(paths: Sequence[Path]) -> np.ndarray` — `(N, K)`: the luma
    quantisation table read from each file's header.
  - `def run_control(features, labels, groups, *, seed: int, n_null: int = 24) -> dict[str, Any]`
    — fits on a group-disjoint half, scores the other, returns
    `{auc, ci_lo, ci_hi, null_lo, null_hi, p, n_fit, n_test}`.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import numpy as np
import pytest

from bench.dfeval_controls import colour_features, run_control


def _planted(rng, n, lo, hi):
    return [rng.integers(lo, hi, (24, 24, 3), dtype=np.uint8) for _ in range(n)]


def test_the_colour_control_finds_a_planted_colour_shortcut():
    # The control must be able to FIRE, or reporting it proves nothing.
    rng = np.random.default_rng(0)
    images = _planted(rng, 60, 0, 110) + _planted(rng, 60, 145, 255)
    labels = np.array([0] * 60 + [1] * 60)
    groups = np.array([f"g{i % 12}" for i in range(120)])
    out = run_control(colour_features(images), labels, groups, seed=0)
    assert out["auc"] > 0.9
    assert out["p"] < 0.1          # and it escapes its own null


def test_the_colour_control_stays_blind_to_a_corpus_without_one():
    rng = np.random.default_rng(0)
    images = _planted(rng, 60, 0, 255) + _planted(rng, 60, 0, 255)
    labels = np.array([0] * 60 + [1] * 60)
    groups = np.array([f"g{i % 12}" for i in range(120)])
    out = run_control(colour_features(images), labels, groups, seed=0)
    assert 0.3 < out["auc"] < 0.7


def test_the_control_fits_and_tests_on_disjoint_groups():
    # Fitting and reporting on the same rows is a curve that has already
    # seen its own exam, and it would make every control fire on every
    # corpus -- which is indistinguishable from a control that never fires.
    rng = np.random.default_rng(1)
    features = rng.normal(size=(200, 6))
    labels = np.array([0, 1] * 100)
    groups = np.array([f"g{i // 10}" for i in range(200)])
    out = run_control(features, labels, groups, seed=0)
    assert out["n_fit"] > 0 and out["n_test"] > 0
    assert out["n_fit"] + out["n_test"] <= 200


def test_a_control_on_noise_does_not_escape_its_null():
    rng = np.random.default_rng(2)
    features = rng.normal(size=(200, 6))
    labels = np.array([0, 1] * 100)
    groups = np.array([f"g{i // 10}" for i in range(200)])
    out = run_control(features, labels, groups, seed=0)
    assert out["p"] > 0.1
    assert out["null_lo"] <= out["auc"] <= out["null_hi"] or out["p"] > 0.1


def test_the_null_is_refused_on_a_one_class_corpus():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="both labels"):
        run_control(rng.normal(size=(40, 6)), np.zeros(40, dtype=int),
                    np.array([f"g{i // 4}" for i in range(40)]), seed=0)


def test_jpeg_table_features_separate_two_encode_qualities(tmp_path):
    # The statistic that labels 99.2% of this project's capture corpus,
    # read here from the file header rather than from the pixels.
    import cv2
    from bench.dfeval_controls import jpeg_table_features

    rng = np.random.default_rng(0)
    paths = []
    for i, q in enumerate([30] * 10 + [95] * 10):
        img = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        p = tmp_path / f"{i}.jpg"
        cv2.imwrite(str(p), img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        paths.append(p)
    x = jpeg_table_features(paths)
    assert x.shape[0] == 20
    assert not np.allclose(x[:10].mean(axis=0), x[10:].mean(axis=0))


def test_jpeg_table_features_report_nothing_for_a_non_jpeg(tmp_path):
    # A PNG has no quantisation table. Returning zeros would be a feature
    # value; the control must be able to say the statistic does not exist
    # for that row and exclude it.
    import cv2
    from bench.dfeval_controls import jpeg_table_features

    p = tmp_path / "x.png"
    cv2.imwrite(str(p), np.zeros((16, 16, 3), np.uint8))
    x = jpeg_table_features([p])
    assert np.isnan(x[0]).all()
```

- [ ] **Step 2: Run them and watch them fail**

Run: `python3 -m pytest tests/bench/test_dfeval_controls.py -q`
Expected: `ModuleNotFoundError: No module named 'bench.dfeval_controls'`.

- [ ] **Step 3: Implement**

`run_control` splits **by group**, fits `LogisticRegression(max_iter=1000)` on the fit half,
scores the test half, computes AUC with `bootstrap_ci_by_group`, then calls
`bench.metrics.permutation_null` with a `refit_and_score` closure that refits on shuffled fit
labels and re-scores the same test half — refitting, because the docstring is explicit that
permuting a fitted model's scores tests a weaker hypothesis. `permutation_p` gives the two-sided
p against `centre=0.5`. Rows whose features are all-NaN (a non-JPEG under
`jpeg_table_features`) are excluded and counted, never zero-filled.

The module docstring records the two measured precedents that justify the module's existence:
DF40's Lab-colour separation at 0.843, and the capture corpus's 99.2% labelled by one header byte.

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/bench/test_dfeval_controls.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add bench/dfeval_controls.py tests/bench/test_dfeval_controls.py
git commit -m "feat: two controls that decide whether this corpus can measure anything"
```

---

### Task 5: The orchestrator

**Files:**
- Create: `bench/eval_dfeval.py`
- Create: `tests/bench/test_eval_dfeval.py`

**Interfaces:**
- Consumes: `corpora.dfeval.load_dfeval_images` (Task 2), `bench.scorefile.{load_score_file, aligned}`
  (Task 3), `bench.dfeval_controls.{colour_features, jpeg_table_features, run_control}` (Task 4),
  `bench.metrics.{auc, tpr_at_fpr, bootstrap_ci_by_group, kish_ess}` (Task 1),
  `bench.protocol.{logo_splits, UnsplittableCorpusError}`, `dfd.detectors.registry.default_registry`.
- Produces:
  - `WAIVER_HEADING: str = "## Guards waived for this corpus"`
  - `SCOPE_HEADING: str = "## What this corpus can and cannot show"`
  - `PUBLISHED_NPR_AUC: float = 0.53`, `ORACLE_LO: float = 0.40`, `ORACLE_HI: float = 0.75`
  - `def score_detectors(records, registry) -> dict[str, dict[str, Any]]` — per detector:
    `{scores: np.ndarray, abstained: np.ndarray, reasons: list[str]}`, aligned to `records`.
  - `def face_subset(records) -> np.ndarray` — boolean mask of `has_face`.
  - `def abstention_by_class(abstained, labels) -> dict[str, tuple[int, int]]` — keys `"real"`
    and `"fake"`, each `(abstained, total)`. Deliberately simpler than
    `bench.runner._abstention_by_class`, which keys fakes by their generator id: this corpus has
    exactly one generator id, so that breakdown would be one bucket wearing a longer name.
  - `def check_oracle(npr_auc: float) -> tuple[bool, str]`
  - `def logo_verdict(records) -> str` — the sentence that goes where a LOGO table would.
  - `def main(argv=None, *, detect=detect_faces, registry=None) -> int`

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import json

import numpy as np
import pytest

from bench.eval_dfeval import (
    ORACLE_HI,
    ORACLE_LO,
    SCOPE_HEADING,
    WAIVER_HEADING,
    abstention_by_class,
    check_oracle,
    face_subset,
    logo_verdict,
)


def test_the_oracle_passes_in_the_published_neighbourhood():
    ok, msg = check_oracle(0.53)
    assert ok and "0.53" in msg


def test_the_oracle_fails_an_inverted_or_broken_result():
    # Below 0.40 on a corpus where the published number is 0.53 means the
    # labels, the ids or the score direction are wrong -- not that our NPR
    # is a slightly different NPR.
    ok, msg = check_oracle(0.22)
    assert not ok and "0.40" in msg


def test_the_oracle_fails_a_result_that_is_too_good():
    # Above 0.75 where every published open-source model sits at chance is
    # not good news; it is the signature of a label leak or misaligned ids.
    # This is the direction nobody checks, and it is the one that ships.
    ok, msg = check_oracle(0.93)
    assert not ok and "0.75" in msg


def test_the_oracle_boundaries_are_inclusive_and_stated():
    assert check_oracle(ORACLE_LO)[0]
    assert check_oracle(ORACLE_HI)[0]


def test_logo_is_refused_and_the_refusal_is_the_recorded_answer(dfeval_root,
                                                                stub_detect):
    from corpora.dfeval import load_dfeval_images

    records, _ = load_dfeval_images(dfeval_root, detect=stub_detect)
    verdict = logo_verdict(records)
    assert "cannot fold" in verdict.lower()
    assert "generator" in verdict.lower()


def test_face_subset_selects_only_faced_records():
    records = [{"has_face": True}, {"has_face": False}, {"has_face": True}]
    assert list(face_subset(records)) == [True, False, True]


def test_abstention_by_class_separates_reals_from_fakes():
    # One aggregate rate cannot show label-correlated abstention, and
    # label-correlated abstention means every metric above it was computed
    # on a label-selected subsample. Slot A abstained on 74.3% of one
    # generator's fakes against 56.9% of reals while that fold carried its
    # best AUC.
    abstained = np.array([True, False, True, True, False, False])
    labels = np.array([0, 0, 0, 1, 1, 1])
    out = abstention_by_class(abstained, labels)
    assert out["real"] == (2, 3)
    assert out["fake"] == (1, 3)


def test_the_report_carries_the_waiver_and_scope_blocks(tmp_path, dfeval_root,
                                                        stub_detect, stub_registry):
    from bench.eval_dfeval import main

    rc = main(["--root", str(dfeval_root), "--out-json", str(tmp_path / "r.json"),
               "--out-md", str(tmp_path / "r.md"), "--allow-oracle-failure"],
              detect=stub_detect, registry=stub_registry)
    assert rc == 0
    md = (tmp_path / "r.md").read_text()
    assert WAIVER_HEADING in md
    assert SCOPE_HEADING in md
    assert "cannot fold" in md.lower()


def test_a_failing_oracle_stops_every_other_number(tmp_path, dfeval_root,
                                                   stub_detect, inverted_registry):
    # The gate's whole purpose: a harness that cannot reproduce a published
    # result on a published corpus has not been shown to measure anything,
    # so it reports the failure and nothing else.
    from bench.eval_dfeval import main

    rc = main(["--root", str(dfeval_root), "--out-json", str(tmp_path / "r.json"),
               "--out-md", str(tmp_path / "r.md")],
              detect=stub_detect, registry=inverted_registry)
    assert rc != 0
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["oracle"]["passed"] is False
    assert "detectors" not in report


def test_no_single_corpus_wide_auc_is_reported(tmp_path, dfeval_root,
                                               stub_detect, stub_registry):
    # For a face detector on a corpus that is not all faces there is no
    # honest aggregate, and printing one invites it to be quoted.
    from bench.eval_dfeval import main

    main(["--root", str(dfeval_root), "--out-json", str(tmp_path / "r.json"),
          "--out-md", str(tmp_path / "r.md"), "--allow-oracle-failure"],
         detect=stub_detect, registry=stub_registry)
    report = json.loads((tmp_path / "r.json").read_text())
    for entry in report["detectors"].values():
        assert "auc" not in entry
        assert "auc_face_subset" in entry
        assert "n_face_subset" in entry


def test_kish_ess_is_reported_beside_every_n(tmp_path, dfeval_root,
                                             stub_detect, stub_registry):
    from bench.eval_dfeval import main

    main(["--root", str(dfeval_root), "--out-json", str(tmp_path / "r.json"),
          "--out-md", str(tmp_path / "r.md"), "--allow-oracle-failure"],
         detect=stub_detect, registry=stub_registry)
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["corpus"]["kish_ess_real"] > 0
    assert report["corpus"]["kish_ess_fake"] > 0


def test_a_baseline_score_file_is_reported_with_its_coverage(
        tmp_path, dfeval_root, stub_detect, stub_registry):
    from bench.eval_dfeval import main
    from corpora.dfeval import load_dfeval_images

    records, _ = load_dfeval_images(dfeval_root, detect=stub_detect)
    csv = tmp_path / "fake_baseline.csv"
    rows = "\n".join(f"{r['sample_id']},0.5" for r in records[:2])
    csv.write_text("sample_id,score\n" + rows + "\n")
    main(["--root", str(dfeval_root), "--out-json", str(tmp_path / "r.json"),
          "--out-md", str(tmp_path / "r.md"), "--scores", str(csv),
          "--allow-oracle-failure"], detect=stub_detect, registry=stub_registry)
    report = json.loads((tmp_path / "r.json").read_text())
    entry = report["baselines"]["fake_baseline"]
    assert entry["coverage"] < 1.0
    assert entry["covered"] == 2
```

Fixtures: `stub_registry` is a `Registry` holding a `SyntheticDetector` plus a stand-in named
`npr` whose score lands inside the oracle band; `inverted_registry` is the same with an `npr`
landing below `ORACLE_LO`. Both avoid needing weight files.

- [ ] **Step 2: Run them and watch them fail**

Run: `python3 -m pytest tests/bench/test_eval_dfeval.py -q`
Expected: `ModuleNotFoundError: No module named 'bench.eval_dfeval'`.

- [ ] **Step 3: Write the orchestrator**

```python
#: The paper's published AUC for NPR on this corpus's image split.
PUBLISHED_NPR_AUC = 0.53

#: The band inside which our slot C must land for anything else to be
#: reported. NOT a tolerance on a reproduction: our NPR is a
#: reimplementation whose constants are recorded as unverified against Tan
#: et al. (see `dfd/detectors/npr.py`), and the paper scored whole images
#: where we score a face ROI at its native resolution. A decimal match is
#: not expected and demanding one would block the harness over a non-bug.
#:
#: What this band IS for is catching gross errors, which have signatures:
#:
#:   below 0.40  the labels, the ids or the score direction are inverted.
#:               The published number is 0.53; landing under 0.40 is not a
#:               weaker NPR, it is a wired-up-backwards one.
#:   above 0.75  every published open-source model on this corpus sits at
#:               chance (UFD 0.56, DistilDIRE 0.52, NPR 0.53). A handcrafted
#:               27+30-feature head beating all of them by 20 points is the
#:               signature of a label leak or misaligned ids, not a
#:               breakthrough. This is the direction nobody checks, and it
#:               is therefore the one that ships.
#:
#: Between them, the difference is attributable to the ROI and
#: implementation differences named above, and the run proceeds.
ORACLE_LO = 0.40
ORACLE_HI = 0.75
```

`main` executes in this order, and the order is the point:

1. load the corpus (Task 2);
2. score every registered detector (`score_detectors`);
3. **the oracle gate.** Compute slot C's AUC over the face subset and call `check_oracle`. On
   failure, write a report containing **only** the `oracle` block and the corpus composition,
   log the failure, and return non-zero — unless `--allow-oracle-failure`, which mirrors
   `bench/eval_swaps.py`'s `--allow-training-window` and exists so the failure itself can be
   investigated;
4. the corpus block: counts, domains, Kish ESS per class, dedup tally, face-subset size;
5. the two controls (Task 4);
6. per detector: AUC over the face subset with a domain-grouped CI, TPR at 1% and 0.1% FPR,
   abstention by class. **No key named `auc`** — the field is `auc_face_subset`, so a reader
   cannot quote a corpus-wide number that was never computed;
7. baselines from `--scores` (repeatable), each with coverage;
8. `logo_verdict`, which calls `logo_splits`, catches `UnsplittableCorpusError` and returns the
   recorded sentence. A plain `ValueError` from `_validate` is **not** caught — that means a
   malformed corpus and must propagate;
9. the markdown report, leading with the waiver block, then the scope block, then the controls,
   then the detectors.

The scope block is prose and must say, in the style of `bench/eval_swaps.py`'s `_SCOPE`: that
this corpus cannot fold and why; that the headline is a face subset and what fraction of the
corpus it is; that the baselines are our reproductions in their own environments, so a
discrepancy against the paper could be theirs, ours or the environment's; that the corpus is 2024
media and generators have moved; and that nothing here compares this system to any vendor.

The waiver block names each waived guard and why: `check_compression_coverage` (the corpus has
one unknown level — an invented `c23` would pass the guard by lying to it) and
`check_identity_disjoint` (in-the-wild media carries no identity labels and nothing is split here
anyway).

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/bench/test_eval_dfeval.py -q`
Expected: 12 passed.

- [ ] **Step 5: Run the gates**

Run: `ruff check . && mypy --strict src/ && python3 -m pytest -q`

- [ ] **Step 6: Commit**

```bash
git add bench/eval_dfeval.py tests/bench/test_eval_dfeval.py
git commit -m "feat: the harness, gated on reproducing a number it did not choose"
```

---

### Task 6: How a baseline is run, written down

No code. The deliverable is that someone else can reproduce a baseline column, and that the
reason each baseline is outside this repo is on the record rather than in someone's memory.

**Files:**
- Create: `baselines/README.md`
- Create: `baselines/ufd/run.md`
- Create: `baselines/distildire/run.md`
- Modify: `.gitignore` (baseline checkouts, weights and outputs)

- [ ] **Step 1: Write `baselines/README.md`**

It must state:

- **Why these live outside the package.** `tests/test_ci_gates.py` enforces exact `==` pins on
  both CI legs; UFD, DistilDIRE and the DeepfakeBench family each carry their own dependency
  trees, and several obtainable baselines are additionally NonCommercial. Under this protocol
  neither their dependencies nor their weights ever sit in the tree of a product that ships.
- **The contract**: a CSV with header `sample_id,score`, one row per image the baseline scored,
  `sample_id` exactly as `corpora.dfeval` emits it, any monotone score scale (AUC is rank-based),
  partial coverage permitted and reported.
- **The licence posture**: baselines are run as internal evaluation only. Nothing they produce is
  published, and no weights they use enter this repository. This rests on the owner ruling in
  spec §9.3; if that ruling goes the other way, the harness still runs and reports our detectors
  alone.
- **The honesty clause**: our reproduction of a published baseline is *our* reproduction. A
  discrepancy against the paper's number could be theirs, ours, or the environment's, and the
  report says so rather than resolving it by assumption.

- [ ] **Step 2: Write one `run.md` per baseline**

Each records: upstream repository and commit, licence as read at source, the exact environment
(its own venv or container, never this repo's), the weights and where they came from, the command
that produces the CSV, and the number the paper reports for it on this split (UFD 0.56,
DistilDIRE 0.52) so a reproduction can be checked against something.

- [ ] **Step 3: Ignore the artefacts**

```gitignore
# Third-party baselines are run OUTSIDE this repo by design (baselines/README.md).
# Only the run instructions are tracked; checkouts, weights and outputs are not.
baselines/*/checkout/
baselines/*/weights/
baselines/*/*.csv
```

- [ ] **Step 4: Verify the ignore rule does what it says**

Run: `mkdir -p baselines/ufd/weights && touch baselines/ufd/weights/x.pt baselines/ufd/out.csv && git status --short baselines/`
Expected: only the `run.md` and `README.md` files appear. Then `rm -rf baselines/ufd/weights baselines/ufd/out.csv`.

- [ ] **Step 5: Commit**

```bash
git add baselines/ .gitignore
git commit -m "docs: how a baseline is run, and why it is not installed here"
```

---

### Task 7: The run — **blocked on corpus access**

Everything above is built and tested against synthetic fixtures. This task is the only one that
needs the gated download, and it should not be started until `docs/EULA-ACCESS.md` records that
access was granted.

**Files:**
- Modify: `corpora/dfeval.py` (`_manifest_rows`, once the real layout is known)
- Modify: `assets/manifest.yaml`
- Modify: `bench/evidence_card.json`
- Modify: `docs/EULA-ACCESS.md`
- Modify: `docs/HANDOFF.md`

- [ ] **Step 1: Request access and record it**

Request on the HuggingFace gate with the `@scoreme.in` address and a link evidencing
deepfake-detection work — this repository, or the deployed service. Record in
`docs/EULA-ACCESS.md`: the URL, the licence (CC-BY-SA-4.0), the gate's requirements, the date
requested, the date granted, and **the ruling that nothing trains on it** (spec §3.1). Note that
the gate opens to anyone accepting the terms in May 2027, which is the fallback if the request is
refused.

- [ ] **Step 2: Confirm the on-disk layout and fix `_manifest_rows`**

```bash
python3 - <<'PY'
from pathlib import Path
root = Path.home() / "Desktop/agents/datasets/deepfake_eval_2024"
for p in sorted(root.rglob("*"))[:40]:
    print(p.relative_to(root), p.stat().st_size if p.is_file() else "<dir>")
PY
```

Then adjust `_manifest_rows` alone, and re-run `tests/corpora/test_dfeval.py` — the fixtures must
be updated to match the real shape, and every other test must pass **unchanged**. If a change is
needed outside `_manifest_rows`, the loader's layout isolation failed and that is worth saying in
the commit rather than quietly spreading the parsing.

- [ ] **Step 3: Check the corpus against the paper before trusting the loader**

```bash
python3 - <<'PY'
from corpora.dfeval import load_dfeval_images
records, skipped = load_dfeval_images("~/Desktop/agents/datasets/deepfake_eval_2024")
reals = sum(1 for r in records if r["label"] == 0)
print(f"{len(records)} records: {reals} real / {len(records)-reals} fake")
print(f"{len({r['domain'] for r in records})} domains")
print(f"{sum(1 for r in records if r['has_face'])} with a face")
print("skipped:", skipped)
PY
```

Expected, from the paper: **1,975 images, 767 real / 1,208 fake, 88 domains**. A material
mismatch means the loader is wrong or the release differs from the paper, and either must be
chased before any number is produced. The face count is **not** in the paper — it is this
project's own first finding about the corpus, and it sizes the gap §6 of the spec predicts.

- [ ] **Step 4: Run the harness**

```bash
python3 -m bench.eval_dfeval \
    --root ~/Desktop/agents/datasets/deepfake_eval_2024 \
    --out-json bench/dfeval_report.json --out-md bench/dfeval_report.md
```

- [ ] **Step 5: Read it in this order, and stop at the first failure**

1. **Did the oracle pass?** If slot C is outside 0.40–0.75, nothing else is reported and nothing
   else should be read. Chase it.
2. **Did either control fire?** If colour or the JPEG table approaches the commercial models'
   0.90, this corpus has a shortcut and every number on it is suspect — say that first, in the
   evidence card, and treat the rest as provisional.
3. **How large is the face subset, and is abstention label-correlated?** If it is, every AUC
   above it was computed on a label-selected subsample.
4. **Only then** read the detectors' numbers against UFD 0.56, DistilDIRE 0.52, NPR 0.53 and the
   commercial 0.90.

- [ ] **Step 6: Register the corpus**

```yaml
  deepfake_eval_2024:
    source: "https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024 — in-the-wild deepfakes collected in 2024 (arXiv 2503.02857). Image split: 1,975 images, 767 real / 1,208 fake, 88 web domains."
    license: "CC-BY-SA-4.0 — ShareAlike. EVALUATION ONLY: nothing in this project trains on it, now or later (spec 2026-11-10-benchmark-harness-design.md §3.1). Clean for an exam nothing is fitted on; a live question about any weights that were."
    commercial_use: false
    evidence_url: "https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024"
    date_checked: "2026-11-__"
    checked_by: "rohit.kohli@scoreme.in"
```

`commercial_use: false` is deliberate and is not a defect: the asset gate's question is whether a
file may ship, and this corpus may not. It is never redistributed and nothing fitted on it exists.

- [ ] **Step 7: Write the evidence-card corpus entry**

Under `corpus`, with a caveat recording: the composition and domain count; the Kish ESS; the
face-subset fraction; what both controls reached; that LOGO cannot fold here and why; and — if
either control fired or the oracle failed — the sentence that this corpus can neither support nor
refute a detector, which the card already carries for DF40.

**Do not add a `detectors` entry that opens the gate from this corpus.** A face-subset AUC on
in-the-wild 2024 media is not the cross-corpus measurement
`dfd.service.evidence.gated_detectors` is asking for, and the sibling spec's slot D measurement
is. If the harness produces a number worth gating on, that is a separate, argued decision.

- [ ] **Step 8: Write the `docs/HANDOFF.md` §0 block**

Headline first, in the house style: what each detector scored on the face subset, how large that
subset was, what the controls reached, whether the oracle passed, and the comparison to UFD /
DistilDIRE / NPR / commercial. Then what is NOT true: that this says nothing about Reality
Defender or DuckDuckGoose, that LOGO did not fold, and that the corpus is 2024 media.

- [ ] **Step 9: Commit**

```bash
git add corpora/dfeval.py tests/corpora/ bench/dfeval_report.json bench/dfeval_report.md \
        assets/manifest.yaml bench/evidence_card.json docs/EULA-ACCESS.md docs/HANDOFF.md
git commit -m "feat: the first number this project has produced on a corpus the field recognises"
```

Commit it whatever it says. This project has retracted a headline before, and the retraction is
in the history where it belongs.

---

## Self-Review

**Spec coverage.** §1 (the problem) → the plan's goal and Task 5's scope block. §2 in-scope item 1
→ Task 2; item 2 → Task 3; item 3 → Tasks 1, 4, 5; item 4 → Task 5's oracle gate; item 5 → Task 7
steps 6–7. §3 (corpus facts) → Task 7 step 3 checks them rather than assuming them. §3.1 (the
training prohibition and ShareAlike) → Global Constraints, Task 6's licence posture, Task 7's
manifest entry. §4.1 → Task 2, including the `dfeval_unknown_mixture` correction. §4.2 → Task 3.
§4.3 → Task 5. §4.4 → Task 1. §5 conditions 1–3 → Task 5 step 3 item 6 and Task 4; condition 4
(LOGO refused) → Task 2's `test_logo_refuses_this_corpus_gracefully` and Task 5's `logo_verdict`;
condition 5 (null on controls only) → Task 4. §6 (the face subset) → Task 2 keeps no-face records,
Task 5 reports `auc_face_subset` and forbids an `auc` key. §7 (the oracle) → Task 5, with both
caveats in the `ORACLE_LO`/`ORACLE_HI` comment. §8 (refusals) → Task 2 and Task 3, one test each.
§9.1 → Task 5's scope block and Task 7 step 8. §9.2 → the scope block. §9.3 (owner actions) →
Task 7 step 1 and Task 6's licence posture.

**Known gaps, stated rather than hidden:**

1. **`_manifest_rows` is written blind.** The dataset card does not publish the layout and the
   viewer is behind the gate, so Task 2 builds against a fixture of our own shape and Task 7 step
   2 reconciles it. This is isolation, not a guess — but if the real layout carries something the
   design did not anticipate (per-item provenance beyond the domain, say), Task 7 will be larger
   than one function.
2. **The domain field is assumed to be recoverable per item.** The paper reports 88 web-domain
   names across the dataset; it does not state that each image carries its domain in the release.
   If it does not, the resampling family is gone and §5's condition 1 has nothing to group on —
   the fallback would be to resample rows and say plainly that the interval is too narrow, which
   is the DF40 mistake knowingly repeated and is bad enough to be worth escalating rather than
   quietly doing. **Check this in Task 7 step 2, before step 3.**
3. **The oracle band is a judgement.** 0.40–0.75 is argued from the published numbers rather than
   derived, and a reasonable person could pick 0.35–0.80. What is not negotiable is that the band
   is fixed in Task 5, before the real NPR number exists, rather than chosen in Task 7 once it
   does.
4. **Nothing here measures the vendors.** By design (spec §9.1), and it is the plan's largest
   honest limitation against the request that prompted it.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-11-10-benchmark-harness.md`.
