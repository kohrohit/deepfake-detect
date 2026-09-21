# Composition Root Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire one runnable path — a file on disk to an immutable audit record — through ingest, face normalization, detectors, calibration, fusion and audit, behind a `dfd score` CLI.

**Architecture:** A new `pipeline.py` holds `normalize()` and `decide()`; `decide` is the only non-test caller of the engine's stages. A new `Policy` object carries the decision thresholds and is passed to *both* `fuse` and `build_audit_record`, so the threshold a record reports is the one that produced its verdict. `AuditRecord` gains `stage_reasons` (schema `"1"` → `"2"`) so an abstention names the stage that caused it. `cli.py` is a thin argparse layer over `decide`.

**Tech Stack:** Python 3.10, numpy, OpenCV (headless), PyTorch, scikit-learn, pytest, ruff, mypy --strict.

**Spec:** `docs/superpowers/specs/2026-09-21-composition-root-design.md`

## Global Constraints

Copied from the repo's existing gates and conventions. Every task's requirements include these.

- **Python 3.10** — no `match`, no `tomllib`, no 3.11+ syntax.
- **`mypy --strict` must pass** over `src/dfd` (`mypy --config-file mypy.ini`). Tests and `bench/` are NOT type-checked; do not contort test code for mypy.
- **`ruff check src bench corpora` must pass.** Line length and rules per `ruff.toml`.
- **Coverage ≥ 85%** (`pytest -q --cov=src/dfd --cov-fail-under=85`).
- **No `print()` anywhere under `src/`** — `tests/test_ci_gates.py::test_no_print_statements_in_src` fails the build on any line starting with `print(`. **The CLI must write with `sys.stdout.write` / `sys.stderr.write`.** This is not a style preference; it is an enforced gate.
- **No bare `except:`** — `tests/test_ci_gates.py::test_no_bare_except_anywhere_in_src` enforces it.
- **New exceptions subclass `DfdError`** (`src/dfd/errors.py`). Raise `InvalidInput` for bad caller input.
- **Logging, never printing, inside library code**: `logger = logging.getLogger(__name__)`.
- **Google-style docstrings with an explicit `Raises:` section**, matching the existing modules.
- **Every test must be proved by mutation**: break the implementation, watch the test fail for the *right reason*, restore, watch it pass. A test nobody watched fail is a hope, not a guard. Each task below names the mutation to run.
- **Both CI legs must stay green**: `requirements-dev.txt` (ceiling) and `requirements-floor.txt` (floor).

### Backward compatibility, already verified

- No existing test asserts `schema_version == "1"` (only that the field exists), so the schema bump breaks nothing.
- `AuditRecord(...)` is constructed in exactly one place, `src/dfd/audit.py:228`. No positional construction elsewhere.
- Every existing `fuse(...)` call passes `n_frames`/`ess` by keyword, so adding a trailing `policy` parameter breaks nothing.

---

### Task 1: Policy object, and `fuse` applying it

**Files:**
- Create: `src/dfd/policy.py`
- Modify: `src/dfd/fusion.py` (constants block at lines 24–41; `fuse` signature at line 95; verdict block near lines 180–190)
- Test: `tests/test_policy.py` (create), `tests/test_fusion.py` (append)

**Interfaces:**
- Consumes: `dfd.errors.InvalidInput`.
- Produces: `dfd.policy.Policy(fake_threshold: float = 1.0, real_threshold: float = -1.0, disagreement_ood: float = 3.0, version: str = "p0-default-v0")`, frozen dataclass; `dfd.policy.DEFAULT_POLICY: Policy`; `fuse(evidence: list[Evidence], n_frames: int = 1, ess: float | None = None, policy: Policy = DEFAULT_POLICY) -> FusedResult`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_policy.py`:

```python
import math

import pytest

from dfd.errors import InvalidInput
from dfd.policy import DEFAULT_POLICY, Policy


def test_default_policy_matches_the_thresholds_fusion_has_always_applied():
    assert DEFAULT_POLICY.fake_threshold == 1.0
    assert DEFAULT_POLICY.real_threshold == -1.0
    assert DEFAULT_POLICY.disagreement_ood == 3.0
    assert DEFAULT_POLICY.version == "p0-default-v0"


def test_policy_is_frozen():
    """A policy that can be edited after a record cites it makes the record a lie."""
    with pytest.raises(Exception):
        DEFAULT_POLICY.fake_threshold = 2.0  # type: ignore[misc]


def test_real_threshold_must_sit_below_fake_threshold():
    with pytest.raises(InvalidInput, match="real_threshold"):
        Policy(fake_threshold=1.0, real_threshold=1.0)


def test_non_finite_thresholds_are_refused():
    """NaN compares false against everything, so a NaN threshold silently makes
    every verdict INSUFFICIENT_EVIDENCE instead of failing."""
    with pytest.raises(InvalidInput, match="fake_threshold"):
        Policy(fake_threshold=math.nan)


def test_disagreement_trigger_must_be_positive():
    with pytest.raises(InvalidInput, match="disagreement_ood"):
        Policy(disagreement_ood=0.0)


def test_version_must_be_a_non_empty_string():
    """policy_version is what an auditor uses to reconstruct the decision."""
    with pytest.raises(InvalidInput, match="version"):
        Policy(version="")
```

Append to `tests/test_fusion.py`:

```python
from dfd.policy import Policy
from dfd import fusion


def test_reexported_constants_match_the_default_policy():
    """Two homes for one number is how they drift. This is the only thing
    stopping fusion's module constants and Policy's fields diverging."""
    assert fusion.FAKE_THRESHOLD == fusion.DEFAULT_POLICY.fake_threshold
    assert fusion.REAL_THRESHOLD == fusion.DEFAULT_POLICY.real_threshold
    assert fusion.DISAGREEMENT_OOD == fusion.DEFAULT_POLICY.disagreement_ood


def test_a_custom_policy_moves_the_fake_boundary():
    """Evidence that is FAKE under the default must be INSUFFICIENT under a
    stricter policy, or the policy argument is decorative."""
    evidence = [_ev(1.5, "a")]
    assert fuse(evidence, n_frames=1).verdict is Verdict.FAKE
    strict = Policy(fake_threshold=2.0, real_threshold=-2.0)
    assert fuse(evidence, n_frames=1, policy=strict).verdict is Verdict.INSUFFICIENT_EVIDENCE


def test_a_custom_policy_moves_the_real_boundary():
    evidence = [_ev(-1.5, "a")]
    assert fuse(evidence, n_frames=1).verdict is Verdict.REAL
    strict = Policy(fake_threshold=2.0, real_threshold=-2.0)
    assert fuse(evidence, n_frames=1, policy=strict).verdict is Verdict.INSUFFICIENT_EVIDENCE


def test_a_custom_policy_moves_the_disagreement_trigger():
    """Disagreement overrides both thresholds, so it needs its own proof."""
    evidence = [_ev(2.0, "a"), _ev(-2.0, "b")]
    assert fuse(evidence, n_frames=1).verdict is Verdict.INSUFFICIENT_EVIDENCE
    touchy = Policy(disagreement_ood=1.0)
    assert fuse(evidence, n_frames=1, policy=touchy).verdict is Verdict.OUT_OF_DISTRIBUTION
```

Note: `_ev` and `Verdict` already exist at the top of `tests/test_fusion.py`; do not redefine them. Add `DEFAULT_POLICY` to fusion's namespace (Step 3) so `fusion.DEFAULT_POLICY` resolves.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python3 -m pytest tests/test_policy.py tests/test_fusion.py -q
```
Expected: `ModuleNotFoundError: No module named 'dfd.policy'`.

- [ ] **Step 3: Write the implementation**

Create `src/dfd/policy.py`:

```python
"""Decision thresholds as versioned configuration (spec §7.1).

These numbers used to be module constants in `fusion.py`, which meant the audit
record's `threshold` field was a *copy* of what fusion applied rather than the
thing itself. A copy drifts. `decide` hands one `Policy` to both `fuse` and
`build_audit_record`, so a record's stated threshold is provably the one that
produced its verdict.

The §7.1 economics — expected fraud loss against friction cost, and the
AUTO-PASS / STEP-UP / MANUAL REVIEW / BLOCK bands — are deliberately absent.
The loss and friction figures are still unsupplied, and bands fitted to
placeholder economics would be numbers nobody can defend.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import InvalidInput


@dataclass(frozen=True)
class Policy:
    """Thresholds applied to fused evidence, with a version for the record.

    Attributes:
        fake_threshold: llr_total at or above which the verdict is FAKE, in nats.
        real_threshold: llr_total at or below which the verdict is REAL, in nats.
        disagreement_ood: disagreement at or above which the verdict is
            OUT_OF_DISTRIBUTION, overriding both thresholds.
        version: identifier recorded in every audit record this policy decides.
    """

    fake_threshold: float = 1.0
    real_threshold: float = -1.0
    disagreement_ood: float = 3.0
    version: str = "p0-default-v0"

    def __post_init__(self) -> None:
        """Validate the thresholds at construction, not at decision time.

        Raises:
            InvalidInput: if any threshold is non-numeric or non-finite, if
                `real_threshold` is not strictly below `fake_threshold`, if
                `disagreement_ood` is not positive, or if `version` is not a
                non-empty string.
        """
        for name in ("fake_threshold", "real_threshold", "disagreement_ood"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise InvalidInput(
                    f"{name} must be a finite number, got {value!r}. A NaN "
                    "threshold compares false against everything, which turns "
                    "every verdict into INSUFFICIENT_EVIDENCE silently.")
        if self.real_threshold >= self.fake_threshold:
            raise InvalidInput(
                f"real_threshold ({self.real_threshold}) must be strictly below "
                f"fake_threshold ({self.fake_threshold}); otherwise the bands "
                "overlap and the verdict depends on comparison order.")
        if self.disagreement_ood <= 0:
            raise InvalidInput(
                f"disagreement_ood must be positive, got {self.disagreement_ood}. "
                "A non-positive trigger makes every decision OUT_OF_DISTRIBUTION, "
                "since disagreement is a minimum of two non-negative sums.")
        if not isinstance(self.version, str) or not self.version:
            raise InvalidInput(
                f"version must be a non-empty string, got {self.version!r}; it is "
                "what an auditor uses to reconstruct which policy decided a case.")


#: The thresholds this repo has applied since fusion was written.
DEFAULT_POLICY = Policy()
```

Modify `src/dfd/fusion.py`. Replace the three constant definitions (keep `MAX_TOTAL_LLR` where it is — it is a cap on evidence, not a decision threshold) with re-exports, and add the import:

```python
from .policy import DEFAULT_POLICY, Policy

# Re-exported so callers that imported these names before `Policy` existed keep
# working. `tests/test_fusion.py::test_reexported_constants_match_the_default_policy`
# asserts each equals its Policy field — two homes for one number is how they drift.
FAKE_THRESHOLD = DEFAULT_POLICY.fake_threshold
REAL_THRESHOLD = DEFAULT_POLICY.real_threshold
DISAGREEMENT_OOD = DEFAULT_POLICY.disagreement_ood
```

Change the signature:

```python
def fuse(evidence: list[Evidence], n_frames: int = 1, ess: float | None = None,
         policy: Policy = DEFAULT_POLICY) -> FusedResult:
```

Add to the docstring's `Args:` section:

```
        policy: thresholds to apply. Defaults to DEFAULT_POLICY, which holds the
            values this function used as module constants before policies
            existed, so an unchanged caller gets unchanged behaviour. Pass the
            same object to `build_audit_record` so the record's threshold is
            the one applied rather than a copy of it.
```

And in the verdict block, replace the three constants with the policy's fields:

```python
    if disagreement >= policy.disagreement_ood:
        verdict = Verdict.OUT_OF_DISTRIBUTION
        logger.debug("Disagreement %s >= %s; OUT_OF_DISTRIBUTION",
                     disagreement, policy.disagreement_ood)
    elif total >= policy.fake_threshold:
        verdict = Verdict.FAKE
    elif total <= policy.real_threshold:
        verdict = Verdict.REAL
    else:
        verdict = Verdict.INSUFFICIENT_EVIDENCE
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python3 -m pytest tests/test_policy.py tests/test_fusion.py -q
```
Expected: all pass, including the pre-existing fusion tests (unchanged behaviour).

- [ ] **Step 5: Prove the tests can fail (mutation)**

Run each mutation, confirm the named test fails, then restore:

| Mutation | Must fail |
|---|---|
| In `fuse`, change `policy.fake_threshold` back to `FAKE_THRESHOLD` | `test_a_custom_policy_moves_the_fake_boundary` |
| In `fuse`, change `policy.real_threshold` back to `REAL_THRESHOLD` | `test_a_custom_policy_moves_the_real_boundary` |
| In `fuse`, change `policy.disagreement_ood` back to `DISAGREEMENT_OOD` | `test_a_custom_policy_moves_the_disagreement_trigger` |
| Set `FAKE_THRESHOLD = 1.5` in fusion instead of the re-export | `test_reexported_constants_match_the_default_policy` |
| Delete the `real_threshold >= fake_threshold` check | `test_real_threshold_must_sit_below_fake_threshold` |
| Delete the `math.isfinite` check | `test_non_finite_thresholds_are_refused` |

- [ ] **Step 6: Run the gates and commit**

```bash
ruff check src bench corpora && mypy --config-file mypy.ini && python3 -m pytest -q
git add src/dfd/policy.py src/dfd/fusion.py tests/test_policy.py tests/test_fusion.py
git commit -m "feat: thresholds become a Policy object fusion applies

The audit record's threshold field was a copy of fusion's module constant.
A copy drifts from the thing it copies, and a record whose stated threshold
is not the one applied is worse than a record with no threshold at all.
fuse now takes the policy that decides, defaulted so every existing caller
is untouched.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `stage_reasons` on the audit record (schema 2)

**Files:**
- Modify: `src/dfd/audit.py` (`AUDIT_SCHEMA_VERSION` line 30; `AuditRecord` fields lines 109–124; `build_audit_record` signature line 151 and body lines 186–243)
- Test: `tests/test_audit.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces: `AuditRecord.stage_reasons: Mapping[str, str]`; `build_audit_record(..., stage_reasons: Mapping[str, str] | None = None)`; `AUDIT_SCHEMA_VERSION == "2"`.

**Why:** `build_audit_record` drops `Evidence.artifacts` when it builds its rows, and the record has no free-form field. Without this, a record can say `INSUFFICIENT_EVIDENCE` while being unable to say whether the face detector had no weights or the frame had no face — two situations with completely different remedies.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_audit.py`:

```python
def test_stage_reasons_reach_the_record():
    r = _record(stage_reasons={"faces": "weights_absent"})
    assert r.stage_reasons["faces"] == "weights_absent"


def test_stage_reasons_default_to_empty_rather_than_none():
    """Every consumer can then index the mapping without a None check."""
    assert dict(_record().stage_reasons) == {}


def test_stage_reasons_are_frozen():
    r = _record(stage_reasons={"faces": "ok"})
    with pytest.raises(TypeError):
        r.stage_reasons["faces"] = "tampered"  # type: ignore[index]


def test_digest_covers_stage_reasons():
    """A field outside the digest is a field an attacker can rewrite."""
    a = _record(stage_reasons={"faces": "ok"})
    b = _record(stage_reasons={"faces": "weights_absent"})
    assert record_digest(a) != record_digest(b)


def test_stage_reasons_are_serialised():
    payload = json.loads(_record(stage_reasons={"faces": "no_face"}).to_json())
    assert payload["stage_reasons"] == {"faces": "no_face"}


def test_non_string_stage_reason_values_are_refused():
    """A nested dict here would be mutable state inside a frozen record."""
    with pytest.raises(InvalidInput, match="stage_reasons"):
        _record(stage_reasons={"faces": {"nested": "value"}})


def test_non_string_stage_reason_keys_are_refused():
    with pytest.raises(InvalidInput, match="stage_reasons"):
        _record(stage_reasons={1: "ok"})


def test_schema_version_is_two_now_that_the_record_gained_a_field():
    """A consumer parsing a schema-1 record will not find stage_reasons."""
    assert _record().schema_version == "2"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python3 -m pytest tests/test_audit.py -q
```
Expected: `TypeError: build_audit_record() got an unexpected keyword argument 'stage_reasons'`, and the schema test failing with `'1' != '2'`.

- [ ] **Step 3: Write the implementation**

In `src/dfd/audit.py`:

1. Bump the version constant:

```python
# "2" adds stage_reasons. A consumer parsing a "1" record will not find that key.
AUDIT_SCHEMA_VERSION = "2"
```

2. Add the field as the **last** field of `AuditRecord` (after `created_at`), so no positional construction anywhere changes meaning:

```python
    created_at: str
    #: Why a stage could not measure what it was asked to, keyed by stage name,
    #: e.g. {"faces": "weights_absent"}. An abstention without a cause is not an
    #: audit trail: "no face detector weights" and "no face in frame" produce
    #: the same verdict and demand completely different remedies.
    stage_reasons: Mapping[str, str]
```

3. In `build_audit_record`, add the parameter after `model_versions` and before `created_at`:

```python
    model_versions: Mapping[str, str],
    stage_reasons: Mapping[str, str] | None = None,
    created_at: str | None = None,
```

4. Validate it next to the existing `model_versions` validation:

```python
    stage_reasons = {} if stage_reasons is None else stage_reasons
    for key, value in stage_reasons.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise InvalidInput(
                "stage_reasons must map strings to strings, got "
                f"{type(key).__name__} -> {type(value).__name__}. A nested "
                "structure here would be mutable state inside a frozen record.")
    _validate_serialisable(stage_reasons, field="stage_reasons")
```

5. Pass it into the constructor, frozen like `model_versions`:

```python
        model_versions=_freeze(dict(model_versions)),
        created_at=created_at,
        stage_reasons=_freeze(dict(stage_reasons)),
```

6. Extend the docstring's `Raises:` section: `... if `stage_reasons` is not a string-to-string mapping;`.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python3 -m pytest tests/test_audit.py -q
```
Expected: all pass, including every pre-existing audit test unchanged.

- [ ] **Step 5: Prove the tests can fail (mutation)**

| Mutation | Must fail |
|---|---|
| Omit `stage_reasons` from `to_json`'s payload (hard-code the field list) | `test_digest_covers_stage_reasons`, `test_stage_reasons_are_serialised` |
| Pass `dict(stage_reasons)` instead of `_freeze(dict(stage_reasons))` | `test_stage_reasons_are_frozen` |
| Delete the key/value `isinstance` loop | `test_non_string_stage_reason_values_are_refused` |
| Leave `AUDIT_SCHEMA_VERSION = "1"` | `test_schema_version_is_two_now_that_the_record_gained_a_field` |

Note on the first mutation: `to_json` builds its payload from `fields(self)`, so the field is picked up automatically — to mutate, replace that comprehension with an explicit list omitting `stage_reasons`.

- [ ] **Step 6: Run the gates and commit**

```bash
ruff check src bench corpora && mypy --config-file mypy.ini && python3 -m pytest -q
git add src/dfd/audit.py tests/test_audit.py
git commit -m "feat: audit records name the stage that could not measure (schema 2)

build_audit_record drops Evidence.artifacts, and the record had no free-form
field, so a record could say INSUFFICIENT_EVIDENCE without being able to say
whether the face detector had no weights or the frame had no face. Those
demand different remedies. stage_reasons follows model_versions exactly, so
it inherits the freeze, the validation and the digest rather than inventing
any of them.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `normalize` — faces and quality onto observations

**Files:**
- Create: `src/dfd/pipeline.py`
- Test: `tests/test_pipeline.py` (create)

**Interfaces:**
- Consumes: `dfd.faces.detect_faces`, `dfd.faces.FaceBox`, `dfd.faces.DEFAULT_MODEL`, `dfd.quality.measure_quality`, `dfd.types.{Observation, Sample, QUALITY_BANDS}`.
- Produces:
  - `FaceDetectFn = Callable[[npt.NDArray[np.uint8], str | Path], tuple[list[FaceBox], str]]`
  - `normalize(sample: Sample, *, detect: FaceDetectFn = _detect_with_reason, face_model: str | Path = DEFAULT_MODEL) -> tuple[Sample, dict[str, str]]`
  - `_worst_band(observations: Sequence[Observation]) -> str`
  - Constants `UNMEASURED = "unmeasured"`, `NO_FACE = "no_face"`, `DEGENERATE_BOX = "degenerate_box"`, `NO_OBSERVATIONS = "no_observations"`, `MIXED = "mixed"`.

**Why a plain callable rather than a Protocol:** `detect_faces` is overloaded on a `Literal[True]/[False]` keyword. A Protocol reproducing those overloads is brittle under `mypy --strict` for no gain, so the seam is a two-argument callable and `_detect_with_reason` adapts the real function to it.

**Watch out:** `measure_quality` slices `frame[y:y+h, x:x+w]` with no clamping. A YuNet box can start at a negative coordinate or run past the frame edge; a negative start silently slices from the far end, and an empty crop makes `cv2.cvtColor` raise. The ROI **must** be clamped before measuring. `faces.align` already clamps — this mirrors it.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline.py`:

```python
from dataclasses import dataclass

import numpy as np
import pytest

from dfd.faces import FaceBox
from dfd.pipeline import (DEGENERATE_BOX, NO_FACE, UNMEASURED, _worst_band,
                          normalize)
from dfd.types import Context, Modality, Observation, Quality, Sample


def _noise(size=256, seed=0):
    """High-variance image: a flat fill has blur_var ~0 and always bands 'reject'."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (size, size, 3), dtype=np.uint8)


def _box(x=10, y=10, w=200, h=200, iod=100.0):
    lm = np.array([[x + 20.0, y + 60.0], [x + 20.0 + iod, y + 60.0],
                   [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    return FaceBox(x=x, y=y, w=w, h=h, landmarks=lm, score=0.99)


def _sample(frames=1, seed=0):
    obs = tuple(
        Observation(t=float(i), payload=_noise(seed=seed + i), roi=None,
                    quality=None, source_id="s1")
        for i in range(frames))
    return Sample(sample_id="s1", modality=Modality.IMAGE, observations=obs,
                  context=Context())


def _detector(boxes, reason="ok"):
    return lambda frame, model_path: (list(boxes), reason)


def test_absent_face_weights_are_recorded_not_swallowed():
    """This is the state of the repo today: the YuNet ONNX is gitignored."""
    s, reasons = normalize(_sample(), detect=_detector([], reason="weights_absent"))
    assert reasons["faces"] == "weights_absent"
    assert s.observations[0].quality is None


def test_a_frame_with_no_face_is_distinguished_from_absent_weights():
    """Same verdict, completely different remedy."""
    _, reasons = normalize(_sample(), detect=_detector([], reason="ok"))
    assert reasons["faces"] == NO_FACE


def test_quality_is_attached_when_a_face_is_found():
    s, reasons = normalize(_sample(), detect=_detector([_box()]))
    assert reasons["faces"] == "ok"
    q = s.observations[0].quality
    assert q is not None and q.band in ("low", "medium", "high")
    assert s.observations[0].roi == (10, 10, 200, 200)


def test_inter_ocular_distance_comes_from_the_landmarks():
    s, _ = normalize(_sample(), detect=_detector([_box(iod=100.0)]))
    assert s.observations[0].quality.inter_ocular_px == pytest.approx(100.0)


def test_the_largest_face_is_the_one_measured():
    """v-CIP is single-subject; the subject is the big face, not the bystander."""
    small, large = _box(x=0, y=0, w=20, h=20), _box(x=30, y=30, w=180, h=180)
    s, _ = normalize(_sample(), detect=_detector([small, large]))
    assert s.observations[0].roi == (30, 30, 180, 180)


def test_a_box_running_past_the_frame_edge_is_clamped_not_crashed():
    """measure_quality slices without clamping; a negative start silently
    slices from the far end and an empty crop raises inside OpenCV."""
    s, reasons = normalize(_sample(), detect=_detector([_box(x=200, y=200, w=400, h=400)]))
    assert reasons["faces"] == "ok"
    x, y, w, h = s.observations[0].roi
    assert x + w <= 256 and y + h <= 256


def test_a_box_entirely_outside_the_frame_is_reported_not_measured():
    s, reasons = normalize(_sample(), detect=_detector([_box(x=300, y=300, w=50, h=50)]))
    assert reasons["faces"] == DEGENERATE_BOX
    assert s.observations[0].quality is None


def test_face_counts_are_recorded_so_a_crowd_is_visible():
    _, reasons = normalize(_sample(frames=2), detect=_detector([_box(), _box(x=40)]))
    assert reasons["frames_with_face"] == "2/2"
    assert reasons["max_faces_in_frame"] == "2"


def test_mixed_outcomes_across_frames_are_not_reported_as_success():
    calls = {"n": 0}

    def flaky(frame, model_path):
        calls["n"] += 1
        return ([], "weights_absent") if calls["n"] == 1 else ([], "ok")

    _, reasons = normalize(_sample(frames=2), detect=flaky)
    assert reasons["faces"] == "mixed"


def test_worst_band_is_worst_not_first_and_not_best():
    """Calibration conditions on the regime that held for the whole sample."""
    def obs(band):
        return Observation(t=0.0, payload=_noise(), roi=None, source_id="s",
                           quality=Quality(inter_ocular_px=100.0, blur_var=200.0,
                                           yaw_deg=0.0, pitch_deg=0.0,
                                           exposure=0.5, band=band))
    assert _worst_band([obs("high"), obs("low")]) == "low"
    assert _worst_band([obs("low"), obs("high")]) == "low"
    assert _worst_band([obs("medium"), obs("reject"), obs("high")]) == "reject"


def test_worst_band_of_nothing_measured_is_unmeasured():
    assert _worst_band(_sample().observations) == UNMEASURED
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python3 -m pytest tests/test_pipeline.py -q
```
Expected: `ModuleNotFoundError: No module named 'dfd.pipeline'`.

- [ ] **Step 3: Write the implementation**

Create `src/dfd/pipeline.py` with the module docstring and the normalize half (`decide` arrives in Task 4):

```python
"""The composition root: one file in, one audit record out (spec §5.1, §7.2).

Every other module in this package is a stage. This is the only place they are
wired together, and the only non-test caller of ingest, faces, quality,
calibration, fusion and audit.

What this path does TODAY is abstain, three times over: the YuNet weights are
absent, the detector weights are absent, and nothing has fitted a calibration
curve. Each cause is recorded separately — `stage_reasons` for the face stage,
the per-detector `reason` for the rest — because "insufficient evidence"
without a cause is not an audit trail.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .faces import DEFAULT_MODEL, FaceBox, detect_faces
from .quality import measure_quality
from .types import QUALITY_BANDS, Observation, Sample

logger = logging.getLogger(__name__)

#: The face stage's seam, kept a plain two-argument callable: `detect_faces` is
#: overloaded on a Literal keyword, and a Protocol reproducing those overloads
#: is brittle under mypy --strict for no gain. `_detect_with_reason` adapts it.
FaceDetectFn = Callable[[npt.NDArray[np.uint8], "str | Path"],
                        "tuple[list[FaceBox], str]"]

#: No observation carried a measured quality, so no band can be named. Not a
#: member of QUALITY_BANDS: it is the absence of a band, not a bad one.
UNMEASURED = "unmeasured"
#: The detector ran and found nothing. Distinct from `weights_absent`, which
#: means it never ran at all.
NO_FACE = "no_face"
#: A box that does not intersect the frame. Reported rather than measured,
#: because a clamped empty crop would make OpenCV raise.
DEGENERATE_BOX = "degenerate_box"
NO_OBSERVATIONS = "no_observations"
MIXED = "mixed"
OK = "ok"


def _detect_with_reason(frame: npt.NDArray[np.uint8],
                        model_path: str | Path) -> tuple[list[FaceBox], str]:
    """Adapt `detect_faces` to the `FaceDetectFn` seam."""
    return detect_faces(frame, model_path, with_reason=True)


def _clamp_roi(shape: tuple[int, ...], box: FaceBox) -> tuple[int, int, int, int] | None:
    """Clamp a detection to the frame, or None if it does not intersect it.

    `measure_quality` slices `frame[y:y + h, x:x + w]` with no clamping, so a
    negative origin would silently slice from the far end of the array and an
    empty crop makes `cv2.cvtColor` raise. `faces.align` clamps for the same
    reason; this is that rule applied one stage earlier.
    """
    height, width = shape[:2]
    x0, y0 = max(0, box.x), max(0, box.y)
    x1, y1 = min(width, box.x + box.w), min(height, box.y + box.h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def _worst_band(observations: Sequence[Observation]) -> str:
    """The worst measured band across observations, or UNMEASURED.

    Worst by position in `types.QUALITY_BANDS` (`reject < low < medium < high`).
    Calibration conditions on the regime that held for the whole sample: the
    mean of `high` and `reject` is a band the sample never occupied, and the
    best band would calibrate a blurry sample as though it were sharp.
    """
    bands = [o.quality.band for o in observations if o.quality is not None]
    if not bands:
        return UNMEASURED
    return min(bands, key=QUALITY_BANDS.index)


def _aggregate(reasons: list[str]) -> str:
    """One sample-level reason from per-frame reasons."""
    if not reasons:
        return NO_OBSERVATIONS
    if OK in reasons:
        return OK
    unique = set(reasons)
    return reasons[0] if len(unique) == 1 else MIXED


def normalize(sample: Sample, *, detect: FaceDetectFn = _detect_with_reason,
              face_model: str | Path = DEFAULT_MODEL) -> tuple[Sample, dict[str, str]]:
    """Attach a face ROI and a quality measurement to every observation.

    Ingest deliberately emits `roi=None, quality=None` — decoding and analysis
    are separate stages. This is the analysis stage, and until it existed the
    only code performing it was the benchmark runner, which fabricated
    landmarks at 35% and 65% of frame width rather than detecting them.

    Args:
        sample: an ingested sample whose observations carry no quality.
        detect: face detection seam; must return (boxes, reason).
        face_model: path passed to `detect`.

    Returns:
        A pair of (sample with normalized observations, stage reasons). The
        reasons carry `faces` (`ok`, `no_face`, `weights_absent`,
        `degenerate_box`, `mixed` or `no_observations`), `frames_with_face`
        as "n/total", and `max_faces_in_frame`.

    Raises:
        No exceptions of its own; `detect` may raise (a corrupt model file
        makes OpenCV raise, which is deliberately not caught — treating
        corruption as absence would hide a deployment failure).
    """
    out: list[Observation] = []
    reasons: list[str] = []
    counts: list[int] = []

    for obs in sample.observations:
        boxes, reason = detect(obs.payload, face_model)
        counts.append(len(boxes))
        if not boxes:
            reasons.append(NO_FACE if reason == OK else reason)
            out.append(obs)
            continue
        box = max(boxes, key=lambda b: b.w * b.h)
        roi = _clamp_roi(obs.payload.shape, box)
        if roi is None:
            logger.warning("face box %s does not intersect the frame; not measured",
                           (box.x, box.y, box.w, box.h))
            reasons.append(DEGENERATE_BOX)
            out.append(obs)
            continue
        quality = measure_quality(obs.payload, roi, box.landmarks[:2])
        out.append(Observation(t=obs.t, payload=obs.payload, roi=roi,
                               quality=quality, source_id=obs.source_id))
        reasons.append(OK)

    with_face = sum(1 for r in reasons if r == OK)
    stage_reasons = {
        "faces": _aggregate(reasons),
        "frames_with_face": f"{with_face}/{len(sample.observations)}",
        "max_faces_in_frame": str(max(counts) if counts else 0),
    }
    logger.debug("normalized %d observations: %s", len(out), stage_reasons)
    return (Sample(sample_id=sample.sample_id, modality=sample.modality,
                   observations=tuple(out), context=sample.context),
            stage_reasons)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python3 -m pytest tests/test_pipeline.py -q
```

- [ ] **Step 5: Prove the tests can fail (mutation)**

| Mutation | Must fail |
|---|---|
| `reasons.append(reason)` without the `NO_FACE` translation | `test_a_frame_with_no_face_is_distinguished_from_absent_weights` |
| `box = boxes[0]` instead of `max(..., key=area)` | `test_the_largest_face_is_the_one_measured` |
| Return `(box.x, box.y, box.w, box.h)` from `_clamp_roi` unclamped | `test_a_box_running_past_the_frame_edge_is_clamped_not_crashed` (expect an OpenCV error or an out-of-frame ROI) |
| `_worst_band` using `max` instead of `min` | `test_worst_band_is_worst_not_first_and_not_best` |
| `_worst_band` returning `bands[0]` | same test (the second assertion) |
| `_aggregate` returning `reasons[0]` unconditionally | `test_mixed_outcomes_across_frames_are_not_reported_as_success` |

- [ ] **Step 6: Run the gates and commit**

```bash
ruff check src bench corpora && mypy --config-file mypy.ini && python3 -m pytest -q
git add src/dfd/pipeline.py tests/test_pipeline.py
git commit -m "feat: normalize attaches real faces and quality to observations

Ingest emits roi=None, quality=None by design, and until now the only code
that filled them in was the benchmark runner, which fabricated landmarks at
35% and 65% of frame width. normalize detects, picks the largest box, clamps
it to the frame (measure_quality slices without clamping, so an off-edge box
either reads from the far end of the array or makes OpenCV raise) and
records why it could not measure when it could not.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `decide` — the composition root

**Files:**
- Modify: `src/dfd/pipeline.py` (append)
- Test: `tests/test_pipeline.py` (append)

**Interfaces:**
- Consumes: Task 1's `Policy`/`DEFAULT_POLICY`, Task 2's `stage_reasons` parameter, Task 3's `normalize`/`_worst_band`.
- Produces:

```python
def decide(
    path: str | Path,
    *,
    registry: Registry,
    calibrators: Mapping[str, Calibrator] | None = None,
    policy: Policy = DEFAULT_POLICY,
    context: Context | None = None,
    limits: Limits = DEFAULT_LIMITS,
    detect: FaceDetectFn = _detect_with_reason,
    face_model: str | Path = DEFAULT_MODEL,
    max_frames: int = DEFAULT_MAX_FRAMES,
    seed: int = 0,
    created_at: str | None = None,
) -> AuditRecord
```

Also `IMAGE_SUFFIXES`, `VIDEO_SUFFIXES` (frozensets of lowercase suffixes).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
import hashlib
import json
from pathlib import Path

import cv2

from dfd.calibration import Calibrator
from dfd.detectors.base import Registry, SyntheticDetector
from dfd.errors import InvalidInput, ResourceLimitExceeded
from dfd.limits import Limits
from dfd.pipeline import decide
from dfd.policy import Policy
from dfd.types import RawScore, Verdict

FIXED_TIME = "2026-09-21T10:00:00+00:00"


@pytest.fixture
def png(tmp_path):
    p = tmp_path / "subject.png"
    cv2.imwrite(str(p), _noise())
    return p


@dataclass(frozen=True)
class _FixedDetector:
    """Returns a chosen score without needing weights or quality."""
    name: str = "fixed"
    version: str = "test-1"
    modalities: frozenset = frozenset({Modality.IMAGE})
    min_quality_band: str = "low"
    value: float = 0.99

    def score(self, obs):
        return RawScore(detector=self.name, version=self.version, score=self.value,
                        abstained=False, reason="ok")


def _registry(*detectors):
    r = Registry()
    for d in detectors:
        r.register(d)
    return r


def _fitted_calibrator(name="fixed", band="high"):
    """Separable training data so a 0.99 score earns a strongly positive llr."""
    scores = [0.95 + 0.001 * i for i in range(20)] + [0.01 * i for i in range(20)]
    labels = [1] * 20 + [0] * 20
    return Calibrator(name).fit(scores, labels, [band] * 40)


def test_the_path_runs_end_to_end_and_abstains_for_stated_reasons(png):
    """Today's real behaviour: no face weights, no detector weights, no
    calibration. The value is that the record names all three separately."""
    record = decide(png, registry=_registry(SyntheticDetector(name="synthetic")),
                    detect=_detector([], reason="weights_absent"),
                    created_at=FIXED_TIME)
    assert record.verdict == Verdict.INSUFFICIENT_EVIDENCE.value
    assert record.stage_reasons["faces"] == "weights_absent"
    assert record.quality_band == UNMEASURED
    assert [e["reason"] for e in record.evidence] != []


def test_the_path_can_actually_reach_a_verdict(png):
    """THE load-bearing test. Every other test here passes on a pipeline that
    always abstains; this is the only one that does not. Without it, 'correctly
    abstaining' and 'broken in a way abstention hides' are indistinguishable."""
    record = decide(
        png,
        registry=_registry(_FixedDetector()),
        calibrators={"fixed": _fitted_calibrator()},
        detect=_detector([_box()]),
        created_at=FIXED_TIME,
    )
    assert record.quality_band == "high", "the injected face must band high"
    assert record.verdict == Verdict.FAKE.value
    assert record.llr_total > 1.0
    assert record.evidence[0]["abstained"] is False


def test_input_sha256_is_the_hash_of_the_file(png):
    expected = hashlib.sha256(png.read_bytes()).hexdigest()
    record = decide(png, registry=_registry(SyntheticDetector(name="s")),
                    detect=_detector([]), created_at=FIXED_TIME)
    assert record.input_sha256 == expected


def test_the_recorded_threshold_is_the_one_applied(png):
    strict = Policy(fake_threshold=4.0, real_threshold=-4.0, version="strict-v1")
    record = decide(png, registry=_registry(SyntheticDetector(name="s")),
                    detect=_detector([]), policy=strict, created_at=FIXED_TIME)
    assert record.threshold == 4.0
    assert record.policy_version == "strict-v1"


def test_an_unknown_extension_is_refused_by_name(png):
    other = png.with_suffix(".xyz")
    other.write_bytes(png.read_bytes())
    with pytest.raises(InvalidInput, match=r"\.xyz"):
        decide(other, registry=_registry(SyntheticDetector(name="s")))


def test_a_missing_file_is_a_dfd_error_not_an_oserror(tmp_path):
    with pytest.raises(InvalidInput, match="missing"):
        decide(tmp_path / "missing.png", registry=_registry(SyntheticDetector(name="s")))


def test_an_undecodable_file_is_invalid_input_not_a_bare_valueerror(tmp_path):
    p = tmp_path / "broken.png"
    p.write_bytes(b"not an image")
    with pytest.raises(InvalidInput, match="decode"):
        decide(p, registry=_registry(SyntheticDetector(name="s")))


def test_the_decode_bomb_defence_is_reachable_through_decide(png):
    """Task 21's limits were exercised by their own unit tests and nothing
    else. This is the caller that makes them real."""
    with pytest.raises(ResourceLimitExceeded):
        decide(png, registry=_registry(SyntheticDetector(name="s")),
               limits=Limits(max_pixels=16))


def test_model_versions_name_every_registered_detector(png):
    record = decide(png, registry=_registry(SyntheticDetector(name="a"),
                                            SyntheticDetector(name="b")),
                    detect=_detector([]), created_at=FIXED_TIME)
    assert set(record.model_versions) == {"a", "b"}


def test_the_record_is_json_serialisable_end_to_end(png):
    record = decide(png, registry=_registry(SyntheticDetector(name="s")),
                    detect=_detector([]), created_at=FIXED_TIME)
    assert json.loads(record.to_json())["schema_version"] == "2"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python3 -m pytest tests/test_pipeline.py -q
```
Expected: `ImportError: cannot import name 'decide' from 'dfd.pipeline'`.

If `test_the_path_can_actually_reach_a_verdict` fails on the band assertion once `decide` exists, the noise fixture is not producing a `high` band — check `iod >= 96`, `blur_var >= 100`, `0.15 <= exposure <= 0.90` against `dfd.quality`'s thresholds and adjust the fixture, **not** the assertion.

- [ ] **Step 3: Write the implementation**

Append to `src/dfd/pipeline.py` (and extend the import block at the top):

```python
import hashlib
import time
from collections.abc import Mapping

from .audit import AuditRecord, build_audit_record
from .calibration import Calibrator
from .detectors.base import Registry
from .errors import InvalidInput
from .fusion import fuse
from .ingest.image import load_image
from .ingest.video import DEFAULT_MAX_FRAMES, load_video
from .limits import DEFAULT_LIMITS, Limits
from .policy import DEFAULT_POLICY, Policy
from .types import Context, Evidence

#: Extensions routed to each ingest adapter. An unlisted extension is refused
#: rather than guessed: `load_image` on a video returns the first frame with no
#: indication that the rest of the file was ignored.
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})


def _sha256(path: Path) -> str:
    """Hash the file in chunks, before anything decodes it.

    Raises:
        InvalidInput: if the file cannot be read. Without this translation a
            missing path raises OSError, which the CLI would report as an
            unexpected failure rather than as bad input.
    """
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise InvalidInput(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def _ingest(path: Path, context: Context, limits: Limits, max_frames: int,
            seed: int) -> Sample:
    """Route to an ingest adapter by extension.

    Raises:
        InvalidInput: if the extension is not one this package ingests, or if
            the adapter reports the file is undecodable.
        ResourceLimitExceeded: propagated from the adapters' header-first
            checks, deliberately untouched.
    """
    suffix = path.suffix.lower()
    if suffix not in IMAGE_SUFFIXES and suffix not in VIDEO_SUFFIXES:
        raise InvalidInput(
            f"unsupported file extension {suffix!r} for {path.name}; "
            f"images: {sorted(IMAGE_SUFFIXES)}, videos: {sorted(VIDEO_SUFFIXES)}")
    try:
        if suffix in IMAGE_SUFFIXES:
            return load_image(path, context, limits)
        return load_video(path, context, max_frames, seed, limits)
    except ValueError as exc:
        # Both adapters document a bare ValueError for an undecodable file or a
        # zero-frame video — one of errors.py's 19 un-migrated raise sites,
        # translated here at the boundary that needs it. Scoped to the adapter
        # call alone, NOT wrapped around the rest of the pipeline: a blanket
        # except ValueError would relabel genuine bugs as bad input.
        raise InvalidInput(f"could not decode {path.name}: {exc}") from exc


def decide(
    path: str | Path,
    *,
    registry: Registry,
    calibrators: Mapping[str, Calibrator] | None = None,
    policy: Policy = DEFAULT_POLICY,
    context: Context | None = None,
    limits: Limits = DEFAULT_LIMITS,
    detect: FaceDetectFn = _detect_with_reason,
    face_model: str | Path = DEFAULT_MODEL,
    max_frames: int = DEFAULT_MAX_FRAMES,
    seed: int = 0,
    created_at: str | None = None,
) -> AuditRecord:
    """Score one file and return its immutable audit record.

    The stages: hash, ingest (limits enforced from the header, before decode),
    normalize (faces and quality), score every registered detector, calibrate
    each raw score on the sample's worst measured band, fuse under `policy`,
    and record — with the same `policy` object, so the record's threshold is
    the one applied rather than a copy of it.

    `n_frames=1` is passed to `fuse` because every detector in this repo
    aggregates internally; passing the frame count would apply the ESS
    discount to already-aggregated evidence, which `fuse` documents as misuse.

    `ood_score` carries `FusedResult.disagreement`. P0 has no Mahalanobis or
    energy OOD head, and disagreement is the only OOD-shaped quantity that
    exists; the field name overstates what it holds. Revisit when P1 lands the
    real head.

    Args:
        path: file to score.
        registry: detectors to consult.
        calibrators: fitted calibrators by detector name. A detector with no
            entry calibrates through an unfitted `Calibrator`, which returns
            llr 0.0 and `uncalibrated_for_band` — the honest answer to "I was
            never calibrated in this regime".
        policy: thresholds to apply and to record.
        context: sample metadata; defaults to an empty `Context`.
        limits: decode limits, enforced before allocation.
        detect: face detection seam.
        face_model: path passed to `detect`.
        max_frames: frame cap for video ingest.
        seed: frame-selection seed for video ingest.
        created_at: ISO-8601 timestamp; injectable so two records describing
            the same decision are genuinely identical.

    Returns:
        An `AuditRecord`. A refusal produces no record: a refused input is not
        a decision.

    Raises:
        InvalidInput: unreadable file, unsupported extension, undecodable file.
        ResourceLimitExceeded: the input exceeds a decode limit.
    """
    file_path = Path(path)
    input_sha256 = _sha256(file_path)
    sample = _ingest(file_path, context or Context(), limits, max_frames, seed)
    sample, stage_reasons = normalize(sample, detect=detect, face_model=face_model)
    band = _worst_band(sample.observations)

    fitted = dict(calibrators or {})
    evidence: list[Evidence] = []
    model_versions: dict[str, str] = {}
    for name in registry.names():
        detector = registry.get(name)
        started = time.perf_counter()
        raw = detector.score(sample.observations)
        logger.debug("detector %s scored in %.1f ms", name,
                     (time.perf_counter() - started) * 1000.0)
        model_versions[name] = detector.version
        calibrator = fitted.get(name) or Calibrator(name)
        evidence.append(calibrator.to_evidence(raw, band))

    fused = fuse(evidence, n_frames=1, policy=policy)
    logger.info("decided %s: verdict=%s band=%s contributing=%d",
                sample.sample_id, fused.verdict.value, band, fused.n_contributing)
    return build_audit_record(
        sample_id=sample.sample_id,
        input_sha256=input_sha256,
        verdict=fused.verdict,
        llr_total=fused.llr_total,
        posterior=fused.posterior,
        evidence=evidence,
        quality_band=band,
        ood_score=fused.disagreement,
        policy_version=policy.version,
        threshold=policy.fake_threshold,
        model_versions=model_versions,
        stage_reasons=stage_reasons,
        created_at=created_at,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python3 -m pytest tests/test_pipeline.py -q
```

- [ ] **Step 5: Prove the tests can fail (mutation)**

| Mutation | Must fail |
|---|---|
| `threshold=1.0` hard-coded instead of `policy.fake_threshold` | `test_the_recorded_threshold_is_the_one_applied` |
| `policy` not passed to `fuse` | same test's verdict half — add an assertion there if it survives |
| `_sha256` hashing `str(path).encode()` instead of the bytes | `test_input_sha256_is_the_hash_of_the_file` |
| Drop the `except OSError` translation in `_sha256` | `test_a_missing_file_is_a_dfd_error_not_an_oserror` |
| Drop the `except ValueError` translation in `_ingest` | `test_an_undecodable_file_is_invalid_input_not_a_bare_valueerror` |
| Route unknown suffixes to `load_image` instead of raising | `test_an_unknown_extension_is_refused_by_name` |
| Pass `DEFAULT_LIMITS` instead of the `limits` argument | `test_the_decode_bomb_defence_is_reachable_through_decide` |
| Pass `Calibrator(name)` unconditionally, ignoring `calibrators` | `test_the_path_can_actually_reach_a_verdict` |
| `stage_reasons` not passed to `build_audit_record` | `test_the_path_runs_end_to_end_and_abstains_for_stated_reasons` |

- [ ] **Step 6: Run the gates and commit**

```bash
ruff check src bench corpora && mypy --config-file mypy.ini && python3 -m pytest -q --cov=src/dfd --cov-fail-under=85
git add src/dfd/pipeline.py tests/test_pipeline.py
git commit -m "feat: decide() wires the engine into one path, file to record

fuse, Calibrator.to_evidence, build_audit_record, load_image/load_video and
detect_faces had no non-test callers, so the stages had never been shown to
fit. decide runs them in sequence and records why each one could not
conclude. It also gives Task 21's decode-bomb defences their first caller
outside their own unit tests.

The path abstains today — no face weights, no detector weights, no fitted
calibration — so one test exists solely to prove it can still reach a
verdict, since every other test here passes on a pipeline that always
abstains.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: The `dfd score` CLI

**Files:**
- Create: `src/dfd/cli.py`, `src/dfd/__main__.py`
- Modify: `src/dfd/detectors/registry.py` (add `default_registry`), `pyproject.toml` (add `[project.scripts]`)
- Test: `tests/test_cli.py` (create)

**Interfaces:**
- Consumes: `decide` from Task 4.
- Produces: `dfd.cli.main(argv: Sequence[str] | None = None) -> int`; `dfd.detectors.registry.default_registry(npr_weights: str | Path = DEFAULT_NPR_WEIGHTS, effnet_weights: str | Path = DEFAULT_EFFNET_WEIGHTS) -> Registry`.

**Enforced constraint:** no `print(` anywhere under `src/`. Use `sys.stdout.write` / `sys.stderr.write`. `tests/test_ci_gates.py::test_no_print_statements_in_src` will fail the build otherwise.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli.py`:

```python
import json
import subprocess
import sys

import cv2
import numpy as np
import pytest

from dfd.cli import main


@pytest.fixture
def png(tmp_path):
    p = tmp_path / "subject.png"
    rng = np.random.default_rng(0)
    cv2.imwrite(str(p), rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    return p


def test_a_decision_exits_zero_even_when_it_abstains(png, capsys):
    """Exit status says whether the tool ran, never what the verdict was."""
    assert main(["score", str(png)]) == 0


def test_stdout_is_exactly_one_json_object(png, capsys):
    main(["score", str(png)])
    out = capsys.readouterr().out
    record = json.loads(out)
    assert record["schema_version"] == "2"
    assert out.count("\n") == 1, "stdout must carry the record and nothing else"


def test_the_human_summary_goes_to_stderr_not_stdout(png, capsys):
    main(["score", str(png)])
    captured = capsys.readouterr()
    assert "verdict=" in captured.err
    assert "verdict=" not in captured.out


def test_the_summary_names_the_face_stage(png, capsys):
    main(["score", str(png)])
    assert "faces=weights_absent" in capsys.readouterr().err


def test_a_missing_file_exits_two_with_a_message_and_no_traceback(tmp_path, capsys):
    code = main(["score", str(tmp_path / "nope.png")])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "Traceback" not in captured.err and captured.err.startswith("dfd:")


def test_an_unsupported_extension_exits_two(tmp_path, capsys):
    p = tmp_path / "a.xyz"
    p.write_bytes(b"x")
    assert main(["score", str(p)]) == 2


def test_pretty_output_is_the_same_record_reformatted(png, capsys):
    main(["score", str(png), "--pretty"])
    pretty = capsys.readouterr().out
    assert "\n  " in pretty
    main(["score", str(png)])
    compact = capsys.readouterr().out
    assert json.loads(pretty) == json.loads(compact)


def test_the_module_runs_as_a_subprocess_with_clean_stdout(png):
    """The real entry point, not an in-process call: proves nothing in the
    import chain prints to stdout and pollutes the record."""
    proc = subprocess.run([sys.executable, "-m", "dfd", "score", str(png)],
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    assert json.loads(proc.stdout)["sample_id"] == "subject"


def test_subject_id_is_accepted_and_does_not_change_sample_identity(png, capsys):
    """Named for what it actually proves. AuditRecord has NO subject field, so
    --subject-id reaches Context and stops there; it cannot be asserted in the
    record. See known gap 6 — do not rename this test to imply otherwise."""
    main(["score", str(png), "--subject-id", "applicant-7"])
    assert json.loads(capsys.readouterr().out)["sample_id"] == "subject"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python3 -m pytest tests/test_cli.py -q
```
Expected: `ModuleNotFoundError: No module named 'dfd.cli'`.

- [ ] **Step 3: Write the implementation**

Add to `src/dfd/detectors/registry.py`, keeping the existing re-export line:

```python
"""Re-export so callers import the registry from an obvious place."""
from __future__ import annotations

from pathlib import Path

from .base import Detector, Registry, SyntheticDetector, abstain  # noqa: F401
from .effnet import EffNetDetector
from .npr import NPRDetector

#: Where a deployment is expected to place weights. Both files are gitignored
#: and absent in this repo, so both detectors abstain with `weights_absent`.
DEFAULT_NPR_WEIGHTS = Path("assets/models/npr.pt")
DEFAULT_EFFNET_WEIGHTS = Path("assets/models/effnet_b4_ffpp.pt")


def default_registry(npr_weights: str | Path = DEFAULT_NPR_WEIGHTS,
                     effnet_weights: str | Path = DEFAULT_EFFNET_WEIGHTS) -> Registry:
    """The detector set the CLI consults.

    Two distinct physics, per spec §6: NPR's upsampling fingerprint (slot C)
    and EfficientNet-B4's learned appearance (slot E). Both abstain when their
    weights are absent, which is every checkout of this repo.
    """
    registry = Registry()
    registry.register(NPRDetector(weights_path=npr_weights))
    registry.register(EffNetDetector(name="effnet_b4", slot="E",
                                     weights_path=effnet_weights))
    return registry
```

Create `src/dfd/cli.py`:

```python
"""Command-line entry point (spec §5.1).

Deliberately thin: argument parsing, the two output streams, and exit codes.
Every decision belongs to `pipeline.decide`.

Output is written with `sys.stdout.write`, not `print`. `src/` is under a CI
gate that forbids `print(` (structured logging, never printing), and a CLI's
record on stdout is its product rather than a log line.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence

from .audit import AuditRecord, record_digest
from .detectors.registry import default_registry
from .errors import DfdError
from .faces import DEFAULT_MODEL
from .ingest.video import DEFAULT_MAX_FRAMES
from .pipeline import decide
from .types import Context

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dfd",
        description="Score a file for manipulation and emit an audit record.")
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser(
        "score", help="score one image or video and print its audit record")
    score.add_argument("path", help="image or video file to score")
    score.add_argument("--subject-id", default=None,
                       help="subject identifier recorded in the sample context")
    score.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
                       help=f"frames to sample from a video (default {DEFAULT_MAX_FRAMES})")
    score.add_argument("--seed", type=int, default=0,
                       help="frame-selection seed for reproducible video sampling")
    score.add_argument("--face-model", default=str(DEFAULT_MODEL),
                       help="path to the face detector weights")
    score.add_argument("--pretty", action="store_true",
                       help="indent the JSON for reading; the digest is always "
                            "taken over the canonical form")
    return parser


def _summary(record: AuditRecord) -> str:
    """One line for a human, on stderr, while stdout stays machine-readable."""
    contributing = sum(1 for row in record.evidence if not row["abstained"])
    return (f"verdict={record.verdict} llr={record.llr_total:.2f} "
            f"band={record.quality_band} "
            f"contributing={contributing}/{len(record.evidence)} "
            f"faces={record.stage_reasons.get('faces', 'n/a')} "
            f"digest={record_digest(record)[:8]}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Returns:
        0 when a decision was produced — any verdict, INSUFFICIENT_EVIDENCE
        included. 2 when the input was refused. An unexpected failure is not
        caught here, so the interpreter exits 1 with its traceback: a bug
        should look like a bug, not like a rejected file.

        The verdict is deliberately absent from the exit status. `if dfd score
        f` would otherwise read a REAL verdict as failure, and `set -e` would
        abort a script on a correct answer. Callers branch on the JSON.
    """
    args = _build_parser().parse_args(argv)
    try:
        record = decide(
            args.path,
            registry=default_registry(),
            context=Context(subject_id=args.subject_id),
            face_model=args.face_model,
            max_frames=args.max_frames,
            seed=args.seed,
        )
    except DfdError as exc:
        sys.stderr.write(f"dfd: {exc}\n")
        return 2

    canonical = record.to_json()
    body = (json.dumps(json.loads(canonical), indent=2, sort_keys=True)
            if args.pretty else canonical)
    sys.stdout.write(body + "\n")
    sys.stderr.write(_summary(record) + "\n")
    return 0
```

Create `src/dfd/__main__.py`:

```python
"""`python -m dfd`."""
from __future__ import annotations

import sys

from .cli import main

sys.exit(main())
```

Add to `pyproject.toml`, after the `dependencies` array (the CI gate parses that array with a regex anchored on `^dependencies` and `^]`, so a new table after it is safe):

```toml
[project.scripts]
dfd = "dfd.cli:main"
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python3 -m pytest tests/test_cli.py -q
```

**On the decode-bomb path:** `ResourceLimitExceeded` subclasses `DfdError`, so it
reaches the same `except` and the same exit 2 as the two refusals tested here.
There is no CLI test for it because `limits` is not a command-line flag and
fabricating a real decode bomb on disk is not worth the fixture; the defence
itself is tested through `decide` in Task 4.

If `test_the_module_runs_as_a_subprocess_with_clean_stdout` fails on an import error, the subprocess lacks the src path — run it as `PYTHONPATH=src python3 -m dfd ...` or reinstall with `pip install -e .`; do not weaken the test, since its whole point is the real entry point.

- [ ] **Step 5: Prove the tests can fail (mutation)**

| Mutation | Must fail |
|---|---|
| Write the summary to `sys.stdout` | `test_the_human_summary_goes_to_stderr_not_stdout`, `test_stdout_is_exactly_one_json_object` |
| `return 1` instead of `2` in the `DfdError` handler | `test_a_missing_file_exits_two_with_a_message_and_no_traceback` |
| Remove the `try/except DfdError` entirely | same test (traceback, non-zero exit) |
| Return `1` when the verdict is not REAL | `test_a_decision_exits_zero_even_when_it_abstains` |
| `--pretty` writing pretty JSON to a different record | `test_pretty_output_is_the_same_record_reformatted` |

- [ ] **Step 6: Run the gates and commit**

```bash
ruff check src bench corpora && mypy --config-file mypy.ini && python3 -m pytest -q --cov=src/dfd --cov-fail-under=85
git add src/dfd/cli.py src/dfd/__main__.py src/dfd/detectors/registry.py pyproject.toml tests/test_cli.py
git commit -m "feat: dfd score — a person can finally run this

argparse over decide(), with the record on stdout and one human line on
stderr so the output pipes into jq. Exit status says whether the tool ran,
never what the verdict was: encoding the verdict would make 'if dfd score f'
read a REAL answer as failure.

Output goes through sys.stdout.write because src/ is under a CI gate that
forbids print(.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Documentation and whole-branch verification

**Files:**
- Modify: `docs/HANDOFF.md` (§3 "There is no composition root" paragraph; §5 recommended next steps), `docs/superpowers/ledger/2026-09-20-p0-execution-ledger.md` (append)
- Test: none new — this task verifies the whole suite.

**Interfaces:** none.

- [ ] **Step 1: Run every gate in both environments**

```bash
ruff check src bench corpora
mypy --config-file mypy.ini
python3 -m pytest -q --cov=src/dfd --cov-fail-under=85
```

Then the floor environment, which CI also runs:

```bash
python3 -m venv /tmp/floorcheck && /tmp/floorcheck/bin/pip install -q -r requirements-floor.txt && /tmp/floorcheck/bin/pip install -q -e .
/tmp/floorcheck/bin/python -m pytest -q
```

- [ ] **Step 2: Run the CLI by hand and paste the real output into the handoff**

```bash
python3 -c "
import cv2, numpy as np
rng = np.random.default_rng(0)
cv2.imwrite('/tmp/dfd-demo.png', rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
"
python3 -m dfd score /tmp/dfd-demo.png | python3 -m json.tool ; echo "exit=${PIPESTATUS[0]}"
```

Do not paraphrase what it printed. The handoff records measured output, and the
expected result is an `insufficient_evidence` record naming `weights_absent` —
if it prints a verdict, something is wrong, because no weights exist.

- [ ] **Step 3: Update `docs/HANDOFF.md`**

Replace the "**There is no composition root.**" paragraph in §3 with what is now true: `decide()` and `dfd score` exist; `fuse`, `Calibrator.to_evidence`, `build_audit_record`, `load_image`/`load_video` and `detect_faces` now have a non-test caller; Task 21's decode-bomb defences are exercised through `decide`. State what is still NOT true: the benchmark runner still builds `Observation`s from dicts and still bypasses ingest, so criteria 4 and 11 remain unmet, and the path abstains on every real input until weights land.

In §5, strike "A composition root" from the recommended steps and leave the RD adapter, the parity wiring and the embedder.

- [ ] **Step 4: Append to the execution ledger**

One entry per task with the ruling and what it costs if wrong, matching the existing format: the policy-object decision, the schema-2 bump, the ROI clamp, the scoped `ValueError` translation, the exit-code choice, and the `ood_score`-carries-disagreement compromise.

- [ ] **Step 5: Commit and push**

```bash
git add docs/HANDOFF.md docs/superpowers/ledger/2026-09-20-p0-execution-ledger.md
git commit -m "docs: the composition root exists, and what it still does not do

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git push origin p0-evidence-core
```

- [ ] **Step 6: Confirm both CI legs pass**

```bash
gh run list --repo kohrohit/deepfake-detect --branch p0-evidence-core --limit 1
```
Expected: `gates (dev)` and `gates (floor)` both SUCCESS.

---

## Known gaps this plan deliberately leaves open

State these in the PR rather than letting a reviewer find them:

1. **`ood_score` carries `disagreement`**, not an OOD score. P0 has no OOD head. Documented in `decide`'s docstring and the spec.
2. **No calibrator persistence.** `decide` accepts fitted calibrators; nothing serialises them, so the CLI always runs uncalibrated.
3. **The benchmark runner still bypasses ingest** and still fabricates landmarks. Criteria 4 and 11 stay unmet.
4. **A refusal emits no record**, so refused inputs leave only a log line.
5. **`--subject-id` is accepted but never recorded.** `AuditRecord` has no
   subject field, so the flag reaches `Context` and goes no further. Either
   drop the flag or extend the record in a later cycle — it currently promises
   something the output does not deliver.
6. **`sample_id` is the file stem**, so two files with the same stem in different directories produce records sharing a `sample_id` and differing only in `input_sha256`.
