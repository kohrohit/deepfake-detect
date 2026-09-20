# P0 — Evidence Core and Benchmark Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Tasks:** 22. Tasks 18-19 came from review findings; Tasks 20-22 from raising the bar to production standards. Task 20 (audit record) is a dropped spec requirement (§7.2), not an enhancement.

**Goal:** Build the modality-agnostic evidence core (Sample → Detector → Evidence → Fusion) and a leave-one-generator-out benchmark harness rigorous enough to prove — or disprove — that this system beats Reality Defender.

**Architecture:** Detectors emit raw scores and abstentions; a quality-conditioned calibrator converts those to log-likelihood ratios; fusion sums LLRs with an effective-sample-size discount for correlated frames. The benchmark harness wraps all of it with five evaluation-hygiene guards, a robustness surface, and a white-box adversarial baseline. Everything is hermetic: the harness is fully testable with zero model weights and zero network, so it can be proven correct before any EULA'd dataset arrives.

**Tech Stack:** Python 3.10.12, PyTorch 2.4.1 (CPU), OpenCV 4.10.0 (YuNet face detector), NumPy 1.26.4, scikit-learn 1.5.0, pytest.

**Spec:** `docs/superpowers/specs/2026-09-20-deepfake-detection-design.md`

## Global Constraints

Copied verbatim from the spec. Every task's requirements implicitly include this section.

- **Correctness first.** Latency is recorded as data, never optimised in P0. The tier cascade is explicitly deferred (spec §6).
- **Likelihood ratios, not scores.** `llr = log[P(obs|fake)/P(obs|real)]`, in nats. `llr == 0.0` means "this observation carries no information" (spec §5.2).
- **Abstention is a first-class output.** Four verdicts: `REAL` / `FAKE` / `INSUFFICIENT_EVIDENCE` / `OUT_OF_DISTRIBUTION` (spec §7).
- **Refuse to answer below a quality floor.** A detector below its quality floor is not consulted and returns an abstention, never a guess (spec §5.3).
- **ML detectors are evidence contributors, never deciders** (spec §3A.4, principle 8). The system must remain useful with every ML slot defeated.
- **Randomisation is a primitive** (spec principle 9). Detector subset selection must be seedable and recorded.
- **Everything is reconstructable after the fact** (spec principle 10). Every benchmark run records seed, dataset manifest hash, and model-version set.
- **Asset manifest is mandatory.** No dataset or weight file enters the repo without a manifest record: source, license, commercial-use verdict, evidence URL, date checked (spec §11).
- **Licensing phase gate.** Research-licensed assets are permitted through build, benchmark and internal demo. They must be swappable by config, never hardcoded (spec §11).
- **Hermetic tests.** No test touches the network. No test requires a model weight file to exist.
- **Python 3.10 typing.** Use `X | None`, not `Optional[X]`. Dataclasses are `frozen=True` unless mutation is required.
- **Ensemble diversity must be in the physics, not the architecture** (spec §6). The three P0 detectors are deliberately three different physics.

### Production engineering standards (raised 2026-09-20, binding on all tasks from Task 3's fix round onward)

This is production code for a BFSI fraud control, not a research prototype. Every task must additionally satisfy:

- **No silent failures.** Never `except:` or `except Exception: pass`. Catch the narrowest exception that can occur, and either handle it with a stated reason or let it propagate. A swallowed error in a fraud detector is a fraud that was approved.
- **Validate at boundaries.** Every public function validates its inputs and raises a typed, named exception with an actionable message. Internal helpers may assume validated input; the boundary is where the check lives.
- **Typed exceptions, not bare ones.** Define a module exception hierarchy rooted at `DfdError`. `ValueError`/`KeyError` are acceptable only where they are the semantically correct answer (a malformed record IS a programming error).
- **Structured logging, never `print`.** Use the stdlib `logging` module with module-level loggers. Log the abstention reason, never the applicant's image data or any PII.
- **Resource limits at every decode boundary.** Media arrives from an adversary (spec §3A). Cap decoded dimensions, frame counts, file sizes and durations, and reject beyond them rather than allocating.
- **Every public callable has complete type hints** and a docstring stating what it does, what it raises, and any non-obvious invariant.
- **Determinism is declared.** Any function whose output depends on randomness takes an explicit `seed`. No implicit global RNG.
- **No hardcoded magic numbers in logic.** Thresholds are module-level named constants with a comment recording their provenance.
- **Lint and type checks must pass**: `ruff check` clean, `mypy --strict` clean on `src/dfd`.

---

## File Structure

```
src/dfd/
  types.py              core dataclasses: Modality, Verdict, Quality, Observation, Context, Sample, RawScore, Evidence
  quality.py            quality metrics (blur, inter-ocular, pose, exposure) + band assignment
  faces.py              YuNet face detection + 5-point alignment
  ingest/base.py        IngestAdapter protocol
  ingest/image.py       image file  → Sample
  ingest/video.py       video file  → Sample (deterministic frame sampling)
  detectors/base.py     Detector protocol, abstention helpers
  detectors/registry.py name → Detector registry, seedable subset selection
  detectors/npr.py      slot C — upsampling fingerprint
  detectors/effnet.py   slots A and E — EfficientNet-B4 backbone, weights-parameterised
  calibration.py        RawScore → Evidence, conditioned on quality band
  fusion.py             Evidence[] → FusedResult, with effective-sample-size discount
  manifest.py           asset manifest schema, loader, commercial-use gate

bench/
  metrics.py            TPR@FPR, AUC, ECE, bootstrap CI grouped by video
  guards.py             the five evaluation-hygiene guards
  protocol.py           leave-one-generator-out split construction
  robustness.py         perturbation surface incl. screenshot / print-recapture
  adversarial.py        white-box PGD baseline
  runner.py             orchestration, seeding, reproducibility record
  report.py             markdown tables, head-to-head vs Reality Defender

corpora/
  rd_cache.py           loads the 24 cached Reality Defender results
  captures.py           loads the 442-session capture corpus

assets/manifest.yaml    the asset manifest
tests/                  mirrors src/ and bench/
```

**Why this split:** `src/dfd` is the shippable engine; `bench/` is the measuring instrument and must never be importable from production paths; `corpora/` holds corpus-specific loaders (NOT `datasets/` — that name is taken by the installed HuggingFace package and a repo-root copy shadows it) that will churn as new corpora arrive. Files that change together live together.

---

### Task 1: Project scaffold and core types

**Files:**
- Create: `pyproject.toml`, `src/dfd/__init__.py`, `src/dfd/types.py`
- Test: `tests/test_types.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Modality`, `Verdict`, `Quality`, `Observation`, `Context`, `Sample`, `RawScore`, `Evidence` — every later task depends on these exact names and field types.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_types.py
import numpy as np
import pytest
from dfd.types import (
    Modality, Verdict, Quality, Observation, Context, Sample, RawScore, Evidence,
)


def _frame() -> np.ndarray:
    return np.zeros((64, 64, 3), dtype=np.uint8)


def test_quality_band_is_explicit_not_derived():
    q = Quality(inter_ocular_px=80.0, blur_var=120.0, yaw_deg=5.0,
                pitch_deg=2.0, exposure=0.5, band="high")
    assert q.band == "high"


def test_observation_carries_source_id_for_video_level_aggregation():
    obs = Observation(t=0.0, payload=_frame(), roi=(0, 0, 64, 64),
                      quality=None, source_id="vid_001")
    assert obs.source_id == "vid_001"


def test_sample_is_immutable():
    s = Sample(sample_id="s1", modality=Modality.IMAGE, observations=(), context=Context())
    with pytest.raises(Exception):
        s.sample_id = "s2"


def test_abstained_raw_score_has_no_score():
    r = RawScore(detector="npr", version="0.1.0", score=None,
                 abstained=True, reason="weights_absent")
    assert r.abstained and r.score is None


def test_zero_llr_means_no_information():
    e = Evidence(detector="npr", detector_version="0.1.0", llr=0.0, raw_score=None,
                 uncertainty=0.0, abstained=True, reason="below_quality_floor")
    assert e.llr == 0.0


def test_four_verdicts_exist():
    assert {v.value for v in Verdict} == {
        "real", "fake", "insufficient_evidence", "out_of_distribution"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dfd'`

- [ ] **Step 3: Write minimal implementation**

```toml
# pyproject.toml
[project]
name = "dfd"
version = "0.1.0"
requires-python = ">=3.10"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src", "."]
testpaths = ["tests"]
```

```python
# src/dfd/types.py
"""Core value types. Every module in the engine speaks these and nothing else."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Modality(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    LIVE = "live"


class Verdict(str, Enum):
    REAL = "real"
    FAKE = "fake"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    OUT_OF_DISTRIBUTION = "out_of_distribution"


# Ordered worst → best. Used for floor comparisons.
QUALITY_BANDS = ("reject", "low", "medium", "high")


@dataclass(frozen=True)
class Quality:
    inter_ocular_px: float
    blur_var: float
    yaw_deg: float
    pitch_deg: float
    exposure: float
    band: str


@dataclass(frozen=True)
class Observation:
    t: float
    payload: np.ndarray
    roi: tuple[int, int, int, int] | None
    quality: Quality | None
    source_id: str


@dataclass(frozen=True)
class Context:
    subject_id: str | None = None
    generator: str | None = None
    compression: str | None = None
    label: int | None = None
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    modality: Modality
    observations: tuple[Observation, ...]
    context: Context


@dataclass(frozen=True)
class RawScore:
    """What a detector emits. Uncalibrated and not comparable across detectors."""
    detector: str
    version: str
    score: float | None
    abstained: bool
    reason: str
    artifacts: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Evidence:
    """A calibrated contribution to the decision. llr is in nats; 0.0 = no information."""
    detector: str
    detector_version: str
    llr: float
    raw_score: float | None
    uncertainty: float
    abstained: bool
    reason: str
    artifacts: dict = field(default_factory=dict)
```

```python
# src/dfd/__init__.py
"""Modality-agnostic deepfake detection engine."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_types.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/dfd/__init__.py src/dfd/types.py tests/test_types.py
git commit -m "feat: core value types for the evidence pipeline"
```

---

### Task 2: Asset manifest and commercial-use gate

**Files:**
- Create: `src/dfd/manifest.py`, `assets/manifest.yaml`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Consumes: nothing
- Produces: `AssetRecord`, `load_manifest(path) -> dict[str, AssetRecord]`, `assert_release_clean(manifest, asset_ids)` — the CI gate.

This implements spec §11. It is deliberately Task 2 because every later task that touches a weight file must register it here first.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_manifest.py
import pytest
from dfd.manifest import AssetRecord, load_manifest, assert_release_clean, NonCommercialAsset

MANIFEST = """
assets:
  yunet_face_detector:
    source: "https://github.com/opencv/opencv_zoo"
    license: "MIT"
    commercial_use: true
    evidence_url: "https://github.com/opencv/opencv_zoo/blob/main/LICENSE"
    date_checked: "2026-09-20"
    checked_by: "kohrohit@gmail.com"
  ffpp_xception_weights:
    source: "DeepfakeBench"
    license: "research-only"
    commercial_use: false
    evidence_url: "https://github.com/SCLBD/DeepfakeBench"
    date_checked: "2026-09-20"
    checked_by: "kohrohit@gmail.com"
"""


def test_load_manifest_parses_records(tmp_path):
    p = tmp_path / "manifest.yaml"
    p.write_text(MANIFEST)
    m = load_manifest(p)
    assert m["yunet_face_detector"].commercial_use is True
    assert m["ffpp_xception_weights"].commercial_use is False


def test_release_gate_passes_on_clean_assets(tmp_path):
    p = tmp_path / "manifest.yaml"
    p.write_text(MANIFEST)
    m = load_manifest(p)
    assert_release_clean(m, ["yunet_face_detector"]) is None


def test_release_gate_blocks_non_commercial_asset(tmp_path):
    p = tmp_path / "manifest.yaml"
    p.write_text(MANIFEST)
    m = load_manifest(p)
    with pytest.raises(NonCommercialAsset) as exc:
        assert_release_clean(m, ["yunet_face_detector", "ffpp_xception_weights"])
    assert "ffpp_xception_weights" in str(exc.value)


def test_unregistered_asset_is_an_error_not_a_pass(tmp_path):
    """An asset with no manifest record must fail closed."""
    p = tmp_path / "manifest.yaml"
    p.write_text(MANIFEST)
    m = load_manifest(p)
    with pytest.raises(NonCommercialAsset):
        assert_release_clean(m, ["some_weights_nobody_registered"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dfd.manifest'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/dfd/manifest.py
"""Asset provenance manifest and the release gate that enforces it (spec §11).

Fails closed: an asset with no record is treated as non-commercial.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class NonCommercialAsset(Exception):
    """Raised when a release bundle references an asset not cleared for commercial use."""


@dataclass(frozen=True)
class AssetRecord:
    asset_id: str
    source: str
    license: str
    commercial_use: bool
    evidence_url: str
    date_checked: str
    checked_by: str


def load_manifest(path: str | Path) -> dict[str, AssetRecord]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    out: dict[str, AssetRecord] = {}
    for asset_id, rec in (data.get("assets") or {}).items():
        out[asset_id] = AssetRecord(
            asset_id=asset_id,
            source=rec["source"],
            license=rec["license"],
            commercial_use=bool(rec["commercial_use"]),
            evidence_url=rec["evidence_url"],
            date_checked=rec["date_checked"],
            checked_by=rec["checked_by"],
        )
    return out


def assert_release_clean(manifest: dict[str, AssetRecord], asset_ids: list[str]) -> None:
    """Raise if any asset is unregistered or not cleared for commercial use."""
    bad: list[str] = []
    for aid in asset_ids:
        rec = manifest.get(aid)
        if rec is None or not rec.commercial_use:
            bad.append(aid)
    if bad:
        raise NonCommercialAsset(
            "assets not cleared for commercial release: " + ", ".join(sorted(bad))
        )
```

```yaml
# assets/manifest.yaml
# Every dataset and weight file used anywhere in this repo must appear here.
# Fails closed: unregistered == non-commercial. See spec §11.
assets: {}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_manifest.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add src/dfd/manifest.py assets/manifest.yaml tests/test_manifest.py
git commit -m "feat: asset provenance manifest with a fail-closed release gate"
```

---

### Task 3: Quality metrics and banding

**Files:**
- Create: `src/dfd/quality.py`
- Test: `tests/test_quality.py`

**Interfaces:**
- Consumes: `Quality` from Task 1
- Produces: `measure_quality(frame, roi, landmarks) -> Quality`, `meets_floor(band, floor) -> bool`

This implements spec §5.3 — the refuse-to-answer gate.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quality.py
import numpy as np
from dfd.quality import measure_quality, meets_floor


def _sharp(h=256, w=256) -> np.ndarray:
    """A high-frequency checkerboard: high Laplacian variance."""
    img = np.indices((h, w)).sum(axis=0) % 2
    return (img * 255).astype(np.uint8)[:, :, None].repeat(3, axis=2)


def _flat(h=256, w=256) -> np.ndarray:
    return np.full((h, w, 3), 128, dtype=np.uint8)


LM_WIDE = np.array([[80.0, 100.0], [176.0, 100.0]])   # 96 px inter-ocular
LM_TIGHT = np.array([[120.0, 100.0], [140.0, 100.0]])  # 20 px inter-ocular


def test_sharp_image_has_higher_blur_var_than_flat():
    sharp = measure_quality(_sharp(), (0, 0, 256, 256), LM_WIDE)
    flat = measure_quality(_flat(), (0, 0, 256, 256), LM_WIDE)
    assert sharp.blur_var > flat.blur_var


def test_inter_ocular_distance_is_measured_from_landmarks():
    q = measure_quality(_sharp(), (0, 0, 256, 256), LM_WIDE)
    assert abs(q.inter_ocular_px - 96.0) < 1e-6


def test_small_face_is_banded_reject():
    q = measure_quality(_sharp(), (0, 0, 256, 256), LM_TIGHT)
    assert q.band == "reject"


def test_flat_image_is_not_banded_high():
    q = measure_quality(_flat(), (0, 0, 256, 256), LM_WIDE)
    assert q.band != "high"


def test_meets_floor_uses_band_ordering():
    assert meets_floor("high", "medium") is True
    assert meets_floor("low", "medium") is False
    assert meets_floor("medium", "medium") is True
    assert meets_floor("reject", "low") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_quality.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dfd.quality'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/dfd/quality.py
"""Quality measurement and banding (spec §5.3).

A detector below its quality floor is not consulted. This is the layer that
turns a confident guess on a 40-pixel blurry face into an honest abstention.

Thresholds here are starting values. Task 18 records the observed distribution
so they can be set from data rather than from intuition.
"""
from __future__ import annotations

import cv2
import numpy as np

from .types import QUALITY_BANDS, Quality

# Starting thresholds. Revisit against the distribution recorded by the benchmark.
MIN_IOD_REJECT = 32.0
MIN_IOD_LOW = 64.0
MIN_IOD_HIGH = 96.0
MIN_BLUR_LOW = 20.0
MIN_BLUR_HIGH = 100.0
MAX_YAW_HIGH = 30.0
EXPOSURE_OK = (0.15, 0.90)


def _laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def measure_quality(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    landmarks: np.ndarray,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
) -> Quality:
    """Measure quality of the face at `roi`.

    landmarks: at least two points, [[lx, ly], [rx, ry]] for left and right eye.
    """
    x, y, w, h = roi
    crop = frame[y : y + h, x : x + w]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop

    iod = float(np.linalg.norm(landmarks[0] - landmarks[1]))
    blur = _laplacian_var(gray)
    exposure = float(gray.mean() / 255.0)

    band = _band(iod, blur, yaw_deg, exposure)
    return Quality(
        inter_ocular_px=iod,
        blur_var=blur,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        exposure=exposure,
        band=band,
    )


def _band(iod: float, blur: float, yaw: float, exposure: float) -> str:
    if iod < MIN_IOD_REJECT or blur < MIN_BLUR_LOW:
        return "reject"
    if not (EXPOSURE_OK[0] <= exposure <= EXPOSURE_OK[1]):
        return "low"
    if iod >= MIN_IOD_HIGH and blur >= MIN_BLUR_HIGH and abs(yaw) <= MAX_YAW_HIGH:
        return "high"
    if iod >= MIN_IOD_LOW:
        return "medium"
    return "low"


def meets_floor(band: str, floor: str) -> bool:
    """True if `band` is at least as good as `floor`."""
    return QUALITY_BANDS.index(band) >= QUALITY_BANDS.index(floor)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_quality.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add src/dfd/quality.py tests/test_quality.py
git commit -m "feat: quality measurement and banding for the abstention gate"
```

---

### Task 4: Face detection and alignment

**Files:**
- Create: `src/dfd/faces.py`
- Modify: `assets/manifest.yaml`
- Test: `tests/test_faces.py`

**Interfaces:**
- Consumes: `measure_quality` from Task 3
- Produces: `FaceBox`, `FaceDetector`, `detect_faces(frame) -> list[FaceBox]`, `align(frame, box, size) -> np.ndarray`

**Why YuNet and not InsightFace:** spec §11 records that InsightFace pretrained models are academic-research-only. YuNet ships via OpenCV Zoo under MIT. This is the licence-clean substitute, and it is registered in the manifest as part of this task.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_faces.py
import numpy as np
import pytest
from dfd.faces import FaceBox, align, detect_faces, WEIGHTS_ABSENT


def _frame(h=480, w=640) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_detect_faces_returns_empty_when_weights_absent(tmp_path):
    """Hermetic: no weight file in CI. Must not raise, must not invent faces."""
    out = detect_faces(_frame(), model_path=tmp_path / "missing.onnx")
    assert out == []


def test_detect_faces_reports_why_it_returned_nothing(tmp_path):
    out, reason = detect_faces(_frame(), model_path=tmp_path / "missing.onnx",
                               with_reason=True)
    assert out == [] and reason == WEIGHTS_ABSENT


def test_align_crops_to_requested_size():
    frame = _frame()
    box = FaceBox(x=100, y=100, w=200, h=200,
                  landmarks=np.array([[150.0, 160.0], [250.0, 160.0]]),
                  score=0.99)
    out = align(frame, box, size=224)
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.uint8


def test_align_clamps_box_to_frame_bounds():
    """A box running off the edge must not raise or produce a wrong-sized crop."""
    frame = _frame(h=100, w=100)
    box = FaceBox(x=80, y=80, w=200, h=200,
                  landmarks=np.array([[90.0, 90.0], [95.0, 90.0]]),
                  score=0.9)
    out = align(frame, box, size=64)
    assert out.shape == (64, 64, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_faces.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dfd.faces'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/dfd/faces.py
"""Face detection and alignment via OpenCV YuNet (MIT, see assets/manifest.yaml).

Fails soft: a missing weight file yields zero faces with a stated reason, never
an exception and never a fabricated detection. This keeps CI hermetic and keeps
the whole pipeline honest about what it could not measure.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

WEIGHTS_ABSENT = "weights_absent"
OK = "ok"

DEFAULT_MODEL = Path("assets/models/face_detection_yunet_2023mar.onnx")


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    landmarks: np.ndarray  # (5, 2): right eye, left eye, nose, right mouth, left mouth
    score: float


def detect_faces(
    frame: np.ndarray,
    model_path: str | Path = DEFAULT_MODEL,
    score_threshold: float = 0.7,
    with_reason: bool = False,
):
    """Detect faces. Returns [] (and a reason) when the model file is absent."""
    path = Path(model_path)
    if not path.exists():
        return ([], WEIGHTS_ABSENT) if with_reason else []

    h, w = frame.shape[:2]
    det = cv2.FaceDetectorYN.create(str(path), "", (w, h),
                                    score_threshold=score_threshold)
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    _, faces = det.detect(bgr)
    out: list[FaceBox] = []
    if faces is not None:
        for f in faces:
            x, y, bw, bh = (int(v) for v in f[:4])
            lms = np.array(f[4:14], dtype=np.float64).reshape(5, 2)
            out.append(FaceBox(x=x, y=y, w=bw, h=bh, landmarks=lms, score=float(f[14])))
    return (out, OK) if with_reason else out


def align(frame: np.ndarray, box: FaceBox, size: int = 224) -> np.ndarray:
    """Crop the face box, clamped to frame bounds, resized to (size, size)."""
    h, w = frame.shape[:2]
    x0 = max(0, box.x)
    y0 = max(0, box.y)
    x1 = min(w, box.x + box.w)
    y1 = min(h, box.y + box.h)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    crop = frame[y0:y1, x0:x1]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
```

Append to `assets/manifest.yaml` (replacing the empty `assets: {}`):

```yaml
assets:
  yunet_face_detector:
    source: "https://github.com/opencv/opencv_zoo — face_detection_yunet_2023mar.onnx"
    license: "MIT"
    commercial_use: true
    evidence_url: "https://github.com/opencv/opencv_zoo/blob/main/LICENSE"
    date_checked: "2026-09-20"
    checked_by: "kohrohit@gmail.com"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_faces.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add src/dfd/faces.py assets/manifest.yaml tests/test_faces.py
git commit -m "feat: YuNet face detection and alignment, failing soft without weights"
```

---

### Task 5: Ingest adapters for image and video

**Files:**
- Create: `src/dfd/ingest/__init__.py`, `src/dfd/ingest/base.py`, `src/dfd/ingest/image.py`, `src/dfd/ingest/video.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `Sample`, `Observation`, `Context`, `Modality` (Task 1); `detect_faces`, `align` (Task 4); `measure_quality` (Task 3)
- Produces: `load_image(path, context) -> Sample`, `load_video(path, context, max_frames, seed) -> Sample`

Frame sampling must be **deterministic given a seed** — spec principle 10 requires runs be reconstructable.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest.py
import cv2
import numpy as np
import pytest
from dfd.ingest.image import load_image
from dfd.ingest.video import load_video, sample_indices
from dfd.types import Context, Modality


@pytest.fixture
def jpg(tmp_path):
    p = tmp_path / "a.jpg"
    img = np.full((256, 256, 3), 120, dtype=np.uint8)
    cv2.imwrite(str(p), img)
    return p


@pytest.fixture
def mp4(tmp_path):
    p = tmp_path / "a.mp4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (128, 128))
    for i in range(30):
        vw.write(np.full((128, 128, 3), i * 8 % 255, dtype=np.uint8))
    vw.release()
    return p


def test_image_yields_exactly_one_observation(jpg):
    s = load_image(jpg, Context(label=0))
    assert s.modality is Modality.IMAGE
    assert len(s.observations) == 1


def test_image_observation_source_id_is_the_sample_id(jpg):
    """Video-level aggregation groups by source_id; an image is its own group."""
    s = load_image(jpg, Context())
    assert s.observations[0].source_id == s.sample_id


def test_video_yields_at_most_max_frames(mp4):
    s = load_video(mp4, Context(), max_frames=5, seed=42)
    assert s.modality is Modality.VIDEO
    assert len(s.observations) <= 5


def test_video_all_observations_share_one_source_id(mp4):
    s = load_video(mp4, Context(), max_frames=5, seed=42)
    assert len({o.source_id for o in s.observations}) == 1


def test_frame_sampling_is_deterministic_given_a_seed():
    a = sample_indices(total=100, k=7, seed=42)
    b = sample_indices(total=100, k=7, seed=42)
    c = sample_indices(total=100, k=7, seed=43)
    assert a == b
    assert a != c


def test_frame_sampling_never_exceeds_available_frames():
    assert sample_indices(total=3, k=10, seed=1) == [0, 1, 2]


def test_video_timestamps_increase(mp4):
    s = load_video(mp4, Context(), max_frames=5, seed=1)
    ts = [o.t for o in s.observations]
    assert ts == sorted(ts)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dfd.ingest'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/dfd/ingest/__init__.py
"""Ingest adapters: any medium in, one Sample out."""
```

```python
# src/dfd/ingest/base.py
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..types import Context, Sample


class IngestAdapter(Protocol):
    def __call__(self, path: str | Path, context: Context) -> Sample: ...
```

```python
# src/dfd/ingest/image.py
from __future__ import annotations

from pathlib import Path

import cv2

from ..types import Context, Modality, Observation, Sample


def load_image(path: str | Path, context: Context) -> Sample:
    path = Path(path)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not decode image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    sample_id = path.stem
    obs = Observation(t=0.0, payload=rgb, roi=None, quality=None, source_id=sample_id)
    return Sample(sample_id=sample_id, modality=Modality.IMAGE,
                  observations=(obs,), context=context)
```

```python
# src/dfd/ingest/video.py
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..types import Context, Modality, Observation, Sample


def sample_indices(total: int, k: int, seed: int) -> list[int]:
    """Deterministic frame indices. Returns all frames when total <= k."""
    if total <= k:
        return list(range(total))
    rng = np.random.default_rng(seed)
    # Stratified: one frame per equal-width bin, jittered inside the bin.
    edges = np.linspace(0, total, k + 1).astype(int)
    idx = [int(rng.integers(edges[i], max(edges[i] + 1, edges[i + 1])))
           for i in range(k)]
    return sorted(set(min(i, total - 1) for i in idx))


def load_video(path: str | Path, context: Context, max_frames: int = 32,
               seed: int = 0) -> Sample:
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    wanted = set(sample_indices(total, max_frames, seed))

    sample_id = path.stem
    obs: list[Observation] = []
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i in wanted:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            obs.append(Observation(t=i / fps, payload=rgb, roi=None,
                                   quality=None, source_id=sample_id))
        i += 1
    cap.release()
    return Sample(sample_id=sample_id, modality=Modality.VIDEO,
                  observations=tuple(obs), context=context)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ingest.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add src/dfd/ingest tests/test_ingest.py
git commit -m "feat: image and video ingest with deterministic frame sampling"
```

---

### Task 6: Detector protocol and seedable registry

**Files:**
- Create: `src/dfd/detectors/__init__.py`, `src/dfd/detectors/base.py`, `src/dfd/detectors/registry.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: `RawScore`, `Observation`, `Modality` (Task 1); `meets_floor` (Task 3)
- Produces: `Detector` protocol, `abstain(...) -> RawScore`, `register(detector)`, `get(name)`, `select_subset(names, k, seed) -> list[str]`, `SyntheticDetector`

`select_subset` implements spec principle 9 (randomisation as a primitive) and §3A.2 (per-deployment randomised configuration). `SyntheticDetector` is what makes the whole harness testable without weights.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry.py
import numpy as np
import pytest
from dfd.detectors.base import SyntheticDetector, abstain
from dfd.detectors.registry import Registry
from dfd.types import Modality, Observation, Quality


def _obs(band="high") -> Observation:
    q = Quality(inter_ocular_px=100, blur_var=200, yaw_deg=0,
                pitch_deg=0, exposure=0.5, band=band)
    return Observation(t=0.0, payload=np.zeros((64, 64, 3), np.uint8),
                       roi=(0, 0, 64, 64), quality=q, source_id="s1")


def test_abstain_helper_produces_zero_information_rawscore():
    r = abstain("npr", "0.1.0", "weights_absent")
    assert r.abstained and r.score is None and r.reason == "weights_absent"


def test_synthetic_detector_is_deterministic():
    d = SyntheticDetector(name="synth", seed=7)
    a = d.score([_obs()])
    b = d.score([_obs()])
    assert a.score == b.score


def test_detector_abstains_below_its_quality_floor():
    d = SyntheticDetector(name="synth", seed=7, min_quality_band="high")
    r = d.score([_obs(band="low")])
    assert r.abstained and r.reason == "below_quality_floor"


def test_registry_round_trips():
    reg = Registry()
    d = SyntheticDetector(name="synth", seed=1)
    reg.register(d)
    assert reg.get("synth") is d
    assert "synth" in reg.names()


def test_registry_rejects_duplicate_names():
    reg = Registry()
    reg.register(SyntheticDetector(name="synth", seed=1))
    with pytest.raises(ValueError):
        reg.register(SyntheticDetector(name="synth", seed=2))


def test_subset_selection_is_seeded_and_reproducible():
    reg = Registry()
    for i in range(5):
        reg.register(SyntheticDetector(name=f"d{i}", seed=i))
    a = reg.select_subset(k=3, seed=99)
    b = reg.select_subset(k=3, seed=99)
    c = reg.select_subset(k=3, seed=100)
    assert a == b and len(a) == 3
    assert a != c


def test_subset_selection_caps_at_registry_size():
    reg = Registry()
    reg.register(SyntheticDetector(name="only", seed=1))
    assert reg.select_subset(k=10, seed=1) == ["only"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dfd.detectors'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/dfd/detectors/__init__.py
"""Detectors: perishable, hot-swappable evidence producers (spec §3, §6)."""
```

```python
# src/dfd/detectors/base.py
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from ..quality import meets_floor
from ..types import Modality, Observation, RawScore

BELOW_FLOOR = "below_quality_floor"
NO_QUALITY = "quality_not_measured"
WEIGHTS_ABSENT = "weights_absent"
OK = "ok"


def abstain(detector: str, version: str, reason: str) -> RawScore:
    return RawScore(detector=detector, version=version, score=None,
                    abstained=True, reason=reason)


class Detector(Protocol):
    name: str
    version: str
    modalities: set[Modality]
    min_quality_band: str

    def score(self, obs: Sequence[Observation]) -> RawScore: ...


@dataclass
class SyntheticDetector:
    """A deterministic stand-in used to test the harness without model weights.

    Its score is a hash of the observation payload, so it is reproducible and
    label-blind. It exists so the benchmark can be proven correct before any
    dataset or weight file arrives.
    """
    name: str
    seed: int = 0
    version: str = "synthetic-1"
    modalities: set[Modality] = field(
        default_factory=lambda: {Modality.IMAGE, Modality.VIDEO})
    min_quality_band: str = "low"

    def score(self, obs: Sequence[Observation]) -> RawScore:
        if not obs:
            return abstain(self.name, self.version, NO_QUALITY)
        usable = []
        for o in obs:
            if o.quality is None:
                continue
            if meets_floor(o.quality.band, self.min_quality_band):
                usable.append(o)
        if not usable:
            reason = NO_QUALITY if obs[0].quality is None else BELOW_FLOOR
            return abstain(self.name, self.version, reason)

        acc = 0
        for o in usable:
            h = hashlib.sha256(np.ascontiguousarray(o.payload).tobytes())
            h.update(str(self.seed).encode())
            acc ^= int.from_bytes(h.digest()[:8], "big")
        return RawScore(detector=self.name, version=self.version,
                        score=(acc % 10_000) / 10_000.0,
                        abstained=False, reason=OK)


class Registry:
    """Holds detectors and draws seeded random subsets (spec principle 9)."""

    def __init__(self) -> None:
        self._d: dict[str, Detector] = {}

    def register(self, detector: Detector) -> None:
        if detector.name in self._d:
            raise ValueError(f"detector already registered: {detector.name}")
        self._d[detector.name] = detector

    def get(self, name: str) -> Detector:
        return self._d[name]

    def names(self) -> list[str]:
        return sorted(self._d)

    def select_subset(self, k: int, seed: int) -> list[str]:
        names = self.names()
        k = min(k, len(names))
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(names, size=k, replace=False).tolist())
```

```python
# src/dfd/detectors/registry.py
"""Re-export so callers import the registry from an obvious place."""
from .base import Detector, Registry, SyntheticDetector, abstain  # noqa: F401
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_registry.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add src/dfd/detectors tests/test_registry.py
git commit -m "feat: detector protocol, seedable registry, synthetic test detector"
```

---

### Task 7: NPR detector (slot C — upsampling fingerprint)

**Files:**
- Create: `src/dfd/detectors/npr.py`
- Modify: `assets/manifest.yaml`
- Test: `tests/test_npr.py`

**Interfaces:**
- Consumes: `Detector`, `abstain`, `WEIGHTS_ABSENT` (Task 6)
- Produces: `npr_feature(img) -> np.ndarray`, `NPRDetector`

**Note for the implementer:** the NPR feature is the residual between an image and its own downsample-then-upsample. **Verify the exact form against the NPR paper** (Tan et al., CVPR 2024, "Rethinking the Up-Sampling Operations…") before trusting the numbers — the test below pins the *property* that matters (nearest-neighbour upsampled content has near-zero residual), not a magic constant, so it stays valid either way.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_npr.py
import numpy as np
import pytest
from dfd.detectors.base import WEIGHTS_ABSENT
from dfd.detectors.npr import NPRDetector, npr_feature
from dfd.types import Modality, Observation, Quality


def _obs(img) -> Observation:
    q = Quality(inter_ocular_px=100, blur_var=200, yaw_deg=0,
                pitch_deg=0, exposure=0.5, band="high")
    return Observation(t=0.0, payload=img, roi=(0, 0, *img.shape[:2][::-1]),
                       quality=q, source_id="s1")


def test_npr_feature_is_near_zero_on_nearest_upsampled_content():
    """An image that IS a 2x nearest upsample has almost no NPR residual.

    This is the physics the detector reads: generator upsampling leaves a
    characteristic residual that natural images do not have.
    """
    small = np.random.default_rng(0).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    upsampled = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)
    feat = npr_feature(upsampled)
    assert np.abs(feat).mean() < 1e-6


def test_npr_feature_is_nonzero_on_natural_noise():
    noise = np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    feat = npr_feature(noise)
    assert np.abs(feat).mean() > 1e-3


def test_npr_feature_preserves_shape():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    assert npr_feature(img).shape == (64, 64, 3)


def test_detector_abstains_when_weights_absent(tmp_path):
    d = NPRDetector(weights_path=tmp_path / "missing.pt")
    r = d.score([_obs(np.zeros((64, 64, 3), np.uint8))])
    assert r.abstained and r.reason == WEIGHTS_ABSENT


def test_detector_declares_its_slot_and_physics():
    d = NPRDetector(weights_path="whatever")
    assert d.name == "npr"
    assert d.slot == "C"
    assert Modality.IMAGE in d.modalities
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_npr.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dfd.detectors.npr'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/dfd/detectors/npr.py
"""Slot C — upsampling fingerprint (spec §6).

Physics: neural generators upsample via transposed convolution or
interpolate-then-convolve, which leaves a periodic residual structure that a
camera's optical chain does not produce. The NPR feature isolates that residual
by subtracting the image's own downsample-then-upsample reconstruction.

Verify the exact feature definition against the NPR paper before relying on
absolute numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from ..quality import meets_floor
from ..types import Modality, Observation, RawScore
from .base import BELOW_FLOOR, NO_QUALITY, OK, WEIGHTS_ABSENT, abstain


def npr_feature(img: np.ndarray, stride: int = 2) -> np.ndarray:
    """Residual between the image and its nearest-neighbour down/up reconstruction."""
    x = img.astype(np.float32) / 255.0
    down = x[::stride, ::stride]
    up = np.repeat(np.repeat(down, stride, axis=0), stride, axis=1)
    up = up[: x.shape[0], : x.shape[1]]
    return x - up


@dataclass
class NPRDetector:
    weights_path: str | Path
    name: str = "npr"
    slot: str = "C"
    version: str = "0.1.0"
    modalities: set[Modality] = field(
        default_factory=lambda: {Modality.IMAGE, Modality.VIDEO})
    min_quality_band: str = "low"
    _model: object | None = None

    def _load(self):
        if self._model is not None:
            return self._model
        import torch
        self._model = torch.load(self.weights_path, map_location="cpu")
        self._model.eval()
        return self._model

    def score(self, obs: Sequence[Observation]) -> RawScore:
        if not Path(self.weights_path).exists():
            return abstain(self.name, self.version, WEIGHTS_ABSENT)
        usable = [o for o in obs
                  if o.quality is not None
                  and meets_floor(o.quality.band, self.min_quality_band)]
        if not usable:
            reason = NO_QUALITY if (not obs or obs[0].quality is None) else BELOW_FLOOR
            return abstain(self.name, self.version, reason)

        import torch
        model = self._load()
        feats = np.stack([npr_feature(o.payload) for o in usable])
        t = torch.from_numpy(feats).permute(0, 3, 1, 2).float()
        with torch.no_grad():
            logits = model(t)
            probs = torch.softmax(logits, dim=1)[:, 1]
        return RawScore(detector=self.name, version=self.version,
                        score=float(probs.mean()), abstained=False, reason=OK,
                        artifacts={"n_observations": len(usable)})
```

Append to `assets/manifest.yaml`:

```yaml
  npr_weights:
    source: "https://github.com/chuangchuangtan/NPR-DeepfakeDetection"
    license: "research-only — VERIFY before any commercial release (spec §11)"
    commercial_use: false
    evidence_url: "https://github.com/chuangchuangtan/NPR-DeepfakeDetection"
    date_checked: "2026-09-20"
    checked_by: "kohrohit@gmail.com"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_npr.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add src/dfd/detectors/npr.py assets/manifest.yaml tests/test_npr.py
git commit -m "feat: NPR detector for the upsampling-fingerprint slot"
```

---

### Task 8: EfficientNet-B4 detector (slots A and E, weights-parameterised)

**Files:**
- Create: `src/dfd/detectors/effnet.py`
- Modify: `assets/manifest.yaml`
- Test: `tests/test_effnet.py`

**Interfaces:**
- Consumes: `Detector`, `abstain` (Task 6)
- Produces: `EffNetDetector(name, slot, weights_path, ...)`

**Why one class for two slots:** SBI (slot A, blending boundary) is a *training scheme*, not an architecture — at inference it is an EfficientNet-B4, identical to the FF++-trained appearance detector in slot E. Making this one weights-parameterised class keeps the code DRY and turns the slot-A vs slot-E comparison into a pure weights comparison, which is exactly what the benchmark should be measuring.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_effnet.py
import numpy as np
import pytest
from dfd.detectors.base import WEIGHTS_ABSENT
from dfd.detectors.effnet import EffNetDetector, preprocess
from dfd.types import Modality, Observation, Quality


def _obs(band="high") -> Observation:
    img = np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8)
    q = Quality(inter_ocular_px=100, blur_var=200, yaw_deg=0,
                pitch_deg=0, exposure=0.5, band=band)
    return Observation(t=0.0, payload=img, roi=(0, 0, 224, 224),
                       quality=q, source_id="s1")


def test_preprocess_produces_nchw_float_in_unit_range():
    img = np.full((224, 224, 3), 255, dtype=np.uint8)
    t = preprocess([img])
    assert t.shape == (1, 3, 224, 224)
    assert t.dtype.name == "float32"
    assert 0.0 <= float(t.min()) and float(t.max()) <= 1.0


def test_preprocess_resizes_to_model_input():
    img = np.zeros((97, 61, 3), dtype=np.uint8)
    assert preprocess([img], size=224).shape == (1, 3, 224, 224)


def test_sbi_and_appearance_are_the_same_class_different_weights(tmp_path):
    sbi = EffNetDetector(name="sbi", slot="A", weights_path=tmp_path / "sbi.pt")
    app = EffNetDetector(name="effnet_ffpp", slot="E", weights_path=tmp_path / "e.pt")
    assert type(sbi) is type(app)
    assert sbi.slot != app.slot


def test_abstains_when_weights_absent(tmp_path):
    d = EffNetDetector(name="sbi", slot="A", weights_path=tmp_path / "missing.pt")
    r = d.score([_obs()])
    assert r.abstained and r.reason == WEIGHTS_ABSENT


def test_abstains_below_quality_floor(tmp_path):
    (tmp_path / "w.pt").write_bytes(b"not-a-real-model")
    d = EffNetDetector(name="sbi", slot="A", weights_path=tmp_path / "w.pt",
                       min_quality_band="high")
    r = d.score([_obs(band="low")])
    assert r.abstained and r.reason == "below_quality_floor"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_effnet.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dfd.detectors.effnet'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/dfd/detectors/effnet.py
"""Slots A and E — EfficientNet-B4 backbone, parameterised by weights (spec §6).

Slot A (SBI): trained with self-blended images; reads the *composite seam*.
Slot E (FF++): trained on FF++ fakes; reads *learned appearance artifacts*.

Same architecture, different physics — because the physics lives in the
training scheme, not the layers. Keeping them one class makes the benchmark's
A-vs-E comparison a clean weights comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from ..quality import meets_floor
from ..types import Modality, Observation, RawScore
from .base import BELOW_FLOOR, NO_QUALITY, OK, WEIGHTS_ABSENT, abstain


def preprocess(images: Sequence[np.ndarray], size: int = 224) -> np.ndarray:
    """RGB uint8 HWC list → float32 NCHW in [0, 1]."""
    out = []
    for img in images:
        if img.shape[0] != size or img.shape[1] != size:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        out.append(img.astype(np.float32) / 255.0)
    arr = np.stack(out)
    return np.ascontiguousarray(arr.transpose(0, 3, 1, 2))


@dataclass
class EffNetDetector:
    name: str
    slot: str
    weights_path: str | Path
    version: str = "0.1.0"
    input_size: int = 224
    modalities: set[Modality] = field(
        default_factory=lambda: {Modality.IMAGE, Modality.VIDEO})
    min_quality_band: str = "low"
    _model: object | None = None

    def _load(self):
        if self._model is not None:
            return self._model
        import torch
        self._model = torch.load(self.weights_path, map_location="cpu")
        self._model.eval()
        return self._model

    def score(self, obs: Sequence[Observation]) -> RawScore:
        if not Path(self.weights_path).exists():
            return abstain(self.name, self.version, WEIGHTS_ABSENT)
        usable = [o for o in obs
                  if o.quality is not None
                  and meets_floor(o.quality.band, self.min_quality_band)]
        if not usable:
            reason = NO_QUALITY if (not obs or obs[0].quality is None) else BELOW_FLOOR
            return abstain(self.name, self.version, reason)

        import torch
        model = self._load()
        t = torch.from_numpy(preprocess([o.payload for o in usable], self.input_size))
        with torch.no_grad():
            logits = model(t)
            probs = torch.softmax(logits, dim=1)[:, 1]
        return RawScore(detector=self.name, version=self.version,
                        score=float(probs.mean()), abstained=False, reason=OK,
                        artifacts={"n_observations": len(usable), "slot": self.slot})
```

Append to `assets/manifest.yaml`:

```yaml
  sbi_effnetb4_weights:
    source: "https://github.com/mapooon/SelfBlendedImages"
    license: "research-only — VERIFY before any commercial release (spec §11)"
    commercial_use: false
    evidence_url: "https://github.com/mapooon/SelfBlendedImages"
    date_checked: "2026-09-20"
    checked_by: "kohrohit@gmail.com"
  ffpp_effnetb4_weights:
    source: "DeepfakeBench"
    license: "research-only — VERIFY before any commercial release (spec §11)"
    commercial_use: false
    evidence_url: "https://github.com/SCLBD/DeepfakeBench"
    date_checked: "2026-09-20"
    checked_by: "kohrohit@gmail.com"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_effnet.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add src/dfd/detectors/effnet.py assets/manifest.yaml tests/test_effnet.py
git commit -m "feat: EfficientNet-B4 detector serving the SBI and appearance slots"
```

---

### Task 9: Quality-conditioned calibration

**Files:**
- Create: `src/dfd/calibration.py`
- Test: `tests/test_calibration.py`

**Interfaces:**
- Consumes: `RawScore`, `Evidence` (Task 1)
- Produces: `Calibrator`, `Calibrator.fit(scores, labels, bands)`, `Calibrator.to_evidence(raw, band) -> Evidence`

This implements spec §7.1. The crucial property is that calibration is **per quality band** — a detector's reliability at 512px is not its reliability at 96px.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calibration.py
import math
import numpy as np
import pytest
from dfd.calibration import Calibrator
from dfd.types import RawScore


def _raw(score, detector="d1"):
    return RawScore(detector=detector, version="0.1.0", score=score,
                    abstained=False, reason="ok")


def _fit_separable(band="high", n=200):
    rng = np.random.default_rng(0)
    real = rng.uniform(0.0, 0.4, n)
    fake = rng.uniform(0.6, 1.0, n)
    scores = np.concatenate([real, fake])
    labels = np.concatenate([np.zeros(n), np.ones(n)])
    bands = [band] * len(scores)
    c = Calibrator(detector="d1")
    c.fit(scores, labels, bands)
    return c


def test_high_score_yields_positive_llr():
    c = _fit_separable()
    ev = c.to_evidence(_raw(0.95), band="high")
    assert ev.llr > 0


def test_low_score_yields_negative_llr():
    c = _fit_separable()
    ev = c.to_evidence(_raw(0.05), band="high")
    assert ev.llr < 0


def test_abstained_rawscore_becomes_zero_llr_evidence():
    c = _fit_separable()
    raw = RawScore(detector="d1", version="0.1.0", score=None,
                   abstained=True, reason="weights_absent")
    ev = c.to_evidence(raw, band="high")
    assert ev.llr == 0.0 and ev.abstained and ev.reason == "weights_absent"


def test_uncalibrated_band_yields_zero_llr_not_a_guess():
    """The honest response to 'I was never calibrated here' is no information."""
    c = _fit_separable(band="high")
    ev = c.to_evidence(_raw(0.95), band="low")
    assert ev.llr == 0.0
    assert ev.abstained and ev.reason == "uncalibrated_for_band"


def test_llr_is_finite_at_the_extremes():
    """Clipping must prevent infinite evidence from one detector."""
    c = _fit_separable()
    for s in (0.0, 1.0):
        ev = c.to_evidence(_raw(s), band="high")
        assert math.isfinite(ev.llr)


def test_separate_bands_are_calibrated_separately():
    rng = np.random.default_rng(1)
    n = 200
    # In 'high' the detector separates; in 'low' it is pure noise.
    hi_s = np.concatenate([rng.uniform(0, .4, n), rng.uniform(.6, 1, n)])
    lo_s = np.concatenate([rng.uniform(0, 1, n), rng.uniform(0, 1, n)])
    labels = np.concatenate([np.zeros(n), np.ones(n)])
    c = Calibrator(detector="d1")
    c.fit(np.concatenate([hi_s, lo_s]),
          np.concatenate([labels, labels]),
          ["high"] * 2 * n + ["low"] * 2 * n)
    hi = abs(c.to_evidence(_raw(0.95), band="high").llr)
    lo = abs(c.to_evidence(_raw(0.95), band="low").llr)
    assert hi > lo
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_calibration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dfd.calibration'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/dfd/calibration.py
"""Raw score → calibrated log-likelihood ratio, conditioned on quality band.

Spec §7.1. A detector's reliability at 512px uncompressed is not its
reliability at 96px after recompression, so one calibration curve per band.

A band with no fitted curve returns llr = 0.0 — the honest answer to "I was
never calibrated in this regime" is 'no information', not an extrapolation.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.linear_model import LogisticRegression

from .types import Evidence, RawScore

UNCALIBRATED = "uncalibrated_for_band"
# Caps any single detector's contribution. Prevents one saturated model from
# dominating the fused posterior (the failure mode seen in RD's cedar models).
MAX_ABS_LLR = 6.0
MIN_FIT_SAMPLES = 20


class Calibrator:
    def __init__(self, detector: str, max_abs_llr: float = MAX_ABS_LLR) -> None:
        self.detector = detector
        self.max_abs_llr = max_abs_llr
        self._models: dict[str, LogisticRegression] = {}
        self._priors: dict[str, float] = {}

    def fit(self, scores, labels, bands) -> "Calibrator":
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=int)
        bands = np.asarray(bands)
        for band in np.unique(bands):
            m = bands == band
            if m.sum() < MIN_FIT_SAMPLES or len(np.unique(labels[m])) < 2:
                continue
            lr = LogisticRegression()
            lr.fit(scores[m].reshape(-1, 1), labels[m])
            self._models[str(band)] = lr
            self._priors[str(band)] = float(labels[m].mean())
        return self

    def to_evidence(self, raw: RawScore, band: str) -> Evidence:
        if raw.abstained or raw.score is None:
            return Evidence(detector=raw.detector, detector_version=raw.version,
                            llr=0.0, raw_score=None, uncertainty=0.0,
                            abstained=True, reason=raw.reason,
                            artifacts=dict(raw.artifacts))

        model = self._models.get(band)
        if model is None:
            return Evidence(detector=raw.detector, detector_version=raw.version,
                            llr=0.0, raw_score=raw.score, uncertainty=0.0,
                            abstained=True, reason=UNCALIBRATED,
                            artifacts=dict(raw.artifacts))

        # Posterior log-odds minus the prior log-odds gives the likelihood ratio.
        post_logodds = float(model.decision_function([[raw.score]])[0])
        prior = self._priors[band]
        prior_logodds = math.log(prior / (1.0 - prior)) if 0 < prior < 1 else 0.0
        llr = post_logodds - prior_logodds
        llr = max(-self.max_abs_llr, min(self.max_abs_llr, llr))

        return Evidence(detector=raw.detector, detector_version=raw.version,
                        llr=llr, raw_score=raw.score,
                        uncertainty=0.0, abstained=False, reason="ok",
                        artifacts=dict(raw.artifacts))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_calibration.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add src/dfd/calibration.py tests/test_calibration.py
git commit -m "feat: quality-conditioned calibration from score to likelihood ratio"
```

---

### Task 10: Fusion with effective-sample-size discount

**Files:**
- Create: `src/dfd/fusion.py`
- Test: `tests/test_fusion.py`

**Interfaces:**
- Consumes: `Evidence`, `Verdict` (Task 1)
- Produces: `FusedResult`, `effective_sample_size(scores) -> float`, `fuse(evidence, n_frames, autocorr, thresholds) -> FusedResult`

This implements spec §9.5, the correlated-frames problem. **Naively summing LLRs across 900 correlated frames produces a posterior of ~1.0 regardless of truth** — the textbook way to build a system that is confidently wrong. The ESS discount is not optional.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fusion.py
import math
import numpy as np
import pytest
from dfd.fusion import FusedResult, effective_sample_size, fuse
from dfd.types import Evidence, Verdict


def _ev(llr, detector="d", abstained=False, reason="ok"):
    return Evidence(detector=detector, detector_version="1", llr=llr,
                    raw_score=None, uncertainty=0.0,
                    abstained=abstained, reason=reason)


def test_all_abstentions_yield_insufficient_evidence():
    r = fuse([_ev(0.0, abstained=True, reason="weights_absent")], n_frames=1)
    assert r.verdict is Verdict.INSUFFICIENT_EVIDENCE


def test_empty_evidence_yields_insufficient_evidence():
    assert fuse([], n_frames=1).verdict is Verdict.INSUFFICIENT_EVIDENCE


def test_positive_llrs_sum_towards_fake():
    r = fuse([_ev(2.0, "a"), _ev(2.0, "b")], n_frames=1)
    assert r.verdict is Verdict.FAKE and r.llr_total > 0


def test_negative_llrs_sum_towards_real():
    r = fuse([_ev(-2.0, "a"), _ev(-2.0, "b")], n_frames=1)
    assert r.verdict is Verdict.REAL and r.llr_total < 0


def test_abstentions_contribute_nothing():
    a = fuse([_ev(2.0, "a")], n_frames=1)
    b = fuse([_ev(2.0, "a"), _ev(0.0, "b", abstained=True, reason="x")], n_frames=1)
    assert a.llr_total == b.llr_total


def test_effective_sample_size_is_one_for_perfectly_correlated_frames():
    constant = np.ones(500)
    assert effective_sample_size(constant) == pytest.approx(1.0, abs=0.5)


def test_effective_sample_size_approaches_n_for_independent_frames():
    indep = np.random.default_rng(0).normal(size=500)
    ess = effective_sample_size(indep)
    assert ess > 100


def test_correlated_frames_do_not_produce_runaway_confidence():
    """900 correlated frames must not yield 900x the evidence of one frame."""
    one = fuse([_ev(1.0, "a")], n_frames=1, ess=1.0)
    many = fuse([_ev(1.0, "a")], n_frames=900, ess=20.0)
    assert many.llr_total < one.llr_total * 900
    assert many.llr_total <= _MAX_TOTAL


_MAX_TOTAL = 20.0


def test_total_llr_is_capped():
    r = fuse([_ev(6.0, f"d{i}") for i in range(20)], n_frames=1)
    assert abs(r.llr_total) <= _MAX_TOTAL


def test_disagreement_is_recorded_as_a_feature():
    """Spec §7.2: disagreement is signal, not noise to be averaged away."""
    r = fuse([_ev(3.0, "a"), _ev(-3.0, "b")], n_frames=1)
    assert r.disagreement > 0
    assert r.verdict is Verdict.OUT_OF_DISTRIBUTION


def test_reasons_are_carried_for_audit():
    r = fuse([_ev(0.0, "a", abstained=True, reason="below_quality_floor")], n_frames=1)
    assert "below_quality_floor" in r.reasons.values()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fusion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dfd.fusion'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/dfd/fusion.py
"""Combine Evidence into a decision (spec §7, §9.5).

Two properties matter more than the arithmetic:

1. Abstentions contribute exactly zero. An abstention is not a vote for 'real'.
2. Correlated frames are discounted by effective sample size. Summing LLRs over
   900 frames of the same pipeline, identity and lighting yields a posterior of
   ~1.0 regardless of truth — confidently wrong. ESS is mandatory, not a refinement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .types import Evidence, Verdict

MAX_TOTAL_LLR = 20.0
FAKE_THRESHOLD = 1.0
REAL_THRESHOLD = -1.0
# Evidence pulling hard in both directions means off-distribution, not 'average them'.
DISAGREEMENT_OOD = 4.0


@dataclass(frozen=True)
class FusedResult:
    verdict: Verdict
    llr_total: float
    posterior: float
    disagreement: float
    n_contributing: int
    ess: float
    reasons: dict = field(default_factory=dict)


def effective_sample_size(series) -> float:
    """ESS from lag-1 autocorrelation: n * (1 - rho) / (1 + rho).

    Perfectly correlated → 1. Independent → ~n.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < 2:
        return float(n)
    if np.std(x) < 1e-12:
        return 1.0
    xc = x - x.mean()
    rho = float(np.dot(xc[:-1], xc[1:]) / np.dot(xc, xc))
    rho = max(-0.999, min(0.999, rho))
    return max(1.0, n * (1.0 - rho) / (1.0 + rho))


def fuse(evidence, n_frames: int = 1, ess: float | None = None) -> FusedResult:
    reasons = {e.detector: e.reason for e in evidence}
    contributing = [e for e in evidence if not e.abstained]

    if not contributing:
        return FusedResult(verdict=Verdict.INSUFFICIENT_EVIDENCE, llr_total=0.0,
                           posterior=0.5, disagreement=0.0, n_contributing=0,
                           ess=0.0, reasons=reasons)

    llrs = np.array([e.llr for e in contributing], dtype=float)
    total = float(llrs.sum())

    # Discount for temporal correlation: evidence scales with independent
    # observations, not with frame count.
    if n_frames > 1:
        eff = ess if ess is not None else 1.0
        # LINEAR in ESS, not sqrt. Log-likelihood ratios ADD for independent
        # evidence, so n observations worth `eff` independent ones carry
        # eff * per_frame_llr -- a factor of eff/n. sqrt scaling belongs to
        # standard errors, not additive evidence. Under sqrt, 900 identical
        # frames still yielded 30.0, saturating the cap and reproducing the
        # very "confidently wrong" failure this discount exists to prevent.
        # NOTE: this applies ONLY when `evidence` is one entry PER FRAME.
        # Detectors in this repo aggregate internally (probs.mean()), so their
        # Evidence is already whole-sample -- call with n_frames=1 and no discount.
        total *= max(1.0, eff) / float(n_frames)

    total = max(-MAX_TOTAL_LLR, min(MAX_TOTAL_LLR, total))

    pos = float(llrs[llrs > 0].sum())
    neg = float(-llrs[llrs < 0].sum())
    disagreement = min(pos, neg)

    posterior = 1.0 / (1.0 + math.exp(-total))

    if disagreement >= DISAGREEMENT_OOD:
        verdict = Verdict.OUT_OF_DISTRIBUTION
    elif total >= FAKE_THRESHOLD:
        verdict = Verdict.FAKE
    elif total <= REAL_THRESHOLD:
        verdict = Verdict.REAL
    else:
        verdict = Verdict.INSUFFICIENT_EVIDENCE

    return FusedResult(verdict=verdict, llr_total=total, posterior=posterior,
                       disagreement=disagreement, n_contributing=len(contributing),
                       ess=float(ess or 1.0), reasons=reasons)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_fusion.py -v`
Expected: PASS, 11 tests

- [ ] **Step 5: Commit**

```bash
git add src/dfd/fusion.py tests/test_fusion.py
git commit -m "feat: evidence fusion with effective-sample-size discounting"
```

---

### Task 11: Benchmark metrics

**Files:**
- Create: `bench/__init__.py`, `bench/metrics.py`
- Test: `tests/bench/test_metrics.py`

**Interfaces:**
- Consumes: nothing from `dfd`
- Produces: `auc(scores, labels)`, `tpr_at_fpr(scores, labels, fpr)`, `ece(probs, labels, bins)`, `bootstrap_ci_by_group(scores, labels, groups, stat_fn, n, seed) -> (lo, hi)`

Spec §8.3. **TPR@FPR is the fraud-relevant metric, not accuracy.** Bootstrap must resample *groups* (videos), never rows — spec §8.2 guard 2.

- [ ] **Step 1: Write the failing test**

```python
# tests/bench/test_metrics.py
import numpy as np
import pytest
from bench.metrics import auc, bootstrap_ci_by_group, ece, tpr_at_fpr


def test_auc_is_one_for_perfect_separation():
    s = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([0, 0, 1, 1])
    assert auc(s, y) == pytest.approx(1.0)


def test_auc_is_half_for_no_separation():
    s = np.array([0.5, 0.5, 0.5, 0.5])
    y = np.array([0, 1, 0, 1])
    assert auc(s, y) == pytest.approx(0.5)


def test_tpr_at_fpr_is_one_for_perfect_separation():
    s = np.concatenate([np.linspace(0, .4, 100), np.linspace(.6, 1, 100)])
    y = np.concatenate([np.zeros(100), np.ones(100)])
    assert tpr_at_fpr(s, y, fpr=0.01) == pytest.approx(1.0)


def test_tpr_at_fpr_is_near_the_fpr_for_random_scores():
    rng = np.random.default_rng(0)
    s = rng.uniform(size=2000)
    y = rng.integers(0, 2, size=2000)
    assert tpr_at_fpr(s, y, fpr=0.1) < 0.25


def test_ece_is_zero_for_perfectly_calibrated_probabilities():
    probs = np.array([0.0] * 50 + [1.0] * 50)
    y = np.array([0] * 50 + [1] * 50)
    assert ece(probs, y, bins=10) == pytest.approx(0.0, abs=1e-9)


def test_ece_is_large_for_confidently_wrong_probabilities():
    probs = np.array([0.99] * 100)
    y = np.zeros(100, dtype=int)
    assert ece(probs, y, bins=10) > 0.9


def test_bootstrap_resamples_groups_not_rows():
    """10 videos x 100 frames must give wider CIs than 1000 independent rows.

    This is spec guard 2: frame-level bootstrapping fabricates precision.
    """
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(10), 100)
    per_video = rng.uniform(size=10)
    s = np.repeat(per_video, 100) + rng.normal(0, .01, 1000)
    y = np.repeat(rng.integers(0, 2, 10), 100)

    lo_g, hi_g = bootstrap_ci_by_group(s, y, groups, auc, n=200, seed=1)
    lo_r, hi_r = bootstrap_ci_by_group(s, y, np.arange(1000), auc, n=200, seed=1)
    assert (hi_g - lo_g) > (hi_r - lo_r)


def test_bootstrap_is_reproducible_given_a_seed():
    rng = np.random.default_rng(2)
    s = rng.uniform(size=200)
    y = rng.integers(0, 2, size=200)
    g = np.repeat(np.arange(20), 10)
    a = bootstrap_ci_by_group(s, y, g, auc, n=100, seed=7)
    b = bootstrap_ci_by_group(s, y, g, auc, n=100, seed=7)
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/bench/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench'`

- [ ] **Step 3: Write minimal implementation**

```python
# bench/__init__.py
"""Benchmark harness. Never imported from production paths."""
```

```python
# bench/metrics.py
"""Metrics that matter for fraud (spec §8.3).

Accuracy is not among them. At a 1-in-10,000 fraud base rate, a detector that
calls everything real is 99.99% accurate and worth nothing. TPR at a fixed,
operationally tolerable FPR is the number that decides whether this ships.
"""
from __future__ import annotations

import numpy as np


def auc(scores, labels) -> float:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # Average ranks within ties so constant scores give exactly 0.5.
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def tpr_at_fpr(scores, labels, fpr: float) -> float:
    """Highest TPR achievable without exceeding `fpr` on the negatives."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    neg, pos = s[y == 0], s[y == 1]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    thr = np.quantile(neg, 1.0 - fpr)
    return float((pos > thr).mean())


def ece(probs, labels, bins: int = 10) -> float:
    """Expected calibration error: mean |confidence - accuracy| over bins."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=int)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if not m.any():
            continue
        total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def bootstrap_ci_by_group(scores, labels, groups, stat_fn,
                          n: int = 1000, seed: int = 0,
                          alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI, resampling GROUPS with replacement (spec §8.2 guard 2).

    Resampling rows instead of videos fabricates precision: 10,000 frames from
    100 videos carry 100 videos' worth of information, not 10,000.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    g = np.asarray(groups)
    uniq = np.unique(g)
    index = {u: np.flatnonzero(g == u) for u in uniq}
    rng = np.random.default_rng(seed)

    stats: list[float] = []
    for _ in range(n):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index[d] for d in drawn])
        v = stat_fn(s[idx], y[idx])
        if np.isfinite(v):
            stats.append(float(v))
    if not stats:
        return (float("nan"), float("nan"))
    return (float(np.quantile(stats, alpha / 2)),
            float(np.quantile(stats, 1 - alpha / 2)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/bench/test_metrics.py -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Commit**

```bash
git add bench/__init__.py bench/metrics.py tests/bench/test_metrics.py
git commit -m "feat: fraud-relevant metrics with group-wise bootstrap CIs"
```

---

### Task 12: The six evaluation-hygiene guards

**Files:**
- Create: `bench/guards.py`
- Test: `tests/bench/test_guards.py`

**Interfaces:**
- Consumes: nothing
- Produces: `GuardViolation`, `IdentityReport`, `ParityReport`, `check_identity_disjoint(train_ids, test_ids, embeddings, threshold) -> IdentityReport`, `check_video_level(sample_ids, groups)`, `check_compression_coverage(records, required)`, `check_uniform_preprocessing(records)`, `check_threshold_provenance(threshold_source)`, `check_demographic_parity(scores, labels, strata, threshold, max_fpr_ratio) -> ParityReport`

Spec §8.2. **These are the difference between a real benchmark and a flattering one.** Each guard raises rather than warns — a benchmark that can be silently run dirty will be.

- [ ] **Step 1: Write the failing test**

```python
# tests/bench/test_guards.py
import numpy as np
import pytest
from bench.guards import (
    GuardViolation, check_compression_coverage, check_demographic_parity,
    check_identity_disjoint, check_threshold_provenance,
    check_uniform_preprocessing, check_video_level,
)


def test_identity_disjoint_passes_when_identities_do_not_overlap():
    emb = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    rep = check_identity_disjoint(["a"], ["b"], emb, threshold=0.9)
    assert rep.violations == 0
    assert rep.max_similarity < 0.9


def test_identity_disjoint_raises_on_the_same_person_in_both_splits():
    emb = {"a": np.array([1.0, 0.0]), "a_dup": np.array([0.99, 0.01])}
    with pytest.raises(GuardViolation) as exc:
        check_identity_disjoint(["a"], ["a_dup"], emb, threshold=0.9)
    assert "identity" in str(exc.value).lower()


def test_identity_report_carries_a_number_not_an_assertion():
    """Spec acceptance criterion 2: report the measurement, do not claim it."""
    emb = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    rep = check_identity_disjoint(["a"], ["b"], emb, threshold=0.9)
    assert isinstance(rep.max_similarity, float)
    assert rep.n_train == 1 and rep.n_test == 1


def test_video_level_raises_when_a_group_is_split_across_samples():
    with pytest.raises(GuardViolation):
        check_video_level(sample_ids=["s1", "s2"], groups=["v1", "v1"])


def test_video_level_passes_for_one_sample_per_group():
    check_video_level(sample_ids=["s1", "s2"], groups=["v1", "v2"])


def test_compression_coverage_raises_when_a_level_is_missing():
    recs = [{"compression": "c23"}, {"compression": "c23"}]
    with pytest.raises(GuardViolation) as exc:
        check_compression_coverage(recs, required=("c0", "c23", "c40"))
    assert "c0" in str(exc.value)


def test_compression_coverage_passes_when_all_present():
    recs = [{"compression": c} for c in ("c0", "c23", "c40")]
    check_compression_coverage(recs, required=("c0", "c23", "c40"))


def test_uniform_preprocessing_raises_on_mixed_detectors():
    recs = [{"face_detector": "yunet", "align": "v1", "label": 0},
            {"face_detector": "retinaface", "align": "v1", "label": 1}]
    with pytest.raises(GuardViolation) as exc:
        check_uniform_preprocessing(recs)
    assert "face_detector" in str(exc.value)


def test_uniform_preprocessing_passes_when_pipeline_is_identical():
    recs = [{"face_detector": "yunet", "align": "v1", "label": 0},
            {"face_detector": "yunet", "align": "v1", "label": 1}]
    check_uniform_preprocessing(recs)


def test_threshold_from_test_set_is_rejected():
    with pytest.raises(GuardViolation):
        check_threshold_provenance("test")


def test_threshold_from_validation_is_accepted():
    check_threshold_provenance("validation")


def test_demographic_parity_reports_a_spread_not_a_mean():
    """Guard 6: an aggregate FPR hides a group rejected three times as often."""
    scores = [0.1, 0.2, 0.9, 0.95] * 5
    labels = [0, 0, 0, 0] * 5
    strata = (["A"] * 2 + ["B"] * 2) * 5
    rep = check_demographic_parity(scores, labels, strata, threshold=0.5,
                                   max_fpr_ratio=100.0)
    assert set(rep.fpr_by_stratum) == {"A", "B"}
    assert rep.fpr_by_stratum["B"] > rep.fpr_by_stratum["A"]


def test_demographic_parity_raises_when_the_ratio_exceeds_the_ceiling():
    scores = [0.1, 0.1, 0.9, 0.9] * 5
    labels = [0, 0, 0, 0] * 5
    strata = (["A"] * 2 + ["B"] * 2) * 5
    with pytest.raises(GuardViolation) as exc:
        check_demographic_parity(scores, labels, strata, threshold=0.5,
                                 max_fpr_ratio=2.0)
    assert "fpr" in str(exc.value).lower()


def test_demographic_parity_passes_when_groups_are_treated_alike():
    scores = [0.1, 0.9, 0.1, 0.9] * 5
    labels = [0, 0, 0, 0] * 5
    strata = (["A"] * 2 + ["B"] * 2) * 5
    rep = check_demographic_parity(scores, labels, strata, threshold=0.5,
                                   max_fpr_ratio=2.0)
    assert rep.max_fpr_ratio == pytest.approx(1.0)


def test_demographic_parity_ignores_strata_with_no_negatives():
    """A stratum with no genuine samples has no measurable FPR; do not divide by zero."""
    scores = [0.1, 0.9, 0.9, 0.9]
    labels = [0, 0, 1, 1]
    strata = ["A", "A", "B", "B"]
    rep = check_demographic_parity(scores, labels, strata, threshold=0.5,
                                   max_fpr_ratio=2.0)
    assert "B" not in rep.fpr_by_stratum
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/bench/test_guards.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.guards'`

- [ ] **Step 3: Write minimal implementation**

```python
# bench/guards.py
"""The six evaluation-hygiene guards (spec §8.2).

Each raises rather than warns. A benchmark that can be silently run dirty will
be run dirty, and every one of these failures inflates results in the flattering
direction — which is exactly why they are easy to leave out.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class GuardViolation(Exception):
    """Raised when an evaluation-hygiene guard fails."""


@dataclass(frozen=True)
class IdentityReport:
    n_train: int
    n_test: int
    max_similarity: float
    violations: int
    threshold: float


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def check_identity_disjoint(train_ids, test_ids, embeddings: dict,
                            threshold: float = 0.6) -> IdentityReport:
    """Guard 1 — identity leakage.

    The same person in train and test teaches the model faces, not forgery.
    Returns a measured report; spec acceptance criterion 2 requires the number,
    not an assertion that it was checked.
    """
    max_sim = 0.0
    violations = 0
    for a in train_ids:
        for b in test_ids:
            if a not in embeddings or b not in embeddings:
                continue
            sim = _cosine(embeddings[a], embeddings[b])
            max_sim = max(max_sim, sim)
            if sim >= threshold:
                violations += 1
    if violations:
        raise GuardViolation(
            f"identity leakage: {violations} train/test pairs at cosine "
            f">= {threshold} (max {max_sim:.4f})")
    return IdentityReport(n_train=len(train_ids), n_test=len(test_ids),
                          max_similarity=max_sim, violations=0,
                          threshold=threshold)


def check_video_level(sample_ids, groups) -> None:
    """Guard 2 — one sample per source video.

    Frames must be aggregated before scoring. 10,000 frames from 100 videos is
    100 independent samples; treating them as 10,000 inflates AUC and shrinks
    confidence intervals dishonestly.
    """
    seen: dict = {}
    for sid, g in zip(sample_ids, groups):
        if g in seen and seen[g] != sid:
            raise GuardViolation(
                f"group {g!r} appears in multiple samples ({seen[g]!r}, {sid!r}); "
                "aggregate to video level before scoring")
        seen[g] = sid


def check_compression_coverage(records, required=("c0", "c23", "c40")) -> None:
    """Guard 3 — evaluate across compression levels, report the worst."""
    present = {r.get("compression") for r in records}
    missing = [c for c in required if c not in present]
    if missing:
        raise GuardViolation(
            f"compression levels missing from evaluation: {', '.join(missing)}")


def check_uniform_preprocessing(records) -> None:
    """Guard 4 — one preprocessing pipeline, applied blind to label.

    Different face detectors on real vs fake is itself a giveaway the model will
    happily learn.
    """
    for key in ("face_detector", "align"):
        by_label: dict = {}
        for r in records:
            by_label.setdefault(r["label"], set()).add(r.get(key))
        all_values = set().union(*by_label.values()) if by_label else set()
        if len(all_values) > 1:
            raise GuardViolation(
                f"non-uniform preprocessing: {key} takes values {sorted(map(str, all_values))}")


def check_threshold_provenance(source: str) -> None:
    """Guard 5 — operating points frozen on validation, never chosen on test."""
    if source != "validation":
        raise GuardViolation(
            f"threshold source must be 'validation', got {source!r}")


@dataclass(frozen=True)
class ParityReport:
    fpr_by_stratum: dict
    tpr_by_stratum: dict
    max_fpr_ratio: float
    ceiling: float


def check_demographic_parity(scores, labels, strata, threshold: float,
                             max_fpr_ratio: float = 2.0) -> ParityReport:
    """Guard 6 — per-stratum error parity (spec §8.2 guard 6).

    An aggregate FPR of 1% is compatible with 0.3% on one group and 4% on
    another. The applicants wrongly rejected are not distributed evenly, and
    the aggregate is precisely the statistic that conceals it.

    Reports the spread and fails when the inter-stratum FPR ratio exceeds the
    ceiling. Strata with no genuine (negative) samples have no measurable FPR
    and are excluded rather than assumed clean.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    g = np.asarray(strata)

    fpr: dict[str, float] = {}
    tpr: dict[str, float] = {}
    for stratum in np.unique(g):
        m = g == stratum
        neg = m & (y == 0)
        pos = m & (y == 1)
        if neg.sum() > 0:
            fpr[str(stratum)] = float((s[neg] > threshold).mean())
        if pos.sum() > 0:
            tpr[str(stratum)] = float((s[pos] > threshold).mean())

    ratio = 1.0
    if len(fpr) >= 2:
        values = [v for v in fpr.values()]
        lo = min(values)
        hi = max(values)
        # An absolute floor keeps a 0%-vs-0.5% difference from reading as infinite.
        ratio = hi / max(lo, 1e-3)

    report = ParityReport(fpr_by_stratum=fpr, tpr_by_stratum=tpr,
                          max_fpr_ratio=float(ratio), ceiling=max_fpr_ratio)
    if ratio > max_fpr_ratio:
        worst = max(fpr, key=fpr.get)
        best = min(fpr, key=fpr.get)
        raise GuardViolation(
            f"demographic FPR disparity {ratio:.2f}x exceeds ceiling "
            f"{max_fpr_ratio:.2f}x: {worst}={fpr[worst]:.4f} vs "
            f"{best}={fpr[best]:.4f}")
    return report
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/bench/test_guards.py -v`
Expected: PASS, 15 tests

- [ ] **Step 5: Commit**

```bash
git add bench/guards.py tests/bench/test_guards.py
git commit -m "feat: six evaluation-hygiene guards that raise rather than warn"
```

---

### Task 13: Leave-one-generator-out protocol

**Files:**
- Create: `bench/protocol.py`
- Test: `tests/bench/test_protocol.py`

**Interfaces:**
- Consumes: nothing (guards in `bench/guards.py` check the *result*; this module makes the result correct by construction)
- Produces: `Split`, `logo_splits(records, seed) -> list[Split]`

Spec §8.1. **This is the only number that predicts field performance**, because the field always brings an unseen generator.

**Record schema.** Every record carries `sample_id`, `subject_id`, `source_id`, `generator`, `label`. `source_id` is the source video (an image is its own source, recorded explicitly — never aliased). `generator` is `None` for reals and a non-empty string for fakes.

**Why subjects, not records, are partitioned.** Spec §8.1 asks for leave-one-generator-out; spec §8.2 guards 1 and 2 ask for identity-disjoint, video-level splits. Those are not independent: in every real forgery corpus one subject is faked by *several* generators, so assigning fakes to sides by generator alone puts subject `pA`'s faceswap video in train and their deepfacelive video in test. The split then teaches faces. Partitioning **subjects** and dropping the fakes whose generator wants one side while their subject sits on the other is the only assignment that satisfies both. The drops are real and are reported, not hidden: on a corpus where every subject is faked by every generator, half the fakes fall out of each fold. A reader who does not see that number will over-read the fold.

- [ ] **Step 1: Write the failing test**

```python
# tests/bench/test_protocol.py
import pytest

from bench.protocol import Split, logo_splits


def rec(sample_id, subject_id, generator, label, source_id=None):
    return {"sample_id": sample_id, "subject_id": subject_id,
            "source_id": source_id or sample_id,
            "generator": generator, "label": label}


# Six real subjects, and five fake subjects each faked by two or three
# generators. The repeated subjects are the point: a splitter that assigns
# fakes by generator alone leaks all five.
RECORDS = [rec(f"r{i}", f"p{i}", None, 0) for i in range(1, 7)] + [
    rec("f1", "pA", "deepfacelive", 1), rec("f2", "pA", "faceswap", 1),
    rec("f3", "pB", "deepfacelive", 1), rec("f4", "pB", "stylegan", 1),
    rec("f5", "pC", "faceswap", 1), rec("f6", "pC", "stylegan", 1),
    rec("f7", "pD", "deepfacelive", 1), rec("f8", "pD", "faceswap", 1),
    rec("f9", "pD", "stylegan", 1),
    rec("f10", "pE", "deepfacelive", 1), rec("f11", "pE", "faceswap", 1),
    rec("f12", "pE", "stylegan", 1),
]

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]


def test_one_split_per_generator():
    splits = logo_splits(RECORDS)
    assert {s.held_out_generator for s in splits} == {
        "deepfacelive", "faceswap", "stylegan"}
    assert len(splits) == 3


@pytest.mark.parametrize("seed", SEEDS)
def test_held_out_generator_never_appears_in_train(seed):
    for s in logo_splits(RECORDS, seed=seed):
        gens = {r["generator"] for r in s.train if r["label"] == 1}
        assert s.held_out_generator not in gens


@pytest.mark.parametrize("seed", SEEDS)
def test_held_out_generator_is_the_only_fake_generator_in_test(seed):
    for s in logo_splits(RECORDS, seed=seed):
        gens = {r["generator"] for r in s.test if r["label"] == 1}
        assert gens == {s.held_out_generator}


@pytest.mark.parametrize("seed", SEEDS)
def test_both_sides_carry_reals_and_fakes(seed):
    """No test reals means no FPR; no test fakes means no TPR."""
    for s in logo_splits(RECORDS, seed=seed):
        for side in (s.train, s.test):
            assert any(r["label"] == 0 for r in side)
            assert any(r["label"] == 1 for r in side)


@pytest.mark.parametrize("seed", SEEDS)
def test_subjects_are_disjoint_across_the_split_including_fake_subjects(seed):
    """Spec §8.2 guard 1. RECORDS fakes each subject with several
    generators, so a by-generator-only assignment fails this."""
    for s in logo_splits(RECORDS, seed=seed):
        tr = {r["subject_id"] for r in s.train}
        te = {r["subject_id"] for r in s.test}
        assert tr.isdisjoint(te), f"{s.held_out_generator}: leaked {tr & te}"


@pytest.mark.parametrize("seed", SEEDS)
def test_no_source_video_straddles_the_split(seed):
    """Spec §8.2 guard 2: a source video belongs wholly to one side."""
    for s in logo_splits(RECORDS, seed=seed):
        tr = {r["source_id"] for r in s.train}
        te = {r["source_id"] for r in s.test}
        assert tr.isdisjoint(te)


@pytest.mark.parametrize("seed", SEEDS)
def test_every_record_is_placed_or_reported_dropped(seed):
    """Nothing vanishes silently."""
    for s in logo_splits(RECORDS, seed=seed):
        placed = s.train_ids() + s.test_ids() + s.dropped_ids()
        assert sorted(placed) == sorted(r["sample_id"] for r in RECORDS)
        assert len(placed) == len(set(placed))


def test_dropped_records_are_exactly_the_identity_conflicts():
    """Each drop is a fake whose generator wants the side its subject is not
    on. On this corpus that is a large fraction, which is why it is counted."""
    for s in logo_splits(RECORDS, seed=0):
        test_subjects = {r["subject_id"] for r in s.test}
        for r in s.dropped_for_identity:
            assert r["label"] == 1
            wants_test = r["generator"] == s.held_out_generator
            assert wants_test != (r["subject_id"] in test_subjects)
        assert len(s.dropped_for_identity) > 0


def test_split_is_deterministic_given_a_seed():
    a = logo_splits(RECORDS, seed=5)
    b = logo_splits(RECORDS, seed=5)
    assert [s.test_ids() for s in a] == [s.test_ids() for s in b]


def test_seed_actually_changes_the_partition():
    """Guards against an implementation that accepts `seed` and ignores it."""
    by_seed = {tuple(tuple(s.test_ids()) for s in logo_splits(RECORDS, seed=k))
               for k in SEEDS}
    assert len(by_seed) > 1


def test_records_without_a_generator_key_are_rejected():
    bad = [{"sample_id": "x", "subject_id": "p", "source_id": "x", "label": 1}]
    with pytest.raises(KeyError, match="missing required"):
        logo_splits(bad)


def test_records_without_a_source_id_are_rejected():
    bad = [{"sample_id": "x", "subject_id": "p", "generator": "g", "label": 1}]
    with pytest.raises(KeyError, match="missing required"):
        logo_splits(bad)


def test_a_fake_with_no_generator_is_rejected():
    """An unattributed fake would silently join the training side of EVERY
    split, which is the one place it can never be measured."""
    bad = [rec("r1", "p1", None, 0), rec("f1", "pA", None, 1)]
    with pytest.raises(ValueError, match="unattributed fake"):
        logo_splits(bad)


def test_a_corpus_with_one_subject_is_rejected():
    """Better to refuse than to emit a split with nothing on one side."""
    bad = [rec("r1", "pA", None, 0), rec("f1", "pA", "deepfacelive", 1)]
    with pytest.raises(ValueError, match="needs at least 2"):
        logo_splits(bad)


def test_a_single_generator_corpus_is_rejected():
    """Holding out the only generator leaves nothing to train on."""
    bad = [rec("r1", "p1", None, 0), rec("r2", "p2", None, 0),
           rec("f1", "pA", "deepfacelive", 1), rec("f2", "pB", "deepfacelive", 1)]
    with pytest.raises(ValueError, match="train fakes"):
        logo_splits(bad)


def test_a_source_carrying_two_subjects_is_rejected():
    bad = [rec("a", "p1", None, 0, source_id="v"),
           rec("b", "p2", None, 0, source_id="v"),
           rec("f", "pA", "deepfacelive", 1)]
    with pytest.raises(ValueError, match="more than one subject/generator"):
        logo_splits(bad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/bench/test_protocol.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.protocol'`

- [ ] **Step 3: Write minimal implementation**

```python
# bench/protocol.py
"""Leave-one-generator-out split construction (spec §8.1).

In-dataset AUC measures memorisation. LOGO measures what happens when a
generator the model has never seen walks through the door — which, in the
field, is every generator eventually.

Two spec §8.2 guards are structural here rather than checked after the fact:
identity disjointness (guard 1) and video-level integrity (guard 2). A split
that leaks a subject or straddles a source video produces a number that cannot
be repaired downstream, so this module refuses to emit one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

REQUIRED_KEYS = ("sample_id", "subject_id", "source_id", "generator", "label")


@dataclass(frozen=True)
class Split:
    """One fold: every fake in `test` comes from `held_out_generator`."""

    held_out_generator: str
    train: list[dict]
    test: list[dict]
    dropped_for_identity: list[dict] = field(default_factory=list)

    def test_ids(self) -> list[str]:
        return [r["sample_id"] for r in self.test]

    def train_ids(self) -> list[str]:
        return [r["sample_id"] for r in self.train]

    def dropped_ids(self) -> list[str]:
        return [r["sample_id"] for r in self.dropped_for_identity]


def _validate(records: list[dict]) -> None:
    for r in records:
        missing = [k for k in REQUIRED_KEYS if k not in r]
        if missing:
            raise KeyError(
                f"record {r.get('sample_id')!r} is missing required "
                f"key(s) {missing}")
        if r["label"] == 1 and not r["generator"]:
            raise ValueError(
                f"fake record {r['sample_id']!r} has no generator; an "
                "unattributed fake would join the training side of every split")

    by_source: dict[str, set] = {}
    for r in records:
        by_source.setdefault(r["source_id"], set()).add(
            (r["subject_id"], r["generator"]))
    for src, pairs in sorted(by_source.items()):
        if len(pairs) > 1:
            raise ValueError(
                f"source {src!r} carries more than one subject/generator pair: "
                f"{sorted(map(str, pairs))}. A source that straddles cannot be "
                "assigned to one side of a split")


def _require_measurable(
    held_out: str, train: list[dict], test: list[dict]
) -> None:
    counts = {
        "train reals": sum(1 for r in train if r["label"] == 0),
        "test reals": sum(1 for r in test if r["label"] == 0),
        "train fakes": sum(1 for r in train if r["label"] == 1),
        "test fakes": sum(1 for r in test if r["label"] == 1),
    }
    empty = sorted(k for k, v in counts.items() if v == 0)
    if empty:
        raise ValueError(
            f"split holding out {held_out!r} has no {' and no '.join(empty)} "
            f"(counts={counts}); without test reals there is no FPR to measure "
            "and without test fakes there is no TPR")


def logo_splits(records: list[dict], seed: int = 0) -> list[Split]:
    """One split per generator, identity-disjoint and video-whole.

    Subjects — not records — are partitioned, so a subject faked by several
    generators cannot appear on both sides. Fakes whose generator wants one
    side while their subject sits on the other are dropped and reported in
    `Split.dropped_for_identity` rather than silently leaked.
    """
    _validate(records)

    subjects = sorted({r["subject_id"] for r in records})
    if len(subjects) < 2:
        raise ValueError(
            f"corpus has {len(subjects)} distinct subject(s); an "
            "identity-disjoint split needs at least 2")

    rng = np.random.default_rng(seed)
    shuffled = [subjects[i] for i in rng.permutation(len(subjects))]
    cut = max(1, len(shuffled) // 2)
    train_subjects = set(shuffled[:cut])

    generators = sorted({r["generator"] for r in records if r["label"] == 1})

    splits: list[Split] = []
    for g in generators:
        train: list[dict] = []
        test: list[dict] = []
        dropped: list[dict] = []
        for r in records:
            subject_side_is_test = r["subject_id"] not in train_subjects
            if r["label"] == 0:
                (test if subject_side_is_test else train).append(r)
                continue
            belongs_in_test = r["generator"] == g
            if belongs_in_test == subject_side_is_test:
                (test if belongs_in_test else train).append(r)
            else:
                dropped.append(r)
        _require_measurable(g, train, test)
        splits.append(Split(g, train, test, dropped))
    return splits
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/bench/test_protocol.py -v`
Expected: PASS, 60 tests (the parametrised ones run once per seed)

- [ ] **Step 5: Prove the identity test can fail**

The identity guard is the one that has historically shipped vacuous. Prove it fires: temporarily replace the fake-assignment branch in `logo_splits` with the unconditional form

```python
            belongs_in_test = r["generator"] == g
            (test if belongs_in_test else train).append(r)
```

run `pytest tests/bench/test_protocol.py -k disjoint`, and confirm it FAILS reporting leaked subjects. Restore, confirm it passes. Record both outputs in the report.

- [ ] **Step 6: Commit**

```bash
git add bench/protocol.py tests/bench/test_protocol.py
git commit -m "feat: leave-one-generator-out split construction"
```

---

### Task 14: Robustness surface

**Files:**
- Create: `bench/robustness.py`
- Test: `tests/bench/test_robustness.py`

**Interfaces:**
- Consumes: nothing
- Produces: `PERTURBATIONS: dict[str, Callable]`, `JPEG_QUALITIES`, `apply_perturbation(img, name, **kw) -> np.ndarray`, `robustness_sweep(img) -> dict[str, np.ndarray]`

Spec §8.3 and acceptance criterion 9. **Screenshot-of-screen and print-recapture are the cheapest laundering steps available to any adversary** and are routinely skipped in published evaluations.

**Two things this task must get right, both measured rather than assumed.**

*The sweep is a curve, not a point.* Spec §8.3 asks for a "JPEG quality sweep". One JPEG at one quality cannot show where a detector falls off, which is the only thing the sweep is for. `robustness_sweep` emits one entry per quality in `JPEG_QUALITIES`.

*Each perturbation is asserted to do the thing its name claims.* Shape-and-dtype assertions pass against a stub that does no work — a defect class this codebase has shipped repeatedly. The mechanism here is high-frequency energy, because that is the evidence NPR and every frequency-domain detector depends on. Measured on a structured 128×128 test image (Laplacian mean, ratio to clean, min/max over five jittered draws):

| perturbation | ratio | assertion |
|---|---|---|
| `resize` | 0.231–0.233 | `< 0.5` |
| `blur` | 0.130–0.131 | `< 0.3` |
| `noise` | 1.018–1.022 | `> 1.0` — noise **adds** high frequency |
| `screenshot_recapture` | 0.360–0.364 | `< 0.6` |
| `print_recapture` | 0.307–0.309 | `< 0.6` |

**JPEG is deliberately absent from that table.** Its blocking artefacts add edges at block boundaries, so high-frequency energy across q=90→10 runs 0.97, 0.98, 0.97, 0.95, 1.00 — flat and non-monotonic. JPEG is asserted on *distortion* instead, which is monotonic over the same sweep (mean |diff| 14.77, 14.82, 15.24, 17.32, 19.08). Do not "fix" the JPEG case by moving it into the energy table.

The fixture is a structured image — gradients, a block edge, fine scanlines — not uniform noise. On uniform noise JPEG *raises* high-frequency energy (1.10×), so any mechanism assertion written against a noise fixture measures the fixture.

- [ ] **Step 1: Write the failing test**

```python
# tests/bench/test_robustness.py
import cv2
import numpy as np
import pytest

from bench.robustness import (
    JPEG_QUALITIES, PERTURBATIONS, apply_perturbation, robustness_sweep,
)


def _img(h=128, w=128):
    """Structured: two gradients, a hard block edge, and fine scanlines.

    Uniform noise is the wrong fixture here — JPEG raises high-frequency
    energy on it, so mechanism assertions would measure the fixture.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    img[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    img[h // 4:3 * h // 4, w // 4:3 * w // 4] = 220
    img[::4, :] = 40
    return img


def _hf_energy(img):
    """Mean |Laplacian| — the high-frequency evidence detectors depend on."""
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.abs(cv2.Laplacian(grey, cv2.CV_32F)).mean())


def _distortion(a, b):
    return float(np.abs(a.astype(np.int32) - b.astype(np.int32)).mean())


def test_every_perturbation_preserves_shape_and_dtype():
    img = _img()
    for name in PERTURBATIONS:
        out = apply_perturbation(img, name)
        assert out.shape == img.shape, name
        assert out.dtype == np.uint8, name


def test_every_perturbation_actually_changes_the_image():
    img = _img()
    for name in PERTURBATIONS:
        assert not np.array_equal(apply_perturbation(img, name), img), name


@pytest.mark.parametrize("name,limit", [
    ("resize", 0.5), ("blur", 0.3),
    ("screenshot_recapture", 0.6), ("print_recapture", 0.6),
])
def test_perturbation_destroys_high_frequency_evidence(name, limit):
    """The mechanism, not just 'the pixels changed'. Measured ratios are
    0.23, 0.13, 0.36 and 0.31 — these limits carry real margin."""
    img = _img()
    assert _hf_energy(apply_perturbation(img, name)) < limit * _hf_energy(img)


def test_noise_adds_high_frequency_rather_than_removing_it():
    """Asserting every perturbation lowers HF energy would be wrong."""
    img = _img()
    assert _hf_energy(apply_perturbation(img, "noise")) > _hf_energy(img)


def test_jpeg_distortion_rises_monotonically_as_quality_falls():
    """JPEG is asserted on distortion, not high-frequency energy: its
    blocking artefacts ADD edges, so HF energy is flat and non-monotonic
    across the sweep."""
    img = _img()
    qualities = sorted(JPEG_QUALITIES, reverse=True)
    d = [_distortion(apply_perturbation(img, "jpeg", quality=q), img)
         for q in qualities]
    assert d == sorted(d), dict(zip(qualities, d))
    assert d[-1] > d[0]


def test_sweep_covers_the_whole_jpeg_quality_curve():
    """Spec §8.3 asks for a sweep. One point is not a curve."""
    out = robustness_sweep(_img())
    for q in JPEG_QUALITIES:
        assert f"jpeg_q{q}" in out
    assert len(JPEG_QUALITIES) >= 4


def test_sweep_returns_clean_plus_every_non_jpeg_perturbation():
    out = robustness_sweep(_img())
    assert "clean" in out
    assert np.array_equal(out["clean"], _img())
    for name in PERTURBATIONS:
        if name != "jpeg":
            assert name in out, name


def test_sweep_is_deterministic():
    """The harness's whole value is reproducibility; two perturbations draw
    from RNGs and nothing else pins them."""
    a, b = robustness_sweep(_img()), robustness_sweep(_img())
    assert set(a) == set(b)
    for k in a:
        assert np.array_equal(a[k], b[k]), k


def test_unknown_perturbation_raises_naming_the_unknown_name():
    with pytest.raises(KeyError, match="teleport"):
        apply_perturbation(_img(), "teleport")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/bench/test_robustness.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.robustness'`

- [ ] **Step 3: Write minimal implementation**

```python
# bench/robustness.py
"""Perturbation surface (spec §8.3, acceptance criterion 9).

Includes the two physical re-capture paths that published evaluations usually
skip and that any adversary can perform for free: photographing a screen, and
printing then re-photographing. Both destroy roughly two thirds of the image's
high-frequency energy, which is the evidence most detectors depend on.

Every perturbation here is a pure function of its input: the two that draw
noise take an explicit seed with a fixed default, so a sweep is reproducible.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

#: The JPEG quality curve. Spec §8.3 asks for a sweep; one point is not a curve.
JPEG_QUALITIES = (90, 70, 50, 30, 10)


def _jpeg(img: np.ndarray, quality: int = 50) -> np.ndarray:
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _resize(img: np.ndarray, scale: float = 0.5) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _blur(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def _noise(img: np.ndarray, sigma: float = 8.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def _screenshot_recapture(img: np.ndarray) -> np.ndarray:
    """Photographing a screen: resample, moiré, glare gradient, recompress.

    Deterministic by construction — the moiré and glare are analytic, so this
    takes no seed.
    """
    out = _resize(img, 0.7)
    h, w = out.shape[:2]
    yy = np.arange(h)[:, None]
    moire = (8.0 * np.sin(2 * np.pi * yy / 3.0)).astype(np.float32)
    out = np.clip(out.astype(np.float32) + moire, 0, 255).astype(np.uint8)
    glare = np.linspace(1.0, 1.12, w, dtype=np.float32)[None, :, None]
    out = np.clip(out.astype(np.float32) * glare, 0, 255).astype(np.uint8)
    return _jpeg(out, quality=70)


def _print_recapture(img: np.ndarray, seed: int = 1) -> np.ndarray:
    """Print then photograph: soft focus, gamut loss, halftone, paper grain."""
    out = _blur(img, 3).astype(np.float32)
    out = np.clip((out - 16.0) * (255.0 / (235.0 - 16.0)), 0, 255)  # gamut
    out = np.round(out / 16.0) * 16.0                               # halftone
    rng = np.random.default_rng(seed)
    out = out + rng.normal(0, 4.0, out.shape)                       # paper grain
    return _jpeg(np.clip(out, 0, 255).astype(np.uint8), quality=75)


PERTURBATIONS: dict[str, Callable[..., np.ndarray]] = {
    "jpeg": _jpeg,
    "resize": _resize,
    "blur": _blur,
    "noise": _noise,
    "screenshot_recapture": _screenshot_recapture,
    "print_recapture": _print_recapture,
}


def apply_perturbation(img: np.ndarray, name: str, **kwargs) -> np.ndarray:
    if name not in PERTURBATIONS:
        raise KeyError(
            f"unknown perturbation: {name!r}; known: {sorted(PERTURBATIONS)}")
    return PERTURBATIONS[name](img, **kwargs)


def robustness_sweep(img: np.ndarray) -> dict[str, np.ndarray]:
    """Clean, the full JPEG quality curve, and one entry per other perturbation.

    JPEG is expanded over `JPEG_QUALITIES` rather than sampled once, because a
    single quality cannot show where a detector falls off.
    """
    out: dict[str, np.ndarray] = {"clean": img}
    for quality in JPEG_QUALITIES:
        out[f"jpeg_q{quality}"] = _jpeg(img, quality=quality)
    for name, fn in PERTURBATIONS.items():
        if name == "jpeg":
            continue
        out[name] = fn(img)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/bench/test_robustness.py -v`
Expected: PASS, 12 tests (the parametrised mechanism test runs four times).

- [ ] **Step 5: Prove the mechanism assertions can fail**

The mechanism tests are the ones carrying this task's claim. Prove they fire: temporarily make `_screenshot_recapture` and `_print_recapture` return `img` unchanged, run `pytest tests/bench/test_robustness.py -k high_frequency`, and confirm BOTH fail. Restore and confirm they pass. Record both outputs in the report.

- [ ] **Step 6: Commit**

```bash
git add bench/robustness.py tests/bench/test_robustness.py
git commit -m "feat: robustness surface including screen and print re-capture"
```

---

### Task 15: White-box adversarial baseline

**Files:**
- Create: `bench/adversarial.py`
- Test: `tests/bench/test_adversarial.py`

**Interfaces:**
- Consumes: `tpr_at_fpr` (Task 11)
- Produces: `pgd_attack(model, x, y, eps, alpha, steps, seed) -> torch.Tensor`, `adversarial_tpr(model, x, y, eps, fpr) -> float`

Spec acceptance criterion 8 and §3A. **This is the criterion that makes the state-sponsored threat model real rather than decorative.** A detector whose adversarial TPR is ~0 is recorded as such and demoted to evidence-only.

**The fixture is tuned so that a working attack collapses the number, and a broken one does not.** This matters more than it sounds. The obvious formulation — assert `attacked <= clean` — is satisfied by a `pgd_attack` that returns its input unchanged, because then `attacked == clean`. Worse, the obvious fixture (negatives at mean pixel 0.2, positives at 0.8, eps=0.3) cannot be attacked at all: the attack moves positives the full 0.3 to 0.50, which still ranks above the clean negatives at 0.20, so TPR stays 1.0 even with a *perfect* attack. Both were measured. With negatives at 0.45, positives at 0.55 and eps=0.2:

| attack | TPR@FPR=0.1 |
|---|---|
| none (clean) | 1.0 |
| real PGD | **0.0** |
| a no-op `pgd_attack` | 1.0 — the test must fail here |

So the assertions are exact values, not an inequality.

**PGD gets a real random start.** Without one this is BIM, not PGD, and — measured — `seed=3` and `seed=999` produce byte-identical tensors, so the `seed` parameter is dead code and its determinism test cannot fail. The start is drawn from an explicit `torch.Generator`, not the global `torch.manual_seed`, so running the attack does not perturb global RNG state for every other test in the suite.

- [ ] **Step 1: Write the failing test**

```python
# tests/bench/test_adversarial.py
import pytest
import torch
import torch.nn as nn

from bench.adversarial import adversarial_tpr, pgd_attack


class TinyModel(nn.Module):
    """A trivially attackable linear detector: mean pixel above 0.5 => fake."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 8 * 8, 2)
        with torch.no_grad():
            self.fc.weight.copy_(torch.cat([
                -torch.ones(1, 192) / 192.0,
                torch.ones(1, 192) / 192.0]))
            self.fc.bias.copy_(torch.tensor([0.5, -0.5]))

    def forward(self, x):
        return self.fc(x.flatten(1))


@pytest.fixture
def model():
    m = TinyModel()
    m.eval()
    return m


def _batch(value, n=8):
    return torch.full((n, 3, 8, 8), value, dtype=torch.float32)


def _labelled():
    """Negatives at 0.45, positives at 0.55 — close enough that eps=0.2
    genuinely flips the ranking. A wider gap makes the attack unmeasurable."""
    x = torch.cat([_batch(0.45, 16), _batch(0.55, 16)])
    y = torch.cat([torch.zeros(16, dtype=torch.long),
                   torch.ones(16, dtype=torch.long)])
    return x, y


def test_pgd_output_stays_within_the_epsilon_ball(model):
    x = _batch(0.5)
    adv = pgd_attack(model, x, torch.ones(8, dtype=torch.long),
                     eps=0.03, alpha=0.01, steps=5)
    assert torch.max(torch.abs(adv - x)).item() <= 0.03 + 1e-6


def test_pgd_output_stays_in_valid_pixel_range(model):
    x = _batch(0.99)
    adv = pgd_attack(model, x, torch.ones(8, dtype=torch.long),
                     eps=0.1, alpha=0.02, steps=5)
    assert adv.min().item() >= 0.0 and adv.max().item() <= 1.0


def test_pgd_actually_moves_the_input(model):
    """A no-op attack must not be able to reach the later assertions."""
    x = _batch(0.5)
    adv = pgd_attack(model, x, torch.ones(8, dtype=torch.long),
                     eps=0.1, alpha=0.02, steps=5)
    assert not torch.equal(adv, x)


def test_pgd_reduces_confidence_on_the_true_class(model):
    x = _batch(0.9)
    y = torch.ones(8, dtype=torch.long)
    before = torch.softmax(model(x), 1)[:, 1].mean().item()
    after = torch.softmax(model(pgd_attack(model, x, y, eps=0.2, alpha=0.05,
                                           steps=20)), 1)[:, 1].mean().item()
    assert after < before - 0.05


def test_pgd_is_deterministic_given_a_seed(model):
    x, y = _batch(0.7), torch.ones(8, dtype=torch.long)
    kw = dict(eps=0.1, alpha=0.02, steps=5)
    assert torch.equal(pgd_attack(model, x, y, seed=3, **kw),
                       pgd_attack(model, x, y, seed=3, **kw))


def test_pgd_seed_is_load_bearing(model):
    """Without a random start the seed is dead code and the determinism
    test above passes for any implementation, seeded or not."""
    x, y = _batch(0.7), torch.ones(8, dtype=torch.long)
    kw = dict(eps=0.1, alpha=0.02, steps=5)
    assert not torch.equal(pgd_attack(model, x, y, seed=3, **kw),
                           pgd_attack(model, x, y, seed=999, **kw))


def test_pgd_does_not_disturb_global_torch_rng(model):
    """The attack must not reseed the RNG every other test draws from."""
    torch.manual_seed(1234)
    expected = torch.randn(4)
    torch.manual_seed(1234)
    pgd_attack(model, _batch(0.5), torch.ones(8, dtype=torch.long),
               eps=0.1, alpha=0.02, steps=5, seed=7)
    assert torch.equal(torch.randn(4), expected)


def test_clean_tpr_is_perfect_on_this_fixture(model):
    """Anchors the collapse below: without this, 'attacked == 0.0' could
    mean the detector never worked."""
    x, y = _labelled()
    assert adversarial_tpr(model, x, y, eps=0.0, fpr=0.1) == 1.0


def test_attack_collapses_tpr_to_zero(model):
    """Acceptance criterion 8. Exact values, not `attacked <= clean` — that
    inequality is satisfied by an attack that does nothing at all."""
    x, y = _labelled()
    assert adversarial_tpr(model, x, y, eps=0.2, fpr=0.1) == 0.0


def test_negatives_are_left_clean(model):
    """The adversary wants fakes to read as real, not the reverse; attacking
    negatives too would understate the detector by moving the threshold."""
    x, y = _labelled()
    before = x[y == 0].clone()
    adversarial_tpr(model, x, y, eps=0.2, fpr=0.1)
    assert torch.equal(x[y == 0], before)


def test_zero_epsilon_is_the_identity(model):
    x = _batch(0.5)
    assert torch.equal(
        pgd_attack(model, x, torch.ones(8, dtype=torch.long), eps=0.0), x)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/bench/test_adversarial.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.adversarial'`

- [ ] **Step 3: Write minimal implementation**

```python
# bench/adversarial.py
"""White-box adversarial baseline (spec §3A, acceptance criterion 8).

The threat model assumes the adversary holds our weights. Under that assumption
a gradient attack against any differentiable detector is not a risk — it is the
expected case. Measuring it is what turns "state-sponsored threat model" from a
sentence in a document into a number in a report.

A detector whose adversarial TPR collapses is not thereby useless. It is
demoted from decider to evidence contributor (spec §3A.4).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .metrics import tpr_at_fpr


def pgd_attack(model, x: torch.Tensor, y: torch.Tensor, eps: float = 0.03,
               alpha: float = 0.01, steps: int = 10,
               seed: int = 0) -> torch.Tensor:
    """Projected gradient descent within an L-inf ball, clamped to [0, 1].

    The random start is what distinguishes PGD from iterative FGSM, and it is
    drawn from a local `torch.Generator` so that attacking does not perturb the
    global RNG state the rest of the suite draws from.
    """
    x = x.detach()
    if eps == 0.0:
        return x.clone()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    start = (torch.rand(x.shape, generator=generator) * 2.0 - 1.0) * eps
    adv = torch.clamp(x + start.to(x.device), 0.0, 1.0).detach()

    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        (grad,) = torch.autograd.grad(loss, adv)
        with torch.no_grad():
            adv = adv + alpha * grad.sign()
            adv = torch.clamp(adv, x - eps, x + eps)
            adv = torch.clamp(adv, 0.0, 1.0)
        adv = adv.detach()
    return adv


def adversarial_tpr(model, x: torch.Tensor, y: torch.Tensor, eps: float,
                    fpr: float = 0.01, alpha: float | None = None,
                    steps: int = 10, seed: int = 0) -> float:
    """TPR@FPR after attacking only the positives (the adversary's goal).

    Negatives are left clean: a fraudster wants fakes to read as real, not the
    reverse, and attacking negatives too would move the threshold and
    understate the detector.
    """
    alpha = alpha if alpha is not None else max(eps / 4.0, 1e-4)
    pos = y == 1
    adv = x.clone()
    if eps > 0 and bool(pos.any()):
        adv[pos] = pgd_attack(model, x[pos], y[pos], eps=eps, alpha=alpha,
                              steps=steps, seed=seed)
    with torch.no_grad():
        scores = torch.softmax(model(adv), dim=1)[:, 1].cpu().numpy()
    return tpr_at_fpr(scores, y.cpu().numpy(), fpr)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/bench/test_adversarial.py -v`
Expected: PASS, 11 tests

- [ ] **Step 5: Prove the collapse test can fail**

`test_attack_collapses_tpr_to_zero` is the assertion carrying acceptance criterion 8, and the formulation it replaced could not fail. Prove this one does: temporarily make `pgd_attack` return `x.clone()` unconditionally, run `pytest tests/bench/test_adversarial.py -k collapses`, and confirm it FAILS reporting 1.0 where 0.0 was expected. Restore, confirm it passes. Record both outputs in the report.

- [ ] **Step 6: Commit**

```bash
git add bench/adversarial.py tests/bench/test_adversarial.py
git commit -m "feat: white-box PGD baseline so the threat model is measured"
```

---

### Task 16: Corpus loaders for the RD cache and capture sessions

**Files:**
- Create: `corpora/__init__.py`, `corpora/rd_cache.py`, `corpora/captures.py`
- Test: `tests/corpora/test_rd_cache.py`, `tests/corpora/test_captures.py`

**Interfaces:**
- Consumes: nothing
- Produces: `load_rd_cache(root) -> list[RDResult]`, `RDResult`, `load_capture_sessions(root) -> list[CaptureSession]`, `CaptureSession`

**The package is `corpora/`, not `datasets/`.** HuggingFace `datasets` 3.0.1 is installed in this environment, alongside `transformers` and `huggingface_hub` which this project already uses. `pyproject.toml` sets `pythonpath = ["src", "."]`, so a repo-root `datasets/` is prepended to `sys.path` and wins. Verified by construction: `import datasets` resolves to the repo package and `from datasets import load_dataset` raises `ImportError: cannot import name 'load_dataset'`. FF++, Celeb-DF and DFDC are routinely loaded through HF `datasets`, so the collision would break the corpora on this project's critical path — and it would surface late, as someone's loader failing for no visible reason, rather than at import. This project has already lost time to exactly this failure once, when a stray `tests/bench/__init__.py` shadowed the real `bench/` package.

These read the two corpora we already own (spec §1.1, §1.2): 24 cached RD responses with per-model breakdowns, and 442 capture sessions of which five are `swapped=true, approved=true` — the fraud that got through.

- [ ] **Step 1: Write the failing test**

```python
# tests/corpora/test_rd_cache.py
import json
import pytest
from corpora.rd_cache import RDResult, load_rd_cache, aggregate_is_max_like


def _write(root, name, verdict, score, models):
    d = root / name
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps(
        {"verdict": verdict, "score": score, "models": models}))


def test_loads_results_and_skips_quota_file(tmp_path):
    _write(tmp_path, "aaa", "MANIPULATED", 0.99,
           [{"name": "m1", "verdict": "MANIPULATED", "score": 0.99}])
    (tmp_path / "quota.json").write_text(json.dumps({"month": "2026-08", "count": 27}))
    out = load_rd_cache(tmp_path)
    assert len(out) == 1
    assert out[0].verdict == "MANIPULATED"


def test_exposes_per_model_scores(tmp_path):
    _write(tmp_path, "aaa", "MANIPULATED", 0.98,
           [{"name": "m1", "verdict": "MANIPULATED", "score": 0.99},
            {"name": "m2", "verdict": "AUTHENTIC", "score": 0.01}])
    r = load_rd_cache(tmp_path)[0]
    assert r.model_scores == {"m1": 0.99, "m2": 0.01}
    assert r.n_manipulated == 1 and r.n_models == 2


def test_detects_max_like_aggregation(tmp_path):
    """Reproduces the spec §1.2 finding: the aggregate tracks the maximum."""
    for i, (v, s, top) in enumerate([("MANIPULATED", 0.99, 0.99),
                                     ("MANIPULATED", 0.98, 0.99),
                                     ("MANIPULATED", 0.96, 0.99)]):
        _write(tmp_path, f"d{i}", v, s,
               [{"name": "m1", "verdict": "MANIPULATED", "score": top},
                {"name": "m2", "verdict": "AUTHENTIC", "score": 0.01}])
    results = load_rd_cache(tmp_path)
    assert aggregate_is_max_like(results, tolerance=0.2) is True


def test_a_mean_tracking_ensemble_is_not_max_like(tmp_path):
    """The negative case. Without it a constant `return True` passes, and this
    function launders one of the three headline findings in the spec rather
    than testing it: RD's aggregate tracking the MAX is why its false-positive
    rate approximates the UNION of its members' FPRs."""
    for i in range(3):
        _write(tmp_path, f"m{i}", "MANIPULATED", 0.50,
               [{"name": "m1", "verdict": "MANIPULATED", "score": 0.99},
                {"name": "m2", "verdict": "AUTHENTIC", "score": 0.01}])
    results = load_rd_cache(tmp_path)
    assert aggregate_is_max_like(results, tolerance=0.2) is False


def test_tolerance_is_load_bearing(tmp_path):
    """Same corpus, two tolerances, two answers — so `tolerance` cannot be
    silently ignored."""
    for i in range(3):
        _write(tmp_path, f"m{i}", "MANIPULATED", 0.50,
               [{"name": "m1", "verdict": "MANIPULATED", "score": 0.99},
                {"name": "m2", "verdict": "AUTHENTIC", "score": 0.01}])
    results = load_rd_cache(tmp_path)
    assert aggregate_is_max_like(results, tolerance=0.4) is False
    assert aggregate_is_max_like(results, tolerance=0.6) is True


def test_an_empty_corpus_is_not_reported_as_max_like(tmp_path):
    assert aggregate_is_max_like(load_rd_cache(tmp_path)) is False


def test_missing_models_key_does_not_crash(tmp_path):
    d = tmp_path / "bbb"
    d.mkdir()
    (d / "result.json").write_text(json.dumps({"verdict": "AUTHENTIC", "score": 0.01}))
    out = load_rd_cache(tmp_path)
    assert out[0].model_scores == {}
```

```python
# tests/corpora/test_captures.py
import json
import logging

import pytest
from corpora.captures import CaptureSession, load_capture_sessions, missed_attacks


def _session(root, name, swapped, approved, verdict="LIVE"):
    d = root / name
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps({
        "session_id": name, "swapped": swapped, "frame_count": 1,
        "scan": {"verdict": verdict},
        "decision": {"approved": approved, "reason": "approved"},
    }))


def test_loads_sessions(tmp_path):
    _session(tmp_path, "s1", False, True)
    out = load_capture_sessions(tmp_path)
    assert len(out) == 1 and out[0].session_id == "s1"


def test_label_is_derived_from_the_swapped_flag(tmp_path):
    _session(tmp_path, "s1", True, True)
    _session(tmp_path, "s2", False, True)
    by_id = {s.session_id: s for s in load_capture_sessions(tmp_path)}
    assert by_id["s1"].label == 1
    assert by_id["s2"].label == 0


def test_missed_attacks_finds_swapped_sessions_that_were_approved(tmp_path):
    """The five sessions in spec §1.1 that constitute the actual fraud."""
    _session(tmp_path, "bad", True, True)
    _session(tmp_path, "caught", True, False)
    _session(tmp_path, "genuine", False, True)
    missed = missed_attacks(load_capture_sessions(tmp_path))
    assert [s.session_id for s in missed] == ["bad"]


def test_malformed_session_is_skipped_not_fatal(tmp_path):
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "results.json").write_text("{not json")
    _session(tmp_path, "ok", False, True)
    assert len(load_capture_sessions(tmp_path)) == 1


def test_a_skipped_session_is_reported_not_swallowed(tmp_path, caplog):
    """442 sessions, of which exactly 5 are the fraud that matters. A session
    dropped in silence could be one of the 5 and nobody would know."""
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "results.json").write_text("{not json")
    _session(tmp_path, "ok", False, True)
    with caplog.at_level(logging.WARNING):
        load_capture_sessions(tmp_path)
    assert "broken" in caplog.text
    assert "1" in caplog.text


@pytest.mark.parametrize("frame_count", [0, 1, 37, 900])
def test_frame_count_is_preserved(tmp_path, frame_count):
    d = tmp_path / "s1"
    d.mkdir()
    (d / "results.json").write_text(json.dumps({
        "session_id": "s1", "swapped": False, "frame_count": frame_count,
        "scan": {"verdict": "LIVE"},
        "decision": {"approved": True, "reason": "approved"},
    }))
    assert load_capture_sessions(tmp_path)[0].frame_count == frame_count
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/corpora -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'corpora.rd_cache'`

- [ ] **Step 3: Write minimal implementation**

```python
# corpora/__init__.py
"""Corpus loaders. One module per corpus; they churn as corpora arrive."""
```

```python
# corpora/rd_cache.py
"""Loader for cached Reality Defender responses (spec §1.2).

These are the free head-to-head data: 24 results with per-model breakdowns,
already paid for. No quota is consumed by reading them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RDResult:
    cache_key: str
    verdict: str
    score: float
    model_scores: dict[str, float] = field(default_factory=dict)
    model_verdicts: dict[str, str] = field(default_factory=dict)

    @property
    def n_models(self) -> int:
        return len(self.model_scores)

    @property
    def n_manipulated(self) -> int:
        return sum(1 for v in self.model_verdicts.values() if v == "MANIPULATED")

    @property
    def max_model_score(self) -> float:
        return max(self.model_scores.values()) if self.model_scores else float("nan")


def load_rd_cache(root: str | Path) -> list[RDResult]:
    out: list[RDResult] = []
    for path in sorted(Path(root).glob("*/result.json")):
        try:
            d = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        models = d.get("models") or []
        out.append(RDResult(
            cache_key=path.parent.name,
            verdict=d.get("verdict", ""),
            score=float(d.get("score", float("nan"))),
            model_scores={m["name"]: float(m["score"]) for m in models},
            model_verdicts={m["name"]: m.get("verdict", "") for m in models},
        ))
    return out


def aggregate_is_max_like(results: list[RDResult], tolerance: float = 0.2) -> bool:
    """True if the vendor's aggregate tracks the maximum member score.

    Matters because a near-max aggregation has a false-positive rate that
    approximates the union of its members' FPRs (spec §1.2).
    """
    diffs = [abs(r.score - r.max_model_score)
             for r in results if r.model_scores and np.isfinite(r.score)]
    if not diffs:
        return False
    return float(np.mean(diffs)) <= tolerance
```

```python
# corpora/captures.py
"""Loader for the 442-session v-CIP capture corpus (spec §1.1).

Five of these sessions are swapped=true and approved=true. They are the actual
fraud, and any candidate system must be measured against them specifically —
aggregate accuracy over 442 sessions would hide all five.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaptureSession:
    session_id: str
    folder: str
    swapped: bool
    approved: bool
    scan_verdict: str | None
    frame_count: int

    @property
    def label(self) -> int:
        """1 = fake (a swap was injected), 0 = genuine."""
        return 1 if self.swapped else 0


def load_capture_sessions(root: str | Path) -> list[CaptureSession]:
    out: list[CaptureSession] = []
    skipped: list[str] = []
    for path in sorted(Path(root).glob("*/results.json")):
        try:
            d = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            # Never silent: a dropped session may be one of the five that are
            # the actual fraud, and aggregate counts would not reveal it.
            logger.warning("skipping unreadable session %s: %s",
                           path.parent.name, exc)
            skipped.append(path.parent.name)
            continue
        out.append(CaptureSession(
            session_id=d.get("session_id", path.parent.name),
            folder=str(path.parent),
            swapped=bool(d.get("swapped", False)),
            approved=bool((d.get("decision") or {}).get("approved", False)),
            scan_verdict=(d.get("scan") or {}).get("verdict"),
            frame_count=int(d.get("frame_count", 0)),
        ))
    if skipped:
        logger.warning("loaded %d capture sessions, skipped %d: %s",
                       len(out), len(skipped), ", ".join(skipped))
    return out


def missed_attacks(sessions: list[CaptureSession]) -> list[CaptureSession]:
    """Swapped sessions that were nonetheless approved — the fraud that got through."""
    return [s for s in sessions if s.swapped and s.approved]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/corpora -v`
Expected: PASS, 16 tests

- [ ] **Step 5: Prove the max-like tests can fail**

`aggregate_is_max_like` encodes a headline claim from the spec, and the version this task replaced asserted only the `True` case — a constant `return True` passed it. Prove the new tests discriminate: temporarily replace the function body with `return True`, run `pytest tests/corpora/test_rd_cache.py -k max_like or tolerance or empty_corpus`, and confirm the mean-tracking, tolerance and empty-corpus tests FAIL while `test_detects_max_like_aggregation` still passes. Then try `return False` and confirm the reverse. Restore and confirm all 16 pass. Record each output in the report.

- [ ] **Step 6: Commit**

```bash
git add corpora tests/corpora
git commit -m "feat: loaders for the RD cache and the v-CIP capture corpus"
```

---

### Task 17: Benchmark runner and head-to-head report
### Task 17: Benchmark runner and head-to-head report

**Files:**
- Create: `bench/runner.py`, `bench/report.py`
- Test: `tests/bench/test_runner.py`, `tests/bench/test_report.py`

**Interfaces:**
- Consumes: everything above
- Produces: `RunConfig`, `RunRecord`, `DetectorResult`, `run_benchmark(records, registry, config) -> RunRecord`, `render_markdown(record) -> str`

This delivers spec acceptance criteria 1, 5 and 10: a reproducible run producing per-detector TPR@FPR, AUC and ECE with all guards active, recorded latency, and a reproducibility record (seed + dataset manifest hash + model-version set).

- [ ] **Step 1: Write the failing test**

```python
# tests/bench/test_runner.py
import numpy as np
import logging

import pytest
from bench.guards import GuardViolation
from bench.runner import (
    RunConfig, RunRecord, dataset_hash, run_benchmark, worst_logo_auc,
)
from dfd.detectors.base import Registry, SyntheticDetector


def _records(n=40):
    out = []
    for i in range(n):
        fake = i % 2 == 0
        out.append({
            "sample_id": f"s{i}",
            "subject_id": f"p{i}",
            # Each image is its own source. Recorded explicitly, never aliased
            # to sample_id: the moment video records arrive, several samples
            # share one source_id and the video-level guard must still bite.
            "source_id": f"src{i}",
            # TWO generators, because leave-one-generator-out is undefined
            # with one: holding out the only generator leaves nothing to
            # train on, and logo_splits refuses such a corpus outright.
            "generator": ["deepfacelive", "faceswap"][(i // 2) % 2] if fake else None,
            "label": 1 if fake else 0,
            "compression": ["c0", "c23", "c40"][i % 3],
            "face_detector": "yunet",
            "align": "v1",
            # Non-square and varying, and at least 128px on the short side.
            # Both matter. A uniform fixture SHAPE hid a crash through every
            # test in Task 14. And a 64x64 image measures quality band
            # "reject", below SyntheticDetector's "low" floor, so the whole
            # corpus abstains: AUC, CI, TPR and ECE all come back nan and the
            # runner's entire metric path goes untested while the suite looks
            # green.
            "image": np.random.default_rng(i).integers(
                0, 255, (128 + (i % 3) * 16, 160 + (i % 5) * 16, 3),
                dtype=np.uint8),
        })
    return out


def _registry():
    reg = Registry()
    reg.register(SyntheticDetector(name="synth_a", seed=1))
    reg.register(SyntheticDetector(name="synth_b", seed=2))
    return reg


def test_run_produces_one_result_per_detector():
    rec = run_benchmark(_records(), _registry(), RunConfig(seed=7))
    assert set(rec.detector_results) == {"synth_a", "synth_b"}


def test_run_records_seed_and_dataset_hash_for_reproducibility():
    """Spec acceptance criterion 10."""
    rec = run_benchmark(_records(), _registry(), RunConfig(seed=7))
    assert rec.seed == 7
    assert len(rec.dataset_hash) == 64


def test_dataset_hash_is_stable():
    a = _records()
    assert dataset_hash(a) == dataset_hash(a)


@pytest.mark.parametrize("field_name,value", [
    ("sample_id", "changed"),
    ("subject_id", "changed"),
    ("source_id", "changed"),
    ("generator", "changed"),
    ("label", 0),
    ("compression", "c99"),
])
def test_dataset_hash_is_sensitive_to_every_identifying_field(field_name, value):
    """Mutating one field was one field's worth of evidence. The hash is what
    ties an audit record to the corpus it was computed on."""
    a, b = _records(), _records()
    b[0][field_name] = value
    assert dataset_hash(a) != dataset_hash(b)


def test_run_records_model_versions():
    rec = run_benchmark(_records(), _registry(), RunConfig(seed=7))
    assert rec.model_versions["synth_a"] == "synthetic-1"


def test_run_records_latency_per_detector():
    """Spec acceptance criterion 5: recorded, not optimised.

    `>= 0.0` would be satisfied by a stub that never measures anything and
    returns 0.0. Scoring 40 records takes real time, so require it.
    """
    rec = run_benchmark(_records(), _registry(), RunConfig(seed=7))
    assert rec.detector_results["synth_a"].p95_latency_ms > 0.0


def test_guards_run_by_default_and_fail_the_run():
    recs = _records()
    for r in recs:
        r["compression"] = "c23"          # violates compression coverage
    with pytest.raises(GuardViolation, match="compression"):
        run_benchmark(recs, _registry(), RunConfig(seed=7))


def test_guards_can_be_waived_only_explicitly():
    recs = _records()
    for r in recs:
        r["compression"] = "c23"
    rec = run_benchmark(recs, _registry(), RunConfig(seed=7, enforce_guards=False))
    assert rec.guards_enforced is False


def test_a_detector_that_abstains_on_everything_reports_rate_one():
    """The previous form asserted only `0.0 <= rate <= 1.0`, which any value
    satisfies — while its own docstring named the property it failed to test."""
    reg = Registry()
    reg.register(SyntheticDetector(name="picky", seed=1, min_quality_band="high"))
    rec = run_benchmark(_records(), reg, RunConfig(seed=7, enforce_guards=False))
    result = rec.detector_results["picky"]
    assert result.abstention_rate == 1.0
    assert result.auc != result.auc          # nan: nothing was scored


def test_a_detector_that_abstains_on_nothing_reports_rate_zero():
    rec = run_benchmark(_records(), _registry(), RunConfig(seed=7))
    assert rec.detector_results["synth_a"].abstention_rate == 0.0


def test_logo_results_exist_for_every_generator():
    """Spec 8.1. Without this the harness reports only in-dataset AUC, which
    the spec describes as measuring memorisation."""
    rec = run_benchmark(_records(), _registry(), RunConfig(seed=7))
    assert set(rec.logo_results) == {"deepfacelive", "faceswap"}
    for folds in rec.logo_results.values():
        assert set(folds) == {"synth_a", "synth_b"}


def test_a_logo_fold_scores_only_its_held_out_generator():
    """The fold must be a strict subset of the corpus, or it is not held out
    at all — a fold silently scoring everything would report in-dataset
    numbers under a LOGO heading, which is worse than reporting neither."""
    records = _records()
    rec = run_benchmark(records, _registry(), RunConfig(seed=7))
    # Without this, an empty logo_results passes by never entering the loop.
    assert len(rec.logo_results) == 2
    for folds in rec.logo_results.values():
        n = folds["synth_a"].n_samples
        assert 0 < n < len(records)


def test_logo_and_in_dataset_numbers_are_reported_separately():
    """They must not be the same object or the same number by construction."""
    rec = run_benchmark(_records(), _registry(), RunConfig(seed=7))
    assert rec.detector_results["synth_a"].n_samples == 40
    # `all(...)` over an empty dict is True, so the count is asserted first.
    assert len(rec.logo_results) == 2
    assert all(f["synth_a"].n_samples < 40 for f in rec.logo_results.values())


def test_worst_logo_auc_takes_the_minimum_not_the_mean():
    """Spec 8.2 guard 3 reports the WORST compression cell for the same
    reason: an average over generators hides the one an attacker will use."""
    rec = run_benchmark(_records(), _registry(), RunConfig(seed=7))
    per_fold = [f["synth_a"].auc for f in rec.logo_results.values()]
    assert worst_logo_auc(rec, "synth_a") == min(per_fold)


def test_a_single_generator_corpus_reports_no_logo_rather_than_failing(caplog):
    """LOGO is undefined with one generator. The in-dataset numbers are still
    valid, so the run degrades rather than raising."""
    records = _records()
    for r in records:
        if r["label"] == 1:
            r["generator"] = "deepfacelive"
    with caplog.at_level(logging.WARNING):
        rec = run_benchmark(records, _registry(),
                            RunConfig(seed=7, enforce_guards=False))
    assert rec.logo_results == {}
    assert "LOGO unavailable" in caplog.text
    assert rec.detector_results["synth_a"].n_samples == 40


def test_run_is_reproducible_given_a_seed():
    a = run_benchmark(_records(), _registry(), RunConfig(seed=7))
    b = run_benchmark(_records(), _registry(), RunConfig(seed=7))
    assert (a.detector_results["synth_a"].auc
            == b.detector_results["synth_a"].auc)
    assert a.dataset_hash == b.dataset_hash
    # Folds are drawn from a seeded permutation; same seed, same folds.
    assert ({g: f["synth_a"].auc for g, f in a.logo_results.items()}
            == {g: f["synth_a"].auc for g, f in b.logo_results.items()})
```

```python
# tests/bench/test_report.py
from dataclasses import replace

from bench.report import render_markdown
from bench.runner import DetectorResult, RunRecord


def _record():
    return RunRecord(
        seed=7, dataset_hash="a" * 64, guards_enforced=True,
        model_versions={"synth_a": "synthetic-1"},
        identity_report=None,
        detector_results={
            "synth_a": DetectorResult(
                detector="synth_a", auc=0.83, auc_ci=(0.71, 0.92),
                tpr_at_1pct=0.42, tpr_at_0p1pct=0.19, ece=0.06,
                adversarial_tpr_at_1pct=0.03,
                abstention_rate=0.10, p95_latency_ms=42.0,
                n_samples=40),
        },
    )


def test_report_contains_the_reproducibility_record():
    md = render_markdown(_record())
    assert "seed" in md.lower()
    assert "a" * 64 in md


def test_report_shows_tpr_at_fpr_not_accuracy():
    md = render_markdown(_record())
    assert "TPR@FPR=1%" in md
    assert "accuracy" not in md.lower()


def test_report_shows_adversarial_column():
    md = render_markdown(_record())
    assert "adversarial" in md.lower()


def test_report_flags_a_detector_defeated_by_adversarial_attack():
    md = render_markdown(_record())
    assert "evidence-only" in md.lower()


def test_report_shows_confidence_intervals():
    md = render_markdown(_record())
    assert "0.71" in md and "0.92" in md


def _logo_record():
    base = _record()
    def _dr(auc):
        return DetectorResult(
            detector="synth_a", auc=auc, auc_ci=(auc - 0.1, auc + 0.1),
            tpr_at_1pct=0.2, tpr_at_0p1pct=0.1, ece=0.05,
            adversarial_tpr_at_1pct=0.03, abstention_rate=0.0,
            p95_latency_ms=1.0, n_samples=12)
    return replace(base, logo_results={
        "deepfacelive": {"synth_a": _dr(0.77)},
        "faceswap": {"synth_a": _dr(0.51)},
    })


def test_report_leads_with_the_worst_held_out_generator():
    """Spec §8.1. The mean of 0.77 and 0.51 is 0.64; reporting that would
    hide the generator an attacker would actually choose."""
    md = render_markdown(_logo_record())
    assert "0.510" in md
    assert "0.640" not in md
    assert md.index("Leave-one-generator-out") < md.index("In-dataset")


def test_report_labels_whole_corpus_numbers_as_memorisation():
    md = render_markdown(_logo_record())
    assert "memorisation" in md.lower()


def test_report_says_so_when_logo_was_not_computed():
    """A missing LOGO number must be stated, not left as a silent absence
    that reads as though the in-dataset table were the result."""
    md = render_markdown(_record())
    assert "not computed" in md.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/bench/test_runner.py tests/bench/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.runner'`

- [ ] **Step 3: Write minimal implementation**

```python
# bench/runner.py
"""Benchmark orchestration (spec §8, acceptance criteria 1, 5, 10).

Guards are enforced by default and can only be waived explicitly, because a
benchmark that is easy to run dirty will be run dirty — and every hygiene
failure flatters the result.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from dfd.quality import measure_quality
from dfd.types import Observation

from .guards import (
    IdentityReport, check_compression_coverage, check_threshold_provenance,
    check_uniform_preprocessing, check_video_level,
)
from .metrics import auc, bootstrap_ci_by_group, ece, tpr_at_fpr
from .protocol import logo_splits

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    seed: int = 0
    enforce_guards: bool = True
    fpr_targets: tuple[float, ...] = (0.01, 0.001)
    bootstrap_n: int = 200
    threshold_source: str = "validation"


@dataclass(frozen=True)
class DetectorResult:
    detector: str
    auc: float
    auc_ci: tuple[float, float]
    tpr_at_1pct: float
    tpr_at_0p1pct: float
    ece: float
    adversarial_tpr_at_1pct: float | None
    abstention_rate: float
    p95_latency_ms: float
    n_samples: int


@dataclass(frozen=True)
class RunRecord:
    seed: int
    dataset_hash: str
    guards_enforced: bool
    model_versions: dict[str, str]
    identity_report: IdentityReport | None
    #: Whole-corpus metrics. THIS IS IN-DATASET PERFORMANCE, which measures
    #: memorisation, not field performance. Never report it as the headline.
    detector_results: dict[str, DetectorResult] = field(default_factory=dict)
    #: held-out generator -> detector -> metrics. Spec 8.1: the only number
    #: that predicts field performance. Empty when the corpus cannot be split.
    logo_results: dict[str, dict[str, DetectorResult]] = field(
        default_factory=dict)


def dataset_hash(records: list[dict]) -> str:
    """Content hash over identifying fields — stable, order-independent."""
    keys = sorted(
        json.dumps({k: r.get(k) for k in
                    ("sample_id", "subject_id", "source_id", "generator",
                     "label", "compression")},
                   sort_keys=True)
        for r in records)
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def _observation(record: dict) -> Observation:
    img = record["image"]
    h, w = img.shape[:2]
    lm = np.array([[w * 0.35, h * 0.4], [w * 0.65, h * 0.4]])
    q = measure_quality(img, (0, 0, w, h), lm)
    return Observation(t=0.0, payload=img, roi=(0, 0, w, h), quality=q,
                       source_id=record["source_id"])


def run_benchmark(records: list[dict], registry, config: RunConfig) -> RunRecord:
    if config.enforce_guards:
        # `groups` MUST identify the SOURCE VIDEO, never the sample id.
        # Passing sample_ids for both arguments makes this guard vacuous: its
        # only failure condition is groups[i] != sample_ids[i], so identical
        # lists can never raise. Records must carry a source-video field; for
        # image records each image is its own source, which must be recorded
        # explicitly rather than aliased to sample_id.
        check_video_level([r["sample_id"] for r in records],
                          [r["source_id"] for r in records])
        check_compression_coverage(records)
        check_uniform_preprocessing(records)
        check_threshold_provenance(config.threshold_source)

    labels = np.array([r["label"] for r in records], dtype=int)
    # The SOURCE video, not the sample id. bootstrap_ci_by_group resamples
    # over these, and resampling over frames rather than videos fabricates
    # precision: measured at 11.9x too narrow (group CI 0.751 vs row 0.063).
    # check_video_level enforces a 1:1 mapping while guards are on, but
    # enforce_guards=False is a supported path and that is exactly where an
    # honest interval matters most.
    groups = np.array([r["source_id"] for r in records])
    observations = [_observation(r) for r in records]

    results: dict[str, DetectorResult] = {}
    versions: dict[str, str] = {}
    scores_by_detector: dict[str, np.ndarray] = {}

    for name in registry.names():
        det = registry.get(name)
        versions[name] = det.version

        scores: list[float] = []
        abstentions = 0
        latencies: list[float] = []

        for obs in observations:
            t0 = time.perf_counter()
            raw = det.score([obs])
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if raw.abstained or raw.score is None:
                abstentions += 1
                scores.append(np.nan)
            else:
                scores.append(float(raw.score))

        s = np.array(scores, dtype=float)
        scores_by_detector[name] = s
        results[name] = _detector_result(
            name, s, labels, groups, latencies, abstentions, config)

    logo_results = _logo_results(records, registry, scores_by_detector,
                                 labels, groups, config)

    return RunRecord(seed=config.seed, dataset_hash=dataset_hash(records),
                     guards_enforced=config.enforce_guards,
                     model_versions=versions, identity_report=None,
                     detector_results=results, logo_results=logo_results)


def _detector_result(name, s, labels, groups, latencies, abstentions,
                     config) -> DetectorResult:
    """Metrics for one detector over one set of rows.

    Split out so a LOGO fold can reuse it verbatim: the fold differs only in
    which rows it passes, never in how the numbers are computed.
    """
    n = len(s)
    valid = np.isfinite(s)
    base = dict(
        detector=name,
        adversarial_tpr_at_1pct=None,
        abstention_rate=abstentions / max(1, n),
        p95_latency_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
        n_samples=n,
    )
    if valid.sum() == 0 or len(np.unique(labels[valid])) < 2:
        nan = float("nan")
        return DetectorResult(auc=nan, auc_ci=(nan, nan), tpr_at_1pct=nan,
                              tpr_at_0p1pct=nan, ece=nan, **base)
    return DetectorResult(
        auc=auc(s[valid], labels[valid]),
        auc_ci=bootstrap_ci_by_group(s[valid], labels[valid], groups[valid],
                                     auc, n=config.bootstrap_n,
                                     seed=config.seed),
        tpr_at_1pct=tpr_at_fpr(s[valid], labels[valid], 0.01),
        tpr_at_0p1pct=tpr_at_fpr(s[valid], labels[valid], 0.001),
        ece=ece(s[valid], labels[valid]),
        **base,
    )


def _logo_results(records, registry, scores_by_detector, labels, groups,
                  config) -> dict[str, dict[str, DetectorResult]]:
    """Per-held-out-generator metrics — spec 8.1, the number that predicts field
    performance.

    Detection is not re-run per fold: a detector's score for a record does not
    depend on which fold the record lands in, so folds slice the scores already
    computed.

    HONEST SCOPE. A full LOGO protocol trains on the fold's train side and
    tests on the held-out one. P0 detectors are not trained here, so what this
    computes is evaluation on the held-out generator's test rows. That is the
    number you report, and it becomes the full protocol once training or
    calibration fitting exists — at which point the fold's train side is also
    where the operating threshold must be frozen (spec 8.2 guard 5).
    """
    try:
        splits = logo_splits(records, seed=config.seed)
    except ValueError as exc:
        # A corpus with one generator, one subject, or no measurable fold.
        # Recorded rather than raised: the in-dataset numbers are still valid.
        logger.warning("LOGO unavailable for this corpus: %s", exc)
        return {}

    position = {r["sample_id"]: i for i, r in enumerate(records)}
    out: dict[str, dict[str, DetectorResult]] = {}
    for split in splits:
        rows = np.array([position[sid] for sid in split.test_ids()], dtype=int)
        out[split.held_out_generator] = {
            name: _detector_result(name, scores_by_detector[name][rows],
                                   labels[rows], groups[rows], [], 0, config)
            for name in registry.names()
        }
    return out


def worst_logo_auc(record: "RunRecord", detector: str) -> float:
    """The weakest held-out generator for one detector.

    Reported in preference to the mean, for the same reason spec 8.2 guard 3
    reports the worst compression cell: an average over generators hides the
    one an attacker will actually use.
    """
    folds = [f[detector].auc for f in record.logo_results.values()
             if detector in f]
    finite = [a for a in folds if a == a]
    return min(finite) if finite else float("nan")
```

```python
# bench/report.py
"""Markdown rendering of a benchmark run.

Deliberately prints no accuracy figure. At fraud base rates accuracy is a
number that always looks good and never means anything.
"""
from __future__ import annotations

from .runner import RunRecord, worst_logo_auc

ADVERSARIAL_FLOOR = 0.10


def _f(x) -> str:
    return "n/a" if x is None or x != x else f"{x:.3f}"


def render_markdown(record: RunRecord) -> str:
    lines: list[str] = []
    lines.append("# Benchmark run\n")
    lines.append("## Reproducibility record\n")
    lines.append(f"- seed: `{record.seed}`")
    lines.append(f"- dataset hash: `{record.dataset_hash}`")
    lines.append(f"- guards enforced: `{record.guards_enforced}`")
    versions = ", ".join(f"{k}={v}" for k, v in sorted(record.model_versions.items()))
    lines.append(f"- model versions: `{versions}`")
    if record.identity_report is not None:
        r = record.identity_report
        lines.append(f"- identity disjointness: max cosine `{r.max_similarity:.4f}` "
                     f"at threshold `{r.threshold}`, {r.violations} violations")
    lines.append("")

    lines.append("## Leave-one-generator-out (spec §8.1)\n")
    if not record.logo_results:
        lines.append(
            "**Not computed for this corpus.** Without a held-out-generator "
            "number there is nothing here that predicts field performance; "
            "the table below measures memorisation only.\n")
    else:
        lines.append("Worst held-out generator per detector — the headline "
                     "number. Reported as the worst rather than the mean for "
                     "the same reason spec §8.2 guard 3 reports the worst "
                     "compression cell: an average hides the generator an "
                     "attacker will actually use.\n")
        detectors = sorted(record.detector_results)
        generators = sorted(record.logo_results)
        lines.append("| detector | worst AUC | "
                     + " | ".join(f"held out {g}" for g in generators) + " |")
        lines.append("|---|---|" + "---|" * len(generators))
        for name in detectors:
            cells = [_f(record.logo_results[g][name].auc) for g in generators]
            lines.append(f"| {name} | **{_f(worst_logo_auc(record, name))}** | "
                         + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## In-dataset results — memorisation, not field performance\n")
    lines.append("These are computed over the whole corpus, with every "
                 "generator seen. Spec §8.1: in-dataset AUC measures "
                 "memorisation. Read the LOGO table above instead.\n")
    lines.append("| detector | AUC | 95% CI | TPR@FPR=1% | TPR@FPR=0.1% | "
                 "adversarial TPR@FPR=1% | ECE | abstained | p95 ms | n |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for name in sorted(record.detector_results):
        d = record.detector_results[name]
        lo, hi = d.auc_ci
        lines.append(
            f"| {d.detector} | {_f(d.auc)} | {_f(lo)}–{_f(hi)} | "
            f"{_f(d.tpr_at_1pct)} | {_f(d.tpr_at_0p1pct)} | "
            f"{_f(d.adversarial_tpr_at_1pct)} | {_f(d.ece)} | "
            f"{d.abstention_rate:.1%} | {d.p95_latency_ms:.1f} | {d.n_samples} |")
    lines.append("")

    demoted = [d for d in record.detector_results.values()
               if d.adversarial_tpr_at_1pct is not None
               and d.adversarial_tpr_at_1pct < ADVERSARIAL_FLOOR]
    if demoted:
        lines.append("## Adversarial demotion\n")
        lines.append(
            f"Adversarial TPR below {ADVERSARIAL_FLOOR:.0%} under white-box PGD. "
            "Per spec §3A.4 these are **evidence-only** and must not decide:\n")
        for d in demoted:
            lines.append(f"- `{d.detector}` — adversarial TPR@FPR=1% "
                         f"{_f(d.adversarial_tpr_at_1pct)}")
        lines.append("")

    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/bench/test_runner.py tests/bench/test_report.py -v`
Expected: PASS, 29 tests (21 runner, 8 report)

- [ ] **Step 5: Run the full suite and commit**

Run: `python -m pytest -q`
Expected: PASS, ~100 tests, no network access, no model weights required

```bash
git add bench/runner.py bench/report.py tests/bench/test_runner.py tests/bench/test_report.py
git commit -m "feat: reproducible benchmark runner and head-to-head report"
```

---

## Self-Review

**Spec coverage.** §5.1 Sample abstraction → Task 1. §5.2 LLR currency → Tasks 9, 10. §5.3 quality gate → Tasks 3, 6. §6 portfolio (slots A, C, E) → Tasks 7, 8. §7.1 calibration → Task 9. §7 four verdicts + disagreement → Tasks 1, 10. §8.1 LOGO → Task 13. §8.2 six guards (incl. demographic parity) → Task 12. §8.3 metrics and robustness → Tasks 11, 14. §8.4 head-to-head → Tasks 16, 17. §9.5 ESS discount → Task 10. §11 manifest → Tasks 2, 4, 7, 8. §3A adversarial → Task 15. Principle 9 randomisation → Task 6 `select_subset`. Principle 10 reconstructability → Task 17.

**Known gaps, deliberately deferred and recorded here so they are not forgotten:**

1. **The LOGO number is now computed, but this is not a trained LOGO protocol.** `run_benchmark` evaluates each detector on each held-out generator's test rows and reports the worst as the headline. A full protocol also *trains* on the fold's train side; P0 detectors are not trained here, so the train side is currently unused. It becomes load-bearing the moment calibration or training lands — and that is also where the operating threshold must be frozen, which would make spec §8.2 guard 5 real rather than the string check on `config.threshold_source` it is today.

2. **Guard 1 (identity leakage) is implemented in Task 12 but not yet wired into `run_benchmark`.** `RunRecord.identity_report` is present and rendered but always `None`, because computing it needs a face-embedding model that is not part of P0's licence-clean set (spec §11 flags InsightFace). **Acceptance criterion 2 is therefore not met by this plan alone** — it needs a follow-up task once an embedding model is chosen. This is the single most important gap; do not close P0 without it.
3. `adversarial_tpr_at_1pct` is computed by Task 15 but wired as `None` in the runner, because it needs a differentiable model and the P0 detectors abstain without weights. Wire it when real weights land.
4. Calibration is fitted per detector but the runner scores raw detector output rather than fused LLRs. End-to-end fusion scoring belongs in P1.
5. **Resolved, not deferred.** `check_video_level` reads a real `source_id` field; records must populate it. For image corpora each image is its own source, recorded explicitly. Aliasing `source_id` to `sample_id` at the call site makes the guard vacuous — its only failure condition is `groups[i] != sample_ids[i]` — so the alias is forbidden rather than tolerated.

**Placeholder scan.** No TBDs. Every step carries runnable code. Threshold constants in `quality.py` are marked as starting values with a stated plan (Task 18 follow-up) rather than left as magic numbers.

**Type consistency.** `RawScore` (Tasks 1, 6, 7, 8) → `Calibrator.to_evidence` (Task 9) → `Evidence` (Tasks 1, 10) → `FusedResult` (Task 10) verified consistent. `meets_floor(band, floor)` signature identical in Tasks 3, 6, 7, 8. `abstain(detector, version, reason)` identical in Tasks 6, 7, 8. `Registry` defined in `base.py` and re-exported from `registry.py`, imported both ways in tests — consistent.

---

### Task 18: Wire the robustness sweep into the runner
### Task 18: Wire the robustness sweep into the runner

**Files:**
- Modify: `bench/runner.py`, `bench/report.py`
- Test: `tests/bench/test_runner_robustness.py`

**Interfaces:**
- Consumes: `robustness_sweep`, `PERTURBATIONS` (Task 14); `RunConfig`, `DetectorResult`, `run_benchmark` (Task 17)
- Produces: `RunConfig.robustness` flag, `DetectorResult.tpr_by_perturbation: dict[str, float]`

**Why this task exists:** Tasks 14 and 17 were both correct in isolation and did not connect — `robustness_sweep` was built, tested, and never called. Spec acceptance criterion 9 requires screenshot-of-screen and print-recapture to be **measured**, and without this task P0 would close believing it measured them. The two cheapest laundering steps available to any adversary would have gone untested.

- [ ] **Step 1: Write the failing test**

```python
# tests/bench/test_runner_robustness.py
import numpy as np
import pytest
from bench.robustness import PERTURBATIONS
from bench.runner import RunConfig, run_benchmark
from dfd.detectors.base import Registry, SyntheticDetector


def _records(n=30):
    out = []
    for i in range(n):
        fake = i % 2 == 0
        out.append({
            "sample_id": f"s{i}", "subject_id": f"p{i}",
            "generator": "deepfacelive" if fake else None,
            "label": 1 if fake else 0,
            "compression": ["c0", "c23", "c40"][i % 3],
            "face_detector": "yunet", "align": "v1",
            "image": np.random.default_rng(i).integers(
                0, 255, (64, 64, 3), dtype=np.uint8),
        })
    return out


def _registry():
    reg = Registry()
    reg.register(SyntheticDetector(name="synth_a", seed=1))
    return reg


def test_robustness_is_off_by_default():
    rec = run_benchmark(_records(), _registry(), RunConfig(seed=1))
    assert rec.detector_results["synth_a"].tpr_by_perturbation == {}


def test_robustness_reports_one_entry_per_perturbation_plus_clean():
    rec = run_benchmark(_records(), _registry(),
                        RunConfig(seed=1, robustness=True))
    got = rec.detector_results["synth_a"].tpr_by_perturbation
    assert "clean" in got
    assert set(PERTURBATIONS).issubset(set(got))


def test_physical_recapture_paths_are_measured():
    """Spec acceptance criterion 9 — the reason this task exists."""
    rec = run_benchmark(_records(), _registry(),
                        RunConfig(seed=1, robustness=True))
    got = rec.detector_results["synth_a"].tpr_by_perturbation
    assert "screenshot_recapture" in got
    assert "print_recapture" in got


def test_robustness_run_is_reproducible():
    a = run_benchmark(_records(), _registry(), RunConfig(seed=1, robustness=True))
    b = run_benchmark(_records(), _registry(), RunConfig(seed=1, robustness=True))
    assert (a.detector_results["synth_a"].tpr_by_perturbation
            == b.detector_results["synth_a"].tpr_by_perturbation)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/bench/test_runner_robustness.py -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'robustness'`

- [ ] **Step 3: Write minimal implementation**

In `bench/runner.py`, add `robustness: bool = False` to `RunConfig`, add
`tpr_by_perturbation: dict = field(default_factory=dict)` to `DetectorResult`,
import `from .robustness import robustness_sweep`, and after the clean scoring
loop for each detector add:

```python
        tpr_by_perturbation: dict[str, float] = {}
        if config.robustness:
            variants: dict[str, list[float]] = {}
            for rec_in, obs in zip(records, observations):
                for pname, pimg in robustness_sweep(rec_in["image"]).items():
                    pobs = Observation(t=obs.t, payload=pimg, roi=obs.roi,
                                       quality=obs.quality,
                                       source_id=obs.source_id)
                    praw = det.score([pobs])
                    variants.setdefault(pname, []).append(
                        float(praw.score) if not praw.abstained
                        and praw.score is not None else np.nan)
            for pname, pscores in variants.items():
                ps = np.array(pscores, dtype=float)
                pv = np.isfinite(ps)
                tpr_by_perturbation[pname] = (
                    tpr_at_fpr(ps[pv], labels[pv], 0.01)
                    if pv.sum() and len(np.unique(labels[pv])) > 1
                    else float("nan"))
```

Pass `tpr_by_perturbation=tpr_by_perturbation` into both `DetectorResult(...)`
constructions in the function (the degenerate-case one and the normal one).

In `bench/report.py`, after the per-detector table, add:

```python
    any_rob = any(d.tpr_by_perturbation for d in record.detector_results.values())
    if any_rob:
        names = sorted({p for d in record.detector_results.values()
                        for p in d.tpr_by_perturbation})
        lines.append("## Robustness — TPR@FPR=1% under perturbation\n")
        lines.append("| detector | " + " | ".join(names) + " |")
        lines.append("|---" * (len(names) + 1) + "|")
        for name in sorted(record.detector_results):
            d = record.detector_results[name]
            row = " | ".join(_f(d.tpr_by_perturbation.get(p)) for p in names)
            lines.append(f"| {d.detector} | {row} |")
        lines.append("")
        lines.append("`screenshot_recapture` and `print_recapture` are the two "
                     "cheapest laundering steps available to an adversary; a "
                     "detector that collapses under them is not deployable "
                     "against the threat model in spec §3A.\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/bench/test_runner_robustness.py tests/bench/test_runner.py tests/bench/test_report.py -v`
Expected: PASS — 4 new tests, and Tasks 17's 14 tests still green

- [ ] **Step 5: Commit**

```bash
git add bench/runner.py bench/report.py tests/bench/test_runner_robustness.py
git commit -m "feat: measure the robustness surface in the benchmark runner"
```

---

### Task 19: Asset enumeration — make the release gate non-vacuous

**Files:**
- Create: `src/dfd/asset_scan.py`
- Test: `tests/test_asset_scan.py`

**Interfaces:**
- Consumes: `load_manifest`, `assert_release_clean`, `NonCommercialAsset` (Task 2)
- Produces: `discover_assets(root) -> list[str]`, `assert_all_assets_registered(root, manifest_path)`

**Why this task exists:** Task 2's `assert_release_clean(manifest, asset_ids)` can only judge assets it is *handed*. Passing it an empty list returns cleanly — a vacuous pass. Nothing in the repo enumerates what assets are actually in use, so the gate currently guarantees nothing about a real release: forget to list a weight file and it ships unchecked, which is precisely the false confidence Task 2 exists to prevent. Spec §12.1 criterion 6 ("asset manifest covering every dataset and weight file in use") is unverifiable without this.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_asset_scan.py
import pytest
from dfd.asset_scan import (
    ASSET_SUFFIXES, assert_all_assets_registered, discover_assets,
)
from dfd.manifest import NonCommercialAsset

MANIFEST = """
assets:
  good_weights:
    source: "s"
    license: "MIT"
    commercial_use: true
    evidence_url: "u"
    date_checked: "2026-09-20"
    checked_by: "k"
"""


def _tree(tmp_path, *names):
    (tmp_path / "assets" / "models").mkdir(parents=True, exist_ok=True)
    for n in names:
        (tmp_path / "assets" / "models" / n).write_bytes(b"x")
    return tmp_path


def test_discovers_weight_files_by_suffix(tmp_path):
    _tree(tmp_path, "good_weights.onnx", "notes.txt")
    found = discover_assets(tmp_path)
    assert "good_weights" in found
    assert "notes" not in found


def test_every_declared_suffix_is_discovered(tmp_path):
    names = [f"a{i}{s}" for i, s in enumerate(sorted(ASSET_SUFFIXES))]
    _tree(tmp_path, *names)
    found = set(discover_assets(tmp_path))
    assert len(found) == len(ASSET_SUFFIXES)


def test_passes_when_every_discovered_asset_is_registered(tmp_path):
    _tree(tmp_path, "good_weights.onnx")
    mp = tmp_path / "manifest.yaml"
    mp.write_text(MANIFEST)
    assert assert_all_assets_registered(tmp_path, mp) is None


def test_raises_on_an_asset_present_on_disk_but_absent_from_the_manifest(tmp_path):
    """The whole point: a weight file nobody registered must fail the build."""
    _tree(tmp_path, "good_weights.onnx", "sneaky_weights.pt")
    mp = tmp_path / "manifest.yaml"
    mp.write_text(MANIFEST)
    with pytest.raises(NonCommercialAsset) as exc:
        assert_all_assets_registered(tmp_path, mp)
    assert "sneaky_weights" in str(exc.value)


def test_empty_tree_is_not_treated_as_success_by_accident(tmp_path):
    """An empty scan must be visibly empty, not a silent pass."""
    (tmp_path / "assets").mkdir()
    assert discover_assets(tmp_path) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_asset_scan.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dfd.asset_scan'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/dfd/asset_scan.py
"""Enumerate assets on disk so the release gate cannot pass vacuously.

`assert_release_clean` judges only the ids it is handed, so handing it nothing
returns cleanly. That is a gate guaranteeing nothing. This module supplies the
list from the filesystem instead of from a human's memory.
"""
from __future__ import annotations

from pathlib import Path

from .manifest import assert_release_clean, load_manifest

# Extensions that carry model weights or dataset payloads.
ASSET_SUFFIXES = (".onnx", ".pt", ".pth", ".safetensors", ".tflite", ".bin", ".npz")


def discover_assets(root: str | Path) -> list[str]:
    """Asset ids (filename stems) for every weight-like file under `root`."""
    found: set[str] = set()
    for path in Path(root).rglob("*"):
        if path.is_file() and path.suffix.lower() in ASSET_SUFFIXES:
            found.add(path.stem)
    return sorted(found)


def assert_all_assets_registered(root: str | Path,
                                 manifest_path: str | Path) -> None:
    """Raise unless every asset on disk is registered and commercially clear."""
    assert_release_clean(load_manifest(manifest_path), discover_assets(root))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_asset_scan.py tests/test_manifest.py -v`
Expected: PASS — 5 new tests, Task 2's 4 still green

- [ ] **Step 5: Commit**

```bash
git add src/dfd/asset_scan.py tests/test_asset_scan.py
git commit -m "feat: enumerate assets from disk so the release gate cannot pass vacuously"
```

---

### Task 20: Immutable audit record

**Files:**
- Create: `src/dfd/errors.py`, `src/dfd/audit.py`
- Test: `tests/test_audit.py`

**Interfaces:**
- Consumes: `Evidence`, `Verdict` (Task 1)
- Produces: `DfdError` hierarchy; `AuditRecord`, `build_audit_record(...) -> AuditRecord`, `AuditRecord.to_json() -> str`, `record_digest(record) -> str`

**Why this task exists:** spec §7.2 requires every decision to emit an immutable record — input hash, model versions, per-detector LLRs, quality metrics, policy version, decision. It is the artifact that makes a rejection defensible to a regulator and doubles as next-cycle training data. **The original 19-task plan had no task for it; this is a dropped spec requirement, not an enhancement.** Without it the system can decide but cannot account for a decision, which is not shippable in BFSI.

**The record is called immutable and tamper-evident, so it has to be both.** Three ways the obvious implementation is neither, all measured:

1. **`@dataclass(frozen=True)` is shallow.** A `dict` field stays mutable: `record.model_versions["npr"] = "tampered"` raises nothing and *changes the digest*, so a record can be altered after the fact and re-digested to match. Containers are frozen here — `MappingProxyType` for mappings, tuples for sequences — and each is asserted to refuse mutation.
2. **Excluding `created_at` from the digest makes backdating invisible.** Verified: moving a timestamp from 2026 to 1999 leaves the digest identical. The only reason to exclude it is so two separately-built records compare equal in a test — which weakens the guarantee to fit the test. Instead `created_at` is an **injectable parameter** defaulting to now, so identical records really are identical and the digest covers the timestamp. For an audit record defended to a regulator, the timestamp is among the most attack-relevant fields there is.
3. **`json.dumps(..., default=str)` silently absorbs anything.** It never raises, so a caller passing a numpy array, bytes, or any other non-JSON value gets it stringified into the record instead of rejected. For a record whose digest is the tamper-evidence, silently absorbing an unexpected type is the wrong failure direction. `default=` is omitted, and a non-JSON value is asserted to raise.

Note also what a PII test must actually do. Asserting `r"\x89PNG" not in r.to_json()` proves nothing — that is seven literal characters no implementation ever inserts, and the assertion passes on a record carrying a whole image. The real discipline is that the record references its input by hash and refuses values it cannot serialise.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_audit.py
import dataclasses
import json

import numpy as np
import pytest

from dfd.audit import AuditRecord, build_audit_record, record_digest
from dfd.errors import DfdError, InvalidInput
from dfd.types import Evidence, Verdict

FIXED_TIME = "2026-09-20T10:00:00+00:00"


def _ev(name, llr, abstained=False, reason="ok"):
    return Evidence(detector=name, detector_version="1.0", llr=llr,
                    raw_score=0.5, uncertainty=0.0,
                    abstained=abstained, reason=reason)


def _record(**kw):
    base = dict(
        sample_id="s1", input_sha256="a" * 64, verdict=Verdict.FAKE,
        llr_total=3.2, posterior=0.96,
        evidence=[_ev("npr", 2.0), _ev("sbi", 1.2)],
        quality_band="high", ood_score=0.1,
        policy_version="policy-1", threshold=1.0,
        model_versions={"npr": "0.1.0", "sbi": "0.1.0"},
        created_at=FIXED_TIME,
    )
    base.update(kw)
    return build_audit_record(**base)


def test_record_carries_every_field_a_regulator_would_ask_for():
    r = _record()
    for name in ("sample_id", "input_sha256", "verdict", "llr_total",
                 "posterior", "policy_version", "threshold", "model_versions",
                 "quality_band", "ood_score", "created_at", "schema_version"):
        assert getattr(r, name) is not None, name


def test_rebinding_a_field_raises():
    with pytest.raises(dataclasses.FrozenInstanceError):
        _record().verdict = Verdict.REAL


def test_model_versions_cannot_be_mutated_in_place():
    """`frozen=True` is shallow: a plain dict field stays writable and
    mutating it changes the digest, so the record is not immutable at all."""
    r = _record()
    with pytest.raises(TypeError):
        r.model_versions["npr"] = "tampered"


def test_evidence_rows_cannot_be_appended_to():
    r = _record()
    with pytest.raises(AttributeError):
        r.evidence.append({"detector": "ghost"})


def test_per_detector_llrs_are_preserved_including_abstentions():
    """An abstention is evidence about the system, not an absence of evidence."""
    r = _record(evidence=[_ev("npr", 2.0),
                          _ev("sbi", 0.0, abstained=True,
                              reason="weights_absent")])
    got = {e["detector"]: e for e in r.evidence}
    assert got["sbi"]["abstained"] is True
    assert got["sbi"]["reason"] == "weights_absent"
    assert got["npr"]["llr"] == 2.0


def test_to_json_round_trips():
    d = json.loads(_record().to_json())
    assert d["sample_id"] == "s1"
    assert d["verdict"] == "fake"
    assert len(d["evidence"]) == 2


def test_digest_is_stable_for_identical_records():
    assert record_digest(_record()) == record_digest(_record())


@pytest.mark.parametrize("field,value", [
    ("verdict", Verdict.REAL),
    ("llr_total", 3.3),
    ("posterior", 0.95),
    ("sample_id", "s2"),
    ("input_sha256", "b" * 64),
    ("quality_band", "low"),
    ("ood_score", 0.2),
    ("policy_version", "policy-2"),
    ("threshold", 1.5),
    ("model_versions", {"npr": "0.2.0", "sbi": "0.1.0"}),
    ("created_at", "1999-01-01T00:00:00+00:00"),
])
def test_digest_changes_when_any_field_changes(field, value):
    """Tamper-evidence, field by field. `created_at` is in this list
    deliberately: excluding it makes backdating a decision invisible."""
    assert record_digest(_record()) != record_digest(_record(**{field: value}))


def test_digest_changes_when_evidence_changes():
    assert record_digest(_record()) != record_digest(
        _record(evidence=[_ev("npr", 9.9), _ev("sbi", 1.2)]))


def test_a_value_that_cannot_be_serialised_is_refused_not_stringified():
    """`json.dumps(default=str)` never raises, so an image passed by mistake
    would be absorbed into the record instead of rejected."""
    with pytest.raises((InvalidInput, TypeError)):
        _record(model_versions={"npr": np.zeros((4, 4), dtype=np.uint8)}).to_json()


def test_the_record_references_its_input_by_hash_only():
    d = json.loads(_record().to_json())
    assert d["input_sha256"] == "a" * 64
    assert not any(k.startswith("image") or k.endswith("bytes") for k in d)


def test_rejects_a_malformed_input_hash():
    with pytest.raises(InvalidInput, match="64 lowercase hex"):
        _record(input_sha256="not-a-hash")


def test_rejects_an_uppercase_hash():
    """Case matters: the same digest in two cases would give two records."""
    with pytest.raises(InvalidInput, match="64 lowercase hex"):
        _record(input_sha256="A" * 64)


def test_rejects_an_empty_sample_id():
    with pytest.raises(InvalidInput, match="sample_id"):
        _record(sample_id="")


def test_invalid_input_is_a_dfd_error():
    """One catchable root for every error this package raises."""
    assert issubclass(InvalidInput, DfdError)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_audit.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dfd.audit'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/dfd/errors.py
"""Exception hierarchy. One catchable root for everything this package raises."""
from __future__ import annotations


class DfdError(Exception):
    """Base for every error raised by the dfd package."""


class InvalidInput(DfdError):
    """A caller supplied an argument that cannot be processed."""


class ResourceLimitExceeded(DfdError):
    """Input exceeded a configured decode or size limit."""
```

```python
# src/dfd/audit.py
"""Immutable per-decision audit record (spec §7.2).

Makes a rejection defensible: what was decided, by which model versions, on
what evidence, under which policy. References the input by SHA-256 and never
carries image bytes — the record is retained far longer than the media, and
BFSI face data is sensitive personal data under India's DPDP Act.

Immutable means immutable in depth: `frozen=True` alone leaves dict and list
fields writable, and mutating one changes the digest, which would let a record
be altered after the fact and re-digested to match.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .errors import InvalidInput
from .types import Evidence, Verdict

logger = logging.getLogger(__name__)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
AUDIT_SCHEMA_VERSION = "1"


def _freeze(value: Any) -> Any:
    """Deep-freeze the containers a record holds."""
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    """Plain-Python view for serialisation. No `default=` fallback: a value
    this cannot render must raise rather than be silently stringified."""
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


@dataclass(frozen=True)
class AuditRecord:
    schema_version: str
    sample_id: str
    input_sha256: str
    verdict: str
    llr_total: float
    posterior: float
    evidence: tuple
    quality_band: str
    ood_score: float
    policy_version: str
    threshold: float
    model_versions: Mapping[str, str]
    created_at: str

    def to_json(self) -> str:
        """Serialise deterministically (sorted keys) so digests are comparable.

        `dataclasses.asdict` is deliberately not used: it deep-copies every
        field value, and a `MappingProxyType` cannot be deep-copied.

        Raises:
            TypeError: if any field holds a value JSON cannot represent.
        """
        payload = {f.name: _thaw(getattr(self, f.name)) for f in fields(self)}
        return json.dumps(payload, sort_keys=True)


def build_audit_record(
    sample_id: str,
    input_sha256: str,
    verdict: Verdict,
    llr_total: float,
    posterior: float,
    evidence: Sequence[Evidence],
    quality_band: str,
    ood_score: float,
    policy_version: str,
    threshold: float,
    model_versions: Mapping[str, str],
    created_at: str | None = None,
) -> AuditRecord:
    """Build an immutable decision record.

    `created_at` is injectable so that two records describing the same decision
    are genuinely identical. It is covered by `record_digest`: a timestamp
    outside the digest makes backdating a decision invisible.

    Raises:
        InvalidInput: if `sample_id` is empty or `input_sha256` is not a
            lowercase 64-character hex digest.
    """
    if not sample_id:
        raise InvalidInput("sample_id must be a non-empty string")
    if not _SHA256_RE.match(input_sha256 or ""):
        raise InvalidInput(
            "input_sha256 must be 64 lowercase hex characters, got "
            f"{input_sha256!r}")

    rows = tuple(
        MappingProxyType({
            "detector": e.detector,
            "version": e.detector_version,
            "llr": float(e.llr),
            "raw_score": None if e.raw_score is None else float(e.raw_score),
            "abstained": bool(e.abstained),
            "reason": e.reason,
        })
        for e in evidence
    )
    record = AuditRecord(
        schema_version=AUDIT_SCHEMA_VERSION,
        sample_id=sample_id,
        input_sha256=input_sha256,
        verdict=verdict.value if isinstance(verdict, Verdict) else str(verdict),
        llr_total=float(llr_total),
        posterior=float(posterior),
        evidence=rows,
        quality_band=quality_band,
        ood_score=float(ood_score),
        policy_version=policy_version,
        threshold=float(threshold),
        model_versions=_freeze(dict(model_versions)),
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
    )
    logger.info("audit record built: sample=%s verdict=%s detectors=%d",
                sample_id, record.verdict, len(rows))
    return record


def record_digest(record: AuditRecord) -> str:
    """Tamper-evident digest over the whole record, timestamp included."""
    return hashlib.sha256(record.to_json().encode()).hexdigest()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_audit.py -v`
Expected: PASS, 25 tests (14 plus the digest test parametrised over 11 fields).

- [ ] **Step 5: Prove the tamper-evidence tests can fail**

Two of these guard properties that the obvious implementation silently lacks. Prove each fires:

1. Make `record_digest` drop the timestamp before hashing (`payload = json.loads(record.to_json()); payload.pop("created_at")`). Run `pytest tests/test_audit.py -k digest_changes` and confirm the `created_at` case FAILS while the others still pass. Restore.
2. Replace `model_versions=_freeze(dict(model_versions))` with `model_versions=dict(model_versions)`. Run `pytest tests/test_audit.py -k mutated` and confirm it FAILS. Restore.

Record all four outputs in the report.

- [ ] **Step 6: Commit**

```bash
git add src/dfd/errors.py src/dfd/audit.py tests/test_audit.py
git commit -m "feat: immutable per-decision audit record and error hierarchy"
```

---

### Task 21: Resource limits at the decode boundary

**Files:**
- Create: `src/dfd/limits.py`
- Modify: `src/dfd/ingest/image.py`, `src/dfd/ingest/video.py`
- Test: `tests/test_limits.py`

**Interfaces:**
- Consumes: `InvalidInput`, `ResourceLimitExceeded` (Task 20); `load_image`, `load_video` (Task 5)
- Produces: `Limits`, `DEFAULT_LIMITS`, `check_file_size(path, limits)`, `check_frame_dims(w, h, limits)`, `probe_image_dims(path)`, `check_image_before_decode(path, limits)`

**Why this task exists:** spec §3A assumes a well-resourced adversary and spec §10 names resource exhaustion via crafted media as an attack surface. A 50,000×50,000 PNG decodes to 7.5 GB and takes the service down — a denial-of-service against a fraud control is itself a fraud enabler, because it forces a fallback path.

**Limits are enforced before allocation, and that word is load-bearing.** The obvious design — check the file size, decode, then check the decoded array's shape — does not defend against this attack at all, for two reasons:

1. **A file-size limit is exactly what a decompression bomb defeats.** Measured: a 12,000×12,000 uniform PNG is **161,331 bytes on disk** — 0.15 MB — and decodes to **0.40 GB**. It passes a 256 MB file limit with five orders of magnitude to spare. The task's own headline example, 50,000×50,000, is roughly 2.8 MB on disk and 7.5 GB decoded.
2. **Checking `decoded.shape` runs after the allocation it exists to prevent.** By the time an array has a shape, the memory is already committed.

So dimensions are read from the image **header**, before any decode. Pillow does this: `Image.open(path).size` returned `(12000, 12000)` in **0.007 s** without decoding a pixel. Only a frame whose header dimensions pass is handed to `cv2.imread`.

**Pillow's own bomb guard must be absorbed, not left to fire on its own.** Pillow warns with `DecompressionBombWarning` above `MAX_IMAGE_PIXELS` (89,478,485 by default) and raises `DecompressionBombError` above twice that. Our `max_pixels` is 33.2 M — stricter than Pillow's — so our check is the one that should speak. `probe_image_dims` suppresses Pillow's warning and converts its error into `ResourceLimitExceeded`, giving one source of truth and keeping test output pristine.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_limits.py
import dataclasses

import cv2
import numpy as np
import pytest

from dfd.errors import DfdError, InvalidInput, ResourceLimitExceeded
from dfd.limits import (
    DEFAULT_LIMITS, Limits, check_file_size, check_frame_dims,
    check_image_before_decode, probe_image_dims,
)
from dfd.ingest.image import load_image
from dfd.types import Context


def _png(path, width, height):
    """A uniform image: large in pixels, tiny on disk. That gap is the attack."""
    cv2.imwrite(str(path), np.zeros((height, width), dtype=np.uint8))
    return path


@pytest.fixture
def context():
    return Context(label=0)


def test_limits_are_a_frozen_value_object():
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_LIMITS.max_pixels = 1


def test_defaults_are_the_documented_values():
    """Asserting only `> 0` would let max_pixels drift to 1 unnoticed."""
    assert DEFAULT_LIMITS.max_pixels == 7680 * 4320
    assert DEFAULT_LIMITS.max_file_bytes == 256 * 1024 * 1024
    assert DEFAULT_LIMITS.max_frames == 10_000
    assert DEFAULT_LIMITS.max_duration_s == 1800.0


def test_oversized_file_is_rejected(tmp_path):
    p = tmp_path / "big.bin"
    p.write_bytes(b"0" * 2048)
    with pytest.raises(ResourceLimitExceeded, match="exceeds limit 1024"):
        check_file_size(p, Limits(max_file_bytes=1024))


def test_file_within_limit_passes(tmp_path):
    p = tmp_path / "ok.bin"
    p.write_bytes(b"0" * 100)
    assert check_file_size(p, Limits(max_file_bytes=1024)) is None


def test_missing_file_is_invalid_input_not_a_resource_limit(tmp_path):
    """`DfdError` alone cannot tell these apart — ResourceLimitExceeded is one."""
    with pytest.raises(InvalidInput, match="not a readable file"):
        check_file_size(tmp_path / "nope.bin", DEFAULT_LIMITS)


@pytest.mark.parametrize("width,height", [
    (50000, 50000),        # the square bomb
    (1, 10 ** 9),          # a degenerate strip: same pixel count, no large side
    (10 ** 9, 1),          # and its transpose
    (7681, 4320),          # one pixel over the cap
])
def test_oversized_dimensions_are_rejected(width, height):
    """Shape is parametrised deliberately. A check that compares each side
    against a maximum instead of the product passes the strips."""
    with pytest.raises(ResourceLimitExceeded, match="exceeds limit"):
        check_frame_dims(width, height, DEFAULT_LIMITS)


@pytest.mark.parametrize("width,height", [(1920, 1080), (7680, 4320), (1, 1)])
def test_dimensions_within_the_cap_pass(width, height):
    assert check_frame_dims(width, height, DEFAULT_LIMITS) is None


@pytest.mark.parametrize("width,height", [(0, 100), (100, 0), (-1, 100), (100, -1)])
def test_non_positive_dimensions_are_invalid_input(width, height):
    with pytest.raises(InvalidInput, match="must be positive"):
        check_frame_dims(width, height, DEFAULT_LIMITS)


def test_probe_reads_dimensions_from_the_header(tmp_path):
    _png(tmp_path / "a.png", 640, 480)
    assert probe_image_dims(tmp_path / "a.png") == (640, 480)


def test_probe_reads_a_bomb_without_decoding_it(tmp_path):
    """12000x12000 is 144M pixels. If this decoded, it would allocate ~0.4GB."""
    _png(tmp_path / "bomb.png", 12000, 12000)
    assert probe_image_dims(tmp_path / "bomb.png") == (12000, 12000)


def test_probe_emits_no_warnings_on_a_bomb(tmp_path, recwarn):
    """Pillow's own DecompressionBombWarning must be absorbed, not leaked:
    this project requires pristine test output."""
    _png(tmp_path / "bomb.png", 12000, 12000)
    probe_image_dims(tmp_path / "bomb.png")
    assert [w.category.__name__ for w in recwarn] == []


def test_probe_rejects_a_file_that_is_not_an_image(tmp_path):
    p = tmp_path / "junk.png"
    p.write_bytes(b"not an image")
    with pytest.raises(InvalidInput, match="could not read image header"):
        probe_image_dims(p)


def test_a_bomb_passes_the_file_size_check_and_is_still_rejected(tmp_path):
    """The measurement that justifies header probing: this file is ~0.15MB,
    far under the 256MB default, and decodes to ~0.4GB."""
    p = _png(tmp_path / "bomb.png", 12000, 12000)
    assert p.stat().st_size < DEFAULT_LIMITS.max_file_bytes
    assert check_file_size(p, DEFAULT_LIMITS) is None
    with pytest.raises(ResourceLimitExceeded, match="exceeds limit"):
        check_image_before_decode(p, DEFAULT_LIMITS)


def test_the_check_runs_before_any_decode(tmp_path, monkeypatch):
    """The whole point. If cv2.imread is reached, the allocation already
    happened and the limit is decorative."""
    p = _png(tmp_path / "bomb.png", 12000, 12000)

    def _boom(*args, **kwargs):
        raise AssertionError("cv2.imread was called — decode preceded the check")

    monkeypatch.setattr(cv2, "imread", _boom)
    with pytest.raises(ResourceLimitExceeded):
        check_image_before_decode(p, DEFAULT_LIMITS)


def test_load_image_rejects_a_bomb_before_decoding_it(tmp_path, monkeypatch, context):
    """The guard must be WIRED IN. Calling the checker directly in every test
    would let the loader enforce nothing while the suite stayed green."""
    p = _png(tmp_path / "bomb.png", 12000, 12000)

    def _boom(*args, **kwargs):
        raise AssertionError("cv2.imread was called — decode preceded the check")

    monkeypatch.setattr(cv2, "imread", _boom)
    with pytest.raises(ResourceLimitExceeded):
        load_image(p, context)


def test_load_image_still_loads_a_normal_image(tmp_path, context):
    _png(tmp_path / "ok.png", 64, 48)
    sample = load_image(tmp_path / "ok.png", context)
    assert sample.observations[0].payload.shape == (48, 64, 3)


def test_resource_limit_exceeded_is_a_dfd_error():
    assert issubclass(ResourceLimitExceeded, DfdError)
    assert issubclass(InvalidInput, DfdError)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_limits.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dfd.limits'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/dfd/limits.py
"""Resource limits enforced at every decode boundary (spec §3A, §10).

Media arrives from an adversary. A crafted image can allocate gigabytes before
any detection logic runs, and a denial-of-service against a fraud control is a
fraud enabler: it forces the fallback path.

Two things this module refuses to do, because both are the usual way this
control is built and neither works:

- It does not treat file size as a proxy for decoded size. A 12,000x12,000
  uniform PNG occupies 161 KB on disk and 0.40 GB decoded; defeating a size
  limit is what a decompression bomb IS.
- It does not inspect a decoded array's shape. By the time an array has a
  shape the memory is already committed.

Dimensions come from the image header instead, before any decode.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .errors import InvalidInput, ResourceLimitExceeded

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Limits:
    # 8K RGB decodes to ~100MB; beyond this nothing legitimate in v-CIP arrives.
    max_pixels: int = 7680 * 4320
    max_file_bytes: int = 256 * 1024 * 1024
    max_frames: int = 10_000
    max_duration_s: float = 1800.0


DEFAULT_LIMITS = Limits()


def check_file_size(path: str | Path, limits: Limits = DEFAULT_LIMITS) -> None:
    """Raise unless the file exists and is within `limits.max_file_bytes`.

    This is a cheap first gate against a merely huge file. It is NOT a defence
    against a decompression bomb — see the module docstring.

    Raises:
        InvalidInput: the path does not exist or is not a regular file.
        ResourceLimitExceeded: the file is larger than the configured maximum.
    """
    p = Path(path)
    if not p.is_file():
        raise InvalidInput(f"not a readable file: {p}")
    size = p.stat().st_size
    if size > limits.max_file_bytes:
        raise ResourceLimitExceeded(
            f"file {p.name} is {size} bytes, exceeds limit {limits.max_file_bytes}")


def check_frame_dims(width: int, height: int,
                     limits: Limits = DEFAULT_LIMITS) -> None:
    """Raise unless the frame dimensions are positive and within the pixel cap.

    The cap is on the PRODUCT, not on either side: a 1 x 10**9 strip carries
    the same allocation as a square bomb and has no large dimension.

    Raises:
        InvalidInput: a dimension is zero or negative.
        ResourceLimitExceeded: width * height exceeds `limits.max_pixels`.
    """
    if width <= 0 or height <= 0:
        raise InvalidInput(
            f"frame dimensions must be positive, got {width}x{height}")
    if width * height > limits.max_pixels:
        raise ResourceLimitExceeded(
            f"frame {width}x{height} = {width * height} pixels, "
            f"exceeds limit {limits.max_pixels}")


def probe_image_dims(path: str | Path) -> tuple[int, int]:
    """Read (width, height) from the image header without decoding it.

    Pillow's own bomb guard is absorbed here rather than allowed to surface:
    its threshold is looser than ours, so our limit should be the one that
    speaks, and its warning would otherwise pollute output.

    Raises:
        InvalidInput: the header cannot be read.
        ResourceLimitExceeded: Pillow refused the image as a bomb outright.
    """
    p = Path(path)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(p) as im:
                return int(im.width), int(im.height)
    except Image.DecompressionBombError as exc:
        raise ResourceLimitExceeded(
            f"{p.name} rejected as a decompression bomb: {exc}") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidInput(f"could not read image header for {p}: {exc}") from exc


def check_image_before_decode(path: str | Path,
                              limits: Limits = DEFAULT_LIMITS) -> tuple[int, int]:
    """Gate an image file before a single pixel is decoded.

    Returns:
        The header (width, height), so callers need not read it twice.

    Raises:
        InvalidInput: unreadable path or unreadable header.
        ResourceLimitExceeded: file too large, or too many pixels.
    """
    check_file_size(path, limits)
    width, height = probe_image_dims(path)
    check_frame_dims(width, height, limits)
    return width, height
```

Then in `src/dfd/ingest/image.py`, call `check_image_before_decode(path)` **before** `cv2.imread` — not after, and not on the decoded array. In `src/dfd/ingest/video.py`, call `check_file_size(path)` before opening and clamp the requested frame count to `limits.max_frames`. Import from `..limits`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_limits.py tests/test_ingest.py -v`
Expected: PASS — 28 new tests, Task 5's still green.

- [ ] **Step 5: Prove the ordering and wiring tests can fail**

These are the two properties the whole task rests on, and both are invisible to a test that merely calls the checker directly.

1. **Ordering.** In `check_image_before_decode`, move `check_file_size` and `check_frame_dims` to run *after* a `cv2.imread(str(path))` call. Run `pytest tests/test_limits.py -k before_any_decode` and confirm it FAILS with the `cv2.imread was called` assertion. Restore.
2. **Wiring.** In `src/dfd/ingest/image.py`, delete the `check_image_before_decode` call. Run `pytest tests/test_limits.py -k load_image_rejects` and confirm it FAILS. Restore.

Record all four outputs in the report.

- [ ] **Step 6: Commit**

```bash
git add src/dfd/limits.py src/dfd/ingest tests/test_limits.py
git commit -m "feat: resource limits enforced before decode, not after"
```

---

### Task 22: CI pipeline with lint, type and coverage gates

**Files:**
- Create: `.github/workflows/ci.yml`, `ruff.toml`, `mypy.ini`
- Modify: `pyproject.toml`
- Test: `tests/test_ci_gates.py`

**Interfaces:**
- Consumes: `assert_all_assets_registered` (Task 19)
- Produces: a CI workflow that fails on lint, type, coverage or unregistered-asset violations

**Why this task exists:** the production standards in Global Constraints are only real if something enforces them. A standard enforced by intention is a standard that decays by the third contributor. This also wires Task 19's asset gate into CI, which is what makes spec §12.1 criterion 6 an enforced property rather than a claim.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ci_gates.py
"""The CI config is itself tested: a gate nobody verifies is a gate that rots."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_ci_workflow_exists():
    assert (ROOT / ".github/workflows/ci.yml").is_file()


def test_ci_runs_every_gate():
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    for gate in ("ruff", "mypy", "pytest"):
        assert gate in ci, f"CI does not run {gate}"


def test_ci_enforces_the_asset_registration_gate():
    """Spec criterion 6 is only real if CI fails on an unregistered weight file."""
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "assert_all_assets_registered" in ci or "asset_scan" in ci


def test_mypy_is_configured_strict():
    cfg = (ROOT / "mypy.ini").read_text()
    assert "strict = True" in cfg or "strict=True" in cfg


def test_ruff_bans_silent_exception_handling():
    """Global constraint: a swallowed error in a fraud detector is an approved fraud."""
    cfg = (ROOT / "ruff.toml").read_text()
    # E722 = bare except; BLE = blind except; S110 = try/except/pass
    assert "E722" in cfg
    assert "BLE" in cfg or "S110" in cfg


def test_no_bare_except_anywhere_in_src():
    """Enforced here too, so the rule holds even if ruff config drifts."""
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip() == "except:":
                offenders.append(f"{path}:{n}")
    assert offenders == [], f"bare except found: {offenders}"


def test_no_print_statements_in_src():
    """Global constraint: structured logging, never print."""
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip().startswith("print("):
                offenders.append(f"{path}:{n}")
    assert offenders == [], f"print() found in src: {offenders}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_ci_gates.py -v`
Expected: FAIL — the workflow and config files do not exist

- [ ] **Step 3: Write minimal implementation**

```toml
# ruff.toml
line-length = 100
target-version = "py310"

[lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "A", "C4", "S", "BLE", "RET", "SIM"]
# E722 bare except, BLE001 blind except, S110 try-except-pass: a swallowed error
# in a fraud detector is a fraud that was approved. Never silence these.
ignore = ["S101"]  # assert is fine in tests

[lint.per-file-ignores]
"tests/*" = ["S", "N802"]
```

```ini
# mypy.ini
[mypy]
python_version = 3.10
strict = True
warn_unreachable = True
files = src/dfd

[mypy-cv2.*]
ignore_missing_imports = True

[mypy-torch.*]
ignore_missing_imports = True

[mypy-sklearn.*]
ignore_missing_imports = True

[mypy-yaml.*]
ignore_missing_imports = True
```

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  gates:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - name: Install
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt
          pip install -e .
      - name: Lint
        run: ruff check .
      - name: Types
        run: mypy --config-file mypy.ini
      - name: Tests
        run: pytest -q --cov=src/dfd --cov-fail-under=85
      - name: Asset registration gate
        run: |
          python -c "
          from dfd.asset_scan import assert_all_assets_registered
          assert_all_assets_registered('.', 'assets/manifest.yaml')
          print('all assets registered and commercially cleared')
          "
```

Also create `requirements-dev.txt` listing: `pytest`, `pytest-cov`, `ruff`, `mypy`, `numpy`, `opencv-python-headless`, `scikit-learn`, `pyyaml`, `torch`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_ci_gates.py -v && ruff check . && mypy --config-file mypy.ini`
Expected: 7 tests PASS; ruff and mypy clean. Fix any violations they surface in existing code — that is the point of the gate.

- [ ] **Step 5: Commit**

```bash
git add .github ruff.toml mypy.ini requirements-dev.txt pyproject.toml tests/test_ci_gates.py
git commit -m "feat: CI gates for lint, strict types, coverage and asset registration"
```
