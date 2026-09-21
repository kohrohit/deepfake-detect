# SBI Corpus and Blend-Seam Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Manufacture a licence-clean training corpus by self-blending the project's own real face captures, and train a CPU-only blending-boundary detector on it — so the pipeline can reach a real verdict without any EULA'd dataset.

**Architecture:** Three layers, each independently testable. `corpora/face_pool.py` extracts aligned face crops with provenance from the 442-session capture corpus. `corpora/sbi.py` turns one real crop into a (real, pseudo-fake) pair by blending the frame with a jittered copy of itself under a landmark-derived mask — no second image, no generator model, nothing licensed. `src/dfd/detectors/blend.py` reads the seam with handcrafted concentric-annulus statistics and a linear model whose coefficients are stored as plain arrays, so loading can never execute code. Training lives at repo root (`training/fit_blend.py`), outside the installed package, because fitting is not inference.

**Tech Stack:** numpy, opencv-python-headless (YuNet face detection, MIT, `commercial_use: true`), scikit-learn (LogisticRegression + StandardScaler, fitted then discarded — only the arrays are kept). No torch: this detector is deliberately not a CNN.

**Spec:** `docs/superpowers/specs/2026-09-20-deepfake-detection-design.md` — §6 slot A (blending boundary), §11 (the clean path). Read with `docs/HANDOFF.md` §0, §4 and `docs/EULA-ACCESS.md` §1, which establish *why* this plan exists: research datasets are non-commercial and their derived data is encumbered, so the shipping detector must be trained on owned material.

## Global Constraints

- **Python `>=3.10`.** Dependency ranges are fixed in `pyproject.toml` and must not be widened: `numpy>=1.26.4,<3`, `opencv-python-headless>=4.10.0.84,<6`, `pillow>=11.0,<13`, `pyyaml>=6.0.3,<7`, `scikit-learn>=1.5,<2`, `torch>=2.4.1,<3`. **Adding any new dependency is out of scope for this plan** — three tests in `tests/test_ci_gates.py` enforce pin/range integrity and will fail.
- **The gates are exactly what `.github/workflows/ci.yml` runs, and nothing else:**
  - `ruff check src bench corpora` — note the scope: `tests/` is deliberately NOT linted (ci.yml:38-43 says so explicitly, and `ruff check .` reports 45 pre-existing errors in `tests/`). Do not lint or "fix" `tests/`.
  - `mypy --config-file mypy.ini` — `mypy.ini` sets `files = src/dfd`, so 25 files in the installed package are checked. `corpora/`, `bench/` and `training/` are deliberately outside it and carry pre-existing `--strict` errors. Do not widen `mypy.ini` and do not attempt to fix those.
  - `pytest -q --cov=src/dfd --cov-fail-under=85`
  Running `mypy --strict` directly on a `corpora/` file needs `MYPYPATH=src` (the package ships no PEP 561 `py.typed` marker) and then surfaces only opencv-stub artifacts — `cv2.imread` is typed as non-optional although it returns `None`, which is the same stub deficiency already documented in `src/dfd/faces.py`. It is not a gate; do not chase it.
- **Coverage gate is ≥85%.** Currently 95.12%.
- **No test may depend on a weight file existing.** `assets/models/face_detection_yunet_2023mar.onnx` is gitignored and absent in CI. Every function that needs face detection takes the detector as an injected callable, defaulting to the real one. Tests inject a stub.
- **Never train on the evaluation set.** The 5 `swapped AND approved` sessions (`20260826-221956-387743`, `20260827-104716-349039`, `20260829-010524-969870`, `20260831-142514-700890`, `20260831-142708-227903`) and the 2 further `swapped` sessions are **evaluation only** and must never enter a training split. This is enforced in code, not by convention (Task 3).
- **Corpus data lives outside the repo.** Loaders and builders take a root path. The capture corpus is at `/home/rohit/Desktop/agents/fraud_gff/deepfake_detection/captures` on the owner's machine; nothing may hardcode it outside a CLI default.
- **`bench/protocol.py` record invariants** (enforced at `bench/protocol.py:67-96`, and the reason for a non-obvious id scheme in Task 3):
  - a record with `label == 1` must have a non-empty `generator`;
  - a record with `label == 0` must have `generator is None`;
  - each `source_id` must carry **exactly one** `(subject_id, generator)` pair.
  Therefore a real crop and the pseudo-fake made from it **cannot share a `source_id`**. They share `subject_id` (so identity-disjoint splitting moves them together) and differ in `source_id`.
- **Commit after every task.** Each task ends green: `python3 -m pytest -q` passes.

---

### Task 1: Face pool — aligned crops with provenance

Extract face crops from capture sessions so later tasks have real faces to blend. Nothing here blends or scores.

**Files:**
- Create: `corpora/face_pool.py`
- Test: `tests/corpora/test_face_pool.py`

**Interfaces:**
- Consumes: `corpora.captures.CaptureSession` and `load_capture_sessions` (existing); `dfd.faces.FaceBox`, `detect_faces`, `align`; `dfd.quality.measure_quality`; `dfd.types.Quality`.
- Produces:
  - `FaceCrop` frozen dataclass with fields `session_id: str`, `frame_index: int`, `image: npt.NDArray[np.uint8]` (HWC uint8, `size`×`size`×3), `box: FaceBox`, `quality: Quality`, `swapped: bool`.
  - `build_face_pool(sessions: Sequence[CaptureSession], *, size: int = 224, detect: DetectFn = detect_faces, max_frames_per_session: int = 2) -> tuple[list[FaceCrop], dict[str, int]]` — takes NO root: `CaptureSession.folder` is already a complete path (`corpora/captures.py:72` sets `folder=str(path.parent)`), so joining a root onto it double-prefixes — returns the crops and a reason→count tally of what was skipped.
  - `DetectFn` type alias: `Callable[[npt.NDArray[np.uint8]], list[FaceBox]]`.
  - Skip reason constants `NO_FRAMES = "no_frames"`, `NO_FACE = "no_face"`, `UNREADABLE = "unreadable"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/corpora/test_face_pool.py`:

```python
"""build_face_pool turns sessions into aligned crops, and says what it dropped."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from corpora.captures import CaptureSession
from corpora.face_pool import NO_FACE, UNREADABLE, build_face_pool


def _session(session_id: str, swapped: bool = False) -> CaptureSession:
    return CaptureSession(session_id=session_id, folder=session_id, swapped=swapped,
                          approved=True, scan_verdict="LIVE", frame_count=2)


def _write_frames(root: Path, session_id: str, n: int = 2) -> None:
    d = root / session_id
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        img = rng.integers(0, 255, (120, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(d / f"frame_{i:02d}.jpg"), img)


def _fake_box():
    from dfd.faces import FaceBox
    lms = np.array([[30.0, 40.0], [60.0, 40.0], [45.0, 55.0],
                    [33.0, 70.0], [57.0, 70.0]])
    return FaceBox(x=20, y=25, w=50, h=60, landmarks=lms, score=0.99)


def test_one_crop_per_frame_with_provenance(tmp_path: Path) -> None:
    _write_frames(tmp_path, "s1", n=2)
    crops, skipped = build_face_pool([_session("s1")], tmp_path,
                                     detect=lambda f: [_fake_box()])
    assert len(crops) == 2
    assert {c.frame_index for c in crops} == {0, 1}
    assert all(c.session_id == "s1" for c in crops)
    assert all(c.image.shape == (224, 224, 3) for c in crops)
    assert all(c.image.dtype == np.uint8 for c in crops)
    assert skipped == {}


def test_swapped_flag_is_carried_from_the_session(tmp_path: Path) -> None:
    _write_frames(tmp_path, "s_fraud", n=1)
    crops, _ = build_face_pool([_session("s_fraud", swapped=True)], tmp_path,
                               detect=lambda f: [_fake_box()])
    assert crops and all(c.swapped for c in crops)


def test_frames_with_no_detected_face_are_skipped_and_counted(tmp_path: Path) -> None:
    _write_frames(tmp_path, "s1", n=2)
    crops, skipped = build_face_pool([_session("s1")], tmp_path, detect=lambda f: [])
    assert crops == []
    assert skipped == {NO_FACE: 2}


def test_an_unreadable_frame_is_counted_not_raised(tmp_path: Path) -> None:
    d = tmp_path / "s1"
    d.mkdir(parents=True)
    (d / "frame_00.jpg").write_bytes(b"not a jpeg")
    crops, skipped = build_face_pool([_session("s1")], tmp_path,
                                     detect=lambda f: [_fake_box()])
    assert crops == []
    assert skipped == {UNREADABLE: 1}


def test_max_frames_per_session_caps_the_pool(tmp_path: Path) -> None:
    _write_frames(tmp_path, "s1", n=5)
    crops, _ = build_face_pool([_session("s1")], tmp_path,
                               detect=lambda f: [_fake_box()],
                               max_frames_per_session=2)
    assert len(crops) == 2
    assert [c.frame_index for c in crops] == [0, 1]


def test_highest_scoring_face_is_chosen_when_several_are_detected(tmp_path: Path) -> None:
    from dfd.faces import FaceBox
    _write_frames(tmp_path, "s1", n=1)
    lms = np.array([[30.0, 40.0], [60.0, 40.0], [45.0, 55.0],
                    [33.0, 70.0], [57.0, 70.0]])
    small = FaceBox(x=0, y=0, w=10, h=10, landmarks=lms, score=0.5)
    big = FaceBox(x=20, y=25, w=50, h=60, landmarks=lms, score=0.95)
    crops, _ = build_face_pool([_session("s1")], tmp_path, detect=lambda f: [small, big])
    assert len(crops) == 1
    assert crops[0].box.score == pytest.approx(0.95)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/corpora/test_face_pool.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'corpora.face_pool'`

- [ ] **Step 3: Write the implementation**

Create `corpora/face_pool.py`:

```python
"""Aligned face crops, with provenance, from the capture corpus.

The pool is the only real-face source this project owns outright, so every
crop carries the session it came from and whether that session was swapped.
Both travel with the crop because the split discipline downstream depends on
them: swapped sessions are evaluation-only and must never be blended into
training data.

Face detection is injected rather than imported-and-called so that tests need
no weight file. The YuNet weights are gitignored and absent in CI.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from dfd.faces import FaceBox, align, detect_faces
from dfd.quality import measure_quality
from dfd.types import Quality

from .captures import CaptureSession

logger = logging.getLogger(__name__)

#: A face detector: frame in, boxes out. `dfd.faces.detect_faces` satisfies it.
DetectFn = Callable[[npt.NDArray[np.uint8]], list[FaceBox]]

NO_FRAMES = "no_frames"
NO_FACE = "no_face"
UNREADABLE = "unreadable"

#: Aligned crop edge length, in pixels. 224 matches `dfd.faces.align`'s default
#: and the resolution the seam features in `dfd.detectors.blend` assume.
DEFAULT_CROP_SIZE = 224

#: Frames taken per session. The corpus holds 1-5 frames per session and 163
#: sessions have exactly 3; taking 2 keeps sessions with few frames from being
#: under-represented relative to sessions with many, which would otherwise
#: weight the pool toward whichever sessions happened to record longest.
DEFAULT_MAX_FRAMES = 2


@dataclass(frozen=True)
class FaceCrop:
    """One aligned face, and everything needed to place it in a split."""
    session_id: str
    frame_index: int
    image: npt.NDArray[np.uint8]
    box: FaceBox
    quality: Quality
    swapped: bool


def build_face_pool(
    sessions: Sequence[CaptureSession],
    root: str | Path,
    *,
    size: int = DEFAULT_CROP_SIZE,
    detect: DetectFn = detect_faces,
    max_frames_per_session: int = DEFAULT_MAX_FRAMES,
) -> tuple[list[FaceCrop], dict[str, int]]:
    """Extract aligned face crops from capture sessions.

    Args:
        sessions: sessions to draw from, as loaded by `load_capture_sessions`.
        root: directory holding one folder per session, each with `frame_NN.jpg`.
        size: edge length of the aligned crop.
        detect: face detector. Injected so tests need no weight file.
        max_frames_per_session: cap on frames taken from any one session.

    Returns:
        A (crops, skipped) pair. `skipped` maps a reason constant to a count,
        and is empty when nothing was dropped. Never raises for bad input:
        a frame that cannot be decoded is counted, not propagated, because
        one corrupt JPEG must not cost the other 441 sessions.
    """
    crops: list[FaceCrop] = []
    skipped: dict[str, int] = {}

    def drop(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for session in sessions:
        folder = Path(root) / session.folder
        frames = sorted(folder.glob("frame_*.jpg"))[:max_frames_per_session]
        if not frames:
            logger.debug("session %s has no frames", session.session_id)
            drop(NO_FRAMES)
            continue

        for frame_path in frames:
            bgr = cv2.imread(str(frame_path))
            if bgr is None:
                logger.warning("unreadable frame: %s", frame_path)
                drop(UNREADABLE)
                continue
            frame: npt.NDArray[np.uint8] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            boxes = detect(frame)
            if not boxes:
                logger.debug("no face in %s", frame_path)
                drop(NO_FACE)
                continue

            box = max(boxes, key=lambda b: b.score)
            quality = measure_quality(
                frame, (box.x, box.y, box.w, box.h), box.landmarks)
            index = int(frame_path.stem.split("_")[-1])
            crops.append(FaceCrop(
                session_id=session.session_id,
                frame_index=index,
                image=align(frame, box, size=size),
                box=box,
                quality=quality,
                swapped=session.swapped,
            ))

    logger.info("face pool: %d crops from %d sessions, skipped %s",
                len(crops), len(sessions), skipped or "nothing")
    return crops, skipped
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/corpora/test_face_pool.py -q`
Expected: 6 passed

- [ ] **Step 5: Prove each test can fail**

For each test, break the implementation, watch it fail for the right reason, restore. A test nobody watched fail is a hope, not a guard (`docs/HANDOFF.md` §6). Specifically:
- Change `max(boxes, key=lambda b: b.score)` to `boxes[0]` → `test_highest_scoring_face_is_chosen_when_several_are_detected` fails.
- Change `swapped=session.swapped` to `swapped=False` → `test_swapped_flag_is_carried_from_the_session` fails.
- Remove the `if bgr is None` branch → `test_an_unreadable_frame_is_counted_not_raised` errors rather than passing.
- Drop the `[:max_frames_per_session]` slice → `test_max_frames_per_session_caps_the_pool` fails.

- [ ] **Step 6: Check the gates and commit**

```bash
python3 -m pytest -q && ruff check . && mypy --strict
git add corpora/face_pool.py tests/corpora/test_face_pool.py
git commit -m "feat: aligned face crops with the provenance a split needs"
```

---

### Task 2: The self-blend transform

One real frame in, one pseudo-fake out. No second image, no generator model — this is the whole reason the corpus is licence-clean.

**Files:**
- Create: `corpora/sbi.py`
- Test: `tests/corpora/test_sbi.py`

**Interfaces:**
- Consumes: `dfd.faces.FaceBox`.
- Produces:
  - `self_blend(frame: npt.NDArray[np.uint8], box: FaceBox, rng: np.random.Generator) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]` — returns `(blended_frame, mask)`; mask is float32 in `[0, 1]`, same H×W as the frame.
  - `face_mask(shape: tuple[int, int], box: FaceBox, rng: np.random.Generator) -> npt.NDArray[np.float32]`
  - `jitter(frame: npt.NDArray[np.uint8], rng: np.random.Generator) -> npt.NDArray[np.uint8]`

- [ ] **Step 1: Write the failing tests**

Create `tests/corpora/test_sbi.py`:

```python
"""The self-blend makes a seam without a second image."""
from __future__ import annotations

import numpy as np

from corpora.sbi import face_mask, jitter, self_blend
from dfd.faces import FaceBox


def _frame(h: int = 200, w: int = 180) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(40, 210, (h, w, 3), dtype=np.uint8)


def _box() -> FaceBox:
    lms = np.array([[70.0, 80.0], [110.0, 80.0], [90.0, 100.0],
                    [75.0, 125.0], [105.0, 125.0]])
    return FaceBox(x=55, y=55, w=70, h=90, landmarks=lms, score=0.99)


def test_blend_is_deterministic_for_a_given_seed() -> None:
    f, b = _frame(), _box()
    a, _ = self_blend(f, b, np.random.default_rng(3))
    c, _ = self_blend(f, b, np.random.default_rng(3))
    assert np.array_equal(a, c)


def test_different_seeds_give_different_blends() -> None:
    f, b = _frame(), _box()
    a, _ = self_blend(f, b, np.random.default_rng(1))
    c, _ = self_blend(f, b, np.random.default_rng(2))
    assert not np.array_equal(a, c)


def test_shape_and_dtype_are_preserved() -> None:
    f, b = _frame(), _box()
    out, mask = self_blend(f, b, np.random.default_rng(0))
    assert out.shape == f.shape
    assert out.dtype == np.uint8
    assert mask.shape == f.shape[:2]
    assert mask.dtype == np.float32


def test_pixels_inside_the_mask_actually_change() -> None:
    f, b = _frame(), _box()
    out, mask = self_blend(f, b, np.random.default_rng(0))
    core = mask > 0.9
    assert core.any(), "mask never reaches full weight; there is no blend"
    changed = (out[core] != f[core]).any(axis=-1).mean()
    assert changed > 0.5


def test_pixels_far_outside_the_mask_are_untouched() -> None:
    f, b = _frame(), _box()
    out, mask = self_blend(f, b, np.random.default_rng(0))
    outside = mask == 0.0
    assert outside.any()
    assert np.array_equal(out[outside], f[outside])


def test_mask_is_bounded_and_covers_part_but_not_all_of_the_frame() -> None:
    m = face_mask((200, 180), _box(), np.random.default_rng(0))
    assert m.min() >= 0.0 and m.max() <= 1.0
    coverage = float((m > 0.5).mean())
    assert 0.0 < coverage < 0.5


def test_mask_has_a_soft_edge_rather_than_a_hard_step() -> None:
    m = face_mask((200, 180), _box(), np.random.default_rng(0))
    partial = ((m > 0.05) & (m < 0.95)).sum()
    assert partial > 0, "a hard-edged mask is a paste, not a blend"


def test_jitter_changes_the_image_without_changing_its_shape() -> None:
    f = _frame()
    j = jitter(f, np.random.default_rng(0))
    assert j.shape == f.shape
    assert j.dtype == np.uint8
    assert not np.array_equal(j, f)


def test_the_blend_leaves_a_measurable_discontinuity() -> None:
    """The point of the whole exercise, stated as what is actually true.

    An earlier draft of this test asserted the seam RAISES gradient energy at
    the mask edge. That is false, and was measured to be false on two separate
    fixtures: the jitter downsamples before it upsamples, so the blended
    region carries LESS detail than what it replaced, and edge energy falls
    (118.7 -> 64.0 on noise, 7.77 -> 4.35 on a textured field). What a
    composite actually guarantees is a DISCONTINUITY — two regions with
    different imaging statistics meeting along a curve — and that is what the
    detector reads and what this asserts. The gap widened 0.77 -> 100.4 and
    0.55 -> 7.0 on those same two fixtures; the 3x bar below is far inside it.
    """
    import cv2
    f, b = _frame(), _box()
    out, mask = self_blend(f, b, np.random.default_rng(0))
    core = mask > 0.9
    outside = mask == 0.0
    assert core.any() and outside.any()

    def lap_mean(img: np.ndarray, m: np.ndarray) -> float:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return float(np.abs(cv2.Laplacian(gray, cv2.CV_64F))[m].mean())

    before = abs(lap_mean(f, core) - lap_mean(f, outside))
    after = abs(lap_mean(out, core) - lap_mean(out, outside))
    assert after > before * 3, (
        f"blend did not create a statistical discontinuity: {before:.2f} -> {after:.2f}")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/corpora/test_sbi.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'corpora.sbi'`

- [ ] **Step 3: Write the implementation**

Create `corpora/sbi.py`:

```python
"""Self-blended images: a pseudo-fake made from one real frame and nothing else.

Shiohara and Yamasaki, CVPR 2022. The method blends a frame with a
photometrically and geometrically jittered copy of *itself* under a
landmark-derived mask. The result has the one artifact every face swap shares
— a composite seam — while the identity, the camera and the lighting are
unchanged, so a detector trained on it learns the seam rather than a
generator's signature.

IMPORTANT: the jitter ranges and mask geometry below are a reimplementation
from the paper's description, not a port of the authors' code, and the exact
constants are UNVERIFIED against it. The physical property — a soft-edged
composite boundary between two versions of the same face — is what matters
and holds either way. Verify the constants before any published claim rests
on the numbers.

Why this matters legally, not just technically: it consumes only real faces
this project owns, and no licensed dataset or generator weight file. See
docs/EULA-ACCESS.md §1.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
import numpy.typing as npt

from dfd.faces import FaceBox

logger = logging.getLogger(__name__)

#: Photometric jitter. Deliberately small: a swap that survives a human
#: reviewer does not shift colour grossly, and an exaggerated range would
#: teach the detector to spot colour casts rather than seams.
BRIGHTNESS_RANGE = (-12.0, 12.0)
CONTRAST_RANGE = (0.92, 1.08)
CHANNEL_GAIN_RANGE = (0.96, 1.04)

#: Resolution jitter: downsample then upsample, so the source carries slightly
#: less detail than the target. Real swaps almost always do, because the
#: generated face is produced at a fixed and usually lower resolution.
RESCALE_RANGE = (0.70, 1.00)

#: Geometric jitter, in fractions of the box. Sub-pixel to a few pixels.
SHIFT_RANGE = (-0.03, 0.03)
SCALE_RANGE = (0.97, 1.03)

#: Mask ellipse, as fractions of the face box. Under 1.0 so the seam falls
#: inside the face rather than on the jawline, where a crop boundary would
#: confound it.
MASK_AXIS_RANGE = (0.62, 0.92)
MASK_CENTRE_JITTER = 0.04
MASK_ANGLE_RANGE = (-15.0, 15.0)

#: Feather width as a fraction of the box's smaller side. A hard edge is a
#: paste, not a blend, and would be trivially detectable for the wrong reason.
FEATHER_RANGE = (0.06, 0.18)


def _uniform(rng: np.random.Generator, lohi: tuple[float, float]) -> float:
    return float(rng.uniform(lohi[0], lohi[1]))


def jitter(frame: npt.NDArray[np.uint8],
           rng: np.random.Generator) -> npt.NDArray[np.uint8]:
    """Return a photometrically and geometrically altered copy of `frame`.

    Args:
        frame: RGB HWC uint8 image.
        rng: seeded generator; the same seed yields the same output.

    Returns:
        An RGB HWC uint8 image of identical shape.
    """
    h, w = frame.shape[:2]
    x = frame.astype(np.float32)

    # Resolution: down then up, losing detail the target still has.
    scale = _uniform(rng, RESCALE_RANGE)
    if scale < 1.0:
        small = cv2.resize(x, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
        x = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # Photometry.
    contrast = _uniform(rng, CONTRAST_RANGE)
    brightness = _uniform(rng, BRIGHTNESS_RANGE)
    gains = np.array([_uniform(rng, CHANNEL_GAIN_RANGE) for _ in range(3)],
                     dtype=np.float32)
    x = x * contrast * gains + brightness

    # Geometry: a small similarity transform about the centre.
    dx = _uniform(rng, SHIFT_RANGE) * w
    dy = _uniform(rng, SHIFT_RANGE) * h
    s = _uniform(rng, SCALE_RANGE)
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.0, s)
    m[0, 2] += dx
    m[1, 2] += dy
    x = cv2.warpAffine(x, m, (w, h), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)

    return np.clip(x, 0, 255).astype(np.uint8)


def face_mask(shape: tuple[int, int], box: FaceBox,
              rng: np.random.Generator) -> npt.NDArray[np.float32]:
    """A soft-edged elliptical mask over the inner face.

    YuNet gives five landmarks, not the 68-point contour the paper's masks are
    built from, so the mask is an ellipse fitted to the box and jittered rather
    than a landmark convex hull. The consequence is stated plainly: seams sit
    on a smoother curve than a real swap's would. Randomising the axes, centre
    and angle keeps the detector from learning one fixed boundary position,
    which is the failure this approximation would otherwise cause.

    Args:
        shape: (height, width) of the frame.
        box: the detected face.
        rng: seeded generator.

    Returns:
        float32 mask in [0, 1], shape `shape`.
    """
    h, w = shape
    canvas = np.zeros((h, w), dtype=np.uint8)

    cx = box.x + box.w / 2.0 + _uniform(rng, (-MASK_CENTRE_JITTER, MASK_CENTRE_JITTER)) * box.w
    cy = box.y + box.h / 2.0 + _uniform(rng, (-MASK_CENTRE_JITTER, MASK_CENTRE_JITTER)) * box.h
    ax = max(1, int(box.w / 2.0 * _uniform(rng, MASK_AXIS_RANGE)))
    ay = max(1, int(box.h / 2.0 * _uniform(rng, MASK_AXIS_RANGE)))
    angle = _uniform(rng, MASK_ANGLE_RANGE)

    cv2.ellipse(canvas, (int(cx), int(cy)), (ax, ay), angle, 0, 360, 255, -1)

    feather = _uniform(rng, FEATHER_RANGE) * min(box.w, box.h)
    k = max(3, int(feather) | 1)  # odd kernel
    soft = cv2.GaussianBlur(canvas.astype(np.float32) / 255.0, (k, k), 0)
    return np.clip(soft, 0.0, 1.0).astype(np.float32)


def self_blend(
    frame: npt.NDArray[np.uint8],
    box: FaceBox,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]:
    """Blend `frame` with a jittered copy of itself under a face mask.

    Args:
        frame: RGB HWC uint8 image.
        box: the detected face to blend over.
        rng: seeded generator; the same seed yields the same pseudo-fake.

    Returns:
        A (blended, mask) pair. `blended` has the same shape and dtype as
        `frame`; `mask` is float32 in [0, 1] and is returned so callers can
        record where the seam is without recomputing it.
    """
    source = jitter(frame, rng)
    mask = face_mask(frame.shape[:2], box, rng)
    m3 = mask[..., None]
    blended = source.astype(np.float32) * m3 + frame.astype(np.float32) * (1.0 - m3)
    logger.debug("self-blend: mask covers %.3f of frame", float((mask > 0.5).mean()))
    return np.clip(blended, 0, 255).astype(np.uint8), mask
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/corpora/test_sbi.py -q`
Expected: 9 passed

- [ ] **Step 5: Prove each test can fail**

- Replace `mask` with `np.ones(...)` → `test_pixels_far_outside_the_mask_are_untouched` fails.
- Replace `mask` with `np.zeros(...)` → `test_pixels_inside_the_mask_actually_change` fails.
- Remove the `GaussianBlur` feather → `test_mask_has_a_soft_edge_rather_than_a_hard_step` fails.
- Return `frame` unchanged from `jitter` → `test_jitter_changes_the_image_without_changing_its_shape` fails, and so does `test_pixels_inside_the_mask_actually_change`.
- Seed the generator internally instead of using the passed `rng` → `test_different_seeds_give_different_blends` fails.
- Return `frame` for `source` (no jitter) → `test_the_blend_leaves_a_measurable_discontinuity` fails, because blending an image with itself is the identity and leaves no discontinuity at all. This is the mutation that matters most: it is the one a plausible-looking refactor could actually introduce.

- [ ] **Step 6: Check the gates and commit**

```bash
python3 -m pytest -q && ruff check . && mypy --strict
git add corpora/sbi.py tests/corpora/test_sbi.py
git commit -m "feat: self-blended pseudo-fakes, from one real frame and nothing else"
```

---

### Task 3: Corpus builder — Samples with split-safe ids

Turn the face pool into benchmark records. This task is where the evaluation set is protected and where the `source_id` invariant is satisfied; both are enforced by tests, not by care.

**Files:**
- Modify: `corpora/sbi.py` (append)
- Test: `tests/corpora/test_sbi_corpus.py`

**Interfaces:**
- Consumes: `FaceCrop` from Task 1; `self_blend` from Task 2; `dfd.types.Sample`, `Observation`, `Context`, `Modality`.
- Produces:
  - `SBI_GENERATOR = "sbi"`
  - `EvaluationOnlySessionError(ValueError)`
  - `build_sbi_corpus(crops: Sequence[FaceCrop], *, seed: int = 0) -> list[Sample]`

- [ ] **Step 1: Write the failing tests**

Create `tests/corpora/test_sbi_corpus.py`:

Create `tests/corpora/test_sbi_corpus.py`:

```python
"""The corpus builder's job is split discipline, not image processing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from corpora.face_pool import FaceCrop
from corpora.sbi import SBI_GENERATOR, EvaluationOnlySessionError, build_sbi_corpus
from dfd.faces import FaceBox
from dfd.types import Modality, Quality


def _crop(session_id: str, frame_index: int = 0, swapped: bool = False) -> FaceCrop:
    lms = np.array([[70.0, 80.0], [110.0, 80.0], [90.0, 100.0],
                    [75.0, 125.0], [105.0, 125.0]])
    rng = np.random.default_rng(len(session_id) * 1000 + frame_index)
    return FaceCrop(
        session_id=session_id,
        frame_index=frame_index,
        image=rng.integers(40, 210, (224, 224, 3), dtype=np.uint8),
        box=FaceBox(x=55, y=55, w=70, h=90, landmarks=lms, score=0.99),
        quality=Quality(inter_ocular_px=40.0, blur_var=120.0, yaw_deg=0.0,
                        pitch_deg=0.0, exposure=0.5, band="high"),
        swapped=swapped,
    )


def test_each_crop_yields_one_real_and_one_fake() -> None:
    samples = build_sbi_corpus([_crop("s1"), _crop("s2")])
    assert len(samples) == 4
    labels = sorted(s.context.label for s in samples)
    assert labels == [0, 0, 1, 1]


def test_reals_carry_no_generator_and_fakes_carry_sbi() -> None:
    """bench/protocol.py:78-86 rejects a corpus that gets this wrong."""
    samples = build_sbi_corpus([_crop("s1")])
    reals = [s for s in samples if s.context.label == 0]
    fakes = [s for s in samples if s.context.label == 1]
    assert all(s.context.generator is None for s in reals)
    assert all(s.context.generator == SBI_GENERATOR for s in fakes)


def test_a_real_and_its_own_fake_share_a_subject_but_not_a_source() -> None:
    """bench/protocol.py:92-96 requires one (subject, generator) pair per source.

    Sharing a source_id would make that source carry both (subj, None) and
    (subj, "sbi") and the corpus would be rejected. Sharing subject_id is
    required in the other direction, so identity-disjoint splitting keeps a
    face and its own pseudo-fake on the same side of every fold.
    """
    samples = build_sbi_corpus([_crop("s1")])
    real = next(s for s in samples if s.context.label == 0)
    fake = next(s for s in samples if s.context.label == 1)
    assert real.context.subject_id == fake.context.subject_id == "s1"
    assert real.observations[0].source_id != fake.observations[0].source_id


def test_a_swapped_session_is_refused_outright() -> None:
    """The 7 swapped sessions are the evaluation set. Blending them would
    train on the only labelled fraud this project has."""
    with pytest.raises(EvaluationOnlySessionError, match="s_fraud"):
        build_sbi_corpus([_crop("ok"), _crop("s_fraud", swapped=True)])


def test_the_corpus_is_reproducible_for_a_given_seed() -> None:
    a = build_sbi_corpus([_crop("s1"), _crop("s2")], seed=5)
    b = build_sbi_corpus([_crop("s1"), _crop("s2")], seed=5)
    for x, y in zip(a, b):
        assert x.sample_id == y.sample_id
        assert np.array_equal(x.observations[0].payload, y.observations[0].payload)


def test_reproducibility_survives_a_different_hash_seed() -> None:
    """Same seed, fresh interpreter, randomised PYTHONHASHSEED: same bytes.

    Within one process, `hash()` and `hashlib` are indistinguishable here, so
    the test above passes either way. This one is the guard that matters —
    str hashing is salted per process, and a corpus seeded from it would be
    irreproducible between runs while every in-process test stayed green.
    """
    import os
    import subprocess
    import sys
    import textwrap

    repo_root = Path(__file__).resolve().parents[2]
    script = textwrap.dedent("""
        import numpy as np
        from corpora.face_pool import FaceCrop
        from corpora.sbi import build_sbi_corpus
        from dfd.faces import FaceBox
        from dfd.types import Quality
        lms = np.array([[70., 80.], [110., 80.], [90., 100.],
                        [75., 125.], [105., 125.]])
        img = np.random.default_rng(11).integers(40, 210, (224, 224, 3),
                                                 dtype=np.uint8)
        crop = FaceCrop(session_id="s1", frame_index=0, image=img,
                        box=FaceBox(x=55, y=55, w=70, h=90, landmarks=lms,
                                    score=0.99),
                        quality=Quality(inter_ocular_px=40.0, blur_var=120.0,
                                        yaw_deg=0.0, pitch_deg=0.0,
                                        exposure=0.5, band="high"),
                        swapped=False)
        s = build_sbi_corpus([crop], seed=5)
        fake = next(x for x in s if x.context.label == 1)
        print(int(fake.observations[0].payload.astype(np.int64).sum()))
    """)
    outs = set()
    for hashseed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hashseed}
        r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                           text=True, env=env, cwd=str(repo_root))
        assert r.returncode == 0, r.stderr
        outs.add(r.stdout.strip())
    assert len(outs) == 1, f"corpus changed with PYTHONHASHSEED: {outs}"


def test_a_different_seed_changes_the_fakes_but_not_the_reals() -> None:
    a = build_sbi_corpus([_crop("s1")], seed=1)
    b = build_sbi_corpus([_crop("s1")], seed=2)
    real_a = next(s for s in a if s.context.label == 0)
    real_b = next(s for s in b if s.context.label == 0)
    fake_a = next(s for s in a if s.context.label == 1)
    fake_b = next(s for s in b if s.context.label == 1)
    assert np.array_equal(real_a.observations[0].payload,
                          real_b.observations[0].payload)
    assert not np.array_equal(fake_a.observations[0].payload,
                              fake_b.observations[0].payload)


def test_samples_are_images_carrying_the_crops_quality() -> None:
    samples = build_sbi_corpus([_crop("s1")])
    for s in samples:
        assert s.modality is Modality.IMAGE
        assert len(s.observations) == 1
        assert s.observations[0].quality is not None
        assert s.observations[0].quality.band == "high"


def test_the_corpus_passes_the_protocol_validator() -> None:
    """The end-to-end contract: a corpus this builder emits is splittable."""
    from bench.protocol import logo_splits
    crops = [_crop(f"s{i}") for i in range(6)]
    samples = build_sbi_corpus(crops)
    records = [
        {"sample_id": s.sample_id,
         "subject_id": s.context.subject_id,
         "source_id": s.observations[0].source_id,
         "generator": s.context.generator,
         "label": s.context.label}
        for s in samples
    ]
    splits = logo_splits(records)
    assert splits, "a single-generator corpus still yields one fold"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/corpora/test_sbi_corpus.py -q`
Expected: FAIL — `ImportError: cannot import name 'SBI_GENERATOR' from 'corpora.sbi'`

- [ ] **Step 3: Write the implementation**

Append to `corpora/sbi.py` (add `import hashlib`, `from collections.abc import Sequence`, and the `dfd.types` and `face_pool` imports at the top of the file):

```python
#: The generator name every self-blended fake carries. LOGO holds generators
#: out one at a time, so this is the single fold a self-blend-only corpus can
#: offer — which is exactly the benchmark weakness the research datasets would
#: have fixed. See docs/HANDOFF.md §4.
SBI_GENERATOR = "sbi"


class EvaluationOnlySessionError(ValueError):
    """Raised when a session reserved for evaluation is offered for blending.

    Its own type, not a bare ValueError, because a caller may reasonably want
    to catch this and filter, while a malformed-crop ValueError means a defect
    and must propagate.
    """


def build_sbi_corpus(crops: Sequence[FaceCrop], *, seed: int = 0) -> list[Sample]:
    """Turn real face crops into a labelled, splittable corpus.

    Each crop yields two samples: the crop itself as a real, and a self-blend
    of it as a fake. They share `subject_id` so identity-disjoint splitting
    moves them together, and differ in `source_id` because
    `bench.protocol._validate` requires each source to carry exactly one
    (subject, generator) pair.

    Args:
        crops: real face crops. Any crop from a swapped session is refused.
        seed: base seed. The same seed yields the same corpus.

    Returns:
        Samples, two per crop, ordered real-then-fake per crop.

    Raises:
        EvaluationOnlySessionError: if any crop comes from a swapped session.
            These are the only labelled fraud this project has; blending them
            would spend the evaluation set on training.
    """
    reserved = sorted({c.session_id for c in crops if c.swapped})
    if reserved:
        raise EvaluationOnlySessionError(
            "refusing to blend evaluation-only sessions: "
            f"{', '.join(reserved)}. Swapped sessions are the held-out fraud "
            "set; filter them out before building a training corpus.")

    samples: list[Sample] = []
    for crop in crops:
        stem = f"{crop.session_id}-{crop.frame_index:02d}"
        # hashlib, not hash(): Python salts str hashing per process unless
        # PYTHONHASHSEED is set, so hash() would make this reproducible within
        # one run and silently irreproducible between runs — the worst of both,
        # because a test calling it twice in one process would still pass.
        digest = hashlib.sha256(stem.encode()).digest()[:8]
        rng = np.random.default_rng([seed, int.from_bytes(digest, "big")])
        blended, _mask = self_blend(crop.image, _crop_box(crop), rng)

        for suffix, payload, label, generator in (
            ("real", crop.image, 0, None),
            ("sbi", blended, 1, SBI_GENERATOR),
        ):
            samples.append(Sample(
                sample_id=f"{stem}-{suffix}",
                modality=Modality.IMAGE,
                observations=(Observation(
                    t=0.0,
                    payload=payload,
                    roi=None,
                    quality=crop.quality,
                    source_id=f"{crop.session_id}:{suffix}",
                ),),
                context=Context(
                    subject_id=crop.session_id,
                    generator=generator,
                    compression=None,
                    label=label,
                ),
            ))

    logger.info("SBI corpus: %d samples from %d crops", len(samples), len(crops))
    return samples


def _crop_box(crop: FaceCrop) -> FaceBox:
    """A box covering the aligned crop.

    `FaceCrop.box` is in the ORIGINAL frame's coordinates; `crop.image` has
    already been cropped and resized by `dfd.faces.align`, so those coordinates
    do not apply to it. Blending under the original box would put the seam
    outside the crop entirely — silently producing pseudo-fakes identical to
    their reals, which every downstream metric would then reward the detector
    for failing to separate.
    """
    h, w = crop.image.shape[:2]
    return FaceBox(x=0, y=0, w=w, h=h,
                   landmarks=crop.box.landmarks.copy(), score=crop.box.score)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/corpora/test_sbi_corpus.py -q`
Expected: 8 passed

- [ ] **Step 5: Prove each test can fail**

- Set `source_id=crop.session_id` for both rows → `test_a_real_and_its_own_fake_share_a_subject_but_not_a_source` fails, and `test_the_corpus_passes_the_protocol_validator` raises from `_validate`.
- Give reals `generator=SBI_GENERATOR` → `test_reals_carry_no_generator_and_fakes_carry_sbi` fails and the validator rejects the corpus.
- Delete the `reserved` check → `test_a_swapped_session_is_refused_outright` fails.
- Use `_crop_box`'s original `crop.box` instead of the crop-space box → `test_a_different_seed_changes_the_fakes_but_not_the_reals` fails, because the blend lands outside the image and the fake equals the real.

- [ ] **Step 6: Check the gates and commit**

```bash
python3 -m pytest -q && ruff check . && mypy --strict
git add corpora/sbi.py tests/corpora/test_sbi_corpus.py
git commit -m "feat: SBI corpus with the ids the split protocol requires"
```

---

### Task 4: Seam features

A pure function: image in, fixed-length float vector out. No model, no state.

**Files:**
- Create: `src/dfd/detectors/blend.py`
- Test: `tests/test_blend_features.py`

**Interfaces:**
- Produces:
  - `ANNULI: tuple[tuple[float, float], ...]`
  - `FEATURE_NAMES: tuple[str, ...]` — length 30
  - `seam_features(img: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]` — shape `(30,)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_blend_features.py`:

```python
"""Seam features read a discontinuity, and say the same thing every time."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from dfd.detectors.blend import FEATURE_NAMES, seam_features


def _flat(size: int = 224) -> np.ndarray:
    return np.full((size, size, 3), 128, dtype=np.uint8)


def _with_ring(size: int = 224) -> np.ndarray:
    """A flat image with a soft bright annulus — a synthetic seam."""
    img = _flat(size)
    cv2.circle(img, (size // 2, size // 2), int(size * 0.3), (200, 200, 200), 6)
    return cv2.GaussianBlur(img, (7, 7), 0)


def test_returns_one_float32_value_per_named_feature() -> None:
    f = seam_features(_flat())
    assert f.shape == (len(FEATURE_NAMES),)
    assert f.dtype == np.float32
    assert len(FEATURE_NAMES) == 30


def test_feature_names_are_unique() -> None:
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_is_deterministic() -> None:
    img = _with_ring()
    assert np.array_equal(seam_features(img), seam_features(img))


def test_all_features_are_finite() -> None:
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    for img in (_flat(), _with_ring(), noisy):
        assert np.isfinite(seam_features(img)).all()


def test_a_ring_raises_residual_energy_where_the_ring_is() -> None:
    """The discriminating property. A flat image has no annular structure;
    one with a ring must differ in the band the ring falls in."""
    flat = seam_features(_flat())
    ring = seam_features(_with_ring())
    i = FEATURE_NAMES.index("residual_mean_b1")
    assert ring[i] > flat[i] + 1e-3


def test_ratio_features_separate_a_ring_from_a_flat_field() -> None:
    flat = seam_features(_flat())
    ring = seam_features(_with_ring())
    i = FEATURE_NAMES.index("residual_logratio_b1_b2")
    assert abs(ring[i] - flat[i]) > 1e-3


def test_rejects_an_image_that_is_not_three_channel() -> None:
    with pytest.raises(ValueError, match="HWC RGB"):
        seam_features(np.zeros((224, 224), dtype=np.uint8))


def test_works_at_a_size_other_than_224() -> None:
    f = seam_features(_with_ring(size=96))
    assert f.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(f).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_blend_features.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dfd.detectors.blend'`

- [ ] **Step 3: Write the implementation**

Create `src/dfd/detectors/blend.py`:

```python
"""Slot A — blending boundary (spec §6).

Physics: a face swap composites a generated inner face onto a real outer face.
However well the colours are matched, the two halves came from different
imaging chains, so their high-frequency residual, local sharpness and colour
statistics differ, and the difference is concentrated on a closed curve
somewhere inside the face. A camera producing a single exposure has no such
curve.

The detector does not know where the seam is. It does not need to: in an
aligned crop the inner face sits near the centre, so statistics computed over
concentric elliptical annuli will straddle the seam wherever it falls, and the
ADJACENT-BAND CONTRASTS — not the raw band values — carry the signal. That is
why the feature vector ends with log-ratios and differences rather than only
per-band means.

This occupies the slot the spec assigns to SBI, with different machinery. The
spec names a CNN trained on self-blended images; hardware here is CPU-only
(docs/HANDOFF.md §1), which rules out training EfficientNet-B4, so the
learned part is a linear model over handcrafted statistics and the
self-blending moves into corpus construction (corpora/sbi.py). The physics
read is the same; the capacity is much lower, and the honest expectation is
correspondingly lower accuracy.

IMPORTANT: the annulus boundaries and the feature set are this project's own
design, not a reimplementation of a published method, and are UNVERIFIED
against any baseline. No published claim may rest on them without measurement.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

#: Normalised elliptical radii bounding each annulus, where 1.0 is the
#: half-width of the crop. The outermost band runs past 1.0 to take in the
#: corners, which would otherwise be measured by nothing.
ANNULI: tuple[tuple[float, float], ...] = (
    (0.00, 0.45),
    (0.45, 0.65),
    (0.65, 0.85),
    (0.85, 1.45),
)

#: Gaussian kernel for the high-pass residual. Small, so the residual keeps
#: the seam's spatial frequency rather than smearing it into the whole face.
RESIDUAL_KERNEL = 5

_PER_BAND = ("residual_mean", "residual_std", "laplacian_var",
             "lab_l_mean", "lab_a_mean", "lab_b_mean")


def _feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for b in range(len(ANNULI)):
        names.extend(f"{stat}_b{b}" for stat in _PER_BAND)
    for b in range(len(ANNULI) - 1):
        names.append(f"residual_logratio_b{b}_b{b + 1}")
    for b in range(len(ANNULI) - 1):
        names.append(f"lab_l_delta_b{b}_b{b + 1}")
    return tuple(names)


FEATURE_NAMES: tuple[str, ...] = _feature_names()


def seam_features(img: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """Compute blending-boundary statistics over concentric annuli.

    Args:
        img: aligned RGB face crop, HWC uint8.

    Returns:
        float32 vector of length `len(FEATURE_NAMES)`, aligned to it
        positionally. Always finite: an empty band contributes zeros rather
        than nan, because a nan would poison the linear model silently while
        a zero is merely uninformative.

    Raises:
        ValueError: if `img` is not a three-channel HWC array.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"seam_features expects an HWC RGB image, got shape {img.shape}")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float64)
    blurred = cv2.GaussianBlur(gray, (RESIDUAL_KERNEL, RESIDUAL_KERNEL), 0)
    residual = np.abs(gray.astype(np.float64) - blurred.astype(np.float64))
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    yy, xx = np.mgrid[0:h, 0:w]
    ry = (yy - (h - 1) / 2.0) / max(1.0, (h - 1) / 2.0)
    rx = (xx - (w - 1) / 2.0) / max(1.0, (w - 1) / 2.0)
    radius = np.sqrt(rx ** 2 + ry ** 2)

    per_band: list[list[float]] = []
    for lo, hi in ANNULI:
        mask = (radius >= lo) & (radius < hi)
        if not mask.any():
            logger.debug("annulus [%.2f, %.2f) is empty at %dx%d", lo, hi, h, w)
            per_band.append([0.0] * len(_PER_BAND))
            continue
        per_band.append([
            float(residual[mask].mean()),
            float(residual[mask].std()),
            float(laplacian[mask].var()),
            float(lab[..., 0][mask].mean()),
            float(lab[..., 1][mask].mean()),
            float(lab[..., 2][mask].mean()),
        ])

    feats: list[float] = [v for band in per_band for v in band]

    # The contrast terms. log1p keeps the ratio finite when a band's residual
    # is zero, which happens on synthetic flat fields and would otherwise
    # divide by zero.
    residual_means = [band[0] for band in per_band]
    lab_l_means = [band[3] for band in per_band]
    feats.extend(float(np.log1p(residual_means[b]) - np.log1p(residual_means[b + 1]))
                 for b in range(len(ANNULI) - 1))
    feats.extend(float(lab_l_means[b] - lab_l_means[b + 1])
                 for b in range(len(ANNULI) - 1))

    return np.asarray(feats, dtype=np.float32)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_blend_features.py -q`
Expected: 8 passed

- [ ] **Step 5: Prove each test can fail**

- Return only the per-band features (drop the contrast terms) → the length assertion and `test_ratio_features_separate_a_ring_from_a_flat_field` fail.
- Replace the elliptical `radius` with a constant → `test_a_ring_raises_residual_energy_where_the_ring_is` fails.
- Remove the `mask.any()` guard and feed a 96-px image → `test_works_at_a_size_other_than_224` produces nan and `test_all_features_are_finite` fails.
- Drop the `ndim` check → `test_rejects_an_image_that_is_not_three_channel` fails.

- [ ] **Step 6: Check the gates and commit**

```bash
python3 -m pytest -q && ruff check . && mypy --strict
git add src/dfd/detectors/blend.py tests/test_blend_features.py
git commit -m "feat: concentric-annulus seam features, slot A"
```

---

### Task 5: The detector, and a model file that cannot execute code

**Files:**
- Modify: `src/dfd/detectors/blend.py` (append)
- Test: `tests/test_blend_detector.py`

**Interfaces:**
- Consumes: `seam_features` (Task 4); `dfd.detectors.base.{abstain, filter_by_quality_floor, OK, WEIGHTS_ABSENT}`.
- Produces:
  - `BlendModel` frozen dataclass: `mean: NDArray[float64]`, `scale: NDArray[float64]`, `coef: NDArray[float64]`, `intercept: float`, `feature_names: tuple[str, ...]`, `version: str`, and `predict_proba(features) -> float`.
  - `MODEL_FILE_VERSION = 1`
  - `save_blend_model(model: BlendModel, path: str | Path) -> None`
  - `load_blend_model(path: str | Path) -> BlendModel`
  - `BlendDetector` frozen dataclass implementing `Detector`, `name="blend_seam"`, `slot="A"`, `version="0.1.0"`, `min_quality_band="medium"`.
  - `DEFAULT_BLEND_WEIGHTS = Path("assets/models/blend_seam.npz")`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_blend_detector.py`:

```python
"""The detector abstains honestly, and its model file is inert data."""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from dfd.detectors.base import WEIGHTS_ABSENT
from dfd.detectors.blend import (
    FEATURE_NAMES,
    BlendDetector,
    BlendModel,
    load_blend_model,
    save_blend_model,
)
from dfd.types import Modality, Observation, Quality


def _model(coef: np.ndarray | None = None) -> BlendModel:
    n = len(FEATURE_NAMES)
    return BlendModel(
        mean=np.zeros(n), scale=np.ones(n),
        coef=np.ones(n) if coef is None else coef,
        intercept=0.0, feature_names=FEATURE_NAMES, version="test-1")


def _obs(band: str = "high", seed: int = 0) -> Observation:
    rng = np.random.default_rng(seed)
    return Observation(
        t=0.0, payload=rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        roi=None,
        quality=Quality(inter_ocular_px=40.0, blur_var=120.0, yaw_deg=0.0,
                        pitch_deg=0.0, exposure=0.5, band=band),
        source_id="s1")


def test_abstains_when_the_model_file_is_absent(tmp_path: Path) -> None:
    d = BlendDetector(weights_path=tmp_path / "nope.npz")
    r = d.score([_obs()])
    assert r.abstained and r.score is None and r.reason == WEIGHTS_ABSENT


def test_abstains_below_the_quality_floor(tmp_path: Path) -> None:
    p = tmp_path / "m.npz"
    save_blend_model(_model(), p)
    r = BlendDetector(weights_path=p).score([_obs(band="low")])
    assert r.abstained and r.reason == "below_quality_floor"


def test_scores_in_the_unit_interval(tmp_path: Path) -> None:
    p = tmp_path / "m.npz"
    save_blend_model(_model(), p)
    r = BlendDetector(weights_path=p).score([_obs()])
    assert not r.abstained
    assert r.score is not None and 0.0 <= r.score <= 1.0


def test_round_trips_through_the_file(tmp_path: Path) -> None:
    p = tmp_path / "m.npz"
    m = _model(coef=np.linspace(-1, 1, len(FEATURE_NAMES)))
    save_blend_model(m, p)
    back = load_blend_model(p)
    assert np.allclose(back.coef, m.coef)
    assert back.feature_names == m.feature_names
    assert back.version == m.version


def test_the_model_file_contains_no_pickle(tmp_path: Path) -> None:
    """The supply-chain property. An npz of plain arrays cannot execute code
    on load; a pickled sklearn estimator can, and that is why one is not used.
    """
    p = tmp_path / "m.npz"
    save_blend_model(_model(), p)
    with zipfile.ZipFile(p) as z:
        for name in z.namelist():
            head = z.read(name)[:80]
            assert b"sklearn" not in head
            assert b"__reduce__" not in head
    loaded = np.load(p, allow_pickle=False)  # must not raise
    assert "coef" in loaded


def test_a_model_whose_features_do_not_match_is_refused(tmp_path: Path) -> None:
    """A stale model file is worse than no model file: it would score
    confidently against the wrong columns."""
    p = tmp_path / "m.npz"
    save_blend_model(_model(), p)
    np.savez(p, **{**dict(np.load(p, allow_pickle=False)),
                   "feature_names": np.array(["wrong"], dtype=np.str_)})
    with pytest.raises(ValueError, match="feature names"):
        load_blend_model(p)


def test_identity_is_stable(tmp_path: Path) -> None:
    d = BlendDetector(weights_path=tmp_path / "x.npz")
    assert d.name == "blend_seam"
    assert d.slot == "A"
    assert Modality.IMAGE in d.modalities
    assert d.min_quality_band == "medium"


def test_two_observations_are_averaged_not_only_the_first(tmp_path: Path) -> None:
    p = tmp_path / "m.npz"
    save_blend_model(_model(), p)
    d = BlendDetector(weights_path=p)
    one = d.score([_obs(seed=1)])
    two = d.score([_obs(seed=1), _obs(seed=2)])
    assert one.score is not None and two.score is not None
    assert one.score != two.score
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_blend_detector.py -q`
Expected: FAIL — `ImportError: cannot import name 'BlendModel'`

- [ ] **Step 3: Write the implementation**

Append to `src/dfd/detectors/blend.py` (add `from collections.abc import Sequence`, `from dataclasses import dataclass, field`, `from pathlib import Path`, `from typing import Literal`, and the `..types` / `.base` imports at the top):

```python
#: On-disk format version for the model file. Bump when the array set changes.
MODEL_FILE_VERSION = 1

#: Where a deployment is expected to place the fitted model. Gitignored and
#: absent in this repo, so the detector abstains on every fresh checkout —
#: the same contract NPR and EffNet already keep.
DEFAULT_BLEND_WEIGHTS = Path("assets/models/blend_seam.npz")


@dataclass(frozen=True)
class BlendModel:
    """A standardiser and a linear model, as plain arrays.

    Deliberately not a pickled sklearn estimator. `joblib.load` and
    `pickle.load` execute arbitrary code from the file they read, and a
    detector's weight file is exactly the artefact an attacker would swap.
    scikit-learn fits this model (see training/fit_blend.py) and is then
    discarded: only the numbers are kept, and `np.load(..., allow_pickle=False)`
    cannot execute anything.
    """
    mean: npt.NDArray[np.float64]
    scale: npt.NDArray[np.float64]
    coef: npt.NDArray[np.float64]
    intercept: float
    feature_names: tuple[str, ...]
    version: str

    def predict_proba(self, features: npt.NDArray[np.float32]) -> float:
        """P(fake) for one feature vector.

        Args:
            features: vector aligned to `feature_names`.

        Returns:
            A probability in [0, 1].
        """
        z = (features.astype(np.float64) - self.mean) / np.where(
            self.scale == 0.0, 1.0, self.scale)
        logit = float(np.dot(z, self.coef) + self.intercept)
        return float(1.0 / (1.0 + np.exp(-logit)))


def save_blend_model(model: BlendModel, path: str | Path) -> None:
    """Write the model as an npz of plain arrays."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        p,
        format_version=np.array(MODEL_FILE_VERSION),
        mean=model.mean, scale=model.scale, coef=model.coef,
        intercept=np.array(model.intercept),
        feature_names=np.array(model.feature_names, dtype=np.str_),
        version=np.array(model.version, dtype=np.str_),
    )
    logger.info("wrote blend model v%s to %s", model.version, p)


def load_blend_model(path: str | Path) -> BlendModel:
    """Read a model file, refusing one that does not match this code.

    Args:
        path: npz written by `save_blend_model`.

    Returns:
        The model.

    Raises:
        ValueError: if the file's format version or feature names disagree
            with this build. A stale model file is worse than none: it would
            score confidently against columns that no longer mean what it
            was fitted on.
    """
    data = np.load(Path(path), allow_pickle=False)
    version = int(data["format_version"])
    if version != MODEL_FILE_VERSION:
        raise ValueError(
            f"blend model format version {version} != {MODEL_FILE_VERSION}")
    names = tuple(str(n) for n in data["feature_names"])
    if names != FEATURE_NAMES:
        raise ValueError(
            "blend model feature names do not match this build: "
            f"file has {len(names)}, code expects {len(FEATURE_NAMES)}")
    return BlendModel(
        mean=np.asarray(data["mean"], dtype=np.float64),
        scale=np.asarray(data["scale"], dtype=np.float64),
        coef=np.asarray(data["coef"], dtype=np.float64),
        intercept=float(data["intercept"]),
        feature_names=names,
        version=str(data["version"]),
    )


@dataclass(frozen=True)
class BlendDetector:
    """Slot A. Scores the blending seam with a linear model over seam features.

    Frozen, like every other detector, so the registry's name→detector
    invariant cannot be broken by mutating identity after registration.
    """
    weights_path: str | Path = DEFAULT_BLEND_WEIGHTS

    name: str = "blend_seam"
    slot: str = "A"
    version: str = "0.1.0"
    modalities: frozenset[Modality] = field(
        default_factory=lambda: frozenset({Modality.IMAGE, Modality.VIDEO}))
    min_quality_band: Literal["low", "medium", "high"] = "medium"

    def score(self, obs: Sequence[Observation]) -> RawScore:
        """Score observations by their mean seam probability.

        Args:
            obs: observations to score.

        Returns:
            A RawScore. Abstains with `weights_absent` when the model file is
            missing, or with the quality-floor reason when nothing is usable.
        """
        usable, reason = filter_by_quality_floor(obs, self.min_quality_band)
        if reason is not None:
            return abstain(self.name, self.version, reason)

        path = Path(self.weights_path)
        if not path.exists():
            logger.warning("blend model absent at %s", path)
            return abstain(self.name, self.version, WEIGHTS_ABSENT)

        model = load_blend_model(path)
        probs = [model.predict_proba(seam_features(o.payload)) for o in usable]
        score = float(np.mean(probs))
        return RawScore(detector=self.name, version=self.version, score=score,
                        abstained=False, reason=OK,
                        artifacts={"n_observations": len(probs),
                                   "model_version": model.version})
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_blend_detector.py -q`
Expected: 8 passed

- [ ] **Step 5: Prove each test can fail**

- Drop the `names != FEATURE_NAMES` check → `test_a_model_whose_features_do_not_match_is_refused` fails.
- Pass `allow_pickle=True` and save a pickled object instead → `test_the_model_file_contains_no_pickle` fails.
- Score `usable[0]` only → `test_two_observations_are_averaged_not_only_the_first` fails.
- Remove the `path.exists()` branch → `test_abstains_when_the_model_file_is_absent` raises `FileNotFoundError` instead of abstaining.

- [ ] **Step 6: Check the gates and commit**

```bash
python3 -m pytest -q && ruff check . && mypy --strict
git add src/dfd/detectors/blend.py tests/test_blend_detector.py
git commit -m "feat: blend-seam detector with an inert model file"
```

---

### Task 6: Fit the model, with the split discipline in the fitter

**Files:**
- Create: `training/__init__.py`, `training/fit_blend.py`
- Test: `tests/test_fit_blend.py`

**Interfaces:**
- Consumes: `build_face_pool`, `build_sbi_corpus`, `seam_features`, `BlendModel`, `save_blend_model`.
- Produces:
  - `split_by_subject(samples: Sequence[Sample], *, holdout_fraction: float = 0.3, seed: int = 0) -> tuple[list[Sample], list[Sample]]`
  - `fit_blend_model(train: Sequence[Sample], *, version: str) -> BlendModel`
  - `evaluate(model: BlendModel, samples: Sequence[Sample]) -> dict[str, float]` returning `{"auc": ..., "n": ..., "n_fake": ...}`
  - `main(argv: Sequence[str] | None = None) -> int`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fit_blend.py`:

```python
"""The fitter must not leak a subject across the split, and must say so."""
from __future__ import annotations

import numpy as np
import pytest

from corpora.face_pool import FaceCrop
from corpora.sbi import build_sbi_corpus
from dfd.faces import FaceBox
from dfd.types import Quality
from training.fit_blend import evaluate, fit_blend_model, split_by_subject


def _crop(session_id: str) -> FaceCrop:
    lms = np.array([[70.0, 80.0], [110.0, 80.0], [90.0, 100.0],
                    [75.0, 125.0], [105.0, 125.0]])
    rng = np.random.default_rng(len(session_id) * 1000)
    return FaceCrop(session_id=session_id, frame_index=0,
                    image=rng.integers(40, 210, (224, 224, 3), dtype=np.uint8),
                    box=FaceBox(x=55, y=55, w=70, h=90, landmarks=lms, score=0.9),
                    quality=Quality(inter_ocular_px=40.0, blur_var=120.0,
                                    yaw_deg=0.0, pitch_deg=0.0, exposure=0.5,
                                    band="high"),
                    swapped=False)


def _corpus(n: int = 20):
    return build_sbi_corpus([_crop(f"s{i}") for i in range(n)])


def test_no_subject_appears_on_both_sides_of_the_split() -> None:
    train, test = split_by_subject(_corpus(), seed=0)
    assert {s.context.subject_id for s in train} & {s.context.subject_id for s in test} == set()


def test_both_sides_carry_both_labels() -> None:
    train, test = split_by_subject(_corpus(), seed=0)
    for side in (train, test):
        assert {s.context.label for s in side} == {0, 1}


def test_the_split_is_reproducible() -> None:
    a, _ = split_by_subject(_corpus(), seed=3)
    b, _ = split_by_subject(_corpus(), seed=3)
    assert [s.sample_id for s in a] == [s.sample_id for s in b]


def test_a_fitted_model_matches_the_current_feature_set() -> None:
    from dfd.detectors.blend import FEATURE_NAMES
    train, _ = split_by_subject(_corpus(), seed=0)
    m = fit_blend_model(train, version="t1")
    assert m.feature_names == FEATURE_NAMES
    assert m.coef.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(m.coef).all()
    assert np.isfinite(m.mean).all() and np.isfinite(m.scale).all()


def test_scale_is_never_zero_so_standardising_cannot_divide_by_zero() -> None:
    train, _ = split_by_subject(_corpus(), seed=0)
    m = fit_blend_model(train, version="t1")
    assert (m.scale != 0).all()


def test_fitting_refuses_a_single_label_corpus() -> None:
    samples = [s for s in _corpus() if s.context.label == 0]
    with pytest.raises(ValueError, match="both labels"):
        fit_blend_model(samples, version="t1")


def test_evaluate_reports_the_fake_count_not_only_the_total() -> None:
    """A headline AUC over 4 fakes is not the same claim as one over 400,
    and the report must make that visible."""
    train, test = split_by_subject(_corpus(), seed=0)
    m = fit_blend_model(train, version="t1")
    out = evaluate(m, test)
    assert set(out) >= {"auc", "n", "n_fake"}
    assert out["n_fake"] == sum(1 for s in test if s.context.label == 1)
    assert 0.0 <= out["auc"] <= 1.0


def test_the_model_separates_self_blends_it_was_trained_on() -> None:
    """A weak but non-vacuous bar: better than chance in-distribution.
    If this fails the features carry no seam signal at all.

    Deliberately weak. The same pipeline measured on synthetic textured
    fixtures scored a held-out AUC of 1.000, and that number means nothing
    about real faces: the fixtures differ from their self-blends in ways a
    camera never would. Raising this bar to match would be asserting a
    property of the fixture generator, not of the detector.
    """
    train, test = split_by_subject(_corpus(n=40), seed=0)
    m = fit_blend_model(train, version="t1")
    assert evaluate(m, test)["auc"] > 0.5
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_fit_blend.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'training'`

- [ ] **Step 3: Write the implementation**

Create `training/__init__.py` (empty) and `training/fit_blend.py`:

```python
"""Fit the blend-seam model. Not part of the installed package: fitting is
not inference, and nothing in src/dfd/ may depend on this module.

Run:
    python3 -m training.fit_blend \\
        --captures /path/to/captures \\
        --out assets/models/blend_seam.npz
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from corpora.captures import load_capture_sessions
from corpora.face_pool import build_face_pool
from corpora.sbi import build_sbi_corpus
from dfd.detectors.blend import FEATURE_NAMES, BlendModel, save_blend_model, seam_features
from dfd.types import Sample

logger = logging.getLogger(__name__)

#: Fraction of SUBJECTS held out. Subjects, never rows: a real crop and the
#: pseudo-fake made from it share a face, so splitting by row would put the
#: same face on both sides and the reported AUC would be meaningless.
DEFAULT_HOLDOUT = 0.3


def split_by_subject(
    samples: Sequence[Sample],
    *,
    holdout_fraction: float = DEFAULT_HOLDOUT,
    seed: int = 0,
) -> tuple[list[Sample], list[Sample]]:
    """Split samples into (train, test) with no subject on both sides.

    Args:
        samples: the corpus.
        holdout_fraction: share of subjects placed in test.
        seed: reproducibility.

    Returns:
        A (train, test) pair.
    """
    subjects = sorted({s.context.subject_id for s in samples
                       if s.context.subject_id is not None})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(subjects))
    n_test = max(1, int(round(len(subjects) * holdout_fraction)))
    test_subjects = {subjects[i] for i in order[:n_test]}

    train = [s for s in samples if s.context.subject_id not in test_subjects]
    test = [s for s in samples if s.context.subject_id in test_subjects]
    logger.info("split: %d train / %d test rows over %d subjects",
                len(train), len(test), len(subjects))
    return train, test


def _matrix(samples: Sequence[Sample]) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([seam_features(s.observations[0].payload) for s in samples])
    y = np.asarray([s.context.label for s in samples], dtype=np.int64)
    return x, y


def fit_blend_model(train: Sequence[Sample], *, version: str) -> BlendModel:
    """Fit the standardiser and logistic model, and keep only the numbers.

    Args:
        train: training samples. Must contain both labels.
        version: version string recorded in the model file.

    Returns:
        The fitted model.

    Raises:
        ValueError: if `train` does not contain both labels — a one-class fit
            produces a model that scores everything the same and reports no
            error at all.
    """
    x, y = _matrix(train)
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("training corpus must contain both labels, got "
                         f"{sorted(set(y.tolist()))}")

    mean = x.mean(axis=0).astype(np.float64)
    scale = x.std(axis=0).astype(np.float64)
    # A constant feature has zero spread. Dividing by it yields inf; replacing
    # the divisor with 1.0 leaves the feature at zero, which is exactly as
    # informative as it actually is.
    scale = np.where(scale == 0.0, 1.0, scale)

    z = (x.astype(np.float64) - mean) / scale
    clf = LogisticRegression(max_iter=1000)
    clf.fit(z, y)

    return BlendModel(mean=mean, scale=scale,
                      coef=np.asarray(clf.coef_[0], dtype=np.float64),
                      intercept=float(clf.intercept_[0]),
                      feature_names=FEATURE_NAMES, version=version)


def evaluate(model: BlendModel, samples: Sequence[Sample]) -> dict[str, float]:
    """Score a held-out set.

    Returns a dict carrying `auc`, `n` and `n_fake`. The fake count is
    reported beside the AUC deliberately: an AUC over a handful of positives
    is a different claim from the same number over hundreds, and a report
    that hides the denominator invites the reader to confuse them.
    """
    from sklearn.metrics import roc_auc_score

    if not samples:
        return {"auc": float("nan"), "n": 0.0, "n_fake": 0.0}
    x, y = _matrix(samples)
    probs = np.asarray([model.predict_proba(row.astype(np.float32)) for row in x])
    n_fake = float((y == 1).sum())
    auc = (float(roc_auc_score(y, probs))
           if len(set(y.tolist())) == 2 else float("nan"))
    return {"auc": auc, "n": float(len(samples)), "n_fake": n_fake}


def main(argv: Sequence[str] | None = None) -> int:
    """Build the corpus, fit, evaluate, write the model and a JSON report."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", required=True, type=Path)
    parser.add_argument("--out", default=Path("assets/models/blend_seam.npz"),
                        type=Path)
    parser.add_argument("--report", default=Path("bench/blend_seam_report.json"),
                        type=Path)
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    sessions = load_capture_sessions(args.captures)
    genuine = [s for s in sessions if not s.swapped]
    logger.info("%d sessions, %d genuine and usable for training",
                len(sessions), len(genuine))

    crops, skipped = build_face_pool(genuine)
    if not crops:
        # Name both causes. The skip tally distinguishes them — NO_FACE means
        # the detector ran and found nothing (or has no weights), NO_FRAMES
        # means the folders held no frame_NN.jpg at all — and a message that
        # guesses one cause sends the reader past the tally that answers it.
        logger.error("no face crops extracted; skip tally: %s. NO_FACE means "
                     "the detector returned nothing (check the YuNet weights at "
                     "assets/models/face_detection_yunet_2023mar.onnx); "
                     "NO_FRAMES means the session folders held no frames.",
                     skipped)
        return 1

    samples = build_sbi_corpus(crops, seed=args.seed)
    train, test = split_by_subject(samples, seed=args.seed)
    model = fit_blend_model(train, version=args.version)
    metrics = evaluate(model, test)

    save_blend_model(model, args.out)
    report = {"version": args.version, "seed": args.seed,
              "n_sessions": len(sessions), "n_genuine": len(genuine),
              "n_crops": len(crops), "skipped": skipped,
              "n_train": len(train), "n_test": len(test), **metrics}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True))
    logger.info("held-out AUC %.3f over %d rows (%d fake)",
                metrics["auc"], int(metrics["n"]), int(metrics["n_fake"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_fit_blend.py -q`
Expected: 8 passed

- [ ] **Step 5: Prove each test can fail**

- Split on `sample_id` instead of `subject_id` → `test_no_subject_appears_on_both_sides_of_the_split` fails.
- Drop the `set(np.unique(y)) != {0, 1}` guard → `test_fitting_refuses_a_single_label_corpus` fails.
- Remove the `np.where(scale == 0.0, 1.0, scale)` replacement and add a constant feature → `test_scale_is_never_zero_so_standardising_cannot_divide_by_zero` fails.
- Return only `auc` from `evaluate` → `test_evaluate_reports_the_fake_count_not_only_the_total` fails.

- [ ] **Step 6: Check the gates and commit**

```bash
python3 -m pytest -q && ruff check . && mypy --strict && mypy --strict --follow-imports=skip training/fit_blend.py
git add training tests/test_fit_blend.py
git commit -m "feat: fit the blend model, with subject-disjoint splitting in the fitter"
```

---

### Task 7: Wire it in — registry, manifest, gitignore, docs

**Files:**
- Modify: `src/dfd/detectors/registry.py`
- Modify: `assets/manifest.yaml`
- Modify: `.gitignore`
- Modify: `docs/HANDOFF.md` (§0 resume block)
- Test: `tests/test_registry.py` (extend), `tests/test_manifest.py` (already covers schema)

**Interfaces:**
- Consumes: `BlendDetector`, `DEFAULT_BLEND_WEIGHTS`.
- Produces: `default_registry` gains a third detector, `blend_seam`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_registry.py`:

```python
def test_default_registry_composes_three_distinct_physics() -> None:
    """Spec §6: the portfolio's value is uncorrelated evidence, so the
    default set must not be three views of the same artifact."""
    from dfd.detectors.registry import default_registry

    registry = default_registry()
    assert registry.names() == ["blend_seam", "effnet_b4", "npr"]
    slots = {registry.get(n).slot for n in registry.names()}  # type: ignore[attr-defined]
    assert slots == {"A", "C", "E"}


def test_blend_seam_abstains_when_its_model_file_is_absent(tmp_path) -> None:
    from dfd.detectors.registry import default_registry
    from dfd.types import Observation, Quality

    registry = default_registry(blend_weights=tmp_path / "absent.npz")
    obs = Observation(
        t=0.0, payload=__import__("numpy").zeros((224, 224, 3), dtype="uint8"),
        roi=None,
        quality=Quality(inter_ocular_px=40.0, blur_var=120.0, yaw_deg=0.0,
                        pitch_deg=0.0, exposure=0.5, band="high"),
        source_id="s")
    result = registry.get("blend_seam").score([obs])
    assert result.abstained and result.reason == "weights_absent"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_registry.py -q`
Expected: FAIL — `default_registry()` returns two names, and `default_registry(blend_weights=...)` is an unexpected keyword.

- [ ] **Step 3: Update the registry**

In `src/dfd/detectors/registry.py`, add the import and the parameter, and update the docstring:

```python
from .blend import DEFAULT_BLEND_WEIGHTS, BlendDetector


def default_registry(npr_weights: str | Path = DEFAULT_NPR_WEIGHTS,
                     effnet_weights: str | Path = DEFAULT_EFFNET_WEIGHTS,
                     blend_weights: str | Path = DEFAULT_BLEND_WEIGHTS) -> Registry:
    """The detector set the CLI consults.

    Three distinct physics, per spec §6: NPR's upsampling fingerprint (slot C),
    the blending seam (slot A), and EfficientNet-B4's learned appearance
    (slot E). All three abstain when their weights are absent, which is every
    fresh checkout of this repo — `blend_seam` is the first of the three whose
    weights this project can actually produce, because it is fitted on a corpus
    manufactured from the project's own captures (corpora/sbi.py) rather than
    on a licensed dataset.

    Raises:
        ValueError: from `Registry.register` if a detector with the same name
            is already registered. Unreachable today: three distinct hardcoded
            names ("npr", "blend_seam", "effnet_b4") cannot collide.
    """
    registry = Registry()
    registry.register(NPRDetector(weights_path=npr_weights))
    registry.register(BlendDetector(weights_path=blend_weights))
    registry.register(EffNetDetector(name="effnet_b4", slot="E",
                                     weights_path=effnet_weights))
    return registry
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_registry.py tests/test_pipeline.py tests/test_cli.py -q`
Expected: all pass. If a pipeline or CLI test asserted a two-detector audit record, update the expectation — the third detector is a deliberate change, and the test should assert three.

- [ ] **Step 5: Register the asset and ignore the artefact**

In `assets/manifest.yaml`, add under `assets:`:

```yaml
  blend_seam_weights:
    source: "Fitted in this repo by training/fit_blend.py from self-blended crops of the project's own capture corpus. No external dataset or generator weight file contributes to it."
    license: "owned — no third-party dataset or model contributed to these coefficients"
    commercial_use: true
    evidence_url: "docs/EULA-ACCESS.md"
    date_checked: "2026-09-21"
    checked_by: "kohrohit@gmail.com"
    files:
      - "assets/models/blend_seam.npz"
```

In `.gitignore`, add:

```
assets/models/blend_seam.npz
bench/blend_seam_report.json
```

Run: `python3 -m pytest tests/test_manifest.py tests/test_asset_scan.py -q`
Expected: pass. The asset gate must accept the new entry; if `files` points at a path that does not exist yet, confirm that is the same shape the gate already tolerates for `npr_weights` (which has no `files` key at all) — if it is not, omit `files` until the model file is actually produced, and say so in the manifest comment.

- [ ] **Step 6: Update the handoff**

In `docs/HANDOFF.md` §0, replace next-step 4 ("A swap corpus…") with a statement of what now exists: the corpus builder, the detector, the fitter, the manifest entry, and — measured, not assumed — the held-out AUC from `bench/blend_seam_report.json` together with its `n_fake`. If the fitter has not been run against the real corpus yet, say exactly that instead of quoting a number.

- [ ] **Step 7: Check every gate and commit**

```bash
python3 -m pytest -q
ruff check .
mypy --strict && mypy --strict --follow-imports=skip training/fit_blend.py
python3 -m pytest --cov=src --cov=corpora --cov=bench --cov-fail-under=85 -q
git add -A
git commit -m "feat: register the blend-seam detector and its owned model asset"
```

---

## What this plan does not do, deliberately

Recorded here so the next reader does not mistake absence for oversight.

1. **It does not fit the model as part of the test suite.** `training/fit_blend.py` runs against the capture corpus on the owner's machine; CI has neither the corpus nor the YuNet weights. The tests fit on synthetic crops instead, which proves the fitter's discipline but says nothing about the detector's accuracy. **The accuracy number only exists once step 6 of Task 6 is run for real.**
2. **It leaves LOGO with one generator.** A self-blend-only corpus has exactly one generator name, so leave-one-generator-out produces a single fold and cannot measure cross-generator transfer — the headline number spec §8.1 asks for. This is the gap the research datasets or a second clean generator family (classical landmark swap with Poisson blending) would close. It is out of scope here and should be its own plan.
3. **It does not touch calibration.** `Calibrator.to_evidence` still returns `uncalibrated_for_band` for every band, so `dfd score` will keep returning `insufficient_evidence` even once the model file exists. Fitting the calibration curve is the next milestone, and it is where the known `_worst_band` defect recorded in the P0 plan's known-gaps block must be settled first.
4. **The only accuracy evidence so far is on synthetic fixtures.** Every reference implementation in this plan was executed before the plan was handed over (the rule in `docs/HANDOFF.md` §6), and all assertions hold — including a held-out AUC of 1.000 separating textured fixtures from their self-blends. **That number is not a detector claim.** Synthetic fixtures differ from their blends in ways a real camera never produces. The first honest accuracy figure arrives only from Task 6 step 6 run against the real capture corpus.
5. **It does not verify the seam-feature design against any baseline.** The annulus geometry and feature set are this project's own and unmeasured. `docs/HANDOFF.md` §6's rule applies: the constants are a hypothesis until something measures them.
6. **It does not evaluate on the 5 swapped-and-approved sessions.** Building that evaluation — the only real fraud this project holds — needs the calibration and reporting path, and deserves its own plan rather than a step at the end of this one.
