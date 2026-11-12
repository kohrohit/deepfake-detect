"""Tests for the identity embedder.

Hermetic by default: CI has no weight file, so every behavioural test drives a
stub recogniser through `Embedder` rather than the real ONNX. The two tests
that need the real model are skipped when it is absent and are the only place
the 38 MB file is touched.
"""
import cv2
import numpy as np
import pytest
from dfd.embed import (
    DEFAULT_MODEL,
    DEFAULT_THRESHOLD,
    EMBEDDING_DIM,
    OK,
    WEIGHTS_ABSENT,
    Embedder,
    _detection_row,
    cosine,
    load_embedder,
)
from dfd.faces import FaceBox, detect_faces

needs_weights = pytest.mark.skipif(
    not DEFAULT_MODEL.exists(),
    reason=f"SFace weights absent at {DEFAULT_MODEL} (gitignored, absent in CI)")


def _box(x=10, y=20, w=100, h=120, score=0.9) -> FaceBox:
    lms = np.array([[30.0, 50.0], [80.0, 50.0], [55.0, 75.0],
                    [35.0, 100.0], [75.0, 100.0]])
    return FaceBox(x=x, y=y, w=w, h=h, landmarks=lms, score=score)


def _frame(h=200, w=200) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


class _Stub:
    """A recogniser that returns what the test tells it to."""

    def __init__(self, feature, aligned=None, raise_align=False):
        self._feature, self._aligned, self._raise = feature, aligned, raise_align
        self.seen_rows = []

    def alignCrop(self, img, row):  # noqa: N802 — cv2's name
        self.seen_rows.append(np.asarray(row).copy())
        if self._raise:
            raise cv2.error("alignCrop failed")
        return np.zeros((112, 112, 3), np.uint8) if self._aligned is None else self._aligned

    def feature(self, aligned):
        return self._feature


def test_load_embedder_reports_absent_weights_rather_than_raising(tmp_path):
    """CI has no weight file. Absence is a reason, never an exception."""
    emb, reason = load_embedder(tmp_path / "missing.onnx")
    assert emb is None
    assert reason == WEIGHTS_ABSENT


def test_detection_row_carries_yunet_field_order():
    """alignCrop reads position, then ten landmark values, then the score.

    A row assembled in any other order aligns to the wrong points and still
    returns a plausible-looking embedding, so the order is asserted rather
    than assumed.
    """
    row = _detection_row(_box(x=10, y=20, w=100, h=120, score=0.9))
    assert row.shape == (15,)
    assert list(row[:4]) == [10.0, 20.0, 100.0, 120.0]
    assert list(row[4:14]) == [30.0, 50.0, 80.0, 50.0, 55.0, 75.0,
                               35.0, 100.0, 75.0, 100.0]
    assert row[14] == pytest.approx(0.9)


def test_embed_returns_a_unit_vector():
    """Callers take dot products as cosines; that is only true if normalised."""
    raw = np.arange(1, EMBEDDING_DIM + 1, dtype=np.float32).reshape(1, -1)
    emb = Embedder(model_path=DEFAULT_MODEL, _net=_Stub(raw))

    v = emb.embed(_frame(), _box())

    assert v is not None
    assert v.shape == (EMBEDDING_DIM,)
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-6)
    # Direction preserved, not just magnitude: a stub that returned zeros
    # would also be "normalised" if this only checked the norm.
    assert np.dot(v, raw.reshape(-1) / np.linalg.norm(raw)) == pytest.approx(1.0, abs=1e-6)


def test_embed_refuses_a_checkpoint_of_the_wrong_width():
    """A different model at the same path makes similarities incomparable."""
    emb = Embedder(model_path=DEFAULT_MODEL,
                   _net=_Stub(np.ones((1, EMBEDDING_DIM + 384), np.float32)))

    with pytest.raises(ValueError, match=f"{EMBEDDING_DIM}-d embedding"):
        emb.embed(_frame(), _box())


def test_embed_refuses_a_zero_vector_rather_than_returning_it():
    """A zero vector has no direction; cosine() would read it as 'dissimilar'."""
    emb = Embedder(model_path=DEFAULT_MODEL,
                   _net=_Stub(np.zeros((1, EMBEDDING_DIM), np.float32)))

    assert emb.embed(_frame(), _box()) is None


def test_embed_returns_none_when_the_crop_cannot_be_aligned():
    emb = Embedder(model_path=DEFAULT_MODEL,
                   _net=_Stub(np.ones((1, EMBEDDING_DIM), np.float32), raise_align=True))

    assert emb.embed(_frame(), _box()) is None


def test_embed_converts_rgb_to_bgr_before_the_recogniser_sees_it():
    """Callers hold RGB (dfd.faces' convention); OpenCV expects BGR.

    Passing RGB straight through would not raise — it would embed a
    colour-swapped face — so the conversion is asserted on the pixels the
    recogniser actually receives.
    """
    seen = {}

    class _Watcher(_Stub):
        def alignCrop(self, img, row):  # noqa: N802
            seen["img"] = np.asarray(img).copy()
            return np.zeros((112, 112, 3), np.uint8)

    frame = np.zeros((32, 32, 3), np.uint8)
    frame[..., 0] = 255  # pure RED in RGB
    emb = Embedder(model_path=DEFAULT_MODEL,
                   _net=_Watcher(np.ones((1, EMBEDDING_DIM), np.float32)))

    emb.embed(frame, _box(x=0, y=0, w=32, h=32))

    # Red in RGB must arrive in the BLUE channel of a BGR array.
    assert seen["img"][..., 2].mean() == 255
    assert seen["img"][..., 0].mean() == 0


def test_cosine_is_one_for_identical_and_zero_for_orthogonal():
    a = np.array([1.0, 0.0, 0.0], np.float32)
    b = np.array([0.0, 1.0, 0.0], np.float32)
    assert cosine(a, a) == pytest.approx(1.0)
    assert cosine(a, b) == pytest.approx(0.0)
    assert cosine(a, -a) == pytest.approx(-1.0)


def test_cosine_normalises_rather_than_assuming_unit_vectors():
    """A caller passing an un-normalised vector must not get a scaled answer."""
    a = np.array([3.0, 4.0], np.float32)
    assert cosine(a, a * 10.0) == pytest.approx(1.0)


def test_cosine_is_zero_when_a_vector_has_no_direction():
    a = np.array([0.0, 0.0], np.float32)
    assert cosine(a, np.array([1.0, 0.0], np.float32)) == 0.0


@needs_weights
def test_the_real_model_loads_and_reports_ok():
    emb, reason = load_embedder()
    assert emb is not None and reason == OK
    assert emb.model_path == DEFAULT_MODEL


@needs_weights
def test_the_real_model_separates_a_face_from_its_own_jittered_copy():
    """The property the guard depends on, on real weights and a real face.

    A photograph and a jittered copy of ITSELF must score far above
    `DEFAULT_THRESHOLD`; the same photograph against a different face must
    score below it. Without both halves this passes for a model that returns
    a constant.
    """
    from pathlib import Path

    from corpora.sbi import jitter

    root = Path.home() / "Desktop/agents/datasets/fairface_sessions"
    if not root.is_dir():
        pytest.skip("FairFace sessions corpus not on this machine")
    emb, _ = load_embedder()
    assert emb is not None

    vecs = []
    for sid in sorted(d.name for d in root.iterdir() if d.is_dir())[:12]:
        bgr = cv2.imread(str(root / sid / "frame_00.jpg"))
        if bgr is None:
            continue
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        boxes = detect_faces(frame)
        if not boxes:
            continue
        box = max(boxes, key=lambda b: b.score)
        v = emb.embed(frame, box)
        j = jitter(frame, np.random.default_rng(0))
        jb = detect_faces(j)
        w = emb.embed(j, max(jb, key=lambda b: b.score)) if jb else None
        if v is not None and w is not None:
            vecs.append((v, w))
    if len(vecs) < 2:
        pytest.skip("no usable face pairs on this machine")

    same = [cosine(v, w) for v, w in vecs]
    different = [cosine(vecs[i][0], vecs[j][0])
                 for i in range(len(vecs)) for j in range(i + 1, len(vecs))]

    assert min(same) > DEFAULT_THRESHOLD, f"same-person pairs fell to {min(same):.3f}"
    assert max(different) < min(same), (
        f"a different-person pair ({max(different):.3f}) scored at or above the "
        f"weakest same-person pair ({min(same):.3f})")
