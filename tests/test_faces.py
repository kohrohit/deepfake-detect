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
