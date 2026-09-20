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
    """Align must actually crop and resize, not just return zeros."""
    # Create a frame filled with 0, with a distinctive bright patch in the box region.
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:300, 100:300] = 255  # Bright patch in the box region

    box = FaceBox(x=100, y=100, w=200, h=200,
                  landmarks=np.array([[150.0, 160.0], [250.0, 160.0]]),
                  score=0.99)
    out = align(frame, box, size=224)

    assert out.shape == (224, 224, 3)
    assert out.dtype == np.uint8
    # The output should have high mean value (bright), not be all zeros.
    # INTER_AREA interpolation will smooth the bright patch but keep it bright.
    assert out.mean() > 200


def test_align_clamps_box_to_frame_bounds():
    """A box running off the edge must not raise or produce a wrong-sized crop."""
    # Create a frame with bright pixels only in the valid region.
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[0:100, 0:100] = 255  # Entire frame is bright

    box = FaceBox(x=80, y=80, w=200, h=200,
                  landmarks=np.array([[90.0, 90.0], [95.0, 90.0]]),
                  score=0.9)
    out = align(frame, box, size=64)

    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8
    # Even though the box extends beyond the frame, the clamped region is bright.
    assert out.mean() > 200


def test_landmarks_array_is_read_only():
    """Mutating landmarks after detection must raise ValueError."""
    frame = _frame()
    box = FaceBox(x=100, y=100, w=200, h=200,
                  landmarks=np.array([[150.0, 160.0], [250.0, 160.0]]),
                  score=0.99)
    # Make the landmarks read-only like detect_faces does
    box.landmarks.setflags(write=False)

    # Attempting to mutate should raise ValueError
    with pytest.raises(ValueError):
        box.landmarks[0, 0] = 999.0


def test_real_manifest_parses_and_is_commercially_clean():
    """The shipped manifest, not a fixture. An unparseable manifest blocks release."""
    from pathlib import Path
    from dfd.manifest import assert_release_clean, load_manifest

    root = Path(__file__).resolve().parents[1]
    m = load_manifest(root / "assets" / "manifest.yaml")
    assert "yunet_face_detector" in m
    assert m["yunet_face_detector"].commercial_use is True
    assert_release_clean(m, ["yunet_face_detector"])
