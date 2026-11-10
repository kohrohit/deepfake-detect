import numpy as np
import pytest

from dfd.faces import WEIGHTS_ABSENT, FaceBox, align, detect_faces, select_largest_face


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


def test_landmarks_are_read_only_on_construction():
    """A FaceBox is evidence. Nothing may mutate it after the fact.

    The test must NOT set the flag itself — construction is what enforces it.
    """
    lms = np.array([[150.0, 160.0], [250.0, 160.0]])
    box = FaceBox(x=100, y=100, w=200, h=200, landmarks=lms, score=0.99)
    # Attempting to mutate should raise ValueError because __post_init__ froze it
    with pytest.raises(ValueError):
        box.landmarks[0, 0] = 999.0


def test_select_largest_face_returns_none_for_no_faces():
    assert select_largest_face([]) is None


def test_select_largest_face_returns_the_one_box_given():
    box = FaceBox(x=10, y=10, w=20, h=20,
                  landmarks=np.array([[15.0, 18.0], [25.0, 18.0]]), score=0.5)
    assert select_largest_face([box]) is box


def test_select_largest_face_picks_area_over_confidence():
    """The exact confusion behind the 2026-09-24 defect, pinned at the source.

    `dfd.pipeline` scores the largest box, not the most confident one. A
    fitter or a corpus loader that picked by `score` instead would fit on
    one face and score another whenever a frame holds two faces and the two
    rules disagree -- which is precisely what happened in
    `training/fit_vcip.py` before this rule was centralised here.
    """
    small_but_confident = FaceBox(
        x=300, y=300, w=50, h=50,
        landmarks=np.array([[310.0, 315.0], [340.0, 315.0]]), score=0.99)
    big_but_less_confident = FaceBox(
        x=10, y=10, w=200, h=200,
        landmarks=np.array([[60.0, 70.0], [160.0, 70.0]]), score=0.70)
    chosen = select_largest_face([small_but_confident, big_but_less_confident])
    assert chosen is big_but_less_confident


def test_real_manifest_parses_and_is_commercially_clean():
    """The shipped manifest, not a fixture. An unparseable manifest blocks release."""
    from pathlib import Path
    from dfd.manifest import assert_release_clean, load_manifest

    root = Path(__file__).resolve().parents[1]
    m = load_manifest(root / "assets" / "manifest.yaml")
    assert "yunet_face_detector" in m
    assert m["yunet_face_detector"].commercial_use is True
    assert_release_clean(m, ["yunet_face_detector"])
