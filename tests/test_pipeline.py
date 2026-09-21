from dataclasses import dataclass

import numpy as np
import pytest

from dfd.faces import FaceBox
from dfd.pipeline import (DEGENERATE_BOX, NO_FACE, NO_OBSERVATIONS,
                          UNMEASURED, _worst_band, normalize)
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


def test_a_box_with_a_negative_origin_is_clamped_to_the_frame():
    """The exact defect this task exists to prevent: `measure_quality` slices
    frame[y:y+h, x:x+w] with no clamp of its own, so a negative x or y must
    never reach it. Every other fixture in this file uses a non-negative
    origin; this is the one that exercises `max(0, box.x)` / `max(0, box.y)`.
    """
    s, reasons = normalize(_sample(),
                           detect=_detector([_box(x=-20, y=-10, w=300, h=200)]))
    assert reasons["faces"] == "ok"
    roi = s.observations[0].roi
    assert roi == (0, 0, 256, 190)
    x, y, w, h = roi
    assert x >= 0 and y >= 0
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


def test_a_sample_with_no_observations_reports_no_observations():
    """NO_OBSERVATIONS is named in normalize's own docstring as a value
    callers may see; an empty observation list is the edge case that
    produces it, and nothing else in this file constructs one."""
    empty = _sample(frames=0)
    assert empty.observations == ()
    _, reasons = normalize(empty, detect=_detector([]))
    assert reasons["faces"] == NO_OBSERVATIONS
    assert reasons["frames_with_face"] == "0/0"
    assert reasons["max_faces_in_frame"] == "0"
