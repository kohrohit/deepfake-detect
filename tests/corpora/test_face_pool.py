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
