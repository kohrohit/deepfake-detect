"""build_face_pool turns sessions into aligned crops, and says what it dropped."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from corpora.captures import CaptureSession
from corpora.face_pool import (DEGENERATE_BOX, DUPLICATE, NO_FACE, UNREADABLE,
                               build_face_pool)


def _session(session_id: str, folder: str, swapped: bool = False) -> CaptureSession:
    # folder is a COMPLETE path, exactly as load_capture_sessions produces it
    # (str(path.parent) from a glob already rooted at the caller's root) —
    # never a bare session id that build_face_pool must re-root.
    return CaptureSession(session_id=session_id, folder=folder, swapped=swapped,
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
    crops, skipped = build_face_pool([_session("s1", str(tmp_path / "s1"))],
                                     detect=lambda f: [_fake_box()])
    assert len(crops) == 2
    assert {c.frame_index for c in crops} == {0, 1}
    assert all(c.session_id == "s1" for c in crops)
    assert all(c.image.shape == (224, 224, 3) for c in crops)
    assert all(c.image.dtype == np.uint8 for c in crops)
    assert skipped == {}


def test_swapped_flag_is_carried_from_the_session(tmp_path: Path) -> None:
    _write_frames(tmp_path, "s_fraud", n=1)
    crops, _ = build_face_pool(
        [_session("s_fraud", str(tmp_path / "s_fraud"), swapped=True)],
        detect=lambda f: [_fake_box()])
    assert crops and all(c.swapped for c in crops)


def test_frames_with_no_detected_face_are_skipped_and_counted(tmp_path: Path) -> None:
    _write_frames(tmp_path, "s1", n=2)
    crops, skipped = build_face_pool([_session("s1", str(tmp_path / "s1"))],
                                     detect=lambda f: [])
    assert crops == []
    assert skipped == {NO_FACE: 2}


def test_an_unreadable_frame_is_counted_not_raised(tmp_path: Path) -> None:
    d = tmp_path / "s1"
    d.mkdir(parents=True)
    (d / "frame_00.jpg").write_bytes(b"not a jpeg")
    crops, skipped = build_face_pool([_session("s1", str(d))],
                                     detect=lambda f: [_fake_box()])
    assert crops == []
    assert skipped == {UNREADABLE: 1}


def test_max_frames_per_session_caps_the_pool(tmp_path: Path) -> None:
    _write_frames(tmp_path, "s1", n=5)
    crops, _ = build_face_pool([_session("s1", str(tmp_path / "s1"))],
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
    crops, _ = build_face_pool([_session("s1", str(tmp_path / "s1"))],
                               detect=lambda f: [small, big])
    assert len(crops) == 1
    assert crops[0].box.score == pytest.approx(0.95)


def test_a_session_folder_is_used_as_the_complete_path_it_is(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CaptureSession.folder is already a full path; build_face_pool must not
    re-root it. A prior version took a `root` kwarg and joined it against
    `session.folder`, which double-prefixed a relative folder into a path
    that does not exist. Regression test for that: run from a cwd where a
    naive `Path(some_root) / session.folder` join would land on a
    non-existent directory, and confirm the real (relative) folder is still
    found.
    """
    monkeypatch.chdir(tmp_path)
    _write_frames(Path("captures"), "s1", n=1)
    session = _session("s1", "captures/s1")
    crops, skipped = build_face_pool([session], detect=lambda f: [_fake_box()])
    assert len(crops) == 1
    assert skipped == {}


def _write_same_frame(root: Path, session_id: str, payload: bytes) -> None:
    d = root / session_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "frame_00.jpg").write_bytes(payload)


def test_byte_identical_frames_across_sessions_yield_one_crop(tmp_path: Path) -> None:
    """The same image enrolled under many session ids is one observation, not many.

    Measured on the real corpus, 2026-09-22: 979 of its 1088 frame files are
    byte-identical to `assets/attack/victim_id.jpg`, spread over 368 of the
    442 sessions. Keeping one crop per session would hand `split_by_subject`
    — which splits on session id — the SAME image on both sides of the
    holdout, so the held-out AUC would measure memorisation of one picture
    and report it as generalisation. Deduplicating is what makes that split
    mean anything.
    """
    src = tmp_path / "src"
    src.mkdir()
    rng = np.random.default_rng(0)
    cv2.imwrite(str(src / "f.jpg"),
                rng.integers(0, 255, (120, 100, 3), dtype=np.uint8))
    payload = (src / "f.jpg").read_bytes()
    for sid in ("s1", "s2", "s3"):
        _write_same_frame(tmp_path, sid, payload)

    crops, skipped = build_face_pool(
        [_session(sid, str(tmp_path / sid)) for sid in ("s1", "s2", "s3")],
        detect=lambda f: [_fake_box()])

    assert len(crops) == 1, "the same image must not enter the pool three times"
    assert crops[0].session_id == "s1", "the first session to carry it wins"
    assert skipped == {DUPLICATE: 2}


def _box(x: int, y: int, w: int, h: int):
    from dfd.faces import FaceBox
    lms = np.array([[30.0, 40.0], [60.0, 40.0], [45.0, 55.0],
                    [33.0, 70.0], [57.0, 70.0]])
    return FaceBox(x=x, y=y, w=w, h=h, landmarks=lms, score=0.95)


def test_a_box_hanging_off_the_frame_is_clamped_not_crashed(tmp_path: Path) -> None:
    """A detection may extend past the frame edge, and usually does on a
    tightly-cropped face dataset.

    Measured 2026-09-22 against FairFace (97,698 pre-aligned 224x224 faces):
    YuNet returned a box reaching outside the image on 146 of the first 300.
    `measure_quality` slices `frame[y:y + h, x:x + w]` with no clamping, so a
    negative origin slices from the FAR END of the array and yields an empty
    crop, and `cv2.cvtColor` then raises. `dfd.pipeline._clamp_roi` and
    `dfd.faces.align` both already guard this; `build_face_pool` was the one
    site that did not, so the whole pool builder crashed on image 2 of any
    public face dataset.
    """
    _write_frames(tmp_path, "s1", n=1)
    crops, skipped = build_face_pool([_session("s1", str(tmp_path / "s1"))],
                                     detect=lambda f: [_box(50, -5, 173, 217)])
    assert len(crops) == 1
    assert skipped == {}


def test_a_box_that_misses_the_frame_entirely_is_counted_not_crashed(
        tmp_path: Path) -> None:
    _write_frames(tmp_path, "s1", n=1)
    crops, skipped = build_face_pool([_session("s1", str(tmp_path / "s1"))],
                                     detect=lambda f: [_box(-500, -500, 100, 100)])
    assert crops == []
    assert skipped == {DEGENERATE_BOX: 1}
