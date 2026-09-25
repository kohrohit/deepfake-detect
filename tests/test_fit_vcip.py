"""The capture-corpus fitter, and the train/serve agreement it must keep.

The recurring defect in this project is a fitter whose preprocessing differs
from its detector's: the weights then describe features nobody computes at
inference. Two such mismatches have already been found in this slot — the
detector scoring the whole frame rather than the face, and the fitter picking
a different face from the pipeline. These tests pin both.
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from dfd.faces import FaceBox


def _box(x, y, w, h, score):
    lms = np.array([[x + w * 0.3, y + h * 0.3], [x + w * 0.7, y + h * 0.3],
                    [x + w * 0.5, y + h * 0.5], [x + w * 0.35, y + h * 0.75],
                    [x + w * 0.65, y + h * 0.75]])
    return FaceBox(x=x, y=y, w=w, h=h, landmarks=lms, score=score)


def test_the_fitter_picks_the_same_face_the_pipeline_scores():
    """A frame with two faces must yield the SAME crop on both paths.

    `dfd.pipeline` selects `max(boxes, key=w * h)` — the LARGEST face. A
    fitter selecting `max(boxes, key=b.score)` — the most CONFIDENT — picks a
    different face whenever the two disagree, and the head is then fitted on
    one face and asked to score another.

    Measured consequence, 2026-09-24: the assembled system scored 0.821 at
    session level against 0.954 for the same head measured out-of-fold by the
    fitter itself — and the assembled run was NOT held out, so it should have
    been optimistic rather than worse. Both are pre-fix figures, quoted as
    the evidence that found the defect; the post-fix out-of-fold number is
    0.957. The serving path is the one that
    cannot change, so the fitter follows it.
    """
    from training.fit_vcip import select_face

    big_but_less_confident = _box(10, 10, 200, 200, 0.70)
    small_but_confident = _box(300, 300, 50, 50, 0.99)
    chosen = select_face([small_but_confident, big_but_less_confident])
    assert chosen is big_but_less_confident, (
        "the fitter picked the most confident face; the pipeline picks the "
        "largest, so the head would be fitted on a different face from the "
        "one it scores")


def test_select_face_matches_the_pipelines_rule_on_random_boxes():
    """Not one hand-built case: the same rule on many shapes.

    A test with a single fixture passes for a fitter that hard-codes the
    answer. This asserts agreement with the pipeline's actual expression.
    """
    from training.fit_vcip import select_face

    rng = np.random.default_rng(0)
    for _ in range(50):
        boxes = [_box(int(rng.integers(0, 100)), int(rng.integers(0, 100)),
                      int(rng.integers(10, 200)), int(rng.integers(10, 200)),
                      float(rng.random()))
                 for _ in range(int(rng.integers(2, 5)))]
        expected = max(boxes, key=lambda b: b.w * b.h)
        assert select_face(boxes) is expected


def test_select_face_returns_none_for_no_faces():
    """A frame with no face has no crop, and inventing one scores background."""
    from training.fit_vcip import select_face
    assert select_face([]) is None


def _session(root, name, *, swapped, n=2, size=(720, 1280)):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    frames = []
    for i in range(n):
        img = rng.integers(0, 255, (*size, 3), dtype=np.uint8)
        cv2.imwrite(str(d / f"frame_{i:02d}.jpg"), img)
        frames.append({"index": i, "swapped": swapped,
                       "saved_as": f"captures/{name}/frame_{i:02d}.jpg"})
    (d / "results.json").write_text(json.dumps(
        {"session_id": name, "swapped": swapped, "frames": frames}))
    return d


def test_it_refuses_a_corpus_with_only_one_label(tmp_path):
    """A head needs both classes. Returning a model fitted on one label
    would write weights that score everything the same way."""
    from training import fit_vcip

    root = tmp_path / "captures"
    for i in range(4):
        _session(root, f"s{i}", swapped=True)
    code = fit_vcip.main(
        ["--captures", str(root), "--out", str(tmp_path / "npr.pt"),
         "--report", str(tmp_path / "r.json"),
         "--calibration", str(tmp_path / "c.json")],
        detect=lambda f: [_box(100, 100, 300, 300, 0.9)])
    assert code == 1
    assert not (tmp_path / "npr.pt").exists()


def test_it_refuses_a_missing_capture_directory(tmp_path):
    from training import fit_vcip
    assert fit_vcip.main(
        ["--captures", str(tmp_path / "nope"),
         "--out", str(tmp_path / "npr.pt"),
         "--report", str(tmp_path / "r.json"),
         "--calibration", str(tmp_path / "c.json")]) == 1


def test_shapes_present_in_only_one_class_are_dropped(tmp_path):
    """Image dimensions correlated with the label are a shortcut: the head
    would learn a property of the recording setup, not of the swap."""
    from training.fit_vcip import _frames, _shapes_in_both_classes

    root = tmp_path / "captures"
    _session(root, "genuine_a", swapped=False, size=(720, 1280))
    _session(root, "swapped_a", swapped=True, size=(720, 1280))
    # A shape only the genuine half ever has.
    _session(root, "genuine_b", swapped=False, size=(544, 768))

    keep = _shapes_in_both_classes(_frames(root))
    assert (720, 1280, 3) in keep
    assert (544, 768, 3) not in keep
