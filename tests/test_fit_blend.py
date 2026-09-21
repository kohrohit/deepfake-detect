"""The fitter must not leak a subject across the split, and must say so."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from corpora.face_pool import FaceCrop
from corpora.sbi import build_sbi_corpus
from dfd.detectors.blend import FEATURE_NAMES, load_blend_model
from dfd.faces import FaceBox
from dfd.types import Quality
from training.fit_blend import evaluate, fit_blend_model, main, split_by_subject


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


def test_scale_is_never_zero_when_a_feature_is_constant_across_the_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fixture above happens to vary on every one of the 30 seam
    features, so it alone never exercises the zero-scale guard: removing
    `np.where(scale == 0.0, 1.0, scale)` still leaves every prior test
    green. This test forces feature 0 to a constant so a fitted model must
    actually divide by that column's (zero) spread."""
    import training.fit_blend as fit_blend_module

    real_seam_features = fit_blend_module.seam_features

    def constant_first_feature(img: np.ndarray) -> np.ndarray:
        feats = real_seam_features(img).copy()
        feats[0] = 5.0
        return feats

    monkeypatch.setattr(fit_blend_module, "seam_features", constant_first_feature)

    train, _ = split_by_subject(_corpus(), seed=0)
    m = fit_blend_model(train, version="t1")
    assert m.scale[0] == 1.0
    assert (m.scale != 0).all()
    assert np.isfinite(m.coef).all()


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
    """What this proves, and what it does not.

    Proves: the wiring. Feature extraction, the subject-disjoint split,
    fitting and evaluate() compose into something that separates THESE
    synthetic fixtures well above chance.

    Does NOT prove anything about real faces: the fixtures differ from
    their self-blends in ways a camera never would, so a high AUC here is
    a property of the fixture generator, not of the detector's real-world
    accuracy. The pipeline measured on synthetic textured fixtures scored
    a held-out AUC of 1.000.

    Why 0.9 and not the weaker-looking 0.5 that used to be here: with pure
    random-noise features run through this exact pipeline (fit and
    evaluate, no seam signal at all), 19 of 40 seeds still passed `> 0.5`
    — that bar was a coin flip, not a check. At `> 0.9`, the same
    noise-feature run failed all 40 of 40 seeds, while the real pipeline
    (this test, unmutated) still passes comfortably (AUC 1.000 at seed=0).
    0.9 is therefore the bar that actually distinguishes "the features
    carry seam signal" from "the model memorised nothing and got lucky."
    """
    train, test = split_by_subject(_corpus(n=40), seed=0)
    m = fit_blend_model(train, version="t1")
    assert evaluate(m, test)["auc"] > 0.9


# --- main(): exercised only against a synthetic capture corpus written into
# tmp_path, via the injected `detect` seam. Never against the real corpus.

def _seed_for(name: str) -> int:
    """A deterministic per-name seed. Not `hash()`: Python salts str hashing
    per process unless PYTHONHASHSEED is set, which would make two calls in
    the same test run agree but not two separate runs — see the identical
    reasoning in corpora/sbi.py for build_sbi_corpus's per-sample seeding."""
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")


def _write_session(root: Path, session_id: str, *, swapped: bool = False,
                   n_frames: int = 1) -> None:
    """A minimal capture session on disk: a results.json plus real, decodable
    frame_NN.jpg files, so build_face_pool's cv2.imread actually succeeds."""
    folder = root / session_id
    folder.mkdir(parents=True)
    (folder / "results.json").write_text(json.dumps({
        "session_id": session_id,
        "swapped": swapped,
        "decision": {"approved": True},
        "scan": {"verdict": "ok"},
        "frame_count": n_frames,
    }))
    rng = np.random.default_rng(_seed_for(session_id))
    for i in range(n_frames):
        img = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(str(folder / f"frame_{i:02d}.jpg"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def _one_box_detector(frame: np.ndarray) -> list[FaceBox]:
    """A stub detector standing in for YuNet: always finds the same box."""
    lms = np.array([[70.0, 80.0], [110.0, 80.0], [90.0, 100.0],
                    [75.0, 125.0], [105.0, 125.0]])
    return [FaceBox(x=55, y=55, w=70, h=90, landmarks=lms, score=0.9)]


def _no_box_detector(frame: np.ndarray) -> list[FaceBox]:
    """A stub detector that never finds a face — exercises the NO_FACE path."""
    return []


def test_main_happy_path_writes_a_loadable_model_and_a_consistent_report(
    tmp_path: Path,
) -> None:
    captures = tmp_path / "captures"
    for i in range(6):
        _write_session(captures, f"real{i}", swapped=False)
    out = tmp_path / "model.npz"
    report = tmp_path / "report.json"

    code = main(
        ["--captures", str(captures), "--out", str(out),
         "--report", str(report), "--seed", "0"],
        detect=_one_box_detector,
    )

    assert code == 0
    model = load_blend_model(out)
    assert model.feature_names == FEATURE_NAMES

    data = json.loads(report.read_text())
    assert set(data) >= {"auc", "n", "n_fake", "n_train", "n_test",
                         "n_sessions", "n_genuine", "n_crops", "skipped",
                         "version", "seed"}
    assert data["n_sessions"] == 6
    assert data["n_genuine"] == 6
    assert data["n_crops"] == 6
    # Every crop yields exactly two samples (real, self-blend), and
    # split_by_subject partitions rows without dropping any.
    assert data["n_train"] + data["n_test"] == 2 * data["n_crops"]
    assert data["n"] == data["n_test"]


def test_main_filters_swapped_sessions_before_blending(tmp_path: Path) -> None:
    """`not s.swapped` in main() is defence-in-depth, not the only guard:
    build_sbi_corpus independently hard-refuses any crop from a swapped
    session with EvaluationOnlySessionError. So removing this filter would
    not leak evaluation-only fraud into training silently — it would fail
    loudly with that exception the moment build_sbi_corpus saw the crop.
    What this test checks is narrower: that the filter, as written, keeps
    the swapped session's crop out of the corpus at all (n_crops counts
    only the six genuine sessions, not the seventh)."""
    captures = tmp_path / "captures"
    for i in range(6):
        _write_session(captures, f"real{i}", swapped=False)
    _write_session(captures, "swapped0", swapped=True)
    out = tmp_path / "model.npz"
    report = tmp_path / "report.json"

    code = main(
        ["--captures", str(captures), "--out", str(out),
         "--report", str(report)],
        detect=_one_box_detector,
    )

    assert code == 0
    data = json.loads(report.read_text())
    assert data["n_sessions"] == 7
    assert data["n_genuine"] == 6
    assert data["n_crops"] == 6


def test_main_returns_1_and_writes_nothing_when_no_faces_are_found(
    tmp_path: Path,
) -> None:
    captures = tmp_path / "captures"
    _write_session(captures, "real0", swapped=False)
    out = tmp_path / "model.npz"
    report = tmp_path / "report.json"

    code = main(
        ["--captures", str(captures), "--out", str(out),
         "--report", str(report)],
        detect=_no_box_detector,
    )

    assert code == 1
    assert not out.exists()
    assert not report.exists()
