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
