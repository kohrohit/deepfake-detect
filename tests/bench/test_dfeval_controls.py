from __future__ import annotations

import numpy as np
import pytest

from bench.dfeval_controls import colour_features, run_control


def _planted(rng, n, lo, hi):
    return [rng.integers(lo, hi, (24, 24, 3), dtype=np.uint8) for _ in range(n)]


def test_the_colour_control_finds_a_planted_colour_shortcut():
    # The control must be able to FIRE, or reporting it proves nothing.
    rng = np.random.default_rng(0)
    images = _planted(rng, 60, 0, 110) + _planted(rng, 60, 145, 255)
    labels = np.array([0] * 60 + [1] * 60)
    groups = np.array([f"g{i % 12}" for i in range(120)])
    out = run_control(colour_features(images), labels, groups, seed=0)
    assert out["auc"] > 0.9
    assert out["p"] < 0.1          # and it escapes its own null


def test_the_colour_control_stays_blind_to_a_corpus_without_one():
    rng = np.random.default_rng(0)
    images = _planted(rng, 60, 0, 255) + _planted(rng, 60, 0, 255)
    labels = np.array([0] * 60 + [1] * 60)
    groups = np.array([f"g{i % 12}" for i in range(120)])
    out = run_control(colour_features(images), labels, groups, seed=0)
    assert 0.3 < out["auc"] < 0.7


def test_the_control_fits_and_tests_on_disjoint_groups():
    # Fitting and reporting on the same rows is a curve that has already
    # seen its own exam, and it would make every control fire on every
    # corpus -- which is indistinguishable from a control that never fires.
    rng = np.random.default_rng(1)
    features = rng.normal(size=(200, 6))
    labels = np.array([0, 1] * 100)
    groups = np.array([f"g{i // 10}" for i in range(200)])
    out = run_control(features, labels, groups, seed=0)
    assert out["n_fit"] > 0 and out["n_test"] > 0
    assert out["n_fit"] + out["n_test"] <= 200


def test_a_control_on_noise_does_not_escape_its_null():
    rng = np.random.default_rng(2)
    features = rng.normal(size=(200, 6))
    labels = np.array([0, 1] * 100)
    groups = np.array([f"g{i // 10}" for i in range(200)])
    out = run_control(features, labels, groups, seed=0)
    assert out["p"] > 0.1
    assert out["null_lo"] <= out["auc"] <= out["null_hi"] or out["p"] > 0.1


def test_the_null_is_refused_on_a_one_class_corpus():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="both labels"):
        run_control(rng.normal(size=(40, 6)), np.zeros(40, dtype=int),
                    np.array([f"g{i // 4}" for i in range(40)]), seed=0)


def test_jpeg_table_features_separate_two_encode_qualities(tmp_path):
    # The statistic that labels 99.2% of this project's capture corpus,
    # read here from the file header rather than from the pixels.
    import cv2

    from bench.dfeval_controls import jpeg_table_features

    rng = np.random.default_rng(0)
    paths = []
    for i, q in enumerate([30] * 10 + [95] * 10):
        img = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        p = tmp_path / f"{i}.jpg"
        cv2.imwrite(str(p), img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        paths.append(p)
    x = jpeg_table_features(paths)
    assert x.shape[0] == 20
    assert not np.allclose(x[:10].mean(axis=0), x[10:].mean(axis=0))


def test_jpeg_table_features_report_nothing_for_a_non_jpeg(tmp_path):
    # A PNG has no quantisation table. Returning zeros would be a feature
    # value; the control must be able to say the statistic does not exist
    # for that row and exclude it.
    import cv2

    from bench.dfeval_controls import jpeg_table_features

    p = tmp_path / "x.png"
    cv2.imwrite(str(p), np.zeros((16, 16, 3), np.uint8))
    x = jpeg_table_features([p])
    assert np.isnan(x[0]).all()
