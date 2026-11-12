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


def test_colour_features_read_their_input_as_RGB():
    # The contract its only caller depends on. `bench.eval_dfeval` passes
    # each record's own frame straight through -- the corpora store RGB --
    # rather than building a second, complete, BGR copy of the decoded
    # corpus to produce six numbers per image. A BGR reading of the same
    # bytes would call this pure red pure blue, and b* would come back
    # BELOW the 128 midpoint instead of far above it.
    red = np.zeros((8, 8, 3), np.uint8)
    red[..., 0] = 255
    l_mean, a_mean, b_mean = colour_features([red])[0][:3]
    assert b_mean > 128      # yellow-positive: red, not blue
    assert a_mean > 128      # magenta-positive


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
    # The observed statistic sits INSIDE the null's own range -- the thing
    # `p > 0.1` is a summary of. Previously `... or out["p"] > 0.1`, which
    # the line above had already asserted, so this clause could never fail.
    assert out["null_lo"] <= out["auc"] <= out["null_hi"]


def test_an_all_nan_feature_row_is_dropped_and_shrinks_n_fit_and_n_test():
    # `jpeg_table_features` returns an all-NaN row for a PNG, and an
    # in-the-wild corpus is mixed JPEG/PNG, so this fires on the first real
    # run -- and it governs the arithmetic of the only control that can
    # disqualify the corpus. The unmeasured rows must leave the fit AND the
    # test half, and the reported n must say so rather than quoting the
    # row count it was handed.
    rng = np.random.default_rng(7)
    labels = np.array([0, 1] * 100)
    groups = np.array([f"g{i // 10}" for i in range(200)])

    clean = rng.normal(size=(200, 6))
    whole = run_control(clean, labels, groups, seed=0)
    assert whole["n_fit"] + whole["n_test"] == 200

    holed = clean.copy()
    holed[::10] = np.nan          # one unmeasurable row in every group
    out = run_control(holed, labels, groups, seed=0)
    assert out["n_fit"] + out["n_test"] == 180
    assert out["n_fit"] > 0 and out["n_test"] > 0


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


def test_run_control_splits_class_pure_groups_without_raising():
    # On the real corpus (Deepfake-Eval-2024, grouped by originating web
    # domain) a domain that only hosts fakes or only contributes reals is
    # not an edge case -- it is what 88 in-the-wild domains look like. A
    # blind group split can strand a whole class on one side; this one must
    # not, whenever an assignment avoiding that exists.
    rng = np.random.default_rng(4)
    rows_per_group = 10
    features = []
    labels = []
    groups = []
    for i in range(4):
        for _ in range(rows_per_group):
            features.append(rng.normal(size=6))
            labels.append(0)
            groups.append(f"real{i}")
    for i in range(4):
        for _ in range(rows_per_group):
            features.append(rng.normal(size=6))
            labels.append(1)
            groups.append(f"fake{i}")
    out = run_control(np.asarray(features), np.asarray(labels),
                      np.asarray(groups), seed=0)
    assert out["n_fit"] > 0
    assert out["n_test"] > 0


def test_run_control_refuses_an_unsplittable_group_structure():
    # One real domain, one fake domain: no assignment of two class-pure
    # groups can give both sides both classes, so the control must refuse
    # with its own message rather than let sklearn fail on a one-class fit.
    rng = np.random.default_rng(5)
    features = np.concatenate([rng.normal(size=(5, 6)), rng.normal(size=(5, 6))])
    labels = np.array([0] * 5 + [1] * 5)
    groups = np.array(["real0"] * 5 + ["fake0"] * 5)
    with pytest.raises(ValueError, match="both labels"):
        run_control(features, labels, groups, seed=0)


def test_a_file_pil_cannot_open_is_not_measured_rather_than_fatal(tmp_path):
    # cv2 and PIL do not read the same set of files, and every path handed
    # to this function came from a record cv2 ALREADY decoded. One such file
    # used to abort the entire run -- detectors, both controls, the report --
    # with an UnidentifiedImageError raised from inside an argument
    # expression. The row is simply not measured, which is what NaN means.
    import shutil

    import cv2

    from bench.dfeval_controls import jpeg_table_features

    rng = np.random.default_rng(6)
    good = tmp_path / "good.jpg"
    cv2.imwrite(str(good), rng.integers(0, 255, (32, 32, 3), dtype=np.uint8))

    # A mislabelled download: the manifest calls it .jpg, the bytes are a
    # format PIL has no plugin for. cv2 sniffs the content and reads it.
    pfm = tmp_path / "x.pfm"
    cv2.imwrite(str(pfm), rng.integers(0, 255, (32, 32, 3), dtype=np.uint8))
    mislabelled = tmp_path / "mislabelled.jpg"
    shutil.move(str(pfm), str(mislabelled))
    assert cv2.imread(str(mislabelled)) is not None

    x = jpeg_table_features([good, mislabelled])
    assert x.shape[0] == 2
    assert not np.isnan(x[0]).any()
    assert np.isnan(x[1]).all()


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
