"""`training.fit_npr` fits slot C and writes a state_dict the registry loads.

The assertions here exist because each alternative ships something that looks
fitted and is not: a one-column head that only fails at score time, a
normalisation left in the fitter so inference sees raw features, a holdout
that reports the fit split back to you, and a preprocessing step whose
omission was measured to turn a 0.63 cross-corpus number into 0.46.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from dfd.detectors.loading import load_model
from dfd.detectors.npr import NPRDetector, NPRStatsNet
from dfd.faces import FaceBox
from dfd.types import Modality, Observation, Quality
from training.fit_npr import main, save_npr_model, split_items, stats_for


def _box() -> FaceBox:
    lms = np.array([[30.0, 40.0], [60.0, 40.0], [45.0, 55.0],
                    [33.0, 70.0], [57.0, 70.0]])
    return FaceBox(x=20, y=25, w=50, h=60, landmarks=lms, score=0.99)


def _detect_one(_frame):
    return [_box()]


def _corpus(root: Path, n: int = 30) -> tuple[Path, Path]:
    """Fakes are nearest-neighbour upsampled; reals are not.

    That is the physics slot C claims to read, so a fitter that works at all
    must separate these two, and one that does not has nothing to do with
    upsampling.
    """
    fakes, reals = root / "fake", root / "real"
    for d in (fakes, reals):
        d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        small = rng.integers(0, 255, (60, 50, 3), dtype=np.uint8)
        up = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)
        cv2.imwrite(str(fakes / f"f{i:03d}.png"), up)
        cv2.imwrite(str(reals / f"r{i:03d}.png"),
                    rng.integers(0, 255, (120, 100, 3), dtype=np.uint8))
    return fakes, reals


def test_the_split_is_deterministic_and_covers_every_item() -> None:
    a, b = split_items(50, holdout_fraction=0.3, seed=1)
    a2, b2 = split_items(50, holdout_fraction=0.3, seed=1)
    a3, _ = split_items(50, holdout_fraction=0.3, seed=2)
    assert sorted(np.r_[a, b].tolist()) == list(range(50))
    assert set(a) & set(b) == set()
    assert a.tolist() == a2.tolist() and b.tolist() == b2.tolist()
    assert a.tolist() != a3.tolist()


def test_stats_are_returned_for_a_face_and_none_without_one(tmp_path: Path) -> None:
    fakes, _ = _corpus(tmp_path, n=1)
    p = next(fakes.iterdir())
    assert stats_for(p, detect=_detect_one, norm_size=224).shape == (
        NPRStatsNet.N_FEATURES,)
    assert stats_for(p, detect=lambda _f: [], norm_size=224) is None
    assert stats_for(tmp_path / "missing.png", detect=_detect_one,
                     norm_size=224) is None


def test_the_saved_head_has_two_columns_and_reproduces_the_logistic(
        tmp_path: Path) -> None:
    """A one-column head loads clean and dies inside `NPRDetector.score`.

    The detector checks for exactly two output classes, so a head saved with
    one column passes every fitter-side assertion and fails at the only place
    nobody is watching.

    It pins the SHAPE and the PROBABILITY, not the particular split of the
    logit across the two columns: softmax over (-w/2, +w/2) and over (0, w)
    are the same function, so both are correct and this test accepts either.
    What it rejects is any saving that changes the classifier.
    """
    rng = np.random.default_rng(0)
    coef = rng.normal(size=NPRStatsNet.N_FEATURES)
    out = tmp_path / "npr.pt"
    save_npr_model(coef, 0.37, np.zeros(NPRStatsNet.N_FEATURES),
                   np.ones(NPRStatsNet.N_FEATURES), out)
    net = load_model(out, NPRStatsNet, False)
    assert net.linear.weight.shape == (2, NPRStatsNet.N_FEATURES)
    x = torch.from_numpy(rng.normal(0, 0.3, (4, 3, 32, 32)).astype(np.float32))
    with torch.no_grad():
        f = net.features(x).numpy()
        got = torch.softmax(net(x), dim=1)[:, 1].numpy()
    expected = 1.0 / (1.0 + np.exp(-(f @ coef + 0.37)))
    assert np.allclose(got, expected, atol=1e-5)


def test_the_normalisation_is_written_into_the_weights(tmp_path: Path) -> None:
    """Left in the fitter, inference would see raw features and be wrong."""
    out = tmp_path / "npr.pt"
    mean = np.arange(NPRStatsNet.N_FEATURES, dtype=float)
    scale = np.full(NPRStatsNet.N_FEATURES, 2.5)
    save_npr_model(np.ones(NPRStatsNet.N_FEATURES), 0.0, mean, scale, out)
    state = torch.load(out, weights_only=True)
    assert np.allclose(state["feature_mean"].numpy(), mean)
    assert np.allclose(state["feature_scale"].numpy(), scale)


def test_it_fits_the_upsampling_signal_and_the_detector_then_scores(
        tmp_path: Path) -> None:
    fakes, reals = _corpus(tmp_path)
    out, report = tmp_path / "npr.pt", tmp_path / "r.json"
    code = main(["--fakes", str(fakes), "--reals", str(reals),
                 "--out", str(out), "--report", str(report)],
                detect=_detect_one)
    assert code == 0
    r = json.loads(report.read_text())
    assert r["n_fakes"] == r["n_reals"] == 30
    # The fixture's two classes differ ONLY by upsampling, so a fitter that
    # reads the residual at all must separate them.
    assert r["holdout_auc_in_family"] > 0.9

    d = NPRDetector(weights_path=out, model_factory=NPRStatsNet)
    img = cv2.imread(str(next(fakes.iterdir())))
    obs = Observation(t=0.0, payload=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                      roi=(0, 0, img.shape[1], img.shape[0]),
                      quality=Quality(inter_ocular_px=100, blur_var=200,
                                      yaw_deg=0, pitch_deg=0, exposure=0.5,
                                      band="high"),
                      source_id="s1")
    scored = d.score([obs])
    assert not scored.abstained
    assert Modality.IMAGE in d.modalities


def test_a_single_label_corpus_is_refused(tmp_path: Path) -> None:
    """Every metric over one label is undefined rather than wrong."""
    fakes, _ = _corpus(tmp_path, n=4)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["--fakes", str(fakes), "--reals", str(empty),
                 "--out", str(tmp_path / "npr.pt")], detect=_detect_one) == 1


def test_the_report_records_the_preprocessing_that_makes_it_transfer(
        tmp_path: Path) -> None:
    """`norm_size` is the difference between 0.63 and chance on an unseen
    corpus (docs/HANDOFF.md §0). A report that does not say which was used
    cannot be compared with another run."""
    fakes, reals = _corpus(tmp_path, n=10)
    report = tmp_path / "r.json"
    main(["--fakes", str(fakes), "--reals", str(reals), "--norm-size", "128",
          "--out", str(tmp_path / "npr.pt"), "--report", str(report)],
         detect=_detect_one)
    assert json.loads(report.read_text())["norm_size"] == 128


def test_the_reported_auc_is_the_holdout_and_not_the_fit_split(
        tmp_path: Path) -> None:
    """Both classes drawn from ONE distribution, so there is nothing to learn.

    27 features over 60 items will still separate the fit split by
    memorisation. A fitter that reported its own fit split back would post a
    confident number here; an honest holdout lands near chance. This is the
    only shape of test that can tell the two apart — a fixture with real
    signal scores ~1.0 either way.
    """
    fakes, reals = tmp_path / "fake", tmp_path / "real"
    for d in (fakes, reals):
        d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for i in range(30):
        for d in (fakes, reals):
            cv2.imwrite(str(d / f"{i:03d}.png"),
                        rng.integers(0, 255, (120, 100, 3), dtype=np.uint8))
    report = tmp_path / "r.json"
    assert main(["--fakes", str(fakes), "--reals", str(reals),
                 "--out", str(tmp_path / "npr.pt"), "--report", str(report)],
                detect=_detect_one) == 0
    assert json.loads(report.read_text())["holdout_auc_in_family"] < 0.75


def test_the_source_is_normalised_before_detection(tmp_path: Path) -> None:
    """The same picture at two source resolutions must give the same features.

    Measured 2026-09-23 (docs/HANDOFF.md §0): without this, a model fitted on
    1024px fakes against 224px reals scores 0.969 in-family and transfers to
    an unseen corpus at 0.461 — chance — because image width alone separates
    the corpora at AUC 1.000. The resize is the whole difference, so it is
    asserted rather than trusted to a comment.
    """
    rng = np.random.default_rng(11)
    small = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    big = cv2.resize(small, (512, 512), interpolation=cv2.INTER_CUBIC)
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    cv2.imwrite(str(a), small)
    cv2.imwrite(str(b), big)
    # **Not an equality claim, and deliberately not, since 2026-09-24.** The
    # feature vector now carries resampling-sensitive terms (the spectral
    # block's peak-to-mean ratio above all), because reading resampling
    # history IS slot C's job — see `NPRStatsNet._spectral`. Asserting
    # invariance would be asserting that the detector cannot see the thing it
    # exists to see.
    #
    # What the resize still buys, and what this asserts, is a large REDUCTION
    # in the gap. Measured on this fixture 2026-09-24:
    #
    #     without the resize   max gap 71.53, mean gap 3.615
    #     with it (both 224)   max gap  7.10, mean gap 0.329
    #
    # A tenfold reduction on both. That is what keeps image width from
    # separating SFHQ at 1024px from FairFace at 224px.
    with_resize = np.abs(np.asarray(stats_for(a, detect=_detect_one, norm_size=224))
                         - np.asarray(stats_for(b, detect=_detect_one, norm_size=224)))
    without = np.abs(np.asarray(stats_for(a, detect=_detect_one, norm_size=128))
                     - np.asarray(stats_for(b, detect=_detect_one, norm_size=512)))
    assert with_resize.max() < without.max() / 5.0, (
        f"the resize reduced the worst gap only from {without.max():.2f} to "
        f"{with_resize.max():.2f}; source resolution is still leaking into "
        f"the features")
    assert with_resize.mean() < without.mean() / 5.0, (
        f"mean gap {without.mean():.3f} -> {with_resize.mean():.3f}")


def test_a_feature_with_no_variance_does_not_divide_by_zero(
        tmp_path: Path) -> None:
    """Flat images give an all-zero residual, so EVERY feature has zero
    variance on the fit split. Without the guard the standardisation divides
    by zero, sklearn is handed non-finite input, and the fitter dies on a
    corpus that is merely degenerate rather than invalid."""
    fakes, reals = tmp_path / "fake", tmp_path / "real"
    for d, v in ((fakes, 120), (reals, 200)):
        d.mkdir(parents=True, exist_ok=True)
        for i in range(15):
            cv2.imwrite(str(d / f"{i:03d}.png"),
                        np.full((120, 100, 3), v, dtype=np.uint8))
    assert main(["--fakes", str(fakes), "--reals", str(reals),
                 "--out", str(tmp_path / "npr.pt")], detect=_detect_one) == 0
