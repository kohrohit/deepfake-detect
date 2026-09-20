import math
import numpy as np
import pytest
from dfd.calibration import Calibrator
from dfd.types import RawScore


def _raw(score, detector="d1"):
    return RawScore(detector=detector, version="0.1.0", score=score,
                    abstained=False, reason="ok")


def _fit_separable(band="high", n=200):
    rng = np.random.default_rng(0)
    real = rng.uniform(0.0, 0.4, n)
    fake = rng.uniform(0.6, 1.0, n)
    scores = np.concatenate([real, fake])
    labels = np.concatenate([np.zeros(n), np.ones(n)])
    bands = [band] * len(scores)
    c = Calibrator(detector="d1")
    c.fit(scores, labels, bands)
    return c


def test_high_score_yields_positive_llr():
    c = _fit_separable()
    ev = c.to_evidence(_raw(0.95), band="high")
    assert ev.llr > 0


def test_low_score_yields_negative_llr():
    c = _fit_separable()
    ev = c.to_evidence(_raw(0.05), band="high")
    assert ev.llr < 0


def test_abstained_rawscore_becomes_zero_llr_evidence():
    c = _fit_separable()
    raw = RawScore(detector="d1", version="0.1.0", score=None,
                   abstained=True, reason="weights_absent")
    ev = c.to_evidence(raw, band="high")
    assert ev.llr == 0.0 and ev.abstained and ev.reason == "weights_absent"


def test_uncalibrated_band_yields_zero_llr_not_a_guess():
    """The honest response to 'I was never calibrated here' is no information."""
    c = _fit_separable(band="high")
    ev = c.to_evidence(_raw(0.95), band="low")
    assert ev.llr == 0.0
    assert ev.abstained and ev.reason == "uncalibrated_for_band"


def test_llr_is_finite_at_the_extremes():
    """Clipping must prevent infinite evidence from one detector."""
    c = _fit_separable()
    for s in (0.0, 1.0):
        ev = c.to_evidence(_raw(s), band="high")
        assert math.isfinite(ev.llr)


def test_separate_bands_are_calibrated_separately():
    rng = np.random.default_rng(1)
    n = 200
    # In 'high' the detector separates; in 'low' it is pure noise.
    hi_s = np.concatenate([rng.uniform(0, .4, n), rng.uniform(.6, 1, n)])
    lo_s = np.concatenate([rng.uniform(0, 1, n), rng.uniform(0, 1, n)])
    labels = np.concatenate([np.zeros(n), np.ones(n)])
    c = Calibrator(detector="d1")
    c.fit(np.concatenate([hi_s, lo_s]),
          np.concatenate([labels, labels]),
          ["high"] * 2 * n + ["low"] * 2 * n)
    hi = abs(c.to_evidence(_raw(0.95), band="high").llr)
    lo = abs(c.to_evidence(_raw(0.95), band="low").llr)
    assert hi > lo


def test_prior_is_subtracted_so_the_output_is_a_likelihood_ratio():
    """With an imbalanced band, llr must differ from the posterior log-odds
    by exactly log(prior/(1-prior)).

    Without this subtraction the quantity is a posterior, and the training
    set's base rate rides along into production, where the fraud rate is
    nothing like a balanced set's. This test uses deliberately imbalanced
    data (270 real, 30 fake → prior = 0.10) to force the prior offset away
    from zero.
    """
    rng = np.random.default_rng(0)
    n_real, n_fake = 270, 30
    scores = np.concatenate([rng.uniform(0.0, 0.4, n_real),
                             rng.uniform(0.6, 1.0, n_fake)])
    labels = np.concatenate([np.zeros(n_real), np.ones(n_fake)])
    c = Calibrator(detector="d").fit(scores, labels, ["high"] * len(scores))

    prior = n_fake / (n_real + n_fake)
    expected_offset = math.log(prior / (1 - prior))
    post = float(c._models["high"].decision_function([[0.95]])[0])
    got = c.to_evidence(_raw(0.95), band="high").llr

    assert got == pytest.approx(post - expected_offset, abs=1e-9)
    assert expected_offset != 0.0, "test is vacuous unless the prior is imbalanced"


def test_llr_is_clipped_to_the_configured_bound():
    """One saturated detector must not dominate the fused posterior.

    Uses a deliberately tiny bound (0.5 nats) so the clip is actually reached;
    a test that never reaches the bound proves nothing about clipping. The last
    assertion is critical: it proves the value is PINNED at the bound, not
    merely happening to sit under it.
    """
    rng = np.random.default_rng(0)
    n = 200
    real = rng.uniform(0.0, 0.4, n)
    fake = rng.uniform(0.6, 1.0, n)
    scores = np.concatenate([real, fake])
    labels = np.concatenate([np.zeros(n), np.ones(n)])
    bands = ["high"] * len(scores)

    c_clipped = Calibrator(detector="d", max_abs_llr=0.5)
    c_clipped.fit(scores, labels, bands)

    # For any score, |llr| must not exceed the bound
    for score in (0.0, 1.0):
        llr = c_clipped.to_evidence(_raw(score), band="high").llr
        assert abs(llr) <= 0.5

    # Critical: the unclipped value for 1.0 must exceed 0.5, proving the clip
    # is actually active on the test data. If unclipped <= 0.5, the test is
    # vacuous (no clip is reached).
    c_unclipped = Calibrator(detector="d", max_abs_llr=10.0)
    c_unclipped.fit(scores, labels, bands)
    unclipped = c_unclipped.to_evidence(_raw(1.0), band="high").llr
    assert abs(unclipped) > 0.5, f"test data does not reach the clip bound; unclipped value {unclipped} <= 0.5"
    # The clipped value should be pinned exactly at the bound
    clipped = c_clipped.to_evidence(_raw(1.0), band="high").llr
    assert clipped == pytest.approx(0.5, abs=1e-9)


def test_too_few_samples_per_band_yields_uncalibrated():
    """MIN_FIT_SAMPLES guard: a band with too few samples is not fitted,
    and querying it yields llr=0.0 with uncalibrated_for_band reason."""
    c = Calibrator(detector="d")
    # Fit 'high' with enough data (200 per class)
    rng = np.random.default_rng(0)
    high_scores = np.concatenate([rng.uniform(0.0, 0.4, 200),
                                  rng.uniform(0.6, 1.0, 200)])
    high_labels = np.concatenate([np.zeros(200), np.ones(200)])
    # 'low' has only 5 samples total (< MIN_FIT_SAMPLES = 20)
    low_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.5])
    low_labels = np.array([0, 0, 1, 1, 0])

    c.fit(
        np.concatenate([high_scores, low_scores]),
        np.concatenate([high_labels, low_labels]),
        ["high"] * len(high_scores) + ["low"] * len(low_scores)
    )

    # 'high' is fitted
    ev_high = c.to_evidence(_raw(0.95), band="high")
    assert ev_high.llr != 0.0 and not ev_high.abstained

    # 'low' is not fitted due to too few samples
    ev_low = c.to_evidence(_raw(0.5), band="low")
    assert ev_low.llr == 0.0 and ev_low.abstained
    assert ev_low.reason == "uncalibrated_for_band"


def test_single_class_per_band_yields_uncalibrated():
    """Single-class guard: a band with only one class present is not fitted,
    and querying it yields llr=0.0 with uncalibrated_for_band reason."""
    c = Calibrator(detector="d")
    # Fit 'high' with both classes (balanced)
    rng = np.random.default_rng(0)
    high_scores = np.concatenate([rng.uniform(0.0, 0.4, 100),
                                  rng.uniform(0.6, 1.0, 100)])
    high_labels = np.concatenate([np.zeros(100), np.ones(100)])
    # 'medium' has only class 0 (real) present
    med_scores = np.array([0.1, 0.2, 0.15, 0.25, 0.3])
    med_labels = np.array([0, 0, 0, 0, 0])

    c.fit(
        np.concatenate([high_scores, med_scores]),
        np.concatenate([high_labels, med_labels]),
        ["high"] * len(high_scores) + ["medium"] * len(med_scores)
    )

    # 'high' is fitted
    ev_high = c.to_evidence(_raw(0.95), band="high")
    assert ev_high.llr != 0.0 and not ev_high.abstained

    # 'medium' is not fitted due to single class
    ev_med = c.to_evidence(_raw(0.2), band="medium")
    assert ev_med.llr == 0.0 and ev_med.abstained
    assert ev_med.reason == "uncalibrated_for_band"


def test_non_binary_labels_raises_valueerror():
    """Labels must be {0, 1}, not {-1, +1} or other encodings. A common
    mistake in ML literature; this test ensures we catch it early and
    explicitly rather than silently falling through to wrong results."""
    c = Calibrator(detector="d")
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    # {-1, +1} encoding (common in ML literature)
    labels_pm1 = np.array([-1, -1, 1, 1])
    with pytest.raises(ValueError, match="Labels must be binary.*got \\[-1.*1\\]"):
        c.fit(scores, labels_pm1, ["high"] * len(scores))

    # {0, 1, 2} encoding (three classes)
    labels_012 = np.array([0, 0, 1, 2])
    with pytest.raises(ValueError, match="Labels must be binary"):
        c.fit(scores, labels_012, ["high"] * len(scores))
