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
