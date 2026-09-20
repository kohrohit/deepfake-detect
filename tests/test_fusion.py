import math
import numpy as np
import pytest
from dfd.fusion import FusedResult, effective_sample_size, fuse
from dfd.types import Evidence, Verdict


def _ev(llr, detector="d", abstained=False, reason="ok"):
    return Evidence(detector=detector, detector_version="1", llr=llr,
                    raw_score=None, uncertainty=0.0,
                    abstained=abstained, reason=reason)


def test_all_abstentions_yield_insufficient_evidence():
    r = fuse([_ev(0.0, abstained=True, reason="weights_absent")], n_frames=1)
    assert r.verdict is Verdict.INSUFFICIENT_EVIDENCE


def test_empty_evidence_yields_insufficient_evidence():
    assert fuse([], n_frames=1).verdict is Verdict.INSUFFICIENT_EVIDENCE


def test_positive_llrs_sum_towards_fake():
    r = fuse([_ev(2.0, "a"), _ev(2.0, "b")], n_frames=1)
    assert r.verdict is Verdict.FAKE and r.llr_total > 0


def test_negative_llrs_sum_towards_real():
    r = fuse([_ev(-2.0, "a"), _ev(-2.0, "b")], n_frames=1)
    assert r.verdict is Verdict.REAL and r.llr_total < 0


def test_abstentions_contribute_nothing():
    a = fuse([_ev(2.0, "a")], n_frames=1)
    b = fuse([_ev(2.0, "a"), _ev(0.0, "b", abstained=True, reason="x")], n_frames=1)
    assert a.llr_total == b.llr_total


def test_effective_sample_size_is_one_for_perfectly_correlated_frames():
    constant = np.ones(500)
    assert effective_sample_size(constant) == pytest.approx(1.0, abs=0.5)


def test_effective_sample_size_approaches_n_for_independent_frames():
    indep = np.random.default_rng(0).normal(size=500)
    ess = effective_sample_size(indep)
    assert ess > 100


def test_correlated_frames_do_not_produce_runaway_confidence():
    """900 correlated frames must not yield 900x the evidence of one frame."""
    one = fuse([_ev(1.0, "a")], n_frames=1, ess=1.0)
    many = fuse([_ev(1.0, "a")], n_frames=900, ess=20.0)
    assert many.llr_total < one.llr_total * 900
    assert many.llr_total <= _MAX_TOTAL


_MAX_TOTAL = 20.0


def test_total_llr_is_capped():
    r = fuse([_ev(6.0, f"d{i}") for i in range(20)], n_frames=1)
    assert abs(r.llr_total) <= _MAX_TOTAL


def test_disagreement_is_recorded_as_a_feature():
    """Spec §7.2: disagreement is signal, not noise to be averaged away."""
    r = fuse([_ev(3.0, "a"), _ev(-3.0, "b")], n_frames=1)
    assert r.disagreement > 0
    assert r.verdict is Verdict.OUT_OF_DISTRIBUTION


def test_reasons_are_carried_for_audit():
    r = fuse([_ev(0.0, "a", abstained=True, reason="below_quality_floor")], n_frames=1)
    assert "below_quality_floor" in r.reasons.values()
