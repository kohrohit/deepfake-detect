import logging
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
    # Use nonzero llr: if filter breaks, this evidence would contribute visibly.
    r = fuse([_ev(9.9, abstained=True, reason="weights_absent")], n_frames=1)
    assert r.verdict is Verdict.INSUFFICIENT_EVIDENCE
    # Only early-return path sets n_contributing=0 and ess=0.0
    assert r.n_contributing == 0 and r.ess == 0.0


def test_empty_evidence_yields_insufficient_evidence():
    r = fuse([], n_frames=1)
    assert r.verdict is Verdict.INSUFFICIENT_EVIDENCE
    # Only early-return path produces these exact values.
    assert r.n_contributing == 0 and r.ess == 0.0


def test_positive_llrs_sum_towards_fake():
    r = fuse([_ev(2.0, "a"), _ev(2.0, "b")], n_frames=1)
    assert r.verdict is Verdict.FAKE and r.llr_total > 0


def test_negative_llrs_sum_towards_real():
    r = fuse([_ev(-2.0, "a"), _ev(-2.0, "b")], n_frames=1)
    assert r.verdict is Verdict.REAL and r.llr_total < 0


def test_abstentions_contribute_nothing():
    # Use nonzero llr for abstained evidence: if filter breaks, this would change the total.
    a = fuse([_ev(2.0, "a")], n_frames=1)
    b = fuse([_ev(2.0, "a"), _ev(9.9, "b", abstained=True, reason="x")], n_frames=1)
    assert a.llr_total == b.llr_total
    assert a.n_contributing == 1 and b.n_contributing == 1


def test_effective_sample_size_is_one_for_perfectly_correlated_frames():
    constant = np.ones(500)
    assert effective_sample_size(constant) == pytest.approx(1.0, abs=0.5)


def test_effective_sample_size_approaches_n_for_independent_frames():
    indep = np.random.default_rng(0).normal(size=500)
    ess = effective_sample_size(indep)
    assert ess > 100


def test_aggregated_detector_path_undiscounted():
    """Real case: single Evidence with n_frames=1 (detector aggregated internally).

    All detectors in this repo compute probs.mean() over frames before emitting
    a score. Calibration was fitted on the aggregated score. The llr is already
    whole-sample. No discount applies.
    """
    r = fuse([_ev(2.0)], n_frames=1)
    assert r.llr_total == pytest.approx(2.0)
    assert r.n_contributing == 1


def test_per_frame_evidence_with_ess_discount():
    """Per-frame evidence list with ESS discount (well below the cap).

    Result must sit comfortably BELOW MAX_TOTAL_LLR, or the assertion is
    blind to bugs. With ess=5.0, n_frames=900, per-frame llr=1.0:
    naive sum = 900, discounted = 900 * (5/900) = 5.0.

    This value is diagnostic, not saturated by the cap:
    - correct linear discount: 900 * (5/900) = 5.0
    - old sqrt formula: 900 * sqrt(5/900) ≈ 67.1 (still below cap)
    - no discount at all: 900 (capped to 20.0)

    If someone changes the numbers back into the saturated region,
    the guard assertion catches it.
    """
    per_frame = [_ev(1.0, f"f{i}") for i in range(900)]
    r = fuse(per_frame, n_frames=900, ess=5.0)
    assert r.llr_total == pytest.approx(5.0, rel=1e-2)
    assert r.llr_total < _MAX_TOTAL, "expected value must not saturate the cap"
    assert 5.0 < 900  # fixture: ess < n_frames


def test_per_frame_evidence_with_ess_equals_one():
    """Pathological case: per-frame list with ess=1 (perfectly correlated).

    900 identical frames, per-frame llr=1.0, ess=1.0 (maximum correlation).
    Naive sum = 900. Discount = 1/900. Result = 1.0 (exactly one frame's worth).
    """
    identical = [_ev(1.0, f"f{i}") for i in range(900)]
    r = fuse(identical, n_frames=900, ess=1.0)
    assert r.llr_total == pytest.approx(1.0)
    assert r.n_contributing == 900


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


def test_effective_sample_size_bounded_by_n_on_anticorrelated_series():
    """Anti-correlated series (e.g. flicker) must not amplify evidence.

    ESS formula on [1, -1, 1, -1, ...] gives rho ≈ -0.999.
    Without ceiling: ESS = n·(1.999/0.001) ≈ 2000n.
    With ceiling: ESS ≤ n.
    """
    anticorr = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(500)])
    ess = effective_sample_size(anticorr)
    assert 1.0 <= ess <= 500.0, f"ESS {ess} not bounded by [1, 500]"


def test_effective_sample_size_n_less_than_3():
    """For n < 3, autocorrelation estimation is meaningless.

    At n=2, any two distinct values give rho=-0.5 exactly, making ESS always 6.0.
    Return n directly to avoid false precision.
    """
    # Multiple different pairs, should all return 2.0, not 6.0
    assert effective_sample_size([5.0, 9.0]) == 2.0
    assert effective_sample_size([1.0, 100.0]) == 2.0
    assert effective_sample_size([-5.0, 3.0]) == 2.0
    # n=1 returns 1.0
    assert effective_sample_size([42.0]) == 1.0
    # n=0 returns 0.0
    assert effective_sample_size([]) == 0.0


def test_ess_clamped_to_n_frames(caplog):
    """ESS > n_frames amplifies instead of discounting; must be clamped.

    With ess=5000, n_frames=900, per-frame llr=1.0:
    naive sum = 900
    unclamped discount: 900 * (5000/900) ≈ 5000 (AMPLIFIES, wrong)
    clamped to n_frames: 900 * (900/900) = 900 -> capped to MAX_TOTAL_LLR=20.0

    Compare with smaller ess: ess=4, n_frames=900, per-frame llr=1.0:
    naive sum = 900, discounted = 900 * (4/900) = 4.0
    """
    per_frame = [_ev(1.0, f"f{i}") for i in range(900)]
    with caplog.at_level(logging.WARNING):
        r_clamped = fuse(per_frame, n_frames=900, ess=5000.0)

    # With ESS clamped to n_frames, discount = 900/900 = 1.0 (no discount)
    # naive sum = 900, capped to MAX_TOTAL_LLR = 20.0
    assert r_clamped.llr_total == pytest.approx(_MAX_TOTAL)

    # Warning should have been logged for the clamp
    assert "exceeds n_frames" in caplog.text
    assert "clamping" in caplog.text

    # Verify that smaller ess gives smaller result (discount applies normally)
    r_normal = fuse(per_frame, n_frames=900, ess=4.0)
    assert r_normal.llr_total == pytest.approx(4.0, rel=1e-2)
    assert r_normal.llr_total < r_clamped.llr_total


def test_n_frames_validation():
    """n_frames < 1 is invalid and raises ValueError."""
    with pytest.raises(ValueError, match="n_frames must be >= 1"):
        fuse([_ev(1.0)], n_frames=0)

    with pytest.raises(ValueError, match="n_frames must be >= 1"):
        fuse([_ev(1.0)], n_frames=-5)
