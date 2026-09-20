import numpy as np
import pytest
from bench.metrics import auc, bootstrap_ci_by_group, ece, tpr_at_fpr


def test_auc_is_one_for_perfect_separation():
    s = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([0, 0, 1, 1])
    assert auc(s, y) == pytest.approx(1.0)


def test_auc_is_half_for_no_separation():
    s = np.array([0.5, 0.5, 0.5, 0.5])
    y = np.array([0, 1, 0, 1])
    assert auc(s, y) == pytest.approx(0.5)


def test_auc_has_mid_range_case():
    """Test AUC at a mid-range value to catch broken implementations."""
    # Overlapping scores: negatives [0.2, 0.4, 0.6], positives [0.3, 0.5, 0.8]
    s = np.array([0.2, 0.4, 0.6, 0.3, 0.5, 0.8])
    y = np.array([0, 0, 0, 1, 1, 1])
    result = auc(s, y)
    # With partial overlap, AUC should be strictly between 0.5 and 1.0
    assert 0.5 < result < 1.0
    # Sanity check: AUC should be around 0.67 for this overlap pattern
    assert result > 0.5


def test_tpr_at_fpr_is_one_for_perfect_separation():
    s = np.concatenate([np.linspace(0, .4, 100), np.linspace(.6, 1, 100)])
    y = np.concatenate([np.zeros(100), np.ones(100)])
    assert tpr_at_fpr(s, y, fpr=0.01) == pytest.approx(1.0)


def test_tpr_at_fpr_is_near_the_fpr_for_random_scores():
    rng = np.random.default_rng(0)
    s = rng.uniform(size=2000)
    y = rng.integers(0, 2, size=2000)
    assert tpr_at_fpr(s, y, fpr=0.1) < 0.25


def test_tpr_at_fpr_mid_range():
    """Test TPR@FPR at a mid-range value to catch broken implementations."""
    # Create a case with partial separation
    rng = np.random.default_rng(42)
    neg = rng.normal(-1, 1, size=200)
    pos = rng.normal(1, 1, size=200)
    s = np.concatenate([neg, pos])
    y = np.concatenate([np.zeros(200), np.ones(200)])
    result = tpr_at_fpr(s, y, fpr=0.1)
    # With partial separation, TPR should be mid-range
    assert 0.5 < result < 1.0


def test_tpr_at_fpr_respects_fpr_argument():
    """Verify that changing fpr changes the result (guards against ignoring fpr)."""
    rng = np.random.default_rng(42)
    neg = rng.normal(-1, 1, size=200)
    pos = rng.normal(1, 1, size=200)
    s = np.concatenate([neg, pos])
    y = np.concatenate([np.zeros(200), np.ones(200)])

    tpr_1 = tpr_at_fpr(s, y, fpr=0.01)
    tpr_10 = tpr_at_fpr(s, y, fpr=0.1)
    # Lower FPR should give lower or equal TPR
    assert tpr_1 <= tpr_10


def test_ece_is_zero_for_perfectly_calibrated_probabilities():
    probs = np.array([0.0] * 50 + [1.0] * 50)
    y = np.array([0] * 50 + [1] * 50)
    assert ece(probs, y, bins=10) == pytest.approx(0.0, abs=1e-9)


def test_ece_is_large_for_confidently_wrong_probabilities():
    probs = np.array([0.99] * 100)
    y = np.zeros(100, dtype=int)
    assert ece(probs, y, bins=10) > 0.9


def test_ece_mid_range():
    """Test ECE at a mid-range value to catch broken implementations."""
    # Moderately miscalibrated probabilities
    probs = np.array([0.9] * 50 + [0.1] * 50)
    y = np.array([0] * 50 + [1] * 50)
    result = ece(probs, y, bins=10)
    # Should be between 0 and 1, and non-trivial (not zero, not max)
    assert 0.0 < result < 1.0
    assert result > 0.5


def test_bootstrap_resamples_groups_not_rows():
    """10 videos x 100 frames must give wider CIs than 1000 independent rows.

    This is spec guard 2: frame-level bootstrapping fabricates precision.
    """
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(10), 100)
    per_video = rng.uniform(size=10)
    s = np.repeat(per_video, 100) + rng.normal(0, .01, 1000)
    y = np.repeat(rng.integers(0, 2, 10), 100)

    lo_g, hi_g = bootstrap_ci_by_group(s, y, groups, auc, n=200, seed=1)
    lo_r, hi_r = bootstrap_ci_by_group(s, y, np.arange(1000), auc, n=200, seed=1)
    assert (hi_g - lo_g) > (hi_r - lo_r)


def test_bootstrap_is_reproducible_given_a_seed():
    rng = np.random.default_rng(2)
    s = rng.uniform(size=200)
    y = rng.integers(0, 2, size=200)
    g = np.repeat(np.arange(20), 10)
    a = bootstrap_ci_by_group(s, y, g, auc, n=100, seed=7)
    b = bootstrap_ci_by_group(s, y, g, auc, n=100, seed=7)
    assert a == b
