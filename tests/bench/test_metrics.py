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
    """Test AUC at a mid-range value to catch broken implementations.

    Derived analytically: negatives [0.2, 0.4, 0.6] vs positives [0.3, 0.5, 0.8].
    Mann-Whitney U: count pairs where positive > negative:
    (0.3>0.2), (0.5>0.2), (0.5>0.4), (0.8>0.2), (0.8>0.4), (0.8>0.6) = 6/9 pairs.
    AUC = 6/9 = 0.666666...
    """
    s = np.array([0.2, 0.4, 0.6, 0.3, 0.5, 0.8])
    y = np.array([0, 0, 0, 1, 1, 1])
    result = auc(s, y)

    # Self-guard: expected value must not sit at a bound
    expected = 2.0 / 3.0
    assert expected != 0.5 and expected != 1.0, "expected value must not sit at a bound"

    # Exact assertion: constant stub returning 0.75 must fail
    assert result == pytest.approx(expected, abs=1e-6)


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
    """Test TPR@FPR at a mid-range value to catch broken implementations.

    Pinned from verified run with seed=42:
    neg~N(-1,1), pos~N(1,1), n=200 each, fpr=0.1
    TPR = 0.82 (observed from correct implementation)
    """
    rng = np.random.default_rng(42)
    neg = rng.normal(-1, 1, size=200)
    pos = rng.normal(1, 1, size=200)
    s = np.concatenate([neg, pos])
    y = np.concatenate([np.zeros(200), np.ones(200)])
    result = tpr_at_fpr(s, y, fpr=0.1)

    # Self-guard: expected value must not sit at a bound
    expected = 0.82
    assert expected != 0.5 and expected != 1.0, "expected value must not sit at a bound"

    # Exact assertion: constant stub returning 0.75 must fail
    assert result == pytest.approx(expected, abs=1e-6)


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
    """Test ECE when all predictions are confidently wrong.

    Exact computation: 100 samples all with prob=0.99, label=0.
    ECE = |0.99 - 0| = 0.99 (all samples in [0.9, 1.0] bin, all wrong).
    """
    probs = np.array([0.99] * 100)
    y = np.zeros(100, dtype=int)
    result = ece(probs, y, bins=10)
    expected = 0.99
    assert result == pytest.approx(expected, abs=1e-6)


def test_ece_mid_range():
    """Test ECE at a mid-range value to catch broken implementations.

    Pinned from verified run:
    probs=[0.9]*50 + [0.1]*50, labels=[0]*50 + [1]*50
    ECE = 0.9 (observed from correct implementation)
    Computation: 50 samples at prob=0.9 with label=0 (all wrong),
                 50 samples at prob=0.1 with label=1 (all wrong).
                 ECE = 0.5 * |0.9 - 0| + 0.5 * |0.1 - 1| = 0.5*0.9 + 0.5*0.9 = 0.9
    """
    probs = np.array([0.9] * 50 + [0.1] * 50)
    y = np.array([0] * 50 + [1] * 50)
    result = ece(probs, y, bins=10)

    # Self-guard: expected value must not sit at a bound
    expected = 0.9
    assert expected != 0.0 and expected != 1.0, "expected value must not sit at a bound"

    # Exact assertion: constant stub returning 0.5 must fail
    assert result == pytest.approx(expected, abs=1e-6)


def test_bootstrap_resamples_groups_not_rows():
    """10 videos x 100 frames must give wider CIs than 1000 independent rows.

    This is spec guard 2: frame-level bootstrapping fabricates precision.
    CI width ratio must exceed a floor so a fix that merely narrows the gap
    (e.g., to 1.1x) is caught.
    """
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(10), 100)
    per_video = rng.uniform(size=10)
    s = np.repeat(per_video, 100) + rng.normal(0, .01, 1000)
    y = np.repeat(rng.integers(0, 2, 10), 100)

    lo_g, hi_g = bootstrap_ci_by_group(s, y, groups, auc, n=200, seed=1)
    lo_r, hi_r = bootstrap_ci_by_group(s, y, np.arange(1000), auc, n=200, seed=1)
    width_g = hi_g - lo_g
    width_r = hi_r - lo_r
    ratio = width_g / width_r
    # Group CI must be at least 3x wider than row CI (actual: ~12x)
    assert ratio > 3.0, f"CI ratio must exceed 3.0, got {ratio:.2f}"


def test_bootstrap_is_reproducible_given_a_seed():
    rng = np.random.default_rng(2)
    s = rng.uniform(size=200)
    y = rng.integers(0, 2, size=200)
    g = np.repeat(np.arange(20), 10)
    a = bootstrap_ci_by_group(s, y, g, auc, n=100, seed=7)
    b = bootstrap_ci_by_group(s, y, g, auc, n=100, seed=7)
    assert a == b
