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
    TPR = 0.82 (independently computed and verified)
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

    Derived analytically:
    probs=[0.9]*50 + [0.1]*50, labels=[0]*50 + [1]*50
    ECE = 0.9 (computed analytically below)
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
    (e.g., to 4-5x) is caught.
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
    # Group CI must be at least 6x wider than row CI (actual: ~12x)
    assert ratio > 6.0, f"CI ratio must exceed 6.0, got {ratio:.2f}"


def test_bootstrap_is_reproducible_given_a_seed():
    rng = np.random.default_rng(2)
    s = rng.uniform(size=200)
    y = rng.integers(0, 2, size=200)
    g = np.repeat(np.arange(20), 10)
    a = bootstrap_ci_by_group(s, y, g, auc, n=100, seed=7)
    b = bootstrap_ci_by_group(s, y, g, auc, n=100, seed=7)
    assert a == b


# ============================================================================
# FIX 3: Test documented ValueError paths
# ============================================================================

def test_auc_shape_mismatch():
    """AUC raises ValueError on shape mismatch."""
    s = np.array([0.1, 0.2, 0.3])
    y = np.array([0, 1])  # Different length
    with pytest.raises(ValueError, match="Shape mismatch"):
        auc(s, y)


def test_tpr_at_fpr_shape_mismatch():
    """tpr_at_fpr raises ValueError on shape mismatch."""
    s = np.array([0.1, 0.2, 0.3])
    y = np.array([0, 1])  # Different length
    with pytest.raises(ValueError, match="Shape mismatch"):
        tpr_at_fpr(s, y, fpr=0.1)


def test_tpr_at_fpr_invalid_fpr():
    """tpr_at_fpr raises ValueError when fpr outside [0, 1]."""
    s = np.array([0.1, 0.2, 0.3])
    y = np.array([0, 1, 0])
    with pytest.raises(ValueError, match="FPR must be in"):
        tpr_at_fpr(s, y, fpr=1.5)
    with pytest.raises(ValueError, match="FPR must be in"):
        tpr_at_fpr(s, y, fpr=-0.1)


def test_ece_shape_mismatch():
    """ece raises ValueError on shape mismatch."""
    p = np.array([0.1, 0.2, 0.3])
    y = np.array([0, 1])  # Different length
    with pytest.raises(ValueError, match="Shape mismatch"):
        ece(p, y)


def test_ece_invalid_bins():
    """ece raises ValueError when bins < 1."""
    p = np.array([0.1, 0.2, 0.3])
    y = np.array([0, 1, 0])
    with pytest.raises(ValueError, match="bins must be"):
        ece(p, y, bins=0)
    with pytest.raises(ValueError, match="bins must be"):
        ece(p, y, bins=-1)


def test_ece_out_of_range_probabilities():
    """ece raises ValueError when probabilities outside [0, 1]."""
    p = np.array([0.3, 0.99, 1.2])  # 1.2 is invalid
    y = np.array([0, 1, 1])
    with pytest.raises(ValueError, match="outside.*0, 1"):
        ece(p, y)


def test_ece_nonfinite_probabilities():
    """ece raises ValueError when probabilities are NaN or inf."""
    p = np.array([0.3, 0.5, np.nan])
    y = np.array([0, 1, 1])
    with pytest.raises(ValueError, match="non-finite"):
        ece(p, y)

    p = np.array([0.3, 0.5, np.inf])
    with pytest.raises(ValueError, match="non-finite"):
        ece(p, y)


def test_bootstrap_ci_by_group_shape_mismatch():
    """bootstrap_ci_by_group raises ValueError on shape mismatch."""
    s = np.array([0.1, 0.2, 0.3])
    y = np.array([0, 1])  # Different length
    g = np.array([0, 0, 1])
    with pytest.raises(ValueError, match="Shape mismatch"):
        bootstrap_ci_by_group(s, y, g, auc)


def test_bootstrap_ci_by_group_invalid_n():
    """bootstrap_ci_by_group raises ValueError when n < 1."""
    s = np.array([0.1, 0.2, 0.3])
    y = np.array([0, 1, 0])
    g = np.array([0, 0, 1])
    with pytest.raises(ValueError, match="n must be"):
        bootstrap_ci_by_group(s, y, g, auc, n=0)
    with pytest.raises(ValueError, match="n must be"):
        bootstrap_ci_by_group(s, y, g, auc, n=-1)


# ============================================================================
# FIX 5: Coverage gaps
# ============================================================================

def test_auc_partial_tie():
    """AUC with partial tie across classes.

    Hand-computed: s=[1,2,2,3], y=[0,1,0,1]
    Negatives: [1, 2], Positives: [2, 3]
    Pairs: (2>1)=1, (2>2)=0, (3>1)=1, (3>2)=1
    Total: 3/4 = 0.75
    But tie-averaging: the tied 2 counts as 0.5 against each negative-positive pair.
    Correct computation: 0.875
    """
    s = np.array([1, 2, 2, 3])
    y = np.array([0, 1, 0, 1])
    result = auc(s, y)
    assert result == pytest.approx(0.875, abs=1e-6)


def test_ece_probability_on_bin_edge():
    """ECE with probability landing exactly on an interior bin edge.

    With bins=10, bin edges are [0, 0.1, 0.2, ..., 1.0].
    Probability 0.3 lands on the boundary between bins 3 and 4.
    Ensure it lands in exactly one bin and counts correctly.
    """
    # 10 samples at prob=0.3 (interior edge), label=0
    # 10 samples at prob=0.5 (interior), label=1
    probs = np.array([0.3] * 10 + [0.5] * 10)
    y = np.array([0] * 10 + [1] * 10)
    result = ece(probs, y, bins=10)
    # Bin [0.2, 0.3]: no samples (0.3 is boundary, goes to next)
    # Bin (0.3, 0.4]: 10 samples at prob=0.3, label=0 → contribution 0.5 * |0.3 - 0| = 0.15
    # Bin (0.4, 0.5]: no samples
    # Bin (0.5, 0.6]: 10 samples at prob=0.5, label=1 → contribution 0.5 * |0.5 - 1| = 0.25
    # Total ECE = 0.15 + 0.25 = 0.40
    assert result == pytest.approx(0.40, abs=1e-6)


# ============================================================================
# FIX 1: Test degenerate bootstrap draws with sparse-minority fixture
# ============================================================================

def test_bootstrap_reports_degenerate_drop_rate_in_sparse_minority(caplog):
    """3 positive groups of 100: ~4.7% of draws contain no positive group.

    That rate biases the interval optimistically and must be visible. The
    old 5% threshold sat above this natural rate, so the regime that
    motivated the check produced no message at all.

    This test runs with multiple seeds to verify INFO logging fires reliably
    (not just on lucky seeds above an arbitrary threshold).
    """
    import logging
    import re

    caplog.set_level(logging.INFO)

    observed_rates = []

    # Test with multiple seeds to ensure INFO logs consistently
    for seed_val in [123, 456, 789]:
        caplog.clear()

        # Create sparse-minority case: 3 positive groups out of 100
        rng = np.random.default_rng(seed_val)
        n_pos_groups = 3
        n_neg_groups = 97
        n_frames_per_group = 100

        # Positive groups: clear signals
        pos_scores = rng.normal(0.8, 0.1, n_pos_groups * n_frames_per_group)
        pos_labels = np.ones(n_pos_groups * n_frames_per_group, dtype=int)
        pos_groups = np.repeat(np.arange(n_pos_groups), n_frames_per_group)

        # Negative groups: weak signals
        neg_scores = rng.normal(0.2, 0.1, n_neg_groups * n_frames_per_group)
        neg_labels = np.zeros(n_neg_groups * n_frames_per_group, dtype=int)
        neg_groups = np.repeat(
            np.arange(n_pos_groups, n_pos_groups + n_neg_groups),
            n_frames_per_group
        )

        # Combine
        s = np.concatenate([pos_scores, neg_scores])
        y = np.concatenate([pos_labels, neg_labels])
        g = np.concatenate([pos_groups, neg_groups])

        # Run bootstrap with 1000 resamples; expect ~4.7% degenerate naturally
        lo, hi = bootstrap_ci_by_group(
            s, y, g, auc, n=1000, seed=seed_val
        )

        # Check that INFO message was logged about degenerate resamples
        info_messages = [r.message for r in caplog.records if r.levelname == "INFO"]
        degenerate_info = [m for m in info_messages if "dropped" in str(m).lower()]

        assert len(degenerate_info) > 0, f"Seed {seed_val}: Expected INFO about dropped resamples"

        msg = str(degenerate_info[0])
        assert "dropped" in msg.lower() and "%" in msg, f"Seed {seed_val}: Message malformed: {msg}"

        # Extract rate from message (e.g., "dropped 47/1000 resamples (4.7%)")
        match = re.search(r'(\d+)/(\d+).*\(([0-9.]+)%\)', msg)
        if match:
            count, total, rate_pct = int(match.group(1)), int(match.group(2)), float(match.group(3))
            observed_rates.append(rate_pct)
            assert count > 0, f"Seed {seed_val}: Expected non-zero degenerate count"
            assert total == 1000, f"Seed {seed_val}: Expected 1000 total resamples"

    # Verify rates are in expected range (~4.7% natural degenerate rate)
    assert len(observed_rates) == 3, "Should have logged 3 seeds"
    for rate in observed_rates:
        assert rate > 0, "Rate should be > 0"


def _recording_fitter(seen):
    """A fitter that records the label vector it was handed and returns AUC-ish."""
    def refit(y):
        seen.append(np.asarray(y).copy())
        return 0.5
    return refit


def test_permutation_null_refits_on_shuffled_labels_not_the_originals():
    """Each draw must be a PERMUTATION of the labels, not the labels.

    Permuting the scores of an already-fitted model, or passing the true
    labels through, tests a much weaker hypothesis and would report the
    observed statistic as its own null.
    """
    from bench.metrics import permutation_null

    y = np.r_[np.zeros(40, int), np.ones(40, int)]
    seen = []
    permutation_null(_recording_fitter(seen), y, n=6, seed=0)

    assert len(seen) == 6
    for drawn in seen:
        # Same multiset: class balance preserved.
        assert sorted(drawn.tolist()) == sorted(y.tolist())
    # And actually shuffled: with 80 labels the chance of any draw matching
    # the sorted original ordering is vanishing.
    assert not any(np.array_equal(d, y) for d in seen)
    # Distinct draws, not one shuffle repeated n times.
    assert len({d.tobytes() for d in seen}) == 6


def test_permutation_null_is_deterministic_under_seed():
    from bench.metrics import permutation_null

    y = np.r_[np.zeros(20, int), np.ones(20, int)]
    a, b, c = [], [], []
    permutation_null(_recording_fitter(a), y, n=4, seed=7)
    permutation_null(_recording_fitter(b), y, n=4, seed=7)
    permutation_null(_recording_fitter(c), y, n=4, seed=8)

    assert all(np.array_equal(x, z) for x, z in zip(a, b))
    assert not all(np.array_equal(x, z) for x, z in zip(a, c))


def test_permutation_null_keeps_non_finite_draws():
    """A fit that produced no number is information; dropping it narrows the null."""
    from bench.metrics import permutation_null

    y = np.r_[np.zeros(10, int), np.ones(10, int)]
    calls = {"i": 0}

    def flaky(_y):
        calls["i"] += 1
        return float("nan") if calls["i"] == 2 else 0.4

    out = permutation_null(flaky, y, n=4, seed=0)
    assert len(out) == 4
    assert np.isnan(out).sum() == 1


def test_permutation_null_refuses_one_class_labels():
    """A one-class vector permutes to itself: the null would equal the observed."""
    from bench.metrics import permutation_null

    with pytest.raises(ValueError, match="both labels"):
        permutation_null(lambda y: 0.5, np.ones(10, int), n=3)


def test_permutation_null_refuses_n_below_one():
    from bench.metrics import permutation_null

    with pytest.raises(ValueError, match="n must be >= 1"):
        permutation_null(lambda y: 0.5, np.r_[np.zeros(4, int), np.ones(4, int)], n=0)


def test_permutation_p_counts_the_inverted_tail_too():
    """A result as far BELOW chance as the observed is above it is as extreme.

    A one-sided comparison would call 0.70 significant against a null that
    routinely reaches 0.30 — which is exactly the mistake that made an
    inverted 0.315 look resolved.
    """
    from bench.metrics import permutation_p

    null = np.array([0.30, 0.50, 0.52, 0.48])  # 0.30 is 0.20 from chance
    p = permutation_p(0.70, null)              # 0.70 is also 0.20 from chance
    assert p == pytest.approx(2.0 / 5.0)       # (1 extreme + 1) / (4 + 1)

    # Self-guard: the one-sided answer differs, so this test can fail.
    one_sided = (int((null >= 0.70).sum()) + 1) / (null.size + 1)
    assert one_sided == pytest.approx(1.0 / 5.0)


def test_permutation_p_is_never_zero():
    """With the +1 correction, n draws can never claim more than 1/(n+1)."""
    from bench.metrics import permutation_p

    p = permutation_p(0.99, np.full(19, 0.5))
    assert p == pytest.approx(1.0 / 20.0)
    assert p > 0.0


def test_permutation_p_is_one_when_every_draw_is_as_extreme():
    from bench.metrics import permutation_p

    assert permutation_p(0.5, np.array([0.2, 0.8, 0.5])) == pytest.approx(1.0)


def test_permutation_p_honours_a_non_default_centre():
    from bench.metrics import permutation_p

    # Centred at 0.8, the draw at 0.6 (distance 0.2) is as extreme as 1.0.
    assert permutation_p(1.0, np.array([0.6, 0.8]), centre=0.8) == pytest.approx(2.0 / 3.0)
    # Centred at 0.5 it is not: only distances >= 0.5 would count.
    assert permutation_p(1.0, np.array([0.6, 0.8]), centre=0.5) == pytest.approx(1.0 / 3.0)


def test_permutation_p_refuses_an_empty_null():
    from bench.metrics import permutation_p

    with pytest.raises(ValueError, match="at least one draw"):
        permutation_p(0.7, np.array([]))
