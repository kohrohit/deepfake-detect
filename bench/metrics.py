"""Metrics that matter for fraud (spec §8.3).

Accuracy is not among them. At a 1-in-10,000 fraud base rate, a detector that
calls everything real is 99.99% accurate and worth nothing. TPR at a fixed,
operationally tolerable FPR is the number that decides whether this ships.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

# Threshold for WARNING when bootstrap drops degenerate draws.
# A drop rate of ~1% already shifts a percentile interval materially.
# INFO is logged at ANY drops; WARNING fires when drop rate exceeds this threshold.
# This matters most in low-fraud regimes where positive groups are sparse and fragile.
# Note: natural degenerate rate at 3-of-100 groups is ~4.7%, so the threshold is
# intentionally placed low to catch drops in the motivating regime.
MAX_DEGENERATE_FRACTION = 0.01


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under ROC curve via Mann-Whitney U statistic.

    Computes the probability that a random positive scores higher than
    a random negative.

    Args:
        scores: Predicted scores (0 to 1 for probabilities, unbounded for logits).
        labels: Binary labels (0 for negative, 1 for positive).

    Returns:
        AUC in [0, 1], or NaN if either class is empty.

    Raises:
        ValueError: If inputs have incompatible shapes.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)

    if s.shape[0] != y.shape[0]:
        msg = f"Shape mismatch: scores {s.shape} vs labels {y.shape}"
        raise ValueError(msg)

    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        logger.warning(
            "Cannot compute AUC: pos=%d, neg=%d", len(pos), len(neg)
        )
        return float("nan")

    # Mann-Whitney: rank positives, subtract minimum possible rank.
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)

    # Average ranks within ties so constant scores give exactly 0.5.
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]

    return float(
        (ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2)
        / (len(pos) * len(neg))
    )


def tpr_at_fpr(
    scores: np.ndarray, labels: np.ndarray, fpr: float
) -> float:
    """Highest TPR achievable without exceeding `fpr` on the negatives.

    Operationally: if we can tolerate flipping 1 in 100 real transactions,
    how many frauds will we catch?

    Caveat: with tied or saturated negative scores, the achieved FPR can fall
    short of the requested value. The error is conservative (TPR is never
    overstated). This quantization matches Reality Defender's observed behaviour,
    making it operationally relevant rather than theoretical.

    Args:
        scores: Predicted scores (higher is more likely positive).
        labels: Binary labels (0 for negative, 1 for positive).
        fpr: Tolerance for false positive rate on negatives, in [0, 1].

    Returns:
        TPR in [0, 1], or NaN if either class is empty.

    Raises:
        ValueError: If fpr not in [0, 1], or shape mismatch.
    """
    if not 0.0 <= fpr <= 1.0:
        msg = f"FPR must be in [0, 1], got {fpr}"
        raise ValueError(msg)

    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)

    if s.shape[0] != y.shape[0]:
        msg = f"Shape mismatch: scores {s.shape} vs labels {y.shape}"
        raise ValueError(msg)

    neg, pos = s[y == 0], s[y == 1]
    if len(neg) == 0 or len(pos) == 0:
        logger.warning(
            "Cannot compute TPR@FPR: pos=%d, neg=%d", len(pos), len(neg)
        )
        return float("nan")

    # Find threshold that lets fpr fraction of negatives pass.
    thr = np.quantile(neg, 1.0 - fpr)
    return float((pos > thr).mean())


def ece(
    probs: np.ndarray, labels: np.ndarray, bins: int = 10
) -> float:
    """Expected calibration error: mean |confidence - accuracy| over bins.

    Measure of whether predicted probabilities match empirical frequencies.
    Divide [0, 1] into bins; within each bin, compute |mean(probs) - mean(labels)|;
    return weighted average by bin size.

    Args:
        probs: Predicted probabilities (MUST be in [0, 1]; NaN and inf raise ValueError).
        labels: Binary labels (0 or 1).
        bins: Number of bins to partition [0, 1] (default 10: standard histogram).

    Returns:
        ECE in [0, 1], or NaN if no samples.

    Raises:
        ValueError: If any probability outside [0, 1] or non-finite, bins < 1,
            or shape mismatch.
    """
    if bins < 1:
        msg = f"bins must be >= 1, got {bins}"
        raise ValueError(msg)

    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=int)

    if p.shape[0] != y.shape[0]:
        msg = f"Shape mismatch: probs {p.shape} vs labels {y.shape}"
        raise ValueError(msg)

    # Validate probability range: no NaN, inf, or out-of-bounds values
    if not np.all(np.isfinite(p)):
        bad_idx = np.where(~np.isfinite(p))[0]
        bad_vals = p[bad_idx]
        msg = (
            f"Probability contains non-finite values at indices {bad_idx.tolist()}: "
            f"{bad_vals.tolist()}"
        )
        raise ValueError(msg)
    if np.any((p < 0.0) | (p > 1.0)):
        bad_idx = np.where((p < 0.0) | (p > 1.0))[0]
        bad_vals = p[bad_idx]
        msg = (
            f"Probability outside [0, 1] at indices {bad_idx.tolist()}: "
            f"{bad_vals.tolist()}"
        )
        raise ValueError(msg)

    if len(p) == 0:
        logger.warning("Cannot compute ECE: no samples")
        return float("nan")

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        # First bin is inclusive on both ends; others exclude lower.
        m = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if not m.any():
            continue
        # Bin contribution: (bin fraction) × |mean prob - mean label|
        total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def bootstrap_ci_by_group(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
    n: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI, resampling GROUPS with replacement (spec §8.2 guard 2).

    Resampling rows instead of videos fabricates precision: 10,000 frames from
    100 videos carry 100 videos' worth of information, not 10,000.

    NOTE: Degenerate resamples (those returning NaN or inf from stat_fn) are dropped.
    The drop count, total, and rate are logged at INFO level. A WARNING is issued if
    the drop rate exceeds ~1%, indicating the CI is conditioned on well-behaved draws
    only and reads optimistically. Even below the warning threshold, dropped draws
    matter in low-fraud regimes where positive groups are scarce — check the INFO
    message to see the actual rate.

    Args:
        scores: Predicted scores (same length as labels and groups).
        labels: Binary labels.
        groups: Group identifier for each sample (e.g., video ID). Samples with
            the same group are resampled together.
        stat_fn: Statistic to compute on each resample. Must accept (scores, labels)
            and return a scalar float.
        n: Number of bootstrap resamples (default 1000: yields ~±3.2% precision at
            percentile level for a true population median).
        seed: Random seed for reproducibility (default 0: allows explicit determinism).
        alpha: Significance level; returns (alpha/2, 1-alpha/2) quantiles.
            (default 0.05: yields 95% confidence interval).

    Returns:
        (lo, hi): Lower and upper bounds of confidence interval.
        Returns (NaN, NaN) if no valid statistics were computed.

    Raises:
        ValueError: If groups has different length than scores/labels, or n < 1.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    g = np.asarray(groups)

    if s.shape[0] != y.shape[0] or s.shape[0] != g.shape[0]:
        msg = (
            f"Shape mismatch: scores {s.shape}, labels {y.shape}, "
            f"groups {g.shape}"
        )
        raise ValueError(msg)

    if n < 1:
        msg = f"n must be >= 1, got {n}"
        raise ValueError(msg)

    uniq = np.unique(g)
    # Map each unique group to indices where it appears.
    index = {u: np.flatnonzero(g == u) for u in uniq}
    rng = np.random.default_rng(seed)

    stats: list[float] = []
    degenerate_count = 0
    for _ in range(n):
        # Resample groups (not rows).
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        # Concatenate all rows of resampled groups.
        idx = np.concatenate([index[d] for d in drawn])
        v = stat_fn(s[idx], y[idx])
        if np.isfinite(v):
            stats.append(float(v))
        else:
            degenerate_count += 1

    degenerate_fraction = degenerate_count / n if n > 0 else 0.0

    # Always log INFO when any draws are dropped; WARNING when rate exceeds threshold.
    if degenerate_count > 0:
        logger.info(
            "Bootstrap dropped %d/%d resamples (%.1f%%) due to degenerate stat_fn returns.",
            degenerate_count, n, degenerate_fraction * 100
        )

    if degenerate_fraction > MAX_DEGENERATE_FRACTION:
        logger.warning(
            "Degenerate rate %.1f%% exceeds threshold (%.0f%%). "
            "CI is conditioned on well-behaved draws and may read optimistically.",
            degenerate_fraction * 100, MAX_DEGENERATE_FRACTION * 100
        )

    if not stats:
        logger.warning(
            "No finite statistics from %d bootstrap resamples", n
        )
        return (float("nan"), float("nan"))

    return (
        float(np.quantile(stats, alpha / 2)),
        float(np.quantile(stats, 1 - alpha / 2)),
    )
