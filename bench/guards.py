"""The six evaluation-hygiene guards (spec §8.2).

Each raises rather than warns. A benchmark that can be silently run dirty will
be run dirty, and every one of these failures inflates results in the flattering
direction — which is exactly why they are easy to leave out.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


class GuardViolation(Exception):
    """Raised when an evaluation-hygiene guard fails."""


@dataclass(frozen=True)
class IdentityReport:
    """Report from identity leakage guard with measured similarity metrics."""
    n_train: int
    n_test: int
    max_similarity: float
    violations: int
    threshold: float


@dataclass(frozen=True)
class ParityReport:
    """Report from demographic parity guard with per-stratum error rates."""
    fpr_by_stratum: dict
    tpr_by_stratum: dict
    max_fpr_ratio: float
    ceiling: float


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 if either vector has zero norm.
    """
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def check_identity_disjoint(train_ids, test_ids, embeddings: dict,
                            threshold: float = 0.6) -> IdentityReport:
    """Guard 1 — identity leakage.

    The same person in train and test teaches the model faces, not forgery.
    Returns a measured report; spec acceptance criterion 2 requires the number,
    not an assertion that it was checked.

    Raises:
        GuardViolation: If any train/test pair has cosine similarity >= threshold.
    """
    max_sim = 0.0
    violations = 0
    for a in train_ids:
        for b in test_ids:
            if a not in embeddings or b not in embeddings:
                continue
            sim = _cosine(embeddings[a], embeddings[b])
            max_sim = max(max_sim, sim)
            if sim >= threshold:
                violations += 1
    if violations:
        raise GuardViolation(
            f"identity leakage: %d train/test pairs at cosine >= %f "
            f"(max %.4f)" % (violations, threshold, max_sim))
    return IdentityReport(n_train=len(train_ids), n_test=len(test_ids),
                          max_similarity=max_sim, violations=0,
                          threshold=threshold)


def check_video_level(sample_ids, groups) -> None:
    """Guard 2 — one sample per source video.

    Frames must be aggregated before scoring. 10,000 frames from 100 videos is
    100 independent samples; treating them as 10,000 inflates AUC and shrinks
    confidence intervals dishonestly.

    Raises:
        GuardViolation: If a video appears in multiple samples.
    """
    seen: dict = {}
    for sid, g in zip(sample_ids, groups):
        if g in seen and seen[g] != sid:
            raise GuardViolation(
                f"group %r appears in multiple samples (%r, %r); "
                "aggregate to video level before scoring" % (g, seen[g], sid))
        seen[g] = sid


def check_compression_coverage(records, required=("c0", "c23", "c40")) -> None:
    """Guard 3 — evaluate across compression levels, report the worst.

    Raises:
        GuardViolation: If any required compression level is missing.
    """
    present = {r.get("compression") for r in records}
    missing = [c for c in required if c not in present]
    if missing:
        raise GuardViolation(
            f"compression levels missing from evaluation: %s" % (
                ', '.join(missing)))


def check_uniform_preprocessing(records) -> None:
    """Guard 4 — one preprocessing pipeline, applied blind to label.

    Different face detectors on real vs fake is itself a giveaway the model will
    happily learn.

    Raises:
        GuardViolation: If preprocessing pipeline parameters vary.
    """
    for key in ("face_detector", "align"):
        by_label: dict = {}
        for r in records:
            by_label.setdefault(r["label"], set()).add(r.get(key))
        all_values = set().union(*by_label.values()) if by_label else set()
        if len(all_values) > 1:
            raise GuardViolation(
                f"non-uniform preprocessing: %s takes values %s" % (
                    key, sorted(map(str, all_values))))


def check_threshold_provenance(source: str) -> None:
    """Guard 5 — operating points frozen on validation, never chosen on test.

    Raises:
        GuardViolation: If threshold source is not 'validation'.
    """
    if source != "validation":
        raise GuardViolation(
            f"threshold source must be 'validation', got %r" % source)


def check_demographic_parity(scores, labels, strata, threshold: float,
                             max_fpr_ratio: float = 2.0) -> ParityReport:
    """Guard 6 — per-stratum error parity (spec §8.2 guard 6).

    An aggregate FPR of 1% is compatible with 0.3% on one group and 4% on
    another. The applicants wrongly rejected are not distributed evenly, and
    the aggregate is precisely the statistic that conceals it.

    Reports the spread and fails when the inter-stratum FPR ratio exceeds the
    ceiling. Strata with no genuine (negative) samples have no measurable FPR
    and are excluded rather than assumed clean.

    Raises:
        GuardViolation: If FPR ratio between any two strata exceeds max_fpr_ratio.

    Returns:
        ParityReport with per-stratum FPR/TPR rates and computed ratio.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    g = np.asarray(strata)

    fpr: dict[str, float] = {}
    tpr: dict[str, float] = {}
    for stratum in np.unique(g):
        m = g == stratum
        neg = m & (y == 0)
        pos = m & (y == 1)
        if neg.sum() > 0:
            fpr[str(stratum)] = float((s[neg] > threshold).mean())
        if pos.sum() > 0:
            tpr[str(stratum)] = float((s[pos] > threshold).mean())

    ratio = 1.0
    if len(fpr) >= 2:
        values = [v for v in fpr.values()]
        lo = min(values)
        hi = max(values)
        # An absolute floor keeps a 0%-vs-0.5% difference from reading as infinite.
        # FLOOR_FOR_RATIO: 1e-3 prevents division by near-zero FPR values
        ratio = hi / max(lo, 1e-3)

    report = ParityReport(fpr_by_stratum=fpr, tpr_by_stratum=tpr,
                          max_fpr_ratio=float(ratio), ceiling=max_fpr_ratio)
    if ratio > max_fpr_ratio:
        worst = max(fpr, key=fpr.get)
        best = min(fpr, key=fpr.get)
        raise GuardViolation(
            f"demographic FPR disparity %.2f%% exceeds ceiling %.2f%%: "
            f"%s=%.4f vs %s=%.4f" % (
                ratio, max_fpr_ratio, worst, fpr[worst], best, fpr[best]))
    return report
