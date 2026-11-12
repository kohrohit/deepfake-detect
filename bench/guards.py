"""The six evaluation-hygiene guards (spec §8.2).

Each raises rather than warns. A benchmark that can be silently run dirty will
be run dirty, and every one of these failures inflates results in the flattering
direction — which is exactly why they are easy to leave out.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class GuardViolation(Exception):
    """Raised when an evaluation-hygiene guard fails."""


@dataclass(frozen=True)
class IdentityReport:
    """Report from identity leakage guard with measured similarity metrics.

    Note: max_similarity=0.0 and violations=0 when train or test ids are empty
    does not confirm low risk, only that zero pairs were compared.

    A missing embedding is not represented here at all: `check_identity_disjoint`
    raises `GuardViolation` before constructing this report if any train or
    test id has no embedding, rather than silently skipping that id and
    returning a report over the pairs that happened to be measurable.
    """
    n_train: int
    n_test: int
    max_similarity: float
    violations: int
    threshold: float
    #: `violations / (n_train * n_test)`, or 0.0 when no pairs were compared.
    #: Reported beside the count because the count alone is unreadable without
    #: the denominator: 25 crossing pairs is leakage in a 10x10 split and is
    #: BELOW the unrelated-face rate in a 1,000x300 one. Defaulted so that
    #: existing constructions (and their tests) keep working unchanged.
    violation_rate: float = 0.0
    #: The rate this run was willing to tolerate. See
    #: `check_identity_disjoint`'s `max_false_match_rate`.
    tolerated_rate: float = 0.0


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


def check_identity_disjoint(
    train_ids: list[str],
    test_ids: list[str],
    embeddings: dict[str, np.ndarray],
    threshold: float = 0.6,
    max_false_match_rate: float = 0.0,
) -> IdentityReport:
    """Guard 1 — identity leakage.

    The same person in train and test teaches the model faces, not forgery.
    Returns a measured report; spec acceptance criterion 2 requires the number,
    not an assertion that it was checked.

    **Why `max_false_match_rate` exists, measured 2026-09-23.** This guard was
    written to raise if ANY pair crosses the threshold, which is right for a
    handful of subjects and wrong at scale. Measured with `dfd.embed`'s SFace
    embedder over 124,251 pairs of DIFFERENT FairFace people: similarity
    reaches 0.657 at the maximum, and 0.21% of unrelated pairs cross 0.363.
    A 1,000 x 300 split therefore produces hundreds of crossings with no
    leakage whatever, and a guard that refuses every honest split is a guard
    somebody switches off. Tolerating a rate — never a count — is what keeps
    this readable as the corpus grows: leakage raises the rate, and pair count
    does not.

    The default stays 0.0, so existing callers keep the strict behaviour and
    nothing is silently loosened. A caller working at scale must pass the rate
    it is willing to attribute to the embedder's own false-match rate, and
    should take that number from a measurement on ITS corpus, not from this
    docstring.

    Args:
        train_ids: ids on the training side.
        test_ids: ids on the test side.
        embeddings: id -> embedding vector. Every id in both lists must appear.
        threshold: cosine similarity at or above which a pair is a crossing.
        max_false_match_rate: the fraction of train x test pairs allowed to
            cross before this is called leakage. 0.0 means any crossing is.

    Raises:
        GuardViolation: if the crossing RATE exceeds `max_false_match_rate`,
            or if any train or test id has no embedding. A missing embedding
            means the comparison for that id was never made; treating it as
            "no similarity found" would certify disjointness that was never
            checked, which is worse than refusing to answer.
        ValueError: if `max_false_match_rate` is outside [0, 1]. A negative
            rate would make every split fail and a rate above 1 would make
            every split pass, and both look like a working guard from the
            outside.
    """
    if not 0.0 <= max_false_match_rate <= 1.0:
        raise ValueError(
            "max_false_match_rate must be in [0, 1], got "
            f"{max_false_match_rate}: outside it this guard silently becomes "
            "either always-fail or always-pass")
    missing = sorted({i for i in (*train_ids, *test_ids) if i not in embeddings})
    if missing:
        raise GuardViolation(
            f"identity guard cannot certify disjointness: no embedding for "
            f"{len(missing)} id(s): {missing}")
    max_sim = 0.0
    violations = 0
    for a in train_ids:
        for b in test_ids:
            sim = _cosine(embeddings[a], embeddings[b])
            max_sim = max(max_sim, sim)
            if sim >= threshold:
                violations += 1
    n_pairs = len(train_ids) * len(test_ids)
    rate = violations / n_pairs if n_pairs else 0.0
    if rate > max_false_match_rate:
        raise GuardViolation(
            f"identity leakage: {violations} of {n_pairs} train/test pairs "
            f"({rate:.4%}) at cosine >= {threshold} (max {max_sim:.4f}), above "
            f"the tolerated {max_false_match_rate:.4%}")
    return IdentityReport(n_train=len(train_ids), n_test=len(test_ids),
                          max_similarity=max_sim, violations=violations,
                          threshold=threshold, violation_rate=rate,
                          tolerated_rate=max_false_match_rate)


def check_subject_partition(
    subject_by_id: dict[str, str],
    embeddings: dict[str, np.ndarray],
    threshold: float = 0.6,
    max_false_match_rate: float = 0.0,
    max_subjects: int = 500,
    seed: int = 0,
) -> IdentityReport:
    """Guard 1, at the corpus level: are the declared SUBJECTS distinct people?

    **Why this exists beside `check_identity_disjoint`.** That function
    certifies one train/test split, which presupposes a split to certify.
    Every corpus this project holds has exactly one generator, so
    `logo_splits` refuses them all and the per-fold check never runs — a
    criterion that is wired and structurally unable to fire. This one needs
    no split. It asks the question the split discipline actually rests on:
    two records carrying DIFFERENT `subject_id` are assumed to be different
    people, and if that assumption is false then every split partitioned on
    subject_id leaks, however carefully it was built.

    That assumption is known to be shaky here and not merely in principle.
    `corpora/sbi.py` and `training/fit_blend.py` both record that
    `subject_id` is a capture SESSION id: one person enrolling twice gets two
    ids. `corpora/df40.py` sets it to a filename family. Neither is an
    identity.

    One representative embedding per subject is compared, not every record:
    the question is whether two SUBJECTS are one person, and comparing every
    record pair multiplies the work by frames-per-subject without changing
    what is being asked.

    Args:
        subject_by_id: sample_id -> subject_id, over the whole corpus.
        embeddings: sample_id -> embedding. A subject whose representative
            has no embedding is dropped from the comparison and counted in
            the report's `n_train` shortfall rather than assumed distinct.
        threshold: cosine at or above which two subjects are one person.
        max_false_match_rate: fraction of subject pairs allowed to cross.
            Unrelated faces cross at a measurable rate (see `dfd.embed`), so
            at scale this must be non-zero to mean anything.
        max_subjects: cap on subjects compared. The comparison is quadratic —
            86,000 subjects is 3.7 billion pairs — so a large corpus is
            SAMPLED, and the report says how many were compared.
        seed: sampling seed.

    Returns:
        An `IdentityReport` whose `n_train` and `n_test` are both the number
        of subjects compared. They are equal because the comparison is a
        corpus against itself rather than one side against another.

    Raises:
        GuardViolation: if the crossing rate exceeds `max_false_match_rate`.
        ValueError: if `max_subjects` is below 2 — one subject has no pair to
            compare and would certify every corpus clean.
    """
    if max_subjects < 2:
        raise ValueError(
            f"max_subjects must be at least 2, got {max_subjects}: a single "
            "subject has no pair to compare and would certify any corpus")
    if not 0.0 <= max_false_match_rate <= 1.0:
        raise ValueError(
            f"max_false_match_rate must be in [0, 1], got {max_false_match_rate}")

    representative: dict[str, str] = {}
    for sample_id, subject in sorted(subject_by_id.items()):
        if subject not in representative and sample_id in embeddings:
            representative[subject] = sample_id

    subjects = sorted(representative)
    if len(subjects) > max_subjects:
        rng = np.random.default_rng(seed)
        picked = rng.choice(len(subjects), size=max_subjects, replace=False)
        subjects = [subjects[i] for i in sorted(picked)]

    vectors = np.array([embeddings[representative[s]] for s in subjects]) \
        if subjects else np.zeros((0, 1))
    n = len(subjects)
    if n < 2:
        return IdentityReport(n_train=n, n_test=n, max_similarity=0.0,
                              violations=0, threshold=threshold,
                              violation_rate=0.0,
                              tolerated_rate=max_false_match_rate)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # A zero-norm vector has no direction; leaving it to divide by zero would
    # fill its row with nan and make `>=` silently False everywhere.
    norms[norms == 0.0] = 1.0
    sims = (vectors / norms) @ (vectors / norms).T
    iu = np.triu_indices(n, k=1)
    pair_sims = sims[iu]
    violations = int((pair_sims >= threshold).sum())
    n_pairs = len(pair_sims)
    rate = violations / n_pairs if n_pairs else 0.0
    max_sim = float(pair_sims.max()) if n_pairs else 0.0

    if rate > max_false_match_rate:
        raise GuardViolation(
            f"declared subjects are not distinct people: {violations} of "
            f"{n_pairs} subject pairs ({rate:.4%}) at cosine >= {threshold} "
            f"(max {max_sim:.4f}), above the tolerated "
            f"{max_false_match_rate:.4%}. Every split partitioned on "
            "subject_id leaks by this much.")
    return IdentityReport(n_train=n, n_test=n, max_similarity=max_sim,
                          violations=violations, threshold=threshold,
                          violation_rate=rate,
                          tolerated_rate=max_false_match_rate)


def check_video_level(
    sample_ids: list[str],
    groups: list[str],
) -> None:
    """Guard 2 — one sample per source video.

    Frames must be aggregated before scoring. 10,000 frames from 100 videos is
    100 independent samples; treating them as 10,000 inflates AUC and shrinks
    confidence intervals dishonestly.

    Args:
        sample_ids: Identifiers for samples (rows).
        groups: Source video identifiers — must uniquely identify the origin
                video for each sample. MUST NOT be the sample_ids themselves,
                as that makes the guard vacuous.

    Raises:
        GuardViolation: If a video appears in multiple samples.
        ValueError: If groups and sample_ids are element-wise identical,
                   indicating the guard would be a no-op.
    """
    # Reject identical sequences: the guard's only failure condition is
    # groups[i] != sample_ids[i], so identical lists are structurally unable to fire.
    if list(sample_ids) == list(groups):
        raise ValueError(
            "check_video_level guard is vacuous: groups are identical to "
            "sample_ids. Pass the SOURCE VIDEO identifier as groups, not "
            "sample_ids; one sample_id per video.")

    seen: dict[str, str] = {}
    for sid, g in zip(sample_ids, groups, strict=True):
        if g in seen and seen[g] != sid:
            raise GuardViolation(
                f"group {g!r} appears in multiple samples ({seen[g]!r}, {sid!r}); "
                "aggregate to video level before scoring")
        seen[g] = sid


def check_compression_coverage(
    records: list[dict[str, Any]],
    required: tuple[str, ...] = ("c0", "c23", "c40"),
) -> None:
    """Guard 3 — evaluate across compression levels, report the worst.

    Raises:
        GuardViolation: If any required compression level is missing.
    """
    present = {r.get("compression") for r in records}
    missing = [c for c in required if c not in present]
    if missing:
        raise GuardViolation(
            f"compression levels missing from evaluation: {', '.join(missing)}")


def check_uniform_preprocessing(
    records: list[dict[str, Any]],
) -> None:
    """Guard 4 — one preprocessing pipeline, applied to entire dataset.

    All samples must use the same face detector and alignment method.
    Applying different preprocessing by label is itself a giveaway the model
    will happily learn.

    Raises:
        GuardViolation: If preprocessing parameters vary across any samples.
    """
    for key in ("face_detector", "align"):
        all_values: set = set()
        for r in records:
            all_values.add(r.get(key))
        if len(all_values) > 1:
            raise GuardViolation(
                f"non-uniform preprocessing: {key} takes values {sorted(map(str, all_values))}")


def check_threshold_provenance(source: str) -> None:
    """Guard 5 — operating points frozen on validation, never chosen on test.

    Raises:
        GuardViolation: If threshold source is not 'validation'.
    """
    if source != "validation":
        raise GuardViolation(
            f"threshold source must be 'validation', got {source!r}")


def check_demographic_parity(
    scores: list[float] | np.ndarray,
    labels: list[int] | np.ndarray,
    strata: list[str] | np.ndarray,
    threshold: float,
    max_fpr_ratio: float = 2.0,
) -> ParityReport:
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
        values = list(fpr.values())
        lo = min(values)
        hi = max(values)
        # Infinity only at a literal zero denominator. Flooring a nonzero
        # denominator pulls the ratio BELOW the true value and silently hides
        # real disparity; for lo=0.0002, hi=0.0016, true ratio is 8.0x but
        # floor would compute 1.6x, swallowing a 4x genuine difference.
        ratio = (float("inf") if hi > 0.0 else 1.0) if lo == 0.0 else hi / lo

    report = ParityReport(fpr_by_stratum=fpr, tpr_by_stratum=tpr,
                          max_fpr_ratio=float(ratio), ceiling=max_fpr_ratio)
    if ratio > max_fpr_ratio:
        worst = max(fpr, key=fpr.get)
        best = min(fpr, key=fpr.get)
        # Format infinity as "inf" (unbounded) for legibility
        ratio_str = ("inf (one stratum has zero false positives)" if np.isinf(ratio)
                     else f"{ratio:.1f}x")
        ceiling_str = f"{max_fpr_ratio:.1f}x"
        raise GuardViolation(
            f"demographic FPR disparity {ratio_str} exceeds ceiling {ceiling_str}: "
            f"{worst}={fpr[worst]:.4f} vs {best}={fpr[best]:.4f}")
    return report
