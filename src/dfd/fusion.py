"""Combine Evidence into a decision (spec §7, §9.5).

Two properties matter more than the arithmetic:

1. Abstentions contribute exactly zero. An abstention is not a vote for 'real'.
2. Correlated frames are discounted by effective sample size. Summing LLRs over
   900 frames of the same pipeline, identity and lighting yields a posterior of
   ~1.0 regardless of truth — confidently wrong. ESS is mandatory, not a refinement.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .policy import DEFAULT_POLICY, Policy
from .types import Evidence, Verdict

logger = logging.getLogger(__name__)

# Maximum aggregate evidence, in nats. Prevents overconfidence from large numbers
# of weak detectors; no combination can exceed this regardless of n_frames or ESS.
# Confidence at ±20 nats ≈ 99.99% posterior. Spec §9.3.
MAX_TOTAL_LLR = 20.0

# Re-exported so callers that imported these names before `Policy` existed keep
# working. `tests/test_fusion.py::test_reexported_constants_match_the_default_policy`
# asserts each equals its Policy field — two homes for one number is how they drift.
FAKE_THRESHOLD = DEFAULT_POLICY.fake_threshold
REAL_THRESHOLD = DEFAULT_POLICY.real_threshold
DISAGREEMENT_OOD = DEFAULT_POLICY.disagreement_ood


@dataclass(frozen=True)
class FusedResult:
    """Outcome of combining Evidence across detectors.

    Attributes:
        verdict: Decision (REAL, FAKE, INSUFFICIENT_EVIDENCE, OUT_OF_DISTRIBUTION).
        llr_total: Aggregate log-likelihood ratio after ESS discount and cap.
        posterior: Posterior probability of being fake (logistic sigmoid of llr_total).
        disagreement: Sum of the smaller of positive and negative evidence streams.
        n_contributing: Count of evidence that was not abstained.
        ess: Effective sample size applied as discount.
        reasons: Detector → reason mapping for audit and root-cause.
    """
    verdict: Verdict
    llr_total: float
    posterior: float
    disagreement: float
    n_contributing: int
    ess: float
    reasons: dict[str, str] = field(default_factory=dict)


def effective_sample_size(series: npt.NDArray[np.float64] | list[float]) -> float:
    """ESS from lag-1 autocorrelation: n * (1 - rho) / (1 + rho).

    Perfectly correlated frames (rho ≈ 1) → ESS ≈ 1.
    Independent frames (rho ≈ 0) → ESS ≈ n.

    ESS is clamped to [1.0, n]. Anti-correlated data (rho → -1) would give
    ESS > n, which contradicts the model: ESS is the number of *independent*
    observations equivalent to these n dependent ones. It cannot exceed n.

    For n < 3, reliable autocorrelation estimation is impossible. At n=2, any
    two distinct values give rho = -0.5 by construction, making ESS always 6.0
    (three times the sample size) regardless of the data. Return n directly.

    Raises:
        No exceptions; returns float(n) for insufficient input.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < 3:
        return float(n)
    if np.std(x) < 1e-12:
        return 1.0
    xc = x - x.mean()
    rho = float(np.dot(xc[:-1], xc[1:]) / np.dot(xc, xc))
    rho = max(-0.999, min(0.999, rho))
    ess = n * (1.0 - rho) / (1.0 + rho)
    return float(min(float(n), max(1.0, ess)))


def fuse(evidence: list[Evidence], n_frames: int = 1, ess: float | None = None,
         policy: Policy = DEFAULT_POLICY) -> FusedResult:
    """Combine Evidence into a single verdict.

    CONTRACT (critical): This function assumes one of two mutually exclusive patterns:

    1. **Per-frame evidence** (n_frames > 1): evidence list contains ONE Evidence per frame.
       The list sum equals (n_frames × per-frame_llr). The ESS discount applies here:
       total = naive_sum × (ESS / n_frames). Useful only when a caller explicitly
       breaks down a sample into per-frame evidence before fusion.

    2. **Aggregated evidence** (n_frames = 1): evidence list contains ONE Evidence per
       detector, where each Evidence.llr is already whole-sample (the detector aggregated
       internally). No discount is applied; Evidence passes through untouched.
       This is the normal path: all detectors in this repo aggregate over frames before
       emitting a score, and calibration was fitted on the aggregated score.

    Misuse: Passing n_frames=900 with a single Evidence (whole-sample llr) triggers the
    discount, destroying the evidence. A WARNING is logged when n_frames > 1 and
    len(evidence) < n_frames, indicating likely misuse.

    Abstentions are filtered out before aggregation. LLRs are summed, then optionally
    discounted. The total is capped at ±MAX_TOTAL_LLR. Disagreement (min of positive
    and negative evidence streams) triggers OUT_OF_DISTRIBUTION if ≥ DISAGREEMENT_OOD.

    Args:
        evidence: Detectors' log-likelihood ratios. Per-frame if n_frames > 1,
                  aggregated if n_frames = 1.
        n_frames: Number of frames in the sample. Use n_frames=1 for detectors that
                  aggregate internally (the normal case). Use n_frames > 1 only if
                  evidence list contains one entry per frame.
        ess: Effective sample size for discount. Applied only when n_frames > 1.
             If None and n_frames > 1, defaults to 1.0 (maximum discount; conservative).
        policy: thresholds to apply. Defaults to DEFAULT_POLICY, which holds the
            values this function used as module constants before policies
            existed, so an unchanged caller gets unchanged behaviour. Pass the
            same object to `build_audit_record` so the record's threshold is
            the one applied rather than a copy of it.

    Returns:
        FusedResult with verdict, aggregate LLR, posterior, disagreement, counts.

    Raises:
        ValueError: if n_frames < 1.
        No other exceptions; returns INSUFFICIENT_EVIDENCE for empty or all-abstained input.
    """
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")

    reasons = {e.detector: e.reason for e in evidence}
    contributing = [e for e in evidence if not e.abstained]

    if not contributing:
        logger.debug("No contributing evidence; returning INSUFFICIENT_EVIDENCE")
        return FusedResult(verdict=Verdict.INSUFFICIENT_EVIDENCE, llr_total=0.0,
                           posterior=0.5, disagreement=0.0, n_contributing=0,
                           ess=0.0, reasons=reasons)

    llrs = np.array([e.llr for e in contributing], dtype=float)
    total = float(llrs.sum())

    # Discount for temporal correlation (per-frame evidence only). Evidence scales with
    # independent observations, not with frame count. If n_frames correlated observations
    # are worth ESS independent ones, total = naive_sum × (ESS / n_frames).
    # This applies ONLY when evidence list contains one entry per frame.
    # When detectors aggregate internally (n_frames=1), no discount applies.
    if n_frames > 1:
        # Symmetric mismatch guard: warn if evidence count differs materially from n_frames.
        if len(contributing) != n_frames:
            logger.warning(
                "ESS discount: n_frames=%d but %d evidence entries provided. "
                "Discount assumes one entry per frame. If evidence is aggregated "
                "(not per-frame), use n_frames=1.",
                n_frames, len(contributing))

        # Clamp ESS to n_frames: ESS > n_frames would amplify evidence, not discount it.
        eff = ess if ess is not None else 1.0
        if ess is not None and ess > n_frames:
            logger.warning(
                "ESS %.2f exceeds n_frames %d; clamping. An ESS above the "
                "observation count would amplify evidence rather than discount it.",
                ess, n_frames)
            eff = float(n_frames)

        if ess is None:
            logger.warning(
                "Fusion ran with n_frames=%d and no ESS data; max-discounting to 1.0",
                n_frames)
        total *= max(1.0, eff) / float(n_frames)

    # Cap to prevent runaway confidence. Spec §9.3.
    total = max(-MAX_TOTAL_LLR, min(MAX_TOTAL_LLR, total))

    # Disagreement = min(positive evidence, negative evidence). If both streams
    # pull hard, the sample is off-distribution, not a confident average.
    pos = float(llrs[llrs > 0].sum())
    neg = float(-llrs[llrs < 0].sum())
    disagreement = min(pos, neg)

    # Posterior = P(fake | data). Spec §8.1.
    posterior = 1.0 / (1.0 + math.exp(-total))

    # Verdict logic. Disagreement wins (signals OOD).
    if disagreement >= policy.disagreement_ood:
        verdict = Verdict.OUT_OF_DISTRIBUTION
        logger.debug("Disagreement %s >= %s; OUT_OF_DISTRIBUTION",
                     disagreement, policy.disagreement_ood)
    elif total >= policy.fake_threshold:
        verdict = Verdict.FAKE
    elif total <= policy.real_threshold:
        verdict = Verdict.REAL
    else:
        verdict = Verdict.INSUFFICIENT_EVIDENCE

    return FusedResult(verdict=verdict, llr_total=total, posterior=posterior,
                       disagreement=disagreement, n_contributing=len(contributing),
                       ess=float(ess or 1.0), reasons=reasons)
