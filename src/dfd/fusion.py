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

from .types import Evidence, Verdict

logger = logging.getLogger(__name__)

# Maximum aggregate evidence, in nats. Prevents overconfidence from large numbers
# of weak detectors; no combination can exceed this regardless of n_frames or ESS.
# Confidence at ±20 nats ≈ 99.99% posterior. Spec §9.3.
MAX_TOTAL_LLR = 20.0

# LLR threshold to emit FAKE verdict. Spec §7.1.
FAKE_THRESHOLD = 1.0

# LLR threshold to emit REAL verdict. Spec §7.1.
REAL_THRESHOLD = -1.0

# Evidence pulling hard in both directions means off-distribution, not 'average them'.
# Two detectors at full opposite confidence is 6-of-10 split case the spec says
# vendors wrongly average away. Spec §7.2.
DISAGREEMENT_OOD = 3.0


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
    reasons: dict = field(default_factory=dict)


def effective_sample_size(series) -> float:
    """ESS from lag-1 autocorrelation: n * (1 - rho) / (1 + rho).

    Perfectly correlated frames (rho ≈ 1) → ESS ≈ 1.
    Independent frames (rho ≈ 0) → ESS ≈ n.

    Raises:
        No exceptions; returns 1.0 for invalid input.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < 2:
        return float(n)
    if np.std(x) < 1e-12:
        return 1.0
    xc = x - x.mean()
    rho = float(np.dot(xc[:-1], xc[1:]) / np.dot(xc, xc))
    rho = max(-0.999, min(0.999, rho))
    return max(1.0, n * (1.0 - rho) / (1.0 + rho))


def fuse(evidence: list[Evidence], n_frames: int = 1, ess: float | None = None) -> FusedResult:
    """Combine Evidence into a single verdict.

    Abstentions are filtered out before aggregation. LLRs are summed, then
    discounted by √(ESS / n_frames) to account for frame-to-frame correlation.
    The total is capped at ±MAX_TOTAL_LLR. Disagreement (min of positive and
    negative evidence streams) triggers OUT_OF_DISTRIBUTION if ≥ DISAGREEMENT_OOD.

    Args:
        evidence: Detectors' calibrated log-likelihood ratios.
        n_frames: Number of frames in the sample (>1 triggers ESS discount).
        ess: Effective sample size. If None and n_frames > 1, defaults to 1.0.

    Returns:
        FusedResult with verdict, aggregate LLR, posterior, disagreement, counts.

    Raises:
        No exceptions; returns INSUFFICIENT_EVIDENCE for empty or all-abstained input.
    """
    reasons = {e.detector: e.reason for e in evidence}
    contributing = [e for e in evidence if not e.abstained]

    if not contributing:
        logger.debug("No contributing evidence; returning INSUFFICIENT_EVIDENCE")
        return FusedResult(verdict=Verdict.INSUFFICIENT_EVIDENCE, llr_total=0.0,
                           posterior=0.5, disagreement=0.0, n_contributing=0,
                           ess=0.0, reasons=reasons)

    llrs = np.array([e.llr for e in contributing], dtype=float)
    total = float(llrs.sum())

    # Discount for temporal correlation: evidence scales with independent
    # observations, not with frame count. √(ESS/n) accounts for the fact that
    # frame-to-frame detector errors are strongly correlated (same pipeline,
    # identity, lighting). Spec §9.5.
    if n_frames > 1:
        eff = ess if ess is not None else 1.0
        total *= math.sqrt(max(1.0, eff) / float(n_frames))

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
    if disagreement >= DISAGREEMENT_OOD:
        verdict = Verdict.OUT_OF_DISTRIBUTION
        logger.debug("Disagreement %s >= %s; OUT_OF_DISTRIBUTION", disagreement, DISAGREEMENT_OOD)
    elif total >= FAKE_THRESHOLD:
        verdict = Verdict.FAKE
    elif total <= REAL_THRESHOLD:
        verdict = Verdict.REAL
    else:
        verdict = Verdict.INSUFFICIENT_EVIDENCE

    return FusedResult(verdict=verdict, llr_total=total, posterior=posterior,
                       disagreement=disagreement, n_contributing=len(contributing),
                       ess=float(ess or 1.0), reasons=reasons)
