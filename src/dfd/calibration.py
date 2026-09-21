"""Raw score → calibrated log-likelihood ratio, conditioned on quality band.

Spec §7.1. A detector's reliability at 512px uncompressed is not its
reliability at 96px after recompression, so one calibration curve per band.

A band with no fitted curve returns llr = 0.0 — the honest answer to "I was
never calibrated in this regime" is 'no information', not an extrapolation.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression

from .types import Evidence, RawScore

logger = logging.getLogger(__name__)

UNCALIBRATED = "uncalibrated_for_band"
# Caps any single detector's contribution. Prevents one saturated model from
# dominating the fused posterior (the failure mode seen in RD's cedar models).
# See spec §1.2: RD telemetry shows models emitting only 0.01 or 0.99.
MAX_ABS_LLR = 6.0
# Total sample count threshold per band (both classes combined). A band with
# fewer than this many samples cannot be reliably calibrated. Note: this is a
# total count, not per-class; a band with 19 fake + 1 real passes this guard
# but would fit a curve off a single real example (degenerate).
MIN_FIT_SAMPLES = 20


class Calibrator:
    """Converts raw scores to log-likelihood ratios, per quality band."""

    def __init__(self, detector: str, max_abs_llr: float = MAX_ABS_LLR) -> None:
        """Initialize a calibrator for a given detector.

        Args:
            detector: Detector identifier.
            max_abs_llr: Maximum absolute value of LLR (clipping bound).

        Raises:
            ValueError: If max_abs_llr is not positive.
        """
        if max_abs_llr <= 0:
            raise ValueError(f"max_abs_llr must be positive, got {max_abs_llr}")
        self.detector = detector
        self.max_abs_llr = max_abs_llr
        self._models: dict[str, LogisticRegression] = {}
        self._priors: dict[str, float] = {}

    def fit(
        self,
        scores: npt.NDArray[np.float64] | Sequence[float],
        labels: npt.NDArray[np.int64] | Sequence[int],
        bands: npt.NDArray[np.str_] | Sequence[str],
    ) -> Calibrator:
        """Fit logistic calibration curves per quality band.

        Args:
            scores: Array of raw detector scores [0, 1].
            labels: Array of ground truth labels (0 for real, 1 for fake).
                Must be binary {0, 1}, not {-1, +1} or other encodings.
            bands: Array of quality band identifiers.

        Returns:
            Self (for chaining).

        Raises:
            ValueError: If inputs have mismatched lengths, labels not binary,
                or invalid dtypes.
        """
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=int)
        bands = np.asarray(bands)

        if not (len(scores) == len(labels) == len(bands)):
            raise ValueError(
                f"Input length mismatch: scores={len(scores)}, "
                f"labels={len(labels)}, bands={len(bands)}"
            )

        # Validate labels are binary {0, 1}
        unique_labels = np.unique(labels)
        if not (len(unique_labels) <= 2 and np.all(np.isin(unique_labels, [0, 1]))):
            raise ValueError(
                f"Labels must be binary {{0, 1}}, got {unique_labels} for detector {self.detector}"
            )

        for band in np.unique(bands):
            m = bands == band
            n_samples = m.sum()
            n_classes = len(np.unique(labels[m]))

            # Skip bands with too few samples or only one class
            if n_samples < MIN_FIT_SAMPLES:
                logger.warning(
                    "Detector %s, band %s: %d < %d samples, skipping fit",
                    self.detector, band, n_samples, MIN_FIT_SAMPLES
                )
                continue
            if n_classes < 2:
                logger.warning(
                    "Detector %s, band %s: only %d class(es), skipping fit",
                    self.detector, band, n_classes
                )
                continue

            lr = LogisticRegression()
            lr.fit(scores[m].reshape(-1, 1), labels[m])
            self._models[str(band)] = lr
            self._priors[str(band)] = float(labels[m].mean())
            logger.info(
                "Detector %s, band %s: fitted with %d samples, prior=%.3f",
                self.detector, band, n_samples, self._priors[str(band)]
            )

        return self

    def to_evidence(self, raw: RawScore, band: str) -> Evidence:
        """Convert a raw score to calibrated evidence on a given band.

        Args:
            raw: Raw detector output.
            band: Quality band (e.g., 'high', 'low').

        Returns:
            Evidence with log-likelihood ratio.

        Raises:
            ValueError: If band is not a string.
        """
        if not isinstance(band, str):
            raise ValueError(f"band must be a string, got {type(band)}")

        # Abstained score → zero LLR with original reason preserved
        if raw.abstained or raw.score is None:
            return Evidence(
                detector=raw.detector,
                detector_version=raw.version,
                llr=0.0,
                raw_score=None,
                uncertainty=0.0,
                abstained=True,
                reason=raw.reason,
                artifacts=dict(raw.artifacts)
            )

        # No calibration for this band → zero LLR (honest "no information")
        model = self._models.get(band)
        if model is None:
            return Evidence(
                detector=raw.detector,
                detector_version=raw.version,
                llr=0.0,
                raw_score=raw.score,
                uncertainty=0.0,
                abstained=True,
                reason=UNCALIBRATED,
                artifacts=dict(raw.artifacts)
            )

        # Posterior log-odds minus prior log-odds gives a LIKELIHOOD RATIO.
        # Subtracting the prior removes the training set's base rate, so it does
        # not ride into production where the fraud rate is orders of magnitude
        # different. Without this subtraction, the quantity is a posterior, not
        # a ratio, and the training set's assumptions contaminate every decision.
        post_logodds = float(model.decision_function([[raw.score]])[0])
        prior = self._priors[band]

        # Avoid log(0) or log(∞) at boundaries
        prior_logodds = math.log(prior / (1.0 - prior)) if 0 < prior < 1 else 0.0

        llr = post_logodds - prior_logodds

        # Clip to prevent one detector from dominating the posterior
        llr = max(-self.max_abs_llr, min(self.max_abs_llr, llr))

        return Evidence(
            detector=raw.detector,
            detector_version=raw.version,
            llr=llr,
            raw_score=raw.score,
            uncertainty=0.0,
            abstained=False,
            reason="ok",
            artifacts=dict(raw.artifacts)
        )
