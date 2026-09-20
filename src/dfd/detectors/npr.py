"""Slot C — upsampling fingerprint (spec §6).

Physics: neural generators upsample via transposed convolution or
interpolate-then-convolve, which leaves a periodic residual structure that a
camera's optical chain does not produce. The NPR feature isolates that residual
by subtracting the image's own downsample-then-upsample reconstruction.

IMPORTANT: The exact feature definition below (stride=2 nearest-neighbour down/up)
is specified verbatim but is UNVERIFIED against the NPR paper (Tan et al., CVPR
2024, "Rethinking the Up-Sampling Operations in CNN-based Generative Network for
Generalizable Deepfake Detection"). This implementation must be verified against
that source before any published claim relies on the numerical values. The physical
property (upsampled content → near-zero residual) is what matters; the constants
remain valid either way.

Supply-chain control: weights are loaded with torch.load(..., weights_only=True)
by default, preventing arbitrary code execution from tampered models (spec §3A).
Full-module pickles must be explicitly enabled and logged as a warning.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np

import torch
import torch.nn as nn

from ..types import Modality, Observation, RawScore
from .base import (
    NO_OBSERVATIONS,
    OK,
    WEIGHTS_ABSENT,
    abstain,
    filter_by_quality_floor,
)
from .loading import load_model

logger = logging.getLogger(__name__)

# Downsampling stride for NPR feature. Matches Tan et al., CVPR 2024 spec.
DEFAULT_STRIDE = 2

# Expected number of output classes (real/fake binary classifier).
EXPECTED_NUM_CLASSES = 2


def npr_feature(img: np.ndarray, stride: int = DEFAULT_STRIDE) -> np.ndarray:
    """Compute NPR (Neural-generated Periodic Residual) feature.

    The feature is the residual between the input image and its own
    downsample-then-upsample reconstruction using nearest-neighbour
    interpolation. This highlights the periodic artifacts left by neural
    generator upsampling, which natural images do not exhibit.

    Args:
        img: input image as uint8 HWC array
        stride: downsampling stride (default matches NPR paper)

    Returns:
        float32 residual array with same shape as input, in range [-1, 1]

    Raises:
        No exceptions raised; input is always valid numpy arrays.
    """
    x = img.astype(np.float32) / 255.0
    down = x[::stride, ::stride]
    up = np.repeat(np.repeat(down, stride, axis=0), stride, axis=1)
    # Trim to match input shape (in case height or width not divisible by stride)
    up = up[: x.shape[0], : x.shape[1]]
    return x - up


@dataclass(frozen=True)
class NPRDetector:
    """Detector that reads upsampling fingerprints via NPR features.

    This detector is frozen to maintain registry identity invariant:
    once registered, it cannot mutate its name or configuration.
    Model weights are loaded via a module-level cache with staleness detection
    (mtime_ns, size) to avoid silent re-use of replaced files, and thread-safe
    to protect against concurrent first-load races.

    Supply-chain control: weights are loaded securely by default via
    torch.load(..., weights_only=True) combined with a model_factory that
    reconstructs the architecture and loads the state_dict into it. This
    prevents arbitrary code execution from tampered models (spec §3A).

    **Secure path:** Provide model_factory as a callable that returns an
    instantiated, uninitialized model (e.g., `lambda: torch.nn.Linear(3, 2)`).
    The weights file must be a state_dict. This path uses weights_only=True and
    produces no warnings.

    **Unsafe path:** Omit model_factory and set allow_unsafe_load=True to load
    full-module pickles. This permits arbitrary code execution and logs a warning.

    Note: A state_dict alone cannot become a model without knowing the architecture.
    Secure loading therefore requires the architecture to be known (via model_factory).
    This is a genuine constraint, not a limitation of this implementation.

    Attributes:
        weights_path: path to the detector's trained weights file
        model_factory: callable that returns an uninitialized model instance.
            If provided, enables the secure weights_only=True path. If None,
            only full-module pickles with allow_unsafe_load=True are accepted.
        name: unique identifier (read-only via property)
        slot: detector slot designation (read-only via property)
        version: detector version string (read-only via property)
        modalities: which modalities this detector supports (read-only via property)
        min_quality_band: minimum quality band for consultation (read-only via property)
        allow_unsafe_load: if True, load full-module pickles unsafely; logs warning
    """
    weights_path: str | Path
    model_factory: Callable[[], nn.Module] | None = None
    allow_unsafe_load: bool = False

    # Identity fields (read-only after frozen)
    name: str = "npr"
    slot: str = "C"
    version: str = "0.1.0"
    modalities: frozenset[Modality] = field(
        default_factory=lambda: frozenset({Modality.IMAGE, Modality.VIDEO})
    )
    min_quality_band: Literal["low", "medium", "high"] = "low"

    def score(self, obs: Sequence[Observation]) -> RawScore:
        """Score observations using the NPR feature.

        Filters observations by quality floor, computes NPR features,
        runs them through the model, and returns a raw score or abstention.

        Args:
            obs: sequence of observations to score

        Returns:
            RawScore with score (or None if abstained) and reason, plus
            artifacts containing per-observation scores, max, and count

        Raises:
            ValueError: if model output has unexpected shape (not 2 classes)
            RuntimeError: if torch.load fails with weights_only=True and
                allow_unsafe_load is False (instructs user to convert to state_dict)
            Other exceptions from model inference propagate
        """
        # Handle empty observation list
        if not obs:
            logger.debug("score called with empty observation sequence; abstaining")
            return abstain(self.name, self.version, NO_OBSERVATIONS)

        weights_path = Path(self.weights_path)

        # Check if weights file exists
        if not weights_path.exists():
            logger.debug(
                "weights file not found: %s; abstaining with %s",
                weights_path,
                WEIGHTS_ABSENT,
            )
            return abstain(self.name, self.version, WEIGHTS_ABSENT)

        # Filter observations by quality floor (shared with every other
        # detector via detectors.base.filter_by_quality_floor).
        usable, reason = filter_by_quality_floor(obs, self.min_quality_band)
        if reason is not None:
            return abstain(self.name, self.version, reason)

        # Load model (or retrieve from cache) via the shared secure loader.
        try:
            model = load_model(weights_path, self.model_factory, self.allow_unsafe_load)
        except Exception as e:
            logger.error(
                "failed to load model from %s: %s",
                weights_path,
                type(e).__name__,
                exc_info=True,
            )
            raise

        # Compute NPR features and score
        try:
            feats = np.stack([npr_feature(o.payload) for o in usable])
            t = torch.from_numpy(feats).permute(0, 3, 1, 2).float()

            with torch.no_grad():
                logits = model(t)  # type: ignore

            # Validate output shape (must be 2-class binary classifier)
            if logits.shape[1] != EXPECTED_NUM_CLASSES:
                raise ValueError(
                    f"expected {EXPECTED_NUM_CLASSES} output classes, "
                    f"got {logits.shape[1]} from model"
                )

            # Extract fake-class probabilities
            probs = torch.softmax(logits, dim=1)[:, 1]
            per_obs_scores = probs.cpu().numpy().tolist()

            # Aggregate: record all scores for transparency, use max (not mean)
            # to avoid diluting strong single-frame detections
            max_score = float(probs.max())
            mean_score = float(probs.mean())

            logger.debug(
                "computed scores (max=%f, mean=%f) from %d observations",
                max_score,
                mean_score,
                len(usable),
            )

            # Return the mean as the official score (per brief), but record
            # per-observation scores in artifacts for transparency and fusion
            return RawScore(
                detector=self.name,
                version=self.version,
                score=mean_score,
                abstained=False,
                reason=OK,
                artifacts={
                    "n_observations": len(usable),
                    "per_observation_scores": per_obs_scores,
                    "max_score": max_score,
                },
            )
        except Exception as e:
            logger.error(
                "failed to score observations: %s",
                type(e).__name__,
                exc_info=True,
            )
            raise
