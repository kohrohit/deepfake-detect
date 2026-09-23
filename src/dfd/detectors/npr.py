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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
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


def npr_feature(
    img: npt.NDArray[np.uint8], stride: int = DEFAULT_STRIDE
) -> npt.NDArray[np.float32]:
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
    x: npt.NDArray[np.float32] = img.astype(np.float32) / 255.0
    down = x[::stride, ::stride]
    up: npt.NDArray[np.float32] = np.repeat(
        np.repeat(down, stride, axis=0), stride, axis=1
    ).astype(np.float32)
    # Trim to match input shape (in case height or width not divisible by stride)
    up = up[: x.shape[0], : x.shape[1]]
    residual: npt.NDArray[np.float32] = (x - up).astype(np.float32)
    return residual


#: The four (dy, dx) sampling phases of a stride-2 grid, minus the one that
#: carries nothing. Nearest-neighbour upsampling REPLICATES the sampled
#: pixel, so `up[2i, 2j] == x[2i, 2j]` and the residual on the even phase is
#: identically zero for every image, real or generated. A statistic computed
#: there is a constant, not a feature.
INFORMATIVE_PHASES: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1))

#: Keeps the log-ratios finite when a phase's residual is exactly zero, which
#: happens on flat fields and on synthetic test images. A nan here would
#: reach the linear layer and poison every output silently.
_RATIO_EPS = 1e-6


class NPRStatsNet(nn.Module):
    """Slot C's model: phase statistics of the NPR residual, then a linear layer.

    **Why this shape and not a CNN.** The hardware ruling is CPU-only
    (docs/HANDOFF.md §1, correction 3), which makes "handcrafted features
    feeding a light model" the only detector this project can actually fit.
    This is that: 27 numbers describing how the upsampling residual is
    distributed across the stride-2 sampling phases, and a `Linear(27, 2)`.
    It trains in seconds and runs in milliseconds.

    **The physics it reads.** A generator that upsamples by replication or
    interpolation leaves content that reconstructs almost exactly under
    downsample-then-upsample, so its residual is small AND its residual is
    distributed differently across the three informative phases than a
    camera's optical chain leaves it. The ratios below are there to read the
    second property, which is scale-free — the absolute magnitudes are not,
    and that matters because resolution has already been measured as this
    project's most reliable shortcut (docs/HANDOFF.md §0).

    **Feature layout**, fixed because a fitted `state_dict` is meaningless
    without it:

        f[ 0: 9]  mean |residual|, channel-major over the three phases
        f[ 9:18]  standard deviation of the residual, same order
        f[18:27]  log-ratios of the phase means, per channel

    **The normalisation travels in the state_dict.** `feature_mean` and
    `feature_scale` are buffers, not fitter-side bookkeeping: a model that
    standardised during fitting and not at inference loads clean, scores
    confidently, and is wrong. `training/fit_npr.py` writes them.
    """

    #: Length of the feature vector `features` returns. A fitted state_dict
    #: is tied to it, so changing it invalidates every saved model.
    N_FEATURES = 27

    #: Declared for the type checker only. `register_buffer` assigns through
    #: `nn.Module.__setattr__`, so a reader of these attributes gets
    #: `Tensor | Module` from `nn.Module.__getattr__` and `forward`'s
    #: arithmetic on them does not type-check. These annotations bind no
    #: value -- the buffers are still created in `__init__`, and still move
    #: with `.to()` and appear in `state_dict()` as buffers must.
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(self.N_FEATURES))
        self.register_buffer("feature_scale", torch.ones(self.N_FEATURES))
        self.linear = nn.Linear(self.N_FEATURES, 2)

    def features(self, residual: torch.Tensor) -> torch.Tensor:
        """Phase statistics of an NPR residual batch.

        Args:
            residual: `(B, 3, H, W)` float tensor, as produced by
                `npr_feature` and permuted to channels-first.

        Returns:
            `(B, N_FEATURES)` float tensor. Always finite: an empty phase
            (an image smaller than the stride) and a flat field both
            contribute zeros rather than nan.
        """
        means: list[torch.Tensor] = []
        stds: list[torch.Tensor] = []
        for dy, dx in INFORMATIVE_PHASES:
            sub = residual[:, :, dy::2, dx::2].flatten(2)
            if sub.shape[-1] == 0:
                zeros = residual.new_zeros(residual.shape[0], residual.shape[1])
                means.append(zeros)
                stds.append(zeros)
                continue
            means.append(sub.abs().mean(-1))
            stds.append(sub.std(-1, unbiased=False))

        mean_stack = torch.stack(means, dim=2)          # (B, C, phase)
        std_stack = torch.stack(stds, dim=2)
        ratios = torch.stack([
            torch.log((mean_stack[:, :, a] + _RATIO_EPS)
                      / (mean_stack[:, :, b] + _RATIO_EPS))
            for a, b in ((0, 1), (0, 2), (1, 2))
        ], dim=2)
        return torch.cat([mean_stack.flatten(1), std_stack.flatten(1),
                          ratios.flatten(1)], dim=1)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        """Two logits per sample, ordered (real, fake) to match `NPRDetector`.

        Args:
            residual: `(B, 3, H, W)` NPR residual batch.

        Returns:
            `(B, 2)` logits. `NPRDetector.score` softmaxes these and takes
            column 1 as P(fake).
        """
        f = (self.features(residual) - self.feature_mean) / self.feature_scale
        out: torch.Tensor = self.linear(f)
        return out


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
                logits = model(t)

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
