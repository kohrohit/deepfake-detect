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
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from ..quality import meets_floor
from ..types import Modality, Observation, RawScore
from .base import BELOW_FLOOR, NO_QUALITY, OK, WEIGHTS_ABSENT, abstain

logger = logging.getLogger(__name__)

# Module-level cache for loaded weights to keep the frozen dataclass immutable.
# Keys are resolved weight file paths; values are loaded model objects.
_MODEL_CACHE: dict[str, object] = {}


def npr_feature(img: np.ndarray, stride: int = 2) -> np.ndarray:
    """Compute NPR (Neural-generated Periodic Residual) feature.

    The feature is the residual between the input image and its own
    downsample-then-upsample reconstruction using nearest-neighbour
    interpolation. This highlights the periodic artifacts left by neural
    generator upsampling, which natural images do not exhibit.

    Args:
        img: input image as uint8 HWC array
        stride: downsampling stride (default 2)

    Returns:
        float32 residual array with same shape as input, in range [-1, 1]
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
    Model weights are loaded via a module-level cache to avoid mutation
    of the immutable dataclass.

    Attributes:
        weights_path: path to the detector's trained weights file
        name: unique identifier (read-only via property)
        slot: detector slot designation (read-only via property)
        version: detector version string (read-only via property)
        modalities: which modalities this detector supports (read-only via property)
        min_quality_band: minimum quality band for consultation (read-only via property)
    """
    weights_path: str | Path

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
            RawScore with score (or None if abstained) and reason

        Raises:
            No exceptions are raised; failures are reported via abstention reason.
            Concrete exceptions from model inference propagate and must be
            caught by the caller.
        """
        weights_path = Path(self.weights_path)

        # Check if weights file exists
        if not weights_path.exists():
            logger.debug(
                "weights file not found: %s; abstaining with %s",
                weights_path,
                WEIGHTS_ABSENT,
            )
            return abstain(self.name, self.version, WEIGHTS_ABSENT)

        # Filter observations by quality floor
        usable = []
        has_measured_below_floor = False

        for o in obs:
            if o.quality is None:
                logger.debug("observation has quality=None; skipping")
                continue
            if meets_floor(o.quality.band, self.min_quality_band):
                usable.append(o)
            else:
                logger.debug(
                    "observation band %s below floor %s; skipping",
                    o.quality.band,
                    self.min_quality_band,
                )
                has_measured_below_floor = True

        # Abstain if no usable observations
        if not usable:
            reason = BELOW_FLOOR if has_measured_below_floor else NO_QUALITY
            logger.debug("no usable observations; abstaining: %s", reason)
            return abstain(self.name, self.version, reason)

        # Load model (or retrieve from cache)
        try:
            model = self._load_model(weights_path)
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
            import torch

            feats = np.stack([npr_feature(o.payload) for o in usable])
            t = torch.from_numpy(feats).permute(0, 3, 1, 2).float()

            with torch.no_grad():
                logits = model(t)  # type: ignore
                probs = torch.softmax(logits, dim=1)[:, 1]

            score_value = float(probs.mean())
            logger.debug(
                "computed score %f from %d observations",
                score_value,
                len(usable),
            )

            return RawScore(
                detector=self.name,
                version=self.version,
                score=score_value,
                abstained=False,
                reason=OK,
                artifacts={"n_observations": len(usable)},
            )
        except Exception as e:
            logger.error(
                "failed to score observations: %s",
                type(e).__name__,
                exc_info=True,
            )
            raise

    def _load_model(self, path: Path) -> object:
        """Load model from disk, caching the result.

        Uses a module-level cache keyed by resolved path to avoid mutable
        state in the frozen dataclass. This preserves immutability while
        allowing lazy loading.

        Args:
            path: path to weights file

        Returns:
            loaded model object

        Raises:
            Exception: any exception from torch.load propagates
        """
        # Resolve to absolute path for cache key
        key = str(path.resolve())

        if key in _MODEL_CACHE:
            logger.debug("using cached model from %s", key)
            return _MODEL_CACHE[key]

        logger.debug("loading model from %s", key)
        import torch

        model = torch.load(key, map_location="cpu")
        model.eval()  # type: ignore
        _MODEL_CACHE[key] = model
        return model
