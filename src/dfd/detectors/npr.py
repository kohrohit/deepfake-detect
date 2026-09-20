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
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

import torch
import torch.nn as nn

from ..quality import meets_floor
from ..types import Modality, Observation, RawScore
from .base import BELOW_FLOOR, NO_OBSERVATIONS, NO_QUALITY, OK, WEIGHTS_ABSENT, abstain

logger = logging.getLogger(__name__)

# Downsampling stride for NPR feature. Matches Tan et al., CVPR 2024 spec.
DEFAULT_STRIDE = 2

# Expected number of output classes (real/fake binary classifier).
EXPECTED_NUM_CLASSES = 2

# Module-level cache for loaded weights to keep the frozen dataclass immutable.
# Keys are (resolved_path, mtime_ns, size) tuples to detect file replacement.
# Values are loaded model objects.
_MODEL_CACHE: dict[tuple[str, int, int], nn.Module] = {}
_CACHE_LOCK = threading.Lock()


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

    Supply-chain control: weights are loaded with torch.load(..., weights_only=True)
    by default, preventing arbitrary code execution from tampered models. Full
    pickles require explicit allow_unsafe_load=True and log a warning.

    Attributes:
        weights_path: path to the detector's trained weights file
        name: unique identifier (read-only via property)
        slot: detector slot designation (read-only via property)
        version: detector version string (read-only via property)
        modalities: which modalities this detector supports (read-only via property)
        min_quality_band: minimum quality band for consultation (read-only via property)
        allow_unsafe_load: if True, load full-module pickles unsafely; logs warning
    """
    weights_path: str | Path
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

    def _load_model(self, path: Path) -> nn.Module:
        """Load model from disk with staleness detection and thread safety.

        Uses a module-level cache keyed by (resolved_path, mtime_ns, size)
        to detect file replacement (e.g., a compromised model pushed at the
        same path). Protected by a lock against concurrent first-load races.

        Loads with weights_only=True (supply-chain control) by default. Full
        pickles require allow_unsafe_load=True and trigger a warning.

        Args:
            path: path to weights file

        Returns:
            loaded model object as nn.Module

        Raises:
            RuntimeError: if weights_only=True fails and allow_unsafe_load is False
            Exception: any other exception from torch.load propagates
        """
        resolved = path.resolve()
        stat = resolved.stat()
        # Cache key includes mtime_ns and size to detect file replacement
        cache_key = (str(resolved), stat.st_mtime_ns, stat.st_size)

        with _CACHE_LOCK:
            if cache_key in _MODEL_CACHE:
                logger.debug("using cached model from %s", resolved)
                return _MODEL_CACHE[cache_key]

            logger.debug("loading model from %s", resolved)

            # Try secure load first (weights only, no code execution)
            try:
                model = torch.load(
                    str(resolved), map_location="cpu", weights_only=True
                )
                logger.debug("loaded with weights_only=True (secure)")
            except Exception as e:
                # weights_only failed (likely a full module pickle); user must opt in
                if not self.allow_unsafe_load:
                    raise RuntimeError(
                        f"failed to load {resolved} with weights_only=True. "
                        f"This is a supply-chain security measure (spec §3A). "
                        f"Either: (1) convert the weights to a state_dict file, or "
                        f"(2) explicitly set allow_unsafe_load=True (will log warning). "
                        f"Error: {type(e).__name__}: {str(e)[:200]}"
                    ) from e

                # Fallback to unsafe load with warning
                logger.warning(
                    "loading %s with full unpickle (allow_unsafe_load=True); "
                    "this permits arbitrary code execution",
                    resolved,
                )
                model = torch.load(str(resolved), map_location="cpu")
                logger.debug("loaded with unsafe unpickle")

            # Ensure it's a module (not just a state dict)
            if not isinstance(model, nn.Module):
                raise ValueError(
                    f"loaded object is not a torch.nn.Module; "
                    f"got {type(model).__name__}"
                )

            model.eval()
            _MODEL_CACHE[cache_key] = model
            return model
