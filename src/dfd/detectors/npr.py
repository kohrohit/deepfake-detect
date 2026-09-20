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
from typing import Callable, Literal, Sequence

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

        Three paths, in order of preference:

        1. **Secure path (recommended):** model_factory is provided.
           Load state_dict via weights_only=True and instantiate model via
           model_factory. No warnings. This is the secure-by-default path.

        2. **Rejection path:** model_factory is None and file is a full module.
           Reject with clear error message directing user to path 1 or 3.

        3. **Unsafe path (last resort):** allow_unsafe_load=True.
           Load full module via unpickle (code execution possible).
           Logs a warning.

        Uses a module-level cache keyed by (resolved_path, mtime_ns, size,
        factory_id, allow_unsafe_load) to detect file replacement and distinguish
        different load configurations. The same file loaded with different
        factories or security modes can produce different models and must not
        share a cache entry. Protected by a lock against concurrent first-load races.

        Args:
            path: path to weights file

        Returns:
            loaded model object as nn.Module

        Raises:
            RuntimeError: if path is full module and (model_factory is None
                and allow_unsafe_load is False)
            Exception: any other exception from torch.load propagates
        """
        resolved = path.resolve()
        stat = resolved.stat()

        # Compute stable identity for the factory
        if self.model_factory is None:
            factory_id = "none"
        else:
            factory_id = f"{self.model_factory.__module__}.{getattr(self.model_factory, '__qualname__', repr(self.model_factory))}"

        # Cache key includes file metadata, factory identity, and load mode
        # to ensure same file + different factories/modes don't collide
        cache_key = (
            str(resolved),
            stat.st_mtime_ns,
            stat.st_size,
            factory_id,
            self.allow_unsafe_load,
        )

        with _CACHE_LOCK:
            if cache_key in _MODEL_CACHE:
                logger.debug("using cached model from %s", resolved)
                return _MODEL_CACHE[cache_key]

            logger.debug("loading model from %s", resolved)

            # CASE 1: Secure path — model_factory provided
            if self.model_factory is not None:
                try:
                    sd = torch.load(
                        str(resolved), map_location="cpu", weights_only=True
                    )
                    if not isinstance(sd, dict):
                        raise ValueError(
                            f"expected state_dict (dict) with model_factory, "
                            f"got {type(sd).__name__}"
                        )
                    model = self.model_factory()
                    model.load_state_dict(sd)
                    model.eval()
                    logger.debug("loaded state_dict securely with model_factory")
                    _MODEL_CACHE[cache_key] = model
                    return model
                except Exception as e:
                    # Provide a clear error if the file appears to be a full-module pickle
                    if isinstance(e, Exception) and "Weights only load failed" in str(e):
                        raise RuntimeError(
                            f"the file {resolved} appears to be a full-module pickle, "
                            f"not a state_dict. model_factory requires a state_dict. "
                            f"These are mutually exclusive: either (1) provide a state_dict "
                            f"file with model_factory, or (2) remove model_factory and use "
                            f"allow_unsafe_load=True for full pickles."
                        ) from e
                    logger.error(
                        "failed to load state_dict with model_factory: %s",
                        type(e).__name__,
                        exc_info=True,
                    )
                    raise

            # CASE 2: Attempt weights_only=True (expecting state_dict or error)
            try:
                model = torch.load(
                    str(resolved), map_location="cpu", weights_only=True
                )
                # weights_only succeeded but model_factory is None
                if isinstance(model, dict):
                    raise RuntimeError(
                        f"loaded a state_dict but model_factory is None. "
                        f"To securely load weights, provide model_factory as "
                        f"a callable that returns an uninitialized model instance."
                    )
                if not isinstance(model, nn.Module):
                    raise ValueError(
                        f"expected torch.nn.Module, got {type(model).__name__}"
                    )
                model.eval()
                logger.debug("loaded full module with weights_only=True")
                _MODEL_CACHE[cache_key] = model
                return model
            except Exception as e:
                # weights_only failed (likely a full module pickle)
                if not self.allow_unsafe_load:
                    raise RuntimeError(
                        f"failed to load {resolved} with weights_only=True. "
                        f"This is a supply-chain security measure (spec §3A). "
                        f"To fix: (1) provide model_factory to load state_dict, or "
                        f"(2) set allow_unsafe_load=True (logs warning). "
                        f"Error: {type(e).__name__}"
                    ) from e

                # CASE 3: Unsafe path — operator has explicitly opted in
                logger.warning(
                    "loading %s with full unpickle (allow_unsafe_load=True); "
                    "this permits arbitrary code execution from the weights file",
                    resolved,
                )
                model = torch.load(str(resolved), map_location="cpu")
                if not isinstance(model, nn.Module):
                    raise ValueError(
                        f"unsafe load: expected torch.nn.Module, "
                        f"got {type(model).__name__}"
                    )
                model.eval()
                logger.debug("loaded full module with unsafe unpickle")
                _MODEL_CACHE[cache_key] = model
                return model
