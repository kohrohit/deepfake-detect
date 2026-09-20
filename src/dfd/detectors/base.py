"""Detector protocol, abstention helper, and synthetic detector for testing."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence

import numpy as np

from ..quality import meets_floor
from ..types import Modality, Observation, RawScore

logger = logging.getLogger(__name__)

BELOW_FLOOR = "below_quality_floor"
NO_QUALITY = "quality_not_measured"
WEIGHTS_ABSENT = "weights_absent"
OK = "ok"


def abstain(detector: str, version: str, reason: str) -> RawScore:
    """Produce a zero-information RawScore indicating the detector abstained.

    Args:
        detector: detector name
        version: detector version
        reason: reason for abstention (e.g., "weights_absent", "below_quality_floor")

    Returns:
        RawScore with abstained=True, score=None, and the given reason
    """
    return RawScore(detector=detector, version=version, score=None,
                    abstained=True, reason=reason)


class Detector(Protocol):
    """Contract for a detector: a callable that scores observations.

    Detectors are hot-swappable and expected to decay in 3-6 months.
    They must implement the Detector protocol to be registered in the harness.
    """
    name: str
    version: str
    modalities: set[Modality]
    min_quality_band: Literal["low", "medium", "high"]

    def score(self, obs: Sequence[Observation]) -> RawScore:
        """Score one or more observations.

        Args:
            obs: sequence of observations to score

        Returns:
            RawScore with score (or None if abstained) and reason if abstained
        """
        ...


@dataclass
class SyntheticDetector:
    """A deterministic stand-in used to test the harness without model weights.

    Its score is a hash of the observation payload, so it is reproducible and
    label-blind (never sees Context.label, only the pixel payload). It exists
    so the benchmark can be proven correct before any dataset or weight file
    arrives.

    Determinism guarantee: same observation payload plus same seed always
    yields the same score. This makes it suitable for repeatable testing.
    """
    name: str
    seed: int = 0
    version: str = "synthetic-1"
    modalities: set[Modality] = field(
        default_factory=lambda: {Modality.IMAGE, Modality.VIDEO})
    min_quality_band: Literal["low", "medium", "high"] = "low"

    def score(self, obs: Sequence[Observation]) -> RawScore:
        """Score observations deterministically using SHA256 hash of payload.

        Args:
            obs: sequence of observations to score

        Returns:
            RawScore with deterministic score based on payload hash, or
            abstention if no usable observations (below quality floor or
            quality not measured)

        Raises:
            No exceptions raised; abstentions are reported via reason field
        """
        if not obs:
            logger.debug("score called with empty observation sequence; abstaining")
            return abstain(self.name, self.version, NO_QUALITY)

        usable = []
        for o in obs:
            if o.quality is None:
                logger.debug("observation has quality=None; skipping")
                continue
            if meets_floor(o.quality.band, self.min_quality_band):
                usable.append(o)
            else:
                logger.debug(
                    "observation band %s below floor %s; skipping",
                    o.quality.band, self.min_quality_band)

        if not usable:
            reason = NO_QUALITY if obs[0].quality is None else BELOW_FLOOR
            logger.debug("no usable observations; abstaining: %s", reason)
            return abstain(self.name, self.version, reason)

        # Hash the payload deterministically using seed
        acc = 0
        for o in usable:
            h = hashlib.sha256(np.ascontiguousarray(o.payload).tobytes())
            h.update(str(self.seed).encode())
            acc ^= int.from_bytes(h.digest()[:8], "big")

        score = (acc % 10_000) / 10_000.0
        logger.debug("score computed: %f", score)
        return RawScore(detector=self.name, version=self.version,
                        score=score,
                        abstained=False, reason=OK)


class Registry:
    """Holds detectors and draws seeded random subsets (spec principle 9).

    Detectors are registered once and can be retrieved by name. Subsets can
    be drawn using a seed to ensure reproducibility across deployments.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._d: dict[str, Detector] = {}

    def register(self, detector: Detector) -> None:
        """Register a detector in the registry.

        Args:
            detector: detector instance to register

        Raises:
            ValueError: if a detector with the same name is already registered
        """
        if detector.name in self._d:
            logger.error("detector already registered: %s", detector.name)
            raise ValueError(f"detector already registered: {detector.name}")
        self._d[detector.name] = detector
        logger.debug("registered detector: %s", detector.name)

    def get(self, name: str) -> Detector:
        """Retrieve a registered detector by name.

        Args:
            name: detector name

        Returns:
            detector instance

        Raises:
            KeyError: if detector not found
        """
        return self._d[name]

    def names(self) -> list[str]:
        """Return sorted list of all registered detector names.

        Returns:
            list of detector names in sorted order
        """
        return sorted(self._d)

    def select_subset(self, k: int, seed: int) -> list[str]:
        """Select a random subset of k detector names using a seeded RNG.

        Args:
            k: number of detectors to select
            seed: random seed for reproducibility

        Returns:
            sorted list of k detector names (or fewer if registry has fewer)
        """
        names = self.names()
        k = min(k, len(names))
        rng = np.random.default_rng(seed)
        subset = rng.choice(names, size=k, replace=False).tolist()
        result = sorted(subset)
        logger.debug("selected subset of %d detectors from %d using seed %d",
                     len(result), len(names), seed)
        return result
