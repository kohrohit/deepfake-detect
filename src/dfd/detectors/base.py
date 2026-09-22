"""Detector protocol, abstention helper, and synthetic detector for testing."""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from ..quality import meets_floor
from ..types import Modality, Observation, RawScore

logger = logging.getLogger(__name__)

BELOW_FLOOR = "below_quality_floor"
NO_QUALITY = "quality_not_measured"
NO_OBSERVATIONS = "no_observations"
WEIGHTS_ABSENT = "weights_absent"
OK = "ok"

# Bytes taken from each SHA-256 digest. 8 gives a 64-bit accumulator, ample
# for a deterministic test double; this value has no statistical significance.
DIGEST_BYTES = 8

# Score quantisation. 10_000 yields 4 decimal places, enough to distinguish
# payloads without implying precision this stand-in does not have.
SCORE_BUCKETS = 10_000


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


def filter_by_quality_floor(
    obs: Sequence[Observation],
    min_quality_band: Literal["low", "medium", "high"],
) -> tuple[list[Observation], str | None]:
    """Select observations meeting the floor, and diagnose why none did.

    Returns (usable, abstention_reason). When `usable` is non-empty the
    reason is None. The diagnosis rule is deliberate and must not be
    reduced to inspecting obs[0]: if ANY observation carried a measured
    quality that failed the floor, the reason is BELOW_FLOOR, because
    "I measured it and it was too poor" is strictly more actionable than
    "I could not measure it". NO_QUALITY applies only when no observation
    carried quality at all; NO_OBSERVATIONS only when the list is empty.

    This lives in one place because the order-dependent version of this
    logic has already shipped and been fixed twice, in two separate files
    (base.SyntheticDetector during Task 6's review, then again independently
    in npr.NPRDetector). A third copy landed in effnet.EffNetDetector before
    this function existed; all three now call this instead.

    Args:
        obs: sequence of observations to filter.
        min_quality_band: minimum quality band a measured observation must
            meet to be considered usable.

    Returns:
        A (usable, reason) pair. `usable` is the list of observations
        meeting the floor (possibly empty). `reason` is None when `usable`
        is non-empty; otherwise one of NO_OBSERVATIONS, NO_QUALITY, or
        BELOW_FLOOR.
    """
    if not obs:
        logger.debug("filter_by_quality_floor: empty observation sequence")
        return [], NO_OBSERVATIONS

    usable: list[Observation] = []
    has_measured_below_floor = False

    for o in obs:
        if o.quality is None:
            logger.debug("observation has quality=None; skipping")
            continue
        if meets_floor(o.quality.band, min_quality_band):
            usable.append(o)
        else:
            logger.debug(
                "observation band %s below floor %s; skipping",
                o.quality.band, min_quality_band)
            has_measured_below_floor = True

    if usable:
        return usable, None

    reason = BELOW_FLOOR if has_measured_below_floor else NO_QUALITY
    logger.debug("no usable observations; abstaining: %s", reason)
    return usable, reason


class Detector(Protocol):
    """Contract for a detector: a callable that scores observations.

    Detectors are hot-swappable and expected to decay in 3-6 months.
    They must implement the Detector protocol to be registered in the harness.

    Identity attributes (name, version, modalities, min_quality_band) are
    declared as read-only properties to prevent mutation after registration,
    which would break the registry's identity invariant.
    """
    @property
    def name(self) -> str:
        """Unique detector identifier. Read-only after registration."""
        ...

    @property
    def slot(self) -> str:
        """Spec §6 physics slot this detector occupies (e.g. "A", "C", "E").

        Slot-diversity of the default registry is a tested invariant
        (`tests/test_registry.py`), so it must be readable on every
        detector, not only the three concrete ones that happen to declare
        it today. Read-only after registration, like `name`.
        """
        ...

    @property
    def version(self) -> str:
        """Detector version for comparison across updates. Read-only."""
        ...

    @property
    def modalities(self) -> frozenset[Modality]:
        """Which modalities this detector can process. Read-only."""
        ...

    @property
    def min_quality_band(self) -> Literal["low", "medium", "high"]:
        """Quality floor declaration. Read-only after registration."""
        ...

    def score(self, obs: Sequence[Observation]) -> RawScore:
        """Score one or more observations.

        Args:
            obs: sequence of observations to score

        Returns:
            RawScore with score (or None if abstained) and reason if abstained
        """
        ...


@dataclass(frozen=True)
class SyntheticDetector:
    """A deterministic stand-in used to test the harness without model weights.

    Its score is a hash of the observation payload, so it is reproducible and
    label-blind (never sees Context.label, only the pixel payload). It exists
    so the benchmark can be proven correct before any dataset or weight file
    arrives.

    Determinism guarantee: same observation payload plus same seed always
    yields the same score. This makes it suitable for repeatable testing.

    Frozen to prevent mutation after registration, which would break the
    registry's identity invariant.
    """
    name: str
    seed: int = 0
    version: str = "synthetic-1"
    # Not one of the spec's real slot letters ("A"/"C"/"E"): this stand-in
    # is registered directly in many tests (`reg.register(SyntheticDetector
    # (...))`) and this class's own docstring claims it "must implement the
    # Detector protocol to be registered in the harness", so it carries a
    # `slot` like every other conforming detector rather than leaving that
    # claim false now that the Protocol declares one.
    slot: str = "synthetic"
    modalities: frozenset[Modality] = field(
        default_factory=lambda: frozenset({Modality.IMAGE, Modality.VIDEO}))
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
        usable, reason = filter_by_quality_floor(obs, self.min_quality_band)
        if reason is not None:
            return abstain(self.name, self.version, reason)

        # Hash the payload deterministically using seed.
        # Use addition (not XOR) so duplicates accumulate rather than annihilate.
        acc = 0
        for o in usable:
            h = hashlib.sha256(np.ascontiguousarray(o.payload).tobytes())
            h.update(str(self.seed).encode())
            acc = (acc + int.from_bytes(h.digest()[:DIGEST_BYTES], "big")) % (2 ** 64)

        score = (acc % SCORE_BUCKETS) / SCORE_BUCKETS
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

        The detector's name is snapshotted at registration time to prevent
        later mutation from breaking the identity invariant (registry maps
        name -> detector, so if a detector rebinds its name, the key becomes
        stale and registry.get(old_name) returns an object calling itself
        by a different name).

        Args:
            detector: detector instance to register

        Raises:
            ValueError: if a detector with the same name is already registered
        """
        # Snapshot the name at registration time, never re-read from detector
        name = detector.name
        if name in self._d:
            logger.error("detector already registered: %s", name)
            raise ValueError(f"detector already registered: {name}")
        self._d[name] = detector
        logger.debug("registered detector: %s", name)

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
            k: number of detectors to select. Must be >= 0; negative values raise ValueError.
            seed: random seed for reproducibility

        Returns:
            sorted list of k detector names (or fewer if registry has fewer)

        Raises:
            ValueError: if k < 0
        """
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")

        names = self.names()
        k = min(k, len(names))
        rng = np.random.default_rng(seed)
        subset = rng.choice(names, size=k, replace=False).tolist()
        result = sorted(subset)
        logger.debug("selected subset of %d detectors from %d using seed %d",
                     len(result), len(names), seed)
        return result
