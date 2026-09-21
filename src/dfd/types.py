"""Core value types. Every module in the engine speaks these and nothing else."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt


class Modality(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    LIVE = "live"


class Verdict(str, Enum):
    REAL = "real"
    FAKE = "fake"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    OUT_OF_DISTRIBUTION = "out_of_distribution"


# Ordered worst → best. Used for floor comparisons.
QUALITY_BANDS = ("reject", "low", "medium", "high")


@dataclass(frozen=True)
class Quality:
    inter_ocular_px: float
    blur_var: float
    yaw_deg: float
    pitch_deg: float
    exposure: float
    band: str


@dataclass(frozen=True)
class Observation:
    t: float
    payload: npt.NDArray[np.uint8]
    roi: tuple[int, int, int, int] | None
    quality: Quality | None
    source_id: str


@dataclass(frozen=True)
class Context:
    subject_id: str | None = None
    generator: str | None = None
    compression: str | None = None
    label: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    modality: Modality
    observations: tuple[Observation, ...]
    context: Context


@dataclass(frozen=True)
class RawScore:
    """What a detector emits. Uncalibrated and not comparable across detectors."""
    detector: str
    version: str
    score: float | None
    abstained: bool
    reason: str
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Evidence:
    """A calibrated contribution to the decision. llr is in nats; 0.0 = no information."""
    detector: str
    detector_version: str
    llr: float
    raw_score: float | None
    uncertainty: float
    abstained: bool
    reason: str
    artifacts: dict[str, Any] = field(default_factory=dict)
