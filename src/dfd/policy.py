"""Decision thresholds as versioned configuration (spec §7.1).

These numbers used to be module constants in `fusion.py`, which meant the audit
record's `threshold` field was a *copy* of what fusion applied rather than the
thing itself. A copy drifts. `decide` hands one `Policy` to both `fuse` and
`build_audit_record`, so a record's stated threshold is provably the one that
produced its verdict.

The §7.1 economics — expected fraud loss against friction cost, and the
AUTO-PASS / STEP-UP / MANUAL REVIEW / BLOCK bands — are deliberately absent.
The loss and friction figures are still unsupplied, and bands fitted to
placeholder economics would be numbers nobody can defend.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import InvalidInput


@dataclass(frozen=True)
class Policy:
    """Thresholds applied to fused evidence, with a version for the record.

    Attributes:
        fake_threshold: llr_total at or above which the verdict is FAKE, in nats.
            Spec §7.1. 1.0 nat ≈ 73% posterior (logit(1.0) = 0.731). Conservative
            threshold to require multi-detector agreement or strong single evidence
            before claiming FAKE.
        real_threshold: llr_total at or below which the verdict is REAL, in nats.
            Spec §7.1. -1.0 nat ≈ 27% posterior (logit(-1.0) = 0.269). Symmetric
            with fake_threshold.
        disagreement_ood: disagreement at or above which the verdict is
            OUT_OF_DISTRIBUTION, overriding both thresholds. Spec §7.2. Evidence
            pulling hard in both directions means off-distribution, not 'average
            them'. Two detectors at full opposite confidence is the 6-of-10 split
            case the spec says vendors wrongly average away.
        version: identifier recorded in every audit record this policy decides.
    """

    fake_threshold: float = 1.0
    real_threshold: float = -1.0
    disagreement_ood: float = 3.0
    version: str = "p0-default-v0"

    def __post_init__(self) -> None:
        """Validate the thresholds at construction, not at decision time.

        Raises:
            InvalidInput: if any threshold is non-numeric or non-finite, if
                `real_threshold` is not strictly below `fake_threshold`, if
                `disagreement_ood` is not positive, or if `version` is not a
                non-empty string.
        """
        for name in ("fake_threshold", "real_threshold", "disagreement_ood"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise InvalidInput(
                    f"{name} must be a finite number, got {value!r}. A NaN "
                    "threshold compares false against everything, which turns "
                    "every verdict into INSUFFICIENT_EVIDENCE silently.")
        if self.real_threshold >= self.fake_threshold:
            raise InvalidInput(
                f"real_threshold ({self.real_threshold}) must be strictly below "
                f"fake_threshold ({self.fake_threshold}); otherwise the bands "
                "overlap and the verdict depends on comparison order.")
        if self.disagreement_ood <= 0:
            raise InvalidInput(
                f"disagreement_ood must be positive, got {self.disagreement_ood}. "
                "A non-positive trigger makes every decision OUT_OF_DISTRIBUTION, "
                "since disagreement is a minimum of two non-negative sums.")
        if not isinstance(self.version, str) or not self.version:
            raise InvalidInput(
                f"version must be a non-empty string, got {self.version!r}; it is "
                "what an auditor uses to reconstruct which policy decided a case.")


#: The thresholds this repo has applied since fusion was written.
DEFAULT_POLICY = Policy()
