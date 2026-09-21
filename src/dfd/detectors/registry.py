"""Re-export so callers import the registry from an obvious place."""
from __future__ import annotations

from pathlib import Path

from .base import Detector, Registry, SyntheticDetector, abstain  # noqa: F401
from .effnet import EffNetDetector
from .npr import NPRDetector

#: Where a deployment is expected to place weights. Both files are gitignored
#: and absent in this repo, so both detectors abstain with `weights_absent`.
DEFAULT_NPR_WEIGHTS = Path("assets/models/npr.pt")
DEFAULT_EFFNET_WEIGHTS = Path("assets/models/effnet_b4_ffpp.pt")


def default_registry(npr_weights: str | Path = DEFAULT_NPR_WEIGHTS,
                     effnet_weights: str | Path = DEFAULT_EFFNET_WEIGHTS) -> Registry:
    """The detector set the CLI consults.

    Two distinct physics, per spec §6: NPR's upsampling fingerprint (slot C)
    and EfficientNet-B4's learned appearance (slot E). Both abstain when their
    weights are absent, which is every checkout of this repo.
    """
    registry = Registry()
    registry.register(NPRDetector(weights_path=npr_weights))
    registry.register(EffNetDetector(name="effnet_b4", slot="E",
                                     weights_path=effnet_weights))
    return registry
