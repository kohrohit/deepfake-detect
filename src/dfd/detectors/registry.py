"""Re-export so callers import the registry from an obvious place."""
from __future__ import annotations

from pathlib import Path

from .base import Detector, Registry, SyntheticDetector, abstain  # noqa: F401
from .blend import DEFAULT_BLEND_WEIGHTS, BlendDetector
from .effnet import EffNetDetector
from .npr import NPRDetector

#: Where a deployment is expected to place weights. Both files are gitignored
#: and absent in this repo, so both detectors abstain with `weights_absent`.
DEFAULT_NPR_WEIGHTS = Path("assets/models/npr.pt")
DEFAULT_EFFNET_WEIGHTS = Path("assets/models/effnet_b4_ffpp.pt")


def default_registry(npr_weights: str | Path = DEFAULT_NPR_WEIGHTS,
                     effnet_weights: str | Path = DEFAULT_EFFNET_WEIGHTS,
                     blend_weights: str | Path = DEFAULT_BLEND_WEIGHTS) -> Registry:
    """The detector set the CLI consults.

    Three distinct physics, per spec §6: NPR's upsampling fingerprint (slot C),
    the blending seam (slot A), and EfficientNet-B4's learned appearance
    (slot E). All three abstain when their weights are absent, which is every
    fresh checkout of this repo — `blend_seam` is the first of the three whose
    weights this project can actually produce, because it is fitted on a corpus
    manufactured from the project's own captures (corpora/sbi.py) rather than
    on a licensed dataset.

    Raises:
        ValueError: from `Registry.register` if a detector with the same name
            is already registered. Unreachable today: three distinct hardcoded
            names ("npr", "blend_seam", "effnet_b4") cannot collide.
    """
    registry = Registry()
    registry.register(NPRDetector(weights_path=npr_weights))
    registry.register(BlendDetector(weights_path=blend_weights))
    registry.register(EffNetDetector(name="effnet_b4", slot="E",
                                     weights_path=effnet_weights))
    return registry
