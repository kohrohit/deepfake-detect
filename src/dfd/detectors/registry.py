"""Re-export so callers import the registry from an obvious place."""
from __future__ import annotations

from pathlib import Path

from .base import Detector, Registry, SyntheticDetector, abstain  # noqa: F401
from .blend import DEFAULT_BLEND_WEIGHTS, BlendDetector
from .effnet import EffNetDetector
from .gend import GendClipDetector
from .npr import NPRDetector, NPRStatsNet

#: Where a deployment is expected to place weights. Both files are gitignored
#: and absent in this repo, so both detectors abstain with `weights_absent`.
DEFAULT_NPR_WEIGHTS = Path("assets/models/npr.pt")
DEFAULT_EFFNET_WEIGHTS = Path("assets/models/effnet_b4_ffpp.pt")
#: Slot D's two files. The head is fitted by training/fit_gend.py; the
#: encoder is fetched and converted (ops/fetch-assets.sh, then
#: training/convert_clip.py). Both gitignored, both absent on a fresh
#: checkout, and slot D names WHICH one is missing when it abstains.
DEFAULT_GEND_WEIGHTS = Path("assets/models/gend_clip.pt")
DEFAULT_CLIP_WEIGHTS = Path("assets/models/clip_vit_l14_visual.pt")


def default_registry(npr_weights: str | Path = DEFAULT_NPR_WEIGHTS,
                     effnet_weights: str | Path = DEFAULT_EFFNET_WEIGHTS,
                     blend_weights: str | Path = DEFAULT_BLEND_WEIGHTS,
                     gend_weights: str | Path = DEFAULT_GEND_WEIGHTS,
                     clip_weights: str | Path = DEFAULT_CLIP_WEIGHTS) -> Registry:
    """The detector set the CLI consults.

    Four distinct physics, per spec §6: NPR's upsampling fingerprint (slot C),
    the blending seam (slot A), EfficientNet-B4's learned appearance (slot E),
    and a linear read-out of CLIP's general embedding (slot D). All four
    abstain when their weights are absent, which is every fresh checkout of
    this repo — `blend_seam` is the first whose weights this project can
    actually produce, because it is fitted on a corpus manufactured from the
    project's own captures (corpora/sbi.py) rather than on a licensed dataset.

    Raises:
        ValueError: from `Registry.register` if a detector with the same name
            is already registered. Unreachable today: four distinct hardcoded
            names ("npr", "blend_seam", "effnet_b4", "gend_clip") cannot
            collide.
    """
    registry = Registry()
    # The factory is what keeps slot C on `loading.load_model`'s SECURE
    # path: with it, weights load under `weights_only=True` as a plain
    # state_dict. Without it the only way to load anything is a full
    # pickle, which is arbitrary code execution from a weights file and is
    # logged as a warning on every call.
    registry.register(NPRDetector(weights_path=npr_weights,
                                  model_factory=NPRStatsNet))
    registry.register(BlendDetector(weights_path=blend_weights))
    registry.register(EffNetDetector(name="effnet_b4", slot="E",
                                     weights_path=effnet_weights))
    # Slot D. TWO paths because the encoder (MIT, 304M parameters, fetched
    # and hash-pinned) and the head (fitted here, 1,538 numbers, CC BY 4.0 by
    # inheritance) are different assets with different licences, and the
    # manifest records them separately.
    registry.register(GendClipDetector(weights_path=gend_weights,
                                       encoder_path=clip_weights))
    return registry
