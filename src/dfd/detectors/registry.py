"""Re-export so callers import the registry from an obvious place."""
from __future__ import annotations

import logging
from pathlib import Path

from .base import Detector, Registry, SyntheticDetector, abstain  # noqa: F401
from .blend import DEFAULT_BLEND_WEIGHTS, BlendDetector
from .effnet import EffNetDetector
from .gend import GendClipDetector
from .npr import NPRDetector, NPRStatsNet

logger = logging.getLogger(__name__)

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

#: Phase 2's pair (plan Task 13). Measured on the capture corpus at session
#: AUC **0.7594** against the frozen pair's **0.663**, and the plan's
#: pre-registered rule returned "tuned encoder ships: YES". Preferred over
#: the frozen pair when BOTH files are present.
DEFAULT_GEND_LNTUNED_WEIGHTS = Path("assets/models/gend_clip_lntuned.pt")
DEFAULT_CLIP_LNTUNED_WEIGHTS = Path("assets/models/clip_vit_l14_lntuned.pt")

#: Slot D's calibration, in the two flavours that match its two weight pairs.
#: `training/fit_calibration.py` writes one file per pair; a curve is fitted on
#: a specific model's scores and means nothing applied to another's.
DEFAULT_CALIBRATION = Path("assets/models/calibration.json")
DEFAULT_CALIBRATION_LNTUNED = Path("assets/models/calibration_lntuned.json")


def slot_d_pair() -> tuple[Path, Path]:
    """`(head, encoder)` for slot D: Phase 2's pair when it is installed.

    BOTH OR NEITHER, and that is the whole rule. The tuned head was fitted
    JOINTLY with the tuned encoder, so a tuned head read through the frozen
    encoder loads cleanly, scores confidently, and describes a model nobody
    fitted. `bench/eval_swaps.py` and `training/fit_calibration.py` refuse
    that combination at the command line; this refuses it by never
    assembling it.

    SHIPPING THE TUNED PAIR DOES NOT MAKE SLOT D DECIDE. The evidence gate
    reads `bench/evidence_card.json`, which still records 0.663, so slot D
    still abstains with `uncalibrated_for_band` on every request. What
    changes is which model produces the raw score sitting beside that
    abstention — the better one, by the only unbiased comparison available.
    A deployment that has not run Phase 2 simply does not have these files
    and keeps the frozen pair; `AuditRecord`'s per-detector version string
    says which was used, so the two are never confused after the fact.
    """
    head, encoder = DEFAULT_GEND_LNTUNED_WEIGHTS, DEFAULT_CLIP_LNTUNED_WEIGHTS
    if head.is_file() and encoder.is_file():
        logger.info("slot D: using the Phase 2 LayerNorm-tuned pair (%s, %s)",
                    encoder.name, head.name)
        return head, encoder
    if head.is_file() or encoder.is_file():
        # Half a pair is the dangerous state, so it is the loud one.
        logger.warning(
            "slot D: only one half of the tuned pair is present (%s=%s, "
            "%s=%s); falling back to the frozen pair. A tuned head read "
            "through the frozen encoder is a model nobody fitted.",
            encoder.name, encoder.is_file(), head.name, head.is_file())
    return DEFAULT_GEND_WEIGHTS, DEFAULT_CLIP_WEIGHTS


def slot_d_calibration() -> Path | None:
    """The calibration file matching whichever pair `slot_d_pair` resolved.

    IT RESOLVES FROM `slot_d_pair`, NOT FROM ITS OWN `is_file` CHECKS, and
    that is the point. A calibration curve is fitted on one model's scores:
    `coef` and `intercept` describe where THAT model puts reals and fakes.
    Applied to a different model's scores it still produces a confident
    number, and the number is wrong.

    This is PR #9's defect wearing different clothes. There, `ServiceConfig`
    defaulted the weight paths to the frozen constants and passed them over
    the top of `slot_d_pair`. Here, `ops/dfd.service` passed
    `--calibration assets/models/calibration.json` as a literal while the
    registry resolved the tuned pair underneath it, so the tuned model's
    scores were read through the frozen model's curve. Measured on the
    instance 2026-11-12: raw 0.008388 in the `reject` band reads llr -3.8441
    under the frozen curve against -4.0174 under its own. Small on that
    frame, systematic across every frame, and invisible in the output --
    both numbers look equally confident.

    Returns:
        The tuned curve when the tuned pair is installed, the frozen curve
        when it is not, and `None` when the tuned pair is installed but its
        curve is absent.

    `None` RATHER THAN A FALLBACK. Returning the frozen curve for a tuned
    pair is the defect this function exists to prevent, so the missing-curve
    case must not land back on it. No calibration makes slot D abstain with
    `uncalibrated_for_band` -- visible and harmless, which is the direction
    this project fails in everywhere else.
    """
    head, _ = slot_d_pair()
    if head != DEFAULT_GEND_LNTUNED_WEIGHTS:
        return DEFAULT_CALIBRATION
    if DEFAULT_CALIBRATION_LNTUNED.is_file():
        logger.info("slot D: using the Phase 2 calibration %s",
                    DEFAULT_CALIBRATION_LNTUNED.name)
        return DEFAULT_CALIBRATION_LNTUNED
    logger.warning(
        "slot D: the tuned pair is installed but its calibration %s is "
        "absent. Running UNCALIBRATED rather than borrowing %s, which was "
        "fitted on the frozen model's scores and would describe a model "
        "nobody fitted.",
        DEFAULT_CALIBRATION_LNTUNED.name, DEFAULT_CALIBRATION.name)
    return None


def default_registry(npr_weights: str | Path = DEFAULT_NPR_WEIGHTS,
                     effnet_weights: str | Path = DEFAULT_EFFNET_WEIGHTS,
                     blend_weights: str | Path = DEFAULT_BLEND_WEIGHTS,
                     gend_weights: str | Path | None = None,
                     clip_weights: str | Path | None = None) -> Registry:
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
    # Resolved together, never independently: see `slot_d_pair`. A caller
    # that overrides one and not the other gets the other's DEFAULT, which
    # would be the mismatched-pair bug by a side door, so both must be given
    # or neither.
    if (gend_weights is None) != (clip_weights is None):
        raise ValueError(
            "gend_weights and clip_weights go together: a head fitted "
            "jointly with a tuned encoder, read through a different "
            "encoder, is a model nobody fitted")
    if gend_weights is None or clip_weights is None:
        gend_weights, clip_weights = slot_d_pair()

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
