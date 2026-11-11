"""Slot D — learned appearance read from a general-purpose encoder (spec §5.1).

Physics: not a handcrafted artefact at all. CLIP's image embedding is a general visual
representation, and the claim GenD (WACV 2026) makes -- 0.912-0.916 mean AUROC over 14 datasets
-- is that a LINEAR read-out of it separates generated faces from photographed ones better than
any single artefact detector generalises. That is a different failure mode from slots A, C and E,
which is the whole point of giving it its own slot letter.

WHAT IS OURS AND WHAT IS NOT. The encoder is OpenAI's CLIP ViT-L/14 under MIT
(`assets/manifest.yaml: clip_vit_l14_visual`). Everything fitted -- the head here, and in Phase 2
the LayerNorm parameters -- is fitted in this repo on FairFace-derived material and inherits CC BY
4.0. GenD's own released weights are trained on FaceForensics++ and are NEVER fetched, loaded or
initialised from: non-commercial research terms make a model fitted on them derived data, which is
the same bar that disqualified `sbi_effnetb4_weights` in this repo's manifest.

TWO WEIGHT FILES, TWO ABSENCES. The encoder and the head are separate assets with separate
licences and separate remedies, so they abstain with separate reasons.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from ..types import Modality, Observation, RawScore
from .base import (
    NO_OBSERVATIONS,
    OK,
    WEIGHTS_ABSENT,
    abstain,
    filter_by_quality_floor,
)
from .clip_vit import CLIPVisualTower, preprocess
from .loading import load_model

logger = logging.getLogger(__name__)

#: The embedding width `GendHead` reads. CLIP ViT-L/14's projected output.
EMBED_DIM = 768

#: Fixed width of the version buffer. Fixed, because `load_state_dict` is
#: strict about buffer shapes and a variable-length one could not be loaded
#: at all. 32 bytes holds "0.1.0-probe-fairface60000" with room.
VERSION_BYTES = 32

#: The observation carried no usable ROI. Mirrors `blend.NO_ROI` and
#: `npr.NO_ROI` deliberately -- one condition, one name across the codebase.
NO_ROI = "roi_absent"

#: The HEAD is present but the encoder file is not. Distinct from
#: WEIGHTS_ABSENT because the remedies differ: fit the head, versus run
#: ops/fetch-assets.sh and training/convert_clip.py.
ENCODER_ABSENT = "encoder_absent"

#: Reported when there is no weights file to read a version from. Describes
#: the CODE -- the preprocessing and the read-out -- never the coefficients.
CODE_VERSION = "0.1.0-unfitted"


class GendHead(nn.Module):
    """Everything this project fits in Phase 1: one linear read-out.

    `Linear(768, 2)` over an L2-normalised CLIP embedding, which is 1,538
    numbers. The encoder stays frozen and is not part of this file, so the
    shipped artefact's licence is FairFace's and the encoder's stays MIT.

    THE VERSION IS A BUFFER, NOT METADATA BESIDE THE FILE. `bench/eval_swaps.py`
    refuses a corpus drawn from a detector's own training window, and it keys
    that refusal on the detector's VERSION string. A version living in a
    sidecar file is a version that can be separated from the coefficients it
    describes, and the first thing anyone does with a weights file is copy it
    somewhere else.
    """

    version_bytes: torch.Tensor

    def __init__(self, in_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, 2)
        self.register_buffer("version_bytes",
                             torch.zeros(VERSION_BYTES, dtype=torch.uint8))

    def set_version(self, value: str) -> None:
        """Record the fitted version. Refuses to truncate.

        Raises:
            ValueError: if the encoded version does not fit. Truncating would
                leave a file declaring a training window that is a prefix of
                the real one -- which reads as a smaller window and lets a
                contaminated evaluation corpus past the offset guard.
        """
        raw = value.encode("utf-8")
        if len(raw) > VERSION_BYTES:
            raise ValueError(
                f"version {value!r} is {len(raw)} bytes, longer than the "
                f"{VERSION_BYTES}-byte buffer; it would be truncated")
        buf = torch.zeros(VERSION_BYTES, dtype=torch.uint8)
        buf[:len(raw)] = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        self.version_bytes = buf

    def version(self) -> str:
        """The recorded version, or "" when none was set."""
        return bytes(self.version_bytes.tolist()).rstrip(b"\x00").decode("utf-8")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.linear(x)
        return out


def embed(tower: nn.Module, crops: Sequence[npt.NDArray[np.uint8]]) -> torch.Tensor:
    """L2-normalised embeddings of face crops, `(N, EMBED_DIM)`.

    Shared by the detector and `training/clip_features.py` on purpose: a
    fitter that normalised and a detector that did not would produce a model
    that loads clean, scores confidently and is wrong -- the defect
    `NPRStatsNet` already carries a comment about.
    """
    size = int(getattr(tower, "image_size", 224))
    batch = torch.cat([preprocess(c, size=size) for c in crops])
    with torch.no_grad():
        out: torch.Tensor = tower(batch)
    normalised: torch.Tensor = out / out.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return normalised


def _crop_to_roi(payload: npt.NDArray[np.uint8],
                 roi: tuple[int, int, int, int] | None,
                 ) -> npt.NDArray[np.uint8] | None:
    """Clamp to the face box, or None.

    No resize here -- `preprocess` does it bicubically at the encoder's own
    input size. Unlike slot C, resampling is not destructive for this slot
    (the encoder resizes to 224 regardless), but it must happen exactly once
    and with CLIP's kernel.
    """
    if roi is None:
        return None
    h, w = payload.shape[:2]
    x, y, rw, rh = roi
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(w, int(x) + int(rw)), min(h, int(y) + int(rh))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return np.ascontiguousarray(payload[y0:y1, x0:x1])


@lru_cache(maxsize=8)
def _version_of(path: str, fingerprint: tuple[int, int]) -> str:
    """The version recorded inside the head at `path`, or CODE_VERSION.

    `fingerprint` is (mtime_ns, size) and is never read: it is in the cache
    key so refitting invalidates the entry. Caching on the path alone would
    pin the first version ever read for the life of the process, and the
    fitter writes to a FIXED path -- so stale would be the normal case.
    Never raises: this is read on the abstention path too.
    """
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        buf = sd["version_bytes"]
        return bytes(buf.tolist()).rstrip(b"\x00").decode("utf-8") or CODE_VERSION
    except Exception:
        logger.debug("cannot read a version from %s", path, exc_info=True)
        return CODE_VERSION


@dataclass(frozen=True)
class GendClipDetector:
    """Slot D. Frozen, so the registry's name -> detector map cannot go stale."""

    weights_path: str | Path
    encoder_path: str | Path
    name: str = "gend_clip"
    slot: str = "D"
    modalities: frozenset[Modality] = field(
        default_factory=lambda: frozenset({Modality.IMAGE, Modality.VIDEO}))
    #: MEASURED 2026-11-11, not chosen. It was provisionally "low" on the
    #: reasoning that an appearance read-out has no use for a blurred face.
    #: The per-band table on the capture corpus (bench/gend_vcip.json,
    #: `encoder_matched`, 509 frames over 125 sessions) says otherwise:
    #:
    #:     band      n   genuine  swapped   frame AUC   TPR@0%FPR
    #:     reject   232      14       218     0.541       0.165
    #:     medium   238     139        99     0.573       0.000
    #:     low       12       2        10     0.700       0.400
    #:     high      27      27         0      none        none
    #:
    #: **The argument rests on the two denominator columns, not on the AUC
    #: column.** A floor above `reject` discards 218 of 328 swapped frames
    #: (66.5%) against 14 of 182 genuine (7.7%) — two thirds of the attacks
    #: thrown away before the detector runs, and almost none of the honest
    #: traffic. That is a security hole, not caution, and it holds whatever
    #: the per-band AUCs turn out to be. Slot C's floor is `reject` for the
    #: same reason and from the same corpus.
    #:
    #: `high` carries 27 genuine frames and no swapped one, so it has no AUC
    #: and could not support a floor even if the rest of the table did.
    #: `low`'s 0.700 is the best cell in the table and is over 12 frames; it
    #: is not a basis for anything.
    min_quality_band: Literal["reject", "low", "medium", "high"] = "reject"
    #: The architectures the two files load into. Fields rather than
    #: hardcoded calls for the same reason `NPRDetector.model_factory` is a
    #: field: `loading.load_model`'s secure path takes a factory, and a test
    #: (or a Phase 2 tuned encoder) must be able to name a different one
    #: without reaching into this module. Defaults are the shipped pair.
    encoder_factory: Callable[[], nn.Module] = CLIPVisualTower.vit_l14
    head_factory: Callable[[], nn.Module] = GendHead

    @property
    def version(self) -> str:
        p = Path(self.weights_path)
        try:
            st = p.stat()
        except OSError:
            return CODE_VERSION
        return _version_of(str(p), (st.st_mtime_ns, st.st_size))

    def score(self, obs: Sequence[Observation]) -> RawScore:
        """Embed every usable face crop, read the head, return mean P(fake).

        Mean is the official score and max rides in artifacts, matching
        `NPRDetector`: a reader comparing two slots on one record must not
        have to discover they aggregate differently.
        """
        if not obs:
            return abstain(self.name, self.version, NO_OBSERVATIONS)
        if not Path(self.weights_path).exists():
            return abstain(self.name, self.version, WEIGHTS_ABSENT)
        if not Path(self.encoder_path).exists():
            logger.debug("encoder missing at %s; abstaining", self.encoder_path)
            return abstain(self.name, self.version, ENCODER_ABSENT)

        usable, reason = filter_by_quality_floor(obs, self.min_quality_band)
        if reason is not None:
            return abstain(self.name, self.version, reason)

        maybe = [_crop_to_roi(o.payload, o.roi) for o in usable]
        if any(c is None for c in maybe):
            return abstain(self.name, self.version, NO_ROI)
        crops = [c for c in maybe if c is not None]

        tower = load_model(self.encoder_path,
                           model_factory=self.encoder_factory)
        head = load_model(self.weights_path, model_factory=self.head_factory)
        with torch.no_grad():
            probs = torch.softmax(head(embed(tower, crops)), dim=1)[:, 1]

        per_obs = probs.cpu().numpy().tolist()
        return RawScore(
            detector=self.name, version=self.version,
            score=float(probs.mean()), abstained=False, reason=OK,
            artifacts={"n_observations": len(usable),
                       "per_observation_scores": per_obs,
                       "max_score": float(probs.max())})
