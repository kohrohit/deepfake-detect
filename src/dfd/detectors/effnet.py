"""Slots A and E — EfficientNet-B4 backbone, parameterised by weights (spec §6).

Slot A (SBI, Self-Blended Images): trained by blending a face with a warped
copy of itself, so the network learns to read the *composite seam* a swapped
face leaves behind — a blending-boundary artifact, not an appearance one.

Slot E (FF++): trained on FaceForensics++ fakes, so the network learns
*learned appearance artifacts* — texture and generator fingerprints in the
synthesized face itself.

Same architecture, different physics: the physics lives entirely in the
training scheme (which weights are loaded), not in the layers. Keeping slots
A and E as one weights-parameterised class turns the A-vs-E benchmark
comparison into a clean weights comparison, with no architectural confound.

Supply-chain control: weights are loaded via detectors.loading.load_model,
which defaults to torch.load(..., weights_only=True) and only permits full
pickle loads when allow_unsafe_load is explicitly set (spec §3A).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn

from ..quality import meets_floor
from ..types import Modality, Observation, RawScore
from .base import BELOW_FLOOR, NO_OBSERVATIONS, NO_QUALITY, OK, WEIGHTS_ABSENT, abstain
from .loading import load_model

logger = logging.getLogger(__name__)

# Expected number of output classes (real/fake binary classifier). Both the
# SBI and FF++ weight sets are trained as 2-class (real, fake) classifiers.
EXPECTED_NUM_CLASSES = 2

# EfficientNet-B4's native input resolution, matching the published SBI and
# DeepfakeBench FF++ training configurations.
DEFAULT_INPUT_SIZE = 224


def preprocess(images: Sequence[np.ndarray], size: int = DEFAULT_INPUT_SIZE) -> np.ndarray:
    """Convert a sequence of RGB uint8 HWC images to a float32 NCHW batch in [0, 1].

    Non-square or mismatched-size inputs are resized to (size, size) with
    area interpolation, which is appropriate for both up- and down-sampling
    face crops without introducing ringing artifacts that could be confused
    with the physics either slot is meant to read.

    Args:
        images: sequence of uint8 HWC (RGB) arrays.
        size: target square side length the model expects.

    Returns:
        float32 array of shape (N, 3, size, size), values in [0, 1].

    Raises:
        No exceptions raised; inputs are assumed to be valid image arrays,
        as produced by the ingest pipeline.
    """
    out = []
    for img in images:
        if img.shape[0] != size or img.shape[1] != size:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        out.append(img.astype(np.float32) / 255.0)
    arr = np.stack(out)
    return np.ascontiguousarray(arr.transpose(0, 3, 1, 2))


@dataclass(frozen=True)
class EffNetDetector:
    """EfficientNet-B4 detector serving slots A (SBI) and E (FF++ appearance).

    Frozen to preserve the registry identity invariant: once registered, a
    detector's name must not drift from the key that indexes it, because
    audit records pin which model version produced a given verdict.

    Model weights are NOT stored on the instance (a frozen dataclass cannot
    hold mutable model state); instead `score` calls the shared, cached
    `detectors.loading.load_model` on every invocation, which is a cheap
    dict lookup after the first real load.

    Attributes:
        name: unique detector identifier.
        slot: "A" (SBI) or "E" (FF++ appearance) — the same architecture,
            different training weights.
        weights_path: path to the detector's trained weights file.
        model_factory: callable returning an uninitialized EfficientNet-B4
            instance. If provided, enables the secure weights_only=True
            load path (see detectors.loading.load_model). If None, only
            full-module pickles with allow_unsafe_load=True are accepted.
        allow_unsafe_load: if True, permits loading full-module pickles via
            unpickling (arbitrary code execution). Logs a WARNING when taken.
        version: detector version string.
        input_size: square side length the backbone expects.
        modalities: which modalities this detector supports.
        min_quality_band: minimum quality band ("low", "medium", or "high")
            an observation must meet to be scored. Never "reject" — that
            band means "unusable by anyone", so a "reject" floor would admit
            exactly the samples the quality gate exists to exclude.
    """
    name: str
    slot: Literal["A", "E"]
    weights_path: str | Path
    model_factory: Callable[[], nn.Module] | None = None
    allow_unsafe_load: bool = False
    version: str = "0.1.0"
    input_size: int = DEFAULT_INPUT_SIZE
    modalities: frozenset[Modality] = field(
        default_factory=lambda: frozenset({Modality.IMAGE, Modality.VIDEO})
    )
    min_quality_band: Literal["low", "medium", "high"] = "low"

    def score(self, obs: Sequence[Observation]) -> RawScore:
        """Score observations with the EfficientNet-B4 backbone.

        Filters observations by quality floor, preprocesses the survivors,
        runs them through the (cached) model, and returns a raw score or
        abstention.

        Args:
            obs: sequence of observations to score.

        Returns:
            RawScore with score (or None if abstained) and reason, plus
            artifacts containing per_observation_scores, max_score,
            n_observations, and slot.

        Raises:
            ValueError: if the model's output has an unexpected number of
                classes (not EXPECTED_NUM_CLASSES).
            RuntimeError: if torch.load fails with weights_only=True and
                allow_unsafe_load is False, or if model_factory and a
                full-module pickle are both supplied (see
                detectors.loading.load_model).
            Other exceptions from model inference propagate.
        """
        if not obs:
            logger.debug("score called with empty observation sequence; abstaining")
            return abstain(self.name, self.version, NO_OBSERVATIONS)

        weights_path = Path(self.weights_path)

        if not weights_path.exists():
            logger.debug(
                "weights file not found: %s; abstaining with %s",
                weights_path,
                WEIGHTS_ABSENT,
            )
            return abstain(self.name, self.version, WEIGHTS_ABSENT)

        # Filter observations by quality floor. An observation is usable only
        # if it carries measured quality that meets the floor. If ANY
        # observation carried measured quality that failed the floor, the
        # abstention reason is BELOW_FLOOR; NO_QUALITY is reported only when
        # NO observation carried quality at all. This must not be inferred
        # from obs[0] alone — that was an order-dependent bug fixed twice
        # already in this codebase (see base.SyntheticDetector, npr.NPRDetector).
        usable: list[Observation] = []
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

        if not usable:
            reason = BELOW_FLOOR if has_measured_below_floor else NO_QUALITY
            logger.debug("no usable observations; abstaining: %s", reason)
            return abstain(self.name, self.version, reason)

        try:
            model = load_model(weights_path, self.model_factory, self.allow_unsafe_load)
        except Exception as e:
            logger.error(
                "failed to load model from %s: %s",
                weights_path,
                type(e).__name__,
                exc_info=True,
            )
            raise

        try:
            batch = preprocess([o.payload for o in usable], self.input_size)
            t = torch.from_numpy(batch)

            with torch.no_grad():
                logits = model(t)  # type: ignore[operator]

            if logits.shape[1] != EXPECTED_NUM_CLASSES:
                raise ValueError(
                    f"expected {EXPECTED_NUM_CLASSES} output classes, "
                    f"got {logits.shape[1]} from model"
                )

            probs = torch.softmax(logits, dim=1)[:, 1]
            per_obs_scores = probs.cpu().numpy().tolist()
            max_score = float(probs.max())
            mean_score = float(probs.mean())

            logger.debug(
                "computed scores (max=%f, mean=%f) from %d observations, slot=%s",
                max_score,
                mean_score,
                len(usable),
                self.slot,
            )

            # score is the mean, but artifacts preserve per-observation detail:
            # averaging in `score` is fusion's job (Task 10), and a mean alone
            # would dilute a strong single-frame detection (one frame at 0.95
            # and four at 0.05 averages to 0.23 — exactly what a
            # partial-duration face swap produces).
            return RawScore(
                detector=self.name,
                version=self.version,
                score=mean_score,
                abstained=False,
                reason=OK,
                artifacts={
                    "per_observation_scores": per_obs_scores,
                    "max_score": max_score,
                    "n_observations": len(usable),
                    "slot": self.slot,
                },
            )
        except Exception as e:
            logger.error(
                "failed to score observations: %s",
                type(e).__name__,
                exc_info=True,
            )
            raise
