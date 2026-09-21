"""Slot A — blending boundary (spec §6).

Physics: a face swap composites a generated inner face onto a real outer face.
However well the colours are matched, the two halves came from different
imaging chains, so their high-frequency residual, local sharpness and colour
statistics differ, and the difference is concentrated on a closed curve
somewhere inside the face. A camera producing a single exposure has no such
curve.

The detector does not know where the seam is. It does not need to: in an
aligned crop the inner face sits near the centre, so statistics computed over
concentric elliptical annuli will straddle the seam wherever it falls, and the
ADJACENT-BAND CONTRASTS — not the raw band values — carry the signal. That is
why the feature vector ends with log-ratios and differences rather than only
per-band means.

This occupies the slot the spec assigns to SBI, with different machinery. The
spec names a CNN trained on self-blended images; hardware here is CPU-only
(docs/HANDOFF.md §1), which rules out training EfficientNet-B4, so the
learned part is a linear model over handcrafted statistics and the
self-blending moves into corpus construction (corpora/sbi.py). The physics
read is the same; the capacity is much lower, and the honest expectation is
correspondingly lower accuracy.

IMPORTANT: the annulus boundaries and the feature set are this project's own
design, not a reimplementation of a published method, and are UNVERIFIED
against any baseline. No published claim may rest on them without measurement.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt

from ..types import Modality, Observation, RawScore
from .base import OK, WEIGHTS_ABSENT, abstain, filter_by_quality_floor

logger = logging.getLogger(__name__)

#: Normalised elliptical radii bounding each annulus, where 1.0 is the
#: half-width of the crop. The outermost band runs past 1.0 to take in the
#: corners, which would otherwise be measured by nothing.
ANNULI: tuple[tuple[float, float], ...] = (
    (0.00, 0.45),
    (0.45, 0.65),
    (0.65, 0.85),
    (0.85, 1.45),
)

#: Gaussian kernel for the high-pass residual. Small, so the residual keeps
#: the seam's spatial frequency rather than smearing it into the whole face.
RESIDUAL_KERNEL = 5

_PER_BAND = ("residual_mean", "residual_std", "laplacian_var",
             "lab_l_mean", "lab_a_mean", "lab_b_mean")


def _feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for b in range(len(ANNULI)):
        names.extend(f"{stat}_b{b}" for stat in _PER_BAND)
    for b in range(len(ANNULI) - 1):
        names.append(f"residual_logratio_b{b}_b{b + 1}")
    for b in range(len(ANNULI) - 1):
        names.append(f"lab_l_delta_b{b}_b{b + 1}")
    return tuple(names)


FEATURE_NAMES: tuple[str, ...] = _feature_names()


def seam_features(img: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """Compute blending-boundary statistics over concentric annuli.

    Args:
        img: aligned RGB face crop, HWC uint8.

    Returns:
        float32 vector of length `len(FEATURE_NAMES)`, aligned to it
        positionally. Always finite: an empty band contributes zeros rather
        than nan, because a nan would poison the linear model silently while
        a zero is merely uninformative.

    Raises:
        ValueError: if `img` is not a three-channel HWC array.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"seam_features expects an HWC RGB image, got shape {img.shape}")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float64)
    blurred = cv2.GaussianBlur(gray, (RESIDUAL_KERNEL, RESIDUAL_KERNEL), 0)
    residual = np.abs(gray.astype(np.float64) - blurred.astype(np.float64))
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    yy, xx = np.mgrid[0:h, 0:w]
    ry = (yy - (h - 1) / 2.0) / max(1.0, (h - 1) / 2.0)
    rx = (xx - (w - 1) / 2.0) / max(1.0, (w - 1) / 2.0)
    radius = np.sqrt(rx ** 2 + ry ** 2)

    per_band: list[list[float]] = []
    for lo, hi in ANNULI:
        mask = (radius >= lo) & (radius < hi)
        if not mask.any():
            logger.debug("annulus [%.2f, %.2f) is empty at %dx%d", lo, hi, h, w)
            per_band.append([0.0] * len(_PER_BAND))
            continue
        per_band.append([
            float(residual[mask].mean()),
            float(residual[mask].std()),
            float(laplacian[mask].var()),
            float(lab[..., 0][mask].mean()),
            float(lab[..., 1][mask].mean()),
            float(lab[..., 2][mask].mean()),
        ])

    feats: list[float] = [v for band in per_band for v in band]

    # The contrast terms. log1p keeps the ratio finite when a band's residual
    # is zero, which happens on synthetic flat fields and would otherwise
    # divide by zero.
    residual_means = [band[0] for band in per_band]
    lab_l_means = [band[3] for band in per_band]
    feats.extend(float(np.log1p(residual_means[b]) - np.log1p(residual_means[b + 1]))
                 for b in range(len(ANNULI) - 1))
    feats.extend(float(lab_l_means[b] - lab_l_means[b + 1])
                 for b in range(len(ANNULI) - 1))

    return np.asarray(feats, dtype=np.float32)


#: On-disk format version for the model file. Bump when the array set changes.
MODEL_FILE_VERSION = 1

#: Where a deployment is expected to place the fitted model. Gitignored and
#: absent in this repo, so the detector abstains on every fresh checkout —
#: the same contract NPR and EffNet already keep.
DEFAULT_BLEND_WEIGHTS = Path("assets/models/blend_seam.npz")


@dataclass(frozen=True)
class BlendModel:
    """A standardiser and a linear model, as plain arrays.

    Deliberately not a pickled sklearn estimator. `joblib.load` and
    `pickle.load` execute arbitrary code from the file they read, and a
    detector's weight file is exactly the artefact an attacker would swap.
    scikit-learn fits this model (see training/fit_blend.py) and is then
    discarded: only the numbers are kept, and `np.load(..., allow_pickle=False)`
    cannot execute anything.
    """
    mean: npt.NDArray[np.float64]
    scale: npt.NDArray[np.float64]
    coef: npt.NDArray[np.float64]
    intercept: float
    feature_names: tuple[str, ...]
    version: str

    def predict_proba(self, features: npt.NDArray[np.float32]) -> float:
        """P(fake) for one feature vector.

        Args:
            features: vector aligned to `feature_names`.

        Returns:
            A probability in [0, 1].
        """
        z = (features.astype(np.float64) - self.mean) / np.where(
            self.scale == 0.0, 1.0, self.scale)
        logit = float(np.dot(z, self.coef) + self.intercept)
        return float(1.0 / (1.0 + np.exp(-logit)))


def save_blend_model(model: BlendModel, path: str | Path) -> None:
    """Write the model as an npz of plain arrays."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        p,
        format_version=np.array(MODEL_FILE_VERSION),
        mean=model.mean, scale=model.scale, coef=model.coef,
        intercept=np.array(model.intercept),
        feature_names=np.array(model.feature_names, dtype=np.str_),
        version=np.array(model.version, dtype=np.str_),
    )
    logger.info("wrote blend model v%s to %s", model.version, p)


def load_blend_model(path: str | Path) -> BlendModel:
    """Read a model file, refusing one that does not match this code.

    Args:
        path: npz written by `save_blend_model`.

    Returns:
        The model.

    Raises:
        ValueError: if the file's format version or feature names disagree
            with this build. A stale model file is worse than none: it would
            score confidently against columns that no longer mean what it
            was fitted on.
    """
    data = np.load(Path(path), allow_pickle=False)
    version = int(data["format_version"])
    if version != MODEL_FILE_VERSION:
        raise ValueError(
            f"blend model format version {version} != {MODEL_FILE_VERSION}")
    names = tuple(str(n) for n in data["feature_names"])
    if names != FEATURE_NAMES:
        raise ValueError(
            "blend model feature names do not match this build: "
            f"file has {len(names)}, code expects {len(FEATURE_NAMES)}")
    return BlendModel(
        mean=np.asarray(data["mean"], dtype=np.float64),
        scale=np.asarray(data["scale"], dtype=np.float64),
        coef=np.asarray(data["coef"], dtype=np.float64),
        intercept=float(data["intercept"]),
        feature_names=names,
        version=str(data["version"]),
    )


@dataclass(frozen=True)
class BlendDetector:
    """Slot A. Scores the blending seam with a linear model over seam features.

    Frozen, like every other detector, so the registry's name→detector
    invariant cannot be broken by mutating identity after registration.
    """
    weights_path: str | Path = DEFAULT_BLEND_WEIGHTS

    name: str = "blend_seam"
    slot: str = "A"
    version: str = "0.1.0"
    modalities: frozenset[Modality] = field(
        default_factory=lambda: frozenset({Modality.IMAGE, Modality.VIDEO}))
    min_quality_band: Literal["low", "medium", "high"] = "medium"

    def score(self, obs: Sequence[Observation]) -> RawScore:
        """Score observations by their mean seam probability.

        Args:
            obs: observations to score.

        Returns:
            A RawScore. Abstains with `weights_absent` when the model file is
            missing, or with the quality-floor reason when nothing is usable.
        """
        usable, reason = filter_by_quality_floor(obs, self.min_quality_band)
        if reason is not None:
            return abstain(self.name, self.version, reason)

        path = Path(self.weights_path)
        if not path.exists():
            logger.warning("blend model absent at %s", path)
            return abstain(self.name, self.version, WEIGHTS_ABSENT)

        model = load_blend_model(path)
        probs = [model.predict_proba(seam_features(o.payload)) for o in usable]
        score = float(np.mean(probs))
        return RawScore(detector=self.name, version=self.version, score=score,
                        abstained=False, reason=OK,
                        artifacts={"n_observations": len(probs),
                                   "model_version": model.version})
