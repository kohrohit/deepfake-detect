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

import inspect
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt

from ..faces import align
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

#: Every field save_blend_model writes, and load_blend_model must find.
#: Checked up front so a truncated or corrupted file raises ValueError
#: naming what is missing, rather than an undocumented KeyError from
#: whichever field happens to be dereferenced first.
_REQUIRED_KEYS = (
    "format_version", "mean", "scale", "coef", "intercept",
    "feature_names", "version",
)

#: Where a deployment is expected to place the fitted model. Gitignored and
#: absent in this repo, so the detector abstains on every fresh checkout —
#: the same contract NPR and EffNet already keep.
DEFAULT_BLEND_WEIGHTS = Path("assets/models/blend_seam.npz")

#: The observation carried no usable ROI: either `roi` was None, or it was
#: present but degenerate/out-of-bounds after clamping to the payload (see
#: `_crop_to_roi`). Distinct from NO_QUALITY/BELOW_FLOOR, which are about
#: capture quality rather than about knowing which pixels are the face, and
#: distinct from WEIGHTS_ABSENT, which means the detector never ran at all.
#: This exists because `seam_features` assumes an aligned face crop (module
#: docstring): scoring a whole frame instead would put the annuli over
#: mostly background, not face, and silently score the wrong pixels rather
#: than say so.
NO_ROI = "roi_absent"


@dataclass(frozen=True)
class BlendModel:
    """A standardiser and a linear model, as plain arrays.

    Deliberately not a pickled sklearn estimator. `joblib.load` and
    `pickle.load` execute arbitrary code from the file they read, and a
    detector's weight file is exactly the artefact an attacker would swap.
    scikit-learn fits this model (see training/fit_blend.py) and is then
    discarded: only the numbers are kept, and `np.load(..., allow_pickle=False)`
    cannot execute anything.

    `frozen=True` stops a caller from rebinding `model.coef = ...` or any
    other field, but it does NOT stop in-place mutation of the numpy arrays
    themselves (`model.coef[:] = ...` or `model.coef *= 2` both work fine
    despite the frozen dataclass). That gap is safe today only because
    `BlendDetector.score` calls `load_blend_model` fresh on every call — a
    new, unshared array set each time — and becomes unsafe the day anyone
    adds a load cache, as `dfd.detectors.loading.load_model` already does
    for NPR and EffNet: a cached `BlendModel` shared across calls would let
    one caller's in-place edit corrupt every other caller's scores.
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
        # A large-magnitude logit (a scaler mismatch on a real fitted model
        # could produce one) drives exp(-logit) to overflow to inf, which
        # numpy reports as a RuntimeWarning by default even though the
        # result (1/(1+inf) == 0.0) is correct either way. Suppress just
        # that warning path; nothing here is silently wrong, only loud.
        with np.errstate(over="ignore"):
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


#: The crop size training actually used. Derived from `dfd.faces.align`'s
#: own default rather than a duplicated literal, so the two can never
#: silently drift apart: `corpora/face_pool.py` builds every training crop
#: by calling `align(..., size=224)` (its own default), and the serving
#: path below must reproduce that transform, not merely approximate it.
_ALIGN_SIZE: int = inspect.signature(align).parameters["size"].default


def _crop_to_roi(
    payload: npt.NDArray[np.uint8],
    roi: tuple[int, int, int, int] | None,
) -> npt.NDArray[np.uint8] | None:
    """Crop `payload` to `roi`, clamped to the frame bounds, then resize.

    Clamping mirrors `dfd.faces.align`'s (`max(0, ...)` / `min(dim, ...)`,
    then a degenerate check) — but the two do NOT then handle a degenerate
    result the same way. `align` returns a black `_ALIGN_SIZE`-square image
    for a box that does not intersect the frame, and that image gets
    scored, with nothing to say it was synthetic. `_crop_to_roi` instead
    returns `None`, so `BlendDetector.score` abstains. Abstaining is the
    correct behaviour here — scoring a fabricated black square as though
    it were a face is worse than saying nothing — so this deliberately
    differs from `align` rather than matching it.

    After clamping, the crop is resized to `(_ALIGN_SIZE, _ALIGN_SIZE)`
    with `cv2.INTER_AREA`, exactly as `align` resizes every training crop.
    This is not optional. `seam_features` normalises its annulus
    GEOMETRY by the image's own height and width (see `ANNULI` and the
    radius computation below), so the annulus BOUNDARIES are scale-free —
    but the statistics computed inside those annuli are not: the residual
    uses a fixed 5px Gaussian kernel and the Laplacian a fixed kernel too,
    both operating at the crop's native resolution, so the same face
    content at a different resolution produces different residual/
    Laplacian statistics even though the annulus geometry lines up.
    Measured directly (40 photo-like fixtures, native-ROI features vs.
    224-square features, deltas in training-set standard deviations):

        ROI 448 -> mean 1.44 sd,  worst residual_std_b3    10.7 sd
        ROI 112 -> mean 15.6 sd,  worst laplacian_var_b3   178.8 sd
        ROI  80 -> mean 27.7 sd,  worst laplacian_var_b3   370.4 sd

    Fed through a model fitted on 224-square crops, mean P(fake) moved
    0.5003 -> 0.95 at ROI 448 and -> 1.0000 at ROI 112 and 80.
    `residual_logratio_b1_b2` — a log-ratio, the exact kind of term the
    module docstring names as carrying the seam signal — moved 0.047 ->
    0.307. The 6 LAB-mean features are scale-stable; the 12 residual/
    Laplacian features are not. `tests/test_blend_features.py::
    test_works_at_a_size_other_than_224` does NOT show a non-224 crop is
    "scored correctly as-is": it asserts only shape and finiteness at
    96px and never compares a 96px feature vector to a 224px one, so it
    cannot and does not support that claim.
    `tests/test_blend_detector.py::test_scores_the_same_face_at_different_roi_sizes`
    pins the fix: the same face content at two different ROI sizes must
    now score the same to a tight tolerance.

    Args:
        payload: the observation's image, HWC.
        roi: `(x, y, w, h)` in `payload`'s own coordinates, or None.

    Returns:
        The cropped-and-resized view, or None if `roi` is None or the
        clamped region is empty.
    """
    if roi is None:
        return None
    h, w = payload.shape[:2]
    x, y, rw, rh = roi
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + rw), min(h, y + rh)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = payload[y0:y1, x0:x1]
    return cv2.resize(crop, (_ALIGN_SIZE, _ALIGN_SIZE),
                      interpolation=cv2.INTER_AREA).astype(np.uint8)


def load_blend_model(path: str | Path) -> BlendModel:
    """Read a model file, refusing one that does not match this code.

    Args:
        path: npz written by `save_blend_model`.

    Returns:
        The model.

    Raises:
        ValueError: if a required field is missing, if format_version is
            not an integer or disagrees with MODEL_FILE_VERSION, or if
            feature_names disagrees with this build. A stale model file is
            worse than none: it would score confidently against columns
            that no longer mean what it was fitted on.
    """
    data = np.load(Path(path), allow_pickle=False)
    missing = [k for k in _REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(
            "blend model file is missing required field(s): "
            + ", ".join(missing))

    # Validate, don't coerce: int(np.array(1.9)) truncates to 1 and would
    # silently accept a corrupted or mistyped format_version as a match.
    raw_version = data["format_version"]
    version_value = float(raw_version)
    if version_value != int(version_value):
        raise ValueError(
            f"blend model format_version is not an integer: {raw_version!r}")
    version = int(version_value)
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
        # Checked before the quality floor, matching NPR and EffNet: on a
        # fresh checkout (no model file, the documented default state) a
        # low-quality observation must not be blamed with
        # "below_quality_floor" for a problem that is actually a missing
        # model — the reason field is what a caller reads to decide what to
        # do next, and "go improve your capture quality" would be wrong.
        path = Path(self.weights_path)
        if not path.exists():
            logger.warning("blend model absent at %s", path)
            return abstain(self.name, self.version, WEIGHTS_ABSENT)

        usable, reason = filter_by_quality_floor(obs, self.min_quality_band)
        if reason is not None:
            return abstain(self.name, self.version, reason)

        # Crop to the face ROI before reading seam features. `seam_features`
        # assumes an aligned face crop (module docstring): the inner face
        # sits near the centre so the concentric annuli straddle the seam.
        # Scoring `o.payload` directly would score whatever the caller
        # ingested — a whole frame in production (`pipeline.normalize`
        # attaches `roi` but leaves `payload` as the frame) — which moves
        # the annuli mostly over background and violates that premise
        # silently. An observation with no usable ROI is not scored on the
        # wrong pixels; it abstains instead, same as filter_by_quality_floor
        # drops observations it cannot use and only abstains once none
        # remain.
        crops = [c for c in (_crop_to_roi(o.payload, o.roi) for o in usable)
                 if c is not None]
        if not crops:
            return abstain(self.name, self.version, NO_ROI)

        model = load_blend_model(path)
        probs = [model.predict_proba(seam_features(c)) for c in crops]
        score = float(np.mean(probs))
        return RawScore(detector=self.name, version=self.version, score=score,
                        abstained=False, reason=OK,
                        artifacts={"n_observations": len(probs),
                                   "model_version": model.version})
