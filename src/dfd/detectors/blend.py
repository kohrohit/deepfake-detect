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

import cv2
import numpy as np
import numpy.typing as npt

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
