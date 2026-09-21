"""Quality measurement and banding (spec §5.3).

A detector below its quality floor is not consulted. This is the layer that
turns a confident guess on a 40-pixel blurry face into an honest abstention.

Thresholds here are starting values. Task 18 records the observed distribution
so they can be set from data rather than from intuition.
"""
from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from .types import QUALITY_BANDS, Quality

# Starting thresholds. Revisit against the distribution recorded by the benchmark.
MIN_IOD_REJECT = 32.0
MIN_IOD_LOW = 64.0
MIN_IOD_HIGH = 96.0
MIN_BLUR_LOW = 20.0
MIN_BLUR_HIGH = 100.0
MAX_YAW_HIGH = 30.0
EXPOSURE_OK = (0.15, 0.90)


def _laplacian_var(gray: npt.NDArray[np.uint8]) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def measure_quality(
    frame: npt.NDArray[np.uint8],
    roi: tuple[int, int, int, int],
    landmarks: npt.NDArray[np.float64],
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
) -> Quality:
    """Measure quality of the face at `roi`.

    landmarks: at least two points, [[lx, ly], [rx, ry]] for left and right eye.
    """
    x, y, w, h = roi
    crop = frame[y : y + h, x : x + w]
    gray = (cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop).astype(np.uint8)

    iod = float(np.linalg.norm(landmarks[0] - landmarks[1]))
    blur = _laplacian_var(gray)
    exposure = float(gray.mean() / 255.0)

    band = _band(iod, blur, yaw_deg, exposure)
    return Quality(
        inter_ocular_px=iod,
        blur_var=blur,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        exposure=exposure,
        band=band,
    )


def _band(iod: float, blur: float, yaw: float, exposure: float) -> str:
    if iod < MIN_IOD_REJECT or blur < MIN_BLUR_LOW:
        return "reject"
    if not (EXPOSURE_OK[0] <= exposure <= EXPOSURE_OK[1]):
        return "low"
    if iod >= MIN_IOD_HIGH and blur >= MIN_BLUR_HIGH and abs(yaw) <= MAX_YAW_HIGH:
        return "high"
    if iod >= MIN_IOD_LOW:
        return "medium"
    return "low"


def meets_floor(band: str, floor: str) -> bool:
    """True if `band` is at least as good as `floor`."""
    return QUALITY_BANDS.index(band) >= QUALITY_BANDS.index(floor)
