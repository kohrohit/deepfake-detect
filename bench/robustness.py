"""Perturbation surface (spec §8.3, acceptance criterion 9).

Includes the two physical re-capture paths that published evaluations usually
skip and that any adversary can perform for free: photographing a screen, and
printing then re-photographing. Both destroy roughly two thirds of the image's
high-frequency energy, which is the evidence most detectors depend on.

Every perturbation here is a pure function of its input: the two that draw
noise take an explicit seed with a fixed default, so a sweep is reproducible.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

#: The JPEG quality curve. Spec §8.3 asks for a sweep; one point is not a curve.
JPEG_QUALITIES = (90, 70, 50, 30, 10)


def _jpeg(img: np.ndarray, quality: int = 50) -> np.ndarray:
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _resize(img: np.ndarray, scale: float = 0.5) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                        interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _blur(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def _noise(img: np.ndarray, sigma: float = 8.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def _screenshot_recapture(img: np.ndarray) -> np.ndarray:
    """Photographing a screen: resample, moiré, glare gradient, recompress.

    Deterministic by construction — the moiré and glare are analytic, so this
    takes no seed.
    """
    out = _resize(img, 0.7)
    h, w = out.shape[:2]
    yy = np.arange(h)[:, None, None]
    moire = (8.0 * np.sin(2 * np.pi * yy / 3.0)).astype(np.float32)
    out = np.clip(out.astype(np.float32) + moire, 0, 255).astype(np.uint8)
    glare = np.linspace(1.0, 1.12, w, dtype=np.float32)[None, :, None]
    out = np.clip(out.astype(np.float32) * glare, 0, 255).astype(np.uint8)
    return _jpeg(out, quality=70)


def _print_recapture(img: np.ndarray, seed: int = 1) -> np.ndarray:
    """Print then photograph: soft focus, gamut loss, halftone, paper grain."""
    out = _blur(img, 3).astype(np.float32)
    out = np.clip((out - 16.0) * (255.0 / (235.0 - 16.0)), 0, 255)  # gamut
    out = np.round(out / 16.0) * 16.0                               # halftone
    rng = np.random.default_rng(seed)
    out = out + rng.normal(0, 4.0, out.shape)                       # paper grain
    return _jpeg(np.clip(out, 0, 255).astype(np.uint8), quality=75)


PERTURBATIONS: dict[str, Callable[..., np.ndarray]] = {
    "jpeg": _jpeg,
    "resize": _resize,
    "blur": _blur,
    "noise": _noise,
    "screenshot_recapture": _screenshot_recapture,
    "print_recapture": _print_recapture,
}


def apply_perturbation(img: np.ndarray, name: str, **kwargs: Any) -> np.ndarray:
    if name not in PERTURBATIONS:
        raise KeyError(
            f"unknown perturbation: {name!r}; known: {sorted(PERTURBATIONS)}")
    return PERTURBATIONS[name](img, **kwargs)


def robustness_sweep(img: np.ndarray) -> dict[str, np.ndarray]:
    """Clean, the full JPEG quality curve, and one entry per other perturbation.

    JPEG is expanded over `JPEG_QUALITIES` rather than sampled once, because a
    single quality cannot show where a detector falls off.
    """
    out: dict[str, np.ndarray] = {"clean": img}
    for quality in JPEG_QUALITIES:
        out[f"jpeg_q{quality}"] = _jpeg(img, quality=quality)
    for name, fn in PERTURBATIONS.items():
        if name == "jpeg":
            continue
        out[name] = fn(img)
    return out
