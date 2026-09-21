"""Seam features read a discontinuity, and say the same thing every time."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from dfd.detectors.blend import FEATURE_NAMES, seam_features


def _flat(size: int = 224) -> np.ndarray:
    return np.full((size, size, 3), 128, dtype=np.uint8)


def _with_ring(size: int = 224) -> np.ndarray:
    """A flat image with a soft bright annulus — a synthetic seam."""
    img = _flat(size)
    cv2.circle(img, (size // 2, size // 2), int(size * 0.3), (200, 200, 200), 6)
    return cv2.GaussianBlur(img, (7, 7), 0)


def test_returns_one_float32_value_per_named_feature() -> None:
    f = seam_features(_flat())
    assert f.shape == (len(FEATURE_NAMES),)
    assert f.dtype == np.float32
    assert len(FEATURE_NAMES) == 30


def test_feature_names_are_unique() -> None:
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_is_deterministic() -> None:
    img = _with_ring()
    assert np.array_equal(seam_features(img), seam_features(img))


def test_all_features_are_finite() -> None:
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    for img in (_flat(), _with_ring(), noisy):
        assert np.isfinite(seam_features(img)).all()


def test_a_ring_raises_residual_energy_where_the_ring_is() -> None:
    """The discriminating property. A flat image has no annular structure;
    one with a ring must differ in the band the ring falls in."""
    flat = seam_features(_flat())
    ring = seam_features(_with_ring())
    i = FEATURE_NAMES.index("residual_mean_b1")
    assert ring[i] > flat[i] + 1e-3


def test_ratio_features_separate_a_ring_from_a_flat_field() -> None:
    flat = seam_features(_flat())
    ring = seam_features(_with_ring())
    i = FEATURE_NAMES.index("residual_logratio_b1_b2")
    assert abs(ring[i] - flat[i]) > 1e-3


def test_rejects_an_image_that_is_not_three_channel() -> None:
    with pytest.raises(ValueError, match="HWC RGB"):
        seam_features(np.zeros((224, 224), dtype=np.uint8))


def test_works_at_a_size_other_than_224() -> None:
    f = seam_features(_with_ring(size=96))
    assert f.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(f).all()


def test_a_tiny_image_with_an_empty_annulus_stays_finite() -> None:
    """At size=3 the discrete radius grid leaves annuli 1 and 2 with no
    pixels at all (verified: only bands 0 and 3 are populated). This is the
    guard `seam_features` documents — an empty band must contribute zeros,
    not nan from an empty-array mean/std/var — and neither the 224 nor the
    96 fixture above ever hits it, so nothing else in this file exercises it."""
    f = seam_features(_flat(size=3))
    assert f.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(f).all()
    i = FEATURE_NAMES.index("residual_mean_b1")
    assert f[i] == 0.0
