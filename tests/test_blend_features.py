"""Seam features read a discontinuity, and say the same thing every time."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from dfd.detectors.blend import ANNULI, FEATURE_NAMES, seam_features


def _flat(size: int = 224) -> np.ndarray:
    return np.full((size, size, 3), 128, dtype=np.uint8)


def _four_band_image(size: int = 224) -> np.ndarray:
    """Four concentric bands with a distinct flat luminance level and a
    distinct checkerboard texture amplitude each, aligned to `ANNULI`.

    The internal-consistency tests below need the three `lab_l_delta_*` and
    three `residual_logratio_*` values to be pairwise distinct and mostly
    non-zero. A fixture where those values happened to coincide (e.g. a
    single ring on a flat field) would let a swapped feature order pass by
    accident, so this fixture is built with band levels/amplitudes chosen
    (and verified, see task-4-report.md) to avoid that coincidence: levels
    30/70/150/200 give L-mean deltas of -47.5/-82.5/-44.0, and amplitudes
    0/15/35/60 give residual log-ratios of roughly -2.32/-0.78/-0.48 — all
    six values distinct and non-zero.
    """
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w]
    ry = (yy - (h - 1) / 2.0) / max(1.0, (h - 1) / 2.0)
    rx = (xx - (w - 1) / 2.0) / max(1.0, (w - 1) / 2.0)
    radius = np.sqrt(rx ** 2 + ry ** 2)

    levels = (30.0, 70.0, 150.0, 200.0)
    amplitudes = (0.0, 15.0, 35.0, 60.0)
    checker = (((xx + yy) % 2) * 2 - 1).astype(np.float64)

    gray = np.zeros((h, w), dtype=np.float64)
    for (lo, hi), level, amp in zip(ANNULI, levels, amplitudes):
        mask = (radius >= lo) & (radius < hi)
        gray[mask] = level + checker[mask] * amp

    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    return np.repeat(gray_u8[:, :, None], 3, axis=2)


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


def test_contrast_features_are_internally_consistent() -> None:
    """`FEATURE_NAMES` positional alignment is a hope unless something checks
    it. The contrast values are each defined in terms of other, independently
    named per-band features, so alignment can be verified without
    recomputing anything from the image: each `lab_l_delta_bK_bK+1` must
    equal the difference of the two `lab_l_mean_b*` values it names, and each
    `residual_logratio_bK_bK+1` must equal the log1p difference of the two
    `residual_mean_b*` values it names. If `_feature_names()` and
    `seam_features()` ever generate these two blocks in different orders,
    this fails; if only one of them is reordered, this also fails.

    This only discriminates because `_four_band_image` was built so the six
    contrast values are pairwise distinct and non-zero (asserted below) —
    a fixture where they coincided would let a reordering pass unnoticed.
    """
    f = seam_features(_four_band_image())
    idx = FEATURE_NAMES.index

    deltas: list[float] = []
    logratios: list[float] = []
    for b in range(len(ANNULI) - 1):
        l_lo = f[idx(f"lab_l_mean_b{b}")]
        l_hi = f[idx(f"lab_l_mean_b{b + 1}")]
        delta = f[idx(f"lab_l_delta_b{b}_b{b + 1}")]
        assert float(delta) == pytest.approx(float(l_lo - l_hi), abs=1e-2)
        deltas.append(float(delta))

        r_lo = f[idx(f"residual_mean_b{b}")]
        r_hi = f[idx(f"residual_mean_b{b + 1}")]
        logratio = f[idx(f"residual_logratio_b{b}_b{b + 1}")]
        expected = np.log1p(float(r_lo)) - np.log1p(float(r_hi))
        assert float(logratio) == pytest.approx(expected, abs=1e-3)
        logratios.append(float(logratio))

    assert len({round(d, 3) for d in deltas}) == len(deltas)
    assert len({round(lr, 3) for lr in logratios}) == len(logratios)
    assert sum(abs(d) > 1e-3 for d in deltas) >= 2
    assert sum(abs(lr) > 1e-3 for lr in logratios) >= 2


def test_contrast_groups_do_not_swap_wholesale() -> None:
    """The two contrast blocks must not only be internally aligned (see
    above) but also appear in the right order and place: every name in the
    first block is a `residual_logratio_*` and every name in the second is a
    `lab_l_delta_*`. Offsets are derived from `ANNULI` and the total feature
    count rather than hardcoded, so this does not silently stop checking
    anything if the per-band stat count ever changes."""
    band_count = len(ANNULI)
    contrast_count = band_count - 1
    per_band_count = (len(FEATURE_NAMES) - 2 * contrast_count) // band_count
    assert per_band_count * band_count + 2 * contrast_count == len(FEATURE_NAMES)

    logratio_start = per_band_count * band_count
    delta_start = logratio_start + contrast_count
    assert delta_start + contrast_count == len(FEATURE_NAMES)

    logratio_names = FEATURE_NAMES[logratio_start:delta_start]
    delta_names = FEATURE_NAMES[delta_start:delta_start + contrast_count]
    assert len(logratio_names) == contrast_count
    assert len(delta_names) == contrast_count
    assert all(name.startswith("residual_logratio_") for name in logratio_names)
    assert all(name.startswith("lab_l_delta_") for name in delta_names)
