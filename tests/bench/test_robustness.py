import cv2
import numpy as np
import pytest

from bench.robustness import (
    JPEG_QUALITIES, PERTURBATIONS, apply_perturbation, robustness_sweep,
)


def _img(h=128, w=128):
    """Structured: two gradients, a hard block edge, and fine scanlines.

    Uniform noise is the wrong fixture here — JPEG raises high-frequency
    energy on it, so mechanism assertions would measure the fixture.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    img[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    img[h // 4:3 * h // 4, w // 4:3 * w // 4] = 220
    img[::4, :] = 40
    return img


def _hf_energy(img):
    """Mean |Laplacian| — the high-frequency evidence detectors depend on."""
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.abs(cv2.Laplacian(grey, cv2.CV_32F)).mean())


def _distortion(a, b):
    return float(np.abs(a.astype(np.int32) - b.astype(np.int32)).mean())


def test_every_perturbation_preserves_shape_and_dtype():
    img = _img()
    for name in PERTURBATIONS:
        out = apply_perturbation(img, name)
        assert out.shape == img.shape, name
        assert out.dtype == np.uint8, name


def test_every_perturbation_actually_changes_the_image():
    img = _img()
    for name in PERTURBATIONS:
        assert not np.array_equal(apply_perturbation(img, name), img), name


@pytest.mark.parametrize("name,limit", [
    ("resize", 0.5), ("blur", 0.3),
    ("screenshot_recapture", 0.6), ("print_recapture", 0.6),
])
def test_perturbation_destroys_high_frequency_evidence(name, limit):
    """The mechanism, not just 'the pixels changed'. Measured ratios are
    0.23, 0.13, 0.36 and 0.31 — these limits carry real margin."""
    img = _img()
    assert _hf_energy(apply_perturbation(img, name)) < limit * _hf_energy(img)


def test_noise_adds_high_frequency_rather_than_removing_it():
    """Asserting every perturbation lowers HF energy would be wrong."""
    img = _img()
    assert _hf_energy(apply_perturbation(img, "noise")) > _hf_energy(img)


def test_jpeg_distortion_rises_monotonically_as_quality_falls():
    """JPEG is asserted on distortion, not high-frequency energy: its
    blocking artefacts ADD edges, so HF energy is flat and non-monotonic
    across the sweep."""
    img = _img()
    qualities = sorted(JPEG_QUALITIES, reverse=True)
    d = [_distortion(apply_perturbation(img, "jpeg", quality=q), img)
         for q in qualities]
    assert d == sorted(d), dict(zip(qualities, d))
    assert d[-1] > d[0]


def test_sweep_covers_the_whole_jpeg_quality_curve():
    """Spec §8.3 asks for a sweep. One point is not a curve."""
    out = robustness_sweep(_img())
    for q in JPEG_QUALITIES:
        assert f"jpeg_q{q}" in out
    assert len(JPEG_QUALITIES) >= 4


def test_sweep_returns_clean_plus_every_non_jpeg_perturbation():
    out = robustness_sweep(_img())
    assert "clean" in out
    assert np.array_equal(out["clean"], _img())
    for name in PERTURBATIONS:
        if name != "jpeg":
            assert name in out, name


def test_sweep_is_deterministic():
    """The harness's whole value is reproducibility; two perturbations draw
    from RNGs and nothing else pins them."""
    a, b = robustness_sweep(_img()), robustness_sweep(_img())
    assert set(a) == set(b)
    for k in a:
        assert np.array_equal(a[k], b[k]), k


def test_unknown_perturbation_raises_naming_the_unknown_name():
    with pytest.raises(KeyError, match="teleport"):
        apply_perturbation(_img(), "teleport")
