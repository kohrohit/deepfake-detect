import cv2
import numpy as np
import pytest

from bench.robustness import (
    JPEG_QUALITIES, PERTURBATIONS, apply_perturbation, robustness_sweep,
)

#: Square, landscape, portrait, and odd-sized. Every test used to run only
#: against the square 128x128 default, so image shape was an entirely
#: untested dimension — that is exactly how a broadcast bug in
#: _screenshot_recapture (only triggers when h != w) shipped with 12/12 green.
SHAPES = [(128, 128), (100, 150), (150, 100), (97, 131)]
SHAPE_IDS = ["square", "landscape", "portrait", "odd"]


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


def _correlation(a, b):
    """Pearson correlation of greyscale intensities against the clean image.

    An HF-energy upper bound alone is satisfied by a degenerate stub such as
    ``np.zeros_like(img)``: a near-black image has ~zero high-frequency
    energy too, and also passes the shape/dtype and "actually changes"
    tests. Correlation with the clean image catches that class of stub —
    measured 0.79-0.99 for the real perturbations, exactly 0.0 for a
    zeros_like stub.
    """
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64).ravel()
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64).ravel()
    return float(np.corrcoef(ga, gb)[0, 1])


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_every_perturbation_preserves_shape_and_dtype(shape):
    img = _img(*shape)
    for name in PERTURBATIONS:
        out = apply_perturbation(img, name)
        assert out.shape == img.shape, name
        assert out.dtype == np.uint8, name


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_every_perturbation_actually_changes_the_image(shape):
    img = _img(*shape)
    for name in PERTURBATIONS:
        assert not np.array_equal(apply_perturbation(img, name), img), name


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("name,limit", [
    ("resize", 0.5), ("blur", 0.3),
    ("screenshot_recapture", 0.6), ("print_recapture", 0.6),
])
def test_perturbation_destroys_high_frequency_evidence(name, limit, shape):
    """The mechanism, not just 'the pixels changed'. Measured ratios are
    0.23, 0.13, 0.36 and 0.31 — these limits carry real margin. Parametrised
    over shape, not just the square default: a broadcast bug in
    screenshot_recapture only fires when height != width."""
    img = _img(*shape)
    assert _hf_energy(apply_perturbation(img, name)) < limit * _hf_energy(img)


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_noise_adds_high_frequency_rather_than_removing_it(shape):
    """Asserting every perturbation lowers HF energy would be wrong."""
    img = _img(*shape)
    assert _hf_energy(apply_perturbation(img, "noise")) > _hf_energy(img)


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("name", [
    "resize", "blur", "screenshot_recapture", "print_recapture",
])
def test_perturbation_preserves_structure(name, shape):
    """An HF-energy upper bound alone is satisfied by a degenerate stub
    like np.zeros_like(img) (~0 HF energy, correlation exactly 0.0).
    Measured correlation of the real perturbations against clean ranges
    0.79-0.99 across shapes; 0.5 leaves more than 0.28 margin below the
    lowest real value."""
    img = _img(*shape)
    assert _correlation(apply_perturbation(img, name), img) > 0.5


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


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_sweep_covers_the_whole_jpeg_quality_curve(shape):
    """Spec §8.3 asks for a sweep. One point is not a curve."""
    out = robustness_sweep(_img(*shape))
    for q in JPEG_QUALITIES:
        assert f"jpeg_q{q}" in out
    assert len(JPEG_QUALITIES) >= 4


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_sweep_returns_clean_plus_every_non_jpeg_perturbation(shape):
    out = robustness_sweep(_img(*shape))
    assert "clean" in out
    assert np.array_equal(out["clean"], _img(*shape))
    for name in PERTURBATIONS:
        if name != "jpeg":
            assert name in out, name


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_sweep_is_deterministic(shape):
    """The harness's whole value is reproducibility; two perturbations draw
    from RNGs and nothing else pins them."""
    a, b = robustness_sweep(_img(*shape)), robustness_sweep(_img(*shape))
    assert set(a) == set(b)
    for k in a:
        assert np.array_equal(a[k], b[k]), k


def test_unknown_perturbation_raises_naming_the_unknown_name():
    with pytest.raises(KeyError, match="teleport"):
        apply_perturbation(_img(), "teleport")
