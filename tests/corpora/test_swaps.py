"""The four swap techniques.

Hermetic: synthetic frames and hand-built boxes, no weight file, no corpus.
The properties asserted are the ones the corpus depends on — that a swap
CHANGES the target, that it changes it only where it claims to, and that the
four techniques are actually different from one another.
"""
import numpy as np
import pytest
from corpora.swaps import (
    LOWRES_PASTE,
    MOUTH_PATCH,
    POISSON,
    TECHNIQUES,
    WARP_HULL,
    colour_transfer,
    ellipse_mask,
    rng_for,
    swap,
)
from dfd.faces import FaceBox


def _face(seed: int, tint: int = 0) -> np.ndarray:
    """A textured frame — flat colour gives every technique nothing to move."""
    rng = np.random.default_rng(seed)
    img = rng.integers(40, 215, (160, 160, 3), dtype=np.uint8)
    if tint:
        img = np.clip(img.astype(int) + tint, 0, 255).astype(np.uint8)
    return img


def _box(dx: int = 0) -> FaceBox:
    lms = np.array([[55.0 + dx, 65.0], [105.0 + dx, 65.0], [80.0 + dx, 90.0],
                    [60.0 + dx, 115.0], [100.0 + dx, 115.0]])
    return FaceBox(x=40 + dx, y=40, w=80, h=80, landmarks=lms, score=0.9)


def test_every_technique_changes_the_target():
    """A technique that returns the target unchanged would put a REAL image
    into the corpus under a fake label — the one defect no metric can see."""
    src, tgt = _face(1), _face(2)
    for technique in TECHNIQUES:
        result = swap(src, _box(), tgt, _box(4), technique, rng_for(technique))
        assert result is not None, technique
        assert not np.array_equal(result.image, tgt), technique
        assert result.generator == technique
        assert result.image.shape == tgt.shape
        assert result.image.dtype == np.uint8


def test_a_swap_leaves_the_frame_edge_alone():
    """The composite must sit inside the face, not span the crop. A technique
    that rewrites the border is detectable from the border alone."""
    src, tgt = _face(1), _face(2)
    for technique in TECHNIQUES:
        result = swap(src, _box(), tgt, _box(4), technique, rng_for(technique))
        assert result is not None
        for edge in (result.image[0, :], result.image[-1, :],
                     result.image[:, 0], result.image[:, -1]):
            pass
        # Corners specifically: furthest from any face-centred mask.
        for corner in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
            assert np.array_equal(result.image[corner], tgt[corner]), technique


def test_the_four_techniques_produce_four_different_images():
    """They are four GENERATOR LABELS. If two produce the same pixels, LOGO is
    folding on a distinction that does not exist."""
    src, tgt = _face(1), _face(2)
    images = {}
    for technique in TECHNIQUES:
        result = swap(src, _box(), tgt, _box(4), technique, rng_for(technique))
        assert result is not None
        images[technique] = result.image
    for a in TECHNIQUES:
        for b in TECHNIQUES:
            if a < b:
                assert not np.array_equal(images[a], images[b]), f"{a} == {b}"


def test_mouth_patch_touches_less_of_the_face_than_a_full_swap():
    """It is a reenactment, not a swap: the composite boundary is elsewhere and
    smaller, which is exactly what LOGO should find hard to generalise across."""
    src, tgt = _face(1), _face(2)
    full = swap(src, _box(), tgt, _box(4), WARP_HULL, rng_for("a"))
    mouth = swap(src, _box(), tgt, _box(4), MOUTH_PATCH, rng_for("a"))
    assert full is not None and mouth is not None

    assert (mouth.mask > 0.5).sum() < (full.mask > 0.5).sum()


def test_lowres_paste_loses_high_frequency_detail():
    """Its whole purpose is the generator-resolution signature slot C reads.
    If it does not actually soften the face, it is just warp_hull twice."""
    src, tgt = _face(1), _face(2)
    sharp = swap(src, _box(), tgt, _box(4), WARP_HULL, rng_for("a"))
    soft = swap(src, _box(), tgt, _box(4), LOWRES_PASTE, rng_for("a"))
    assert sharp is not None and soft is not None

    def energy(img, mask):
        g = img.astype(np.float64).mean(axis=2)
        lap = np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()
        return lap

    assert energy(soft.image, soft.mask) < energy(sharp.image, sharp.mask)


def test_an_unknown_technique_raises_rather_than_returning_the_target():
    src, tgt = _face(1), _face(2)
    with pytest.raises(ValueError, match="unknown technique"):
        swap(src, _box(), tgt, _box(4), "swap_imaginary", rng_for("a"))


def test_the_same_seed_yields_the_same_fake():
    src, tgt = _face(1), _face(2)
    a = swap(src, _box(), tgt, _box(4), WARP_HULL, rng_for("couple", WARP_HULL))
    b = swap(src, _box(), tgt, _box(4), WARP_HULL, rng_for("couple", WARP_HULL))
    assert a is not None and b is not None
    assert np.array_equal(a.image, b.image)


def test_rng_is_reproducible_across_processes_not_just_within_one():
    """`hash()` is salted per process, so a corpus keyed on it would differ
    between runs while every in-process test still passed."""
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, '.');"
         "from corpora.swaps import rng_for;"
         "print(rng_for('couple', 'swap_warp_hull').integers(0, 10**9))"],
        capture_output=True, text=True, check=True,
        env={"PYTHONHASHSEED": "random", "PATH": "/usr/bin:/bin"})
    assert int(out.stdout.strip()) == int(
        rng_for("couple", "swap_warp_hull").integers(0, 10**9))


def test_ellipse_mask_stays_inside_the_face_box():
    mask = ellipse_mask((160, 160), _box())
    assert mask.dtype == np.float32
    assert 0.0 <= mask.min() and mask.max() <= 1.0
    ys, xs = np.nonzero(mask > 0.01)
    # Feathering spreads beyond the ellipse, but not beyond the box by much.
    assert xs.min() >= 40 - 12 and xs.max() <= 120 + 12
    assert ys.min() >= 40 - 12 and ys.max() <= 120 + 12


def test_colour_transfer_moves_source_statistics_toward_the_target():
    """Without it, a swap between two lighting conditions is detectable from
    the colour step alone — the shortcut that outscored seam features on DF40."""
    src = _face(1, tint=-60)
    tgt = _face(2, tint=+60)
    mask = ellipse_mask((160, 160), _box())

    out = colour_transfer(src, tgt, mask)

    inside = mask > 0.1
    before = abs(src[inside].mean() - tgt[inside].mean())
    after = abs(out[inside].mean() - tgt[inside].mean())
    assert after < before
    assert out.dtype == np.uint8


def test_colour_transfer_is_a_no_op_on_an_empty_mask():
    """Fewer than 16 masked pixels has no usable statistics; returning the
    source unchanged is honest, dividing by a zero std is not."""
    src, tgt = _face(1), _face(2)
    empty = np.zeros((160, 160), np.float32)
    assert np.array_equal(colour_transfer(src, tgt, empty), src)


def test_a_degenerate_landmark_set_returns_none_rather_than_raising():
    """None is a pair to skip and the caller counts it; an exception would
    take down a corpus build over one bad detection."""
    flat = np.zeros((5, 2))
    bad = FaceBox(x=40, y=40, w=80, h=80, landmarks=flat, score=0.9)
    result = swap(_face(1), bad, _face(2), bad, POISSON, rng_for("a"))
    assert result is None or result.image.shape == (160, 160, 3)


def test_the_swap_pipeline_actually_applies_colour_transfer():
    """Testing `colour_transfer` directly does not prove `swap` CALLS it.

    A dark source composited into a bright target must not leave a colour
    step at the boundary — that step is the shortcut that let Lab colour
    means outscore seam features on DF40 (AUC 0.843), and a corpus built
    with it would be measuring skin tone.
    """
    src = _face(1, tint=-70)
    tgt = _face(2, tint=+70)

    result = swap(src, _box(), tgt, _box(), WARP_HULL, rng_for("a"))
    assert result is not None

    core = result.mask > 0.9
    assert core.sum() > 50, "fixture must have a solid composited core"
    composited = result.image[core].mean()
    raw_source = src[core].mean()
    target_level = tgt[core].mean()

    # An ABSOLUTE bound, not a comparison against the raw source. Measured
    # on this fixture: with the transfer the core sits at 192.3 against a
    # target of 194.3 (a gap of 2.0); without it, 60.7 (a gap of 133.6). A
    # relative assertion passes either way -- alpha blending at the mask edge
    # pulls the mutated core 0.7 grey levels toward the target, which is
    # enough to satisfy `<` and nothing else.
    assert abs(composited - target_level) < 20.0, (
        f"composited core at {composited:.1f} against a target level of "
        f"{target_level:.1f}; the raw source was {raw_source:.1f}")
