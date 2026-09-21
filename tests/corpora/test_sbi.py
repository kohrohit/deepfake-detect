"""The self-blend makes a seam without a second image."""
from __future__ import annotations

import numpy as np

from corpora.sbi import face_mask, jitter, self_blend
from dfd.faces import FaceBox


def _frame(h: int = 200, w: int = 180) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(40, 210, (h, w, 3), dtype=np.uint8)


def _box() -> FaceBox:
    lms = np.array([[70.0, 80.0], [110.0, 80.0], [90.0, 100.0],
                    [75.0, 125.0], [105.0, 125.0]])
    return FaceBox(x=55, y=55, w=70, h=90, landmarks=lms, score=0.99)


def test_blend_is_deterministic_for_a_given_seed() -> None:
    f, b = _frame(), _box()
    a, _ = self_blend(f, b, np.random.default_rng(3))
    c, _ = self_blend(f, b, np.random.default_rng(3))
    assert np.array_equal(a, c)


def test_different_seeds_give_different_blends() -> None:
    f, b = _frame(), _box()
    a, _ = self_blend(f, b, np.random.default_rng(1))
    c, _ = self_blend(f, b, np.random.default_rng(2))
    assert not np.array_equal(a, c)


def test_shape_and_dtype_are_preserved() -> None:
    f, b = _frame(), _box()
    out, mask = self_blend(f, b, np.random.default_rng(0))
    assert out.shape == f.shape
    assert out.dtype == np.uint8
    assert mask.shape == f.shape[:2]
    assert mask.dtype == np.float32


def test_pixels_inside_the_mask_actually_change() -> None:
    f, b = _frame(), _box()
    out, mask = self_blend(f, b, np.random.default_rng(0))
    core = mask > 0.9
    assert core.any(), "mask never reaches full weight; there is no blend"
    changed = (out[core] != f[core]).any(axis=-1).mean()
    assert changed > 0.5


def test_pixels_far_outside_the_mask_are_untouched() -> None:
    f, b = _frame(), _box()
    out, mask = self_blend(f, b, np.random.default_rng(0))
    outside = mask == 0.0
    assert outside.any()
    assert np.array_equal(out[outside], f[outside])


def test_mask_is_bounded_and_covers_part_but_not_all_of_the_frame() -> None:
    m = face_mask((200, 180), _box(), np.random.default_rng(0))
    assert m.min() >= 0.0 and m.max() <= 1.0
    coverage = float((m > 0.5).mean())
    assert 0.0 < coverage < 0.5


def test_mask_has_a_soft_edge_rather_than_a_hard_step() -> None:
    m = face_mask((200, 180), _box(), np.random.default_rng(0))
    partial = ((m > 0.05) & (m < 0.95)).sum()
    assert partial > 0, "a hard-edged mask is a paste, not a blend"


def test_jitter_changes_the_image_without_changing_its_shape() -> None:
    f = _frame()
    j = jitter(f, np.random.default_rng(0))
    assert j.shape == f.shape
    assert j.dtype == np.uint8
    assert not np.array_equal(j, f)


def test_the_blend_leaves_a_measurable_discontinuity() -> None:
    """The point of the whole exercise, stated as what is actually true.

    An earlier draft of this test asserted the seam RAISES gradient energy at
    the mask edge. That is false, and was measured to be false on two separate
    fixtures: the jitter downsamples before it upsamples, so the blended
    region carries LESS detail than what it replaced, and edge energy falls
    (118.7 -> 64.0 on noise, 7.77 -> 4.35 on a textured field). What a
    composite actually guarantees is a DISCONTINUITY — two regions with
    different imaging statistics meeting along a curve — and that is what the
    detector reads and what this asserts. The gap widened 0.77 -> 100.4 and
    0.55 -> 7.0 on those same two fixtures; the 3x bar below is far inside it.
    """
    import cv2
    f, b = _frame(), _box()
    out, mask = self_blend(f, b, np.random.default_rng(0))
    core = mask > 0.9
    outside = mask == 0.0
    assert core.any() and outside.any()

    def lap_mean(img: np.ndarray, m: np.ndarray) -> float:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return float(np.abs(cv2.Laplacian(gray, cv2.CV_64F))[m].mean())

    before = abs(lap_mean(f, core) - lap_mean(f, outside))
    after = abs(lap_mean(out, core) - lap_mean(out, outside))
    assert after > before * 3, (
        f"blend did not create a statistical discontinuity: {before:.2f} -> {after:.2f}")
