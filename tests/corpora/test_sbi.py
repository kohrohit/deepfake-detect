"""The self-blend makes a seam without a second image."""
from __future__ import annotations

import math

import numpy as np

from corpora.sbi import MASK_AXIS_RANGE, face_mask, jitter, self_blend
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
    shape = (200, 180)
    box = _box()
    m = face_mask(shape, box, np.random.default_rng(0))
    assert m.min() >= 0.0 and m.max() <= 1.0
    coverage = float((m > 0.5).mean())
    # The old bound here was 0 < coverage < 0.5 -- true of almost any mask,
    # including a box-agnostic one, since the box is only 17.5% of the frame
    # and real coverage sits around 6-9%. Derive a tighter bound from the
    # geometry instead of the frame: MASK_AXIS_RANGE sets the ellipse's
    # semi-axes as fractions of the *half*-box, so the raw ellipse area at
    # each end of that range is pi * (axis_frac / 2)**2 * box_area. The
    # feathered mask's >0.5 area tracks that raw area closely because the
    # Gaussian blur is symmetric about the boundary. A +/-20% margin still
    # rejects a mask of grossly the wrong size (an unscaled radius, a fixed
    # blob) without depending on the exact feather draw.
    box_area = box.w * box.h
    frame_area = shape[0] * shape[1]
    lo_area_frac = math.pi * (MASK_AXIS_RANGE[0] / 2) ** 2
    hi_area_frac = math.pi * (MASK_AXIS_RANGE[1] / 2) ** 2
    lo = 0.8 * lo_area_frac * box_area / frame_area
    hi = 1.2 * hi_area_frac * box_area / frame_area
    assert lo < coverage < hi, (
        f"coverage {coverage:.4f} outside box-derived bound [{lo:.4f}, {hi:.4f}]")


def test_mask_geometry_varies_with_the_seed() -> None:
    shape = (200, 180)
    box = _box()
    coverages = []
    centroids = []
    for seed in range(8):
        m = face_mask(shape, box, np.random.default_rng(seed))
        coverages.append(float((m > 0.5).mean()))
        ys, xs = np.nonzero(m > 0.5)
        centroids.append((float(ys.mean()), float(xs.mean())))

    # A rare coincidence could make one pair of seeds agree; it cannot make
    # the whole spread this wide. A box- and rng-agnostic mask (a fixed
    # ellipse at a fixed position) would score 0 on both lines below.
    coverage_spread = max(coverages) - min(coverages)
    assert coverage_spread > 0.01, (
        f"coverage barely moves across seeds: spread {coverage_spread:.4f}")

    ys_ = [c[0] for c in centroids]
    xs_ = [c[1] for c in centroids]
    assert max(ys_) - min(ys_) > 1.0, "mask centroid row is frozen across seeds"
    assert max(xs_) - min(xs_) > 1.0, "mask centroid column is frozen across seeds"


def test_mask_follows_the_face_box() -> None:
    shape = (200, 180)
    small = FaceBox(x=10, y=10, w=40, h=50, landmarks=np.zeros((5, 2)), score=0.99)
    large = FaceBox(x=70, y=70, w=90, h=110, landmarks=np.zeros((5, 2)), score=0.99)

    m_small = face_mask(shape, small, np.random.default_rng(0))
    m_large = face_mask(shape, large, np.random.default_rng(0))

    ys_s, xs_s = np.nonzero(m_small > 0.5)
    ys_l, xs_l = np.nonzero(m_large > 0.5)
    centroid_small = (float(ys_s.mean()), float(xs_s.mean()))
    centroid_large = (float(ys_l.mean()), float(xs_l.mean()))

    expected_small = (small.y + small.h / 2.0, small.x + small.w / 2.0)
    expected_large = (large.y + large.h / 2.0, large.x + large.w / 2.0)

    # Centroids track their own box centre (within the mask's own centre
    # jitter, a few pixels) -- a box-agnostic mask would put both centroids
    # in the same place regardless of which box was passed.
    assert math.dist(centroid_small, expected_small) < 5.0
    assert math.dist(centroid_large, expected_large) < 5.0
    assert math.dist(centroid_small, centroid_large) > 50.0, (
        "mask centroid did not move with the box"
    )

    # A box twice the linear size should cover noticeably more of the frame.
    coverage_small = float((m_small > 0.5).mean())
    coverage_large = float((m_large > 0.5).mean())
    assert coverage_large > coverage_small * 2, (
        f"larger box did not yield larger coverage: {coverage_small:.4f} -> "
        f"{coverage_large:.4f}"
    )


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
