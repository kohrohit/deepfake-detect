import numpy as np
from dfd.quality import measure_quality, meets_floor
from dfd.types import QUALITY_BANDS


def _sharp(h=256, w=256) -> np.ndarray:
    """A high-frequency checkerboard: high Laplacian variance."""
    img = np.indices((h, w)).sum(axis=0) % 2
    return (img * 255).astype(np.uint8)[:, :, None].repeat(3, axis=2)


def _flat(h=256, w=256) -> np.ndarray:
    return np.full((h, w, 3), 128, dtype=np.uint8)


LM_WIDE = np.array([[80.0, 100.0], [176.0, 100.0]])   # 96 px inter-ocular
LM_TIGHT = np.array([[120.0, 100.0], [140.0, 100.0]])  # 20 px inter-ocular


def test_sharp_image_has_higher_blur_var_than_flat():
    sharp = measure_quality(_sharp(), (0, 0, 256, 256), LM_WIDE)
    flat = measure_quality(_flat(), (0, 0, 256, 256), LM_WIDE)
    assert sharp.blur_var > flat.blur_var


def test_inter_ocular_distance_is_measured_from_landmarks():
    q = measure_quality(_sharp(), (0, 0, 256, 256), LM_WIDE)
    assert abs(q.inter_ocular_px - 96.0) < 1e-6


def test_small_face_is_banded_reject():
    q = measure_quality(_sharp(), (0, 0, 256, 256), LM_TIGHT)
    assert q.band == "reject"


def test_flat_image_is_banded_reject_for_insufficient_detail():
    """Zero Laplacian variance means no measurable detail: reject, not a guess."""
    q = measure_quality(_flat(), (0, 0, 256, 256), LM_WIDE)
    assert q.band == "reject"


def test_meets_floor_uses_band_ordering():
    assert meets_floor("high", "medium") is True
    assert meets_floor("low", "medium") is False
    assert meets_floor("medium", "medium") is True
    assert meets_floor("reject", "low") is False


def test_meets_floor_is_reflexive_for_every_band():
    """'At least as good as' includes equality — reject satisfies a reject floor.

    Documented deliberately: no detector declares min_quality_band="reject",
    and the detector config (Task 6) will forbid it at the type level.
    """
    for band in QUALITY_BANDS:
        assert meets_floor(band, band) is True
    assert meets_floor("reject", "low") is False
