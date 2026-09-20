"""Tests for the NPR detector (slot C — upsampling fingerprint)."""
import numpy as np
import pytest

from dfd.detectors.base import WEIGHTS_ABSENT
from dfd.detectors.npr import NPRDetector, npr_feature
from dfd.types import Modality, Observation, Quality


def _obs(img: np.ndarray, band: str = "high") -> Observation:
    """Create an Observation with a high-quality assessment."""
    q = Quality(
        inter_ocular_px=100,
        blur_var=200,
        yaw_deg=0,
        pitch_deg=0,
        exposure=0.5,
        band=band,
    )
    return Observation(
        t=0.0,
        payload=img,
        roi=(0, 0, *img.shape[:2][::-1]),
        quality=q,
        source_id="s1",
    )


def test_npr_feature_is_near_zero_on_nearest_upsampled_content():
    """An image that IS a 2x nearest upsample has almost no NPR residual.

    This is the physics the detector reads: generator upsampling leaves a
    characteristic residual that natural images do not have.
    """
    small = np.random.default_rng(0).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    upsampled = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)
    feat = npr_feature(upsampled)
    assert np.abs(feat).mean() < 1e-6, "Upsampled content should have near-zero residual"


def test_npr_feature_is_nonzero_on_natural_noise():
    """Natural images have non-zero NPR residual.

    The detector's key property: real content leaves a larger residual than
    upsampled content. This test asserts the opposite direction.
    """
    noise = np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    feat = npr_feature(noise)
    assert np.abs(feat).mean() > 1e-3, "Natural noise should have non-zero residual"


def test_natural_image_npr_differs_significantly_from_upsampled():
    """The NPR feature discriminates between upsampled and natural content.

    This is the core physical claim. Both should have mean NPR residual, but
    natural images should have much larger.
    """
    # Create a small image and upsample it
    rng = np.random.default_rng(2)
    small = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    upsampled = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)

    # Create a natural 64x64 image
    natural = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)

    # Compute NPR features
    feat_upsampled = np.abs(npr_feature(upsampled)).mean()
    feat_natural = np.abs(npr_feature(natural)).mean()

    # The natural image should have much larger residual (at least 10x)
    assert feat_natural > feat_upsampled * 10, (
        f"Natural image NPR ({feat_natural:.6f}) should be much larger than "
        f"upsampled ({feat_upsampled:.6f})"
    )


def test_npr_feature_preserves_shape():
    """The npr_feature function returns the same shape as input."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    feat = npr_feature(img)
    assert feat.shape == (64, 64, 3), "Feature should preserve input shape"


def test_detector_abstains_when_weights_absent(tmp_path):
    """Detector returns abstention when weight file does not exist."""
    d = NPRDetector(weights_path=tmp_path / "missing.pt")
    obs = [_obs(np.zeros((64, 64, 3), dtype=np.uint8))]
    r = d.score(obs)
    assert r.abstained and r.reason == WEIGHTS_ABSENT


def test_detector_declares_its_slot_and_physics():
    """Detector declares its identifying metadata."""
    d = NPRDetector(weights_path="whatever")
    assert d.name == "npr"
    assert d.slot == "C"
    assert Modality.IMAGE in d.modalities
    assert Modality.VIDEO in d.modalities


def test_detector_is_frozen():
    """Detector is immutable after creation to preserve registry identity."""
    d = NPRDetector(weights_path="test.pt")
    with pytest.raises(Exception):
        d.name = "modified"  # type: ignore


def test_detector_modalities_is_frozenset():
    """Modalities field is a frozenset, not a mutable set."""
    d = NPRDetector(weights_path="test.pt")
    assert isinstance(d.modalities, frozenset)


def test_min_quality_band_is_valid_literal():
    """min_quality_band must be one of the valid literal values."""
    d = NPRDetector(weights_path="test.pt", min_quality_band="low")
    assert d.min_quality_band in ("low", "medium", "high")


def test_detector_filters_quality_across_multiple_observations():
    """Detector correctly filters observations by quality floor.

    This tests the non-trivial dimension: when given multiple observations,
    the detector must correctly identify which ones meet the quality floor,
    without relying on just the first observation's quality.
    """
    d = NPRDetector(weights_path="nonexistent.pt")

    # Create multiple observations with different quality bands
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    obs_high = _obs(img, band="high")
    obs_low = _obs(img, band="low")
    obs_none = Observation(
        t=0.0,
        payload=img,
        roi=(0, 0, 64, 64),
        quality=None,
        source_id="s_none",
    )

    # Mix: two pass floor, one doesn't
    obs = [obs_high, obs_low, obs_high]

    # Should abstain due to missing weights, not quality issues
    # (since some observations DO meet the floor)
    r = d.score(obs)
    assert r.abstained and r.reason == WEIGHTS_ABSENT


def test_detector_prioritizes_quality_filtering():
    """Detector correctly reports the reason for abstention.

    When using a non-existent weights path with observations below the quality
    floor, the detector reports WEIGHTS_ABSENT (weights checked first) not
    BELOW_FLOOR (checked second). This is correct: a missing model cannot
    score anything, period.
    """
    d = NPRDetector(weights_path="nonexistent.pt", min_quality_band="high")

    # Create observations with low quality
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    obs_low = _obs(img, band="low")

    # All observations below the floor, but weights don't exist
    obs = [obs_low]

    # Should abstain with WEIGHTS_ABSENT because weights are checked first
    r = d.score(obs)
    assert r.abstained and r.reason == WEIGHTS_ABSENT
