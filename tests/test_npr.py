"""Tests for the NPR detector (slot C — upsampling fingerprint)."""
import logging

import numpy as np
import pytest
import torch
import torch.nn as nn

from dfd.detectors.base import (
    BELOW_FLOOR,
    NO_OBSERVATIONS,
    NO_QUALITY,
    WEIGHTS_ABSENT,
)
from dfd.detectors.npr import NPRDetector, npr_feature
from dfd.types import Modality, Observation, Quality


class _Tiny2Class(nn.Module):
    """Minimal 2-class model for testing the real scoring path."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: NCHW tensor from npr_feature (N, 3, H, W)
        # Average spatial dims to 1D, feed to linear layer
        return self.fc(x.mean(dim=(2, 3)))


class _NegatedTiny2Class(nn.Module):
    """Same architecture as _Tiny2Class but negates the output logits.

    Used to prove that two `model_factory` values loading the SAME state_dict
    file produce genuinely different models: same weights, opposite-signed
    logits, so softmax output differs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -self.fc(x.mean(dim=(2, 3)))


class _Tiny3Class(nn.Module):
    """3-class model for testing invalid output shape error."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.mean(dim=(2, 3)))


@pytest.fixture
def tiny_weights(tmp_path):
    """Create a minimal dummy model and save its state_dict.

    This fixture creates a state_dict (secure-loadable format) that the detector
    can load with weights_only=True when given a matching model_factory.
    """
    p = tmp_path / "tiny.pt"
    model = _Tiny2Class()
    torch.save(model.state_dict(), p)
    return p


@pytest.fixture
def tiny_model_factory():
    """Factory function that creates an uninitialized _Tiny2Class."""
    def factory():
        return _Tiny2Class()
    return factory


def _obs(img: np.ndarray, band: str = "high") -> Observation:
    """Create an Observation with a quality assessment."""
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


# ============================================================================
# Feature Tests (NPR physics)
# ============================================================================


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


def test_npr_feature_handles_odd_dimensions():
    """NPR feature correctly handles non-power-of-2 dimensions.

    The code includes trim logic to handle height/width not divisible by stride.
    This tests an odd dimension case (63x65) to ensure no pixel misalignment.
    """
    img = np.random.default_rng(5).integers(0, 255, (63, 65, 3), dtype=np.uint8)
    feat = npr_feature(img)
    assert feat.shape == (63, 65, 3)
    # Odd dimensions should still produce valid residuals
    assert not np.isnan(feat).any()


# ============================================================================
# Detector Identity and Protocol Tests
# ============================================================================


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


# ============================================================================
# Abstention Tests (no real scoring)
# ============================================================================


def test_detector_abstains_when_weights_absent(tmp_path):
    """Detector returns abstention when weight file does not exist."""
    d = NPRDetector(weights_path=tmp_path / "missing.pt")
    obs = [_obs(np.zeros((64, 64, 3), dtype=np.uint8))]
    r = d.score(obs)
    assert r.abstained and r.reason == WEIGHTS_ABSENT


def test_detector_abstains_on_empty_observation_list():
    """Detector returns NO_OBSERVATIONS for empty input."""
    d = NPRDetector(weights_path="dummy.pt")
    r = d.score([])
    assert r.abstained and r.reason == NO_OBSERVATIONS


# ============================================================================
# Secure Path Tests (model_factory + state_dict, weights_only=True)
# ============================================================================


def test_detector_produces_real_score_via_secure_path(tiny_weights, tiny_model_factory):
    """The detector's secure path produces a real score in [0, 1].

    This exercises the full path: quality filtering, secure model loading via
    weights_only=True + model_factory, NPR feature computation, inference,
    and score aggregation.
    """
    d = NPRDetector(weights_path=tiny_weights, model_factory=tiny_model_factory)
    img = np.random.default_rng(10).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    obs = [_obs(img)]

    r = d.score(obs)

    # Must score, not abstain
    assert not r.abstained, f"Expected score, got abstention: {r.reason}"
    assert r.score is not None
    # Score must be in valid probability range
    assert 0.0 <= r.score <= 1.0, f"Score out of range: {r.score}"
    # Must record observations
    assert r.artifacts["n_observations"] == 1


def test_detector_records_per_observation_scores_secure_path(tiny_weights, tiny_model_factory):
    """Detector records individual per-observation scores in artifacts (secure path).

    Per-observation scores allow downstream fusion to make informed decisions
    about which frames are most confident, avoiding the dilution that would
    come from averaging during detection.
    """
    d = NPRDetector(weights_path=tiny_weights, model_factory=tiny_model_factory)

    # Create 3 observations
    rng = np.random.default_rng(11)
    imgs = [rng.integers(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
    obs = [_obs(img) for img in imgs]

    r = d.score(obs)

    assert not r.abstained
    assert "per_observation_scores" in r.artifacts
    assert len(r.artifacts["per_observation_scores"]) == 3
    assert "max_score" in r.artifacts
    # Max must be >= mean
    assert r.artifacts["max_score"] >= r.score


def test_detector_mixed_batch_order_independent_secure_path(tiny_weights, tiny_model_factory):
    """Order of observations does not affect quality filtering (secure path).

    This pins the Task-6 bug class: if filtering relied on obs[0], then
    reordering would change which observations are used. Test both orderings.
    """
    d = NPRDetector(
        weights_path=tiny_weights,
        model_factory=tiny_model_factory,
        min_quality_band="high",
    )

    img = np.random.default_rng(12).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    obs_high = _obs(img, band="high")
    obs_low = _obs(img, band="low")

    # Ordering 1: high first, low second
    r1 = d.score([obs_high, obs_low])
    # Ordering 2: low first, high second
    r2 = d.score([obs_low, obs_high])

    # Both should give the same result (high observation scored)
    assert r1.score == r2.score, (
        f"Order-dependent filtering detected: "
        f"[high, low]={r1.score} vs [low, high]={r2.score}"
    )
    assert r1.artifacts["n_observations"] == 1
    assert r2.artifacts["n_observations"] == 1


def test_detector_quality_filtering_with_mixed_batch_secure_path(tiny_weights, tiny_model_factory):
    """Detector correctly filters observations by quality floor (secure path).

    When given a mix of usable and unusable observations, the detector should
    score only the usable ones, as recorded in n_observations artifact.
    """
    d = NPRDetector(
        weights_path=tiny_weights,
        model_factory=tiny_model_factory,
        min_quality_band="high",
    )

    img = np.random.default_rng(13).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    # 3 high-quality, 2 low-quality, 1 no-quality
    obs = [
        _obs(img, band="high"),
        _obs(img, band="low"),
        _obs(img, band="high"),
        Observation(
            t=0.0,
            payload=img,
            roi=(0, 0, 64, 64),
            quality=None,
            source_id="s_none",
        ),
        _obs(img, band="low"),
        _obs(img, band="high"),
    ]

    r = d.score(obs)

    # Should score only the 3 high-quality observations
    assert not r.abstained
    assert r.artifacts["n_observations"] == 3
    assert len(r.artifacts["per_observation_scores"]) == 3


def test_detector_secure_path_logs_no_warning(tiny_weights, tiny_model_factory, caplog):
    """Secure path (model_factory + state_dict) produces no warning log.

    This proves the secure path is the expected, warning-free default.
    """
    d = NPRDetector(weights_path=tiny_weights, model_factory=tiny_model_factory)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    obs = [_obs(img)]

    with caplog.at_level(logging.WARNING):
        r = d.score(obs)

    assert not r.abstained
    # No warning should have been logged
    assert not any("allow_unsafe_load" in record.message for record in caplog.records)


# ============================================================================
# Unsafe Path Tests (only 2, for the explicit bypass)
# ============================================================================


def test_detector_rejects_full_pickle_without_model_factory_or_unsafe_load(tmp_path):
    """Detector rejects full-module pickle when model_factory=None and allow_unsafe_load=False.

    This is the security gate: full pickles require explicit opt-in.
    """
    # Save a full module (not state_dict)
    p = tmp_path / "full_module.pt"
    torch.save(_Tiny2Class(), p)

    d = NPRDetector(weights_path=p, model_factory=None, allow_unsafe_load=False)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    obs = [_obs(img)]

    with pytest.raises(RuntimeError, match="model_factory"):
        d.score(obs)


def test_detector_loads_full_pickle_with_unsafe_load_and_logs_warning(tmp_path, caplog):
    """Detector loads full-module pickle when allow_unsafe_load=True and logs a warning."""
    # Save a full module
    p = tmp_path / "full_module.pt"
    torch.save(_Tiny2Class(), p)

    d = NPRDetector(weights_path=p, model_factory=None, allow_unsafe_load=True)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    obs = [_obs(img)]

    with caplog.at_level(logging.WARNING):
        r = d.score(obs)

    # Must score successfully
    assert not r.abstained
    # Must have logged a warning about unsafe load
    assert any("allow_unsafe_load" in record.message for record in caplog.records)
    assert any("arbitrary code execution" in record.message for record in caplog.records)


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_detector_fails_on_invalid_model_output_shape(tmp_path, tiny_model_factory):
    """Detector raises ValueError when model output has wrong number of classes."""
    # Save a 3-class model as state_dict
    p = tmp_path / "bad_classes.pt"
    model_3class = _Tiny3Class()
    torch.save(model_3class.state_dict(), p)

    # Factory for 3-class model
    def factory_3class():
        return _Tiny3Class()

    d = NPRDetector(weights_path=p, model_factory=factory_3class)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    obs = [_obs(img)]

    with pytest.raises(ValueError, match="expected 2 output classes"):
        d.score(obs)


def test_model_factory_with_full_module_pickle_gives_actionable_error(tmp_path, tiny_model_factory):
    """A full-module pickle supplied alongside model_factory gets a clear, actionable error.

    Previously this path re-raised torch's raw weights_only unpickling failure —
    internal WeightsUnpickler diagnostics instead of a statement of the actual
    conflict. The file is still correctly rejected either way (this is a
    usability fix, not a security fix); the message must name the conflict:
    model_factory expects a state_dict, but the file is a full module.
    """
    p = tmp_path / "full_module.pt"
    torch.save(_Tiny2Class(), p)

    d = NPRDetector(weights_path=p, model_factory=tiny_model_factory)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    obs = [_obs(img)]

    with pytest.raises(RuntimeError, match="mutually exclusive"):
        d.score(obs)


def test_detector_cache_invalidation_on_file_replacement(tmp_path, tiny_model_factory):
    """Cache is invalidated when weights file is replaced (mtime/size change)."""
    weights_path = tmp_path / "weights.pt"

    # Create and save first state_dict
    model1 = _Tiny2Class()
    torch.save(model1.state_dict(), weights_path)
    d = NPRDetector(weights_path=weights_path, model_factory=tiny_model_factory)

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    obs = [_obs(img)]

    # First load
    r1 = d.score(obs)
    assert not r1.abstained

    # Replace the weights file (simulate model rotation)
    import time

    time.sleep(0.01)  # Ensure mtime changes
    model2 = _Tiny2Class()
    # Modify the model to produce different output
    with torch.no_grad():
        model2.fc.weight.fill_(2.0)
    torch.save(model2.state_dict(), weights_path)

    # Second load should get the new model (not cached)
    r2 = d.score(obs)
    assert not r2.abstained
    # CRITICAL: assert the score actually changed
    assert r1.score != r2.score, (
        "cache served the stale model after the weights file was replaced"
    )


def test_model_cache_distinguishes_different_factories_over_same_file(tmp_path):
    """Two detectors sharing one weights file but different factories must not collide.

    Regression test for the Round 3 bug: the cache key was
    (resolved_path, mtime_ns, size) only, so a second NPRDetector built with a
    different model_factory over the same file silently received the first
    detector's cached model (the wrong architecture). The two factories here
    load the identical state_dict but negate the logits, so a correct cache
    key must yield genuinely different scores.
    """
    weights_path = tmp_path / "shared.pt"
    torch.save(_Tiny2Class().state_dict(), weights_path)

    def factory_positive():
        return _Tiny2Class()

    def factory_negated():
        return _NegatedTiny2Class()

    d_positive = NPRDetector(weights_path=weights_path, model_factory=factory_positive)
    d_negated = NPRDetector(weights_path=weights_path, model_factory=factory_negated)

    img = np.random.default_rng(20).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    obs = [_obs(img)]

    r_positive = d_positive.score(obs)
    r_negated = d_negated.score(obs)

    assert not r_positive.abstained and not r_negated.abstained
    # CRITICAL: assert the two factories' scores actually differ
    assert r_positive.score != r_negated.score, (
        "cache served one factory's model to the other factory's detector: "
        f"factory_positive={r_positive.score} vs factory_negated={r_negated.score}"
    )


# ============================================================================
# Integration Tests
# ============================================================================


def test_detector_abstains_below_quality_floor_secure_path(tiny_weights, tiny_model_factory):
    """Detector reports BELOW_FLOOR when all observations fail quality check (secure path)."""
    d = NPRDetector(
        weights_path=tiny_weights,
        model_factory=tiny_model_factory,
        min_quality_band="high",
    )

    img = np.random.default_rng(14).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    obs = [_obs(img, band="low"), _obs(img, band="low")]

    r = d.score(obs)

    assert r.abstained and r.reason == BELOW_FLOOR


def test_detector_abstains_quality_not_measured_secure_path(tiny_weights, tiny_model_factory):
    """Detector reports NO_QUALITY when no observation has quality measured (secure path)."""
    d = NPRDetector(weights_path=tiny_weights, model_factory=tiny_model_factory)

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    obs = [
        Observation(t=0.0, payload=img, roi=(0, 0, 64, 64), quality=None, source_id="s1"),
        Observation(t=1.0, payload=img, roi=(0, 0, 64, 64), quality=None, source_id="s1"),
    ]

    r = d.score(obs)

    assert r.abstained and r.reason == NO_QUALITY
