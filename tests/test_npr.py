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
from dfd.detectors.npr import NPRDetector, NPRStatsNet, npr_feature
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


# --- Slot C's actual architecture (added 2026-09-23) -----------------------
#
# Until now `npr` shipped a feature function and no model, so the slot
# abstained with `weights_absent` on every input this project has ever
# scored. `NPRStatsNet` is the light head the CPU-only ruling calls for
# (docs/HANDOFF.md §1, correction 3): handcrafted statistics of the
# upsampling residual feeding a linear layer, fitted by
# `training/fit_npr.py`.

def test_the_residual_is_identically_zero_on_the_even_phase():
    """Not a bug — the definition. Nearest-neighbour upsampling REPLICATES
    the sampled pixel, so `up[2i, 2j] == x[2i, 2j]` for every image, real or
    generated. Any statistic computed over that phase is a constant 0 and
    carries no information, which is why `NPRStatsNet` reads the other three.
    """
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    r = npr_feature(img)
    assert np.abs(r[0::2, 0::2]).max() == 0.0
    assert np.abs(r[1::2, 1::2]).max() > 0.0


def test_stats_net_emits_two_logits_per_sample():
    net = NPRStatsNet()
    out = net(torch.zeros(4, 3, 32, 32))
    assert out.shape == (4, 2)


def test_stats_are_finite_on_a_flat_image():
    """A constant image has a zero residual in every phase, so every ratio is
    0/0. A NaN here would poison the linear layer silently."""
    net = NPRStatsNet()
    f = net.features(torch.zeros(1, 3, 32, 32))
    assert f.shape == (1, NPRStatsNet.N_FEATURES)
    assert torch.isfinite(f).all()


def test_an_image_too_small_to_have_every_phase_is_still_finite():
    """A 1x1 crop leaves all three informative phases EMPTY.

    Without the guard, `mean` over an empty tensor is nan, and torch does
    not raise — the nan reaches the linear layer and every logit becomes
    nan. The detector's quality floor should keep crops this small out, but
    "should" is not a guard, and a nan that only appears on degenerate input
    is the kind that reaches production.
    """
    net = NPRStatsNet()
    f = net.features(torch.zeros(1, 3, 1, 1))
    assert f.shape == (1, NPRStatsNet.N_FEATURES)
    assert torch.isfinite(f).all()


def test_stats_separate_upsampled_content_from_natural_content():
    """The physics the slot exists for, asserted rather than assumed.

    Upsampled content reconstructs almost exactly under downsample-then-
    upsample, so its residual is small; natural high-frequency content does
    not, so its residual is large.
    """
    rng = np.random.default_rng(1)
    natural = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    small = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    upsampled = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)
    net = NPRStatsNet()
    def energy(img):
        r = torch.from_numpy(npr_feature(img)).permute(2, 0, 1)[None]
        return float(net.features(r)[0, :9].abs().mean())
    assert energy(upsampled) < energy(natural) / 2


def test_the_normalisation_buffers_are_applied():
    """They travel in the state_dict, so a fitted model that ignored them
    would load clean and score as though it had never been standardised."""
    net = NPRStatsNet()
    rng = np.random.default_rng(2)
    x = torch.from_numpy(rng.normal(0, 0.2, (2, 3, 32, 32)).astype(np.float32))
    before = net(x).clone()
    with torch.no_grad():
        net.feature_mean.add_(1.0)
        net.feature_scale.mul_(3.0)
    assert not torch.allclose(before, net(x))


def test_a_fitted_state_dict_loads_through_the_secure_path(tmp_path):
    """End to end: what `training/fit_npr.py` writes is what the registry's
    `model_factory` can load with weights_only=True."""
    net = NPRStatsNet()
    with torch.no_grad():
        net.linear.weight.copy_(torch.randn(2, NPRStatsNet.N_FEATURES))
    p = tmp_path / "npr.pt"
    torch.save(net.state_dict(), p)
    d = NPRDetector(weights_path=p, model_factory=NPRStatsNet)
    rng = np.random.default_rng(3)
    img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    score = d.score([_obs(img)])
    assert not score.abstained
    assert 0.0 <= score.score <= 1.0


def test_the_registry_wires_the_factory_so_npr_never_needs_unsafe_loading():
    """Without this the slot could only load a full pickle, which is the
    path `loading.load_model` logs a warning for on every call."""
    from dfd.detectors.registry import default_registry
    npr = default_registry().get("npr")
    assert npr.model_factory is NPRStatsNet
    assert npr.allow_unsafe_load is False


# --- Magnitude features, added 2026-09-24 ------------------------------
#
# Measured on 506 frames of the v-CIP capture corpus (106 sessions, real
# `inswapper_128` swaps of genuine capture frames): the 27 phase features
# alone reach AUC 0.861 but catch only 4.4% of swapped sessions at a ZERO
# false-alarm budget, which is the operating point the product needs. Adding
# per-channel magnitude moments and a coarse spectral profile takes the same
# crops to 45.6% at the same budget (AUC 0.957).
#
# Re-measured 2026-09-25 by `bench.vcip_controls.preprocessing_and_blocks`;
# this comment read 0.869/7.4% and 92.6% until then, from a run made before
# `training.fit_vcip.select_face` was corrected to the pipeline's rule.
#
# The phase features are KEPT rather than replaced: they read a scale-free
# property (how residual energy distributes across sampling phases) that the
# magnitude features cannot express, and the combination beat either alone.

def test_feature_vector_has_the_declared_length():
    """`N_FEATURES` is what a fitted state_dict is tied to. A mismatch here
    loads clean and scores nonsense."""
    import torch

    from dfd.detectors.npr import NPRStatsNet
    net = NPRStatsNet()
    r = torch.randn(4, 3, 64, 64)
    assert net.features(r).shape == (4, NPRStatsNet.N_FEATURES)


def test_the_spectral_block_reads_something_the_phase_block_cannot():
    """The capability being added, as an exact case rather than an argument.

    Shuffling the pixels WITHIN each sampling phase is a permutation of that
    phase's values, so every per-phase mean, std and ratio is bit-identical
    afterwards — the 27 phase features cannot see it at all. The spatial
    structure, and therefore the spectrum, is destroyed.

    That is exactly the distinction that matters in the field: an upsampled
    generator patch is smooth where sensor noise is not, while both can carry
    the same per-phase magnitude. On the capture corpus the phase block alone
    caught 4.4% of swapped sessions at a zero false-alarm budget and the full
    vector caught 45.6%, which is why the layout changed.
    """
    import torch

    from dfd.detectors.npr import NPRStatsNet
    net = NPRStatsNet()
    g = torch.Generator().manual_seed(0)
    r = torch.randn(1, 3, 64, 64, generator=g)

    shuffled = r.clone()
    for dy in (0, 1):
        for dx in (0, 1):
            block = shuffled[:, :, dy::2, dx::2]
            flat = block.reshape(block.shape[0], block.shape[1], -1)
            perm = torch.randperm(flat.shape[-1], generator=g)
            shuffled[:, :, dy::2, dx::2] = flat[:, :, perm].reshape(block.shape)

    a, b = net.features(r), net.features(shuffled)
    n_phase = NPRStatsNet.N_PHASE_FEATURES

    # The premise: the phase block genuinely cannot see this.
    assert torch.allclose(a[:, :n_phase], b[:, :n_phase], atol=1e-4), (
        "the within-phase shuffle changed the phase block, so this test is "
        "not measuring what it claims")

    assert NPRStatsNet.N_FEATURES > n_phase, (
        "no spectral block exists; the layout reads phase statistics only")
    assert not torch.allclose(a[:, n_phase:], b[:, n_phase:], atol=1e-3), (
        "the spectral block did not move when the spectrum was destroyed "
        "and every phase statistic held constant; it is not reading "
        "frequency")


def test_features_are_finite_on_a_flat_field_and_a_tiny_image():
    """A flat field divides by zero in every ratio, and an image smaller
    than the stride leaves a phase empty. Both must give numbers, because a
    nan reaches the linear layer and comes out as a confident score."""
    import torch

    from dfd.detectors.npr import NPRStatsNet
    net = NPRStatsNet()
    for r in (torch.zeros(1, 3, 32, 32), torch.zeros(1, 3, 1, 1),
              torch.ones(1, 3, 8, 8)):
        assert torch.isfinite(net.features(r)).all(), f"non-finite on {tuple(r.shape)}"


# --- The detector must read the FACE, at its native resolution ---------
#
# Added 2026-09-24. `score` computed `npr_feature(o.payload)` — the WHOLE
# frame — while every measurement that justified this slot was made on the
# face crop. On the v-CIP capture corpus (720x1280 frames, median face 237px)
# that difference is the whole result: resizing the face to a 224 square
# DOWNSAMPLES the median face and low-pass filters away the upsampling
# fingerprint, taking the zero-false-alarm catch rate from 45.6% to 1.5%
# (re-measured 2026-09-25; this line read 89.7% to 51.5% until then).
#
# So this detector crops to the ROI and does NOT resize, which is the
# opposite of `dfd.detectors.blend._crop_to_roi`. The reason they differ:
# seam features normalise annulus GEOMETRY and need a fixed scale, while the
# NPR residual IS a scale-dependent quantity and normalising it away is the
# defect. Whatever the fitter does here, the detector must match exactly.

def _roi_obs(payload, roi):
    """Like `_obs`, but with an explicit ROI — including None."""
    q = Quality(inter_ocular_px=100, blur_var=200, yaw_deg=0, pitch_deg=0,
                exposure=0.5, band="high")
    return Observation(t=0.0, payload=payload, roi=roi, quality=q,
                       source_id="s0")


def test_score_reads_only_the_roi_pixels(tiny_weights, tiny_model_factory):
    """Scoring a face inside a large frame must equal scoring that crop
    alone. If the whole frame is read, surrounding pixels move the score."""
    import numpy as np

    from dfd.detectors.npr import NPRDetector
    rng = np.random.default_rng(0)
    face = rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)

    frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    frame[100:196, 200:296] = face

    d = NPRDetector(weights_path=tiny_weights,
                    model_factory=tiny_model_factory, min_quality_band="low")
    in_frame = d.score([_roi_obs(frame, (200, 100, 96, 96))])
    alone = d.score([_roi_obs(face, (0, 0, 96, 96))])

    assert not in_frame.abstained and not alone.abstained
    assert in_frame.score == pytest.approx(alone.score, abs=1e-6)


def test_the_crop_keeps_the_rois_own_dimensions():
    """The contract the fitter must match, asserted directly.

    `blend._crop_to_roi` resizes every crop to a 224 square because seam
    features need a fixed scale. This one must NOT, because the NPR residual
    is the difference between an image and its own stride-2 reconstruction —
    a scale-dependent quantity that resampling rewrites.

    Tested on the helper rather than through `score`, because an untrained
    linear layer saturates softmax and hides the difference at the output
    even when the features differ.
    """
    import numpy as np

    from dfd.detectors.npr import _crop_to_roi_native
    frame = np.random.default_rng(3).integers(0, 255, (480, 640, 3),
                                              dtype=np.uint8)
    for w, h in ((64, 64), (237, 190), (400, 300)):
        crop = _crop_to_roi_native(frame, (10, 20, w, h))
        assert crop is not None
        assert crop.shape[:2] == (h, w), (
            f"ROI {w}x{h} produced a {crop.shape[1]}x{crop.shape[0]} crop; "
            f"the crop was resampled")


def test_the_crop_is_a_view_of_the_requested_region():
    """Right size, wrong pixels is the failure this catches — an off-by-one
    or a transposed (x, y) would still pass a shape assertion."""
    import numpy as np

    from dfd.detectors.npr import _crop_to_roi_native
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[30:70, 50:120] = 255
    crop = _crop_to_roi_native(frame, (50, 30, 70, 40))
    assert crop is not None
    assert (crop == 255).all(), "the crop is not the region that was asked for"


def test_the_crop_is_clamped_to_the_frame():
    """A detector may hand back a box hanging off the frame edge, which is
    routine on a tightly framed face. An unclamped negative origin slices
    from the FAR END of the array and yields the wrong pixels silently."""
    import numpy as np

    from dfd.detectors.npr import _crop_to_roi_native
    frame = np.random.default_rng(4).integers(0, 255, (100, 100, 3),
                                              dtype=np.uint8)
    crop = _crop_to_roi_native(frame, (-20, -20, 60, 60))
    assert crop is not None
    assert crop.shape[:2] == (40, 40)
    assert np.array_equal(crop, frame[0:40, 0:40])
    assert _crop_to_roi_native(frame, (99, 99, 1, 1)) is None


def test_score_abstains_when_there_is_no_roi(tiny_weights, tiny_model_factory):
    """No ROI means the detector does not know which pixels are the face.
    Scoring the whole frame instead is how it silently reads background."""
    import numpy as np

    from dfd.detectors.npr import NPRDetector, NO_ROI
    d = NPRDetector(weights_path=tiny_weights,
                    model_factory=tiny_model_factory, min_quality_band="low")
    frame = np.random.default_rng(2).integers(0, 255, (128, 128, 3),
                                              dtype=np.uint8)
    out = d.score([_roi_obs(frame, None)])
    assert out.abstained
    assert out.reason == NO_ROI


def test_the_quality_floor_does_not_discard_the_reject_band():
    """Slot C must score blurred faces, not refuse them.

    Measured on the v-CIP capture corpus, per band, held-out scores from
    folds split on session (`bench.vcip_controls.per_band`, re-measured
    2026-09-25):

        band      n    genuine  swapped   AUC    caught at zero false alarms
        reject   240        14      226   0.803   26.5%
        medium   244       140      104   0.947   85.6%
        low        8         2        6   1.000  100.0%
        high      14        14        0     —       —

    And the floor at `low` was discarding the `reject` band entirely — 226
    of 336 swapped frames (67.3%) against 14 of 170 genuine.

    **The composition columns are the argument, not the AUC columns**, which
    is why this default survived their re-measurement unchanged. The table
    here until 2026-09-25 read 0.992/94.2% for `reject` and 0.957/29.8% for
    `medium` — very nearly the other way round — from a run made before
    `training.fit_vcip.select_face` was corrected to the pipeline's rule.
    The n, genuine and swapped columns are identical in both.

    The asymmetry has a mechanism rather than being a quirk: `inswapper_128`
    emits a 128x128 face pasted back upscaled, so a swap BLURS, and a
    sharpness-based floor reads blur as a bad capture. For every other
    detector low quality means low reliability; for this one it is positively
    correlated with the artefact being looked for. A floor above `reject`
    therefore discards the most detectable attacks before the detector runs,
    which is a security hole rather than caution.

    Reduced reliability at low quality is still real, and it is expressed
    where it belongs — the per-band calibration curve (`dfd.calibration`),
    which is exactly what per-band calibration is for.

    READ THE DENOMINATORS: the reject band holds 14 genuine frames and the
    low band holds 2, so their false-alarm columns rest on 14 and 2
    negatives; `low`'s 1.000 over eight frames means nothing at all, and
    `high` holds no swapped frame so it has no AUC. What the re-measured
    table does show clearly is the reliability drop — `reject` is now the
    worst band rather than the best — which is the thing per-band
    calibration exists to carry.
    """
    from dfd.detectors.npr import NPRDetector
    assert NPRDetector(weights_path="unused.pt").min_quality_band == "reject"


def test_a_reject_band_observation_is_scored_not_refused(
        tiny_weights, tiny_model_factory):
    """The behaviour the floor change exists to produce."""
    import numpy as np

    from dfd.detectors.npr import NPRDetector
    q = Quality(inter_ocular_px=10, blur_var=1.0, yaw_deg=0, pitch_deg=0,
                exposure=0.5, band="reject")
    img = np.random.default_rng(5).integers(0, 255, (64, 64, 3),
                                            dtype=np.uint8)
    obs = Observation(t=0.0, payload=img, roi=(0, 0, 64, 64), quality=q,
                      source_id="s0")
    out = NPRDetector(weights_path=tiny_weights,
                      model_factory=tiny_model_factory).score([obs])
    assert not out.abstained, f"refused a reject-band frame: {out.reason}"
