"""Tests for the EfficientNet-B4 detector (slots A — SBI, and E — FF++ appearance)."""
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
from dfd.detectors.effnet import EffNetDetector, preprocess
from dfd.types import Modality, Observation, Quality


class _Tiny2Class(nn.Module):
    """Minimal 2-class stand-in for EfficientNet-B4: same (N,3,H,W) -> (N,2)
    contract, cheap enough to run in a unit test."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.mean(dim=(2, 3)))


class _Tiny3Class(nn.Module):
    """3-class model for testing the output-shape validation error."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.mean(dim=(2, 3)))


def _tiny_factory():
    return _Tiny2Class()


def _obs(img: np.ndarray, band: str = "high", quality: bool = True) -> Observation:
    """Build an Observation, optionally with quality=None."""
    q = None
    if quality:
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


@pytest.fixture
def weights_a(tmp_path):
    """A state_dict whose two output rows differ, so softmax is not
    trivially 0.5 (symmetric weights would make class 0 and class 1 logits
    identical regardless of input, masking any real distinguishing signal)."""
    p = tmp_path / "sbi.pt"
    model = _Tiny2Class()
    with torch.no_grad():
        model.fc.weight.copy_(torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]))
        model.fc.bias.fill_(0.0)
    torch.save(model.state_dict(), p)
    return p


@pytest.fixture
def weights_b(tmp_path):
    """A state_dict with distinctly different (and oppositely-biased) weights
    from weights_a, so the two slots should score the same input differently."""
    p = tmp_path / "ffpp.pt"
    model = _Tiny2Class()
    with torch.no_grad():
        model.fc.weight.copy_(torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]))
        model.fc.bias.copy_(torch.tensor([2.0, -2.0]))
    torch.save(model.state_dict(), p)
    return p


# ============================================================================
# preprocess()
# ============================================================================


def test_preprocess_produces_nchw_float32_in_unit_range():
    img = np.full((224, 224, 3), 255, dtype=np.uint8)
    t = preprocess([img])
    assert t.shape == (1, 3, 224, 224)
    assert t.dtype == np.float32
    assert 0.0 <= float(t.min()) and float(t.max()) <= 1.0


def test_preprocess_resizes_nonsquare_input_correctly():
    """A non-square input is resized to the requested square size, and the
    resize is not a crop or a distorting reshape: a solid-colour image stays
    solid after resizing, proving the pixels were actually resampled rather
    than merely reinterpreted."""
    img = np.zeros((97, 61, 3), dtype=np.uint8)
    img[:, :] = (10, 20, 30)
    out = preprocess([img], size=224)
    assert out.shape == (1, 3, 224, 224)
    # Every resized pixel should still equal the (normalised) solid colour.
    expected = np.array([10, 20, 30], dtype=np.float32) / 255.0
    for c in range(3):
        assert np.allclose(out[0, c], expected[c], atol=1e-6)


# ============================================================================
# Identity / protocol
# ============================================================================


def test_detector_is_frozen(tmp_path):
    d = EffNetDetector(name="sbi", slot="A", weights_path=tmp_path / "w.pt")
    with pytest.raises(Exception):
        d.name = "modified"  # type: ignore


def test_detector_modalities_is_frozenset(tmp_path):
    d = EffNetDetector(name="sbi", slot="A", weights_path=tmp_path / "w.pt")
    assert isinstance(d.modalities, frozenset)
    assert Modality.IMAGE in d.modalities
    assert Modality.VIDEO in d.modalities


# ============================================================================
# Abstention paths (no real scoring)
# ============================================================================


def test_empty_observation_list_gives_no_observations(tmp_path):
    d = EffNetDetector(name="sbi", slot="A", weights_path=tmp_path / "w.pt")
    r = d.score([])
    assert r.abstained and r.reason == NO_OBSERVATIONS


def test_abstains_when_weights_absent(tmp_path):
    d = EffNetDetector(name="sbi", slot="A", weights_path=tmp_path / "missing.pt")
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    r = d.score([_obs(img)])
    assert r.abstained and r.reason == WEIGHTS_ABSENT


# ============================================================================
# Real scoring path (secure: model_factory + state_dict)
# ============================================================================


def test_real_score_on_secure_path_in_unit_range_no_warning(weights_a, caplog):
    d = EffNetDetector(
        name="sbi", slot="A", weights_path=weights_a, model_factory=_tiny_factory
    )
    img = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)

    with caplog.at_level(logging.WARNING):
        r = d.score([_obs(img)])

    assert not r.abstained, f"expected a score, got abstention: {r.reason}"
    assert r.score is not None
    assert 0.0 <= r.score <= 1.0
    assert not any("allow_unsafe_load" in rec.message for rec in caplog.records)


def test_artifacts_carry_required_fields(weights_a):
    d = EffNetDetector(
        name="sbi", slot="A", weights_path=weights_a, model_factory=_tiny_factory
    )
    rng = np.random.default_rng(1)
    imgs = [rng.integers(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
    r = d.score([_obs(img) for img in imgs])

    assert not r.abstained
    assert "per_observation_scores" in r.artifacts
    assert len(r.artifacts["per_observation_scores"]) == 3
    assert "max_score" in r.artifacts
    assert r.artifacts["max_score"] >= r.score
    assert r.artifacts["n_observations"] == 3
    assert r.artifacts["slot"] == "A"


def test_wrong_shaped_model_output_raises_clear_error(tmp_path):
    p = tmp_path / "bad.pt"
    torch.save(_Tiny3Class().state_dict(), p)

    def factory_3class():
        return _Tiny3Class()

    d = EffNetDetector(name="sbi", slot="A", weights_path=p, model_factory=factory_3class)
    img = np.zeros((64, 64, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="expected 2 output classes"):
        d.score([_obs(img)])


# ============================================================================
# Slots A and E: same class, different weights
# ============================================================================


def test_slots_a_and_e_are_same_class_with_different_scores(weights_a, weights_b):
    sbi = EffNetDetector(name="sbi", slot="A", weights_path=weights_a, model_factory=_tiny_factory)
    appearance = EffNetDetector(
        name="effnet_ffpp", slot="E", weights_path=weights_b, model_factory=_tiny_factory
    )
    assert type(sbi) is type(appearance)
    assert sbi.slot != appearance.slot

    img = np.random.default_rng(2).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    r_a = sbi.score([_obs(img)])
    r_e = appearance.score([_obs(img)])

    assert not r_a.abstained and not r_e.abstained
    assert r_a.score != r_e.score, (
        "same architecture but different weights produced identical scores; "
        f"slot A={r_a.score} slot E={r_e.score}"
    )


# ============================================================================
# Mixed-batch abstention: order independence
# ============================================================================


def test_mixed_batch_abstention_is_order_independent(tmp_path):
    """A batch with one measured-but-below-floor observation and one
    unmeasured (quality=None) observation must abstain with BELOW_FLOOR in
    BOTH orderings — not NO_QUALITY, and not dependent on which one is
    obs[0]. This is the exact order-dependent bug class already fixed twice
    elsewhere in this codebase (base.SyntheticDetector, npr.NPRDetector)."""
    (tmp_path / "w.pt").write_bytes(b"not-a-real-model")  # never reached: abstains first
    d = EffNetDetector(
        name="sbi", slot="A", weights_path=tmp_path / "w.pt", min_quality_band="high"
    )
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    below = _obs(img, band="low")
    unmeasured = _obs(img, quality=False)

    r1 = d.score([unmeasured, below])
    r2 = d.score([below, unmeasured])

    assert r1.abstained and r1.reason == BELOW_FLOOR, (
        f"[unmeasured, below] gave {r1.reason}, expected {BELOW_FLOOR}"
    )
    assert r2.abstained and r2.reason == BELOW_FLOOR, (
        f"[below, unmeasured] gave {r2.reason}, expected {BELOW_FLOOR}"
    )


def test_abstains_quality_not_measured_when_nothing_was_measured(tmp_path):
    # The weights file must exist (else WEIGHTS_ABSENT would fire first) but
    # its contents are irrelevant: this test only exercises the
    # quality-filtering branch, which runs before the model is ever loaded.
    (tmp_path / "w.pt").write_bytes(b"not-a-real-model")
    d = EffNetDetector(name="sbi", slot="A", weights_path=tmp_path / "w.pt")
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    r = d.score([_obs(img, quality=False), _obs(img, quality=False)])
    assert r.abstained and r.reason == NO_QUALITY
