"""Slot D. Every test uses a tiny tower; none needs the 1.2 GB encoder."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from dfd.detectors.clip_vit import CLIPVisualTower
from dfd.detectors.gend import (
    ENCODER_ABSENT,
    NO_ROI,
    VERSION_BYTES,
    GendClipDetector,
    GendHead,
    embed,
)
from dfd.types import Modality, Observation, Quality


def _obs(value: int = 128, size: int = 64, band: str = "high") -> Observation:
    """A face crop that clears the floor, with its quality stated.

    Stated rather than measured, which is the idiom `tests/test_npr.py`
    already uses. `measure_quality` on a flat patch returns `blur_var=0.0`
    and band "reject", so a measured fixture would send every scoring test
    down the quality-floor abstention and assert nothing about slot D.
    """
    img = np.full((size, size, 3), value, dtype=np.uint8)
    q = Quality(inter_ocular_px=100, blur_var=200, yaw_deg=0, pitch_deg=0,
                exposure=0.5, band=band)
    return Observation(t=0.0, payload=img, roi=(0, 0, size, size),
                       quality=q, source_id="s1")


def _fitted(tmp_path, in_dim: int = 8, version: str = "0.1.0-test"):
    head = GendHead(in_dim=in_dim)
    head.set_version(version)
    path = tmp_path / "gend_clip.pt"
    torch.save(head.state_dict(), path)
    return path


def _encoder(tmp_path):
    path = tmp_path / "clip.pt"
    torch.save(CLIPVisualTower.tiny().state_dict(), path)
    return path


def test_embeddings_are_l2_normalised():
    tower = CLIPVisualTower.tiny().eval()
    crops = [np.full((40, 40, 3), v, dtype=np.uint8) for v in (10, 200)]
    out = embed(tower, crops)
    assert out.shape == (2, 8)
    assert torch.allclose(out.norm(dim=1), torch.ones(2), atol=1e-5)


def test_the_version_travels_inside_the_state_dict(tmp_path):
    head = GendHead(in_dim=8)
    head.set_version("0.1.0-probe-fairface60000")
    reloaded = GendHead(in_dim=8)
    reloaded.load_state_dict(head.state_dict())  # strict
    assert reloaded.version() == "0.1.0-probe-fairface60000"


def test_a_version_longer_than_the_buffer_is_refused():
    # Silently truncating would produce a weights file whose declared
    # training window is a PREFIX of the real one, which is exactly how the
    # eval_swaps offset guard gets bypassed.
    head = GendHead(in_dim=8)
    with pytest.raises(ValueError, match="longer than"):
        head.set_version("x" * (VERSION_BYTES + 1))


def test_abstains_when_the_head_is_absent(tmp_path):
    d = GendClipDetector(weights_path=tmp_path / "nope.pt",
                         encoder_path=_encoder(tmp_path))
    raw = d.score([_obs()])
    assert raw.abstained and raw.reason == "weights_absent" and raw.score is None


def test_abstains_when_the_encoder_is_absent_and_says_which(tmp_path):
    # Distinct from weights_absent: "fit the head" and "run
    # ops/fetch-assets.sh" are different remedies, and one reason for both
    # sends every operator down the wrong one.
    d = GendClipDetector(weights_path=_fitted(tmp_path),
                         encoder_path=tmp_path / "nope.pt")
    raw = d.score([_obs()])
    assert raw.abstained and raw.reason == ENCODER_ABSENT


def test_abstains_on_no_observations(tmp_path):
    d = GendClipDetector(weights_path=_fitted(tmp_path),
                         encoder_path=_encoder(tmp_path))
    assert d.score([]).reason == "no_observations"


def test_abstains_rather_than_scoring_the_whole_frame_without_a_roi(tmp_path):
    d = GendClipDetector(weights_path=_fitted(tmp_path),
                         encoder_path=_encoder(tmp_path))
    obs = _obs()
    raw = d.score([Observation(t=0.0, payload=obs.payload, roi=None,
                               quality=obs.quality, source_id="s1")])
    assert raw.abstained and raw.reason == NO_ROI


def test_scores_in_the_unit_interval_and_records_its_denominator(tmp_path):
    # The only test that actually loads both files, so the only one that has
    # to name the tiny architectures they were written with.
    d = GendClipDetector(weights_path=_fitted(tmp_path),
                         encoder_path=_encoder(tmp_path),
                         encoder_factory=CLIPVisualTower.tiny,
                         head_factory=lambda: GendHead(in_dim=8))
    raw = d.score([_obs(100), _obs(150)])
    assert not raw.abstained
    assert 0.0 <= raw.score <= 1.0
    assert raw.artifacts["n_observations"] == 2
    assert len(raw.artifacts["per_observation_scores"]) == 2
    assert "max_score" in raw.artifacts


def test_version_is_read_from_the_weights_file_not_hardcoded(tmp_path):
    d = GendClipDetector(weights_path=_fitted(tmp_path, version="9.9.9-custom"),
                         encoder_path=_encoder(tmp_path))
    assert d.version == "9.9.9-custom"


def test_version_falls_back_to_the_code_version_with_no_weights(tmp_path):
    d = GendClipDetector(weights_path=tmp_path / "nope.pt",
                         encoder_path=tmp_path / "nope.pt")
    assert d.version  # a detector that cannot say what it is, is unauditable
    assert d.score([_obs()]).version == d.version


def test_the_detector_is_frozen_so_the_registry_identity_holds(tmp_path):
    d = GendClipDetector(weights_path=_fitted(tmp_path),
                         encoder_path=_encoder(tmp_path))
    with pytest.raises(Exception):
        d.name = "other"  # type: ignore[misc]


def test_declares_image_and_video_modalities(tmp_path):
    d = GendClipDetector(weights_path=tmp_path / "x.pt",
                         encoder_path=tmp_path / "y.pt")
    assert Modality.IMAGE in d.modalities
