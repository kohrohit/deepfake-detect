"""The detector abstains honestly, and its model file is inert data."""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from dfd.detectors.base import WEIGHTS_ABSENT
from dfd.detectors.blend import (
    FEATURE_NAMES,
    BlendDetector,
    BlendModel,
    load_blend_model,
    save_blend_model,
)
from dfd.types import Modality, Observation, Quality


def _model(coef: np.ndarray | None = None) -> BlendModel:
    n = len(FEATURE_NAMES)
    # scale=1e5 (not 1.0): seam_features on an unstandardised random image
    # runs laplacian_var into the tens of thousands, so with scale=1.0 and
    # coef=ones the logit saturates the float64 sigmoid to exactly 1.0 for
    # every observation, making test_two_observations_are_averaged pass
    # vacuously (both means compare 1.0 == 1.0) regardless of whether the
    # implementation actually averages. 1e5 keeps z-scores O(1) so the two
    # scores differ and the averaging test discriminates real behaviour.
    return BlendModel(
        mean=np.zeros(n), scale=np.full(n, 1e5),
        coef=np.ones(n) if coef is None else coef,
        intercept=0.0, feature_names=FEATURE_NAMES, version="test-1")


def _obs(band: str = "high", seed: int = 0) -> Observation:
    rng = np.random.default_rng(seed)
    return Observation(
        t=0.0, payload=rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        roi=None,
        quality=Quality(inter_ocular_px=40.0, blur_var=120.0, yaw_deg=0.0,
                        pitch_deg=0.0, exposure=0.5, band=band),
        source_id="s1")


def test_abstains_when_the_model_file_is_absent(tmp_path: Path) -> None:
    d = BlendDetector(weights_path=tmp_path / "nope.npz")
    r = d.score([_obs()])
    assert r.abstained and r.score is None and r.reason == WEIGHTS_ABSENT


def test_abstains_below_the_quality_floor(tmp_path: Path) -> None:
    p = tmp_path / "m.npz"
    save_blend_model(_model(), p)
    r = BlendDetector(weights_path=p).score([_obs(band="low")])
    assert r.abstained and r.reason == "below_quality_floor"


def test_scores_in_the_unit_interval(tmp_path: Path) -> None:
    p = tmp_path / "m.npz"
    save_blend_model(_model(), p)
    r = BlendDetector(weights_path=p).score([_obs()])
    assert not r.abstained
    assert r.score is not None and 0.0 <= r.score <= 1.0


def test_round_trips_through_the_file(tmp_path: Path) -> None:
    p = tmp_path / "m.npz"
    m = _model(coef=np.linspace(-1, 1, len(FEATURE_NAMES)))
    save_blend_model(m, p)
    back = load_blend_model(p)
    assert np.allclose(back.coef, m.coef)
    assert back.feature_names == m.feature_names
    assert back.version == m.version


def test_the_model_file_contains_no_pickle(tmp_path: Path) -> None:
    """The supply-chain property. An npz of plain arrays cannot execute code
    on load; a pickled sklearn estimator can, and that is why one is not used.
    """
    p = tmp_path / "m.npz"
    save_blend_model(_model(), p)
    with zipfile.ZipFile(p) as z:
        for name in z.namelist():
            head = z.read(name)[:80]
            assert b"sklearn" not in head
            assert b"__reduce__" not in head
    loaded = np.load(p, allow_pickle=False)  # must not raise
    assert "coef" in loaded
    # `"coef" in loaded` only checks the npz's member list; NpzFile is lazy
    # and does not decode an array until it is indexed. Force a real read of
    # every field, the way load_blend_model does, so this test cannot pass
    # on a file that merely looks clean without actually being decodable.
    for key in loaded.files:
        np.asarray(loaded[key])


def test_a_pickled_field_is_refused_by_allow_pickle_false(tmp_path: Path) -> None:
    """The actual guard is allow_pickle=False at load time, not scanning file
    bytes for suspicious substrings.

    A discovered gap: the byte-substring scan above only inspects the first
    80 bytes of each zip member, but an object-dtype array's .npy header
    (the descr/shape/fortran_order dict, padded to alignment) runs well past
    80 bytes before the pickle opcodes begin — confirmed by writing an
    object array and observing "sklearn" appears in the full 315-byte
    payload but not in bytes[:80]. So that scan would pass on a tampered
    file that carries a pickled non-array payload in one field, as long as
    the attacker's marker text does not happen to fall in the header. This
    test proves the property that actually matters: a pickled field is
    refused when read with allow_pickle=False, by both raw numpy and by
    load_blend_model.
    """
    p = tmp_path / "m.npz"
    save_blend_model(_model(), p)
    tampered = dict(np.load(p, allow_pickle=True))
    # Object dtype, but with plain floats as elements: a tampering that
    # allow_pickle=False must refuse on dtype alone, not one that would
    # incidentally fail float64 conversion anyway (a string payload would
    # raise for that unrelated reason and mask a mutation that dropped the
    # allow_pickle=False guard from load_blend_model).
    tampered["coef"] = np.array([1.0] * len(FEATURE_NAMES), dtype=object)
    np.savez(p, **tampered)
    with pytest.raises(ValueError):
        np.asarray(np.load(p, allow_pickle=False)["coef"])
    with pytest.raises(ValueError):
        load_blend_model(p)


def test_a_model_with_a_different_format_version_is_refused(tmp_path: Path) -> None:
    """The other half of the stale-model guard. Both checks in
    load_blend_model are load-bearing; the feature-name mismatch test above
    does not exercise this branch, and it must be proven independently."""
    p = tmp_path / "m.npz"
    save_blend_model(_model(), p)
    np.savez(p, **{**dict(np.load(p, allow_pickle=False)),
                   "format_version": np.array(999)})
    with pytest.raises(ValueError, match="format version"):
        load_blend_model(p)


def test_a_model_whose_features_do_not_match_is_refused(tmp_path: Path) -> None:
    """A stale model file is worse than no model file: it would score
    confidently against the wrong columns."""
    p = tmp_path / "m.npz"
    save_blend_model(_model(), p)
    np.savez(p, **{**dict(np.load(p, allow_pickle=False)),
                   "feature_names": np.array(["wrong"], dtype=np.str_)})
    with pytest.raises(ValueError, match="feature names"):
        load_blend_model(p)


def test_identity_is_stable(tmp_path: Path) -> None:
    d = BlendDetector(weights_path=tmp_path / "x.npz")
    assert d.name == "blend_seam"
    assert d.slot == "A"
    assert Modality.IMAGE in d.modalities
    assert d.min_quality_band == "medium"


def test_two_observations_are_averaged_not_only_the_first(tmp_path: Path) -> None:
    p = tmp_path / "m.npz"
    save_blend_model(_model(), p)
    d = BlendDetector(weights_path=p)
    one = d.score([_obs(seed=1)])
    two = d.score([_obs(seed=1), _obs(seed=2)])
    assert one.score is not None and two.score is not None
    assert one.score != two.score
