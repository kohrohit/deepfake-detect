"""The detector abstains honestly, and its model file is inert data."""
from __future__ import annotations

import warnings
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from dfd.detectors.base import WEIGHTS_ABSENT
from dfd.detectors.blend import (
    CODE_VERSION,
    FEATURE_NAMES,
    NO_ROI,
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


def _obs(band: str = "high", seed: int = 0,
         roi: tuple[int, int, int, int] | None = (0, 0, 224, 224)) -> Observation:
    rng = np.random.default_rng(seed)
    return Observation(
        t=0.0, payload=rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        roi=roi,
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


def test_weights_absent_takes_priority_over_quality_when_both_apply(
        tmp_path: Path) -> None:
    """Ordering matters for the audit record's `reason` field, which a
    caller reads to decide what to do next. NPR and EffNet both check
    weights_path.exists() before filtering by quality; Blend must match, or
    a fresh checkout (no model file, the documented default state) blames a
    low-quality capture for a problem that is actually a missing model —
    someone would go improve their capture quality and still get nothing."""
    d = BlendDetector(weights_path=tmp_path / "nope.npz")
    r = d.score([_obs(band="low")])
    assert r.abstained and r.reason == WEIGHTS_ABSENT


def test_scores_in_the_unit_interval(tmp_path: Path) -> None:
    p = tmp_path / "m.npz"
    save_blend_model(_model(), p)
    r = BlendDetector(weights_path=p).score([_obs()])
    assert not r.abstained
    assert r.score is not None and 0.0 <= r.score <= 1.0


def test_predict_proba_clamps_extreme_logits_without_warning() -> None:
    """A large-magnitude logit (e.g. a scaler mismatch on a real fitted
    model) drives exp() to overflow or underflow; the clamped result (0.0
    or 1.0) is correct either way, but this repo keeps test and log output
    pristine, so the overflow path must not emit RuntimeWarning."""
    n = len(FEATURE_NAMES)
    features = np.zeros(n, dtype=np.float32)
    hot = BlendModel(mean=np.zeros(n), scale=np.ones(n), coef=np.zeros(n),
                      intercept=1e6, feature_names=FEATURE_NAMES, version="t")
    cold = BlendModel(mean=np.zeros(n), scale=np.ones(n), coef=np.zeros(n),
                       intercept=-1e6, feature_names=FEATURE_NAMES, version="t")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert hot.predict_proba(features) == 1.0
        assert cold.predict_proba(features) == 0.0


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
                   "format_version": np.array(2)})
    with pytest.raises(ValueError, match="format version"):
        load_blend_model(p)


def test_a_non_integral_format_version_is_refused(tmp_path: Path) -> None:
    """The staleness guard must validate, not coerce. `int(np.array(1.9))`
    truncates to 1 rather than rejecting the value, which would silently
    accept a corrupted or mistyped version field as a match for
    MODEL_FILE_VERSION == 1. A stale (or malformed) model file must be
    refused, not used."""
    p = tmp_path / "m.npz"
    save_blend_model(_model(), p)
    np.savez(p, **{**dict(np.load(p, allow_pickle=False)),
                   "format_version": np.array(1.9)})
    with pytest.raises(ValueError, match="format_version"):
        load_blend_model(p)


def test_a_missing_field_is_refused_with_valueerror_naming_it(
        tmp_path: Path) -> None:
    """The docstring promises ValueError; nothing wraps load_blend_model in
    BlendDetector.score, so an undocumented KeyError from a truncated or
    corrupted file would crash the caller instead of being catchable per
    the documented contract."""
    p = tmp_path / "m.npz"
    save_blend_model(_model(), p)
    data = dict(np.load(p, allow_pickle=False))
    del data["format_version"]
    np.savez(p, **data)
    with pytest.raises(ValueError, match="format_version"):
        load_blend_model(p)


def test_several_missing_fields_are_all_named(tmp_path: Path) -> None:
    p = tmp_path / "m.npz"
    save_blend_model(_model(), p)
    data = dict(np.load(p, allow_pickle=False))
    del data["format_version"]
    del data["coef"]
    del data["version"]
    np.savez(p, **data)
    with pytest.raises(ValueError) as excinfo:
        load_blend_model(p)
    msg = str(excinfo.value)
    # "version" is a substring of "format_version", so a naive `in` check
    # on "version" would pass even if only "format_version" were reported.
    # Extract the actual comma-separated field list to check all three are
    # named as distinct entries, not merely as an accidental substring.
    named = {tok.strip().strip("'\"") for tok in msg.split(":")[-1].split(",")}
    assert named == {"format_version", "coef", "version"}


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


def test_abstains_when_roi_is_none(tmp_path: Path) -> None:
    """Train/serve skew guard (finding 1). Scoring `o.payload` whole would
    silently put the annuli over whatever the caller ingested -- a full
    frame in production -- instead of the aligned face crop the model was
    fitted on. An observation carrying no ROI must abstain, not be scored
    on the wrong pixels."""
    p = tmp_path / "m.npz"
    save_blend_model(_model(), p)
    r = BlendDetector(weights_path=p).score([_obs(roi=None)])
    assert r.abstained and r.score is None and r.reason == NO_ROI


def _ring_crop(size: int = 224) -> np.ndarray:
    """A small textured face-like crop: a soft-edged bright annulus on a
    flat field, distinguishable from a flat background by seam_features."""
    img = np.full((size, size, 3), 128, dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), int(size * 0.3), (200, 200, 200), 6)
    return cv2.GaussianBlur(img, (7, 7), 0)


def test_scores_the_roi_crop_not_the_whole_frame(tmp_path: Path) -> None:
    """The assertion that pins finding 1's fix.

    A crop is embedded off-centre in a much larger, differently-textured
    frame. Scoring through the ROI must equal scoring the same crop passed
    alone (same pixels, same features, same score); scoring the whole
    frame -- what the pre-fix code did, since it read `o.payload` and
    ignored `o.roi` -- must NOT equal either, because the annuli then fall
    mostly over background rather than over the face-like crop.
    """
    p = tmp_path / "m.npz"
    save_blend_model(_model(coef=np.linspace(-1, 1, len(FEATURE_NAMES))), p)
    detector = BlendDetector(weights_path=p)

    crop = _ring_crop()
    frame = np.full((480, 640, 3), 60, dtype=np.uint8)
    x0, y0 = 120, 90
    frame[y0:y0 + 224, x0:x0 + 224] = crop
    roi = (x0, y0, 224, 224)
    quality = Quality(inter_ocular_px=40.0, blur_var=120.0, yaw_deg=0.0,
                      pitch_deg=0.0, exposure=0.5, band="high")

    cropped_alone = Observation(t=0.0, payload=crop, roi=(0, 0, 224, 224),
                                quality=quality, source_id="a")
    embedded = Observation(t=0.0, payload=frame, roi=roi,
                           quality=quality, source_id="b")
    whole_frame = Observation(t=0.0, payload=frame, roi=(0, 0, 640, 480),
                              quality=quality, source_id="c")

    r_cropped = detector.score([cropped_alone])
    r_embedded = detector.score([embedded])
    r_whole = detector.score([whole_frame])

    assert not r_cropped.abstained and not r_embedded.abstained and not r_whole.abstained
    assert r_embedded.score == pytest.approx(r_cropped.score)
    assert r_whole.score != pytest.approx(r_embedded.score)


def _textured_face(size: int) -> np.ndarray:
    """A richer, noisier fixture than `_ring_crop`: fixed-kernel residual and
    Laplacian statistics are sensitive to fine texture in a way a smooth
    ring is not, so this is what actually exercises the scale-skew this
    module's fix addresses."""
    rng = np.random.default_rng(42)
    img = rng.integers(30, 220, (size, size, 3)).astype(np.uint8)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.circle(img, (size // 2, size // 2), int(size * 0.29), (200, 60, 60), max(2, size // 45))
    return cv2.GaussianBlur(img, (9, 9), 0)


def _embed_same_face_at(native_size: int, canonical: np.ndarray,
                        x: int = 50, y: int = 40) -> Observation:
    """The SAME face content as `canonical` (448x448), downsampled to
    `native_size` the way a lower-resolution capture would produce it, then
    embedded in its own frame with an ROI matching its native size. Two
    calls with different `native_size` are "the same face at two different
    ROI sizes" -- what BlendDetector.score must now treat equivalently."""
    content = cv2.resize(canonical, (native_size, native_size),
                         interpolation=cv2.INTER_AREA)
    frame = np.full((native_size + 200, native_size + 200, 3), 90, dtype=np.uint8)
    frame[y:y + native_size, x:x + native_size] = content
    quality = Quality(inter_ocular_px=40.0, blur_var=120.0, yaw_deg=0.0,
                      pitch_deg=0.0, exposure=0.5, band="high")
    return Observation(t=0.0, payload=frame, roi=(x, y, native_size, native_size),
                       quality=quality, source_id=f"native{native_size}")


def test_scores_the_same_face_at_different_roi_sizes(tmp_path: Path) -> None:
    """Pins the resize fix. Measured directly: with the resize, scores at
    ROI 224 vs 80 for the same face content are 0.50434 vs 0.50421 (delta
    0.00012); WITHOUT the resize (the pre-fix behaviour) the same pair
    scored 0.50434 vs 0.50606 (delta 0.00172) -- about 14x larger, and
    well outside the tolerance below. `_model()`'s default (coef=ones,
    scale=1e5) is used deliberately: the same config the module's other
    tests use to keep logits off the sigmoid's saturated tails."""
    p = tmp_path / "m.npz"
    save_blend_model(_model(), p)
    detector = BlendDetector(weights_path=p)

    canonical = _textured_face(448)
    small = detector.score([_embed_same_face_at(80, canonical)])
    large = detector.score([_embed_same_face_at(224, canonical)])

    assert not small.abstained and not large.abstained
    assert small.score == pytest.approx(large.score, abs=5e-4)


def test_the_224_path_is_unchanged_by_the_resize(tmp_path: Path) -> None:
    """Regression guard: resizing a crop that is ALREADY 224x224 must be a
    no-op (cv2.resize at a 1:1 ratio returns the input unchanged -- verified
    directly: `cv2.resize(x, x.shape[:2], interpolation=cv2.INTER_AREA)` is
    bit-identical to `x` for a random array), so the already-correct case
    from before this fix must score exactly what computing the feature
    vector directly, with no crop/resize step at all, gives."""
    from dfd.detectors.blend import seam_features

    p = tmp_path / "m.npz"
    m = _model(coef=np.linspace(-1, 1, len(FEATURE_NAMES)))
    save_blend_model(m, p)
    detector = BlendDetector(weights_path=p)

    crop = _ring_crop()
    expected = m.predict_proba(seam_features(crop))

    direct = detector.score([Observation(
        t=0.0, payload=crop, roi=(0, 0, 224, 224),
        quality=Quality(inter_ocular_px=40.0, blur_var=120.0, yaw_deg=0.0,
                        pitch_deg=0.0, exposure=0.5, band="high"),
        source_id="s")])
    assert not direct.abstained
    assert direct.score == pytest.approx(expected)


def test_the_reported_version_is_the_version_of_the_weights_on_disk(
        tmp_path: Path) -> None:
    """A run record naming `0.1.0` for any weights file cannot be reproduced.

    `bench.runner` writes `detector.version` into `model_versions`, which is
    the field a reader uses to answer "which model produced this AUC". A
    hardcoded code version answers it wrongly the moment the weights are
    refitted, and refitting is the normal case for this detector — it is the
    one whose weights this project produces itself.
    """
    path = tmp_path / "blend.npz"
    save_blend_model(_model_versioned("2.0.0-fairface10k"), path)
    assert BlendDetector(weights_path=path).version == "2.0.0-fairface10k"


def test_refitting_the_weights_changes_the_reported_version(
        tmp_path: Path) -> None:
    """Whatever caching the lookup uses must not outlive the file it read."""
    path = tmp_path / "blend.npz"
    det = BlendDetector(weights_path=path)
    save_blend_model(_model_versioned("1.0.0"), path)
    assert det.version == "1.0.0"
    save_blend_model(_model_versioned("1.1.0"), path)
    assert det.version == "1.1.0"


def test_the_version_falls_back_to_the_code_version_when_weights_are_absent(
        tmp_path: Path) -> None:
    """An abstention still carries a version, and must not raise reading it."""
    det = BlendDetector(weights_path=tmp_path / "missing.npz")
    assert det.version == CODE_VERSION
    assert det.score([_obs()]).version == CODE_VERSION


def _model_versioned(version: str) -> BlendModel:
    n = len(FEATURE_NAMES)
    return BlendModel(mean=np.zeros(n), scale=np.full(n, 1e5),
                      coef=np.ones(n), intercept=0.0,
                      feature_names=FEATURE_NAMES, version=version)
