"""The corpus controls, and the two ways a control silently stops being one.

`training/fit_vcip.py` reported a limits list asserting an encoder shortcut
"at AUC 0.873" and a re-encoding control that removed it. Neither had a
committed script; both were wrong. These tests pin the properties that make
the replacement a control rather than a paragraph:

1. the matched quality must reproduce the target table EXACTLY, because a
   near miss leaves a weaker shortcut and still reports as a control; and
2. the control must REFUSE when it cannot remove the shortcut, rather than
   report a number computed over a corpus that still carries it.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bench import vcip_controls
from dfd.faces import FaceBox


def _box(x: int, y: int, w: int, h: int, score: float) -> FaceBox:
    lms = np.array([[x + w * 0.3, y + h * 0.3], [x + w * 0.7, y + h * 0.3],
                    [x + w * 0.5, y + h * 0.5], [x + w * 0.35, y + h * 0.75],
                    [x + w * 0.65, y + h * 0.75]])
    return FaceBox(x=x, y=y, w=w, h=h, landmarks=lms, score=score)


def _detect_one(_frame):
    return [_box(20, 20, 120, 120, 0.9)]


def _frame(rng: np.random.Generator) -> np.ndarray:
    """A 200x200 frame with structure, so its residual is not degenerate."""
    base = rng.integers(40, 210, (200, 200, 3), dtype=np.uint8)
    return cv2.GaussianBlur(base, (5, 5), 0)


def _corpus(root: Path, *, sessions: int = 24, genuine_quality: int = 80,
            swapped_quality: int = 95, frames_per: int = 4) -> Path:
    """A capture corpus shaped like the real one, with a header shortcut.

    The two halves are written at different JPEG qualities, exactly as the
    real corpus's capture path and swap writer do, so the quantisation tables
    separate the labels before any pixel is read.
    """
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(sessions):
        label = i % 2
        quality = swapped_quality if label else genuine_quality
        session = root / f"2026090{i // 10}-{i:06d}"
        session.mkdir(exist_ok=True)
        meta = {"session_id": session.name, "swapped": label, "frames": []}
        for j in range(frames_per):
            name = f"frame_{j:02d}.jpg"
            cv2.imwrite(str(session / name), _frame(rng),
                        [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            meta["frames"].append({"index": j, "saved_as": name,
                                   "swapped": label})
        (session / "results.json").write_text(json.dumps(meta))
    return root


# --- the exactness rule ------------------------------------------------


def test_the_matched_quality_must_reproduce_the_whole_table_not_its_first_byte(
        tmp_path: Path) -> None:
    """A near miss must return None, not a quality.

    On the real corpus the first luma coefficient matches at qualities 93, 94
    and 95 and only 95 reproduces all 128 coefficients. A search that stopped
    at the first coefficient would hand back 93, re-encode the genuine half
    onto a table still differing from the fakes' in 63 places, and report the
    result as a shortcut-free number.
    """
    rng = np.random.default_rng(1)
    sample = _frame(rng)
    target = vcip_controls.quant_table(vcip_controls._encode(sample, 95))
    assert target is not None

    near = vcip_controls.quant_table(vcip_controls._encode(sample, 94))
    assert near is not None
    assert near[0][0] == target[0][0], (
        "this fixture no longer reproduces the near-miss condition the test "
        "exists for: pick qualities whose first coefficient agrees")
    assert near != target

    assert vcip_controls.find_matching_quality(
        sample, target, qualities=(93, 94)) is None, (
        "a quality whose FIRST coefficient matches was accepted as the match")
    assert vcip_controls.find_matching_quality(
        sample, target, qualities=(93, 94, 95)) == 95


def test_a_lossless_image_has_no_table_and_says_so_with_none(
        tmp_path: Path) -> None:
    """None, never an empty tuple.

    Two files carrying no tables must not compare equal on their tables —
    that comparison is the whole of `find_matching_quality`, and `() == ()`
    would make every PNG an exact match for every other.
    """
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), _frame(np.random.default_rng(2)))
    assert vcip_controls.quant_table(path) is None


# --- refusing rather than reporting ------------------------------------


def test_the_control_refuses_when_no_quality_reproduces_the_table(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No match must mean `measured: False`, not a number.

    The corpus still carries the shortcut in that case, so any AUC computed
    over it measures the JPEG header. Reporting one is the exact failure the
    replaced limits list made.
    """
    root = _corpus(tmp_path / "captures")
    monkeypatch.setattr(vcip_controls, "find_matching_quality",
                        lambda *a, **k: None)
    out = vcip_controls.matched_encoder(root, detect=_detect_one)
    assert out["measured"] is False
    assert "no encoder quality" in out["why"]
    assert "auc" not in out


def test_the_control_refuses_when_more_than_one_table_survives(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Surviving tables are counted, and more than one aborts the control.

    The re-encode is the only thing collapsing the halves onto one table. If
    it does not — a writer that varies its table by image size, say — the
    shortcut is still there and the number would be uninterpretable.
    """
    root = _corpus(tmp_path / "captures")
    # Return the genuine half's own quality, so re-encoding preserves rather
    # than replaces its table and both tables survive.
    monkeypatch.setattr(vcip_controls, "find_matching_quality",
                        lambda *a, **k: 80)
    out = vcip_controls.matched_encoder(root, detect=_detect_one)
    assert out["measured"] is False
    assert out["distinct_tables"] > 1
    assert "still present" in out["why"]
    assert "auc" not in out


def test_the_control_collapses_the_corpus_onto_one_table_when_it_runs(
        tmp_path: Path) -> None:
    """The successful path must leave exactly one table behind.

    This is the property that makes the number readable, so it is asserted
    rather than assumed: a control that reports `measured: True` while two
    tables remain has measured the header.
    """
    root = _corpus(tmp_path / "captures")
    out = vcip_controls.matched_encoder(root, detect=_detect_one)
    assert out["measured"] is True, out.get("why")
    assert out["distinct_tables"] == 1
    assert out["quality"] == 95
    assert out["auc"] is not None


# --- the shortcut itself ------------------------------------------------


def test_the_header_lookup_is_counted_over_the_majority_of_each_table(
        tmp_path: Path) -> None:
    """The shortcut is reported as accuracy, not only as AUC.

    On the real corpus the tables label 99.2% of frames with no pixels read.
    An AUC alone lets a reader file that under "a weak confound"; the
    accuracy of a one-byte lookup does not.
    """
    root = _corpus(tmp_path / "captures")
    out = vcip_controls.encoder_shortcut(root)
    assert out["measured"] is True
    assert out["distinct_tables"] == 2
    assert out["header_lookup_accuracy"] == pytest.approx(1.0), (
        "a corpus whose halves were written at different qualities must be "
        "perfectly separable from the header alone")
    assert sum(row["genuine"] + row["swapped"] for row in out["by_table"]) \
        == out["frames"]


def test_every_control_reports_both_the_frame_and_the_session_basis(
        tmp_path: Path) -> None:
    """Both bases travel together, because they disagree by a wide margin.

    On the real corpus the shortcut scores 0.985 per frame and 0.935 per
    session. A control that reported one basis would let the flattering one
    be quoted against the other.
    """
    root = _corpus(tmp_path / "captures")
    out = vcip_controls.encoder_shortcut(root)
    assert out["frame_auc"] is not None
    assert out["auc"] is not None
    assert out["frames"] > out["sessions"]


# --- preprocessing, varied one thing at a time --------------------------


def test_the_preprocessing_comparison_uses_the_shipped_face_rule(
        tmp_path: Path) -> None:
    """Both arms select the largest face, so only preprocessing differs.

    The numbers this module replaced varied the face rule and the crop at
    once: the aligned arm picked the most confident face and the native arm
    the largest, so their difference was not attributable to either.
    """
    rng = np.random.default_rng(3)
    frame = _frame(rng)
    seen: list[list[FaceBox]] = []

    def detect(_frame):
        boxes = [_box(150, 150, 40, 40, 0.99),      # small, most confident
                 _box(10, 10, 120, 120, 0.60)]      # large, least confident
        seen.append(boxes)
        return boxes

    from training.fit_vcip import select_face
    largest = _box(10, 10, 120, 120, 0.60)
    most_confident = _box(150, 150, 40, 40, 0.99)
    assert select_face(detect(frame)) is seen[-1][1], (
        "the shipped rule no longer picks the largest face; this test's "
        "premise is gone")

    net = vcip_controls.NPRStatsNet()
    native = vcip_controls._features(frame, detect=detect, net=net,
                                     aligned=False)
    assert native is not None
    # What the two candidate rules would each produce, computed directly.
    from_largest = vcip_controls._features(
        frame, detect=lambda _f: [largest], net=net, aligned=False)
    from_most_confident = vcip_controls._features(
        frame, detect=lambda _f: [most_confident], net=net, aligned=False)
    assert from_largest is not None and from_most_confident is not None
    assert not np.allclose(from_largest, from_most_confident), (
        "the two faces yield the same vector, so this frame cannot tell the "
        "rules apart")
    assert np.allclose(native, from_largest), (
        "the control scored a face the pipeline would not have scored")

    aligned = vcip_controls._features(frame, detect=detect, net=net,
                                      aligned=True)
    assert aligned is not None
    assert native.shape == aligned.shape
    assert not np.allclose(native, aligned), (
        "the two preprocessings produced the same vector, so the comparison "
        "they feed measures nothing")


def test_the_two_block_arms_are_handed_different_columns() -> None:
    """The arms must partition the vector: disjoint, and together complete.

    The two slices differ by one colon. Get it wrong and both arms are the
    same features under two names, and the ablation reports a finding —
    "the blocks perform alike" — instead of an error.
    """
    x = np.arange(4 * vcip_controls.NPRStatsNet.N_FEATURES, dtype=np.float64
                  ).reshape(4, vcip_controls.NPRStatsNet.N_FEATURES)
    phase, spectral = vcip_controls.block_views(x)
    split = vcip_controls.NPRStatsNet.N_PHASE_FEATURES
    assert np.array_equal(phase, x[:, :split])
    assert np.array_equal(spectral, x[:, split:])
    assert phase.shape[1] + spectral.shape[1] == x.shape[1]
    assert not np.array_equal(phase[:, :1], spectral[:, :1]), (
        "the two arms start at the same column, so they are not a partition")


def test_the_block_arms_grade_the_same_frames(tmp_path: Path) -> None:
    """Phase-only and spectral-only come off ONE feature matrix.

    Re-extracting for each block would let the two arms disagree about which
    frames were usable, which is how an ablation stops being one.
    """
    root = _corpus(tmp_path / "captures")
    out = vcip_controls.preprocessing_and_blocks(root, detect=_detect_one)
    full = out["native"]
    phase = out["native_phase_block_only"]
    spectral = out["native_spectral_block_only"]
    assert phase["n_features"] + spectral["n_features"] == \
        vcip_controls.NPRStatsNet.N_FEATURES
    assert phase["frames"] == full["frames"] == spectral["frames"]


# --- the per-band table under the quality floor -------------------------


def test_a_band_holding_one_class_reports_no_auc_rather_than_a_number(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`high` holds 14 genuine frames and no swap on the real corpus.

    An AUC over one class is not a weak number, it is not a number; and the
    per-band table is what `min_quality_band = "reject"` is argued from, so
    a placeholder there would be read as evidence.
    """
    import training.fit_vcip as fit_vcip

    root = _corpus(tmp_path / "captures", sessions=12)
    rng = np.random.default_rng(7)

    def fake_features_for(path, *, detect, net):
        label = json.loads((path.parent / "results.json").read_text())["swapped"]
        # Genuine frames alone reach the `high` band; both classes land in
        # `medium`, so one band is single-class and the other is not.
        band = "high" if label == 0 and path.name.endswith("00.jpg") else "medium"
        return rng.normal(size=vcip_controls.NPRStatsNet.N_FEATURES) + label, band

    monkeypatch.setattr(fit_vcip, "features_for", fake_features_for)
    out = vcip_controls.per_band(root, detect=_detect_one)
    assert out["measured"] is True
    assert out["high"]["swapped"] == 0
    assert out["high"]["auc"] is None
    assert out["high"]["caught_at_fpr_0.0"] is None
    assert out["medium"]["auc"] is not None


def test_main_writes_a_report_with_every_control_in_it(
        tmp_path: Path) -> None:
    root = _corpus(tmp_path / "captures")
    out = tmp_path / "controls.json"
    assert vcip_controls.main(["--captures", str(root), "--out", str(out)],
                              detect=_detect_one) == 0
    report = json.loads(out.read_text())
    assert set(report) >= {"encoder_shortcut", "matched_encoder",
                           "preprocessing_and_blocks"}
    assert report["captures"] == str(root)


def test_main_refuses_a_missing_capture_directory(tmp_path: Path) -> None:
    assert vcip_controls.main(
        ["--captures", str(tmp_path / "nope"),
         "--out", str(tmp_path / "out.json")], detect=_detect_one) == 1
