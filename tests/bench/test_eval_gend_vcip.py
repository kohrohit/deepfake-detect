"""Slot D against the traffic it is for. Evaluation only; this fits nothing."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from bench.eval_gend_vcip import per_band, session_scores
from dfd.detectors.clip_vit import CLIPVisualTower
from dfd.detectors.gend import GendHead


def test_this_module_fits_nothing(monkeypatch):
    # The guard that matters more than any number this module produces: a
    # capture-corpus fit is the one thing spec §6.4 forbids outright, and it
    # is forbidden for a reason no measurement can overturn. Importing a
    # fitter here would be the first step toward one.
    import bench.eval_gend_vcip as m

    assert not hasattr(m, "fit_head")
    assert not hasattr(m, "fit_gend")
    source = __import__("inspect").getsource(m)
    assert "LogisticRegression" not in source
    assert "training.fit_gend" not in source


def test_per_band_table_reports_its_denominators():
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    labels = np.array([0, 1, 0, 1, 0, 1])
    bands = np.array(["reject", "reject", "medium", "medium", "low", "low"])
    table = per_band(scores, labels, bands)
    assert set(table) == {"reject", "medium", "low"}
    for band, row in table.items():
        assert set(row) >= {"n", "n_genuine", "n_swapped", "auc",
                            "tpr_at_zero_fpr"}
        assert row["n"] == row["n_genuine"] + row["n_swapped"]


def test_a_band_with_one_class_reports_no_auc_rather_than_a_number():
    # `high` held 14 genuine frames and no swapped one when slot C was
    # measured on this corpus. An AUC there would be invented, and a
    # fabricated 1.0 in a per-band table is exactly the kind of number that
    # gets quoted later without its denominator.
    scores = np.array([0.1, 0.2, 0.3, 0.9])
    labels = np.array([0, 0, 0, 1])
    bands = np.array(["high", "high", "high", "medium"])
    table = per_band(scores, labels, bands)
    assert table["high"]["auc"] is None
    assert table["high"]["n_swapped"] == 0
    assert table["medium"]["auc"] is None  # one class on that side too


def test_sessions_are_the_unit_not_frames():
    # Frames of one session share the subject, the camera, the lighting and
    # the encode. Scoring them as independent rows is how 506 frames read as
    # 506 observations when they carry about 106.
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    labels = np.array([1, 1, 0, 0])
    sessions = np.array(["s1", "s1", "s2", "s2"])
    s, y, ids = session_scores(scores, labels, sessions)
    assert list(ids) == ["s1", "s2"]
    assert len(s) == len(y) == 2


def test_a_session_with_conflicting_frame_labels_is_refused():
    # A session is genuine or swapped; frames of it disagreeing means the
    # corpus loader mislabelled something, and averaging that away would
    # produce a plausible number from a broken input.
    with pytest.raises(ValueError, match="conflicting labels"):
        session_scores(np.array([0.9, 0.1]), np.array([1, 0]),
                       np.array(["s1", "s1"]))


@dataclass
class _CapturePair:
    root: Path
    encoder: Path
    head: Path
    out: Path


def _session(root, name, *, swapped, n=2, size=(240, 320), quality=None):
    """One capture session, in `tests/test_fit_vcip.py`'s own shape.

    `quality` writes the swapped half through an explicit JPEG quality so the
    two halves land on different quantisation tables — which is the corpus
    property the encoder control exists to remove, reproduced here so the
    control has something to find.
    """
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    frames = []
    for i in range(n):
        img = rng.integers(60, 200, (*size, 3), dtype=np.uint8)
        # A face-shaped blob the stub detector is pointed at.
        cv2.ellipse(img, (size[1] // 2, size[0] // 2),
                    (size[1] // 6, size[0] // 4), 0, 0, 360,
                    (200, 180, 170), -1)
        params = ([int(cv2.IMWRITE_JPEG_QUALITY), quality]
                  if quality is not None else [])
        cv2.imwrite(str(d / f"frame_{i:02d}.jpg"), img, params)
        frames.append({"index": i, "swapped": swapped,
                       "saved_as": f"captures/{name}/frame_{i:02d}.jpg"})
    (d / "results.json").write_text(json.dumps(
        {"session_id": name, "swapped": swapped, "frames": frames}))
    return d


@pytest.fixture
def capture_pair_fixture(tmp_path):
    root = tmp_path / "captures"
    for i in range(3):
        _session(root, f"genuine_{i}", swapped=False, quality=None)
        _session(root, f"swapped_{i}", swapped=True, quality=95)
    encoder = tmp_path / "clip.pt"
    torch.save(CLIPVisualTower.tiny().state_dict(), encoder)
    head = tmp_path / "gend.pt"
    h = GendHead(in_dim=8)
    h.set_version("0.0.1-test")
    torch.save(h.state_dict(), head)
    return _CapturePair(root=root, encoder=encoder, head=head,
                        out=tmp_path / "out.json")


def test_the_encoder_shortcut_control_runs_and_is_reported(capture_pair_fixture):
    # 99.2% of this corpus is labelled by one JPEG header byte: the swap
    # pipeline re-encodes at OpenCV quality 95 and the capture path never
    # re-encodes at all. The control re-encodes the GENUINE half alone, at
    # the quality that reproduces the swapped half's luma table EXACTLY --
    # `bench.vcip_controls` finds it by search and refuses if no quality
    # matches or if more than one table survives. Both numbers are reported;
    # the controlled one is the one to quote.
    from bench.eval_gend_vcip import main

    rc = main(["--captures", str(capture_pair_fixture.root),
               "--encoder", str(capture_pair_fixture.encoder),
               "--weights", str(capture_pair_fixture.head),
               "--out", str(capture_pair_fixture.out)],
              encoder_factory=CLIPVisualTower.tiny,
              head_factory=lambda: GendHead(in_dim=8))
    assert rc == 0
    report = json.loads(capture_pair_fixture.out.read_text())
    assert "uncontrolled" in report and "encoder_matched" in report
    assert report["encoder_matched"]["quality"] is not None


def test_the_session_auc_carries_an_interval():
    """0.7594 against a 0.75 floor is a margin of 0.0094 on 125 sessions.

    Without an interval, "clears the evidence floor" and "cannot be
    distinguished from the evidence floor" are the same printed number. The
    resampling unit is the SESSION, because frames within one are not
    independent -- `kish_ess_frames_by_session` is 106.9 over 509 frames.
    """
    from bench.eval_gend_vcip import _block

    rng = np.random.default_rng(0)
    scores, labels, sessions, bands = [], [], [], []
    for i in range(40):
        label = i % 2
        for _ in range(4):
            # Deliberately overlapping: a ceiling AUC of 1.0 would make the
            # interval degenerate and prove nothing about it.
            scores.append(float(rng.normal(0.55 if label else 0.45, 0.5)))
            labels.append(label)
            sessions.append(f"s{i}")
            bands.append("medium")
    block = _block(scores, labels, sessions, bands, {})
    lo, hi = block["session_auc_ci95"]
    assert block["session_auc"] < 1.0, "fixture is at ceiling"
    assert lo <= block["session_auc"] <= hi
    assert hi - lo > 0.0, "a zero-width interval is not an interval"
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0
    assert "sessions ARE the resampling unit" in block["session_auc_ci95_note"]


def test_the_block_keeps_per_session_scores_for_pairing():
    # Phase 2's whole question is whether tuning moved the capture-corpus
    # number. Both arms score the SAME sessions, so the difference can be
    # taken on those sessions -- but only if the scores survive the report.
    from bench.eval_gend_vcip import _block

    rng = np.random.default_rng(1)
    scores, labels, sessions, bands = [], [], [], []
    for i in range(10):
        for _ in range(3):
            scores.append(float(rng.random()))
            labels.append(i % 2)
            sessions.append(f"s{i}")
            bands.append("medium")
    rows = _block(scores, labels, sessions, bands, {})["sessions"]
    assert len(rows) == 10
    assert {r["session_id"] for r in rows} == {f"s{i}" for i in range(10)}
    assert all(set(r) == {"session_id", "label", "score"} for r in rows)
