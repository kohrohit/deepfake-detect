"""The swap-corpus eval entry point, and the contamination rule it enforces.

`bench/swap_corpus_report.md` — the first leave-one-generator-out result this
project produced — was written by an uncommitted script, so the headline
number could not be reproduced from the repository. This module is that
script. The tests that matter most are the ones keeping the two facts that
make its numbers readable attached to them: the training-window offset, and
that these are classical compositing swaps rather than generator output.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bench import eval_swaps
from dfd.detectors.base import Registry
from dfd.faces import FaceBox
from dfd.types import RawScore


def _box() -> FaceBox:
    lms = np.array([[70.0, 80.0], [130.0, 80.0], [100.0, 110.0],
                    [76.0, 140.0], [124.0, 140.0]])
    return FaceBox(x=50, y=50, w=100, h=110, landmarks=lms, score=0.99)


def _detect_one(_frame):
    return [_box()]


class _MeanDetector:
    name = "mean_intensity"
    slot = "A"
    version = "test-1"
    modalities = frozenset()

    def score(self, obs):
        value = float(np.mean([np.mean(o.payload) for o in obs]) / 255.0)
        return RawScore(detector=self.name, version=self.version, score=value,
                        abstained=False, reason="ok")


def _registry() -> Registry:
    registry = Registry()
    registry.register(_MeanDetector())
    return registry


def _sessions(root: Path, n: int = 12) -> Path:
    """FairFace session folders, as `training/export_fairface.py` writes them."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        d = root / f"fairface-train-{i:06d}"
        d.mkdir(exist_ok=True)
        img = rng.integers(0, 255, (240, 240, 3), dtype=np.uint8)
        cv2.imwrite(str(d / "frame_00.jpg"), img)
        (d / "results.json").write_text(json.dumps(
            {"demographics": {"race": "White", "gender": "Male"}}))
    return root


def test_it_writes_both_reports(tmp_path: Path) -> None:
    md, js = tmp_path / "r.md", tmp_path / "r.json"
    code = eval_swaps.main(
        ["--root", str(_sessions(tmp_path / "s")), "--offset", "0",
         "--allow-training-window", "--out-md", str(md), "--out-json", str(js)],
        detect=_detect_one, registry=_registry())
    assert code == 0
    assert "mean_intensity" in md.read_text()
    assert json.loads(js.read_text())["corpus"]["offset"] == 0


def test_it_refuses_an_offset_inside_the_detectors_training_window(tmp_path):
    """`blend_seam` v0.2.0-fairface10k was fitted on FairFace sessions
    0-9,999. A corpus built below that offset evaluates it on its own
    training set, which has already produced one wrong headline (0.923).

    `build_swap_corpus` cannot check this — the window is a property of the
    weights, not of the corpus — but this entry point knows which weights
    are loaded, so here the documented rule becomes an enforced one.
    """
    class _Fitted(_MeanDetector):
        name = "blend_seam"
        version = "0.2.0-fairface10k"

    registry = Registry()
    registry.register(_Fitted())
    with pytest.raises(SystemExit) as excinfo:
        eval_swaps.main(
            ["--root", str(_sessions(tmp_path / "s")), "--offset", "9999",
             "--out-md", str(tmp_path / "r.md"),
             "--out-json", str(tmp_path / "r.json")],
            detect=_detect_one, registry=registry)
    assert "10000" in str(excinfo.value)
    assert "blend_seam" in str(excinfo.value)


def test_it_allows_an_offset_at_the_edge_of_the_training_window(tmp_path):
    """The rule is `offset >= 10000`, not `> 10000`. Session 10,000 is the
    first one the fit never saw; refusing it would be an off-by-one that
    costs a tenth of the usable corpus and reads as caution."""
    class _Fitted(_MeanDetector):
        name = "blend_seam"
        version = "0.2.0-fairface10k"

    registry = Registry()
    registry.register(_Fitted())
    code = eval_swaps.main(
        ["--root", str(_sessions(tmp_path / "s")), "--offset", "10000",
         "--out-md", str(tmp_path / "r.md"),
         "--out-json", str(tmp_path / "r.json")],
        detect=_detect_one, registry=registry)
    # No records survive an offset past the fixture, which is a corpus
    # error (exit 1), not the contamination refusal (SystemExit with a
    # message). Reaching either means the offset check passed.
    assert code == 1


def test_a_detector_with_no_declared_training_window_is_not_gated(tmp_path):
    """The gate must key on the WEIGHTS, not on the detector name. A future
    `blend_seam` refitted elsewhere carries a different version string and
    must not inherit this window."""
    class _Refitted(_MeanDetector):
        name = "blend_seam"
        version = "0.3.0-somewhere-else"

    registry = Registry()
    registry.register(_Refitted())
    code = eval_swaps.main(
        ["--root", str(_sessions(tmp_path / "s")), "--offset", "0",
         "--out-md", str(tmp_path / "r.md"),
         "--out-json", str(tmp_path / "r.json")],
        detect=_detect_one, registry=registry)
    assert code == 0


def test_the_report_carries_what_this_corpus_cannot_show(tmp_path: Path) -> None:
    """These are classical compositing swaps. A number from them must never
    travel as though it showed a detector beats FSGAN or InSwapper."""
    md = tmp_path / "r.md"
    eval_swaps.main(
        ["--root", str(_sessions(tmp_path / "s")), "--offset", "0",
         "--allow-training-window", "--out-md", str(md),
         "--out-json", str(tmp_path / "r.json")],
        detect=_detect_one, registry=_registry())
    text = md.read_text().lower()
    assert "classical" in text
    assert "not generator output" in text
