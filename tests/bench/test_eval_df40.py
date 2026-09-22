"""The DF40 eval entry point runs the benchmark and says what it waived.

Every number this script produces is guard-waived — the corpus carries one
unlabelled compression level and one unknown generator — so the tests that
matter most here are the ones that keep that fact attached to the number.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from bench import eval_df40
from dfd.detectors.base import Registry
from dfd.faces import FaceBox
from dfd.types import RawScore


def _box() -> FaceBox:
    lms = np.array([[30.0, 40.0], [60.0, 40.0], [45.0, 55.0],
                    [33.0, 70.0], [57.0, 70.0]])
    return FaceBox(x=20, y=25, w=50, h=60, landmarks=lms, score=0.99)


def _detect_one(_frame):
    return [_box()]


class _MeanDetector:
    """Scores the crop's mean intensity, so real and fake fixtures separate."""
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


def _corpus(root: Path, n: int = 4) -> None:
    for sub, low in (("fake", 160), ("real", 0)):
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(1 if sub == "fake" else 2)
        for i in range(n):
            img = rng.integers(low, low + 90, (120, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(d / f"{sub[0]}{i}.png"), img)


def test_it_writes_both_reports(tmp_path: Path) -> None:
    _corpus(tmp_path / "corpus")
    md, js = tmp_path / "r.md", tmp_path / "r.json"
    code = eval_df40.main(
        ["--root", str(tmp_path / "corpus"), "--out-md", str(md),
         "--out-json", str(js)], detect=_detect_one, registry=_registry())
    assert code == 0
    assert "mean_intensity" in md.read_text()
    assert json.loads(js.read_text())["detector_results"]["mean_intensity"]["n_samples"] == 8


def test_the_report_records_that_the_guards_were_waived(tmp_path: Path) -> None:
    """A guard-waived AUC that does not say so will be quoted as a clean one."""
    _corpus(tmp_path / "corpus")
    md, js = tmp_path / "r.md", tmp_path / "r.json"
    eval_df40.main(["--root", str(tmp_path / "corpus"), "--out-md", str(md),
                    "--out-json", str(js)],
                   detect=_detect_one, registry=_registry())
    text = md.read_text()
    assert "guards enforced: `False`" in text
    assert eval_df40.WAIVER_HEADING in text
    # The reasons, not merely the fact.
    assert "compression" in text
    assert "generator" in text
    assert json.loads(js.read_text())["guards_enforced"] is False


def test_the_report_records_what_the_loader_skipped(tmp_path: Path) -> None:
    _corpus(tmp_path / "corpus")
    md, js = tmp_path / "r.md", tmp_path / "r.json"
    # Fails detection on every other image, which cuts across both labels —
    # skipping by brightness would skip one label entirely and the run would
    # abort for that reason instead of exercising this one.
    calls = iter(range(100))
    eval_df40.main(["--root", str(tmp_path / "corpus"), "--out-md", str(md),
                    "--out-json", str(js)],
                   detect=lambda _f: [] if next(calls) % 2 else [_box()],
                   registry=_registry())
    loaded = json.loads(js.read_text())["corpus"]
    assert loaded["records"] == 4
    assert loaded["skipped"]["no_face"] == 4


def test_a_corpus_with_no_usable_records_fails_rather_than_reporting_nan(
        tmp_path: Path) -> None:
    _corpus(tmp_path / "corpus")
    md, js = tmp_path / "r.md", tmp_path / "r.json"
    code = eval_df40.main(["--root", str(tmp_path / "corpus"), "--out-md",
                           str(md), "--out-json", str(js)],
                          detect=lambda _f: [], registry=_registry())
    assert code == 1
    assert not md.exists()
    assert not js.exists()


def test_a_corpus_with_only_one_label_fails_rather_than_reporting_nan(
        tmp_path: Path) -> None:
    """AUC over one label is undefined; reporting it as nan invites a retry."""
    _corpus(tmp_path / "corpus")
    md, js = tmp_path / "r.md", tmp_path / "r.json"
    code = eval_df40.main(
        ["--root", str(tmp_path / "corpus"), "--out-md", str(md),
         "--out-json", str(js)],
        detect=lambda f: [_box()] if np.mean(f) > 100 else [],
        registry=_registry())
    assert code == 1
    assert not js.exists()


def test_the_limit_and_seed_reach_the_loader(tmp_path: Path) -> None:
    _corpus(tmp_path / "corpus", n=10)
    md, js = tmp_path / "r.md", tmp_path / "r.json"
    eval_df40.main(["--root", str(tmp_path / "corpus"), "--limit", "6",
                    "--seed", "3", "--out-md", str(md), "--out-json", str(js)],
                   detect=_detect_one, registry=_registry())
    loaded = json.loads(js.read_text())
    assert loaded["corpus"]["records"] == 6
    assert loaded["corpus"]["seed"] == 3
