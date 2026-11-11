"""The calibration fitter: fit on validation, never on what you report."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from dfd.calibration import Calibrator, load_calibrators
from dfd.detectors.base import Registry
from dfd.faces import FaceBox
from dfd.types import RawScore
from training.fit_calibration import band_coverage, load_in_domain_bands, main, split_by_source


class _MeanDetector:
    """Separates the fixtures by brightness, so a curve is fittable."""
    name = "mean_intensity"
    slot = "A"
    version = "test-1"
    modalities = frozenset()

    def score(self, obs):
        v = float(np.mean([np.mean(o.payload) for o in obs]) / 255.0)
        return RawScore(detector=self.name, version=self.version, score=v,
                        abstained=False, reason="ok")


class _AbstainingDetector:
    name = "always_abstains"
    slot = "C"
    version = "test-1"
    modalities = frozenset()

    def score(self, obs):
        return RawScore(detector=self.name, version=self.version, score=None,
                        abstained=True, reason="weights_absent")


def _registry() -> Registry:
    r = Registry()
    r.register(_MeanDetector())
    r.register(_AbstainingDetector())
    return r


def _box() -> FaceBox:
    lms = np.array([[30.0, 40.0], [60.0, 40.0], [45.0, 55.0],
                    [33.0, 70.0], [57.0, 70.0]])
    return FaceBox(x=20, y=25, w=50, h=60, landmarks=lms, score=0.99)


def _corpus(root: Path, n: int = 40, per_source: int = 4) -> Path:
    """A corpus with SOURCE structure, four frames to a source.

    The names carry a frame index (`f000_2.png`) because `corpora.df40`
    groups frames of one filename family into one source (2026-09-23), and
    `split_by_source` assigns a whole source to one side. Flat names
    (`f000.png`) all land in that loader's `unnumbered` bucket, which leaves
    exactly two sources — one per label — and a split that is single-label
    on both sides, so nothing is fittable and every curve is skipped.
    """
    for sub, low in (("fake", 150), ("real", 0)):
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(1 if sub == "fake" else 2)
        for i in range(n):
            img = rng.integers(low, low + 100, (300, 260, 3), dtype=np.uint8)
            cv2.imwrite(str(d / f"{sub[0]}{i // per_source:03d}_{i % per_source}.png"),
                        img)
    return root


def test_the_split_puts_no_source_on_both_sides() -> None:
    records = [{"source_id": f"s{i % 7}", "label": i % 2} for i in range(40)]
    fit, holdout = split_by_source(records, holdout_fraction=0.4, seed=0)
    assert {r["source_id"] for r in fit} & {r["source_id"] for r in holdout} == set()
    assert len(fit) + len(holdout) == len(records)


def test_the_split_is_deterministic_under_a_seed() -> None:
    records = [{"source_id": f"s{i}", "label": i % 2} for i in range(40)]
    a, _ = split_by_source(records, holdout_fraction=0.3, seed=5)
    b, _ = split_by_source(records, holdout_fraction=0.3, seed=5)
    c, _ = split_by_source(records, holdout_fraction=0.3, seed=6)
    ids = lambda rs: sorted(r["source_id"] for r in rs)  # noqa: E731
    assert ids(a) == ids(b)
    assert ids(a) != ids(c)


def test_it_writes_curves_that_load_back(tmp_path: Path) -> None:
    out = tmp_path / "cal.json"
    code = main(["--corpus", str(_corpus(tmp_path / "c")), "--out", str(out),
                 "--report", str(tmp_path / "r.json")],
                registry=_registry(), detect=lambda f: [_box()])
    assert code == 0
    assert "mean_intensity" in load_calibrators(out)


def test_a_detector_that_only_abstains_gets_no_curve(tmp_path: Path) -> None:
    """A curve fitted on nothing would be a fabricated licence to decide."""
    out = tmp_path / "cal.json"
    main(["--corpus", str(_corpus(tmp_path / "c")), "--out", str(out),
          "--report", str(tmp_path / "r.json")],
         registry=_registry(), detect=lambda f: [_box()])
    assert "always_abstains" not in load_calibrators(out)


def test_the_report_records_the_holdout_auc_not_the_fit_auc(
        tmp_path: Path) -> None:
    report = tmp_path / "r.json"
    main(["--corpus", str(_corpus(tmp_path / "c")), "--out",
          str(tmp_path / "cal.json"), "--report", str(report)],
         registry=_registry(), detect=lambda f: [_box()])
    data = json.loads(report.read_text())
    assert data["split"]["holdout_records"] > 0
    assert data["detectors"]["mean_intensity"]["holdout_auc"] is not None
    assert data["detectors"]["mean_intensity"]["fitted_on"] == "fit split"


def test_a_corpus_with_one_label_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "c"
    (root / "fake").mkdir(parents=True)
    (root / "real").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(30):
        cv2.imwrite(str(root / "fake" / f"f{i}.png"),
                    rng.integers(0, 255, (300, 260, 3), dtype=np.uint8))
    with pytest.raises(SystemExit) as exc:
        main(["--corpus", str(root), "--out", str(tmp_path / "cal.json"),
              "--report", str(tmp_path / "r.json")],
             registry=_registry(), detect=lambda f: [_box()])
    assert exc.value.code == 1


def test_calibration_can_be_fitted_from_the_swap_corpus(
        tmp_path, tiny_fairface_root, stub_detect):
    # The point: the one corpus this project has ruled unable to measure
    # anything must not be the only thing that can produce a calibration
    # curve for a detector allowed to decide.
    # `registry=_registry()` rather than the default, which every other test
    # in this file also does -- and here it is load-bearing, not stylistic.
    # `default_registry()` now carries slot D, whose encoder is 1.2 GB. Where
    # that file exists these two tests spend four minutes running ViT-L/14;
    # where it does not (CI) slot D abstains instantly. A test that takes
    # four minutes on one box and two seconds on another is exercising two
    # different paths, and neither of them is this test's subject.
    rc = main(["--corpus-kind", "swaps", "--root", str(tiny_fairface_root),
               "--offset", "0", "--limit", "8",
               "--out", str(tmp_path / "cal.json"),
               "--report", str(tmp_path / "rep.json")],
              registry=_registry(), detect=stub_detect)
    assert rc == 0
    assert json.loads((tmp_path / "cal.json").read_text())


def test_the_report_records_which_corpus_the_curve_came_from(
        tmp_path, tiny_fairface_root, stub_detect):
    # Two curves for one detector, fitted on two corpora, are two different
    # claims about what a score means. A report that does not say which is
    # which cannot be read a month later.
    main(["--corpus-kind", "swaps", "--root", str(tiny_fairface_root),
          "--offset", "0", "--limit", "8",
          "--out", str(tmp_path / "cal.json"),
          "--report", str(tmp_path / "rep.json")],
         registry=_registry(), detect=stub_detect)
    assert json.loads((tmp_path / "rep.json").read_text())["corpus_kind"] == "swaps"


def test_an_unknown_corpus_kind_is_refused_by_the_parser():
    with pytest.raises(SystemExit):
        main(["--corpus-kind", "df40-but-ok", "--corpus", "x"])


def test_a_band_seen_but_never_fitted_is_not_reported_as_covered():
    # `Calibrator.fit` refuses a band under MIN_FIT_SAMPLES, so the bands a
    # curve HAS and the bands its fit split SAW are two different lists. The
    # EC2 run of 2026-11-11 saw `low` on 13 rows and fitted nothing there;
    # a report whose single `bands` field said "low" claimed a curve that
    # does not exist, on the one field Task 9 Step 4 tells a reader to read.
    cal = Calibrator("d").fit(
        scores=[0.1] * 10 + [0.9] * 10 + [0.2, 0.8],
        labels=[0] * 10 + [1] * 10 + [0, 1],
        bands=["medium"] * 20 + ["low", "low"])
    assert cal.fitted_bands == ["medium"]


def test_coverage_names_the_in_domain_bands_the_curve_will_abstain_on():
    # The failure this guards against is a curve that looks fitted and is
    # silent in production: an uncovered band returns llr 0.0, and a reader
    # of the report has no way to see that unless the report counts the
    # capture frames sitting in those bands.
    cov = band_coverage(["medium", "reject"],
                        {"reject": 232, "medium": 238, "low": 12, "high": 27})
    assert cov["uncovered_bands"] == ["high", "low"]
    assert cov["capture_frames_covered"] == 470
    assert cov["capture_frames"] == 509
    assert cov["fraction_covered"] == pytest.approx(470 / 509)


def test_coverage_of_nothing_is_said_rather_than_divided_by():
    cov = band_coverage([], {})
    assert cov["fraction_covered"] is None
    assert cov["uncovered_bands"] == []


def test_the_in_domain_band_table_comes_from_the_controlled_half(tmp_path):
    # `uncontrolled` is the half that one JPEG header byte labels. Reading
    # the band histogram from it would describe a corpus nobody may quote.
    p = tmp_path / "gend_vcip.json"
    p.write_text(json.dumps({
        "encoder_matched": {"per_band_frames": {"medium": {"n": 238},
                                                "reject": {"n": 232}}},
        "uncontrolled": {"per_band_frames": {"medium": {"n": 1}}}}))
    assert load_in_domain_bands(p) == {"medium": 238, "reject": 232}
