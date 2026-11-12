"""A fitted calibrator survives a round trip to disk, as inert data.

Without this the service has no way to carry calibration across a restart,
and `decide` falls back to `uncalibrated_for_band` — an abstention that looks
identical to "never fitted".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dfd.calibration import (
    Calibrator,
    load_calibrators,
    save_calibrators,
)
from dfd.types import RawScore


def _fitted(detector: str = "blend_seam", seed: int = 0) -> Calibrator:
    rng = np.random.default_rng(seed)
    n = 400
    labels = np.concatenate([np.zeros(n // 2, int), np.ones(n // 2, int)])
    scores = np.clip(rng.normal(0.35, 0.15, n) + labels * 0.3, 0.0, 1.0)
    # Bands ALTERNATE rather than following the label split: a band that
    # holds only one class is skipped by `fit`, and a "round trip" over two
    # skipped bands compares 0.0 against 0.0 and passes without testing
    # anything.
    bands = np.array(["high" if i % 2 else "medium" for i in range(n)])
    return Calibrator(detector).fit(scores, labels, bands)


def _raw(score: float, detector: str = "blend_seam") -> RawScore:
    return RawScore(detector=detector, version="v1", score=score,
                    abstained=False, reason="ok")


def test_a_loaded_calibrator_gives_the_same_llr_as_the_fitted_one(
        tmp_path: Path) -> None:
    fitted = _fitted()
    path = tmp_path / "cal.json"
    save_calibrators({"blend_seam": fitted}, path)
    loaded = load_calibrators(path)["blend_seam"]
    for band in ("high", "medium"):
        for score in (0.05, 0.4, 0.55, 0.95):
            assert loaded.to_evidence(_raw(score), band).llr == pytest.approx(
                fitted.to_evidence(_raw(score), band).llr, abs=1e-9)


def test_a_band_that_was_never_fitted_stays_uncalibrated_after_a_round_trip(
        tmp_path: Path) -> None:
    """Round-tripping must not invent a curve for a band that had none."""
    path = tmp_path / "cal.json"
    save_calibrators({"blend_seam": _fitted()}, path)
    loaded = load_calibrators(path)["blend_seam"]
    ev = loaded.to_evidence(_raw(0.9), "low")
    assert ev.abstained
    assert ev.reason == "uncalibrated_for_band"


def test_the_file_is_plain_json_that_executes_nothing(tmp_path: Path) -> None:
    """Weight files are the artefact an attacker swaps; pickle executes."""
    path = tmp_path / "cal.json"
    save_calibrators({"blend_seam": _fitted()}, path)
    data = json.loads(path.read_text())
    assert data["format_version"] == 1
    band = data["detectors"]["blend_seam"]["bands"]["high"]
    assert {"coef", "intercept", "prior"} <= set(band)


def test_every_detector_round_trips_not_only_the_first(tmp_path: Path) -> None:
    path = tmp_path / "cal.json"
    save_calibrators({"a": _fitted("a", 1), "b": _fitted("b", 2)}, path)
    loaded = load_calibrators(path)
    assert sorted(loaded) == ["a", "b"]
    assert loaded["a"].detector == "a"
    assert loaded["b"].to_evidence(_raw(0.8, "b"), "high").llr != 0.0


def test_the_llr_clip_survives_the_round_trip(tmp_path: Path) -> None:
    """max_abs_llr is what stops one saturated model dominating the fusion."""
    path = tmp_path / "cal.json"
    cal = Calibrator("blend_seam", max_abs_llr=0.5)
    rng = np.random.default_rng(3)
    labels = np.concatenate([np.zeros(200, int), np.ones(200, int)])
    scores = np.concatenate([rng.normal(0.1, 0.02, 200),
                             rng.normal(0.9, 0.02, 200)])
    cal.fit(scores, labels, np.array(["high"] * 400))
    save_calibrators({"blend_seam": cal}, path)
    loaded = load_calibrators(path)["blend_seam"]
    assert loaded.max_abs_llr == 0.5
    assert abs(loaded.to_evidence(_raw(0.99), "high").llr) <= 0.5


def test_a_file_from_a_future_format_is_refused_rather_than_guessed(
        tmp_path: Path) -> None:
    path = tmp_path / "cal.json"
    path.write_text(json.dumps({"format_version": 99, "detectors": {}}))
    with pytest.raises(ValueError, match="format_version"):
        load_calibrators(path)
