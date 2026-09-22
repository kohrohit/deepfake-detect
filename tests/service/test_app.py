"""The composition root: what gets wired, and what it says when it cannot."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from dfd.calibration import Calibrator, save_calibrators
from dfd.service.app import ServiceConfig, build_service


def _config(tmp_path: Path, **kw) -> ServiceConfig:
    kw.setdefault("evidence_path", tmp_path / "evidence.json")
    return ServiceConfig(
        host="127.0.0.1", port=0,
        db_path=tmp_path / "db.sqlite3",
        inbox=tmp_path / "inbox",
        workdir=tmp_path / "work",
        **kw)


def test_building_creates_the_inbox_and_workdir(tmp_path: Path) -> None:
    """The operator is told to drop files in a directory that must exist."""
    service = build_service(_config(tmp_path))
    assert (tmp_path / "inbox").is_dir()
    assert (tmp_path / "work").is_dir()
    service.close()


def test_the_registry_carries_the_three_declared_slots(tmp_path: Path) -> None:
    service = build_service(_config(tmp_path))
    assert sorted(service.registry.names()) == ["blend_seam", "effnet_b4", "npr"]
    service.close()


def test_a_calibration_file_is_loaded_when_present(tmp_path: Path) -> None:
    # With an evidence card that clears the floor — the gate is tested on its
    # own below, and without a passing card this would only ever assert the
    # gate, never the loading.
    service = build_service(_config(
        tmp_path,
        calibration_path=_calibration_for(tmp_path, "blend_seam"),
        evidence_path=_card(tmp_path, {
            "blend_seam": {"auc": 0.93, "corpus": "df40",
                           "trained_on": "fairface", "note": ""}})))
    assert "blend_seam" in service.calibrators
    service.close()


def test_a_missing_calibration_file_is_not_silently_treated_as_calibrated(
        tmp_path: Path) -> None:
    """No file means no curves, and every detector abstains as uncalibrated.

    The failure this guards is the opposite: quietly proceeding with an empty
    mapping AND logging nothing, so a mistyped path looks exactly like a
    deployment that was never calibrated.
    """
    service = build_service(_config(tmp_path,
                                    calibration_path=tmp_path / "nope.json"))
    assert service.calibrators == {}
    assert service.warnings
    assert any("calibration" in w for w in service.warnings)
    service.close()


def test_a_malformed_calibration_file_fails_the_startup(tmp_path: Path) -> None:
    """Starting anyway would decide every sample with silent zero evidence."""
    bad = tmp_path / "cal.json"
    bad.write_text('{"format_version": 99, "detectors": {}}')
    with pytest.raises(ValueError, match="format_version"):
        build_service(_config(tmp_path, calibration_path=bad))


def test_starting_serves_health_and_stopping_releases_the_port(
        tmp_path: Path) -> None:
    service = build_service(_config(tmp_path))
    service.start()
    try:
        url = f"http://127.0.0.1:{service.port}/health"
        with urllib.request.urlopen(url, timeout=10) as r:
            assert json.loads(r.read())["status"] == "ok"
    finally:
        service.stop()
    with pytest.raises(OSError):
        urllib.request.urlopen(f"http://127.0.0.1:{service.port}/health",
                               timeout=2)


def test_a_file_dropped_in_the_inbox_is_picked_up_and_recorded(
        tmp_path: Path) -> None:
    """End to end through the real worker thread: drop a file, get a row."""
    service = build_service(_config(tmp_path, poll_interval=0.05))
    service.start()
    try:
        (tmp_path / "inbox" / "note.txt").write_text("not an image")
        deadline = 20.0
        while deadline > 0:
            rows = service.store.recent()
            if rows and rows[0].status in ("done", "failed"):
                break
            import time as _t
            _t.sleep(0.1)
            deadline -= 0.1
        rows = service.store.recent()
        assert rows, "the watcher never picked the file up"
        # A .txt is refused by the pipeline, and a refusal is a recorded
        # outcome rather than a crash or a silent drop.
        assert rows[0].status == "failed"
        assert rows[0].filename == "note.txt"
    finally:
        service.stop()


def test_stopping_a_service_that_was_never_started_is_harmless(
        tmp_path: Path) -> None:
    build_service(_config(tmp_path)).stop()


def _card(tmp_path: Path, detectors: dict) -> Path:
    path = tmp_path / "card.json"
    path.write_text(json.dumps({
        "format_version": 1, "generated_at": "2026-09-23T00:00:00Z",
        "auc_floor": 0.75, "detectors": detectors}))
    return path


def _calibration_for(tmp_path: Path, *names: str) -> Path:
    rng = np.random.default_rng(0)
    labels = np.array([i % 2 for i in range(200)])
    scores = np.clip(rng.normal(0.4, 0.1, 200) + labels * 0.25, 0, 1)
    cals = {n: Calibrator(n).fit(scores, labels, np.array(["high"] * 200))
            for n in names}
    path = tmp_path / "cal.json"
    save_calibrators(cals, path)
    return path


def test_a_detector_below_the_evidence_floor_is_not_calibrated(
        tmp_path: Path) -> None:
    """The gate. A calibrated detector moves the verdict; this one must not.

    Dropping the curve rather than the detector is deliberate: the raw score
    still reaches the audit record, so the measurement that would one day
    lift it above the floor keeps accumulating.
    """
    service = build_service(_config(
        tmp_path,
        calibration_path=_calibration_for(tmp_path, "blend_seam"),
        evidence_path=_card(tmp_path, {
            "blend_seam": {"auc": 0.289, "corpus": "df40", "note": ""}})))
    assert service.calibrators == {}
    assert any("0.289" in w or "below" in w for w in service.warnings)
    service.close()


def test_a_detector_above_the_evidence_floor_keeps_its_calibration(
        tmp_path: Path) -> None:
    service = build_service(_config(
        tmp_path,
        calibration_path=_calibration_for(tmp_path, "blend_seam"),
        evidence_path=_card(tmp_path, {
            "blend_seam": {"auc": 0.93, "corpus": "df40",
                           "trained_on": "fairface", "note": ""}})))
    assert "blend_seam" in service.calibrators
    service.close()


def test_a_missing_evidence_card_leaves_nothing_calibrated(
        tmp_path: Path) -> None:
    """No card means no measured performance, which must gate everything out."""
    service = build_service(_config(
        tmp_path,
        calibration_path=_calibration_for(tmp_path, "blend_seam"),
        evidence_path=tmp_path / "absent.json"))
    assert service.calibrators == {}
    assert any("evidence card" in w for w in service.warnings)
    service.close()
