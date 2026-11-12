"""The service's composition root: where flags become a `ServiceConfig`.

This file exists because of PR #9. That bug lived entirely in this layer --
`ServiceConfig` defaulted slot D's weights to the frozen constants and passed
them over the top of `slot_d_pair()` -- and its own commit message recorded
why the tests missed it: "They exercised slot_d_pair() and default_registry()
and both were correct; nothing covered the service's config defaults, which
is the only place the two met."

Nothing covered them afterwards either. The calibration flag then repeated
the defect in a second place, so the layer gets a test file.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from dfd.service.__main__ import _build_parser, config_from_args


def _config(argv: list[str]):
    return config_from_args(_build_parser().parse_args(argv))


def test_the_calibration_default_follows_the_resolved_slot_d_pair(tmp_path):
    """Omitting --calibration must ask the registry, not a constant.

    `ops/dfd.service` passed the frozen path as a literal, so an instance
    running Phase 2's weights scored with the tuned model and calibrated
    with the frozen model's curve.
    """
    curve = tmp_path / "resolved.json"
    curve.write_text(json.dumps({"format_version": 1, "detectors": {}}))

    with mock.patch("dfd.service.__main__.slot_d_calibration",
                    return_value=curve) as resolver:
        config = _config([])

    resolver.assert_called_once()
    assert config.calibration_path == curve


def test_an_explicit_calibration_still_wins_over_the_resolver(tmp_path):
    curve = tmp_path / "explicit.json"
    curve.write_text(json.dumps({"format_version": 1, "detectors": {}}))

    with mock.patch("dfd.service.__main__.slot_d_calibration") as resolver:
        config = _config(["--calibration", str(curve)])

    resolver.assert_not_called()
    assert config.calibration_path == Path(str(curve))


def test_an_unresolvable_calibration_stays_none(tmp_path):
    """None is the safe value, and `build_service` already warns on it.

    It means slot D abstains with `uncalibrated_for_band`, which is the
    behaviour this whole change prefers over borrowing the frozen curve.
    """
    with mock.patch("dfd.service.__main__.slot_d_calibration",
                    return_value=None):
        assert _config([]).calibration_path is None


def test_slot_ds_weights_are_still_left_for_the_registry_to_resolve():
    """PR #9's fix, asserted where it actually lives.

    `config_from_args` must not name the weight paths; `None` is what lets
    `default_registry` call `slot_d_pair()`.
    """
    config = _config([])
    assert config.gend_weights is None
    assert config.clip_weights is None


def test_the_shipped_unit_does_not_hardcode_a_calibration_path():
    """The deployed artefact, not just the code path.

    `ops/dfd.service` is what systemd actually runs. A correct default in
    `config_from_args` is inert if the unit passes a literal over the top
    of it -- which is exactly how this defect reached the instance.
    """
    unit = (Path(__file__).resolve().parents[2] / "ops" / "dfd.service")
    # Comments are allowed to mention the flag -- the unit explains at
    # length why it does not pass one. What must not contain it is the
    # command systemd runs.
    directives = [ln for ln in unit.read_text().splitlines()
                  if not ln.lstrip().startswith("#")]
    assert "--calibration" not in "\n".join(directives)
