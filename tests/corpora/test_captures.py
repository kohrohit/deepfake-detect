import json
import logging

import pytest
from corpora.captures import load_capture_sessions, missed_attacks


def _session(root, name, swapped, approved, verdict="LIVE"):
    d = root / name
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps({
        "session_id": name, "swapped": swapped, "frame_count": 1,
        "scan": {"verdict": verdict},
        "decision": {"approved": approved, "reason": "approved"},
    }))


def test_loads_sessions(tmp_path):
    _session(tmp_path, "s1", False, True)
    out = load_capture_sessions(tmp_path)
    assert len(out) == 1 and out[0].session_id == "s1"


def test_label_is_derived_from_the_swapped_flag(tmp_path):
    _session(tmp_path, "s1", True, True)
    _session(tmp_path, "s2", False, True)
    by_id = {s.session_id: s for s in load_capture_sessions(tmp_path)}
    assert by_id["s1"].label == 1
    assert by_id["s2"].label == 0


def test_missed_attacks_finds_swapped_sessions_that_were_approved(tmp_path):
    """The five sessions in spec §1.1 that constitute the actual fraud."""
    _session(tmp_path, "bad", True, True)
    _session(tmp_path, "caught", True, False)
    _session(tmp_path, "genuine", False, True)
    missed = missed_attacks(load_capture_sessions(tmp_path))
    assert [s.session_id for s in missed] == ["bad"]


def test_malformed_session_is_skipped_not_fatal(tmp_path):
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "results.json").write_text("{not json")
    _session(tmp_path, "ok", False, True)
    assert len(load_capture_sessions(tmp_path)) == 1


def test_wrong_shape_session_is_skipped_not_fatal(tmp_path):
    """A results.json that parses fine but is a list or a string, not an
    object, must not take the other 441 sessions down with it."""
    (tmp_path / "listshaped").mkdir()
    (tmp_path / "listshaped" / "results.json").write_text(json.dumps([1, 2, 3]))
    (tmp_path / "stringshaped").mkdir()
    (tmp_path / "stringshaped" / "results.json").write_text(json.dumps("nope"))
    _session(tmp_path, "ok", False, True)
    out = load_capture_sessions(tmp_path)
    assert [s.session_id for s in out] == ["ok"]


def test_non_numeric_frame_count_is_skipped_not_fatal(tmp_path):
    d = tmp_path / "badcount"
    d.mkdir()
    (d / "results.json").write_text(json.dumps({
        "session_id": "badcount", "swapped": False, "frame_count": "many",
        "scan": {"verdict": "LIVE"},
        "decision": {"approved": True, "reason": "approved"},
    }))
    _session(tmp_path, "ok", False, True)
    out = load_capture_sessions(tmp_path)
    assert [s.session_id for s in out] == ["ok"]


def test_a_skipped_session_is_reported_not_swallowed(tmp_path, caplog):
    """442 sessions, of which exactly 5 are the fraud that matters. A session
    dropped in silence could be one of the 5 and nobody would know.

    Neither assertion below can be satisfied by the JSONDecodeError message
    alone: that message ("Expecting property name enclosed in double quotes:
    line 1 column 2 (char 1)") contains the digit "1" three times over on its
    own, so a bare `"1" in caplog.text` would pass even if the skip count
    were never computed at all. Asserting the literal phrase the summary log
    line emits (`"skipped 1"`) ties the check to the actual count. Likewise,
    "broken" only survives here because the session name is threaded through
    real tracking state (the `skipped` list) into the logged text, not
    because it happens to appear in the parse-error string — verified by
    stripping the name out of both log format strings, which makes this
    assertion fail as expected.
    """
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "results.json").write_text("{not json")
    _session(tmp_path, "ok", False, True)
    with caplog.at_level(logging.WARNING):
        load_capture_sessions(tmp_path)
    assert "broken" in caplog.text
    assert "skipped 1" in caplog.text


@pytest.mark.parametrize("frame_count", [0, 1, 37, 900])
def test_frame_count_is_preserved(tmp_path, frame_count):
    d = tmp_path / "s1"
    d.mkdir()
    (d / "results.json").write_text(json.dumps({
        "session_id": "s1", "swapped": False, "frame_count": frame_count,
        "scan": {"verdict": "LIVE"},
        "decision": {"approved": True, "reason": "approved"},
    }))
    assert load_capture_sessions(tmp_path)[0].frame_count == frame_count
