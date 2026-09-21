import json
import subprocess
import sys
from datetime import datetime, timezone
from unittest import mock

import cv2
import numpy as np
import pytest

from dfd.cli import main


class _FrozenDatetime(datetime):
    """Pins `dfd.audit`'s wall clock so two independent `decide()` calls in
    the same test produce an identical `created_at` and are therefore
    comparable. Without this, `created_at` legitimately differs between the
    two calls below — that is correct production behaviour, not a bug — and
    the test would fail on wall-clock jitter rather than on anything this
    CLI does wrong. That is a test outcome depending on machine timing, the
    same category of problem as Ruling P3's weights-on-disk issue."""

    @classmethod
    def now(cls, tz=None):
        return datetime(2024, 1, 1, tzinfo=timezone.utc)

# Guaranteed not to exist: forces the face stage to abstain with
# weights_absent regardless of whether this machine happens to have the real
# YuNet weights checked out under assets/models/ (Ruling P3 — the weights
# file is gitignored, untracked, absent in CI, but present on some dev
# machines, so any test asserting a face-stage reason must pin the model
# path explicitly rather than rely on the DEFAULT_MODEL path's environment).
MISSING_FACE_MODEL = "/nonexistent/yunet.onnx"


@pytest.fixture
def png(tmp_path):
    p = tmp_path / "subject.png"
    rng = np.random.default_rng(0)
    cv2.imwrite(str(p), rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    return p


@pytest.fixture
def mp4(tmp_path):
    """A real decodable clip, so the video branch of the CLI is exercised."""
    p = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (256, 256))
    rng = np.random.default_rng(0)
    for _ in range(30):
        vw.write(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    vw.release()
    return p


def test_the_video_flags_reach_the_pipeline(mp4, capsys):
    """`--max-frames` and `--seed` had no test that passed them at all, so
    nothing connected the flags to `decide`'s arguments. The frame count in
    `stage_reasons` is the observable end of `--max-frames`; `--seed`'s effect
    on which frames are chosen is asserted in
    tests/test_pipeline.py::test_the_video_seed_reaches_the_frame_sampler,
    which can inspect the score rather than only the record."""
    assert main(["score", str(mp4), "--face-model", MISSING_FACE_MODEL,
                 "--max-frames", "3", "--seed", "5"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["stage_reasons"]["frames_with_face"] == "0/3"


def test_a_decision_exits_zero_even_when_it_abstains(png, capsys):
    """Exit status says whether the tool ran, never what the verdict was."""
    assert main(["score", str(png)]) == 0


def test_stdout_is_exactly_one_json_object(png, capsys):
    main(["score", str(png)])
    out = capsys.readouterr().out
    record = json.loads(out)
    assert record["schema_version"] == "2"
    assert out.count("\n") == 1, "stdout must carry the record and nothing else"


def test_the_record_carries_one_evidence_row_per_default_detector(png, capsys):
    """The CLI's registry is `default_registry()`, whose composition nothing
    asserted. Two detectors in, two evidence rows out — dropping either
    `register(...)` call silently halves the evidence the product consults,
    and `insufficient_evidence` looks identical either way."""
    main(["score", str(png), "--face-model", MISSING_FACE_MODEL])
    record = json.loads(capsys.readouterr().out)
    assert [row["detector"] for row in record["evidence"]] == ["effnet_b4", "npr"]
    assert sorted(record["model_versions"]) == ["effnet_b4", "npr"]


def test_the_human_summary_goes_to_stderr_not_stdout(png, capsys):
    main(["score", str(png)])
    captured = capsys.readouterr()
    assert "verdict=" in captured.err
    assert "verdict=" not in captured.out


def test_the_summary_names_the_face_stage(png, capsys):
    """Forces the face model path to one guaranteed absent (Ruling P3):
    the DEFAULT_MODEL path may or may not exist on the machine running this
    test, and this assertion must not depend on that."""
    main(["score", str(png), "--face-model", MISSING_FACE_MODEL])
    assert "faces=weights_absent" in capsys.readouterr().err


def test_a_missing_file_exits_two_with_a_message_and_no_traceback(tmp_path, capsys):
    code = main(["score", str(tmp_path / "nope.png")])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "Traceback" not in captured.err and captured.err.startswith("dfd:")


def test_an_unsupported_extension_exits_two(tmp_path, capsys):
    p = tmp_path / "a.xyz"
    p.write_bytes(b"x")
    assert main(["score", str(p)]) == 2


def test_pretty_output_is_the_same_record_reformatted(png, capsys):
    with mock.patch("dfd.audit.datetime", _FrozenDatetime):
        main(["score", str(png), "--pretty"])
        pretty = capsys.readouterr().out
        assert "\n  " in pretty
        main(["score", str(png)])
        compact = capsys.readouterr().out
    assert json.loads(pretty) == json.loads(compact)


def test_the_module_runs_as_a_subprocess_with_clean_stdout(png):
    """The real entry point, not an in-process call: proves nothing in the
    import chain prints to stdout and pollutes the record."""
    proc = subprocess.run([sys.executable, "-m", "dfd", "score", str(png)],
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    assert json.loads(proc.stdout)["sample_id"] == "subject"


def test_subject_id_is_accepted_and_does_not_change_sample_identity(png, capsys):
    """Named for what it actually proves. AuditRecord has NO subject field, so
    --subject-id reaches Context and stops there; it cannot be asserted in the
    record. See known gap 6 — do not rename this test to imply otherwise."""
    main(["score", str(png), "--subject-id", "applicant-7"])
    assert json.loads(capsys.readouterr().out)["sample_id"] == "subject"


def test_logging_reaches_stderr_formatted_and_never_touches_the_record(png):
    """The whole point of the stdout/stderr split is that log output must
    never contaminate the record. Run as a real subprocess, at the loudest
    verbosity, because in-process pytest installs its own root handlers and
    `basicConfig` is then a deliberate no-op — only a subprocess shows what a
    shell actually sees.

    The `WARNING dfd.faces:` prefix is the assertion that matters: before
    `main` configured logging, that line arrived through Python's lastResort
    handler as a bare message with no level and no logger name, and every
    INFO/DEBUG diagnostic in `decide` was unreachable from the only entry
    point that exists.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "dfd", "score", str(png),
         "--face-model", MISSING_FACE_MODEL, "-vv"],
        capture_output=True, text=True, check=False)

    assert proc.returncode == 0
    assert json.loads(proc.stdout)["schema_version"] == "2"
    assert proc.stdout.count("\n") == 1, \
        "stdout must carry exactly one JSON object even with logging on"
    assert f"WARNING dfd.faces: Face detector weights absent at {MISSING_FACE_MODEL}" \
        in proc.stderr
    assert "DEBUG dfd.pipeline:" in proc.stderr
    assert "INFO dfd.pipeline:" in proc.stderr
    assert proc.stderr.rstrip().splitlines()[-1].startswith("verdict="), \
        "the human summary stays the last line of stderr"


def test_the_default_verbosity_keeps_stderr_quiet(png):
    """Default is WARNING: a script parsing stderr must not suddenly receive
    every stage's DEBUG line because logging was switched on."""
    proc = subprocess.run(
        [sys.executable, "-m", "dfd", "score", str(png),
         "--face-model", MISSING_FACE_MODEL],
        capture_output=True, text=True, check=False)

    assert proc.returncode == 0
    assert "DEBUG" not in proc.stderr
    assert "INFO" not in proc.stderr
    assert "WARNING dfd.faces:" in proc.stderr
    assert proc.stderr.rstrip().splitlines()[-1].startswith("verdict=")


def test_max_frames_below_one_is_a_usage_error_not_a_corrupt_file(png, capsys):
    """`--max-frames 0` reached `load_video`, which raised its zero-frame
    ValueError, which `_ingest` relabelled `could not decode <file>` — telling
    a user who mistyped a flag that their video is corrupt. Rejected at the
    parser now, with the flag named."""
    with pytest.raises(SystemExit) as exc:
        main(["score", str(png), "--max-frames", "0"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--max-frames" in err and "must be >= 1" in err
    assert "could not decode" not in err
