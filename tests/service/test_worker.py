"""The worker: pick files up safely, and never die of one bad file."""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from dfd.errors import InvalidInput
from dfd.service.store import Store
from dfd.service.worker import Worker


class _Record:
    """Enough of an AuditRecord for the worker's contract."""
    def __init__(self, verdict: str = "INSUFFICIENT_EVIDENCE") -> None:
        self.verdict = verdict
        self.llr_total = 0.5
        self.ood_score = 0.1

    def to_json(self) -> str:
        return '{"verdict": "%s"}' % self.verdict


def _worker(tmp_path: Path, decide=None, **kw) -> Worker:
    return Worker(
        store=Store(tmp_path / "db.sqlite3"),
        decide=decide or (lambda path: _Record()),
        inbox=tmp_path / "inbox",
        workdir=tmp_path / "work",
        **kw)


def _drop(inbox: Path, name: str, data: bytes = b"\xff\xd8\xff\xe0jpegbytes") -> Path:
    inbox.mkdir(parents=True, exist_ok=True)
    p = inbox / name
    p.write_bytes(data)
    return p


def test_a_file_in_the_inbox_is_enqueued_with_its_content_hash(
        tmp_path: Path) -> None:
    w = _worker(tmp_path)
    _drop(w.inbox, "a.jpg", b"abc")
    # Twice: a file is taken only once its size has held still (see
    # test_a_file_still_being_written_is_left_alone_until_it_settles).
    w.scan_inbox()
    assert w.scan_inbox() == 1
    row = w.store.recent()[0]
    assert row.filename == "a.jpg"
    assert row.content_sha256 == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    assert row.status == "queued"


def test_the_file_is_moved_out_of_the_inbox_so_it_is_not_enqueued_twice(
        tmp_path: Path) -> None:
    w = _worker(tmp_path)
    _drop(w.inbox, "a.jpg")
    w.scan_inbox()
    assert w.scan_inbox() == 1
    assert w.scan_inbox() == 0
    assert list(w.inbox.iterdir()) == []
    assert Path(w.store.recent()[0].path).exists()


def test_a_file_still_being_written_is_left_alone_until_it_settles(
        tmp_path: Path) -> None:
    """A half-copied file decoded as truncated is a fabricated refusal."""
    w = _worker(tmp_path)
    p = _drop(w.inbox, "big.jpg", b"partial")
    assert w.scan_inbox() == 0          # first sighting: size not yet confirmed
    p.write_bytes(b"partial-and-more")  # still growing
    assert w.scan_inbox() == 0
    assert w.scan_inbox() == 1          # unchanged since the last look


def test_two_files_with_the_same_name_do_not_overwrite_each_other(
        tmp_path: Path) -> None:
    w = _worker(tmp_path)
    _drop(w.inbox, "a.jpg", b"first")
    w.scan_inbox()
    w.scan_inbox()
    _drop(w.inbox, "a.jpg", b"second")
    w.scan_inbox()
    w.scan_inbox()
    paths = {r.path for r in w.store.recent()}
    assert len(paths) == 2
    assert {Path(p).read_bytes() for p in paths} == {b"first", b"second"}


def test_dotfiles_and_partial_downloads_are_ignored_entirely(
        tmp_path: Path) -> None:
    """These are never submissions, and a failed row for each is noise."""
    w = _worker(tmp_path)
    for name in (".DS_Store", "x.jpg.part", "y.crdownload", ".hidden.jpg"):
        _drop(w.inbox, name)
    assert w.scan_inbox() == 0
    assert w.scan_inbox() == 0
    assert w.store.recent() == []


def test_processing_stores_the_verdict_from_the_decision(
        tmp_path: Path) -> None:
    w = _worker(tmp_path, decide=lambda path: _Record("FAKE"))
    _drop(w.inbox, "a.jpg")
    w.scan_inbox(); w.scan_inbox()
    assert w.process_one() is True
    row = w.store.recent()[0]
    assert row.status == "done"
    assert row.verdict == "FAKE"
    assert row.record_json == '{"verdict": "FAKE"}'
    assert len(row.record_digest) == 64


def test_a_refused_input_is_recorded_as_failed_not_as_a_verdict(
        tmp_path: Path) -> None:
    def decide(path):
        raise InvalidInput("unsupported file extension '.txt'")
    w = _worker(tmp_path, decide=decide)
    _drop(w.inbox, "a.txt")
    w.scan_inbox(); w.scan_inbox()
    w.process_one()
    row = w.store.recent()[0]
    assert row.status == "failed"
    assert row.verdict is None
    assert "unsupported" in row.error


def test_an_unexpected_exception_fails_one_submission_not_the_worker(
        tmp_path: Path) -> None:
    """One corrupt file must not stop the service from scoring the next."""
    seen: list[str] = []

    def decide(path):
        seen.append(Path(path).name)
        if len(seen) == 1:
            raise RuntimeError("something in torch fell over")
        return _Record("REAL")

    w = _worker(tmp_path, decide=decide)
    _drop(w.inbox, "bad.jpg", b"1")
    _drop(w.inbox, "good.jpg", b"2")
    w.scan_inbox(); w.scan_inbox()
    w.process_one()
    w.process_one()
    rows = {r.filename: r for r in w.store.recent()}
    assert rows["bad.jpg"].status == "failed"
    assert "torch fell over" in rows["bad.jpg"].error
    assert rows["good.jpg"].status == "done"
    assert rows["good.jpg"].verdict == "REAL"


def test_processing_an_empty_queue_reports_that_it_did_nothing(
        tmp_path: Path) -> None:
    assert _worker(tmp_path).process_one() is False


def test_run_until_stopped_drains_the_queue_and_then_exits(
        tmp_path: Path) -> None:
    w = _worker(tmp_path, poll_interval=0.01)
    _drop(w.inbox, "a.jpg")
    stop = threading.Event()
    t = threading.Thread(target=w.run_until_stopped, args=(stop,))
    t.start()
    try:
        deadline = 5.0
        while deadline > 0 and (not w.store.recent()
                                or w.store.recent()[0].status != "done"):
            threading.Event().wait(0.05)
            deadline -= 0.05
    finally:
        stop.set()
        t.join(timeout=5)
    assert not t.is_alive()
    assert w.store.recent()[0].status == "done"


def test_a_submission_left_running_by_a_crash_is_retried_on_startup(
        tmp_path: Path) -> None:
    w = _worker(tmp_path)
    _drop(w.inbox, "a.jpg")
    w.scan_inbox(); w.scan_inbox()
    w.store.claim_next()          # a worker took it, then the process died
    assert w.recover() == 1
    assert w.process_one() is True
    assert w.store.recent()[0].status == "done"


def test_submitting_bytes_directly_enqueues_them_without_the_inbox(
        tmp_path: Path) -> None:
    """The API path: no file on disk yet, and the worker owns where it goes."""
    w = _worker(tmp_path)
    sid = w.submit_bytes(b"jpegbytes", filename="upload.jpg", source="api")
    row = w.store.get(sid)
    assert row.status == "queued"
    assert Path(row.path).read_bytes() == b"jpegbytes"
    assert w.process_one() is True


def test_an_empty_upload_is_refused_before_it_reaches_the_queue(
        tmp_path: Path) -> None:
    w = _worker(tmp_path)
    with pytest.raises(InvalidInput, match="empty"):
        w.submit_bytes(b"", filename="upload.jpg", source="api")
    assert w.store.recent() == []


def _age(path: Path, days: float) -> None:
    import os, time
    t = time.time() - days * 86400
    os.utime(path, (t, t))


def test_pruning_removes_files_older_than_the_retention_window(
        tmp_path: Path) -> None:
    """An always-on service that never deletes anything fills the disk."""
    w = _worker(tmp_path)
    old = w.submit_bytes(b"old", filename="old.jpg", source="api")
    new = w.submit_bytes(b"new", filename="new.jpg", source="api")
    # Decided, not pending: a queued file is protected whatever its age,
    # which has its own test below.
    w.process_one(); w.process_one()
    _age(Path(w.store.get(old).path), days=40)
    assert w.prune_workdir(retain_days=30) == 1
    assert not Path(w.store.get(old).path).exists()
    assert Path(w.store.get(new).path).exists()


def test_pruning_never_removes_the_record_of_the_decision(
        tmp_path: Path) -> None:
    """The file is evidence; the audit record IS the decision. Keep it."""
    w = _worker(tmp_path, decide=lambda p: _Record("REAL"))
    sid = w.submit_bytes(b"old", filename="old.jpg", source="api")
    w.process_one()
    _age(Path(w.store.get(sid).path), days=40)
    w.prune_workdir(retain_days=30)
    row = w.store.get(sid)
    assert row.verdict == "REAL"
    assert row.record_json is not None


def test_a_retention_of_zero_days_disables_pruning(tmp_path: Path) -> None:
    """Zero means 'keep everything', not 'delete everything'."""
    w = _worker(tmp_path)
    sid = w.submit_bytes(b"old", filename="old.jpg", source="api")
    _age(Path(w.store.get(sid).path), days=400)
    assert w.prune_workdir(retain_days=0) == 0
    assert Path(w.store.get(sid).path).exists()


def test_a_queued_submission_is_never_pruned_out_from_under_the_worker(
        tmp_path: Path) -> None:
    """Deleting a file that is still waiting to be scored turns a pending
    decision into a fabricated failure."""
    w = _worker(tmp_path)
    sid = w.submit_bytes(b"old", filename="old.jpg", source="api")
    _age(Path(w.store.get(sid).path), days=400)
    assert w.prune_workdir(retain_days=1) == 0
    assert w.process_one() is True
