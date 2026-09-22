"""The submission store: durable, and safe for two workers at once.

A queue that can hand the same row to two workers double-charges every
downstream cost and writes two audit records for one decision. A queue that
loses a row when a worker dies silently drops evidence. Both are tested here
rather than assumed from SQLite's reputation.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from dfd.service.store import Store


@pytest.fixture()
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "dfd.sqlite3")


def test_a_submission_starts_queued_and_comes_back_by_id(store: Store) -> None:
    sid = store.enqueue(path="/tmp/a.jpg", filename="a.jpg", source="api",
                        content_sha256="ab" * 32)
    row = store.get(sid)
    assert row is not None
    assert row.status == "queued"
    assert row.filename == "a.jpg"
    assert row.verdict is None


def test_claiming_returns_the_oldest_queued_submission(store: Store) -> None:
    first = store.enqueue(path="/tmp/a.jpg", filename="a.jpg", source="api",
                          content_sha256="a" * 64)
    store.enqueue(path="/tmp/b.jpg", filename="b.jpg", source="api",
                  content_sha256="b" * 64)
    claimed = store.claim_next()
    assert claimed is not None and claimed.id == first
    assert store.get(first).status == "running"


def test_two_claims_never_return_the_same_submission(store: Store) -> None:
    """The invariant the whole queue exists for."""
    store.enqueue(path="/tmp/a.jpg", filename="a.jpg", source="api",
                  content_sha256="a" * 64)
    a = store.claim_next()
    b = store.claim_next()
    assert a is not None
    assert b is None


def test_claiming_an_empty_queue_returns_none(store: Store) -> None:
    assert store.claim_next() is None


def test_finishing_records_the_verdict_and_the_record(store: Store) -> None:
    sid = store.enqueue(path="/tmp/a.jpg", filename="a.jpg", source="api",
                        content_sha256="a" * 64)
    store.claim_next()
    store.finish(sid, verdict="INSUFFICIENT_EVIDENCE", llr_total=0.0,
                 ood_score=0.0, record_json='{"k": 1}', record_digest="d" * 64)
    row = store.get(sid)
    assert row.status == "done"
    assert row.verdict == "INSUFFICIENT_EVIDENCE"
    assert row.record_json == '{"k": 1}'
    assert row.finished_at is not None


def test_failing_records_the_reason_and_does_not_pretend_to_a_verdict(
        store: Store) -> None:
    sid = store.enqueue(path="/tmp/a.jpg", filename="a.jpg", source="api",
                        content_sha256="a" * 64)
    store.claim_next()
    store.fail(sid, "unsupported file extension '.txt'")
    row = store.get(sid)
    assert row.status == "failed"
    assert row.verdict is None
    assert "unsupported" in row.error


def test_a_submission_left_running_by_a_crash_is_requeued(store: Store) -> None:
    """A worker killed mid-decision must not take the submission with it."""
    sid = store.enqueue(path="/tmp/a.jpg", filename="a.jpg", source="api",
                        content_sha256="a" * 64)
    store.claim_next()
    assert store.requeue_running() == 1
    assert store.get(sid).status == "queued"
    assert store.claim_next().id == sid


def test_rows_survive_reopening_the_database(tmp_path: Path) -> None:
    path = tmp_path / "dfd.sqlite3"
    sid = Store(path).enqueue(path="/tmp/a.jpg", filename="a.jpg",
                              source="watch", content_sha256="a" * 64)
    assert Store(path).get(sid).filename == "a.jpg"


def test_recent_lists_newest_first_and_honours_the_limit(store: Store) -> None:
    ids = [store.enqueue(path=f"/tmp/{i}.jpg", filename=f"{i}.jpg",
                         source="api", content_sha256=str(i) * 64)
           for i in range(5)]
    recent = store.recent(limit=3)
    assert [r.id for r in recent] == ids[::-1][:3]


def test_stats_count_by_status_and_verdict(store: Store) -> None:
    a = store.enqueue(path="/tmp/a.jpg", filename="a.jpg", source="api",
                      content_sha256="a" * 64)
    b = store.enqueue(path="/tmp/b.jpg", filename="b.jpg", source="api",
                      content_sha256="b" * 64)
    store.claim_next()
    store.finish(a, verdict="FAKE", llr_total=2.0, ood_score=0.1,
                 record_json="{}", record_digest="d" * 64)
    store.claim_next()
    store.fail(b, "boom")
    stats = store.stats()
    assert stats["status"]["done"] == 1
    assert stats["status"]["failed"] == 1
    assert stats["verdict"]["FAKE"] == 1


def test_the_same_file_submitted_twice_is_two_submissions(store: Store) -> None:
    """Re-scanning the same bytes is a legitimate request, not a duplicate.

    Deduplicating here would silently answer the second caller with the first
    caller's decision, made under a different model version.
    """
    kwargs = dict(path="/tmp/a.jpg", filename="a.jpg", source="api",
                  content_sha256="a" * 64)
    assert store.enqueue(**kwargs) != store.enqueue(**kwargs)


def test_finishing_a_submission_that_was_never_claimed_is_refused(
        store: Store) -> None:
    sid = store.enqueue(path="/tmp/a.jpg", filename="a.jpg", source="api",
                        content_sha256="a" * 64)
    with pytest.raises(ValueError, match="not running"):
        store.finish(sid, verdict="REAL", llr_total=-2.0, ood_score=0.0,
                     record_json="{}", record_digest="d" * 64)


def test_the_schema_is_created_once_and_reopening_does_not_reset_it(
        tmp_path: Path) -> None:
    path = tmp_path / "dfd.sqlite3"
    Store(path)
    Store(path)
    with sqlite3.connect(path) as conn:
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "submissions" in names
