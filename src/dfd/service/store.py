"""Durable submission queue and result store, on SQLite.

One table, because there is one thing to record: a submission and what
happened to it. The queue discipline lives in `claim_next`, which moves a row
from `queued` to `running` inside a single conditional UPDATE — two workers
racing cannot both win that update, so neither can be handed the same file.
"""
from __future__ import annotations

import logging
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions (
    id              TEXT PRIMARY KEY,
    content_sha256  TEXT NOT NULL,
    filename        TEXT NOT NULL,
    source          TEXT NOT NULL,
    path            TEXT NOT NULL,
    received_at     TEXT NOT NULL,
    started_at      TEXT,
    finished_at     TEXT,
    status          TEXT NOT NULL,
    verdict         TEXT,
    llr_total       REAL,
    ood_score       REAL,
    error           TEXT,
    record_json     TEXT,
    record_digest   TEXT
);
CREATE INDEX IF NOT EXISTS submissions_status_received
    ON submissions (status, received_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class Submission:
    """One file, and everything the service knows about what happened to it."""
    id: str
    content_sha256: str
    filename: str
    source: str
    path: str
    received_at: str
    started_at: str | None
    finished_at: str | None
    status: str
    verdict: str | None
    llr_total: float | None
    ood_score: float | None
    error: str | None
    record_json: str | None
    record_digest: str | None

    def summary(self) -> dict[str, Any]:
        """The row as the API returns it — without the full audit record.

        The record is often tens of kilobytes and a listing returns fifty of
        them; it is fetched per id instead.
        """
        return {
            "id": self.id,
            "filename": self.filename,
            "source": self.source,
            "content_sha256": self.content_sha256,
            "received_at": self.received_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "verdict": self.verdict,
            "llr_total": self.llr_total,
            "ood_score": self.ood_score,
            "error": self.error,
            "record_digest": self.record_digest,
        }


class Store:
    """SQLite-backed store. Safe to construct repeatedly on the same path."""

    def __init__(self, path: str | Path) -> None:
        """Open (and if needed create) the database at `path`.

        Args:
            path: database file; parent directories are created.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            # WAL: a reader (the HTTP thread serving the dashboard) must not
            # block on the writer (the worker finishing a decision), and the
            # default rollback journal makes exactly that happen under load.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # A connection per operation, not one shared across threads: sqlite3
        # objects are not safe to share between threads by default, and this
        # store is read by the HTTP threads while a worker writes.
        conn = sqlite3.connect(self.path, timeout=30.0,
                               isolation_level="IMMEDIATE")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def enqueue(self, *, path: str, filename: str, source: str,
                content_sha256: str) -> str:
        """Record a new submission in `queued` and return its id."""
        sid = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO submissions (id, content_sha256, filename, "
                "source, path, received_at, status) VALUES (?,?,?,?,?,?,?)",
                (sid, content_sha256, filename, source, str(path), _now(),
                 QUEUED))
        logger.info("queued %s (%s, from %s)", sid, filename, source)
        return sid

    def claim_next(self) -> Submission | None:
        """Atomically take the oldest queued submission, or None.

        The claim is a single `UPDATE ... WHERE id = ? AND status = 'queued'`
        under an IMMEDIATE transaction. Selecting a candidate and then
        updating it unconditionally is the version of this that hands one
        file to two workers, which is why the status is in the WHERE clause
        and the rowcount is checked.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM submissions WHERE status = ? "
                "ORDER BY received_at, rowid LIMIT 1", (QUEUED,)).fetchone()
            if row is None:
                return None
            cur = conn.execute(
                "UPDATE submissions SET status = ?, started_at = ? "
                "WHERE id = ? AND status = ?",
                (RUNNING, _now(), row["id"], QUEUED))
            if cur.rowcount != 1:
                # Another worker won the race between the SELECT and the
                # UPDATE. Returning None costs one idle poll; returning the
                # row would be the double-claim this method exists to stop.
                return None
            claimed = conn.execute("SELECT * FROM submissions WHERE id = ?",
                                   (row["id"],)).fetchone()
        return _to_submission(claimed)

    def finish(self, sid: str, *, verdict: str, llr_total: float,
               ood_score: float, record_json: str,
               record_digest: str) -> None:
        """Record a completed decision.

        Raises:
            ValueError: if the submission is not currently `running`. A
                finish against a queued or already-finished row means the
                caller lost track of the queue, and overwriting the earlier
                result would destroy the audit trail rather than report the
                bug.
        """
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE submissions SET status = ?, finished_at = ?, "
                "verdict = ?, llr_total = ?, ood_score = ?, record_json = ?, "
                "record_digest = ? WHERE id = ? AND status = ?",
                (DONE, _now(), verdict, llr_total, ood_score, record_json,
                 record_digest, sid, RUNNING))
            if cur.rowcount != 1:
                raise ValueError(f"submission {sid} is not running")

    def fail(self, sid: str, error: str) -> None:
        """Record that a submission could not be decided, and why.

        Raises:
            ValueError: if the submission is not currently `running`.
        """
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE submissions SET status = ?, finished_at = ?, "
                "error = ? WHERE id = ? AND status = ?",
                (FAILED, _now(), error, sid, RUNNING))
            if cur.rowcount != 1:
                raise ValueError(f"submission {sid} is not running")

    def requeue_running(self) -> int:
        """Return anything left `running` to the queue. Returns the count.

        Called at startup. A submission is only `running` while a worker
        holds it, so any row in that state after a restart belongs to a
        worker that died — and the file is still on disk, so the work is
        repeatable.
        """
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE submissions SET status = ?, started_at = NULL "
                "WHERE status = ?", (QUEUED, RUNNING))
            count = int(cur.rowcount)
        if count:
            logger.warning("requeued %d submission(s) left running by a "
                           "previous process", count)
        return count

    def get(self, sid: str) -> Submission | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM submissions WHERE id = ?",
                               (sid,)).fetchone()
        return _to_submission(row) if row is not None else None

    def recent(self, limit: int = 50, status: str | None = None
               ) -> list[Submission]:
        """The newest submissions first, at most `limit`."""
        sql = "SELECT * FROM submissions"
        params: tuple[Any, ...] = ()
        if status is not None:
            sql += " WHERE status = ?"
            params = (status,)
        sql += " ORDER BY received_at DESC, rowid DESC LIMIT ?"
        with self._connect() as conn:
            rows = conn.execute(sql, (*params, int(limit))).fetchall()
        return [_to_submission(r) for r in rows]

    def unfinished_paths(self) -> set[str]:
        """Files belonging to submissions that have not been decided yet.

        Retention must never delete one of these: the file is the input to a
        decision that has not happened, and removing it turns a pending
        submission into a fabricated failure.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT path FROM submissions WHERE status IN (?, ?)",
                (QUEUED, RUNNING)).fetchall()
        return {r["path"] for r in rows}

    def stats(self) -> dict[str, dict[str, int]]:
        """Counts by status and by verdict, for the dashboard and /health."""
        with self._connect() as conn:
            by_status = {r["status"]: r["n"] for r in conn.execute(
                "SELECT status, COUNT(*) AS n FROM submissions GROUP BY status")}
            by_verdict = {r["verdict"]: r["n"] for r in conn.execute(
                "SELECT verdict, COUNT(*) AS n FROM submissions "
                "WHERE verdict IS NOT NULL GROUP BY verdict")}
        return {"status": by_status, "verdict": by_verdict}


def _to_submission(row: sqlite3.Row) -> Submission:
    return Submission(
        id=row["id"], content_sha256=row["content_sha256"],
        filename=row["filename"], source=row["source"], path=row["path"],
        received_at=row["received_at"], started_at=row["started_at"],
        finished_at=row["finished_at"], status=row["status"],
        verdict=row["verdict"], llr_total=row["llr_total"],
        ood_score=row["ood_score"], error=row["error"],
        record_json=row["record_json"], record_digest=row["record_digest"])
