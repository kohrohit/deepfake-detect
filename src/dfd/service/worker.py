"""Picks files up, scores them through `decide`, and records what happened.

The worker owns two things the decision library deliberately does not: where
a submitted file lives, and what to do when scoring one raises. Both are
operational concerns, and both are where an always-on system actually fails.
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from ..errors import DfdError, InvalidInput
from .store import Store

logger = logging.getLogger(__name__)

#: Never submissions. A dotfile is an editor's or the OS's business, and the
#: three suffixes are what browsers and copy tools name a file that is still
#: arriving — enqueuing those produces a failed row per download, which buries
#: the failures that mean something.
IGNORED_SUFFIXES = (".part", ".crdownload", ".tmp", ".partial", ".download")

#: A file must be the same size on two consecutive scans before it is taken.
#: A file copied into the inbox is visible long before it is complete, and a
#: truncated decode is recorded as a refusal — a fabricated one, about a file
#: that was fine.
STABLE_SCANS = 2


class _Record(Protocol):
    """What the worker needs from a decision, and nothing more.

    Read-only PROPERTIES, not bare attributes: `AuditRecord` is a frozen
    dataclass, and a protocol that declares settable attributes is not
    satisfied by one — the record would be rejected by the type checker for
    being immutable, which is the property it was given on purpose.
    """

    @property
    def verdict(self) -> Any: ...
    @property
    def llr_total(self) -> float: ...
    @property
    def ood_score(self) -> float: ...

    def to_json(self) -> str: ...


DecideFn = Callable[[Path], _Record]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


class Worker:
    """Moves files from an inbox (or an upload) into the queue, and drains it."""

    def __init__(self, *, store: Store, decide: DecideFn, inbox: str | Path,
                 workdir: str | Path, poll_interval: float = 1.0) -> None:
        """
        Args:
            store: where submissions and results live.
            decide: called with a path, returns an audit record. Injected so
                the worker is testable without weights, and so the expensive
                composition (registry, calibrators) happens once at startup.
            inbox: directory watched for dropped files.
            workdir: where taken files are kept. Files are MOVED here, which
                is what makes a second scan a no-op.
            poll_interval: seconds between scans in `run_until_stopped`.
        """
        self.store = store
        self.decide = decide
        self.inbox = Path(inbox)
        self.workdir = Path(workdir)
        self.poll_interval = float(poll_interval)
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.workdir.mkdir(parents=True, exist_ok=True)
        # name -> (size, consecutive scans at that size). Held in memory
        # deliberately: it is a property of THIS process's observations, and
        # persisting it would let a stale entry wave through a file that is
        # being written right now.
        self._seen: dict[str, tuple[int, int]] = {}

    def recover(self) -> int:
        """Requeue anything a previous process left running. Returns the count."""
        return self.store.requeue_running()

    def scan_inbox(self) -> int:
        """Take every settled file in the inbox. Returns how many were taken."""
        if not self.inbox.is_dir():
            return 0
        taken = 0
        for path in sorted(self.inbox.iterdir()):
            if not path.is_file():
                continue
            name = path.name
            if name.startswith(".") or path.suffix.lower() in IGNORED_SUFFIXES:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                # Vanished between listing and stat — someone moved it out
                # from under us, which is not an error, just a non-event.
                self._seen.pop(name, None)
                continue
            previous_size, scans = self._seen.get(name, (-1, 0))
            scans = scans + 1 if size == previous_size else 1
            self._seen[name] = (size, scans)
            if scans < STABLE_SCANS:
                logger.debug("%s not settled yet (size %d, scan %d)",
                             name, size, scans)
                continue
            self._seen.pop(name, None)
            if size == 0:
                logger.warning("ignoring empty file %s", name)
                path.unlink(missing_ok=True)
                continue
            self._take(path, source="watch")
            taken += 1
        return taken

    def submit_bytes(self, data: bytes, *, filename: str,
                     source: str = "api") -> str:
        """Enqueue uploaded bytes directly. Returns the submission id.

        Raises:
            InvalidInput: if `data` is empty. An empty upload is a client
                bug, and a queued row for it would be answered minutes later
                with a decode failure instead of immediately with the truth.
        """
        if not data:
            raise InvalidInput("refusing an empty upload")
        target = self._reserve(Path(filename).name or "upload")
        target.write_bytes(data)
        return self._enqueue(target, Path(filename).name or "upload", source)

    def process_one(self) -> bool:
        """Score one queued submission. Returns False if the queue was empty.

        Never raises for a bad file. A decision that fails is that
        submission's failure; a worker that dies of it is every later
        submission's failure too.
        """
        claimed = self.store.claim_next()
        if claimed is None:
            return False
        try:
            record = self.decide(Path(claimed.path))
        except DfdError as exc:
            # A refusal: unreadable, unsupported, over a decode limit. The
            # message is the useful part and is recorded verbatim.
            logger.warning("refused %s: %s", claimed.filename, exc)
            self.store.fail(claimed.id, str(exc))
            return True
        except Exception as exc:  # noqa: BLE001 - see the docstring above
            logger.exception("scoring %s failed", claimed.filename)
            self.store.fail(claimed.id, f"{type(exc).__name__}: {exc}")
            return True

        payload = record.to_json()
        self.store.finish(
            claimed.id,
            verdict=str(getattr(record.verdict, "value", record.verdict)),
            llr_total=float(record.llr_total),
            ood_score=float(record.ood_score),
            record_json=payload,
            record_digest=hashlib.sha256(payload.encode()).hexdigest())
        logger.info("decided %s: %s", claimed.filename,
                    getattr(record.verdict, "value", record.verdict))
        return True

    def run_until_stopped(self, stop: threading.Event) -> None:
        """Scan and drain until `stop` is set."""
        logger.info("worker running: inbox=%s workdir=%s", self.inbox,
                    self.workdir)
        while not stop.is_set():
            try:
                self.scan_inbox()
                worked = self.process_one()
            except Exception:  # noqa: BLE001
                # The loop itself must survive a transient filesystem or
                # database error; the alternative is a service that is up,
                # accepting files, and silently scoring none of them.
                logger.exception("worker loop error")
                worked = False
            if not worked:
                stop.wait(self.poll_interval)
        logger.info("worker stopped")

    def _reserve(self, filename: str) -> Path:
        """A collision-free path under workdir, keeping the original name."""
        stem = Path(filename).stem or "file"
        suffix = Path(filename).suffix
        day = self.workdir / datetime.now(timezone.utc).strftime("%Y-%m-%d")
        day.mkdir(parents=True, exist_ok=True)
        return day / f"{_utc_stamp()}-{uuid.uuid4().hex[:8]}-{stem}{suffix}"

    def _take(self, path: Path, *, source: str) -> str:
        target = self._reserve(path.name)
        # move, not copy: the inbox must end up empty, or the next scan takes
        # the same file again.
        shutil.move(str(path), str(target))
        return self._enqueue(target, path.name, source)

    def _enqueue(self, target: Path, filename: str, source: str) -> str:
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        return self.store.enqueue(path=str(target), filename=filename,
                                  source=source, content_sha256=digest)
