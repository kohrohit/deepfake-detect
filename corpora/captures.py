"""Loader for the 442-session v-CIP capture corpus (spec §1.1).

Five of these sessions are swapped=true and approved=true. They are the actual
fraud, and any candidate system must be measured against them specifically —
aggregate accuracy over 442 sessions would hide all five.

COUNT THE IMAGES, NOT THE SESSIONS. 442 is a count of session folders. Measured
2026-09-22, by hashing every frame: those folders hold 1088 frame files but only
58 DISTINCT images, because 979 of the files are byte-identical to a single demo
asset replayed as the captured frame across 368 of the sessions. The genuine
face pool `corpora.face_pool.build_face_pool` extracts from the 435 non-swapped
sessions is 19 crops, not 435. Nothing in this module deduplicates — it reports
sessions, faithfully — so any caller sizing a corpus from `len(...)` of its
result is sizing it from a number that does not mean what it looks like. See
docs/HANDOFF.md §0.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaptureSession:
    session_id: str
    folder: str
    swapped: bool
    approved: bool
    scan_verdict: str | None
    frame_count: int

    @property
    def label(self) -> int:
        """1 = fake (a swap was injected), 0 = genuine."""
        return 1 if self.swapped else 0


def load_capture_sessions(root: str | Path) -> list[CaptureSession]:
    out: list[CaptureSession] = []
    skipped: list[str] = []
    for path in sorted(Path(root).glob("*/results.json")):
        try:
            d = json.loads(path.read_text())
            if not isinstance(d, dict):
                raise TypeError(
                    f"expected a JSON object, got {type(d).__name__}")
            decision = d.get("decision")
            decision = decision if isinstance(decision, dict) else {}
            scan = d.get("scan")
            scan = scan if isinstance(scan, dict) else {}
            # Validate and coerce every field here, inside the try. Nothing
            # past this point should be able to raise TypeError/ValueError/
            # KeyError/AttributeError for reasons unrelated to bad input —
            # those exceptions must still mean "malformed session", not
            # "bug in our own field-building code" or "bug in the
            # constructor call below", which is why CaptureSession(...) is
            # built outside this block.
            session_id = d.get("session_id", path.parent.name)
            swapped = bool(d.get("swapped", False))
            approved = bool(decision.get("approved", False))
            scan_verdict = scan.get("verdict")
            frame_count = int(d.get("frame_count", 0))
        except (json.JSONDecodeError, OSError, TypeError, ValueError,
                KeyError, AttributeError) as exc:
            # Never silent: a dropped session may be one of the five that are
            # the actual fraud, and aggregate counts would not reveal it. This
            # catches JSON that parses fine but is the wrong shape (a list, a
            # string, a field of the wrong type), not just decode/IO failures
            # — one malformed session must not take the other 441 down with it.
            logger.warning("skipping unreadable session %s: %s",
                           path.parent.name, exc)
            skipped.append(path.parent.name)
            continue
        # Outside the try: a typo'd keyword or missing required field here
        # is a defect in this module, not bad input, and must raise.
        out.append(CaptureSession(
            session_id=session_id,
            folder=str(path.parent),
            swapped=swapped,
            approved=approved,
            scan_verdict=scan_verdict,
            frame_count=frame_count,
        ))
    if skipped:
        logger.warning("loaded %d capture sessions, skipped %d: %s",
                       len(out), len(skipped), ", ".join(skipped))
    return out


def missed_attacks(sessions: list[CaptureSession]) -> list[CaptureSession]:
    """Swapped sessions that were nonetheless approved — the fraud that got through."""
    return [s for s in sessions if s.swapped and s.approved]
