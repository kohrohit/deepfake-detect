"""Loader for the 442-session v-CIP capture corpus (spec §1.1).

Five of these sessions are swapped=true and approved=true. They are the actual
fraud, and any candidate system must be measured against them specifically —
aggregate accuracy over 442 sessions would hide all five.
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
        except (json.JSONDecodeError, OSError) as exc:
            # Never silent: a dropped session may be one of the five that are
            # the actual fraud, and aggregate counts would not reveal it.
            logger.warning("skipping unreadable session %s: %s",
                           path.parent.name, exc)
            skipped.append(path.parent.name)
            continue
        out.append(CaptureSession(
            session_id=d.get("session_id", path.parent.name),
            folder=str(path.parent),
            swapped=bool(d.get("swapped", False)),
            approved=bool((d.get("decision") or {}).get("approved", False)),
            scan_verdict=(d.get("scan") or {}).get("verdict"),
            frame_count=int(d.get("frame_count", 0)),
        ))
    if skipped:
        logger.warning("loaded %d capture sessions, skipped %d: %s",
                       len(out), len(skipped), ", ".join(skipped))
    return out


def missed_attacks(sessions: list[CaptureSession]) -> list[CaptureSession]:
    """Swapped sessions that were nonetheless approved — the fraud that got through."""
    return [s for s in sessions if s.swapped and s.approved]
