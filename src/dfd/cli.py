"""Command-line entry point (spec §5.1).

Deliberately thin: argument parsing, the two output streams, and exit codes.
Every decision belongs to `pipeline.decide`.

Output is written with `sys.stdout.write`, not `print`. `src/` is under a CI
gate that forbids `print(` (structured logging, never printing), and a CLI's
record on stdout is its product rather than a log line.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence

from .audit import AuditRecord, record_digest
from .detectors.registry import default_registry
from .errors import DfdError
from .faces import DEFAULT_MODEL
from .ingest.video import DEFAULT_MAX_FRAMES
from .pipeline import decide
from .types import Context

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dfd",
        description="Score a file for manipulation and emit an audit record.")
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser(
        "score", help="score one image or video and print its audit record")
    score.add_argument("path", help="image or video file to score")
    score.add_argument("--subject-id", default=None,
                       help="subject identifier recorded in the sample context")
    score.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
                       help=f"frames to sample from a video (default {DEFAULT_MAX_FRAMES})")
    score.add_argument("--seed", type=int, default=0,
                       help="frame-selection seed for reproducible video sampling")
    score.add_argument("--face-model", default=str(DEFAULT_MODEL),
                       help="path to the face detector weights")
    score.add_argument("--pretty", action="store_true",
                       help="indent the JSON for reading; the digest is always "
                            "taken over the canonical form")
    return parser


def _summary(record: AuditRecord) -> str:
    """One line for a human, on stderr, while stdout stays machine-readable."""
    contributing = sum(1 for row in record.evidence if not row["abstained"])
    return (f"verdict={record.verdict} llr={record.llr_total:.2f} "
            f"band={record.quality_band} "
            f"contributing={contributing}/{len(record.evidence)} "
            f"faces={record.stage_reasons.get('faces', 'n/a')} "
            f"digest={record_digest(record)[:8]}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: command-line arguments, excluding the program name. Defaults
            to `sys.argv[1:]` when None.

    Returns:
        0 when a decision was produced — any verdict, INSUFFICIENT_EVIDENCE
        included. 2 when the input was refused. An unexpected failure is not
        caught here, so the interpreter exits 1 with its traceback: a bug
        should look like a bug, not like a rejected file.

        The verdict is deliberately absent from the exit status. `if dfd score
        f` would otherwise read a REAL verdict as failure, and `set -e` would
        abort a script on a correct answer. Callers branch on the JSON.

    Raises:
        DfdError: never — caught here and converted to exit code 2. Any
            other exception propagates, producing exit code 1.
    """
    args = _build_parser().parse_args(argv)
    try:
        record = decide(
            args.path,
            registry=default_registry(),
            context=Context(subject_id=args.subject_id),
            face_model=args.face_model,
            max_frames=args.max_frames,
            seed=args.seed,
        )
    except DfdError as exc:
        sys.stderr.write(f"dfd: {exc}\n")
        return 2

    canonical = record.to_json()
    body = (json.dumps(json.loads(canonical), indent=2, sort_keys=True)
            if args.pretty else canonical)
    sys.stdout.write(body + "\n")
    sys.stderr.write(_summary(record) + "\n")
    return 0
