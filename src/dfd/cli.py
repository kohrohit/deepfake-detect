"""Command-line entry point (spec §5.1).

Deliberately thin: argument parsing, the two output streams, and exit codes.
Every decision belongs to `pipeline.decide`.

Output is written with `sys.stdout.write`, not `print`. `src/` is under a CI
gate that forbids `print(` (structured logging, never printing), and a CLI's
record on stdout is its product rather than a log line.

This is also the only entry point, so it is the only place that can configure
logging. Without `basicConfig` every module's `logger` fell through to Python's
`lastResort` handler, which writes bare unformatted messages to stderr at
WARNING and drops everything below — so `faces.py`'s "Face detector weights
absent at ..." appeared with no level and no logger name, and `decide`'s own
`logger.debug`/`logger.info` diagnostics were unreachable from any shell.
Log records go to stderr; the record stays alone on stdout.
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

#: Verbosity ladder for `-v`. Index by the `-v` count, clamped to the end.
#: WARNING by default so the stderr contract stays near-silent for scripts;
#: `-v` opens `decide`'s stage-level INFO, `-vv` its per-detector DEBUG timings.
LOG_LEVELS = (logging.WARNING, logging.INFO, logging.DEBUG)

#: Level and logger name are part of the point: an unformatted line cannot be
#: filtered, attributed, or distinguished from the summary line.
LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"


def _positive_int(value: str) -> int:
    """Parse an argument that must be at least 1.

    `--max-frames 0` otherwise reaches `load_video`, which raises its
    zero-frame `ValueError`, which `_ingest` relabels `could not decode
    <file>` — telling a user who mistyped a flag that their video is corrupt.

    Args:
        value: the raw argument string.

    Returns:
        The parsed integer.

    Raises:
        ArgumentTypeError: if `value` is not an integer >= 1. This is
            argparse's own contract for a `type=` callable (it renders the
            message as a usage error and exits 2), so it is deliberately not
            a `DfdError`: nothing has been ingested yet, and no `DfdError`
            would be caught by `main`'s handler at parse time anyway.
    """
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {number}")
    return number


def _configure_logging(verbosity: int) -> None:
    """Send log records to stderr, formatted, at the requested level.

    Args:
        verbosity: count of `-v` flags; clamped to the end of `LOG_LEVELS`.

    Raises:
        No exceptions of its own. `basicConfig` is a no-op when the root
        logger already has handlers (an embedding host, or pytest), which is
        correct: a library caller's logging configuration wins over ours.
    """
    logging.basicConfig(level=LOG_LEVELS[min(verbosity, len(LOG_LEVELS) - 1)],
                        stream=sys.stderr, format=LOG_FORMAT)


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
    score.add_argument("--max-frames", type=_positive_int, default=DEFAULT_MAX_FRAMES,
                       help=f"frames to sample from a video, at least 1 "
                            f"(default {DEFAULT_MAX_FRAMES})")
    score.add_argument("--seed", type=int, default=0,
                       help="frame-selection seed for reproducible video sampling")
    score.add_argument("--face-model", default=str(DEFAULT_MODEL),
                       help="path to the face detector weights")
    score.add_argument("-v", "--verbose", action="count", default=0,
                       help="log to stderr at INFO; repeat (-vv) for DEBUG. "
                            "stdout carries the record either way")
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

        stderr carries the one-line summary, plus any log record at or above
        the configured level (WARNING by default, INFO at `-v`, DEBUG at
        `-vv`). stdout carries the record and nothing else, whatever the
        verbosity.

    Raises:
        DfdError: never — caught here and converted to exit code 2. Any
            other exception propagates, producing exit code 1.
        SystemExit: from `argparse` on a malformed argument (including
            `--max-frames` below 1), with the usage message on stderr and
            exit code 2 — the same code a refused input produces.
    """
    args = _build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    registry = default_registry()
    logger.debug("scoring %s with detectors %s", args.path, registry.names())
    try:
        record = decide(
            args.path,
            registry=registry,
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
