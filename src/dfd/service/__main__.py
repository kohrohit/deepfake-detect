"""`python3 -m dfd.service` — run the service until told to stop.

Every path is a flag with a default under ~/.local/share/dfd, so a first run
needs no configuration and no root. Signals are handled explicitly: SIGTERM
is what systemd sends, and a service that ignores it is killed mid-decision
and leaves a submission stuck in `running` — which is why `Service.start`
requeues those on the way back up.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from collections.abc import Sequence
from pathlib import Path

from ..policy import DEFAULT_POLICY, Policy
from .app import DEFAULT_ROOT, ServiceConfig, build_service

logger = logging.getLogger("dfd.service")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m dfd.service",
        description="Run the deepfake-detection service: watch a folder, "
                    "score what lands in it, serve the results.")
    p.add_argument("--host", default="127.0.0.1",
                   help="interface to bind (default: %(default)s — there is "
                        "no authentication, see dfd.service.api)")
    p.add_argument("--port", type=int, default=8077)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                   help="base directory for the database, inbox and workdir "
                        "(default: %(default)s)")
    p.add_argument("--inbox", type=Path, default=None,
                   help="watched directory (default: <root>/inbox)")
    p.add_argument("--workdir", type=Path, default=None,
                   help="where taken files are kept (default: <root>/work)")
    p.add_argument("--db", type=Path, default=None,
                   help="SQLite database (default: <root>/dfd.sqlite3)")
    p.add_argument("--calibration", type=Path, default=None,
                   help="calibration JSON from training.fit_calibration. "
                        "Without it every verdict is insufficient_evidence.")
    p.add_argument("--evidence", type=Path,
                   default=Path("bench/evidence_card.json"),
                   help="measured detector report card shown on the dashboard")
    p.add_argument("--blend-weights", type=Path,
                   default=Path("assets/models/blend_seam.npz"))
    p.add_argument("--face-model", type=Path,
                   default=Path("assets/models/face_detection_yunet_2023mar.onnx"))
    p.add_argument("--workers", type=int, default=1,
                   help="worker threads (default: %(default)s)")
    p.add_argument("--poll-interval", type=float, default=1.0,
                   help="seconds between inbox scans (default: %(default)s)")
    p.add_argument("--retain-days", type=int, default=30,
                   help="delete scored files older than this many days; 0 "
                        "keeps everything. Audit records are never deleted "
                        "(default: %(default)s)")
    p.add_argument("--max-frames", type=int, default=32,
                   help="frames sampled per video (default: %(default)s)")
    p.add_argument("--fake-threshold", type=float,
                   default=DEFAULT_POLICY.fake_threshold)
    p.add_argument("--real-threshold", type=float,
                   default=DEFAULT_POLICY.real_threshold)
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p


def config_from_args(args: argparse.Namespace) -> ServiceConfig:
    """Turn parsed flags into the frozen config the service is built from."""
    root = Path(args.root)
    return ServiceConfig(
        host=args.host,
        port=args.port,
        db_path=args.db or root / "dfd.sqlite3",
        inbox=args.inbox or root / "inbox",
        workdir=args.workdir or root / "work",
        evidence_path=args.evidence,
        calibration_path=args.calibration,
        blend_weights=args.blend_weights,
        face_model=args.face_model,
        policy=Policy(fake_threshold=args.fake_threshold,
                      real_threshold=args.real_threshold),
        poll_interval=args.poll_interval,
        worker_threads=max(1, args.workers),
        max_frames=args.max_frames,
        retain_days=args.retain_days,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose >= 2 else
        logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # The service's own logger stays at INFO even without -v: the startup
    # banner, the warnings about missing weights, and one line per decision
    # are the minimum a running service owes its operator.
    logging.getLogger("dfd.service").setLevel(logging.INFO)
    logging.getLogger("dfd.service.worker").setLevel(logging.INFO)

    service = build_service(config_from_args(args))
    service.start()

    # stderr, not stdout, and not `print`: `src/` is under a CI gate that
    # bans print (tests/test_ci_gates.py), and under systemd this banner is
    # what `journalctl --user -u dfd` shows on every start — which is the
    # only place the missing-weights warnings are ever read.
    banner = [
        f"dfd service on http://{args.host}:{service.port}",
        f"  inbox:  {service.config.inbox}",
        f"  db:     {service.config.db_path}",
        *(f"  WARN:   {w}" for w in service.warnings),
    ]
    sys.stderr.write("\n".join(banner) + "\n")
    sys.stderr.flush()

    stop = threading.Event()

    def _handle(signum: int, _frame: object) -> None:
        logger.info("received %s, shutting down", signal.Signals(signum).name)
        stop.set()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)
    try:
        stop.wait()
    finally:
        service.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
