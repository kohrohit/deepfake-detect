"""Composition root for the service: build everything once, wire it, run it.

`dfd.pipeline.decide` takes its registry, calibrators and policy as
arguments precisely so that the choice of what to run is made in ONE place
that a reader can audit. This is that place. Nothing below it constructs a
detector or reads a weight file.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ..audit import AuditRecord
from ..calibration import Calibrator, load_calibrators
from ..detectors.blend import DEFAULT_BLEND_WEIGHTS
from ..detectors.registry import (
    DEFAULT_EFFNET_WEIGHTS,
    DEFAULT_NPR_WEIGHTS,
    default_registry,
)
from ..faces import DEFAULT_MODEL
from ..limits import DEFAULT_LIMITS, Limits
from ..pipeline import decide
from ..policy import DEFAULT_POLICY, Policy
from .api import DEFAULT_MAX_UPLOAD_BYTES, make_server
from .evidence import DEFAULT_AUC_FLOOR, CardError, gated_detectors, load_card
from .store import Store
from .worker import Worker

logger = logging.getLogger(__name__)

DEFAULT_ROOT = Path.home() / ".local/share/dfd"


@dataclass(frozen=True)
class ServiceConfig:
    """Everything the service needs, in one auditable object."""
    host: str = "127.0.0.1"
    port: int = 8077
    db_path: Path = DEFAULT_ROOT / "dfd.sqlite3"
    inbox: Path = DEFAULT_ROOT / "inbox"
    workdir: Path = DEFAULT_ROOT / "work"
    evidence_path: Path = Path("bench/evidence_card.json")
    calibration_path: Path | None = None
    blend_weights: Path = Path(DEFAULT_BLEND_WEIGHTS)
    npr_weights: Path = Path(DEFAULT_NPR_WEIGHTS)
    effnet_weights: Path = Path(DEFAULT_EFFNET_WEIGHTS)
    face_model: Path = Path(DEFAULT_MODEL)
    policy: Policy = DEFAULT_POLICY
    limits: Limits = DEFAULT_LIMITS
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    poll_interval: float = 1.0
    worker_threads: int = 1
    max_frames: int = 32
    #: Minimum measured AUC before a detector may contribute evidence. The
    #: card cannot lower it (see `dfd.service.evidence.gated_detectors`).
    auc_floor: float = DEFAULT_AUC_FLOOR


@dataclass
class Service:
    """A built, not-yet-running service. `start()` then `stop()`."""
    config: ServiceConfig
    store: Store
    worker: Worker
    registry: object
    calibrators: dict[str, Calibrator]
    httpd: object
    warnings: list[str] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _threads: list[threading.Thread] = field(default_factory=list)
    _serving: threading.Thread | None = None

    @property
    def port(self) -> int:
        """The bound port. Meaningful after construction, because the socket
        is opened by `make_server`, not by `start` — so a test asking for
        port 0 can discover what it actually got."""
        return int(self.httpd.server_address[1])  # type: ignore[attr-defined]

    def start(self) -> None:
        """Requeue orphans, start the workers, then serve."""
        recovered = self.worker.recover()
        if recovered:
            self.warnings.append(
                f"requeued {recovered} submission(s) left running by a "
                "previous process")
        for i in range(self.config.worker_threads):
            t = threading.Thread(target=self.worker.run_until_stopped,
                                 args=(self._stop,), name=f"dfd-worker-{i}",
                                 daemon=True)
            t.start()
            self._threads.append(t)
        self._serving = threading.Thread(
            target=self.httpd.serve_forever,  # type: ignore[attr-defined]
            name="dfd-http", daemon=True)
        self._serving.start()
        logger.info("dfd service listening on http://%s:%d",
                    self.config.host, self.port)

    def stop(self) -> None:
        """Stop serving and drain the worker threads. Safe if never started."""
        self._stop.set()
        if self._serving is not None:
            # ONLY when serve_forever actually ran. `BaseServer.shutdown`
            # waits on an event that only `serve_forever` sets, so calling it
            # on a server that was built and never served blocks forever — a
            # hang in the shutdown path, which is the worst place for one.
            self.httpd.shutdown()  # type: ignore[attr-defined]
        for t in self._threads:
            t.join(timeout=10)
        if self._serving is not None:
            self._serving.join(timeout=10)
        self.close()

    def close(self) -> None:
        """Release the listening socket without having served."""
        self.httpd.server_close()  # type: ignore[attr-defined]


def build_service(config: ServiceConfig) -> Service:
    """Wire the whole service. Does not start it.

    Raises:
        ValueError: if a calibration file is present but unreadable by this
            code. Starting anyway would give every sample zero evidence and
            the verdict `insufficient_evidence`, which is indistinguishable
            from an honest abstention and would hide the misconfiguration
            for as long as nobody looked.
    """
    warnings: list[str] = []

    registry = default_registry(npr_weights=config.npr_weights,
                                effnet_weights=config.effnet_weights,
                                blend_weights=config.blend_weights)
    for name in registry.names():
        version = registry.get(name).version
        logger.info("detector %s version %s", name, version)

    # THE EVIDENCE GATE. Computed before calibration is loaded, because what
    # it gates IS the calibration: a detector with no curve returns llr 0.0
    # and cannot move a verdict, while its raw score still reaches the audit
    # record. See dfd/service/evidence.py for why this fails closed.
    allowed: set[str] = set()
    try:
        card = load_card(config.evidence_path)
        allowed = gated_detectors(card, floor=config.auc_floor)
        for name in sorted(set(registry.names()) - allowed):
            entry = card["detectors"].get(name, {})
            auc = entry.get("auc")
            warnings.append(
                f"detector {name} is gated out of deciding: measured AUC "
                f"{auc if auc is not None else 'never measured'} is below the "
                f"{config.auc_floor} floor. Its raw score is still recorded.")
    except CardError as exc:
        warnings.append(
            f"{exc}. No detector can be calibrated without measured "
            "performance, so every verdict will be insufficient_evidence.")

    calibrators: dict[str, Calibrator] = {}
    if config.calibration_path is None:
        warnings.append(
            "no calibration file configured: every detector will abstain "
            "with uncalibrated_for_band and every verdict will be "
            "insufficient_evidence")
    elif not Path(config.calibration_path).is_file():
        # NOT silent. A mistyped path and a never-calibrated deployment
        # produce identical behaviour, and only this message tells them apart.
        warnings.append(
            f"calibration file {config.calibration_path} does not exist: "
            "running uncalibrated, so every verdict will be "
            "insufficient_evidence")
    else:
        loaded = load_calibrators(config.calibration_path)
        calibrators = {k: v for k, v in loaded.items() if k in allowed}
        for name in sorted(set(loaded) - allowed):
            # A calibration file can carry a curve for a detector the gate
            # refuses. Dropping it here rather than trusting the file is the
            # whole point: the file is produced by whoever ran the fitter,
            # and the gate is produced by what was measured.
            warnings.append(
                f"discarded the calibration curve for {name}: it has not "
                "been measured above the evidence floor")

    if not Path(config.face_model).is_file():
        warnings.append(
            f"face detector weights {config.face_model} are absent: no face "
            "will be found, so nothing will be scored")
    if not Path(config.blend_weights).is_file():
        warnings.append(
            f"blend_seam weights {config.blend_weights} are absent: that "
            "detector will abstain with weights_absent")

    store = Store(config.db_path)

    def scorer(path: Path) -> AuditRecord:
        # A named function rather than a `functools.partial`: partial erases
        # the call signature to (*args, **kwargs), which no longer satisfies
        # the worker's `Callable[[Path], _Record]` and hides a genuine
        # mismatch from the type checker.
        return decide(path, registry=registry, calibrators=calibrators,
                      policy=config.policy, limits=config.limits,
                      face_model=config.face_model,
                      max_frames=config.max_frames)
    worker = Worker(store=store, decide=scorer, inbox=config.inbox,
                    workdir=config.workdir,
                    poll_interval=config.poll_interval)
    httpd = make_server(config.host, config.port, store=store, worker=worker,
                        evidence_path=config.evidence_path,
                        max_upload_bytes=config.max_upload_bytes)

    for w in warnings:
        logger.warning("%s", w)
    return Service(config=config, store=store, worker=worker,
                   registry=registry, calibrators=calibrators, httpd=httpd,
                   warnings=warnings)
