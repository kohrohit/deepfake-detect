"""Benchmark orchestration (spec §8, acceptance criteria 1, 5, 10).

Guards are enforced by default and can only be waived explicitly, because a
benchmark that is easy to run dirty will be run dirty — and every hygiene
failure flatters the result.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from dfd.quality import measure_quality
from dfd.types import Observation

from .guards import (
    IdentityReport, check_compression_coverage, check_threshold_provenance,
    check_uniform_preprocessing, check_video_level,
)
from .metrics import auc, bootstrap_ci_by_group, ece, tpr_at_fpr
from .protocol import logo_splits

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    seed: int = 0
    enforce_guards: bool = True
    fpr_targets: tuple[float, ...] = (0.01, 0.001)
    bootstrap_n: int = 200
    threshold_source: str = "validation"


@dataclass(frozen=True)
class DetectorResult:
    detector: str
    auc: float
    auc_ci: tuple[float, float]
    tpr_at_1pct: float
    tpr_at_0p1pct: float
    ece: float
    adversarial_tpr_at_1pct: float | None
    abstention_rate: float
    p95_latency_ms: float
    n_samples: int


@dataclass(frozen=True)
class RunRecord:
    seed: int
    dataset_hash: str
    guards_enforced: bool
    model_versions: dict[str, str]
    identity_report: IdentityReport | None
    #: Whole-corpus metrics. THIS IS IN-DATASET PERFORMANCE, which measures
    #: memorisation, not field performance. Never report it as the headline.
    detector_results: dict[str, DetectorResult] = field(default_factory=dict)
    #: held-out generator -> detector -> metrics. Spec 8.1: the only number
    #: that predicts field performance. Empty when the corpus cannot be split.
    logo_results: dict[str, dict[str, DetectorResult]] = field(
        default_factory=dict)


def dataset_hash(records: list[dict]) -> str:
    """Content hash over identifying fields — stable, order-independent."""
    keys = sorted(
        json.dumps({k: r.get(k) for k in
                    ("sample_id", "subject_id", "source_id", "generator",
                     "label", "compression")},
                   sort_keys=True)
        for r in records)
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def _observation(record: dict) -> Observation:
    img = record["image"]
    h, w = img.shape[:2]
    lm = np.array([[w * 0.35, h * 0.4], [w * 0.65, h * 0.4]])
    q = measure_quality(img, (0, 0, w, h), lm)
    return Observation(t=0.0, payload=img, roi=(0, 0, w, h), quality=q,
                       source_id=record["source_id"])


def run_benchmark(records: list[dict], registry, config: RunConfig) -> RunRecord:
    if config.enforce_guards:
        # `groups` MUST identify the SOURCE VIDEO, never the sample id.
        # Passing sample_ids for both arguments makes this guard vacuous: its
        # only failure condition is groups[i] != sample_ids[i], so identical
        # lists can never raise. Records must carry a source-video field; for
        # image records each image is its own source, which must be recorded
        # explicitly rather than aliased to sample_id.
        check_video_level([r["sample_id"] for r in records],
                          [r["source_id"] for r in records])
        check_compression_coverage(records)
        check_uniform_preprocessing(records)
        check_threshold_provenance(config.threshold_source)

    labels = np.array([r["label"] for r in records], dtype=int)
    # The SOURCE video, not the sample id. bootstrap_ci_by_group resamples
    # over these, and resampling over frames rather than videos fabricates
    # precision: measured at 11.9x too narrow (group CI 0.751 vs row 0.063).
    # check_video_level enforces a 1:1 mapping while guards are on, but
    # enforce_guards=False is a supported path and that is exactly where an
    # honest interval matters most.
    groups = np.array([r["source_id"] for r in records])
    observations = [_observation(r) for r in records]

    results: dict[str, DetectorResult] = {}
    versions: dict[str, str] = {}
    scores_by_detector: dict[str, np.ndarray] = {}

    for name in registry.names():
        det = registry.get(name)
        versions[name] = det.version

        scores: list[float] = []
        abstentions = 0
        latencies: list[float] = []

        for obs in observations:
            t0 = time.perf_counter()
            raw = det.score([obs])
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if raw.abstained or raw.score is None:
                abstentions += 1
                scores.append(np.nan)
            else:
                scores.append(float(raw.score))

        s = np.array(scores, dtype=float)
        scores_by_detector[name] = s
        results[name] = _detector_result(
            name, s, labels, groups, latencies, abstentions, config)

    logo_results = _logo_results(records, registry, scores_by_detector,
                                 labels, groups, config)

    return RunRecord(seed=config.seed, dataset_hash=dataset_hash(records),
                     guards_enforced=config.enforce_guards,
                     model_versions=versions, identity_report=None,
                     detector_results=results, logo_results=logo_results)


def _detector_result(name, s, labels, groups, latencies, abstentions,
                     config) -> DetectorResult:
    """Metrics for one detector over one set of rows.

    Split out so a LOGO fold can reuse it verbatim: the fold differs only in
    which rows it passes, never in how the numbers are computed.
    """
    n = len(s)
    valid = np.isfinite(s)
    base = dict(
        detector=name,
        adversarial_tpr_at_1pct=None,
        abstention_rate=abstentions / max(1, n),
        p95_latency_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
        n_samples=n,
    )
    if valid.sum() == 0 or len(np.unique(labels[valid])) < 2:
        nan = float("nan")
        return DetectorResult(auc=nan, auc_ci=(nan, nan), tpr_at_1pct=nan,
                              tpr_at_0p1pct=nan, ece=nan, **base)
    return DetectorResult(
        auc=auc(s[valid], labels[valid]),
        auc_ci=bootstrap_ci_by_group(s[valid], labels[valid], groups[valid],
                                     auc, n=config.bootstrap_n,
                                     seed=config.seed),
        tpr_at_1pct=tpr_at_fpr(s[valid], labels[valid], 0.01),
        tpr_at_0p1pct=tpr_at_fpr(s[valid], labels[valid], 0.001),
        ece=ece(s[valid], labels[valid]),
        **base,
    )


def _logo_results(records, registry, scores_by_detector, labels, groups,
                  config) -> dict[str, dict[str, DetectorResult]]:
    """Per-held-out-generator metrics — spec 8.1, the number that predicts field
    performance.

    Detection is not re-run per fold: a detector's score for a record does not
    depend on which fold the record lands in, so folds slice the scores already
    computed.

    HONEST SCOPE. A full LOGO protocol trains on the fold's train side and
    tests on the held-out one. P0 detectors are not trained here, so what this
    computes is evaluation on the held-out generator's test rows. That is the
    number you report, and it becomes the full protocol once training or
    calibration fitting exists — at which point the fold's train side is also
    where the operating threshold must be frozen (spec 8.2 guard 5).
    """
    try:
        splits = logo_splits(records, seed=config.seed)
    except ValueError as exc:
        # A corpus with one generator, one subject, or no measurable fold.
        # Recorded rather than raised: the in-dataset numbers are still valid.
        logger.warning("LOGO unavailable for this corpus: %s", exc)
        return {}

    position = {r["sample_id"]: i for i, r in enumerate(records)}
    out: dict[str, dict[str, DetectorResult]] = {}
    for split in splits:
        rows = np.array([position[sid] for sid in split.test_ids()], dtype=int)
        out[split.held_out_generator] = {
            name: _detector_result(name, scores_by_detector[name][rows],
                                   labels[rows], groups[rows], [], 0, config)
            for name in registry.names()
        }
    return out


def worst_logo_auc(record: "RunRecord", detector: str) -> float:
    """The weakest held-out generator for one detector.

    Reported in preference to the mean, for the same reason spec 8.2 guard 3
    reports the worst compression cell: an average over generators hides the
    one an attacker will actually use.
    """
    folds = [f[detector].auc for f in record.logo_results.values()
             if detector in f]
    finite = [a for a in folds if a == a]
    return min(finite) if finite else float("nan")
