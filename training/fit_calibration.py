"""Fit per-band calibration curves. Not part of the installed package.

    python3 -m training.fit_calibration \\
        --corpus ~/Desktop/agents/datasets/df40_eval/extracted/val \\
        --out assets/models/calibration.json

Spec §8.2 guard 5: operating points are frozen on validation, never chosen
on test. That discipline is structural here — the corpus is split by SOURCE
before anything is fitted, the curves see only the fit side, and every number
in the report comes from the holdout side. A calibration fitted and reported
on the same rows is a curve that has already seen its own exam.

Fitting a curve does NOT license a detector to decide. That is the evidence
gate's job (`dfd/service/evidence.py`), and it reads measured cross-corpus
AUC, not this file. The two are deliberately separate: this answers "how do
I turn this detector's score into nats", the gate answers "may this detector
speak at all".
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bench.metrics import auc, ece
from corpora.df40 import load_df40_records
from corpora.face_pool import DetectFn
from dfd.calibration import Calibrator, save_calibrators
from dfd.detectors.registry import default_registry
from dfd.faces import detect_faces
from dfd.quality import measure_quality
from dfd.types import Observation

logger = logging.getLogger(__name__)

DEFAULT_HOLDOUT = 0.3


def split_by_source(records: Sequence[dict[str, Any]], *,
                    holdout_fraction: float = DEFAULT_HOLDOUT,
                    seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Split records into (fit, holdout), whole sources on one side only.

    Sources, never rows: frames of one video share everything a calibration
    curve could memorise, and splitting by row puts the same face on both
    sides — which makes the reported holdout AUC a measurement of nothing.
    """
    sources = sorted({r["source_id"] for r in records})
    rng = np.random.default_rng(seed)
    rng.shuffle(sources)
    n_holdout = max(1, int(round(len(sources) * holdout_fraction)))
    holdout_ids = set(sources[:n_holdout])
    fit = [r for r in records if r["source_id"] not in holdout_ids]
    holdout = [r for r in records if r["source_id"] in holdout_ids]
    return fit, holdout


def _observe(record: dict[str, Any]) -> Observation:
    img = record["image"]
    h, w = img.shape[:2]
    lm = np.array([[w * 0.35, h * 0.4], [w * 0.65, h * 0.4]])
    return Observation(t=0.0, payload=img, roi=(0, 0, w, h),
                       quality=measure_quality(img, (0, 0, w, h), lm),
                       source_id=record["source_id"])


def _score(records: Sequence[dict[str, Any]], detector: Any
           ) -> tuple[list[float], list[int], list[str]]:
    """Scores, labels and bands for the rows this detector could score."""
    scores: list[float] = []
    labels: list[int] = []
    bands: list[str] = []
    for record in records:
        obs = _observe(record)
        raw = detector.score([obs])
        if raw.abstained or raw.score is None:
            continue
        scores.append(float(raw.score))
        labels.append(int(record["label"]))
        bands.append(obs.quality.band)
    return scores, labels, bands


def main(argv: Sequence[str] | None = None, *, registry: Any = None,
         detect: DetectFn = detect_faces) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path,
                        help="directory holding fake/ and real/")
    parser.add_argument("--out", default=Path("assets/models/calibration.json"),
                        type=Path)
    parser.add_argument("--report",
                        default=Path("bench/calibration_report.json"), type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--holdout", type=float, default=DEFAULT_HOLDOUT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    records, skipped = load_df40_records(args.corpus, limit=args.limit,
                                         seed=args.seed, detect=detect)
    labels = {r["label"] for r in records}
    if len(labels) < 2:
        # One label cannot fit a curve, and `LogisticRegression` would raise
        # deep inside sklearn rather than here where the cause is legible.
        logger.error("corpus has only label(s) %s; calibration needs both",
                     sorted(labels))
        raise SystemExit(1)

    fit, holdout = split_by_source(records, holdout_fraction=args.holdout,
                                   seed=args.seed)
    logger.info("%d records (skipped %s): %d fit / %d holdout over %d sources",
                len(records), skipped or "nothing", len(fit), len(holdout),
                len({r["source_id"] for r in records}))

    registry = registry if registry is not None else default_registry()
    calibrators: dict[str, Calibrator] = {}
    report: dict[str, Any] = {
        "corpus": str(args.corpus),
        "seed": args.seed,
        "split": {"fit_records": len(fit), "holdout_records": len(holdout),
                  "holdout_fraction": args.holdout},
        "detectors": {},
    }

    for name in registry.names():
        detector = registry.get(name)
        f_scores, f_labels, f_bands = _score(fit, detector)
        if len(set(f_labels)) < 2:
            # Either the detector abstained on everything, or it only ever
            # saw one class. A curve fitted on that is a fabricated licence
            # to decide, so none is written.
            logger.warning("%s: no fittable rows on the fit split, skipping",
                           name)
            report["detectors"][name] = {
                "fitted_on": None, "fit_rows": len(f_scores),
                "holdout_auc": None, "holdout_ece": None,
                "note": "abstained on everything, or only one label reached it"}
            continue

        cal = Calibrator(name).fit(np.array(f_scores), np.array(f_labels),
                                   np.array(f_bands))
        calibrators[name] = cal

        h_scores, h_labels, _ = _score(holdout, detector)
        measurable = len(set(h_labels)) > 1
        report["detectors"][name] = {
            # Named rather than implied: the one question a reader of a
            # calibration report asks is which rows the curve has already
            # seen, and "fit split" is the only answer that is not a guess.
            "fitted_on": "fit split",
            "fit_rows": len(f_scores),
            "holdout_rows": len(h_scores),
            "holdout_auc": (float(auc(np.array(h_scores), np.array(h_labels)))
                            if measurable else None),
            "holdout_ece": (float(ece(np.array(h_scores), np.array(h_labels)))
                            if measurable else None),
            "bands": sorted(set(f_bands)),
        }
        logger.info("%s: fitted on %d rows, holdout AUC %s", name,
                    len(f_scores), report["detectors"][name]["holdout_auc"])

    save_calibrators(calibrators, args.out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True,
                                      allow_nan=False))
    logger.info("wrote %s and %s", args.out, args.report)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
