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
from corpora.provenance import corpus_label as corpus_id
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


#: Task 8's per-band table on the capture corpus — the only material in this
#: project that looks like live v-CIP traffic, and so the only thing that can
#: say which bands a shipped curve must cover.
DEFAULT_IN_DOMAIN_BANDS = Path("bench/gend_vcip.json")


def load_in_domain_bands(path: str | Path) -> dict[str, int]:
    """Quality band -> capture frames, from Task 8's matched-encoder table.

    `encoder_matched`, never `uncontrolled`: the uncontrolled half of that
    report is the one a single JPEG header byte labels for 99.2% of rows,
    and `bench/gend_vcip.json`'s own `how_to_read` says it is reported so
    the shortcut is visible, not so it can be cited.
    """
    report = json.loads(Path(path).read_text())
    table = report["encoder_matched"]["per_band_frames"]
    return {band: int(row["n"]) for band, row in table.items()}


def band_coverage(fitted: Sequence[str],
                  frames_by_band: dict[str, int]) -> dict[str, Any]:
    """What a curve fitted on these bands would and would not speak to.

    A band with no curve returns `llr 0.0` and `uncalibrated_for_band` — the
    honest answer, and also a detector that cannot move a verdict. Counting
    the capture frames sitting in those bands is what turns that from a
    footnote into a number a reader can weigh.
    """
    have = set(fitted)
    covered = sorted(b for b in frames_by_band if b in have)
    uncovered = sorted(b for b in frames_by_band if b not in have)
    total = sum(frames_by_band.values())
    n_covered = sum(frames_by_band[b] for b in covered)
    return {
        "covered_bands": covered,
        "uncovered_bands": uncovered,
        "capture_frames_covered": n_covered,
        "capture_frames": total,
        "fraction_covered": (n_covered / total) if total else None,
        "uncovered_frames_by_band": {b: frames_by_band[b] for b in uncovered},
    }


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
    parser.add_argument("--corpus-kind", choices=("df40", "swaps"),
                        default="df40",
                        help="where the curve comes from. `df40` is the "
                             "original source and remains the default so "
                             "every existing invocation is unchanged — but "
                             "this project has ruled that DF40 can neither "
                             "support nor refute anything (every number on "
                             "it sits inside its own permutation null), and "
                             "a calibration curve is a claim about what a "
                             "score MEANS. `swaps` builds the paired, "
                             "identically-encoded FairFace corpus instead.")
    parser.add_argument("--corpus", type=Path,
                        help="df40 only: directory holding fake/ and real/")
    parser.add_argument("--root", type=Path,
                        help="swaps only: the FairFace session directory")
    parser.add_argument("--offset", type=int, default=0,
                        help="swaps only: photographs to skip. MUST be at or "
                             "beyond the training window of any detector "
                             "being calibrated (bench/eval_swaps.py: "
                             "TRAINING_WINDOWS)")
    parser.add_argument("--in-domain-bands", type=Path,
                        default=DEFAULT_IN_DOMAIN_BANDS,
                        help="Task 8's per-band capture table, against which "
                             "every fitted curve's band coverage is reported. "
                             "Absent, the report says so rather than passing "
                             "an unchecked curve off as a checked one.")
    parser.add_argument("--out", default=Path("assets/models/calibration.json"),
                        type=Path)
    parser.add_argument("--report",
                        default=Path("bench/calibration_report.json"), type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--holdout", type=float, default=DEFAULT_HOLDOUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-training-window", action="store_true",
                        help="fit even though --offset falls inside a "
                             "declared training window. For FIXTURES and "
                             "for deliberately measuring the contamination "
                             "-- never for a curve that ships. Mirrors "
                             "bench/eval_swaps.py's flag of the same name.")
    parser.add_argument("--gend-weights", type=Path, default=None,
                        help="override slot D's HEAD path")
    parser.add_argument("--gend-encoder", type=Path, default=None,
                        help="override slot D's ENCODER path, so Phase 2's "
                             "tuned pair is calibrated by the same procedure "
                             "as the frozen one. Pass both or neither.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    # BOTH refusals BEFORE the corpus is built, which takes minutes to hours
    # including face detection over the whole session directory. A refusal
    # that arrives after the work is a refusal the caller learns to route
    # around -- `bench/eval_swaps.py` carries the same rule for the same
    # reason.
    if (args.gend_weights is None) != (args.gend_encoder is None):
        parser.error("--gend-weights and --gend-encoder go together: a head "
                     "fitted jointly with a tuned encoder, read through the "
                     "frozen one, produces a curve that means nothing")

    if args.corpus_kind == "swaps" and not args.allow_training_window:
        # THE HELP TEXT SAID "MUST" AND NOTHING ENFORCED IT, and --offset
        # defaults to 0. The default invocation therefore fitted slot D's
        # score->LLR curve on photographs 0-1,999 -- precisely the window
        # the shipped head 0.1.0-probe-fairface2000 was fitted on. A
        # calibration curve is a claim about what a score MEANS, so one
        # fitted on the model's own training data is over-confident exactly
        # where it is trusted most. bench/eval_gend.py and bench/eval_swaps.py
        # have both refused this since they were written; this did not.
        from bench.eval_swaps import TRAINING_WINDOWS
        window = max(TRAINING_WINDOWS.values())
        if args.offset < window:
            logger.error(
                "--offset %d is inside a declared training window (the "
                "largest is %d, see bench/eval_swaps.py: TRAINING_WINDOWS). "
                "A curve fitted there has already seen its own exam. "
                "Refusing.", args.offset, window)
            return 1

    if args.corpus_kind == "df40":
        if args.corpus is None:
            parser.error("--corpus is required with --corpus-kind df40")
        records, skipped = load_df40_records(args.corpus, limit=args.limit,
                                             seed=args.seed, detect=detect)
        corpus_label = corpus_id(args.corpus)
    else:
        if args.root is None or args.limit is None:
            parser.error("--root and --limit are required with "
                         "--corpus-kind swaps")
        # The same builder the fitter and the LOGO harness use, so the
        # calibration corpus is paired and identically re-encoded like
        # everything else. Its records already carry `image`, `source_id`
        # and `label`, which is exactly what `_score`/`_observe` read.
        from training.fit_gend import build_training_records
        records, skipped = build_training_records(
            args.root, limit=args.limit, offset=args.offset, seed=args.seed,
            detect=detect)
        corpus_label = (f"{corpus_id(args.root)} [offset {args.offset}, "
                        f"limit {args.limit}]")
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

    if registry is None:
        overrides = {k: v for k, v in (
            ("gend_weights", args.gend_weights),
            ("clip_weights", args.gend_encoder)) if v is not None}
        registry = default_registry(**overrides)
    calibrators: dict[str, Calibrator] = {}
    try:
        in_domain = load_in_domain_bands(args.in_domain_bands)
        in_domain_source = str(args.in_domain_bands)
    except (OSError, KeyError, ValueError):
        logger.warning("no in-domain band table at %s; band coverage against "
                       "the capture corpus is NOT checked in this report",
                       args.in_domain_bands)
        in_domain, in_domain_source = {}, None
    report: dict[str, Any] = {
        "corpus_kind": args.corpus_kind,
        "corpus": corpus_label,
        "skipped": skipped,
        "seed": args.seed,
        "split": {"fit_records": len(fit), "holdout_records": len(holdout),
                  "holdout_fraction": args.holdout},
        "in_domain_bands": {
            "source": in_domain_source,
            "what": "encoder_matched.per_band_frames — the controlled half",
            "capture_frames_by_band": in_domain or None,
        },
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
            # Two lists, not one. `fit` drops a band under MIN_FIT_SAMPLES or
            # with one class, so what the split SAW and what the curve HAS
            # differ — and it is the second that decides whether the detector
            # can speak at runtime. `bench/vcip_fit.json` names them the same
            # way for the same reason.
            "bands_fitted": cal.fitted_bands,
            "bands_seen": sorted(set(f_bands)),
            "in_domain_coverage": (band_coverage(cal.fitted_bands, in_domain)
                                   if in_domain else None),
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
