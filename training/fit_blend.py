"""Fit the blend-seam model. Not part of the installed package: fitting is
not inference, and nothing in src/dfd/ may depend on this module.

Run:
    python3 -m training.fit_blend \\
        --captures /path/to/captures \\
        --out assets/models/blend_seam.npz
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from corpora.captures import load_capture_sessions
from corpora.face_pool import build_face_pool
from corpora.sbi import build_sbi_corpus
from dfd.detectors.blend import FEATURE_NAMES, BlendModel, save_blend_model, seam_features
from dfd.types import Sample

logger = logging.getLogger(__name__)

#: Fraction of SUBJECTS held out. Subjects, never rows: a real crop and the
#: pseudo-fake made from it share a face, so splitting by row would put the
#: same face on both sides and the reported AUC would be meaningless.
DEFAULT_HOLDOUT = 0.3


def split_by_subject(
    samples: Sequence[Sample],
    *,
    holdout_fraction: float = DEFAULT_HOLDOUT,
    seed: int = 0,
) -> tuple[list[Sample], list[Sample]]:
    """Split samples into (train, test) with no subject on both sides.

    Args:
        samples: the corpus.
        holdout_fraction: share of subjects placed in test.
        seed: reproducibility.

    Returns:
        A (train, test) pair.
    """
    subjects = sorted({s.context.subject_id for s in samples
                       if s.context.subject_id is not None})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(subjects))
    n_test = max(1, int(round(len(subjects) * holdout_fraction)))
    test_subjects = {subjects[i] for i in order[:n_test]}

    train = [s for s in samples if s.context.subject_id not in test_subjects]
    test = [s for s in samples if s.context.subject_id in test_subjects]
    logger.info("split: %d train / %d test rows over %d subjects",
                len(train), len(test), len(subjects))
    return train, test


def _matrix(samples: Sequence[Sample]) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([seam_features(s.observations[0].payload) for s in samples])
    y = np.asarray([s.context.label for s in samples], dtype=np.int64)
    return x, y


def fit_blend_model(train: Sequence[Sample], *, version: str) -> BlendModel:
    """Fit the standardiser and logistic model, and keep only the numbers.

    Args:
        train: training samples. Must contain both labels.
        version: version string recorded in the model file.

    Returns:
        The fitted model.

    Raises:
        ValueError: if `train` does not contain both labels — a one-class fit
            produces a model that scores everything the same and reports no
            error at all.
    """
    x, y = _matrix(train)
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("training corpus must contain both labels, got "
                         f"{sorted(set(y.tolist()))}")

    mean = x.mean(axis=0).astype(np.float64)
    scale = x.std(axis=0).astype(np.float64)
    # A constant feature has zero spread. Dividing by it yields inf; replacing
    # the divisor with 1.0 leaves the feature at zero, which is exactly as
    # informative as it actually is.
    scale = np.where(scale == 0.0, 1.0, scale)

    z = (x.astype(np.float64) - mean) / scale
    clf = LogisticRegression(max_iter=1000)
    clf.fit(z, y)

    return BlendModel(mean=mean, scale=scale,
                      coef=np.asarray(clf.coef_[0], dtype=np.float64),
                      intercept=float(clf.intercept_[0]),
                      feature_names=FEATURE_NAMES, version=version)


def evaluate(model: BlendModel, samples: Sequence[Sample]) -> dict[str, float]:
    """Score a held-out set.

    Returns a dict carrying `auc`, `n` and `n_fake`. The fake count is
    reported beside the AUC deliberately: an AUC over a handful of positives
    is a different claim from the same number over hundreds, and a report
    that hides the denominator invites the reader to confuse them.
    """
    from sklearn.metrics import roc_auc_score

    if not samples:
        return {"auc": float("nan"), "n": 0.0, "n_fake": 0.0}
    x, y = _matrix(samples)
    probs = np.asarray([model.predict_proba(row.astype(np.float32)) for row in x])
    n_fake = float((y == 1).sum())
    auc = (float(roc_auc_score(y, probs))
           if len(set(y.tolist())) == 2 else float("nan"))
    return {"auc": auc, "n": float(len(samples)), "n_fake": n_fake}


def main(argv: Sequence[str] | None = None) -> int:
    """Build the corpus, fit, evaluate, write the model and a JSON report."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", required=True, type=Path)
    parser.add_argument("--out", default=Path("assets/models/blend_seam.npz"),
                        type=Path)
    parser.add_argument("--report", default=Path("bench/blend_seam_report.json"),
                        type=Path)
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    sessions = load_capture_sessions(args.captures)
    genuine = [s for s in sessions if not s.swapped]
    logger.info("%d sessions, %d genuine and usable for training",
                len(sessions), len(genuine))

    crops, skipped = build_face_pool(genuine)
    if not crops:
        # Name both causes. The skip tally distinguishes them — NO_FACE means
        # the detector ran and found nothing (or has no weights), NO_FRAMES
        # means the folders held no frame_NN.jpg at all — and a message that
        # guesses one cause sends the reader past the tally that answers it.
        logger.error("no face crops extracted; skip tally: %s. NO_FACE means "
                     "the detector returned nothing (check the YuNet weights at "
                     "assets/models/face_detection_yunet_2023mar.onnx); "
                     "NO_FRAMES means the session folders held no frames.",
                     skipped)
        return 1

    samples = build_sbi_corpus(crops, seed=args.seed)
    train, test = split_by_subject(samples, seed=args.seed)
    model = fit_blend_model(train, version=args.version)
    metrics = evaluate(model, test)

    save_blend_model(model, args.out)
    report = {"version": args.version, "seed": args.seed,
              "n_sessions": len(sessions), "n_genuine": len(genuine),
              "n_crops": len(crops), "skipped": skipped,
              "n_train": len(train), "n_test": len(test), **metrics}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True))
    logger.info("held-out AUC %.3f over %d rows (%d fake)",
                metrics["auc"], int(metrics["n"]), int(metrics["n_fake"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
