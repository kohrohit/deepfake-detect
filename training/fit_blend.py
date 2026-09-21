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
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from corpora.captures import load_capture_sessions
from corpora.face_pool import DetectFn, build_face_pool
from corpora.sbi import build_sbi_corpus
from dfd.detectors.blend import FEATURE_NAMES, BlendModel, save_blend_model, seam_features
from dfd.faces import detect_faces
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

    IMPORTANT: "subject" here is `Context.subject_id`, which `corpora.sbi`
    sets to the capture SESSION id, not a person identity (see
    `corpora/sbi.py`'s note where `subject_id` is assigned). This split is
    therefore SESSION-disjoint, not identity-disjoint: a person who
    enrolled in more than one of the 442 sessions receives a distinct
    subject_id per session and can still land on both sides of the split,
    which is exactly the "same face on both sides" failure this function
    exists to prevent. Real identity-disjointness needs a face embedder
    that does not exist in this repo yet (`docs/HANDOFF.md` §0 next-step
    3 records the same gap for benchmark criterion 2; it applies equally
    here).

    Args:
        samples: the corpus.
        holdout_fraction: share of subjects placed in test.
        seed: reproducibility.

    Returns:
        A (train, test) pair.

    Note:
        `n_test = max(1, round(len(subjects) * holdout_fraction))` always
        holds out at least one subject: `holdout_fraction=0.0` still yields
        one test subject, and `holdout_fraction=1.0` (or a corpus with only
        one subject) empties `train` entirely. This function does not
        validate that; an empty `train` reaches `fit_blend_model`, where
        `_matrix([])` calls `np.stack([])`, which raises numpy's own
        generic "need at least one array to concatenate" rather than the
        intended "must contain both labels" message. Not guarded here
        deliberately — `main()` never calls this with a holdout fraction
        outside (0, 1), so the case is unreachable in practice.
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


def _json_safe(value: object) -> object:
    """Replace non-finite floats with None, recursively, before json.dumps.

    `evaluate()` legitimately returns `auc=nan` for a one-class test set
    (its own docstring) -- that is not a bug, it is the honest "AUC is
    undefined here" answer, and this report exists to record it, not to
    hide it behind a crash. `json.dumps`'s default behaviour, though,
    would emit the bare token `NaN`: valid to Python's own parser but not
    to any conforming JSON consumer, the same defect `src/dfd/audit.py`
    guards against with `allow_nan=False`. Converting known non-finite
    values to `null` here -- rather than raising, as `audit.py` does for
    an audit record -- keeps `main()` finishing the write it started
    instead of crashing mid-report over a state this module already
    handles deliberately; `null` is also the more useful reading for a
    metric, where "undefined" is closer to the truth than "absent".
    `allow_nan=False` is still passed at the call site below as a safety
    net: if some other, un-anticipated non-finite value reaches the
    report by a path this function does not walk, it fails loudly rather
    than silently writing invalid JSON.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def main(argv: Sequence[str] | None = None, *, detect: DetectFn = detect_faces) -> int:
    """Build the corpus, fit, evaluate, write the model and a JSON report.

    Args:
        argv: CLI arguments, or None to read `sys.argv`.
        detect: face detector, forwarded to `build_face_pool`. Injected —
            not a CLI flag — purely so this function can be exercised in
            tests against a synthetic corpus with no YuNet weight file, the
            same seam `corpora/face_pool.py` already exposes for exactly
            that reason.
    """
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
    # Defence in depth, not the only guard: build_sbi_corpus hard-refuses
    # any crop from a swapped session with EvaluationOnlySessionError, so
    # removing this filter would surface loudly as that exception rather
    # than silently leaking evaluation-only fraud into training.
    genuine = [s for s in sessions if not s.swapped]
    logger.info("%d sessions, %d genuine and usable for training",
                len(sessions), len(genuine))

    crops, skipped = build_face_pool(genuine, detect=detect)
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
    args.report.write_text(json.dumps(_json_safe(report), indent=2,
                                       sort_keys=True, allow_nan=False))
    logger.info("held-out AUC %.3f over %d rows (%d fake)",
                metrics["auc"], int(metrics["n"]), int(metrics["n_fake"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
