"""Fit slot C — the NPR upsampling-fingerprint head. Not part of the
installed package: fitting is not inference, and nothing in src/dfd/ may
depend on this module.

Run:
    python3 -m training.fit_npr \\
        --fakes ~/Desktop/agents/datasets/sfhq_part3/probe \\
        --reals ~/Desktop/agents/datasets/fairface_sessions \\
        --out assets/models/npr.pt --report bench/npr_fit.json

WHY THIS SLOT, AND WHY NOW. Measured 2026-09-23 (docs/HANDOFF.md §0), slot
A's 30-dimension seam vector saturates at ~500 training examples: growing the
training set 237-fold moved its cross-corpus number by 0.004. More fakes buy
nothing for those features. Slot C reads a DIFFERENT physical property — how
a generator's upsampling distributes energy across the stride-2 sampling
phases — so it is the cheapest untried lever, and SFHQ is exactly the data it
wants.

THE PREPROCESSING IS NOT INCIDENTAL. Every source is resized to `--norm-size`
BEFORE face detection. Measured the same day: used at native resolution, SFHQ
(1024px) against FairFace (224px) gives an in-family AUC of 0.969 that
transfers to an unseen corpus at 0.461 — chance — because image width alone
separates the two corpora at AUC 1.000. Normalising first costs in-family
performance and buys transfer. Anything fitted here without it is fitting the
resolution.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from corpora.face_pool import DetectFn
from dfd.detectors.npr import NPRStatsNet, npr_feature
from dfd.faces import align, clamp_roi, detect_faces

logger = logging.getLogger(__name__)

#: Every source is resized to this square before detection. See the module
#: docstring: this is the difference between transferring and not.
DEFAULT_NORM_SIZE = 224

#: Fraction of items held out. Items, not rows — each image here is an
#: independent draw (a generator sample, or one FairFace photograph), which
#: is the assumption `corpora.sfhq` states and `corpora.df40` could not.
DEFAULT_HOLDOUT = 0.3

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def _images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)


def stats_for(
    path: Path,
    *,
    detect: DetectFn,
    norm_size: int,
) -> npt.NDArray[np.float32] | None:
    """The 27 NPR phase statistics for one image, or None if it yields no face.

    Args:
        path: image file.
        detect: face detector, injected so tests need no weight file.
        norm_size: square the SOURCE is resized to before detection.

    Returns:
        A float32 vector of length `NPRStatsNet.N_FEATURES`, or None when the
        image is unreadable, holds no detectable face, or the detection does
        not overlap the frame.
    """
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    bgr = cv2.resize(bgr, (norm_size, norm_size), interpolation=cv2.INTER_AREA)
    frame: npt.NDArray[np.uint8] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    boxes = detect(frame)
    if not boxes:
        return None
    box = max(boxes, key=lambda b: b.score)
    if clamp_roi(frame.shape, box) is None:
        return None
    crop = align(frame, box, size=norm_size)
    residual = torch.from_numpy(npr_feature(crop)).permute(2, 0, 1)[None]
    with torch.no_grad():
        feats = NPRStatsNet().features(residual)
    return feats[0].numpy().astype(np.float32)


def split_items(n: int, *, holdout_fraction: float, seed: int) -> tuple[
        npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Index split into (fit, holdout). Deterministic under `seed`."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(round(n * (1.0 - holdout_fraction)))
    return idx[:cut], idx[cut:]


def save_npr_model(
    coef: npt.NDArray[np.float64],
    intercept: float,
    mean: npt.NDArray[np.float64],
    scale: npt.NDArray[np.float64],
    path: str | Path,
) -> None:
    """Write a fitted head as a state_dict `NPRStatsNet` can load securely.

    The binary head is written as the TWO-column layer `NPRDetector` expects:
    column 1 is the fake logit, column 0 its negation, so a softmax over them
    reproduces the logistic probability exactly. Writing a one-column layer
    would load fine and then fail the detector's `EXPECTED_NUM_CLASSES` check
    at score time, which is the wrong place to find out.

    Args:
        coef: `(N_FEATURES,)` logistic coefficients on STANDARDISED features.
        intercept: logistic intercept.
        mean: per-feature mean used to standardise.
        scale: per-feature standard deviation used to standardise.
        path: destination `.pt`.
    """
    net = NPRStatsNet()
    with torch.no_grad():
        net.feature_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        net.feature_scale.copy_(torch.as_tensor(scale, dtype=torch.float32))
        w = torch.as_tensor(coef, dtype=torch.float32)
        net.linear.weight.copy_(torch.stack([-w / 2.0, w / 2.0]))
        b = float(intercept)
        net.linear.bias.copy_(torch.tensor([-b / 2.0, b / 2.0]))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), path)


def main(argv: Sequence[str] | None = None, *,
         detect: DetectFn = detect_faces) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fakes", required=True, type=Path)
    parser.add_argument("--reals", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap images taken per class")
    parser.add_argument("--norm-size", type=int, default=DEFAULT_NORM_SIZE)
    parser.add_argument("--holdout", type=float, default=DEFAULT_HOLDOUT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    rows: list[npt.NDArray[np.float32]] = []
    y: list[int] = []
    skipped = {"fake": 0, "real": 0}
    for label, root in ((1, args.fakes), (0, args.reals)):
        paths = _images(Path(root))
        if args.limit is not None:
            paths = paths[:args.limit]
        for p in paths:
            f = stats_for(p, detect=detect, norm_size=args.norm_size)
            if f is None:
                skipped["fake" if label else "real"] += 1
                continue
            rows.append(f)
            y.append(label)
    if len(set(y)) < 2:
        logger.error("need both labels; got %s", sorted(set(y)))
        return 1

    feats_all, ya = np.asarray(rows, dtype=np.float64), np.asarray(y, dtype=int)
    fit_i, hold_i = split_items(len(ya), holdout_fraction=args.holdout,
                                seed=args.seed)
    if len(set(ya[fit_i])) < 2 or len(set(ya[hold_i])) < 2:
        logger.error("a split is single-label; nothing measurable")
        return 1

    mean = feats_all[fit_i].mean(axis=0)
    scale = feats_all[fit_i].std(axis=0)
    # A feature that never varies on the fit split would divide by zero and
    # send every downstream value to inf. Held at 1.0, it contributes a
    # constant the intercept absorbs.
    scale[scale == 0.0] = 1.0
    model = LogisticRegression(max_iter=5000, class_weight="balanced")
    model.fit((feats_all[fit_i] - mean) / scale, ya[fit_i])
    held = model.predict_proba((feats_all[hold_i] - mean) / scale)[:, 1]
    auc = float(roc_auc_score(ya[hold_i], held))

    save_npr_model(model.coef_[0], float(model.intercept_[0]), mean, scale,
                   args.out)
    report = {
        "n_fit": int(len(fit_i)), "n_holdout": int(len(hold_i)),
        "n_fakes": int((ya == 1).sum()), "n_reals": int((ya == 0).sum()),
        "skipped": skipped, "norm_size": int(args.norm_size),
        "seed": int(args.seed),
        # NOT a field-performance number. Items are disjoint; the two CORPORA
        # are not, and a pair of corpora from different sources is separable
        # on source alone — measured at AUC 0.843 on DF40 (docs/HANDOFF.md).
        "holdout_auc_in_family": auc,
        "out": str(args.out),
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    logger.info("npr fit: %s", json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
