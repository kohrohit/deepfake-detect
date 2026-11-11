"""Slot D's leave-one-generator-out measurement, and the null that judges it.

WHAT THIS LOGO IS, AND HOW IT DIFFERS FROM `bench/runner.py`'s. `runner._logo_results` SLICES
already-computed scores, because P0's detectors are not trained in the harness; its own docstring
calls that honest scope and says it becomes the full protocol once fitting exists. Fitting exists
now, and `training/clip_features.FeatureCache` makes a per-fold refit cost milliseconds, so this
module runs the full protocol: for each held-out generator, REFIT the head on the other families
and score the held-out one.

## What this corpus can and cannot show

- **These are classical composites, not generator output.** Self-blends and four OpenCV
  compositing techniques. A detector that beats them has NOT been shown to beat FSGAN, SimSwap,
  or any diffusion face. `corpora/swaps.py` is the authority on what each technique does; read it
  before quoting a number from here.
- **Every number is memorisation unless `--offset` is at or beyond the fitted detector's training
  window.** `bench/eval_swaps.py: TRAINING_WINDOWS` records the windows. This has produced a
  fabricated headline once already (blend_seam, a worst-generator 0.923 that became 0.527 when
  rebuilt at the right offset).
- **A fold is both generator-disjoint and group-disjoint.** Holding out a generator does not hold
  out the REALS: every couple contributes its reals to every family's fold. So each fold also
  splits couples, and tests only on couples the fit never saw. A fold that tests on training
  reals is half contaminated and reads as a strong fold.
- **The permutation null, not the point estimate, decides whether there is a result.** Every
  number this project has reported on DF40 sits inside the null a model that learnt nothing
  produces there.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression

from bench.metrics import auc, bootstrap_ci_by_group, kish_ess, permutation_null, permutation_p
from dfd.faces import detect_faces

logger = logging.getLogger(__name__)

#: Shuffles for the permutation null. 12 is what this project used on DF40,
#: and it bounds the smallest reachable two-sided p at ~0.08.
DEFAULT_SHUFFLES = 12

#: The label a real record carries in the generator column.
REAL = "real"


def colour_features(images: Sequence[npt.NDArray[np.uint8]]) -> npt.NDArray[np.float32]:
    """Per-channel Lab means and standard deviations. Six numbers per image.

    The control that already caught one shortcut: Lab colour means alone
    separated DF40's two halves at 0.843, and matching resolution and
    container RAISED that to 0.904. If this reaches slot D's AUC, slot D has
    measured colour.
    """
    rows = []
    for img in images:
        lab = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2LAB)
        flat = lab.reshape(-1, 3).astype(np.float32)
        rows.append(np.concatenate([flat.mean(0), flat.std(0)]))
    return np.asarray(rows, dtype=np.float32)


def jpeg_table_features(images: Sequence[npt.NDArray[np.uint8]],
                        *, quality: int) -> npt.NDArray[np.float32]:
    """The luma quantisation table each image would be stored under.

    ONE BYTE OF THIS LABELLED 99.2% OF THIS PROJECT'S CAPTURE CORPUS
    (`bench/vcip_controls.py`), which is why it is a control here.

    On a corpus built by `training/fit_gend.build_training_records` this is
    CONSTANT by construction -- every record passes through one encoder at
    one quality, which is precisely what `reencode` exists to arrange. A
    constant feature is reported as such rather than fitted: a logistic
    regression on it returns 0.5 with no information, and dressing that up as
    a measurement would invite the reader to think an encoder shortcut was
    searched for and not found, when in fact it was designed out.
    """
    rows = []
    for img in images:
        ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(img)[:, :, ::-1],
                               [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise ValueError("JPEG encode failed while building the control")
        # The quantisation table lives in the DQT segment; rather than parse
        # it, take the encoded length and the first bytes of the stream,
        # which vary with the table and with nothing else at fixed content.
        raw = np.frombuffer(buf.tobytes()[:64], dtype=np.uint8)
        rows.append(np.pad(raw, (0, 64 - raw.size)).astype(np.float32))
    return np.asarray(rows, dtype=np.float32)


def _control_auc(features: npt.NDArray[np.float32], labels: npt.NDArray[np.int64],
                 groups: npt.NDArray[Any]) -> dict[str, Any]:
    """Fit a control on `features` and report what it reaches.

    A zero-variance feature set short-circuits to 0.5 and says so, instead of
    letting sklearn return an arbitrary constant that reads as a measurement.
    """
    if float(features.std(axis=0).max()) == 0.0:
        return {"auc": 0.5, "constant": True,
                "note": "the feature is identical on every record, so there "
                        "is nothing for a control to separate. Not a measured "
                        "0.5 -- a structural one."}
    probs = LogisticRegression(max_iter=1000).fit(
        features, labels).predict_proba(features)[:, 1]
    value = auc(probs, labels)
    lo, hi = bootstrap_ci_by_group(probs, labels, groups, auc, n=200, seed=0)
    return {"auc": value, "constant": False, "ci95": [lo, hi],
            "note": "fitted and scored IN SAMPLE, which is the strongest form "
                    "of this control: if even an in-sample shortcut model "
                    "cannot reach the detector, the detector is not reading "
                    "the shortcut."}


def _half_split(groups: npt.NDArray[Any], *, seed: int) -> npt.NDArray[np.bool_]:
    """A boolean mask selecting half the GROUPS for the test side."""
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    order = rng.permutation(unique.size)
    test_groups = set(unique[order[:max(1, unique.size // 2)]].tolist())
    return np.asarray([g in test_groups for g in groups], dtype=bool)


def logo_refit(features: npt.NDArray[np.float32],
               labels: npt.NDArray[np.int64],
               generators: npt.NDArray[Any],
               groups: npt.NDArray[Any],
               *, seed: int = 0,
               n_bootstrap: int = 1000) -> dict[str, dict[str, float]]:
    """Held-out generator -> `{auc, n, n_fake, ci_lo, ci_hi}`, refit per fold.

    Each fold is disjoint in BOTH directions: the held-out generator's fakes
    are never fitted on, and the couples the fold is scored over are never
    fitted on either. Without the second half the fold's reals are training
    reals, which is half a contaminated measurement and reads as a strong
    fold.

    Raises:
        ValueError: when fewer than two fake families are present. One
            generator cannot be left out of itself, and a "LOGO" over one
            family is an in-distribution number wearing the name of a
            transfer one.
    """
    families = sorted({str(g) for g in generators if str(g) != REAL})
    if len(families) < 2:
        raise ValueError(
            f"leave-one-generator-out needs at least two fake families, got "
            f"one generator: {families}. A single-family corpus cannot "
            f"produce a transfer number.")

    out: dict[str, dict[str, float]] = {}
    for held, fold in logo_fold_scores(features, labels, generators, groups,
                                       seed=seed).items():
        probs, y_test = fold["probs"], fold["y"]
        lo, hi = bootstrap_ci_by_group(probs, y_test, groups[fold["rows"]],
                                       auc, n=n_bootstrap, seed=seed)
        out[held] = {"auc": auc(probs, y_test), "n": len(fold["rows"]),
                     "n_fake": int(y_test.sum()), "ci_lo": lo, "ci_hi": hi}
    return out


def logo_fold_scores(features: npt.NDArray[np.float32],
                     labels: npt.NDArray[np.int64],
                     generators: npt.NDArray[Any],
                     groups: npt.NDArray[Any],
                     *, seed: int = 0,
                     ) -> dict[str, dict[str, npt.NDArray[Any]]]:
    """Held-out generator -> the fold's test ROWS, their scores and labels.

    The same refit `logo_refit` publishes an AUC for, with the per-record
    out-of-fold scores kept instead of discarded. They are what makes a
    PAIRED comparison possible: two encoders measured over this same corpus
    can be differenced on the same rows, which is a different and much
    stronger statement than two independent bootstrap intervals that happen
    not to overlap.

    `rows` indexes the ORIGINAL arrays, so a caller can recover each score's
    sample id without re-deriving the split.
    """
    families = sorted({str(g) for g in generators if str(g) != REAL})
    is_test = _half_split(groups, seed=seed)
    out: dict[str, dict[str, npt.NDArray[Any]]] = {}
    for held in families:
        family = np.asarray([str(g) for g in generators])
        train = ~is_test & ((family == REAL) | (family != held))
        test = is_test & ((family == REAL) | (family == held))
        y_train, y_test = labels[train], labels[test]
        if len(set(y_train.tolist())) != 2 or len(set(y_test.tolist())) != 2:
            logger.warning("fold %s has one class on a side; skipped", held)
            continue
        clf = LogisticRegression(max_iter=2000).fit(features[train], y_train)
        probs = clf.predict_proba(features[test])[:, 1]
        out[held] = {"rows": np.flatnonzero(test), "probs": probs, "y": y_test}
    return out


def _null_for_worst(features: npt.NDArray[np.float32],
                    labels: npt.NDArray[np.int64],
                    generators: npt.NDArray[Any],
                    groups: npt.NDArray[Any],
                    *, held: str, seed: int, shuffles: int) -> np.ndarray:
    """The permutation null for one fold: refit on SHUFFLED training labels.

    Shuffles the training labels only. The evaluation side keeps its true
    labels, because the question is "what does a model that learnt nothing
    score here", not "what does a random scoring of a random corpus look
    like".
    """
    is_test = _half_split(groups, seed=seed)
    family = np.asarray([str(g) for g in generators])
    train = ~is_test & ((family == REAL) | (family != held))
    test = is_test & ((family == REAL) | (family == held))
    x_train, x_test = features[train], features[test]
    y_test = labels[test]

    def refit_and_score(shuffled: np.ndarray) -> float:
        if len(set(shuffled.tolist())) != 2:
            return float("nan")
        clf = LogisticRegression(max_iter=2000).fit(x_train, shuffled)
        return auc(clf.predict_proba(x_test)[:, 1], y_test)

    return permutation_null(refit_and_score, labels[train], n=shuffles,
                            seed=seed)


def _report_md(report: dict[str, Any]) -> str:
    """The report card, worst fold first. Good news never leads."""
    folds = report["logo"]
    worst = min(folds, key=lambda k: folds[k]["auc"]) if folds else None
    best = max(folds, key=lambda k: folds[k]["auc"]) if folds else None
    lines = [
        "# Slot D (`gend_clip`) — leave-one-generator-out",
        "",
        f"Measured {report['measured_on']} on {report['n_records']} records "
        f"from {report['n_photographs']} photographs at offset "
        f"{report['offset']}, {report['n_subjects']} couples.",
        "",
        "## Corpus",
        "",
        f"- {report['n_records']} records, {report['n_fake']} fake, "
        f"{report['n_subjects']} couples",
        f"- families: {report['families']}",
        f"- skipped: {report['skipped']}",
        f"- Kish ESS, fakes by family: **{report['kish_ess_fake']:.2f}** of "
        f"{len(report['families']) - 1} families",
        f"- Kish ESS, records by couple: **{report['kish_ess_group']:.2f}** of "
        f"{report['n_subjects']} couples",
        "",
        "## Shortcut controls — read these before the folds",
        "",
        f"- **colour (Lab means and stds)**: AUC {report['control_colour']['auc']:.3f}"
        f" — {report['control_colour']['note']}",
        f"- **JPEG quantisation**: AUC {report['control_jpeg']['auc']:.3f}"
        f" — {report['control_jpeg']['note']}",
        "",
        "## Folds",
        "",
        "| held-out generator | AUC | 95% CI (by couple) | n | n fake |",
        "|---|---|---|---|---|",
    ]
    for name in sorted(folds):
        f = folds[name]
        lines.append(f"| {name} | {f['auc']:.3f} | "
                     f"{f['ci_lo']:.3f}–{f['ci_hi']:.3f} | {f['n']} | "
                     f"{f['n_fake']} |")
    lines += ["", "## The headline is the worst fold", ""]
    if worst is not None:
        lines += [
            f"**Worst fold: `{worst}` at AUC {folds[worst]['auc']:.3f}** "
            f"(95% CI {folds[worst]['ci_lo']:.3f}–{folds[worst]['ci_hi']:.3f}).",
            "",
            f"The best fold, `{best}` at {folds[best]['auc']:.3f}, is "
            f"explicitly distrusted: it is the family most like the ones "
            f"fitted on, and quoting it would describe this corpus's "
            f"internal similarity rather than slot D's transfer.",
            "",
            "## The null that judges it",
            "",
            f"Permutation null for the worst fold over "
            f"{report['n_shuffles']} shuffles of the TRAINING labels: "
            f"**{report['null_lo']:.3f}–{report['null_hi']:.3f}**, "
            f"two-sided p = **{report['permutation_p']:.2f}**.",
            "",
            report["verdict"],
        ]
    lines += ["", "## " + "What this corpus can and cannot show", "",
              __doc__.split("## What this corpus can and cannot show")[1].strip()]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None, *, detect: Any = detect_faces) -> int:
    from datetime import date

    from dfd.detectors.clip_vit import CLIPVisualTower
    from dfd.detectors.loading import load_model
    from training.clip_features import FeatureCache
    from training.fit_gend import (
        REENCODE_QUALITY,
        build_training_records,
        encoder_cache_id,
    )

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--offset", type=int, required=True,
                        help="MUST be at or beyond the fitted detector's "
                             "training window (bench/eval_swaps.py: "
                             "TRAINING_WINDOWS). Below it every number here "
                             "is memorisation.")
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--encoder", type=Path,
                        default=Path("assets/models/clip_vit_l14_visual.pt"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffles", type=int, default=DEFAULT_SHUFFLES)
    parser.add_argument("--out-json", type=Path,
                        default=Path("bench/gend_report.json"))
    parser.add_argument("--out-md", type=Path,
                        default=Path("bench/gend_report.md"))
    args = parser.parse_args(argv)

    from bench.eval_swaps import TRAINING_WINDOWS
    window = max(TRAINING_WINDOWS.values())
    if args.offset < window:
        logger.error(
            "--offset %d is inside a declared training window (the largest "
            "is %d, see bench/eval_swaps.py: TRAINING_WINDOWS). Every number "
            "this run produced would be memorisation. Refusing.",
            args.offset, window)
        return 1

    records, skipped = build_training_records(
        args.root, limit=args.limit, offset=args.offset, seed=args.seed,
        detect=detect)
    if not records:
        logger.error("no records built from %s; skip tally: %s", args.root,
                     skipped)
        return 1

    tower = load_model(args.encoder, model_factory=CLIPVisualTower.vit_l14)
    tower.eval()
    # Through the shared helper, not a second copy of the f-string: this and
    # training/fit_gend.py must key the cache identically or a measurement
    # silently reads the fit's embeddings from a different encoder.
    cache = FeatureCache(args.cache, encoder_id=encoder_cache_id(args.encoder))
    images = [r["image"] for r in records]
    features = cache.embed_all(tower, images)
    labels = np.asarray([r["label"] for r in records], dtype=np.int64)
    generators = np.asarray([r["generator"] or REAL for r in records])
    groups = np.asarray([r["subject_id"] for r in records])
    logger.info("embeddings: %d hits, %d misses", cache.stats["hits"],
                cache.stats["misses"])

    families: dict[str, int] = {}
    for g in generators:
        families[str(g)] = families.get(str(g), 0) + 1
    fake_counts = [v for k, v in families.items() if k != REAL]
    group_counts = list(
        {g: int((groups == g).sum()) for g in np.unique(groups)}.values())

    folds = logo_refit(features, labels, generators, groups, seed=args.seed)
    if not folds:
        logger.error("no usable fold; nothing to report")
        return 1
    worst = min(folds, key=lambda k: folds[k]["auc"])

    # The per-record out-of-fold scores, beside the report and always --
    # never behind a flag. They are what lets a LATER run of a different
    # encoder be compared to this one on the SAME rows (bench/
    # compare_lntuned.py) instead of by holding two bootstrap intervals up
    # to the light. A run whose scores were discarded has to be repeated to
    # be compared, and on this corpus that is an hour of forward passes.
    samples = np.asarray([r["sample_id"] for r in records])
    scorefile = args.out_json.with_suffix(".scores.json")
    scored = logo_fold_scores(features, labels, generators, groups,
                              seed=args.seed)
    scorefile.write_text(json.dumps({
        "encoder": str(args.encoder),
        "root": str(args.root), "offset": args.offset, "limit": args.limit,
        "seed": args.seed,
        "note": ("out-of-fold scores, one row per record per fold, from the "
                 "same refit bench/gend_report.json publishes an AUC for. "
                 "Join two of these on sample_id to compare two encoders "
                 "paired over one corpus."),
        "folds": {
            held: [{"sample_id": str(samples[i]),
                    "subject_id": str(groups[i]),
                    "generator": str(generators[i]),
                    "label": int(labels[i]),
                    "score": float(score)}
                   for i, score in zip(fold["rows"], fold["probs"], strict=True)]
            for held, fold in scored.items()},
    }, sort_keys=True, allow_nan=False) + "\n")
    logger.info("wrote %s (%d folds, %d scored rows)", scorefile, len(scored),
                sum(len(f["rows"]) for f in scored.values()))

    null = _null_for_worst(features, labels, generators, groups, held=worst,
                           seed=args.seed, shuffles=args.shuffles)
    finite = null[np.isfinite(null)]
    p = permutation_p(folds[worst]["auc"], finite)

    inside = bool(finite.size and
                  finite.min() <= folds[worst]["auc"] <= finite.max())
    colour = _control_auc(colour_features(images), labels, groups)
    jpeg = _control_auc(
        jpeg_table_features(images, quality=REENCODE_QUALITY), labels, groups)

    if inside:
        verdict = (
            f"**There is no result.** The worst fold's AUC "
            f"{folds[worst]['auc']:.3f} lies INSIDE the permutation null "
            f"{finite.min():.3f}–{finite.max():.3f}, which is the range a "
            f"model fitted on shuffled labels reaches here. Nothing below "
            f"this line may be quoted as slot D's performance, and the "
            f"evidence card must not carry it.")
    elif folds[worst]["auc"] < 0.75:
        verdict = (
            f"**Outside the null, below the floor.** The worst fold "
            f"{folds[worst]['auc']:.3f} is outside the null "
            f"{finite.min():.3f}–{finite.max():.3f} (p = {p:.2f}), so there "
            f"is signal — but it is under the 0.75 evidence floor, so slot D "
            f"does not open the gate.")
    else:
        verdict = (
            f"**Outside the null and above the floor.** The worst fold "
            f"{folds[worst]['auc']:.3f} clears 0.75 and sits outside the "
            f"null {finite.min():.3f}–{finite.max():.3f} (p = {p:.2f}). "
            f"Whether that becomes evidence is a separate, argued decision: "
            f"this corpus is classical composites, not generator output.")

    report: dict[str, Any] = {
        "measured_on": date.today().isoformat(),
        "root": str(args.root), "offset": args.offset, "limit": args.limit,
        "seed": args.seed, "n_shuffles": args.shuffles,
        "n_photographs": args.limit, "n_records": len(records),
        "n_fake": int(labels.sum()),
        "n_subjects": int(np.unique(groups).size),
        "families": families, "skipped": skipped,
        "kish_ess_fake": kish_ess(fake_counts),
        "kish_ess_group": kish_ess(group_counts),
        "control_colour": colour, "control_jpeg": jpeg,
        "logo": folds, "worst_fold": worst,
        "null_lo": float(finite.min()), "null_hi": float(finite.max()),
        "permutation_p": p, "inside_null": inside, "verdict": verdict,
        "cache_hits": cache.stats["hits"], "cache_misses": cache.stats["misses"],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True,
                                        default=float) + "\n")
    args.out_md.write_text(_report_md(report))
    logger.info("worst fold %s at %.3f; null %.3f-%.3f; p=%.2f",
                worst, folds[worst]["auc"], finite.min(), finite.max(), p)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
