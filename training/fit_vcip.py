"""Fit slot C on the v-CIP capture corpus — real swaps, real capture path.

Run:
    python3 -m training.fit_vcip \\
        --captures ~/Desktop/vkyc-server/captures \\
        --out assets/models/npr.pt --report bench/vcip_fit.json

**Why this exists beside `training/fit_npr.py`.** That fitter resizes every
source to a square before detection, which is right for its problem (SFHQ at
1024px against FairFace at 224px, where image width alone separates the
corpora at AUC 1.000). It is wrong for this one. Here both halves are frames
from the SAME capture path, so there is no resolution confound to normalise
away — and `inswapper_128` emits a 128x128 face that is pasted back upscaled,
so the fingerprint lives at the face's native scale. Resizing destroys it:

    aligned to a 224 square   AUC 0.695, caught  1.5% at zero false alarms
    native ROI crop           AUC 0.957, caught 45.6% at zero false alarms

Both over 506 frames from 106 sessions, folds split on session, re-measured
2026-09-25 by `bench/vcip_controls.py` with the face-selection rule held
fixed across the two arms. (This block read 0.924/51.5% and 0.992/89.7%
until then — a run made before `select_face` below was corrected, and never
recomputed after it. The conclusion is unchanged and the gap is wider than
those figures showed.) So this fitter crops to the native ROI, exactly as
`dfd.detectors.npr._crop_to_roi_native` does at inference. **If one changes,
the other must.** A fitter whose preprocessing differs from its detector's
produces weights that load cleanly and describe features nobody computes.

WHAT THE RESULTING WEIGHTS MAY AND MAY NOT BE USED FOR. Three limits, all
measured, all written into the report this produces:

1. **One swapper.** Every fake here is `inswapper_128`. Nothing measured says
   the head transfers to SimSwap, FaceFusion or a diffusion pipeline, and an
   attacker changes tools for free.
2. **~12-20 distinct people**, so the same faces sit on both sides of every
   session-grouped fold. An identity-disjoint number is not computable on
   this corpus at all; it was attempted and the permutation control came
   back p=0.636, meaning the split had become noise.
3. **An encoder shortcut labels 99.2% of this corpus from the JPEG header.**
   The swap pipeline decodes a capture frame, swaps the face and re-encodes
   at OpenCV quality 95; the capture path never re-encodes. So the first
   luma quantisation coefficient alone splits it — 2 for 336 swapped frames
   against 4 genuine, 1 or 3 for 171 genuine against no swapped frame — and
   a model reading the tables and no pixels reaches frame AUC **0.985**,
   above this head's own 0.957. Measured 2026-09-25 by
   `bench.vcip_controls.encoder_shortcut`. **The number this limit carried
   until then, "AUC 0.873", understated it, and the control it cited —
   re-encoding BOTH halves — is not the right control:** the fakes were
   already compressed twice, so that leaves them compressed three times and
   the halves still asymmetric.
   **What the head survives.** `bench.vcip_controls.matched_encoder`
   re-encodes the GENUINE half alone at the quality reproducing the swapped
   half's table exactly, collapsing the corpus from three tables to one.
   With the shortcut arithmetically gone the head holds session AUC 0.949
   and catches 30.9% of swapped sessions at a zero false-alarm budget,
   against 0.957 and 45.6% uncontrolled. **Quote the controlled pair.**
   What that still cannot remove is the fakes' first compression, which
   happened inside the swap pipeline rather than in the camera.

Taken together: these weights are a v0 for internal evaluation and triage.
They are not calibrated for an automatic reject.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from corpora.face_pool import DetectFn
from dfd.calibration import Calibrator, save_calibrators
from dfd.detectors.npr import NPRStatsNet, _crop_to_roi_native, npr_feature
from dfd.faces import FaceBox, clamp_roi, detect_faces
from dfd.quality import measure_quality

from .fit_npr import save_npr_model

logger = logging.getLogger(__name__)

#: Frames smaller than this on the shorter side of the face crop are
#: dropped. Below roughly this size there is not enough spectrum left for
#: the band means to mean anything, and `_crop_to_roi_native` itself only
#: refuses a crop under 2px.
MIN_FACE_PX = 32

#: Minimum sessions needed before a 5-fold session-grouped estimate is worth
#: printing. Fewer than this and the folds are single-session, which reports
#: variance rather than performance.
MIN_SESSIONS = 10


def _frames(root: Path) -> list[tuple[Path, int, str]]:
    """Every capture frame with its label and session.

    The label is read per-FRAME where the session records one, falling back
    to the session's own flag. A session can hold frames of both kinds, and
    collapsing them to the session label would mislabel individual frames.
    """
    out: list[tuple[Path, int, str]] = []
    for meta_path in sorted(root.glob("*/results.json")):
        meta = json.loads(meta_path.read_text())
        sess = meta.get("session_id") or meta_path.parent.name
        for frame in meta.get("frames") or []:
            saved = frame.get("saved_as")
            path = (root / sess / Path(saved).name if saved
                    else root / sess / f"frame_{int(frame['index']):02d}.jpg")
            if path.is_file():
                label = int(frame.get("swapped", meta.get("swapped", 0)))
                out.append((path, label, sess))
    return out


def _shapes_in_both_classes(rows: Sequence[tuple[Path, int, str]]
                            ) -> set[tuple[int, ...]]:
    """Frame shapes that BOTH classes contain.

    Measured on this corpus: genuine frames include 547x768, 944x1024 and
    973x432 shapes that no swapped frame has. Left in, image dimensions
    alone would partly separate the halves, and the head would learn a
    property of the recording setup rather than of the swap. This is the
    same control `bench.eval_df40` needed and could not apply.
    """
    seen: dict[tuple[int, ...], set[int]] = {}
    for path, label, _ in rows:
        img = cv2.imread(str(path))
        if img is not None:
            seen.setdefault(img.shape, set()).add(label)
    return {shape for shape, labels in seen.items() if labels == {0, 1}}


def select_face(boxes: Sequence[FaceBox]) -> FaceBox | None:
    """The face the PIPELINE would score, or None.

    Largest by area, matching `dfd.pipeline` exactly — not the most
    confident, which is the obvious alternative and was what this fitter did
    until 2026-09-24. On any frame with two faces the two rules disagree, and
    the head is then fitted on one face and asked to score another.

    Measured consequence: the assembled system scored 0.821 at session level
    against 0.954 for the same head measured out-of-fold here — and that run
    was NOT held out, so it should have been optimistic rather than worse.
    (Both figures are from the run that FOUND this, with the old rule still
    in place. After the fix the out-of-fold number is 0.957; every other
    figure measured before it had to be recomputed, which is what
    `bench/vcip_controls.json` is.)

    The serving path is the one that cannot change, so this follows it.
    `tests/test_fit_vcip.py` asserts the two agree over random box sets.
    """
    return max(boxes, key=lambda b: b.w * b.h) if boxes else None


def features_for(path: Path, *, detect: DetectFn, net: NPRStatsNet,
                 ) -> tuple[npt.NDArray[np.float64], str] | None:
    """The `NPRStatsNet` feature vector for one frame's face, or None.

    Preprocessing is `_crop_to_roi_native` — the detector's own — so the
    fitted head describes exactly what inference computes.

    Returns:
        A (features, quality_band) pair, or None when no face is usable. The
        band travels with the vector because calibration is per band and a
        second pass over the corpus to recover it would risk measuring
        quality differently the second time.
    """
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    box = select_face(detect(rgb))
    if box is None:
        return None
    roi = clamp_roi(rgb.shape, box)
    crop = _crop_to_roi_native(rgb, roi)
    if crop is None or min(crop.shape[:2]) < MIN_FACE_PX:
        return None
    # The pipeline measures quality from the SELECTED box's landmarks, not
    # from boxes[0] — another way the two paths can silently disagree.
    band = measure_quality(rgb, roi, box.landmarks[:2]).band
    residual = torch.from_numpy(npr_feature(crop)).permute(2, 0, 1)[None]
    with torch.no_grad():
        vector = net.features(residual.float())[0].numpy().astype(np.float64)
    return vector, band


def _session_metrics(scores: npt.NDArray[np.float64],
                     labels: npt.NDArray[np.int_],
                     groups: npt.NDArray[np.str_]) -> dict[str, Any]:
    """Catch rate at false-alarm BUDGETS, aggregated per session.

    The product decides once per session, so the operating point belongs
    there. Sessions are scored by the MAX over their frames: the question is
    whether ANY frame is synthetic, and a mean lets four clean frames bury
    one swap.
    """
    from bench.metrics import auc

    per: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for score, label, group in zip(scores, labels, groups, strict=True):
        if np.isfinite(score):
            per[str(group)].append((float(score), int(label)))
    ids = sorted(per)
    sess_scores = np.array([max(v for v, _ in per[i]) for i in ids])
    sess_labels = np.array([int(round(float(np.mean([lb for _, lb in per[i]]))))
                            for i in ids])

    out: dict[str, Any] = {
        "sessions": len(ids),
        "genuine": int((sess_labels == 0).sum()),
        "swapped": int((sess_labels == 1).sum()),
    }
    if len(np.unique(sess_labels)) < 2:
        out["auc"] = None
        return out
    out["auc"] = float(auc(sess_scores, sess_labels))
    for budget in (0.0, 0.026, 0.05, 0.10):
        best = 0.0
        for threshold in np.unique(sess_scores):
            pred = (sess_scores >= threshold).astype(int)
            fpr = float(((pred == 1) & (sess_labels == 0)).sum()
                        / max(1, (sess_labels == 0).sum()))
            if fpr <= budget:
                best = max(best, float(((pred == 1) & (sess_labels == 1)).sum()
                                       / max(1, (sess_labels == 1).sum())))
        out[f"caught_at_fpr_{budget}"] = best
    return out


def main(argv: Sequence[str] | None = None, *,
         detect: DetectFn = detect_faces) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", required=True, type=Path,
                        help="the v-CIP capture directory, one folder per "
                             "session holding frame_NN.jpg and results.json")
    parser.add_argument("--out", type=Path, default=Path("assets/models/npr.pt"))
    parser.add_argument("--report", type=Path,
                        default=Path("bench/vcip_fit.json"))
    parser.add_argument("--calibration", type=Path,
                        default=Path("assets/models/calibration.json"),
                        help="where to write per-band calibration curves. "
                             "Without these the detector scores and then "
                             "abstains with `uncalibrated_for_band`, and "
                             "every verdict stays insufficient_evidence")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--C", dest="c", type=float, default=1.0,
                        help="inverse regularisation strength")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.captures.is_dir():
        logger.error("no capture directory at %s", args.captures)
        return 1

    rows = _frames(args.captures)
    if not rows:
        logger.error("no frames found under %s", args.captures)
        return 1
    keep = _shapes_in_both_classes(rows)
    logger.info("frame shapes present in both classes: %s", sorted(keep))

    net = NPRStatsNet()
    feats: list[npt.NDArray[np.float64]] = []
    labels: list[int] = []
    groups: list[str] = []
    band_of: list[str] = []
    skipped: Counter[str] = Counter()
    for path, label, sess in rows:
        img = cv2.imread(str(path))
        if img is None:
            skipped["unreadable"] += 1
            continue
        if img.shape not in keep:
            skipped["shape_in_one_class_only"] += 1
            continue
        got = features_for(path, detect=detect, net=net)
        if got is None:
            skipped["no_usable_face"] += 1
            continue
        vector, band = got
        feats.append(vector)
        labels.append(label)
        groups.append(sess)
        band_of.append(band)

    if not feats:
        logger.error("nothing usable under %s (skipped: %s)",
                     args.captures, dict(skipped) or "nothing")
        return 1
    features = np.asarray(feats, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    g = np.asarray(groups)
    if len(np.unique(y)) < 2:
        logger.error("only label(s) %s survived; a head needs both",
                     sorted(set(y.tolist())))
        return 1
    logger.info("%d frames | %d genuine / %d swapped | %d sessions | "
                "skipped %s", len(y), int((y == 0).sum()), int((y == 1).sum()),
                len(set(g)), dict(skipped) or "nothing")

    mean = features.mean(axis=0)
    # Guard a constant feature: dividing by its zero spread yields inf, which
    # reaches the linear layer and leaves it as a confident score.
    scale = features.std(axis=0)
    scale[scale < 1e-9] = 1.0
    standardised = (features - mean) / scale

    held_out: dict[str, Any] = {"measured": False}
    oof = np.full(len(y), np.nan)
    n_sessions = len(set(g))
    if n_sessions >= MIN_SESSIONS:
        for train, test in GroupKFold(n_splits=5).split(standardised, y, g):
            if len(np.unique(y[train])) < 2:
                continue
            fold = LogisticRegression(max_iter=5000, C=args.c,
                                      random_state=args.seed)
            fold.fit(standardised[train], y[train])
            oof[test] = fold.predict_proba(standardised[test])[:, 1]
        held_out = {"measured": True, "folds": "GroupKFold(5) on session",
                    **_session_metrics(oof, y, g)}
        logger.info("held-out (session folds): AUC %s, caught %.1f%% at a "
                    "zero false-alarm budget",
                    held_out.get("auc"),
                    100 * float(held_out.get("caught_at_fpr_0.0") or 0.0))
    else:
        logger.warning("only %d sessions; skipping the held-out estimate "
                       "rather than reporting fold variance as performance",
                       n_sessions)

    model = LogisticRegression(max_iter=5000, C=args.c, random_state=args.seed)
    model.fit(standardised, y)
    save_npr_model(model.coef_[0], float(model.intercept_[0]), mean, scale,
                   args.out)
    logger.info("wrote %s", args.out)

    # Calibration, per quality band. Without it the detector computes a score
    # and then abstains with `uncalibrated_for_band`, so every verdict stays
    # `insufficient_evidence` however good the head is.
    #
    # Fitted on the OUT-OF-FOLD scores where they exist, never on the final
    # model's own training scores: a curve fitted on scores the head has
    # already seen is a curve that has sat its own exam, and it reports
    # confidence the detector has not earned.
    calibration: dict[str, Any] = {"written": False}
    if held_out.get("measured") and np.isfinite(oof).any():
        usable = np.isfinite(oof)
        cal = Calibrator("npr")
        cal.fit(oof[usable], y[usable], np.asarray(band_of)[usable])
        fitted = sorted(cal._models)
        if fitted:
            save_calibrators({"npr": cal}, args.calibration)
            logger.info("wrote %s (bands: %s)", args.calibration,
                        ", ".join(fitted))
        priors = {b: float(np.asarray(band_of)[usable][
            np.asarray(band_of)[usable] == b].size) for b in fitted}
        calibration = {
            "written": bool(fitted),
            "path": str(args.calibration),
            "bands_fitted": fitted,
            "bands_seen": sorted(set(band_of)),
            "n_per_fitted_band": priors,
            "corpus_prior_fake": float(y.mean()),
            "WARNING": (
                "These curves are conditioned on THIS corpus's base rate — "
                f"{float(y.mean()):.1%} of frames are swapped. Live v-CIP "
                "traffic is fraudulent at a rate orders of magnitude lower, "
                "so the probability this calibration reports is P(swap | "
                "score, corpus), NOT P(swap | score, production). Read it as "
                "a ranking score for triage. Re-fit the prior against "
                "measured live base rates before any number is shown to a "
                "reviewer as a probability."),
        }
    else:
        logger.warning("no out-of-fold scores; calibration not written, so "
                       "the detector will abstain with uncalibrated_for_band")

    report = {
        "fitter": "training.fit_vcip",
        "captures": str(args.captures),
        "preprocessing": "native ROI crop (dfd.detectors.npr."
                         "_crop_to_roi_native); NO resize",
        "n_features": int(NPRStatsNet.N_FEATURES),
        "frames": len(y),
        "genuine": int((y == 0).sum()),
        "swapped": int((y == 1).sum()),
        "sessions": n_sessions,
        "skipped": dict(skipped),
        "seed": args.seed,
        "C": args.c,
        "held_out": held_out,
        "calibration": calibration,
        "limits": [
            "ONE SWAPPER: every fake is inswapper_128. Transfer to SimSwap, "
            "FaceFusion or any diffusion pipeline is unmeasured.",
            "~12-20 DISTINCT PEOPLE: the same faces sit on both sides of "
            "every session-grouped fold. An identity-disjoint number is not "
            "computable on this corpus (attempted; permutation p=0.636).",
            "ENCODER SHORTCUT IN THE CORPUS: the JPEG quantisation tables "
            "alone label 99.2% of frames, no pixels read, at frame AUC "
            "0.985 — above this head's own 0.957. The head survives the "
            "control that removes it (genuine half re-encoded onto the "
            "swapped half's table: session AUC 0.949, 30.9% caught at a "
            "zero false-alarm budget, against 0.957 and 45.6% here). QUOTE "
            "THE CONTROLLED PAIR, and see bench/vcip_controls.json. Any "
            "other model fitted on this corpus needs the same control.",
            "NOT CALIBRATED: these scores are not probabilities. Use for "
            "triage and internal evaluation, never for an automatic reject.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False))
    logger.info("wrote %s", args.report)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
