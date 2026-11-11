"""Slot D against the traffic it is actually for. EVALUATION ONLY — this fits nothing.

The capture corpus is the only material on this machine that looks like live v-CIP video, and a
head fitted on Flickr photographs faces a real domain shift towards it. Measuring that shift is
useful. Fitting on this corpus is forbidden outright (spec §6.4) and no measurement can overturn
that, so this module imports no fitter, and `tests/bench/test_eval_gend_vcip.py` asserts it by
reading this file's own source.

THE CORPUS LABELS ITSELF, AND THAT IS WHY THERE ARE TWO NUMBERS. The swap pipeline decodes a
capture frame, swaps, and re-encodes at OpenCV quality 95; the capture path never re-encodes. The
first luma quantisation coefficient alone gets **99.2%** of frames right with no pixel read. So an
uncontrolled number here is worthless however good it looks, and this module reports:

- `uncontrolled` — every frame as it sits on disk. Reported so the size of the shortcut is
  visible, never to be quoted.
- `encoder_matched` — the GENUINE half alone re-encoded at the quality that reproduces the swapped
  half's quantisation table EXACTLY. `bench.vcip_controls.find_matching_quality` finds it by
  search; if no quality matches, or if more than one table survives, the control refuses rather
  than reporting, because a near miss leaves a weaker version of the shortcut and reads as a
  control. **This is the pair to quote.**

SESSIONS ARE THE UNIT, NOT FRAMES. Frames of one session share the subject, the camera, the
lighting and the encode. 554 frames carry about 140 sessions' worth of information, and scoring
them as independent rows is how a corpus reports precision it does not have.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import torch.nn as nn

from bench.metrics import auc, kish_ess, tpr_at_fpr
from bench.vcip_controls import _encode, find_matching_quality, quant_table
from corpora.face_pool import DetectFn
from dfd.detectors.clip_vit import CLIPVisualTower
from dfd.detectors.gend import GendClipDetector, GendHead
from dfd.faces import detect_faces
from dfd.quality import measure_quality
from dfd.types import Observation
from training.fit_vcip import _frames, select_face

logger = logging.getLogger(__name__)

#: The false-alarm budget the product is quoted at. Zero, because a v-CIP
#: flow that rejects genuine applicants has a cost nobody has priced.
ZERO_FPR = 0.0


def session_scores(scores: npt.NDArray[np.float64],
                   labels: npt.NDArray[np.int_],
                   sessions: npt.NDArray[Any],
                   ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int_],
                              npt.NDArray[Any]]:
    """Mean score per session, its label, and the session ids, sorted.

    Raises:
        ValueError: if any session's frames carry conflicting labels. A
            session is genuine or swapped; frames of it disagreeing means the
            loader mislabelled something, and averaging that away would turn
            a broken input into a plausible number.
    """
    ids = sorted({str(s) for s in sessions})
    as_str = np.asarray([str(s) for s in sessions])
    means, out_labels = [], []
    for sid in ids:
        mask = as_str == sid
        found = set(np.asarray(labels)[mask].tolist())
        if len(found) > 1:
            raise ValueError(
                f"session {sid!r} has frames with conflicting labels "
                f"{sorted(found)}; refusing to average them into one score")
        means.append(float(np.asarray(scores)[mask].mean()))
        out_labels.append(int(found.pop()))
    return (np.asarray(means, dtype=float),
            np.asarray(out_labels, dtype=int),
            np.asarray(ids))


def per_band(scores: npt.NDArray[np.float64],
             labels: npt.NDArray[np.int_],
             bands: npt.NDArray[Any],
             ) -> dict[str, dict[str, float | int | None]]:
    """Band -> `{n, n_genuine, n_swapped, auc, tpr_at_zero_fpr}`.

    `auc` is None wherever one class is absent, and the denominators travel
    with it. When slot C was measured on this corpus the `high` band held 14
    genuine frames and no swapped one; an AUC there would have been invented,
    and a fabricated 1.0 in a per-band table is exactly the number that gets
    quoted later without its denominator.
    """
    table: dict[str, dict[str, float | int | None]] = {}
    as_str = np.asarray([str(b) for b in bands])
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    for band in sorted(set(as_str.tolist())):
        mask = as_str == band
        yb, sb = y[mask], s[mask]
        n_swapped = int(yb.sum())
        n_genuine = int(mask.sum()) - n_swapped
        both = n_swapped > 0 and n_genuine > 0
        table[band] = {
            "n": int(mask.sum()),
            "n_genuine": n_genuine,
            "n_swapped": n_swapped,
            "auc": auc(sb, yb) if both else None,
            "tpr_at_zero_fpr": (tpr_at_fpr(sb, yb, ZERO_FPR) if both else None),
        }
    return table


def _clamped_roi(box: Any, shape: tuple[int, int],
                 ) -> tuple[int, int, int, int] | None:
    """The face box clipped to the frame, or None when nothing is left.

    YuNet reports boxes that run off the edge of a frame, and every consumer
    that crops without clamping inherits that. `GendClipDetector` clamps its
    own crop; `measure_quality` does not, so it is clamped here before it is
    handed one.
    """
    h, w = shape
    x0, y0 = max(0, int(box.x)), max(0, int(box.y))
    x1 = min(w, int(box.x) + int(box.w))
    y1 = min(h, int(box.y) + int(box.h))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def _score_frames(rows: Sequence[tuple[Path, int, str]],
                  detector: GendClipDetector,
                  *, detect: DetectFn,
                  reencode_genuine_at: int | None,
                  ) -> tuple[list[float], list[int], list[str], list[str],
                             dict[str, int], set[Any]]:
    """Score every frame through `GendClipDetector.score`, the deployed path.

    Scoring through the detector rather than reimplementing its read-out is
    the point: a measurement of a reimplementation measures the
    reimplementation. `reencode_genuine_at` re-encodes the genuine half alone
    at that JPEG quality, which is the encoder control.
    """
    scores: list[float] = []
    labels: list[int] = []
    sessions: list[str] = []
    bands: list[str] = []
    skipped: dict[str, int] = {}
    tables: set[Any] = set()

    for path, label, session in rows:
        image = cv2.imread(str(path))
        if image is None:
            skipped["unreadable"] = skipped.get("unreadable", 0) + 1
            continue
        table = quant_table(path)
        if label == 0 and reencode_genuine_at is not None:
            raw = _encode(image, reencode_genuine_at)
            decoded = (cv2.imdecode(np.frombuffer(raw, np.uint8),
                                    cv2.IMREAD_COLOR)
                       if raw is not None else None)
            if decoded is None:
                skipped["reencode_failed"] = skipped.get("reencode_failed", 0) + 1
                continue
            image, table = decoded, quant_table(raw)
        if table is not None:
            tables.add(table)

        box = select_face(detect(image))
        if box is None:
            skipped["no_face"] = skipped.get("no_face", 0) + 1
            continue
        rgb = image[:, :, ::-1]
        roi = _clamped_roi(box, rgb.shape[:2])
        if roi is None:
            # A detector box can sit partly or wholly outside the frame, and
            # `measure_quality` crops without clamping -- which on this
            # corpus raises `!_src.empty()` out of cvtColor and takes the
            # whole run down. One unmeasurable frame is a skip with a
            # reason, not a failed evaluation of the other 553.
            skipped["roi_outside_frame"] = skipped.get("roi_outside_frame", 0) + 1
            continue
        quality = measure_quality(rgb, roi, box.landmarks)
        raw_score = detector.score([Observation(
            t=0.0, payload=np.ascontiguousarray(rgb), roi=roi,
            quality=quality, source_id=session)])
        if raw_score.abstained or raw_score.score is None:
            skipped[raw_score.reason] = skipped.get(raw_score.reason, 0) + 1
            continue
        scores.append(float(raw_score.score))
        labels.append(int(label))
        sessions.append(session)
        bands.append(str(quality.band))
    return scores, labels, sessions, bands, skipped, tables


def _block(scores: list[float], labels: list[int], sessions: list[str],
           bands: list[str], skipped: dict[str, int]) -> dict[str, Any]:
    """One measurement block: frames, sessions, the per-band table."""
    if not scores:
        return {"measured": False, "why": "no frame produced a score",
                "skipped": skipped}
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    g = np.asarray(sessions)
    both_frames = bool(y.sum()) and bool((1 - y).sum())
    sess_s, sess_y, sess_ids = session_scores(s, y, g)
    both_sessions = bool(sess_y.sum()) and bool((1 - sess_y).sum())
    counts = [int((g == sid).sum()) for sid in sess_ids]
    return {
        "measured": True,
        "n_frames": int(s.size),
        "n_sessions": int(sess_ids.size),
        "n_genuine_sessions": int((sess_y == 0).sum()),
        "n_swapped_sessions": int(sess_y.sum()),
        "kish_ess_frames_by_session": kish_ess(counts),
        "frame_auc": auc(s, y) if both_frames else None,
        "session_auc": auc(sess_s, sess_y) if both_sessions else None,
        "session_tpr_at_zero_fpr": (tpr_at_fpr(sess_s, sess_y, ZERO_FPR)
                                    if both_sessions else None),
        "per_band_frames": per_band(s, y, np.asarray(bands)),
        "skipped": skipped,
    }


def main(argv: Sequence[str] | None = None, *,
         detect: DetectFn = detect_faces,
         encoder_factory: Callable[[], nn.Module] = CLIPVisualTower.vit_l14,
         head_factory: Callable[[], nn.Module] = GendHead) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", required=True, type=Path)
    parser.add_argument("--encoder", type=Path,
                        default=Path("assets/models/clip_vit_l14_visual.pt"))
    parser.add_argument("--weights", type=Path,
                        default=Path("assets/models/gend_clip.pt"))
    parser.add_argument("--out", type=Path,
                        default=Path("bench/gend_vcip.json"))
    args = parser.parse_args(argv)

    rows = _frames(args.captures)
    if not rows:
        logger.error("no frames under %s", args.captures)
        return 1
    # The floor is deliberately `reject` HERE and only here: this run is what
    # DECIDES slot D's floor, so filtering by the floor first would make the
    # per-band table a function of the answer it is supposed to produce.
    detector = GendClipDetector(
        weights_path=args.weights, encoder_path=args.encoder,
        min_quality_band="reject", encoder_factory=encoder_factory,
        head_factory=head_factory)

    logger.info("scoring %d frames uncontrolled", len(rows))
    s0, y0, g0, b0, skipped0, _tables0 = _score_frames(
        rows, detector, detect=detect, reencode_genuine_at=None)
    uncontrolled = _block(s0, y0, g0, b0, skipped0)

    target = None
    for path, label, _ in rows:
        if label == 1:
            target = quant_table(path)
            if target is not None:
                break
    sample = next((img for img in (cv2.imread(str(p)) for p, lb, _ in rows
                                   if lb == 0) if img is not None), None)
    quality = (find_matching_quality(sample, target)
               if target is not None and sample is not None else None)
    if quality is None:
        matched: dict[str, Any] = {
            "measured": False, "quality": None,
            "why": "no encoder quality reproduces the swapped half's table "
                   "exactly; a near miss would leave a weaker version of the "
                   "shortcut and report as a control"}
    else:
        logger.info("scoring the genuine half re-encoded at quality %d", quality)
        s, y, g, b, skipped, tables = _score_frames(
            rows, detector, detect=detect, reencode_genuine_at=quality)
        if len(tables) > 1:
            matched = {"measured": False, "quality": quality,
                       "distinct_tables": len(tables),
                       "why": "more than one quantisation table survived the "
                              "match, so the shortcut is still present"}
        else:
            matched = {"quality": quality, "distinct_tables": len(tables),
                       "what": f"genuine half re-encoded at quality {quality}, "
                               f"the one reproducing the swapped half's table "
                               f"exactly",
                       **_block(s, y, g, b, skipped)}

    report = {
        "captures": str(args.captures),
        "weights": str(args.weights),
        "encoder": str(args.encoder),
        "uncontrolled": uncontrolled,
        "encoder_matched": matched,
        "how_to_read": (
            "Quote `encoder_matched`. `uncontrolled` is reported so the size "
            "of the encoder shortcut is visible, not so it can be cited: one "
            "JPEG header byte labels 99.2% of this corpus. EVALUATION ONLY — "
            "nothing here was fitted on these captures, and spec §6.4 "
            "forbids it."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True,
                                   default=float) + "\n")
    logger.info("uncontrolled session AUC %s; encoder-matched %s",
                uncontrolled.get("session_auc"), matched.get("session_auc"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
