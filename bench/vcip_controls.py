"""The controls that decide whether the v-CIP capture number means anything.

    python3 -m bench.vcip_controls \\
        --captures ~/Desktop/vkyc-server/captures \\
        --out bench/vcip_controls.json

**Why this file exists.** `training/fit_vcip.py` reports a held-out session
AUC of 0.957 on this corpus, and its report carried a limits list asserting
an encoder shortcut "at AUC 0.873" and a re-encoding control that removed it.
Neither number had a committed script behind it. Both were wrong when
re-measured on 2026-09-25: the shortcut is far stronger than 0.873, and the
control that was quoted is not the control the corpus needs.

**The shortcut, measured.** The swap pipeline decodes a capture frame, swaps
the face and re-encodes at OpenCV quality 95. The capture path never
re-encodes. So every fake carries the writer's quantisation table and every
genuine frame carries the camera's, and the corpus is labelled by one byte of
the JPEG header before a single pixel is read:

    first luma coefficient == 2  ->  336 swapped,   4 genuine
    first luma coefficient in {1, 3}  ->  0 swapped, 171 genuine

That is 99.2% of the corpus from the header, and a model fitted on the
tables alone — no face, no pixels — reaches frame AUC 0.985 against the
head's own 0.957. **Any result on this corpus is worthless until it is shown
to survive with that shortcut removed.**

**The control that removes it.** Re-encoding BOTH halves (what the old limits
list described) leaves the halves asymmetric: the fakes were already
compressed twice, so they end up compressed three times and the genuine half
twice. `matched_encoder` instead re-encodes the GENUINE half alone, at the
quality that reproduces the swapped half's table exactly — discovered here by
search, never assumed, and the control refuses to run if no quality matches.
Both halves then carry one identical table and the same decode-encode
history. The corpus goes from three distinct tables to one.

**What survives it, measured 2026-09-25:** session AUC 0.957 -> 0.949, and
30.9% of swapped sessions caught at a zero false-alarm budget against 45.6%
uncontrolled. The signal is real and it is smaller than the raw number.

**What this control still does not remove.** The fakes' first compression
happened inside the swap pipeline and the genuine half's inside the camera.
Matching the final encoder cannot match that history, so a double-compression
trace remains available to the head. Removing it needs a corpus whose halves
share an imaging chain end to end, which this one does not.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from corpora.face_pool import DetectFn
from dfd.detectors.npr import NPRStatsNet, _crop_to_roi_native, npr_feature
from dfd.faces import align, clamp_roi, detect_faces
from training.fit_vcip import (
    MIN_FACE_PX,
    _frames,
    _session_metrics,
    _shapes_in_both_classes,
    select_face,
)

from .metrics import auc, tpr_at_fpr

logger = logging.getLogger(__name__)

#: Qualities searched for one reproducing the swapped half's quantisation
#: table. Below 70 the search is pointless — no capture writer emits tables
#: that coarse — and the control fails loudly rather than settling for a
#: near miss, because a near miss leaves a weaker version of the very
#: shortcut it exists to remove.
SEARCH_QUALITIES: tuple[int, ...] = tuple(range(70, 101))

#: The square `training/fit_npr.py` resizes its sources to. Used here only to
#: re-measure what that preprocessing costs on this corpus.
ALIGNED_SIZE = 224

QuantTable = tuple[tuple[int, ...], ...]


def quant_table(payload: bytes | Path) -> QuantTable | None:
    """The JPEG quantisation tables of an encoded image, or None.

    Returns:
        One tuple per table, in table order, or None when the bytes carry no
        quantisation tables at all (a PNG, or a decoded array re-saved
        losslessly). None is NOT an empty tuple: an empty tuple would
        compare equal between two files that share no table, which is the
        one comparison this function exists to make.
    """
    source: Any = io.BytesIO(payload) if isinstance(payload, bytes) else payload
    with Image.open(source) as image:
        tables = getattr(image, "quantization", None)
        if not tables:
            return None
        return tuple(tuple(int(v) for v in tables[k]) for k in sorted(tables))


def _encode(bgr: npt.NDArray[np.uint8], quality: int) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return bytes(buf.tobytes()) if ok else None


def find_matching_quality(sample: npt.NDArray[np.uint8], target: QuantTable,
                          *, qualities: Sequence[int] = SEARCH_QUALITIES,
                          ) -> int | None:
    """The encoder quality whose table is EXACTLY `target`, or None.

    Exact rather than nearest. The first luma coefficient alone matches at
    qualities 93, 94 and 95 on this corpus; only 95 reproduces the whole
    table, and settling for 93 would leave 63 coefficients still separating
    the halves — a quieter version of the shortcut being removed, which is
    worse than no control because it reports as one.
    """
    for quality in qualities:
        raw = _encode(sample, quality)
        if raw is not None and quant_table(raw) == target:
            return quality
    return None


def _features(bgr: npt.NDArray[np.uint8], *, detect: DetectFn,
              net: NPRStatsNet, aligned: bool,
              ) -> npt.NDArray[np.float64] | None:
    """The head's feature vector for one decoded frame, or None.

    Face selection is `training.fit_vcip.select_face` — the shipped rule —
    in every variant, so a preprocessing comparison varies preprocessing
    alone. The numbers this replaced varied both at once.
    """
    rgb: npt.NDArray[np.uint8] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    box = select_face(detect(rgb))
    if box is None:
        return None
    roi = clamp_roi(rgb.shape, box)
    if roi is None:
        return None
    crop = (align(rgb, box, size=ALIGNED_SIZE) if aligned
            else _crop_to_roi_native(rgb, roi))
    if crop is None or min(crop.shape[:2]) < MIN_FACE_PX:
        return None
    residual = torch.from_numpy(npr_feature(crop)).permute(2, 0, 1)[None]
    with torch.no_grad():
        return net.features(residual.float())[0].numpy().astype(np.float64)


def held_out_scores(features: npt.NDArray[np.float64],
                    labels: npt.NDArray[np.int_],
                    groups: npt.NDArray[np.str_], *, seed: int = 0,
                    c: float = 1.0) -> npt.NDArray[np.float64]:
    """Out-of-fold probabilities under `GroupKFold(5)` on session.

    Identical to `training.fit_vcip`'s own loop, standardiser included, so a
    control here is comparable with the headline it is controlling.
    """
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-9] = 1.0
    standardised = (features - mean) / scale
    out = np.full(len(labels), np.nan)
    for train, test in GroupKFold(n_splits=5).split(standardised, labels, groups):
        if len(np.unique(labels[train])) < 2:
            continue
        fold = LogisticRegression(max_iter=5000, C=c, random_state=seed)
        fold.fit(standardised[train], labels[train])
        out[test] = fold.predict_proba(standardised[test])[:, 1]
    return out


def _measure(features: npt.NDArray[np.float64], labels: npt.NDArray[np.int_],
             groups: npt.NDArray[np.str_], *, seed: int = 0,
             c: float = 1.0) -> dict[str, Any]:
    """Frame-level AUC and the session-level operating points, together.

    Both bases travel with every control because they disagree here by a
    wide margin — the encoder shortcut scores 0.985 per frame and 0.935 per
    session — and a number quoted without its basis invites the reader to
    pick the flattering one.
    """
    scores = held_out_scores(features, labels, groups, seed=seed, c=c)
    usable = np.isfinite(scores)
    out: dict[str, Any] = {"frames": int(len(labels))}
    if usable.any() and len(np.unique(labels[usable])) > 1:
        out["frame_auc"] = float(auc(scores[usable], labels[usable]))
    else:
        out["frame_auc"] = None
    out.update(_session_metrics(scores, labels, groups))
    return out


def _rows(captures: Path) -> tuple[list[tuple[Path, int, str]],
                                   set[tuple[int, ...]]]:
    rows = _frames(captures)
    return rows, _shapes_in_both_classes(rows)


def encoder_shortcut(captures: Path) -> dict[str, Any]:
    """How much of the corpus the JPEG header alone labels. No pixels read.

    The tables are the feature vector. If this scores anywhere near the
    head, every number fitted on this corpus is suspect until a control
    removes it — which is what `matched_encoder` is for.
    """
    rows, keep = _rows(captures)
    vectors: list[npt.NDArray[np.float64]] = []
    labels: list[int] = []
    groups: list[str] = []
    contingency: dict[QuantTable, Counter[int]] = {}
    for path, label, session in rows:
        image = cv2.imread(str(path))
        if image is None or image.shape not in keep:
            continue
        table = quant_table(path)
        if table is None:
            continue
        contingency.setdefault(table, Counter())[label] += 1
        vectors.append(np.concatenate([np.asarray(t, dtype=np.float64)
                                       for t in table]))
        labels.append(label)
        groups.append(session)
    if not vectors:
        return {"measured": False, "why": "no frame carried a quantisation table"}

    width = min(len(v) for v in vectors)
    x = np.asarray([v[:width] for v in vectors], dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    g = np.asarray(groups)
    by_table = [
        {"first_luma_coefficient": int(table[0][0]),
         "genuine": int(counts[0]), "swapped": int(counts[1])}
        for table, counts in sorted(contingency.items(),
                                    key=lambda kv: -sum(kv[1].values()))
    ]
    # The majority-label lookup: how much of the corpus a reader gets right
    # by reading the header and nothing else.
    correct = sum(max(row["genuine"], row["swapped"]) for row in by_table)
    total = sum(row["genuine"] + row["swapped"] for row in by_table)
    return {
        "measured": True,
        "what": "logistic regression on the quantisation tables alone",
        "distinct_tables": len(by_table),
        "by_table": by_table,
        "header_lookup_accuracy": correct / total if total else None,
        **_measure(x, y, g),
    }


def matched_encoder(captures: Path, *, detect: DetectFn = detect_faces,
                    ) -> dict[str, Any]:
    """The head, re-measured with both halves on one encoder.

    The genuine half alone is re-encoded at the quality reproducing the
    swapped half's table. Refuses rather than reports if that quality cannot
    be found, or if more than one table survives: either way the shortcut is
    still in the corpus and a number computed over it would be the thing
    this function exists to rule out.
    """
    rows, keep = _rows(captures)
    target: QuantTable | None = None
    for path, label, _ in rows:
        if label == 1:
            image = cv2.imread(str(path))
            if image is not None and image.shape in keep:
                target = quant_table(path)
                if target is not None:
                    break
    if target is None:
        return {"measured": False, "why": "no swapped frame carries a table"}

    sample = next((img for img in (cv2.imread(str(p)) for p, lb, _ in rows
                                   if lb == 0)
                   if img is not None), None)
    if sample is None:
        return {"measured": False, "why": "no readable genuine frame"}
    quality = find_matching_quality(sample, target)
    if quality is None:
        return {"measured": False,
                "why": "no encoder quality reproduces the swapped half's "
                       "table exactly; a near miss would leave a weaker "
                       "version of the shortcut and report as a control"}

    net = NPRStatsNet()
    vectors: list[npt.NDArray[np.float64]] = []
    labels: list[int] = []
    groups: list[str] = []
    tables: set[QuantTable] = set()
    for path, label, session in rows:
        image = cv2.imread(str(path))
        if image is None or image.shape not in keep:
            continue
        table = quant_table(path)
        if label == 0:
            raw = _encode(image, quality)
            if raw is None:
                continue
            decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                continue
            image, table = decoded, quant_table(raw)
        vector = _features(image, detect=detect, net=net, aligned=False)
        if vector is None:
            continue
        if table is not None:
            tables.add(table)
        vectors.append(vector)
        labels.append(label)
        groups.append(session)
    if not vectors:
        return {"measured": False, "why": "no usable face after re-encoding"}
    if len(tables) > 1:
        return {"measured": False, "distinct_tables": len(tables),
                "why": "more than one quantisation table survived the match, "
                       "so the shortcut is still present"}
    return {
        "measured": True,
        "what": f"genuine half re-encoded at quality {quality}, the one "
                f"reproducing the swapped half's table exactly",
        "quality": quality,
        "distinct_tables": len(tables),
        **_measure(np.asarray(vectors), np.asarray(labels, dtype=int),
                   np.asarray(groups)),
    }


def block_views(features: npt.NDArray[np.float64],
                ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """The phase columns and the spectral columns, as a partition.

    Split out so a test can assert the two arms are handed DIFFERENT columns
    that together account for every one. Inline, the two slices differ by a
    single colon and a wrong one produces an ablation whose arms are the
    same features under two names — which reports as a finding rather than
    as an error.
    """
    split = NPRStatsNet.N_PHASE_FEATURES
    if features.shape[1] != NPRStatsNet.N_FEATURES:
        msg = (f"expected {NPRStatsNet.N_FEATURES} columns, "
               f"got {features.shape[1]}")
        raise ValueError(msg)
    return features[:, :split].copy(), features[:, split:].copy()


def preprocessing_and_blocks(captures: Path, *,
                             detect: DetectFn = detect_faces,
                             ) -> dict[str, Any]:
    """Native-ROI versus aligned, and each feature block on its own.

    One pass per preprocessing; the block ablations reuse the native pass's
    matrix by slicing it, because refitting on a subset of columns is the
    whole difference between them.
    """
    rows, keep = _rows(captures)
    net = NPRStatsNet()
    out: dict[str, Any] = {}
    for aligned in (False, True):
        vectors: list[npt.NDArray[np.float64]] = []
        labels: list[int] = []
        groups: list[str] = []
        for path, label, session in rows:
            image = cv2.imread(str(path))
            if image is None or image.shape not in keep:
                continue
            if aligned:
                image = cv2.resize(image, (ALIGNED_SIZE, ALIGNED_SIZE),
                                   interpolation=cv2.INTER_AREA)
            vector = _features(image, detect=detect, net=net, aligned=aligned)
            if vector is None:
                continue
            vectors.append(vector)
            labels.append(label)
            groups.append(session)
        if not vectors:
            out["aligned" if aligned else "native"] = {"measured": False}
            continue
        x = np.asarray(vectors)
        y = np.asarray(labels, dtype=int)
        g = np.asarray(groups)
        out["aligned" if aligned else "native"] = {"measured": True,
                                                   **_measure(x, y, g)}
        if not aligned:
            phase, spectral = block_views(x)
            out["native_phase_block_only"] = {
                "measured": True, "n_features": int(phase.shape[1]),
                **_measure(phase, y, g)}
            out["native_spectral_block_only"] = {
                "measured": True, "n_features": int(spectral.shape[1]),
                **_measure(spectral, y, g)}
    return out


def per_band(captures: Path, *, detect: DetectFn = detect_faces,
             ) -> dict[str, Any]:
    """The head's held-out scores, split by the quality band of the frame.

    `NPRDetector.min_quality_band` is `"reject"` — the only detector here
    that scores frames every other one refuses — and the argument for it is
    that low quality is CORRELATED with the artefact rather than with
    unreliability: `inswapper_128` pastes back a 128x128 face, so a swap
    blurs, and a sharpness floor reads that blur as a bad capture.

    That argument rests on this table, so this recomputes it rather than
    quoting it. Frame-level, because a band is a property of a frame and a
    session can hold frames of several.
    """
    rows, keep = _rows(captures)
    from training.fit_vcip import features_for

    net = NPRStatsNet()
    vectors: list[npt.NDArray[np.float64]] = []
    labels: list[int] = []
    groups: list[str] = []
    bands: list[str] = []
    for path, label, session in rows:
        image = cv2.imread(str(path))
        if image is None or image.shape not in keep:
            continue
        got = features_for(path, detect=detect, net=net)
        if got is None:
            continue
        vector, band = got
        vectors.append(vector)
        labels.append(label)
        groups.append(session)
        bands.append(band)
    if not vectors:
        return {"measured": False, "why": "no usable face"}

    y = np.asarray(labels, dtype=int)
    scores = held_out_scores(np.asarray(vectors), y, np.asarray(groups))
    band_of = np.asarray(bands)
    out: dict[str, Any] = {"measured": True, "basis": "frame"}
    for band in sorted(set(bands)):
        here = (band_of == band) & np.isfinite(scores)
        s, labels_here = scores[here], y[here]
        row: dict[str, Any] = {
            "n": int(here.sum()),
            "genuine": int((labels_here == 0).sum()),
            "swapped": int((labels_here == 1).sum()),
        }
        if len(np.unique(labels_here)) > 1:
            row["auc"] = float(auc(s, labels_here))
            row["caught_at_fpr_0.0"] = float(tpr_at_fpr(s, labels_here, 0.0))
        else:
            # One class in a band is not a failure — it is what a band with
            # 14 genuine frames looks like when the fold boundary moves —
            # but an AUC over it would be a number with no meaning.
            row["auc"] = None
            row["caught_at_fpr_0.0"] = None
        out[band] = row
    return out


def main(argv: Sequence[str] | None = None, *,
         detect: DetectFn = detect_faces) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("bench/vcip_controls.json"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.captures.is_dir():
        logger.error("no capture directory at %s", args.captures)
        return 1

    report: dict[str, Any] = {
        "controls": "bench.vcip_controls",
        "captures": str(args.captures),
        "encoder_shortcut": encoder_shortcut(args.captures),
        "matched_encoder": matched_encoder(args.captures, detect=detect),
        "preprocessing_and_blocks": preprocessing_and_blocks(
            args.captures, detect=detect),
        "per_band": per_band(args.captures, detect=detect),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False))
    logger.info("wrote %s", args.out)
    shortcut = report["encoder_shortcut"]
    matched = report["matched_encoder"]
    if shortcut.get("measured"):
        logger.info("encoder shortcut: frame AUC %.3f from the header alone",
                    shortcut["frame_auc"])
    if matched.get("measured"):
        logger.info("head with the shortcut removed: session AUC %.3f, "
                    "caught %.1f%% at a zero false-alarm budget",
                    matched["auc"], 100 * matched["caught_at_fpr_0.0"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
