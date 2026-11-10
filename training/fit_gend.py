"""Fit slot D's head: a Linear(768, 2) over frozen CLIP embeddings.

WHAT TRAINS ON WHAT. Only the FairFace session directory
(`training/export_fairface.py`'s output, CC BY 4.0) and composites this repo makes from it. The
v-CIP capture corpus and its `inswapper_128` swaps are EVALUATION ONLY and are not reachable from
here: this module takes a `--root` that is the FairFace directory, and `corpora.sbi` hard-refuses
any swapped session with `EvaluationOnlySessionError` if one ever were. No generator weight file,
no licensed deepfake dataset, and nothing derived from FaceForensics++ contributes a number.

FIVE GENERATOR FAMILIES, which is what lets LOGO fold five ways in Task 7: `sbi` (a self-blend of
one photograph) and the four `corpora.swaps` techniques (one identity composited into another's
photograph). SFHQ is deliberately NOT here -- it saturates (measured: 500 images reach what
204,716 reach) and it cannot be paired, so it is a held-out control rather than a training family.

PAIRING AND RE-ENCODING ARE NOT OPTIONAL (spec §6.3). Every fake is made from a specific real
crop, and both halves pass through the identical encode. One JPEG header byte labels 99.2% of
this project's capture corpus; a detector handed reals at one encode and fakes at another learns
the encoder and reports it as detection.
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
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from corpora.captures import load_capture_sessions
from corpora.face_pool import DetectFn, build_face_pool
from corpora.sbi import build_sbi_corpus
from corpora.swap_corpus import build_swap_corpus
from dfd.detectors.clip_vit import CLIPVisualTower
from dfd.detectors.gend import GendHead
from dfd.detectors.loading import load_model
from dfd.faces import detect_faces
from training.clip_features import FeatureCache
from training.fit_blend import _json_safe

logger = logging.getLogger(__name__)

#: Both halves of every pair go through this, once. The value matches the
#: existing swap pipeline's OpenCV quality 95 so that a future corpus mixing
#: the two does not acquire a header shortcut at the join.
REENCODE_QUALITY = 95

#: What every record declares, so the one-encode property is checkable in
#: the records themselves rather than only in this file's history.
COMPRESSION = f"jpeg_q{REENCODE_QUALITY}"

#: The id `corpora.sbi` gives its fakes, repeated here so the family set is
#: readable in one place.
SBI_GENERATOR = "sbi"

#: Default held-out fraction, by SUBJECT. Matches `training/fit_blend.py`.
DEFAULT_HOLDOUT = 0.2


def reencode(img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Encode to JPEG at a fixed quality and decode back.

    This is not compression for storage -- the arrays stay in memory. It is
    for making the encode history of reals and fakes IDENTICAL. 99.2% of this
    project's capture corpus is labelled by one byte of the JPEG header
    (bench/vcip_controls.py), which is what happens when the two halves of a
    corpus are written by different encoders.
    """
    ok, buf = cv2.imencode(".jpg", img[:, :, ::-1],
                           [int(cv2.IMWRITE_JPEG_QUALITY), REENCODE_QUALITY])
    if not ok:
        raise ValueError("JPEG encode failed")
    out: npt.NDArray[np.uint8] = cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1]
    return out


def build_training_records(
    root: str | Path,
    *,
    limit: int,
    offset: int,
    seed: int,
    detect: DetectFn = detect_faces,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Reals and all five fake families, every record through one encode.

    Returns `(records, skipped)` in `bench.runner`'s record shape.

    SUBJECT IDS ARE COUPLE IDS, INCLUDING FOR THE SELF-BLENDS. `corpora.sbi`
    files a sample under the photograph it came from, while
    `corpora.swap_corpus` files one under the COUPLE. Left alone, one
    photograph would carry two different `subject_id`s, and a
    subject-disjoint split would happily put its self-blend pair in train and
    its couple in test -- putting the same face, at the same alignment, on
    both sides. That is the contamination this project has already paid for
    once (blend_seam's fabricated 0.923). Couples are disjoint
    (`swap_corpus._couples`), so adopting the couple id for both families
    makes one photograph exactly one subject.
    """
    root = Path(root)
    skipped: dict[str, int] = {}

    swap_records, swap_skipped = build_swap_corpus(
        root, limit=limit, offset=offset, seed=seed, detect=detect)
    skipped.update(swap_skipped)

    # person_id -> couple. `_record`'s sample_id for a real is
    # "<couple>/real/<person_id>", which is the only place the join key is
    # exposed; reading it here rather than re-deriving the pairing keeps one
    # authority for who is coupled with whom.
    couple_of: dict[str, str] = {}
    for record in swap_records:
        if record["label"] == 0:
            couple_of[record["sample_id"].rsplit("/", 1)[1]] = record["subject_id"]

    sessions = load_capture_sessions(root)
    # Defence in depth, not the only guard: build_sbi_corpus hard-refuses any
    # crop from a swapped session with EvaluationOnlySessionError.
    genuine = [s for s in sessions if not s.swapped]
    chosen = sorted(genuine, key=lambda s: s.session_id)[offset:offset + limit]
    crops, pool_skipped = build_face_pool(chosen, detect=detect)
    for reason, count in pool_skipped.items():
        skipped[reason] = skipped.get(reason, 0) + count

    records: list[dict[str, Any]] = list(swap_records)
    for sample in build_sbi_corpus(crops, seed=seed):
        photograph = sample.context.subject_id
        records.append({
            "sample_id": f"sbi/{sample.sample_id}",
            "source_id": f"sbi/{sample.sample_id}",
            "subject_id": couple_of.get(photograph, f"solo/{photograph}"),
            "stratum": "sbi",
            "generator": sample.context.generator,
            "label": sample.context.label,
            "compression": COMPRESSION,
            "face_detector": "yunet",
            "align": "v1",
            "image": sample.observations[0].payload,
        })

    for record in records:
        record["image"] = reencode(record["image"])
        record["compression"] = COMPRESSION

    logger.info("training corpus: %d records, %d fake, %d subjects; skipped %s",
                len(records), sum(r["label"] for r in records),
                len({r["subject_id"] for r in records}), skipped)
    return records, skipped


def fit_head(features: npt.NDArray[np.float32],
             labels: npt.NDArray[np.int64],
             *, version: str) -> GendHead:
    """Logistic regression on cached embeddings, copied into a torch Linear.

    sklearn rather than a torch training loop because the fitted object is
    1,538 numbers over an L2-normalised 768-d input: it is a convex problem
    with a closed enough solution that an optimiser's schedule, seed and
    stopping rule would be three more things to defend and nothing to gain.
    The coefficients are copied into `Linear(768, 2)` so the DETECTOR's
    forward path is torch end to end and `loading.load_model`'s secure path
    applies unchanged.

    The two-row layout reproduces sklearn's probabilities exactly:
    softmax([-w.x/2 - b/2, +w.x/2 + b/2])[1] == sigmoid(w.x + b).
    """
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("training corpus must contain both labels, got "
                         f"{sorted(set(labels.tolist()))}")
    clf = LogisticRegression(max_iter=2000).fit(features, labels)
    w = np.asarray(clf.coef_[0], dtype=np.float32)
    b = float(clf.intercept_[0])
    head = GendHead(in_dim=features.shape[1])
    with torch.no_grad():
        head.linear.weight.copy_(torch.from_numpy(np.stack([-w / 2, w / 2])))
        head.linear.bias.copy_(torch.tensor([-b / 2, b / 2]))
    head.set_version(version)
    return head


def _split_by_subject(records: Sequence[dict[str, Any]], *, holdout: float,
                      seed: int) -> tuple[list[int], list[int]]:
    """Row indices for (train, test), disjoint by `subject_id`.

    Subjects, never rows: a couple's six records are six views of two
    photographs, and splitting them apart puts the same face on both sides.
    """
    subjects = sorted({r["subject_id"] for r in records})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(subjects))
    n_test = max(1, int(round(len(subjects) * holdout)))
    test_subjects = {subjects[i] for i in order[:n_test]}
    train = [i for i, r in enumerate(records)
             if r["subject_id"] not in test_subjects]
    test = [i for i, r in enumerate(records)
            if r["subject_id"] in test_subjects]
    return train, test


def main(argv: Sequence[str] | None = None, *,
         detect: DetectFn = detect_faces) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path,
                        help="the FairFace session directory")
    parser.add_argument("--limit", type=int, required=True,
                        help="photographs to consume. THIS IS THE TRAINING "
                             "WINDOW and must match bench/eval_swaps.py's "
                             "TRAINING_WINDOWS entry for the version written")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--cache", required=True, type=Path,
                        help="embedding cache root, OUTSIDE the repository")
    parser.add_argument("--encoder", type=Path,
                        default=Path("assets/models/clip_vit_l14_visual.pt"))
    parser.add_argument("--out", type=Path,
                        default=Path("assets/models/gend_clip.pt"))
    parser.add_argument("--report", type=Path,
                        default=Path("bench/gend_fit.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout", type=float, default=DEFAULT_HOLDOUT)
    args = parser.parse_args(argv)

    records, skipped = build_training_records(
        args.root, limit=args.limit, offset=args.offset, seed=args.seed,
        detect=detect)
    if not records:
        logger.error("no training records built from %s; skip tally: %s",
                     args.root, skipped)
        return 1

    tower = load_model(args.encoder, model_factory=CLIPVisualTower.vit_l14)
    tower.eval()
    cache = FeatureCache(args.cache, encoder_id=f"clip-l14:{args.encoder.name}")
    features = cache.embed_all(tower, [r["image"] for r in records])
    labels = np.asarray([r["label"] for r in records], dtype=np.int64)
    logger.info("embeddings: %d hits, %d misses", cache.stats["hits"],
                cache.stats["misses"])

    train, test = _split_by_subject(records, holdout=args.holdout,
                                    seed=args.seed)
    version = f"0.1.0-probe-fairface{args.limit}"
    head = fit_head(features[train], labels[train], version=version)

    with torch.no_grad():
        probs = torch.softmax(head(torch.from_numpy(features[test])),
                              dim=1)[:, 1].numpy()
    held_out = labels[test]
    # An in-domain held-out AUC is NOT a result. It is a smoke test that the
    # pipeline is wired, and it belongs in no evidence card.
    auc = (float(roc_auc_score(held_out, probs))
           if len(set(held_out.tolist())) == 2 else None)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), args.out)

    families: dict[str, int] = {}
    for record in records:
        key = record["generator"] or "real"
        families[key] = families.get(key, 0) + 1
    report = {
        "version": version,
        "seed": args.seed,
        "root": str(args.root),
        "offset": args.offset,
        # The training window: the number in the version string, the number
        # in bench/eval_swaps.py's TRAINING_WINDOWS and this must agree.
        "n_photographs": args.limit,
        "n_records": len(records),
        "n_subjects": len({r["subject_id"] for r in records}),
        "families": families,
        "skipped": skipped,
        "cache_hits": cache.stats["hits"],
        "cache_misses": cache.stats["misses"],
        "n_train": len(train),
        "n_test": len(test),
        "held_out_auc": auc,
        "held_out_auc_note": (
            "IN-DOMAIN. A smoke test that the pipeline is wired, not a "
            "result. It does not appear in bench/evidence_card.json and "
            "must not be quoted as slot D's performance."),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(_json_safe(report), indent=2,
                                      sort_keys=True, allow_nan=False) + "\n")
    logger.info("wrote %s (version %s); in-domain held-out AUC %s over %d rows",
                args.out, version, auc, len(test))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
