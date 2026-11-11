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
from dataclasses import dataclass
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
from dfd.detectors.clip_vit import CLIPVisualTower, preprocess
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

#: Phase 2's fitted encoder. A COMPLETE state_dict, never a delta: a complete
#: one loads through `loading.load_model`'s strict secure path with the
#: existing `CLIPVisualTower.vit_l14` factory and no loading machinery
#: changes at all. A delta would need a partial, non-strict load, which is
#: the kind of special case that eventually loads half a model in silence.
DEFAULT_LNTUNED_ENCODER = Path("assets/models/clip_vit_l14_lntuned.pt")

#: The head fitted JOINTLY with that encoder, and a separate file from the
#: frozen probe's. Phase 2's head reads a different encoder's embeddings, so
#: the two are not interchangeable -- and `--out`'s default pointing at the
#: SHIPPED head would have let one overnight run overwrite the only artefact
#: that is currently measured.
DEFAULT_LNTUNED_HEAD = Path("assets/models/gend_clip_lntuned.pt")

#: Phase 2 defaults. Five epochs because `bench/clip_latency.md` measures one
#: epoch over the 9,887-record corpus at 2.3 hours of forward-equivalent work
#: on the deployment instance, which makes five an overnight job and ten a
#: weekend. AdamW at 1e-4 is the conventional starting point for tuning
#: normalisation parameters alone; it is a default, not a measured optimum,
#: and the per-epoch holdout curve in the report is how you would find out.
DEFAULT_EPOCHS = 5
DEFAULT_LR = 1e-4

#: Sixteen 224px crops through a ViT-L/14 forward AND backward hold roughly
#: 5 GB of activations on the 15 GB instance. Lower it before concluding the
#: box cannot run Phase 2.
DEFAULT_TUNE_BATCH = 16


def encoder_cache_id(path: str | Path) -> str:
    """The `FeatureCache` id for an encoder FILE.

    One authority for the id, because the cache is content-addressed on the
    IMAGE and the encoder id is the only thing separating two models'
    embeddings of the same pixels. Keyed on the file NAME rather than on the
    file's hash: hashing 1.2 GB on every run costs seconds and would
    invalidate every entry already on the instance. The trap that leaves
    open, stated rather than hidden -- overwriting an encoder file IN PLACE
    with different weights keeps the old id and serves the old embeddings,
    and the only symptom is a disappointing AUC. Nothing here writes to the
    frozen encoder's path, and the manifest registers each file separately.
    """
    return f"clip-l14:{Path(path).name}"


#: The tuned encoder's cache id. Distinct from the frozen encoder's, which is
#: the whole point: sharing it would serve Phase 1's embeddings to Phase 2.
ENCODER_ID_LNTUNED = encoder_cache_id(DEFAULT_LNTUNED_ENCODER)


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


@dataclass(frozen=True)
class TuningTrace:
    """What an overnight run did, so it does not have to be repeated to find out.

    The per-epoch holdout curve is the only evidence that early stopping did
    anything at all. Recomputing a final AUC afterwards cannot recover it,
    and a 12-hour job whose shape is unknowable is a job somebody runs twice.
    """

    epochs: int
    lr: float
    batch_size: int
    seed: int
    n_train: int
    n_holdout: int
    n_tuned_parameters: int
    n_frozen_parameters: int
    #: One entry per epoch, in order. `None` where the holdout carried a
    #: single class and no AUC exists -- recorded rather than filled with
    #: 0.5, which would read as a measurement.
    holdout_auc: tuple[float | None, ...]
    #: 1-based, and 0 when no epoch scored at all.
    selected_epoch: int
    selected_auc: float | None
    train_loss: tuple[float, ...]


def _embed_for_training(tower: CLIPVisualTower,
                        images: Sequence[npt.NDArray[np.uint8]]) -> torch.Tensor:
    """L2-normalised embeddings WITH gradients attached.

    `dfd.detectors.gend.embed` is this same arithmetic under
    `torch.no_grad()`, and it cannot be reused here because a cached or
    detached embedding cannot be back-propagated through -- which is the
    whole reason Phase 2 costs 2.3 hours an epoch where Phase 1 cost one pass.
    THE TWO MUST AGREE: a tuner that normalised differently from the detector
    would fit coefficients the detector never sees. `tests/test_fit_gend.py`
    asserts they return the same vectors on the same crops; that test is the
    guard, not this comment.
    """
    size = int(tower.image_size)
    batch = torch.cat([preprocess(c, size=size) for c in images])
    out: torch.Tensor = tower(batch)
    return out / out.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _score_rows(tower: CLIPVisualTower, head: GendHead,
                records: Sequence[dict[str, Any]], rows: Sequence[int],
                *, batch_size: int) -> npt.NDArray[np.float32]:
    """P(fake) for `rows`, encoder in eval mode and no gradients."""
    was_training = tower.training
    tower.eval()
    head.eval()
    chunks: list[npt.NDArray[np.float32]] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            idx = rows[start:start + batch_size]
            features = _embed_for_training(tower, [records[i]["image"] for i in idx])
            probs = torch.softmax(head(features), dim=1)[:, 1]
            chunks.append(probs.numpy().astype(np.float32))
    tower.train(was_training)
    head.train(was_training)
    if not chunks:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(chunks)


def tune_layernorms(
    tower: CLIPVisualTower,
    records: Sequence[dict[str, Any]],
    *,
    epochs: int,
    lr: float,
    batch_size: int = DEFAULT_TUNE_BATCH,
    seed: int = 0,
    holdout: float = DEFAULT_HOLDOUT,
    version: str | None = None,
) -> tuple[CLIPVisualTower, GendHead, TuningTrace]:
    """Tune every LayerNorm and the head jointly; freeze everything else.

    102,400 parameters out of ~304M on ViT-L/14 -- 0.03%. The restriction is
    not thrift: tuning the attention and MLP weights on 2,000 photographs of
    FairFace would fit the corpus rather than the task, and it would make the
    shipped encoder a derived work of material whose biometric-consent
    question (owner-action item 6) is still open in a way a renormalisation
    of OpenAI's own features is not.

    `tower` IS MUTATED and also returned; the caller gets the same object.
    Cloning 1.2 GB of frozen weights to leave the input untouched would
    double the peak memory of the only machine that can run this at all.

    Early stopping is "keep the best holdout epoch", not a patience counter.
    At five epochs a patience rule never fires, and the thing it protects
    against -- shipping an overfitted last epoch because the run was too
    expensive to repeat -- is exactly what keeping the best epoch prevents.

    Raises:
        ValueError: if the corpus is not two-class, before any work is done.
    """
    labels = np.asarray([r["label"] for r in records], dtype=np.int64)
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("training corpus must contain both labels, got "
                         f"{sorted(set(labels.tolist()))}")

    torch.manual_seed(seed)
    tunable = set(tower.layernorm_parameter_names())
    for name, parameter in tower.named_parameters():
        parameter.requires_grad_(name in tunable)
    n_tuned = sum(p.numel() for n, p in tower.named_parameters() if n in tunable)
    n_frozen = sum(p.numel() for p in tower.parameters()) - n_tuned

    head = GendHead(in_dim=int(tower.proj.shape[1]))
    if version is not None:
        head.set_version(version)
    optimiser = torch.optim.AdamW(
        [p for p in tower.parameters() if p.requires_grad] + list(head.parameters()),
        lr=lr)

    train, test = _split_by_subject(records, holdout=holdout, seed=seed)
    logger.info("tuning %d LayerNorm parameters (%.3f%% of %d) over %d records, "
                "%d held out by subject, %d epochs at lr %g",
                n_tuned, 100.0 * n_tuned / (n_tuned + n_frozen),
                n_tuned + n_frozen, len(train), len(test), epochs, lr)

    rng = np.random.default_rng(seed)
    # Only the tuned tensors are kept, because everything else is frozen and
    # byte-identical in every epoch. Keeping the whole state_dict would hold
    # a second 1.2 GB copy for nothing.
    best_ln: dict[str, torch.Tensor] | None = None
    best_head: dict[str, torch.Tensor] | None = None
    best_auc: float | None = None
    best_epoch = 0
    aucs: list[float | None] = []
    losses: list[float] = []

    for epoch in range(1, epochs + 1):
        tower.train()
        head.train()
        order = rng.permutation(len(train))
        total, seen = 0.0, 0
        for start in range(0, len(order), batch_size):
            rows = [train[i] for i in order[start:start + batch_size]]
            features = _embed_for_training(tower, [records[i]["image"] for i in rows])
            loss = torch.nn.functional.cross_entropy(
                head(features), torch.from_numpy(labels[rows]))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += float(loss.item()) * len(rows)
            seen += len(rows)
        losses.append(total / seen if seen else float("nan"))

        held = labels[test]
        auc: float | None = None
        if len(set(held.tolist())) == 2:
            probs = _score_rows(tower, head, records, test, batch_size=batch_size)
            auc = float(roc_auc_score(held, probs))
        aucs.append(auc)
        logger.info("epoch %d/%d: train loss %.4f, holdout AUC %s",
                    epoch, epochs, losses[-1], auc)
        if auc is not None and (best_auc is None or auc > best_auc):
            best_auc, best_epoch = auc, epoch
            state = tower.state_dict()
            best_ln = {k: state[k].detach().clone() for k in tunable}
            best_head = {k: v.detach().clone() for k, v in head.state_dict().items()}

    if best_ln is not None and best_head is not None:
        # Restored through a FULL, strict load rather than `strict=False`
        # over the subset: a partial load is the failure mode this task's
        # whole artefact design exists to avoid, and allowing it here would
        # be the one place it is permitted.
        restored = tower.state_dict()
        restored.update(best_ln)
        tower.load_state_dict(restored)
        head.load_state_dict(best_head)
        logger.info("kept epoch %d (holdout AUC %.4f) of %d", best_epoch,
                    best_auc if best_auc is not None else float("nan"), epochs)
    else:
        # Every epoch's holdout was single-class, so nothing was comparable
        # and the last epoch is what the caller gets. Said out loud because
        # the artefact would otherwise look selected when it was not.
        logger.warning("no epoch could be scored (the holdout carried one "
                       "class); keeping the LAST epoch, not the best")

    tower.eval()
    head.eval()
    trace = TuningTrace(
        epochs=epochs, lr=lr, batch_size=batch_size, seed=seed,
        n_train=len(train), n_holdout=len(test),
        n_tuned_parameters=n_tuned, n_frozen_parameters=n_frozen,
        holdout_auc=tuple(aucs), selected_epoch=best_epoch,
        selected_auc=best_auc, train_loss=tuple(losses))
    return tower, head, trace


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
    parser.add_argument("--out", type=Path, default=None,
                        help="where the head is written. Defaults to "
                             f"{DEFAULT_LNTUNED_HEAD} under "
                             "--tune-layernorms and assets/models/"
                             "gend_clip.pt otherwise -- the two heads read "
                             "different encoders and are not "
                             "interchangeable.")
    parser.add_argument("--report", type=Path,
                        default=Path("bench/gend_fit.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout", type=float, default=DEFAULT_HOLDOUT)
    parser.add_argument("--tune-layernorms", action="store_true",
                        help="PHASE 2. Tune the encoder's LayerNorms jointly "
                             "with the head instead of fitting a probe over "
                             "frozen embeddings. The feature cache cannot "
                             "help -- a cached embedding cannot be "
                             "back-propagated through -- so this costs one "
                             "forward AND backward pass per record per "
                             "epoch. Measured: ~2.3 hours an epoch over "
                             "9,887 records on the deployment instance, "
                             "~6x that on a workstation whose torch is a "
                             "CUDA build (bench/clip_latency.md).")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TUNE_BATCH)
    parser.add_argument("--encoder-out", type=Path,
                        default=DEFAULT_LNTUNED_ENCODER,
                        help="where --tune-layernorms writes the COMPLETE "
                             "tuned encoder state_dict (~1.2 GB)")
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = (DEFAULT_LNTUNED_HEAD if args.tune_layernorms
                    else Path("assets/models/gend_clip.pt"))

    records, skipped = build_training_records(
        args.root, limit=args.limit, offset=args.offset, seed=args.seed,
        detect=detect)
    if not records:
        logger.error("no training records built from %s; skip tally: %s",
                     args.root, skipped)
        return 1

    tower = load_model(args.encoder, model_factory=CLIPVisualTower.vit_l14)
    labels = np.asarray([r["label"] for r in records], dtype=np.int64)
    train, test = _split_by_subject(records, holdout=args.holdout,
                                    seed=args.seed)
    trace: TuningTrace | None = None
    cache_stats = {"hits": 0, "misses": 0}

    if args.tune_layernorms:
        version = f"0.2.0-lntuned-fairface{args.limit}"
        tower, head, trace = tune_layernorms(
            tower, records, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, seed=args.seed, holdout=args.holdout,
            version=version)
        args.encoder_out.parent.mkdir(parents=True, exist_ok=True)
        # The WHOLE encoder, so it loads back through the same strict
        # `CLIPVisualTower.vit_l14` factory with no loading changes at all.
        torch.save(tower.state_dict(), args.encoder_out)
        logger.info("wrote the tuned encoder to %s", args.encoder_out)
        probs = _score_rows(tower, head, records, test,
                            batch_size=args.batch_size)
    else:
        tower.eval()
        cache = FeatureCache(args.cache,
                             encoder_id=encoder_cache_id(args.encoder))
        features = cache.embed_all(tower, [r["image"] for r in records])
        cache_stats = dict(cache.stats)
        logger.info("embeddings: %d hits, %d misses", cache.stats["hits"],
                    cache.stats["misses"])
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
        "cache_hits": cache_stats["hits"],
        "cache_misses": cache_stats["misses"],
        "encoder": str(args.encoder),
        "encoder_out": (str(args.encoder_out) if args.tune_layernorms
                        else None),
        # Phase 2's shape, or null for a Phase 1 probe. The per-epoch holdout
        # curve is here so an overnight run never has to be repeated to find
        # out whether early stopping did anything.
        "tuning": None if trace is None else {
            "epochs": trace.epochs,
            "lr": trace.lr,
            "batch_size": trace.batch_size,
            "n_tuned_parameters": trace.n_tuned_parameters,
            "n_frozen_parameters": trace.n_frozen_parameters,
            "holdout_auc_per_epoch": list(trace.holdout_auc),
            "train_loss_per_epoch": list(trace.train_loss),
            "selected_epoch": trace.selected_epoch,
            "selected_auc": trace.selected_auc,
        },
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
