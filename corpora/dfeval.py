"""Deepfake-Eval-2024 — an in-the-wild benchmark — as benchmark records.

Deepfake-Eval-2024 (Chandra et al.) is a 2024 collection of 1,975 images
(767 real / 1,208 fake) pulled from 88 web domains — ordinary social-media
and news posts, not a lab-generated forensics set. That is the entire reason
it is worth loading: every other corpus this project reads is studio-clean,
and the gap this harness exists to measure (spec §6) is between a lab number
and what happens when the input is whatever a platform's re-encode pipeline
produced. See `assets/manifest.yaml` once the download lands (Task 7) for the
registered licence entry; until then, treat this data the way
`assets/manifest.yaml`'s own header treats anything unregistered — fails
closed, non-commercial, EVALUATION ONLY. It must never be used to fit a
shipping detector: the dataset carries no per-item generator attribution (see
`DFEVAL_GENERATOR` below), so there is no way to know what a model trained on
it would actually be learning to recognise.

**`_manifest_rows` is written blind, on purpose.** The dataset is gated on
HuggingFace and nobody on this project has been granted access yet. Every
test below drives this loader through a synthetic fixture in the layout that
function assumes (one `metadata.csv` beside the image files, columns
`item_id,filename,label,domain`) rather than against the real archive. When
the gated download lands, reconciling the assumed layout against the real one
is a change to `_manifest_rows` alone — every other function in this module
only consumes `DFEvalRow` values and does not care where they came from.

**Images with no detected face are KEPT, not skipped.** Every other corpus
here is a face crop by construction; this one is raw in-the-wild media, and
some of it has no face for a detector to find at all — a screenshot, a
caption, a non-facial frame from a video. Dropping those images would hide
exactly the abstention gap this harness is built to report, so they stay in
the corpus with `has_face=False` and `roi=None` rather than vanishing from
the denominator.
"""
from __future__ import annotations

import csv
import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from dfd.faces import FaceBox, clamp_roi, detect_faces

from .face_pool import DUPLICATE, UNREADABLE, DetectFn

__all__ = [
    "ALIGN", "COMPRESSION", "CONTRADICTORY_DUPLICATE", "DFEVAL_GENERATOR",
    "DUPLICATE", "DFEvalRow", "FACE_DETECTOR", "UNLABELLED", "UNREADABLE",
    "load_dfeval_images",
]

logger = logging.getLogger(__name__)

#: The single generator id every fake carries. One value, so a LOGO split is
#: refused by `bench.protocol` rather than fabricated -- the same device
#: `corpora.df40` uses, and for the same reason. The name says outright that
#: the techniques are unknown, because they are: Deepfake-Eval-2024 carries
#: no per-item generator labels. Inventing SEVERAL ids would be the lie;
#: inventing one that admits ignorance is what makes the refusal work.
DFEVAL_GENERATOR = "dfeval_unknown_mixture"

#: Recorded rather than guessed. The images arrive from 88 web domains at
#: whatever quality each platform re-encoded them to, and `corpora.df40` and
#: `corpora.sfhq` both set the precedent: an invented "c23" passes
#: `check_compression_coverage` by lying to it.
COMPRESSION = "unknown"

FACE_DETECTOR = "yunet_2023mar"
#: Not an alignment at all -- the ROI is passed through at its native
#: resolution and each detector crops as it sees fit. Recorded under the key
#: `check_uniform_preprocessing` reads so that the value is uniform and
#: honest rather than absent.
ALIGN = "native_roi"

#: A manifest row whose label token is neither "real" nor "fake". Counted,
#: never silently coerced to one label or the other: a malformed row is not
#: evidence about either class.
UNLABELLED = "unlabelled"

#: Message prefix for the ValueError raised when one content hash is claimed
#: by both labels. Not a skip reason -- `bench`'s skip tally counts rows this
#: loader can resolve on its own; a labelling contradiction is not one of
#: them, because dropping either row would silently pick a winner between two
#: claims this loader has no basis to arbitrate between.
CONTRADICTORY_DUPLICATE = "contradictory duplicate"

#: Manifest label tokens -> the 0/1 this loader works in. Anything else
#: (missing, blank, a third category the real dataset card does not document
#: yet) maps to `_UNLABELLED_SENTINEL` and is dropped under `UNLABELLED`
#: rather than guessed into one of the two real classes.
_LABEL_TOKENS: dict[str, int] = {"real": 0, "fake": 1}
_UNLABELLED_SENTINEL = -1
_LABEL_NAMES = {0: "real", 1: "fake"}

_MANIFEST_NAME = "metadata.csv"

_GATE_MESSAGE = (
    "Deepfake-Eval-2024 is gated on HuggingFace: request access at "
    "https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024, "
    "accept the dataset's terms, authenticate (huggingface-cli login or "
    "HF_TOKEN), then download it with the gate's own tooling before "
    "pointing `root` at the result"
)


@dataclass(frozen=True)
class DFEvalRow:
    """One manifest row, before any image has been opened."""
    path: Path
    label: int
    domain: str
    item_id: str


def _manifest_rows(root: Path) -> list[DFEvalRow]:
    """Read `root/metadata.csv` -- the ONLY function that knows the layout.

    **This layout is provisional.** The HuggingFace dataset card for
    Deepfake-Eval-2024 does not publish an on-disk structure and the gate
    blocks inspecting the real one, so the shape assumed here -- a flat
    image directory with one CSV beside it, columns `item_id`, `filename`
    (relative to `root`), `label` (`"real"` or `"fake"`), `domain` -- is a
    reasonable guess, not a verified fact. Task 7 reconciles it against the
    actual download, and that reconciliation should touch only this
    function: everything downstream consumes `DFEvalRow` values and has no
    opinion about where they came from.

    Args:
        root: the corpus directory. Already confirmed to exist by the caller.

    Returns:
        One `DFEvalRow` per manifest row, in file order.

    Raises:
        FileNotFoundError: if `root` has no `metadata.csv`.
    """
    manifest = root / _MANIFEST_NAME
    if not manifest.is_file():
        raise FileNotFoundError(
            f"{root} has no {_MANIFEST_NAME}; see corpora.dfeval's module "
            "docstring for the (provisional) layout this loader expects")

    rows: list[DFEvalRow] = []
    with manifest.open(newline="", encoding="utf-8") as fh:
        for record in csv.DictReader(fh):
            token = (record.get("label") or "").strip().lower()
            label = _LABEL_TOKENS.get(token, _UNLABELLED_SENTINEL)
            rows.append(DFEvalRow(
                path=root / record["filename"],
                label=label,
                domain=record["domain"],
                item_id=record["item_id"],
            ))
    return rows


def _sample(rows: Sequence[DFEvalRow], k: int, seed: int) -> list[DFEvalRow]:
    if k >= len(rows):
        return list(rows)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(rows), size=k, replace=False)
    return [rows[int(i)] for i in sorted(idx)]


def _largest_face(boxes: Sequence[FaceBox]) -> FaceBox | None:
    """The face `training.fit_vcip.select_face` would pick.

    Duplicated here rather than imported: `training` already depends on
    `corpora` (`training/fit_vcip.py` imports `corpora.face_pool`), so
    importing `training.fit_vcip` from here would invert that dependency
    and drag torch and scikit-learn into every corpus load for one line of
    arithmetic. The RULE is what must match, not the import path -- largest
    by area, not most confident -- because a face rule that disagrees
    between the fitter and the evaluator is the defect that made every slot
    C number in this repo stale and forced a re-measurement
    (`training/fit_vcip.py:select_face`).
    """
    return max(boxes, key=lambda b: b.w * b.h) if boxes else None


def load_dfeval_images(
    root: str | Path,
    *,
    limit: int | None = None,
    seed: int = 0,
    detect: DetectFn = detect_faces,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Load Deepfake-Eval-2024 images under `root` as benchmark records.

    Unlike every other corpus in this repo, records here are NOT pre-cropped
    to a face. `image` is the whole decoded frame; `roi` and `landmarks` (set
    only when a face was found) tell `bench.runner._observation` where in
    that frame the face is, at its native resolution (`ALIGN ==
    "native_roi"`) -- there is no resize, because resizing in the loader
    would fabricate a scale the real deployment path does not apply.

    Deduplication hashes the DECODED array, not the file bytes: a platform
    that re-encodes identical pixels through a different JPEG quality still
    produces the same decoded image, and hashing the file would miss that.
    The converse gap is real too and not covered here -- a NEAR-duplicate
    (recompressed, resized, lightly cropped) decodes to different bytes and
    this loader will count it as two distinct images, same as
    `corpora.face_pool` and `corpora.sfhq` already do.

    Args:
        root: the corpus directory (see `_manifest_rows` for the layout
            this loader expects beneath it).
        limit: how many records to return; fewer if rows are dropped.
        seed: selects which rows the limit takes.
        detect: face detector. Injected so tests need no weight file.

    Returns:
        A (records, skipped) pair. `skipped` maps a reason constant
        (`UNREADABLE`, `DUPLICATE`, `UNLABELLED`) to a count, and is empty
        when nothing was dropped.

    Raises:
        FileNotFoundError: if `root` is not a directory (naming the
            HuggingFace gate this corpus sits behind) or its manifest lists
            no rows.
        ValueError: if one content hash is claimed by both labels --
            `CONTRADICTORY_DUPLICATE` names the hash and both item ids. This
            refuses the WHOLE corpus rather than dropping one side, because
            there is no basis here to pick a winner between two contradicting
            claims about the same pixels.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory. {_GATE_MESSAGE}")

    rows = _manifest_rows(root)
    if not rows:
        raise FileNotFoundError(f"{root}'s {_MANIFEST_NAME} lists no rows")
    if limit is not None:
        rows = _sample(rows, max(limit, 0), seed)

    records: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    # digest -> (label, item_id) of the row that first claimed it, so a
    # second claim under the opposite label can name what it contradicts.
    seen: dict[str, tuple[int, str]] = {}

    def drop(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for row in rows:
        if row.label not in (0, 1):
            logger.debug("unlabelled manifest row: %s", row.item_id)
            drop(UNLABELLED)
            continue

        bgr = cv2.imread(str(row.path))
        if bgr is None:
            logger.warning("unreadable image: %s", row.path)
            drop(UNREADABLE)
            continue
        frame: npt.NDArray[np.uint8] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        digest = hashlib.sha256(frame.tobytes()).hexdigest()
        if digest in seen:
            prev_label, prev_item_id = seen[digest]
            if prev_label == row.label:
                logger.debug("duplicate image %s (matches %s)",
                             row.item_id, prev_item_id)
                drop(DUPLICATE)
                continue
            raise ValueError(
                f"{CONTRADICTORY_DUPLICATE}: hash {digest} is labelled "
                f"{_LABEL_NAMES[prev_label]} by {prev_item_id!r} and "
                f"{_LABEL_NAMES[row.label]} by {row.item_id!r}; identical "
                "pixels cannot carry opposite labels")
        seen[digest] = (row.label, row.item_id)

        box = _largest_face(detect(frame))
        roi = clamp_roi(frame.shape, box) if box is not None else None
        has_face = roi is not None
        landmarks = box.landmarks if has_face and box is not None else None

        sample_id = f"dfeval/{row.item_id}"
        records.append({
            "sample_id": sample_id,
            # Each image is its own source and its own subject: the
            # manifest carries no video or identity grouping (see the
            # module docstring on what `_manifest_rows` does and does not
            # know). If a future download exposes either, this is the line
            # that must change.
            "source_id": sample_id,
            "subject_id": sample_id,
            "generator": DFEVAL_GENERATOR if row.label == 1 else None,
            "label": row.label,
            "compression": COMPRESSION,
            "face_detector": FACE_DETECTOR,
            "align": ALIGN,
            "image": frame,
            "domain": row.domain,
            "has_face": has_face,
            "roi": roi,
            "landmarks": landmarks,
        })

    logger.info("dfeval: %d records from %s, skipped %s",
                len(records), root, skipped or "nothing")
    return records, skipped
