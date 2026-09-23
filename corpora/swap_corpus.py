"""A licence-clean, multi-generator evaluation corpus built from FairFace.

**The first corpus in this project that leave-one-generator-out can fold.**
Every other one carries a single generator — `sbi` for self-blends,
`df40_unknown_mixture` for the repackaging — and `bench.protocol.logo_splits`
correctly refuses a corpus whose only generator is the one being held out. So
the number spec §8.1 calls the only one that predicts field performance has
never been computed here. Four techniques (`corpora.swaps`) is what changes
that.

**COUPLES, NOT PHOTOGRAPHS.** Each `subject_id` is a PAIR of FairFace people,
and every record made from that pair — both reals and all four fakes — carries
it. This is not bookkeeping convenience; it is what makes an identity-disjoint
split possible at all. A swap composites person A's face into person B's
photograph, so the fake carries A's identity (measured: cosine 0.56 to A
against 0.12 to B) while carrying B's lighting, background and camera. Filing
it under either person alone would put that identity on both sides of a
subject-partitioned split. Filing the couple as one subject moves all six
records together, and `bench.guards.check_subject_partition` then verifies
against pixels that different couples really are different people.

**Pairing is WITHIN a stratum** — same FairFace race and gender — for two
reasons. It makes the swap physically plausible, and it gives every record in
a couple one unambiguous `stratum`, which is what acceptance criterion 11
needs and what DF40 cannot supply at all.

**Read `corpora/swaps.py` before reporting any number from this corpus.**
These are classical compositing swaps, not generator output. The corpus can
refute a detector and can support LOGO across these four techniques; it cannot
show that a detector beats FSGAN.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dfd.faces import FaceBox, align, clamp_roi, detect_faces

from .face_pool import DetectFn
from .swaps import TECHNIQUES, rng_for, swap

logger = logging.getLogger(__name__)

#: Every record is aligned to this square, matching `corpora.face_pool` and
#: the resolution `dfd.detectors.blend`'s features assume.
CROP_SIZE = 224

#: Recorded on every record. These images are PNG-encoded in memory only and
#: never re-compressed, so claiming a JPEG quality level would be a lie; the
#: benchmark's compression guard reads this and will refuse to report a
#: worst-compression cell rather than invent one.
COMPRESSION = "none"

FACE_DETECTOR = "yunet"
ALIGN = "v1"

NO_FACE = "no_face"
DEGENERATE_BOX = "degenerate_box"
UNREADABLE = "unreadable"
SWAP_FAILED = "swap_failed"
NO_STRATUM = "no_stratum"


@dataclass(frozen=True)
class _Face:
    """One FairFace photograph, detected and ready to composite."""
    person_id: str
    stratum: str
    frame: np.ndarray
    box: FaceBox


def _stratum(meta: dict[str, Any]) -> str | None:
    demo = meta.get("demographics") or {}
    race, gender = demo.get("race"), demo.get("gender")
    return f"{race}|{gender}" if race and gender else None


def _load_faces(root: Path, *, limit: int | None, detect: DetectFn,
                skipped: dict[str, int]) -> Iterator[_Face]:
    sessions = sorted(d for d in root.iterdir() if d.is_dir())
    if limit is not None:
        sessions = sessions[:limit]
    for folder in sessions:
        meta_path = folder / "results.json"
        frame_path = folder / "frame_00.jpg"
        if not meta_path.is_file() or not frame_path.is_file():
            skipped[UNREADABLE] = skipped.get(UNREADABLE, 0) + 1
            continue
        stratum = _stratum(json.loads(meta_path.read_text()))
        if stratum is None:
            # Not skipped quietly: a record with no stratum would silently
            # shrink criterion 11's denominator.
            skipped[NO_STRATUM] = skipped.get(NO_STRATUM, 0) + 1
            continue
        bgr = cv2.imread(str(frame_path))
        if bgr is None:
            skipped[UNREADABLE] = skipped.get(UNREADABLE, 0) + 1
            continue
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        boxes = detect(frame)
        if not boxes:
            skipped[NO_FACE] = skipped.get(NO_FACE, 0) + 1
            continue
        box = max(boxes, key=lambda b: b.score)
        if clamp_roi(frame.shape, box) is None:
            skipped[DEGENERATE_BOX] = skipped.get(DEGENERATE_BOX, 0) + 1
            continue
        yield _Face(person_id=folder.name, stratum=stratum, frame=frame, box=box)


def _couples(faces: Sequence[_Face], *, seed: int) -> list[tuple[_Face, _Face]]:
    """Disjoint pairs, both members from one stratum.

    Shuffled within each stratum before pairing so that couples are not an
    artefact of directory order — FairFace's shards are not randomised, and
    adjacent files can share a source photograph's origin.
    """
    by_stratum: dict[str, list[_Face]] = {}
    for face in faces:
        by_stratum.setdefault(face.stratum, []).append(face)

    out: list[tuple[_Face, _Face]] = []
    for stratum in sorted(by_stratum):
        members = by_stratum[stratum]
        rng = np.random.default_rng([seed, len(members)])
        order = rng.permutation(len(members))
        # An odd stratum leaves one face unpaired; it is dropped rather than
        # paired across strata, which would break the one-stratum-per-couple
        # property criterion 11 depends on.
        for i in range(0, len(order) - 1, 2):
            out.append((members[order[i]], members[order[i + 1]]))
    logger.info("couples: %d from %d faces across %d strata",
                len(out), len(faces), len(by_stratum))
    return out


def build_swap_corpus(
    root: str | Path,
    *,
    limit: int | None = None,
    seed: int = 0,
    detect: DetectFn = detect_faces,
    techniques: Sequence[str] = TECHNIQUES,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build the corpus as `bench.runner` records.

    Args:
        root: the FairFace session directory (as `training/export_fairface.py`
            writes it): one folder per photograph, holding `frame_00.jpg` and
            a `results.json` carrying demographics.
        limit: cap on photographs read, before pairing.
        seed: pairing and jitter seed. The same seed yields the same corpus.
        detect: face detector, injected so tests need no weight file.
        techniques: which generators to emit. Defaults to all four; a caller
            wanting a two-generator corpus passes two, and LOGO still folds.

    Returns:
        A (records, skipped) pair. Each couple contributes two REAL records
        and one FAKE per technique, all sharing one `subject_id` and one
        `stratum`.

    Raises:
        FileNotFoundError: if `root` is not a directory. A corpus builder
            that returns an empty list for a mistyped path invites the caller
            to conclude the corpus is empty rather than the path wrong.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"no FairFace session directory at {root}")

    skipped: dict[str, int] = {}
    faces = list(_load_faces(root, limit=limit, detect=detect, skipped=skipped))
    records: list[dict[str, Any]] = []

    for source, target in _couples(faces, seed=seed):
        couple = f"{source.person_id}+{target.person_id}"
        for face in (source, target):
            records.append(_record(
                sample_id=f"{couple}/real/{face.person_id}",
                subject_id=couple, stratum=face.stratum, generator=None,
                label=0, image=align(face.frame, face.box, size=CROP_SIZE)))

        for technique in techniques:
            result = swap(source.frame, source.box, target.frame, target.box,
                          technique, rng_for(couple, technique, seed=seed))
            if result is None:
                skipped[SWAP_FAILED] = skipped.get(SWAP_FAILED, 0) + 1
                continue
            # Detected afresh on the composite, not reused from the target:
            # a swap moves the face, and aligning a fake with the target's
            # box would crop it differently from how the same pipeline crops
            # a real — a preprocessing difference perfectly correlated with
            # the label, which is the defect this whole corpus exists to
            # avoid.
            boxes = detect(result.image)
            if not boxes:
                skipped[NO_FACE] = skipped.get(NO_FACE, 0) + 1
                continue
            box = max(boxes, key=lambda b: b.score)
            if clamp_roi(result.image.shape, box) is None:
                skipped[DEGENERATE_BOX] = skipped.get(DEGENERATE_BOX, 0) + 1
                continue
            records.append(_record(
                sample_id=f"{couple}/{technique}",
                subject_id=couple, stratum=target.stratum,
                generator=technique, label=1,
                image=align(result.image, box, size=CROP_SIZE)))

    logger.info("swap corpus: %d records (%d fake) from %d faces, skipped %s",
                len(records), sum(r["label"] for r in records), len(faces),
                skipped or "nothing")
    return records, skipped


def _record(*, sample_id: str, subject_id: str, stratum: str,
            generator: str | None, label: int, image: np.ndarray) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        # One photograph per record, so each is its own source. Recorded
        # explicitly rather than aliased to sample_id — `check_video_level`
        # refuses a corpus where the two are the same list, and rightly.
        "source_id": sample_id,
        "subject_id": subject_id,
        "stratum": stratum,
        "generator": generator,
        "label": label,
        "compression": COMPRESSION,
        "face_detector": FACE_DETECTOR,
        "align": ALIGN,
        "image": image,
    }
