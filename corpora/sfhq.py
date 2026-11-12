"""SFHQ part 3 — licence-clean synthetic faces, as benchmark records.

`assets/manifest.yaml:sfhq_part3` records what this data is. Two things make
it different from every other fake supply this project has looked at, and both
change what may be claimed from it:

- **It is licence-clean.** No depicted real person exists, so the biometric
  question `fairface_corpus` leaves open does not arise here at all, and the
  declared licence permits commercial use. It is therefore the first fake
  supply that a SHIPPING detector may be fitted on. Read the archive's own
  LICENSE at ingest: the Kaggle page declares CC0 and the upstream GitHub
  repository declares MIT, and this repo has already been bitten once by a
  platform's licence tag (`df40_eval_subset`).
- **Its generator label is real, not guessed.** Part 3 is documented as pure
  StyleGAN2 sampling, so `SFHQ_PT3_GENERATOR` is a fact about the data rather
  than a filename family read hopefully — the thing `corpora.df40` refuses to
  do. That makes these records usable on a leave-one-generator-out axis once
  a second generator exists beside them.

**What it is NOT.** Entire-face synthesis is one of the four DF40 families and
not the one a v-CIP attack usually belongs to: a synthesised face has no
composite boundary, so slot A (`dfd.detectors.blend`) has nothing to find here
by construction. These records are the positive class for a SYNTHESIS
detector, not a swap detector, and a number measured over them says nothing
about swaps.

**Read the shortcut warning before fitting anything on this against FairFace.**
Two corpora from different sources separate on colour and compression alone —
measured at AUC 0.843 on DF40, and the separation ROSE to 0.904 when
resolution and format were matched (docs/HANDOFF.md §0). FairFace is Flickr
JPEG and SFHQ is 1024px generator output; there is every reason to expect the
same shortcut here, and no reason to assume it away. Measure it first.
"""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from dfd.faces import align, clamp_roi, detect_faces

from .face_pool import DEGENERATE_BOX, DUPLICATE, NO_FACE, UNREADABLE, DetectFn

__all__ = [
    "ALIGN", "DEGENERATE_BOX", "DUPLICATE", "FACE_DETECTOR", "NO_FACE",
    "SFHQ_COMPRESSION", "SFHQ_PT3_GENERATOR", "UNREADABLE",
    "load_sfhq_records",
]

logger = logging.getLogger(__name__)

#: Documented by the dataset, not inferred from filenames. See the module
#: docstring on why that distinction decides whether a LOGO axis is honest.
SFHQ_PT3_GENERATOR = "sfhq_pt3_stylegan2"

#: The archive ships JPEG at one quality nobody states. Recorded as unknown
#: for the same reason `corpora.df40` does: guard 3 wants c0/c23/c40, and an
#: invented compression label passes a guard by lying to it.
SFHQ_COMPRESSION = "unknown"

FACE_DETECTOR = "yunet_2023mar"
ALIGN = "landmark_similarity_224"

DEFAULT_CROP_SIZE = 224

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def _images(root: Path) -> list[Path]:
    """Every image under `root`, recursively, in a stable order.

    Recursive because the archive nests a curated `a small samples (750
    images)/` folder beside the full set, and a non-recursive scan of the
    archive root would silently load nothing.
    """
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)


def _sample(paths: Sequence[Path], k: int, seed: int) -> list[Path]:
    if k >= len(paths):
        return list(paths)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), size=k, replace=False)
    return [paths[int(i)] for i in sorted(idx)]


def load_sfhq_records(
    root: str | Path,
    *,
    limit: int | None = None,
    seed: int = 0,
    detect: DetectFn = detect_faces,
    size: int = DEFAULT_CROP_SIZE,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Load SFHQ images under `root` as label-1 benchmark records.

    Preprocessing matches every other corpus in this repo — YuNet detection
    then `dfd.faces.align` — so crops fitted on one are comparable to crops
    evaluated on another. Crops are deduplicated by content hash across the
    whole load, as `corpora.face_pool` does, because a duplicate image
    entering twice under two ids is the defect that cost this project its
    capture corpus (docs/HANDOFF.md §0).

    Each image is its own source and its own subject. Unlike `corpora.df40`,
    where that claim was false and narrowed every interval, here it is what
    the data is: independent draws from a generator, depicting nobody. If a
    future part of SFHQ ships an identity or seed grouping, this is the line
    that must change.

    Args:
        root: a directory of images, searched recursively.
        limit: how many records to return; fewer if images are skipped.
        seed: selects which images the limit takes.
        detect: face detector. Injected so tests need no weight file.
        size: aligned crop edge length.

    Returns:
        A (records, skipped) pair. `skipped` maps a reason constant to a
        count and is empty when nothing was dropped.

    Raises:
        FileNotFoundError: if `root` is not a directory, or holds no image.
            An empty single-label corpus produces metrics that are undefined
            rather than wrong, which is harder to notice.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory")

    paths = _images(root)
    if not paths:
        raise FileNotFoundError(f"{root} holds no image files")
    if limit is not None:
        paths = _sample(paths, max(limit, 0), seed)

    records: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    seen: set[str] = set()

    def drop(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for path in paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            logger.warning("unreadable image: %s", path)
            drop(UNREADABLE)
            continue
        frame: npt.NDArray[np.uint8] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        boxes = detect(frame)
        if not boxes:
            logger.debug("no face in %s", path)
            drop(NO_FACE)
            continue
        box = max(boxes, key=lambda b: b.score)
        if clamp_roi(frame.shape, box) is None:
            logger.debug("box misses the frame in %s", path)
            drop(DEGENERATE_BOX)
            continue

        aligned = align(frame, box, size=size)
        digest = hashlib.sha256(aligned.tobytes()).hexdigest()
        if digest in seen:
            logger.debug("duplicate crop from %s", path)
            drop(DUPLICATE)
            continue
        seen.add(digest)

        # Relative to `root`, so an id stays stable if the corpus moves, and
        # stays distinct across the archive's nested sample folder.
        sample_id = f"sfhq/{path.relative_to(root).as_posix()}"
        records.append({
            "sample_id": sample_id,
            "source_id": sample_id,
            "subject_id": sample_id,
            "generator": SFHQ_PT3_GENERATOR,
            "label": 1,
            "compression": SFHQ_COMPRESSION,
            "face_detector": FACE_DETECTOR,
            "align": ALIGN,
            "image": aligned,
        })

    logger.info("sfhq: %d records from %s, skipped %s",
                len(records), root, skipped or "nothing")
    return records, skipped
