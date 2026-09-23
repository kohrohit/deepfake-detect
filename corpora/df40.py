"""The ungated DF40 test split, as benchmark records.

`assets/manifest.yaml:df40_eval_subset` records what this data is: a
repackaging of the DF40 test split with a flat `fake/` vs `real/` layout and
NO per-technique label. DF40 itself spans 40 techniques; this copy does not
say which produced which image. Filenames fall into families that may track
technique, and this loader deliberately does not use them: a guessed grouping
used as a generator axis would produce a leave-one-generator-out number that
looks like generalisation and measures nothing. Every fake therefore carries
one generator id, `df40_unknown_mixture`, which makes `bench.protocol`
refuse a LOGO split outright rather than emit a flattering one.

Preprocessing matches training: YuNet detection then `dfd.faces.align`, the
same path `corpora.face_pool` takes, so a seam model fitted on aligned crops
is evaluated on aligned crops. What does NOT match is identity: DF40's
frames come from upstream forensics corpora and carry no subject label, so
each image is treated as its own subject. Where several images are in fact
the same person, that assumption is wrong and the bootstrap interval is
narrower than the truth. It is recorded here because it cannot be fixed from
this data.
"""
from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from dfd.faces import align, clamp_roi, detect_faces

from .face_pool import DEGENERATE_BOX, DUPLICATE, NO_FACE, UNREADABLE, DetectFn

__all__ = [
    "ALIGN", "DEGENERATE_BOX", "DF40_GENERATOR", "DUPLICATE", "FACE_DETECTOR",
    "NO_FACE", "UNGROUPED", "UNKNOWN_COMPRESSION", "UNREADABLE",
    "load_df40_records", "source_group",
]

logger = logging.getLogger(__name__)

#: The single generator id every fake carries. One value, so a LOGO split is
#: impossible by construction — see the module docstring.
DF40_GENERATOR = "df40_unknown_mixture"

#: Guard 3 (`bench.guards.check_compression_coverage`) wants c0/c23/c40. This
#: corpus arrives at one unlabelled compression level, so it is recorded as
#: unknown and the guard fails — correctly. A benchmark run over this data is
#: a guard-waived run, and must be reported as one.
UNKNOWN_COMPRESSION = "unknown"

FACE_DETECTOR = "yunet_2023mar"
ALIGN = "landmark_similarity_224"

#: Directory name -> label, exactly as the archive lays it out.
LABEL_DIRS: dict[str, int] = {"fake": 1, "real": 0}

DEFAULT_CROP_SIZE = 224

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


#: The source token given to a filename that carries no frame index. Every
#: such file in one label directory shares it, so they resample as ONE
#: observation. That understates the evidence deliberately — see
#: `source_group`.
UNGROUPED = "unnumbered"

#: A trailing frame index: `_100`, `-12`, ` (3)`, `(3)`. The separator (or the
#: parentheses) is required. Without one, `00042.png` would be read as frame
#: 42 of an empty-named video, which is how a filename guess starts inventing
#: structure that is not there.
_FRAME_INDEX = re.compile(r"(?:[ _-]\(?\d+\)?|\(\d+\))$")


def source_group(label_dir: str, filename: str) -> str:
    """The source id shared by every frame of one filename family.

    This is the ONE use this module makes of filename structure, and it is
    not the use the module docstring refuses. A guessed *generator* label
    fabricates a LOGO axis and makes a benchmark look like it generalises.
    A guessed *source group* only decides what `bench.runner` resamples over,
    and the guess here is built to fail in the safe direction: names it
    cannot parse merge into one group rather than splitting into many, so
    the error it can make is to claim LESS independence than the data has,
    never more.

    Measured on the ungated DF40 test split, 2026-09-23: 1,601 fakes fall
    into 45 families, the largest holding 999 of them (62%), for a Kish
    effective sample size of **2.4**. Resampling those rows as if they were
    independent — which this loader did until now — reports a confidence
    interval roughly 26x narrower than the data supports.

    Args:
        label_dir: `"fake"` or `"real"`. Part of the id, so a family name
            appearing under both never merges across the label boundary: a
            source that straddles generators is refused by `bench.protocol`,
            correctly.
        filename: the image's file name, extension included.

    Returns:
        `"{label_dir}/{family}"`, or `"{label_dir}/unnumbered"` when the name
        carries no frame index to strip.
    """
    stem = Path(filename).stem
    trimmed = _FRAME_INDEX.sub("", stem)
    if trimmed == stem or not trimmed:
        return f"{label_dir}/{UNGROUPED}"
    return f"{label_dir}/{trimmed}"


def _images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir()
                  if p.suffix.lower() in _IMAGE_SUFFIXES)


def _sample(paths: Sequence[Path], k: int, seed: int) -> list[Path]:
    if k >= len(paths):
        return list(paths)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), size=k, replace=False)
    return [paths[int(i)] for i in sorted(idx)]


def load_df40_records(
    root: str | Path,
    *,
    limit: int | None = None,
    seed: int = 0,
    detect: DetectFn = detect_faces,
    size: int = DEFAULT_CROP_SIZE,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Load the DF40 eval subset at `root` into `bench.runner` records.

    Args:
        root: a directory holding both `fake/` and `real/`.
        limit: total records wanted, split evenly between the two labels.
            Fewer are returned when a label has fewer images, or when crops
            are skipped.
        seed: selects which images the limit takes.
        detect: face detector. Injected so tests need no weight file.
        size: aligned crop edge length.

    Returns:
        A (records, skipped) pair. `skipped` maps a reason constant to a
        count and is empty when nothing was dropped.

    Raises:
        FileNotFoundError: if either label directory is absent. Loading only
            the half that exists would produce a single-label corpus, and
            every metric over one label is undefined rather than wrong —
            which is harder to notice.
    """
    root = Path(root)
    for name in LABEL_DIRS:
        if not (root / name).is_dir():
            raise FileNotFoundError(
                f"{root} has no {name!r} directory; the DF40 eval subset is "
                "laid out as fake/ and real/")

    per_label = None if limit is None else max(limit // len(LABEL_DIRS), 0)

    records: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    seen: set[str] = set()

    def drop(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for name, label in LABEL_DIRS.items():
        paths = _images(root / name)
        if per_label is not None:
            paths = _sample(paths, per_label, seed)
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

            # The FULL filename, not the stem: DF40's test/fake holds
            # both `0.png` (1024px) and `0.jpg` (256px), and they are
            # different images from different filename families. A
            # stem id collides on 20 such pairs, and `bench.runner`
            # refuses the whole run over it — correctly.
            sample_id = f"{name}/{path.name}"
            # Frames of one filename family are ONE source and ONE subject.
            # Both were `sample_id` until 2026-09-23, which told the
            # bootstrap that 999 frames of a single video were 999
            # independent observations — see `source_group` for the measured
            # cost. subject_id tracks source_id because `bench.protocol`
            # refuses a source that straddles subjects, and because frames
            # of one video are in fact one person.
            group = source_group(name, path.name)
            records.append({
                "sample_id": sample_id,
                "source_id": group,
                "subject_id": group,
                "generator": DF40_GENERATOR if label == 1 else None,
                "label": label,
                "compression": UNKNOWN_COMPRESSION,
                "face_detector": FACE_DETECTOR,
                "align": ALIGN,
                "image": aligned,
            })

    logger.info("df40: %d records from %s, skipped %s",
                len(records), root, skipped or "nothing")
    return records, skipped
