"""Identity embeddings for a loaded corpus — the input acceptance criterion 2 needs.

`bench.runner` takes embeddings and refuses to compute them: embedding means
running a face detector and a recogniser over every record, which is corpus
work, and the orchestrator holding that code would make every benchmark run
depend on a weight file it does not otherwise need.

THE CROP IS RE-DETECTED, AND THAT IS NOT FREE. A loader hands the runner an
ALIGNED crop and does not keep the `FaceBox` it aligned with, so the landmarks
SFace needs are gone by the time anything wants an embedding. Re-detecting on
the crop recovers them in the crop's own coordinates, which is what
`Embedder.embed` wants — but a face that aligns cleanly can still fail to
re-detect once cropped (it fills the frame, and YuNet is trained on faces
within a scene). Those records are reported as UNEMBEDDED rather than skipped
quietly, because `check_identity_disjoint` refuses to certify a split
containing an id it has no embedding for, and a caller needs to know how much
of its corpus that is before deciding what the certificate is worth.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from dfd.embed import Embedder, load_embedder
from dfd.faces import clamp_roi, detect_faces

from .face_pool import DetectFn

logger = logging.getLogger(__name__)


def embeddings_for_records(
    records: Sequence[dict[str, Any]],
    *,
    embedder: Embedder | None = None,
    detect: DetectFn = detect_faces,
) -> tuple[dict[str, npt.NDArray[np.float32]], dict[str, int]]:
    """Embed every record's face crop, keyed by `sample_id`.

    Args:
        records: as produced by a corpus loader; each needs `sample_id` and
            an `image` (RGB HWC uint8 aligned crop).
        embedder: a loaded `dfd.embed.Embedder`, or None to load the default.
            When the weight file is absent this returns no embeddings and a
            reason, rather than raising: a deployment without the recogniser
            must still be able to run the benchmark, with criterion 2
            reported as unmeasured.
        detect: face detector, injected so tests need no weight file.

    Returns:
        An (embeddings, skipped) pair. `skipped` counts `no_face` (the crop
        would not re-detect), `degenerate_box` and `no_embedding` (the
        recogniser refused the aligned crop), plus `weights_absent` set to
        the full record count when there is no embedder at all.
    """
    skipped: dict[str, int] = {}

    def drop(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    if embedder is None:
        embedder, reason = load_embedder()
        if embedder is None:
            logger.warning("no identity embedder (%s); criterion 2 cannot be "
                           "measured on this run", reason)
            return {}, {reason: len(records)}

    out: dict[str, npt.NDArray[np.float32]] = {}
    for record in records:
        frame = record["image"]
        boxes = detect(frame)
        if not boxes:
            drop("no_face")
            continue
        box = max(boxes, key=lambda b: b.score)
        if clamp_roi(frame.shape, box) is None:
            drop("degenerate_box")
            continue
        vector = embedder.embed(frame, box)
        if vector is None:
            drop("no_embedding")
            continue
        out[record["sample_id"]] = vector

    logger.info("identity: embedded %d of %d records, skipped %s",
                len(out), len(records), skipped or "nothing")
    return out, skipped
