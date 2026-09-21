"""The composition root: one file in, one audit record out (spec §5.1, §7.2).

Every other module in this package is a stage. This is the only place they are
wired together, and the only non-test caller of ingest, faces, quality,
calibration, fusion and audit.

What this path does TODAY is abstain, three times over: the YuNet weights are
absent, the detector weights are absent, and nothing has fitted a calibration
curve. Each cause is recorded separately — `stage_reasons` for the face stage,
the per-detector `reason` for the rest — because "insufficient evidence"
without a cause is not an audit trail.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .faces import DEFAULT_MODEL, FaceBox, detect_faces
from .quality import measure_quality
from .types import QUALITY_BANDS, Observation, Sample

logger = logging.getLogger(__name__)

#: The face stage's seam, kept a plain two-argument callable: `detect_faces` is
#: overloaded on a Literal keyword, and a Protocol reproducing those overloads
#: is brittle under mypy --strict for no gain. `_detect_with_reason` adapts it.
FaceDetectFn = Callable[[npt.NDArray[np.uint8], "str | Path"],
                        "tuple[list[FaceBox], str]"]

#: No observation carried a measured quality, so no band can be named. Not a
#: member of QUALITY_BANDS: it is the absence of a band, not a bad one.
UNMEASURED = "unmeasured"
#: The detector ran and found nothing. Distinct from `weights_absent`, which
#: means it never ran at all.
NO_FACE = "no_face"
#: A box that does not intersect the frame. Reported rather than measured,
#: because a clamped empty crop would make OpenCV raise.
DEGENERATE_BOX = "degenerate_box"
NO_OBSERVATIONS = "no_observations"
MIXED = "mixed"
OK = "ok"


def _detect_with_reason(frame: npt.NDArray[np.uint8],
                        model_path: str | Path) -> tuple[list[FaceBox], str]:
    """Adapt `detect_faces` to the `FaceDetectFn` seam."""
    return detect_faces(frame, model_path, with_reason=True)


def _clamp_roi(shape: tuple[int, ...], box: FaceBox) -> tuple[int, int, int, int] | None:
    """Clamp a detection to the frame, or None if it does not intersect it.

    `measure_quality` slices `frame[y:y + h, x:x + w]` with no clamping, so a
    negative origin would silently slice from the far end of the array and an
    empty crop makes `cv2.cvtColor` raise. `faces.align` clamps for the same
    reason; this is that rule applied one stage earlier.
    """
    height, width = shape[:2]
    x0, y0 = max(0, box.x), max(0, box.y)
    x1, y1 = min(width, box.x + box.w), min(height, box.y + box.h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def _worst_band(observations: Sequence[Observation]) -> str:
    """The worst measured band across observations, or UNMEASURED.

    Worst by position in `types.QUALITY_BANDS` (`reject < low < medium < high`).
    Calibration conditions on the regime that held for the whole sample: the
    mean of `high` and `reject` is a band the sample never occupied, and the
    best band would calibrate a blurry sample as though it were sharp.
    """
    bands = [o.quality.band for o in observations if o.quality is not None]
    if not bands:
        return UNMEASURED
    return min(bands, key=QUALITY_BANDS.index)


def _aggregate(reasons: list[str]) -> str:
    """One sample-level reason from per-frame reasons."""
    if not reasons:
        return NO_OBSERVATIONS
    if OK in reasons:
        return OK
    unique = set(reasons)
    return reasons[0] if len(unique) == 1 else MIXED


def normalize(sample: Sample, *, detect: FaceDetectFn = _detect_with_reason,
              face_model: str | Path = DEFAULT_MODEL) -> tuple[Sample, dict[str, str]]:
    """Attach a face ROI and a quality measurement to every observation.

    Ingest deliberately emits `roi=None, quality=None` — decoding and analysis
    are separate stages. This is the analysis stage, and until it existed the
    only code performing it was the benchmark runner, which fabricated
    landmarks at 35% and 65% of frame width rather than detecting them.

    Args:
        sample: an ingested sample whose observations carry no quality.
        detect: face detection seam; must return (boxes, reason).
        face_model: path passed to `detect`.

    Returns:
        A pair of (sample with normalized observations, stage reasons). The
        reasons carry `faces` (`ok`, `no_face`, `weights_absent`,
        `degenerate_box`, `mixed` or `no_observations`), `frames_with_face`
        as "n/total", and `max_faces_in_frame`.

    Raises:
        No exceptions of its own; `detect` may raise (a corrupt model file
        makes OpenCV raise, which is deliberately not caught — treating
        corruption as absence would hide a deployment failure).
    """
    out: list[Observation] = []
    reasons: list[str] = []
    counts: list[int] = []

    for obs in sample.observations:
        boxes, reason = detect(obs.payload, face_model)
        counts.append(len(boxes))
        if not boxes:
            reasons.append(NO_FACE if reason == OK else reason)
            out.append(obs)
            continue
        box = max(boxes, key=lambda b: b.w * b.h)
        roi = _clamp_roi(obs.payload.shape, box)
        if roi is None:
            logger.warning("face box %s does not intersect the frame; not measured",
                           (box.x, box.y, box.w, box.h))
            reasons.append(DEGENERATE_BOX)
            out.append(obs)
            continue
        quality = measure_quality(obs.payload, roi, box.landmarks[:2])
        out.append(Observation(t=obs.t, payload=obs.payload, roi=roi,
                               quality=quality, source_id=obs.source_id))
        reasons.append(OK)

    with_face = sum(1 for r in reasons if r == OK)
    stage_reasons = {
        "faces": _aggregate(reasons),
        "frames_with_face": f"{with_face}/{len(sample.observations)}",
        "max_faces_in_frame": str(max(counts) if counts else 0),
    }
    logger.debug("normalized %d observations: %s", len(out), stage_reasons)
    return (Sample(sample_id=sample.sample_id, modality=sample.modality,
                   observations=tuple(out), context=sample.context),
            stage_reasons)
