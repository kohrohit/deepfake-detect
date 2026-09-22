"""Face detection and alignment via OpenCV YuNet (MIT, see assets/manifest.yaml).

Fails soft when the weight file is absent: yields zero faces with a stated reason,
never an exception and never a fabricated detection. This keeps CI hermetic and
keeps the whole pipeline honest about what it could not measure.

A corrupted (present but unparseable) model file will raise cv2.error, which is
not caught — silently treating corruption as absence would hide deployment failures.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, overload

import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

WEIGHTS_ABSENT = "weights_absent"
OK = "ok"

# YuNet's conventional default confidence threshold. Unvalidated against our data.
DEFAULT_SCORE_THRESHOLD = 0.7

DEFAULT_MODEL = Path("assets/models/face_detection_yunet_2023mar.onnx")


@dataclass(frozen=True)
class FaceBox:
    """Detection result: position, landmarks, and confidence.

    A FaceBox is evidence and immutable. The landmarks array is made read-only
    at construction, so the array object is frozen even if passed from outside.
    Callers needing a mutable copy must call .copy() on the landmarks array.
    """
    x: int
    y: int
    w: int
    h: int
    # (5, 2) landmark points. YuNet's documented order is believed to be
    # right eye, left eye, nose, right mouth corner, left mouth corner — but this
    # has NOT been verified against real model output in this repo (no weights on
    # disk). Only inter-ocular DISTANCE is consumed today, which is symmetric and
    # therefore insensitive to the order. Verify before any consumer needs eye
    # identity (roll correction, gaze).
    landmarks: npt.NDArray[np.float64]
    score: float

    def __post_init__(self) -> None:
        """Freeze the landmark array: a FaceBox is evidence, not a scratch buffer."""
        self.landmarks.setflags(write=False)


@overload
def detect_faces(
    frame: npt.NDArray[np.uint8],
    model_path: str | Path = DEFAULT_MODEL,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    with_reason: Literal[False] = False,
) -> list[FaceBox]: ...


@overload
def detect_faces(
    frame: npt.NDArray[np.uint8],
    model_path: str | Path = DEFAULT_MODEL,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    with_reason: Literal[True] = ...,
) -> tuple[list[FaceBox], str]: ...


def detect_faces(
    frame: npt.NDArray[np.uint8],
    model_path: str | Path = DEFAULT_MODEL,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    with_reason: bool = False,
) -> list[FaceBox] | tuple[list[FaceBox], str]:
    """Detect faces. Returns [] (and a reason) when the model file is absent.

    May raise cv2.error if the model file is present but corrupt.
    """
    path = Path(model_path)
    if not path.exists():
        logger.warning("Face detector weights absent at %s", path)
        return ([], WEIGHTS_ABSENT) if with_reason else []

    h, w = frame.shape[:2]
    det = cv2.FaceDetectorYN.create(str(path), "", (w, h),
                                    score_threshold=score_threshold)
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    _, faces = det.detect(bgr)
    out: list[FaceBox] = []
    if faces is not None:
        # opencv>=5 ships stubs typing detect() as MatLike, and mypy cannot know an
        # ndarray's ndim, so `for f in faces` infers f as a scalar. Normalise to a
        # typed 2-D array and index rows: same iteration over axis 0, but checkable.
        rows: npt.NDArray[np.float64] = np.asarray(faces, dtype=np.float64)
        for i in range(rows.shape[0]):
            row: npt.NDArray[np.float64] = rows[i]
            x, y, bw, bh = (int(v) for v in row[:4])
            lms = np.array(row[4:14], dtype=np.float64).reshape(5, 2)
            out.append(FaceBox(x=x, y=y, w=bw, h=bh, landmarks=lms, score=float(row[14])))
    logger.debug("detected %d faces", len(out))
    return (out, OK) if with_reason else out


def clamp_roi(shape: tuple[int, ...],
              box: FaceBox) -> tuple[int, int, int, int] | None:
    """Clamp a detection to the frame, or None if it does not intersect it.

    A detector may return a box that hangs off the frame edge, and on a
    tightly-cropped face dataset it usually does: measured 2026-09-22, YuNet
    put the box outside the image on 146 of the first 300 FairFace crops.
    `measure_quality` slices `frame[y:y + h, x:x + w]` with no clamping, so a
    negative origin slices from the FAR END of the array and yields an empty
    crop, which makes `cv2.cvtColor` raise.

    This lives here, beside `FaceBox`, because three call sites need the same
    rule and two of them had each solved it privately: `align` clamps inline,
    and `dfd.pipeline` had its own `_clamp_roi`. `corpora.face_pool` was the
    third and had no clamp at all, which is why the pool builder crashed on
    the second image of any public face dataset. One rule, one place.

    Args:
        shape: the frame's shape; only the first two entries are read.
        box: the detection to clamp.

    Returns:
        An (x, y, w, h) ROI inside the frame, or None when the intersection is
        smaller than 2x2 px — too small for any quality measurement to mean
        anything, and the size at which OpenCV starts raising instead.
    """
    height, width = shape[:2]
    x0, y0 = max(0, box.x), max(0, box.y)
    x1, y1 = min(width, box.x + box.w), min(height, box.y + box.h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def align(frame: npt.NDArray[np.uint8], box: FaceBox, size: int = 224) -> npt.NDArray[np.uint8]:
    """Crop the face box, clamped to frame bounds, resized to (size, size)."""
    h, w = frame.shape[:2]
    x0 = max(0, box.x)
    y0 = max(0, box.y)
    x1 = min(w, box.x + box.w)
    y1 = min(h, box.y + box.h)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    crop = frame[y0:y1, x0:x1]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA).astype(np.uint8)
