"""Aligned face crops, with provenance, from the capture corpus.

The pool is the only real-face source this project owns outright, so every
crop carries the session it came from and whether that session was swapped.
Both travel with the crop because the split discipline downstream depends on
them: swapped sessions are evaluation-only and must never be blended into
training data.

Face detection is injected rather than imported-and-called so that tests need
no weight file. The YuNet weights are gitignored and absent in CI.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from dfd.faces import FaceBox, align, detect_faces
from dfd.quality import measure_quality
from dfd.types import Quality

from .captures import CaptureSession

logger = logging.getLogger(__name__)

#: A face detector: frame in, boxes out. `dfd.faces.detect_faces` satisfies it.
DetectFn = Callable[[npt.NDArray[np.uint8]], list[FaceBox]]

NO_FRAMES = "no_frames"
NO_FACE = "no_face"
UNREADABLE = "unreadable"

#: Aligned crop edge length, in pixels. 224 matches `dfd.faces.align`'s default
#: and the resolution the seam features in `dfd.detectors.blend` assume.
DEFAULT_CROP_SIZE = 224

#: Frames taken per session. The corpus holds 1-5 frames per session and 163
#: sessions have exactly 3; taking 2 keeps sessions with few frames from being
#: under-represented relative to sessions with many, which would otherwise
#: weight the pool toward whichever sessions happened to record longest.
DEFAULT_MAX_FRAMES = 2


@dataclass(frozen=True)
class FaceCrop:
    """One aligned face, and everything needed to place it in a split."""
    session_id: str
    frame_index: int
    image: npt.NDArray[np.uint8]
    box: FaceBox
    quality: Quality
    swapped: bool


def build_face_pool(
    sessions: Sequence[CaptureSession],
    root: str | Path,
    *,
    size: int = DEFAULT_CROP_SIZE,
    detect: DetectFn = detect_faces,
    max_frames_per_session: int = DEFAULT_MAX_FRAMES,
) -> tuple[list[FaceCrop], dict[str, int]]:
    """Extract aligned face crops from capture sessions.

    Args:
        sessions: sessions to draw from, as loaded by `load_capture_sessions`.
        root: directory holding one folder per session, each with `frame_NN.jpg`.
        size: edge length of the aligned crop.
        detect: face detector. Injected so tests need no weight file.
        max_frames_per_session: cap on frames taken from any one session.

    Returns:
        A (crops, skipped) pair. `skipped` maps a reason constant to a count,
        and is empty when nothing was dropped. Never raises for bad input:
        a frame that cannot be decoded is counted, not propagated, because
        one corrupt JPEG must not cost the other 441 sessions.
    """
    crops: list[FaceCrop] = []
    skipped: dict[str, int] = {}

    def drop(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for session in sessions:
        folder = Path(root) / session.folder
        frames = sorted(folder.glob("frame_*.jpg"))[:max_frames_per_session]
        if not frames:
            logger.debug("session %s has no frames", session.session_id)
            drop(NO_FRAMES)
            continue

        for frame_path in frames:
            bgr = cv2.imread(str(frame_path))
            if bgr is None:
                logger.warning("unreadable frame: %s", frame_path)
                drop(UNREADABLE)
                continue
            frame: npt.NDArray[np.uint8] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            boxes = detect(frame)
            if not boxes:
                logger.debug("no face in %s", frame_path)
                drop(NO_FACE)
                continue

            box = max(boxes, key=lambda b: b.score)
            quality = measure_quality(
                frame, (box.x, box.y, box.w, box.h), box.landmarks)
            index = int(frame_path.stem.split("_")[-1])
            crops.append(FaceCrop(
                session_id=session.session_id,
                frame_index=index,
                image=align(frame, box, size=size),
                box=box,
                quality=quality,
                swapped=session.swapped,
            ))

    logger.info("face pool: %d crops from %d sessions, skipped %s",
                len(crops), len(sessions), skipped or "nothing")
    return crops, skipped
