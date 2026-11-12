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

import hashlib
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from dfd.faces import FaceBox, align, clamp_roi, detect_faces
from dfd.quality import measure_quality
from dfd.types import Quality

from .captures import CaptureSession

logger = logging.getLogger(__name__)

#: A face detector: frame in, boxes out. `dfd.faces.detect_faces` satisfies it.
DetectFn = Callable[[npt.NDArray[np.uint8]], list[FaceBox]]

NO_FRAMES = "no_frames"
NO_FACE = "no_face"
UNREADABLE = "unreadable"
#: A crop byte-identical to one already in the pool. Counted, never silent:
#: on the real corpus this is the single largest skip reason by an order of
#: magnitude (see `build_face_pool`), and a reader who cannot see it would
#: read a pool of 58 distinct images as a pool of 1088.
DUPLICATE = "duplicate"
#: A detection that does not overlap the frame by at least 2x2 px after
#: clamping. Mirrors `dfd.pipeline.DEGENERATE_BOX`, deliberately: the same
#: condition should not have two names across the codebase.
DEGENERATE_BOX = "degenerate_box"

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
    *,
    size: int = DEFAULT_CROP_SIZE,
    detect: DetectFn = detect_faces,
    max_frames_per_session: int = DEFAULT_MAX_FRAMES,
) -> tuple[list[FaceCrop], dict[str, int]]:
    """Extract aligned face crops from capture sessions.

    Args:
        sessions: sessions to draw from, as loaded by `load_capture_sessions`.
        size: edge length of the aligned crop.
        detect: face detector. Injected so tests need no weight file.
        max_frames_per_session: cap on frames taken from any one session.

    Returns:
        A (crops, skipped) pair. `skipped` maps a reason constant to a count,
        and is empty when nothing was dropped. Never raises for bad input:
        a frame that cannot be decoded is counted, not propagated, because
        one corrupt JPEG must not cost the other 441 sessions.

    Crops are DEDUPLICATED by content across the whole pool, and every drop
    is counted under `DUPLICATE`. This is not an optimisation. Measured on
    the real capture corpus, 2026-09-22: its 1088 frame files hold only 58
    distinct images, and 979 of those files — spread over 368 of the 442
    sessions — are byte-identical to `assets/attack/victim_id.jpg`, a demo
    asset replayed as the captured frame. Without this, one image would
    enter the pool hundreds of times under hundreds of session ids, and
    `training.fit_blend.split_by_subject` — which splits on session id —
    would place that same image on both sides of the holdout. The held-out
    AUC would then measure memorisation of a single picture and report it
    as generalisation. Deduplication is what makes that split mean anything.

    The hash is taken over the ALIGNED CROP, not the source frame, because
    the crop is what reaches training: two frames that differ only outside
    the face box align to the same pixels and are the same observation.

    There is deliberately no `root` parameter. `CaptureSession.folder`, as
    produced by `load_capture_sessions`, is already a complete path (it is
    built there as `str(path.parent)` from a glob rooted at the caller's
    `root`) — not a bare session id relative to some root the caller must
    supply again. An earlier version of this function took a `root` and
    joined it against `session.folder`: with an absolute root that
    "worked" only because `Path.joinpath` discards its left operand
    whenever the right operand is itself absolute, and with a relative
    root it silently double-prefixed the path (`root/root/session_id`),
    so every session reported `NO_FRAMES` even though the frames were on
    disk. If you find yourself wanting to add `root` back, don't: pass a
    `session.folder` that is already correct instead.
    """
    crops: list[FaceCrop] = []
    skipped: dict[str, int] = {}
    seen: set[str] = set()

    def drop(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for session in sessions:
        folder = Path(session.folder)
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
            roi = clamp_roi(frame.shape, box)
            if roi is None:
                logger.debug("box misses the frame in %s", frame_path)
                drop(DEGENERATE_BOX)
                continue
            quality = measure_quality(frame, roi, box.landmarks)
            index = int(frame_path.stem.split("_")[-1])
            aligned = align(frame, box, size=size)
            digest = hashlib.sha256(aligned.tobytes()).hexdigest()
            if digest in seen:
                logger.debug("duplicate crop from %s", frame_path)
                drop(DUPLICATE)
                continue
            seen.add(digest)
            crops.append(FaceCrop(
                session_id=session.session_id,
                frame_index=index,
                image=aligned,
                box=box,
                quality=quality,
                swapped=session.swapped,
            ))

    logger.info("face pool: %d crops from %d sessions, skipped %s",
                len(crops), len(sessions), skipped or "nothing")
    return crops, skipped
