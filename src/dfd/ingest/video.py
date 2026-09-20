from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..types import Context, Modality, Observation, Sample

logger = logging.getLogger(__name__)

# Fallback when the container reports no frame rate. 25.0 is the PAL/broadcast
# convention and is used only to derive observation timestamps, never for decoding.
DEFAULT_FPS = 25.0

# Default maximum number of frames to extract from a video.
DEFAULT_MAX_FRAMES = 32


def sample_indices(total: int, k: int, seed: int) -> list[int]:
    """Deterministic frame indices. Returns all frames when total <= k."""
    if total <= k:
        return list(range(total))
    rng = np.random.default_rng(seed)
    # Stratified: one frame per equal-width bin, jittered inside the bin.
    edges = np.linspace(0, total, k + 1).astype(int)
    idx = [int(rng.integers(edges[i], max(edges[i] + 1, edges[i + 1])))
           for i in range(k)]
    return sorted(set(min(i, total - 1) for i in idx))


def load_video(path: str | Path, context: Context, max_frames: int = DEFAULT_MAX_FRAMES,
               seed: int = 0) -> Sample:
    """Load video frames into a Sample.

    Decodes a video file and extracts a deterministic subset of frames using
    stratified sampling. If frame-count metadata is unavailable or unreliable,
    falls back to sequential read of the first max_frames.

    Args:
        path: Path to video file.
        context: Metadata context.
        max_frames: Maximum number of frames to extract.
        seed: Random seed for deterministic frame selection (when total > 0).

    Returns:
        Sample with Observation per extracted frame, ordered by timestamp.

    Raises:
        ValueError: If video cannot be opened or decoding yields zero frames.
    """
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS

    # Determine which frames to extract.
    if total <= 0:
        logger.warning("frame-count metadata unusable for %s; falling back to sequential read", path)
        wanted = None  # Will read sequentially and keep first max_frames
    else:
        wanted = set(sample_indices(total, max_frames, seed))

    sample_id = path.stem
    obs: list[Observation] = []
    i = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            # If metadata was unusable, keep first max_frames; else keep wanted indices.
            should_keep = (wanted is None and i < max_frames) or (wanted is not None and i in wanted)
            if should_keep:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                obs.append(Observation(t=i / fps, payload=rgb, roi=None,
                                       quality=None, source_id=sample_id))
            i += 1
    finally:
        cap.release()

    if not obs:
        raise ValueError(f"could not decode any frames from video: {path}")

    logger.debug("extracted %d observations from video %s", len(obs), path)
    return Sample(sample_id=sample_id, modality=Modality.VIDEO,
                  observations=tuple(obs), context=context)
