from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..errors import ResourceLimitExceeded
from ..limits import DEFAULT_LIMITS, Limits, check_file_size, check_frame_dims
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
               seed: int = 0, limits: Limits = DEFAULT_LIMITS) -> Sample:
    """Load video frames into a Sample.

    Decodes a video file and extracts a deterministic subset of frames using
    stratified sampling. If frame-count metadata is unavailable or unreliable,
    falls back to sequential read of the first max_frames.

    Every gate here reads container metadata before a frame is decoded (spec
    §3A, §10) — the same header-before-decode discipline `check_image_before_
    decode` uses for images, applied to what `VideoCapture` exposes at open
    time:

    - `check_file_size` runs before the container is opened at all.
    - The container's own declared frame width/height (`CAP_PROP_FRAME_
      WIDTH`/`HEIGHT`, populated at open, before any `cap.read()`) is
      checked against `limits.max_pixels` before the read loop starts. A
      container declaring 0x0 is refused rather than read unguarded.
    - The container's declared duration (`CAP_PROP_FRAME_COUNT / fps`, also
      read before the loop) is checked against `limits.max_duration_s`
      before the loop starts, so a well-compressed multi-hour file that
      passes the byte-size cap is still refused before a single frame
      decodes.
    - The requested `max_frames` is clamped to `limits.max_frames`, and the
      decode loop itself stops as soon as every needed frame has been
      retained — it does not keep decoding to the end of the stream after
      the quota is met. The clamp bounds decode work, not merely how many
      observations are kept.

    When frame-count metadata is unusable (`total <= 0`), duration cannot be
    computed from it, so only the dimension gate and the `max_frames` cutoff
    apply in that fallback path.

    Args:
        path: Path to video file.
        context: Metadata context.
        max_frames: Maximum number of frames to extract.
        seed: Random seed for deterministic frame selection (when total > 0).
        limits: Resource limits to enforce before and during decoding.

    Returns:
        Sample with Observation per extracted frame, ordered by timestamp.

    Raises:
        InvalidInput: If the path is unreadable, or the container declares
            non-positive frame dimensions.
        ResourceLimitExceeded: If the file exceeds `limits.max_file_bytes`,
            the declared frame dimensions exceed `limits.max_pixels`, or the
            declared duration exceeds `limits.max_duration_s`.
        ValueError: If video cannot be opened or decoding yields zero frames.
    """
    path = Path(path)
    check_file_size(path, limits)
    max_frames = min(max_frames, limits.max_frames)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        check_frame_dims(width, height, limits)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS

        # Determine which frames to extract.
        if total <= 0:
            logger.warning("frame-count metadata unusable for %s; falling back to sequential read", path)
            wanted = None  # Will read sequentially and keep first max_frames
        else:
            duration_s = total / fps
            if duration_s > limits.max_duration_s:
                raise ResourceLimitExceeded(
                    f"video {path.name} declares {duration_s:.1f}s, "
                    f"exceeds limit {limits.max_duration_s}s")
            wanted = set(sample_indices(total, max_frames, seed))

        sample_id = path.stem
        obs: list[Observation] = []
        i = 0
        needed = max_frames if wanted is None else len(wanted)
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
            if len(obs) >= needed:
                break
    finally:
        cap.release()

    if not obs:
        raise ValueError(f"could not decode any frames from video: {path}")

    logger.debug("extracted %d observations from video %s", len(obs), path)
    return Sample(sample_id=sample_id, modality=Modality.VIDEO,
                  observations=tuple(obs), context=context)
