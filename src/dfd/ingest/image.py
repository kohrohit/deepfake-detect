from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..limits import DEFAULT_LIMITS, Limits, check_image_before_decode
from ..types import Context, Modality, Observation, Sample

logger = logging.getLogger(__name__)


def load_image(path: str | Path, context: Context,
                limits: Limits = DEFAULT_LIMITS) -> Sample:
    """Load image into a Sample.

    Decodes an image file into a single Observation. No face detection or
    quality measurement is performed; analysis is a separate stage.

    Header dimensions are checked against `limits` before any decode (spec
    §3A, §10): a crafted image can allocate gigabytes on decode while
    occupying kilobytes on disk, so the file size and pixel count are gated
    from the header first. See `dfd.limits` for why.

    Args:
        path: Path to image file.
        context: Metadata context.
        limits: Resource limits to enforce before decoding.

    Returns:
        Sample with a single Observation.

    Raises:
        InvalidInput: If the path is unreadable or the header can't be read.
        ResourceLimitExceeded: If the file or its dimensions exceed `limits`.
        ValueError: If image cannot be decoded.
    """
    path = Path(path)
    check_image_before_decode(path, limits)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not decode image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)
    sample_id = path.stem
    obs = Observation(t=0.0, payload=rgb, roi=None, quality=None, source_id=sample_id)
    logger.debug("extracted 1 observation from image %s", path)
    return Sample(sample_id=sample_id, modality=Modality.IMAGE,
                  observations=(obs,), context=context)
