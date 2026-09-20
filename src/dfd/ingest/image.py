from __future__ import annotations

import logging
from pathlib import Path

import cv2

from ..types import Context, Modality, Observation, Sample

logger = logging.getLogger(__name__)


def load_image(path: str | Path, context: Context) -> Sample:
    """Load image into a Sample.

    Decodes an image file into a single Observation. No face detection or
    quality measurement is performed; analysis is a separate stage.

    Args:
        path: Path to image file.
        context: Metadata context.

    Returns:
        Sample with a single Observation.

    Raises:
        ValueError: If image cannot be decoded.
    """
    path = Path(path)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not decode image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    sample_id = path.stem
    obs = Observation(t=0.0, payload=rgb, roi=None, quality=None, source_id=sample_id)
    logger.debug("extracted 1 observation from image %s", path)
    return Sample(sample_id=sample_id, modality=Modality.IMAGE,
                  observations=(obs,), context=context)
