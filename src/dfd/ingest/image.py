from __future__ import annotations

from pathlib import Path

import cv2

from ..types import Context, Modality, Observation, Sample


def load_image(path: str | Path, context: Context) -> Sample:
    path = Path(path)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not decode image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    sample_id = path.stem
    obs = Observation(t=0.0, payload=rgb, roi=None, quality=None, source_id=sample_id)
    return Sample(sample_id=sample_id, modality=Modality.IMAGE,
                  observations=(obs,), context=context)
