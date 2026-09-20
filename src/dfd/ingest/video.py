from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..types import Context, Modality, Observation, Sample


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


def load_video(path: str | Path, context: Context, max_frames: int = 32,
               seed: int = 0) -> Sample:
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    wanted = set(sample_indices(total, max_frames, seed))

    sample_id = path.stem
    obs: list[Observation] = []
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i in wanted:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            obs.append(Observation(t=i / fps, payload=rgb, roi=None,
                                   quality=None, source_id=sample_id))
        i += 1
    cap.release()
    return Sample(sample_id=sample_id, modality=Modality.VIDEO,
                  observations=tuple(obs), context=context)
