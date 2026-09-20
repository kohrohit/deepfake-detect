"""Face detection and alignment via OpenCV YuNet (MIT, see assets/manifest.yaml).

Fails soft: a missing weight file yields zero faces with a stated reason, never
an exception and never a fabricated detection. This keeps CI hermetic and keeps
the whole pipeline honest about what it could not measure.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

WEIGHTS_ABSENT = "weights_absent"
OK = "ok"

DEFAULT_MODEL = Path("assets/models/face_detection_yunet_2023mar.onnx")


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    landmarks: np.ndarray  # (5, 2): right eye, left eye, nose, right mouth, left mouth
    score: float


def detect_faces(
    frame: np.ndarray,
    model_path: str | Path = DEFAULT_MODEL,
    score_threshold: float = 0.7,
    with_reason: bool = False,
):
    """Detect faces. Returns [] (and a reason) when the model file is absent."""
    path = Path(model_path)
    if not path.exists():
        return ([], WEIGHTS_ABSENT) if with_reason else []

    h, w = frame.shape[:2]
    det = cv2.FaceDetectorYN.create(str(path), "", (w, h),
                                    score_threshold=score_threshold)
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    _, faces = det.detect(bgr)
    out: list[FaceBox] = []
    if faces is not None:
        for f in faces:
            x, y, bw, bh = (int(v) for v in f[:4])
            lms = np.array(f[4:14], dtype=np.float64).reshape(5, 2)
            out.append(FaceBox(x=x, y=y, w=bw, h=bh, landmarks=lms, score=float(f[14])))
    return (out, OK) if with_reason else out


def align(frame: np.ndarray, box: FaceBox, size: int = 224) -> np.ndarray:
    """Crop the face box, clamped to frame bounds, resized to (size, size)."""
    h, w = frame.shape[:2]
    x0 = max(0, box.x)
    y0 = max(0, box.y)
    x1 = min(w, box.x + box.w)
    y1 = min(h, box.y + box.h)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    crop = frame[y0:y1, x0:x1]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
