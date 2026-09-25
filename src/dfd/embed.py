"""Face identity embeddings via OpenCV SFace (Apache-2.0, see assets/manifest.yaml).

**What this is for, and it is not recognition.** Spec acceptance criterion 2
requires a benchmark to *measure* that no identity appears on both sides of a
split. `bench.guards.check_identity_disjoint` has been able to measure it
since the beginning and has had nothing to measure with: `identity_report` was
hardcoded `None`, and the split discipline everywhere else in this repo is
session-disjointness — a person who enrolled twice gets two session ids and
can sit on both sides of a split that looks clean. This module closes that.

**Why SFace and not ArcFace.** The gap has been recorded as "no ArcFace
embedder exists" since 2026-09-21, but ArcFace's usable weights are
research-only, and an embedder is not an ordinary dependency: it decides
whether a split is certified disjoint, so a research-only embedder makes
every split certified with it research-only too. SFace ships in OpenCV Zoo
under Apache-2.0 beside the YuNet detector this repo already uses, needs no
new Python dependency, and runs on CPU in single-digit milliseconds. It is a
weaker recogniser than ArcFace. For this use the question is only whether two
crops are the same person at a threshold, and the cost of being wrong is
asymmetric: a false *match* refuses a split that was fine, a false *miss*
certifies one that was not. That asymmetry argued for a deliberately low
threshold; measuring it on this project's own faces overruled that argument,
and `DEFAULT_THRESHOLD` carries both the number and why.

Fails soft the way `dfd.faces` does: absent weights yield `None` and a stated
reason, never an exception and never a fabricated embedding. A present but
corrupt file raises, because silently treating corruption as absence hides a
deployment failure.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from .faces import FaceBox

logger = logging.getLogger(__name__)

WEIGHTS_ABSENT = "weights_absent"
NO_FACE = "no_face"
OK = "ok"

DEFAULT_MODEL = Path("assets/models/face_recognition_sface_2021dec.onnx")

#: SFace's output width. Asserted at embed time rather than trusted: a
#: different checkpoint at the same path would otherwise silently change what
#: a cosine similarity means.
EMBEDDING_DIM = 128

#: Cosine similarity at or above which two crops are called the same person.
#:
#: MEASURED on this project's own faces, 2026-09-23, not taken from a paper:
#: 499 FairFace photographs (one per subject, so all 124,251 cross pairs are
#: different people) against a jittered copy of each (`corpora.sbi.jitter`,
#: the same transformation the pseudo-fake pipeline applies).
#:
#:     different people   p50 0.042   p99 0.293   p99.9 0.394   max 0.657
#:     same person        min 0.326   p1  0.745   p50  0.967
#:
#:     threshold   different pairs flagged   same pairs missed
#:       0.30            0.8451%                  0.00%
#:       0.363           0.2141%                  0.20%
#:       0.45            0.0241%                  0.20%
#:       0.60            0.0016%                  0.20%
#:
#: The first draft of this module set 0.30, reasoning that a guard should err
#: toward refusing splits. The measurement says that reasoning does not
#: survive contact with pair COUNTS: flagging 0.85% of unrelated pairs means a
#: 1,000 x 300 split produces ~2,500 "violations" that are not leakage, so the
#: guard would refuse every split and be turned off within a week. 0.363 —
#: which is also OpenCV's own verification point — is where the two error
#: rates cross on this data.
#:
#: **The distributions overlap and no threshold removes that.** Unrelated
#: faces reach 0.657 here; the same person under jitter can fall to 0.326. A
#: guard reading the MAXIMUM pair is therefore a screen for small splits only
#: — see `bench.guards.check_identity_disjoint`'s `max_false_match_rate`,
#: which exists because of this measurement.
DEFAULT_THRESHOLD = 0.363


@dataclass(frozen=True)
class Embedder:
    """A loaded SFace recogniser, and the path it came from.

    Holds the cv2 object rather than reloading per call: loading is ~40 MB of
    ONNX parsing and the benchmark embeds thousands of crops. Not thread-safe
    — cv2's recogniser is stateful — so callers that fan out must hold one per
    worker (`corpora`-side extraction does exactly that).
    """
    model_path: Path
    _net: cv2.FaceRecognizerSF

    def embed(self, frame: npt.NDArray[np.uint8],
              box: FaceBox) -> npt.NDArray[np.float32] | None:
        """The L2-normalised identity vector for one detected face.

        Args:
            frame: RGB HWC uint8 image, in the coordinate space `box` was
                detected in. RGB because that is what `dfd.faces.detect_faces`
                takes and what every caller in this repo already holds;
                converted to BGR here because OpenCV's recogniser expects it.
                Passing BGR in would not raise — it would quietly return
                embeddings of a colour-swapped face, which is why the
                convention is stated rather than inferred.
            box: a detection from `dfd.faces.detect_faces` on that same frame.
                Its landmarks are what `alignCrop` uses; a box built by hand
                with placeholder landmarks (as `corpora.sbi._crop_box` makes)
                will align to the wrong points and must not be passed here.

        Returns:
            A float32 vector of length `EMBEDDING_DIM`, L2-normalised so that
            a dot product IS the cosine similarity, or None when the crop
            cannot be aligned.

        Raises:
            ValueError: if the loaded model returns a vector of unexpected
                width — a different checkpoint at the same path.
        """
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        row = _detection_row(box)
        try:
            aligned = self._net.alignCrop(bgr, row)
        except cv2.error:
            logger.debug("alignCrop failed for box %s", box)
            return None
        if aligned is None or aligned.size == 0:
            return None
        feat = np.asarray(self._net.feature(aligned), dtype=np.float32).reshape(-1)
        if feat.shape[0] != EMBEDDING_DIM:
            raise ValueError(
                f"expected a {EMBEDDING_DIM}-d embedding from {self.model_path}, "
                f"got {feat.shape[0]}: this is not the checkpoint this module "
                "was calibrated against, and its similarities are not comparable")
        norm = float(np.linalg.norm(feat))
        if norm == 0.0:
            # Not a normal outcome, and not an error either: a zero vector has
            # no direction, so no similarity to anything is defined. Refusing
            # is what keeps the guard from reading 0.0 as "dissimilar".
            logger.warning("zero-norm embedding from %s; refusing it",
                           self.model_path)
            return None
        return (feat / norm).astype(np.float32)


def _detection_row(box: FaceBox) -> npt.NDArray[np.float32]:
    """The 15-value detection row `alignCrop` expects.

    OpenCV's recogniser takes YuNet's raw output row — x, y, w, h, then five
    (x, y) landmark pairs, then the score — not a structured object.
    `dfd.faces.detect_faces` parses that row into a `FaceBox`; this rebuilds
    it, in the same order, so the two stay a matched pair. Rebuilding rather
    than carrying the raw row on `FaceBox` keeps the detector's output format
    from leaking into every consumer of a detection.
    """
    return np.array(
        [box.x, box.y, box.w, box.h, *box.landmarks.reshape(-1).tolist(), box.score],
        dtype=np.float32)


def load_embedder(model_path: str | Path = DEFAULT_MODEL) -> tuple[Embedder | None, str]:
    """Load the recogniser, or report why not.

    Args:
        model_path: the SFace `.onnx`.

    Returns:
        An (embedder, reason) pair. The embedder is None exactly when reason
        is `WEIGHTS_ABSENT`.

    Raises:
        cv2.error: the file exists and cannot be parsed. Deliberately not
            caught: a corrupt model on a deployment that believes it has one
            is a different failure from having none, and only one of them is
            fixed by downloading the file again.
    """
    path = Path(model_path)
    if not path.exists():
        logger.warning("Face embedder weights absent at %s", path)
        return None, WEIGHTS_ABSENT
    net = cv2.FaceRecognizerSF.create(str(path), "")
    logger.debug("loaded face embedder from %s", path)
    return Embedder(model_path=path, _net=net), OK


def cosine(a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]) -> float:
    """Cosine similarity between two embeddings from `Embedder.embed`.

    Both are already L2-normalised, so this is their dot product; it is
    written as a full cosine anyway so that a caller passing an un-normalised
    vector from somewhere else still gets the right answer rather than a
    silently scaled one.

    Returns:
        A similarity in [-1, 1]; 0.0 if either vector has zero norm.
    """
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
