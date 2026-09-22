"""Self-blended images: a pseudo-fake made from one real frame and nothing else.

Shiohara and Yamasaki, CVPR 2022. The method blends a frame with a
photometrically and geometrically jittered copy of *itself* under a
landmark-derived mask. The result has the one artifact every face swap shares
— a composite seam — while the identity, the camera and the lighting are
unchanged, so a detector trained on it learns the seam rather than a
generator's signature.

IMPORTANT: the jitter ranges and mask geometry below are a reimplementation
from the paper's description, not a port of the authors' code, and the exact
constants are UNVERIFIED against it. The physical property — a soft-edged
composite boundary between two versions of the same face — is what matters
and holds either way. Verify the constants before any published claim rests
on the numbers.

Why this matters legally, not just technically: it consumes only real faces
this project owns, and no licensed dataset or generator weight file. See
docs/EULA-ACCESS.md §1.
"""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence

import cv2
import numpy as np
import numpy.typing as npt

from dfd.faces import FaceBox
from dfd.types import Context, Modality, Observation, Sample

from .face_pool import FaceCrop

logger = logging.getLogger(__name__)

#: Photometric jitter. Deliberately small: a swap that survives a human
#: reviewer does not shift colour grossly, and an exaggerated range would
#: teach the detector to spot colour casts rather than seams.
BRIGHTNESS_RANGE = (-12.0, 12.0)
CONTRAST_RANGE = (0.92, 1.08)
CHANNEL_GAIN_RANGE = (0.96, 1.04)

#: Resolution jitter: downsample then upsample, so the source carries slightly
#: less detail than the target. Real swaps almost always do, because the
#: generated face is produced at a fixed and usually lower resolution.
RESCALE_RANGE = (0.70, 1.00)

#: Geometric jitter, in fractions of the box. Sub-pixel to a few pixels.
SHIFT_RANGE = (-0.03, 0.03)
SCALE_RANGE = (0.97, 1.03)

#: Mask ellipse, as fractions of the face box. Under 1.0 so the seam falls
#: inside the face rather than on the jawline, where a crop boundary would
#: confound it.
MASK_AXIS_RANGE = (0.62, 0.92)
MASK_CENTRE_JITTER = 0.04
MASK_ANGLE_RANGE = (-15.0, 15.0)

#: Feather width as a fraction of the box's smaller side. A hard edge is a
#: paste, not a blend, and would be trivially detectable for the wrong reason.
FEATHER_RANGE = (0.06, 0.18)


def _uniform(rng: np.random.Generator, lohi: tuple[float, float]) -> float:
    return float(rng.uniform(lohi[0], lohi[1]))


def jitter(frame: npt.NDArray[np.uint8],
           rng: np.random.Generator) -> npt.NDArray[np.uint8]:
    """Return a photometrically and geometrically altered copy of `frame`.

    Args:
        frame: RGB HWC uint8 image.
        rng: seeded generator; the same seed yields the same output.

    Returns:
        An RGB HWC uint8 image of identical shape.
    """
    h, w = frame.shape[:2]
    x = frame.astype(np.float32)

    # Resolution: down then up, losing detail the target still has.
    scale = _uniform(rng, RESCALE_RANGE)
    if scale < 1.0:
        small = cv2.resize(x, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
        x = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # Photometry.
    contrast = _uniform(rng, CONTRAST_RANGE)
    brightness = _uniform(rng, BRIGHTNESS_RANGE)
    gains = np.array([_uniform(rng, CHANNEL_GAIN_RANGE) for _ in range(3)],
                     dtype=np.float32)
    x = x * contrast * gains + brightness

    # Geometry: a small similarity transform about the centre.
    dx = _uniform(rng, SHIFT_RANGE) * w
    dy = _uniform(rng, SHIFT_RANGE) * h
    s = _uniform(rng, SCALE_RANGE)
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.0, s)
    m[0, 2] += dx
    m[1, 2] += dy
    x = cv2.warpAffine(x, m, (w, h), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)

    # Round rather than truncate: astype(uint8) alone floors, which would
    # give every non-integer value here a systematic ~0.5-greylevel negative
    # bias -- an artifact that is deterministic and correlated with the seam
    # in exactly the module whose job is to avoid teaching the detector an
    # unintended one.
    return np.rint(np.clip(x, 0, 255)).astype(np.uint8)


def face_mask(shape: tuple[int, int], box: FaceBox,
              rng: np.random.Generator) -> npt.NDArray[np.float32]:
    """A soft-edged elliptical mask over the inner face.

    YuNet gives five landmarks, not the 68-point contour the paper's masks are
    built from, so the mask is an ellipse fitted to the box and jittered rather
    than a landmark convex hull. The consequence is stated plainly: seams sit
    on a smoother curve than a real swap's would. Randomising the axes, centre
    and angle keeps the detector from learning one fixed boundary position,
    which is the failure this approximation would otherwise cause.

    Args:
        shape: (height, width) of the frame.
        box: the detected face.
        rng: seeded generator.

    Returns:
        float32 mask in [0, 1], shape `shape`.
    """
    h, w = shape
    canvas = np.zeros((h, w), dtype=np.uint8)

    cx = box.x + box.w / 2.0 + _uniform(rng, (-MASK_CENTRE_JITTER, MASK_CENTRE_JITTER)) * box.w
    cy = box.y + box.h / 2.0 + _uniform(rng, (-MASK_CENTRE_JITTER, MASK_CENTRE_JITTER)) * box.h
    ax = max(1, int(box.w / 2.0 * _uniform(rng, MASK_AXIS_RANGE)))
    ay = max(1, int(box.h / 2.0 * _uniform(rng, MASK_AXIS_RANGE)))
    angle = _uniform(rng, MASK_ANGLE_RANGE)

    cv2.ellipse(canvas, (int(cx), int(cy)), (ax, ay), angle, 0, 360, 255, -1)

    feather = _uniform(rng, FEATHER_RANGE) * min(box.w, box.h)
    k = max(3, int(feather) | 1)  # odd kernel
    soft = cv2.GaussianBlur(canvas.astype(np.float32) / 255.0, (k, k), 0)
    return np.clip(soft, 0.0, 1.0).astype(np.float32)


def self_blend(
    frame: npt.NDArray[np.uint8],
    box: FaceBox,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]:
    """Blend `frame` with a jittered copy of itself under a face mask.

    Args:
        frame: RGB HWC uint8 image.
        box: the detected face to blend over.
        rng: seeded generator; the same seed yields the same pseudo-fake.

    Returns:
        A (blended, mask) pair. `blended` has the same shape and dtype as
        `frame`; `mask` is float32 in [0, 1] and is returned so callers can
        record where the seam is without recomputing it.
    """
    source = jitter(frame, rng)
    mask = face_mask(frame.shape[:2], box, rng)
    m3 = mask[..., None]
    blended = source.astype(np.float32) * m3 + frame.astype(np.float32) * (1.0 - m3)
    logger.debug("self-blend: mask covers %.3f of frame", float((mask > 0.5).mean()))
    # Round rather than truncate, for the same reason as in jitter(): a floor
    # would bias every blended pixel where 0 < mask < 1 by about half a
    # greylevel, a deterministic artifact perfectly correlated with the seam.
    return np.rint(np.clip(blended, 0, 255)).astype(np.uint8), mask


#: The generator name every self-blended fake carries. LOGO holds each
#: generator out in turn and trains on the rest, so a corpus with only this
#: one generator offers LOGO no folds at all -- holding out the only
#: generator leaves nothing to train on, and `bench.protocol.logo_splits`
#: correctly refuses with `UnsplittableCorpusError` rather than fabricate
#: one. A second, licence-clean generator family is therefore a
#: precondition for running the LOGO benchmark at all, not an optional
#: improvement. See docs/HANDOFF.md §4.
SBI_GENERATOR = "sbi"


class EvaluationOnlySessionError(ValueError):
    """Raised when a session reserved for evaluation is offered for blending.

    Its own type, not a bare ValueError, because a caller may reasonably want
    to catch this and filter, while a malformed-crop ValueError means a defect
    and must propagate.
    """


def build_sbi_corpus(crops: Sequence[FaceCrop], *, seed: int = 0) -> list[Sample]:
    """Turn real face crops into a labelled, splittable corpus.

    Each crop yields two samples: the crop itself as a real, and a self-blend
    of it as a fake. They share `subject_id` so identity-disjoint splitting
    moves them together, and differ in `source_id` because
    `bench.protocol._validate` requires each source to carry exactly one
    (subject, generator) pair.

    `subject_id` below is set to `crop.session_id` -- a capture SESSION id
    (a timestamp like "20260826-221956-387743"), not a person identity.
    Splitting on it is therefore SESSION-disjoint, not identity-disjoint: a
    person who enrolled in more than one of the 442 sessions gets a
    distinct subject_id per session and can still end up on both sides of
    a split in `training/fit_blend.py`. Genuine identity-disjointness needs
    a face embedder this repo does not have yet (`docs/HANDOFF.md` §0
    next-step 3), and its absence is a correctness gap for this fitter, not
    only for benchmark criterion 2.

    This function's refusal to blend evaluation-only sessions (below)
    depends entirely on `corpora.captures.load_capture_sessions`' metadata:
    `FaceCrop.swapped` comes from `CaptureSession.swapped`, which that
    loader defaults to `False` for a session whose `results.json` has no
    `swapped` key at all (`corpora/captures.py`: `bool(d.get("swapped",
    False))`). A session missing that key would silently look genuine to
    this function and be eligible for blending. As of 2026-09-22 all 442
    real sessions carry the key explicitly (verified against the corpus at
    `/home/rohit/Desktop/agents/fraud_gff/deepfake_detection/captures`),
    so this is a recorded dependency, not a currently-live gap.

    Args:
        crops: real face crops. Any crop from a swapped session is refused.
        seed: base seed. The same seed yields the same corpus.

    Returns:
        Samples, two per crop, ordered real-then-fake per crop.

    Raises:
        EvaluationOnlySessionError: if any crop comes from a swapped session.
            These are the only labelled fraud this project has; blending them
            would spend the evaluation set on training.
    """
    reserved = sorted({c.session_id for c in crops if c.swapped})
    if reserved:
        raise EvaluationOnlySessionError(
            "refusing to blend evaluation-only sessions: "
            f"{', '.join(reserved)}. Swapped sessions are the held-out fraud "
            "set; filter them out before building a training corpus.")

    samples: list[Sample] = []
    for crop in crops:
        stem = f"{crop.session_id}-{crop.frame_index:02d}"
        # hashlib, not hash(): Python salts str hashing per process unless
        # PYTHONHASHSEED is set, so hash() would make this reproducible within
        # one run and silently irreproducible between runs — the worst of both,
        # because a test calling it twice in one process would still pass.
        digest = hashlib.sha256(stem.encode()).digest()[:8]
        rng = np.random.default_rng([seed, int.from_bytes(digest, "big")])
        blended, _mask = self_blend(crop.image, _crop_box(crop), rng)

        for suffix, payload, label, generator in (
            ("real", crop.image, 0, None),
            ("sbi", blended, 1, SBI_GENERATOR),
        ):
            samples.append(Sample(
                sample_id=f"{stem}-{suffix}",
                modality=Modality.IMAGE,
                observations=(Observation(
                    t=0.0,
                    payload=payload,
                    roi=None,
                    quality=crop.quality,
                    source_id=f"{crop.session_id}:{suffix}",
                ),),
                context=Context(
                    # Session-disjoint, not identity-disjoint: see the
                    # docstring above.
                    subject_id=crop.session_id,
                    generator=generator,
                    compression=None,
                    label=label,
                ),
            ))

    logger.info("SBI corpus: %d samples from %d crops", len(samples), len(crops))
    return samples


def _crop_box(crop: FaceCrop) -> FaceBox:
    """A box covering the aligned crop.

    `FaceCrop.box` is in the ORIGINAL frame's coordinates; `crop.image` has
    already been cropped and resized by `dfd.faces.align`, so those coordinates
    do not apply to it. Blending under the original box would put the seam
    outside the crop entirely — silently producing pseudo-fakes identical to
    their reals, which every downstream metric would then reward the detector
    for failing to separate.

    That remap covers x/y/w/h only: the returned box's `landmarks` are
    copied straight from `crop.box`, so they stay in the ORIGINAL frame's
    coordinate space while `x/y/w/h` are now in the crop's. Nothing
    downstream currently reads this box's landmarks (`face_mask` uses only
    `x/y/w/h`), so the asymmetry is inert today -- but it is real, and a
    future caller that reaches for `.landmarks` here would get points that
    do not correspond to the box they came with.
    """
    h, w = crop.image.shape[:2]
    return FaceBox(x=0, y=0, w=w, h=h,
                   landmarks=crop.box.landmarks.copy(), score=crop.box.score)
