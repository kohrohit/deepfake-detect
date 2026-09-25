"""Licence-clean face swaps: one person's face composited into another's photograph.

**Why this exists, and why it is not `corpora/sbi.py`.** Self-blending makes a
pseudo-fake from ONE photograph, which is the right training target and cannot
serve as an evaluation corpus: there is no second identity, and every fake
carries the single generator label `sbi`, so `bench.protocol.logo_splits`
refuses the corpus and leave-one-generator-out — the only number spec §8.1
says predicts field performance — has never once run in this project.

This module makes real swaps between two real photographs, by four techniques
that leave four DIFFERENT physical signatures. Four generator labels is what
turns a corpus into one LOGO can fold. Measured against the three properties
`docs/EULA-ACCESS.md` §4a says an evaluation corpus needs:

1. **Real and fake halves share an imaging chain.** Both sides are FairFace
   photographs. Nothing here can learn "smooth means fake" the way every fit
   against SFHQ did (docs/HANDOFF.md, the retracted slot C result).
2. **Sources are not one family.** 43,179 disjoint couples drawn from 86,358
   distinct photographs, against DF40-repackaged's effective n of 2.4.
3. **Per-technique labels**, because we are the technique.

**WHAT THIS CORPUS IS NOT.** These are classical compositing swaps — warp,
mask, blend — not GAN or diffusion output. A detector that beats them has not
been shown to beat FSGAN or InSwapper, and no number measured here may be
reported as if it had. What it CAN do is refute a detector (one that cannot
find a hard-blended composite boundary will not find a subtle one) and support
leave-one-generator-out ACROSS THESE FOUR, which is a real generalisation test
between real techniques even though all four are classical. The research
datasets remain the thing to buy; this is what can be measured while waiting,
and it is more than zero, which is what is measurable today.

**Landmark limitation, stated once.** YuNet gives five points, not the 68-point
contour these methods assume, so masks are ellipses fitted to the face box
rather than jaw-following hulls, and alignment is a five-point similarity
transform rather than a piecewise warp. Seams therefore sit on smoother curves
than a production swap's would. Same approximation `corpora/sbi.py` documents,
same consequence.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from dfd.faces import FaceBox

logger = logging.getLogger(__name__)

#: The four techniques, in the order they are declared. These strings are
#: GENERATOR LABELS and reach `bench.protocol.logo_splits` directly, so
#: renaming one invalidates comparison against every report already written.
WARP_HULL = "swap_warp_hull"
POISSON = "swap_poisson"
MOUTH_PATCH = "swap_mouth_patch"
LOWRES_PASTE = "swap_lowres_paste"

TECHNIQUES = (WARP_HULL, POISSON, MOUTH_PATCH, LOWRES_PASTE)

#: Mask ellipse axes as a fraction of the face box, per technique. Under 1.0
#: so the boundary falls inside the face rather than on the jawline, where a
#: crop edge would confound it — `corpora/sbi.py` gives the same reason.
MASK_AXIS = 0.78
#: Feather width as a fraction of the box's smaller side. A hard edge is a
#: paste, not a blend, and a detector that learns to find it has learned
#: nothing transferable.
FEATHER = 0.10
#: How far `LOWRES_PASTE` downsamples the source face before putting it back.
#: This is the one technique aimed squarely at slot C: a generated face
#: arrives at a fixed, usually lower resolution than the frame it lands in,
#: and the upsampling that follows is the fingerprint NPR features read.
LOWRES_SCALE = 0.25


@dataclass(frozen=True)
class SwapResult:
    """A composited fake, and what made it."""
    image: npt.NDArray[np.uint8]
    generator: str
    #: float32 in [0, 1]; where the composite boundary actually is. Kept so a
    #: future localisation head has a target and so tests can assert a swap
    #: changed the pixels it claimed to.
    mask: npt.NDArray[np.float32]


def _similarity_transform(src: npt.NDArray[np.float64],
                          dst: npt.NDArray[np.float64]) -> npt.NDArray[np.float64] | None:
    """The 2x3 similarity mapping `src` landmarks onto `dst`, or None.

    `estimateAffinePartial2D` returns rotation, uniform scale and translation
    only — no shear and no per-axis scale. That restriction is deliberate: a
    full affine fitted to five noisy points distorts the face enough to be
    detectable as distortion rather than as a swap, which would teach a
    detector the wrong artifact.
    """
    m, _ = cv2.estimateAffinePartial2D(src.astype(np.float32),
                                       dst.astype(np.float32), method=cv2.LMEDS)
    return None if m is None else np.asarray(m, dtype=np.float64)


def ellipse_mask(shape: tuple[int, int], box: FaceBox, *,
                 axis: float = MASK_AXIS,
                 feather: float = FEATHER) -> npt.NDArray[np.float32]:
    """A soft-edged ellipse over the inner face, float32 in [0, 1]."""
    h, w = shape
    canvas = np.zeros((h, w), dtype=np.uint8)
    cx, cy = box.x + box.w / 2.0, box.y + box.h / 2.0
    ax = max(1, int(box.w / 2.0 * axis))
    ay = max(1, int(box.h / 2.0 * axis))
    cv2.ellipse(canvas, (int(cx), int(cy)), (ax, ay), 0, 0, 360, 255, -1)
    k = max(3, int(feather * min(box.w, box.h)) | 1)
    soft = cv2.GaussianBlur(canvas.astype(np.float32) / 255.0, (k, k), 0)
    return np.clip(soft, 0.0, 1.0).astype(np.float32)


def colour_transfer(source: npt.NDArray[np.uint8], target: npt.NDArray[np.uint8],
                    mask: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    """Match `source`'s colour statistics to `target`'s, inside `mask`.

    Reinhard's method in Lab. Without it, a swap between two people under
    different lighting is detectable from the colour step alone — which is
    the shortcut that made colour means outscore seam features on DF40 (AUC
    0.843, docs/HANDOFF.md). A corpus that leaves it in is measuring skin
    tone, not forgery.
    """
    m = mask > 0.1
    if m.sum() < 16:
        return source
    s_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    t_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)
    out = s_lab.copy()
    for c in range(3):
        s_mean, s_std = s_lab[..., c][m].mean(), s_lab[..., c][m].std()
        t_mean, t_std = t_lab[..., c][m].mean(), t_lab[..., c][m].std()
        if s_std < 1e-6:
            continue
        out[..., c] = (s_lab[..., c] - s_mean) * (t_std / s_std) + t_mean
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def _aligned_source(source: npt.NDArray[np.uint8], source_box: FaceBox,
                    target: npt.NDArray[np.uint8],
                    target_box: FaceBox) -> npt.NDArray[np.uint8] | None:
    m = _similarity_transform(source_box.landmarks, target_box.landmarks)
    if m is None:
        return None
    h, w = target.shape[:2]
    return cv2.warpAffine(source, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101).astype(np.uint8)


def swap(source: npt.NDArray[np.uint8], source_box: FaceBox,
         target: npt.NDArray[np.uint8], target_box: FaceBox,
         technique: str, rng: np.random.Generator) -> SwapResult | None:
    """Composite `source`'s face into `target`'s photograph.

    Args:
        source: RGB HWC uint8 photograph supplying the identity.
        source_box: its detection, whose landmarks drive alignment.
        target: RGB HWC uint8 photograph receiving the face. The result has
            this image's shape, lighting and background.
        target_box: its detection.
        technique: one of `TECHNIQUES`.
        rng: seeded generator; the same seed yields the same fake.

    Returns:
        A `SwapResult`, or None when the two faces cannot be aligned (a
        degenerate landmark set). None is not an error — it is a pair to
        skip, and the caller counts it.

    Raises:
        ValueError: on an unknown technique. Silently returning the target
            unchanged would put a REAL image into the corpus under a fake
            label, which is the one defect no downstream metric could see.
    """
    if technique not in TECHNIQUES:
        raise ValueError(f"unknown technique {technique!r}; "
                         f"expected one of {list(TECHNIQUES)}")

    warped = _aligned_source(source, source_box, target, target_box)
    if warped is None:
        return None
    mask = ellipse_mask(target.shape[:2], target_box)
    if float(mask.max()) <= 0.0:
        return None

    if technique == MOUTH_PATCH:
        # Reenactment family: only the mouth moves. The composite boundary
        # sits in a different place from a full swap's, which is the point —
        # a detector tuned to jaw-line seams should measurably struggle here,
        # and LOGO is what will show that.
        mask = _mouth_mask(target.shape[:2], target_box)

    if technique == LOWRES_PASTE:
        h, w = target.shape[:2]
        small = cv2.resize(warped, (max(1, int(w * LOWRES_SCALE)),
                                    max(1, int(h * LOWRES_SCALE))),
                           interpolation=cv2.INTER_AREA)
        warped = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

    warped = colour_transfer(warped, target, mask)

    if technique == POISSON:
        # Gradient-domain compositing leaves no colour step at the boundary
        # at all, so a detector reading the seam as a colour discontinuity
        # is blind to it by construction. That is why it is a separate
        # generator rather than a variant.
        binary = (mask > 0.5).astype(np.uint8) * 255
        ys, xs = np.nonzero(binary)
        if len(xs) < 16:
            return None
        centre = (int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2))
        try:
            blended = cv2.seamlessClone(
                cv2.cvtColor(warped, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(target, cv2.COLOR_RGB2BGR),
                binary, centre, cv2.NORMAL_CLONE)
        except cv2.error:
            # seamlessClone refuses masks that touch the border. A pair this
            # happens to is skipped, not fudged into a different technique.
            return None
        out = cv2.cvtColor(np.asarray(blended, dtype=np.uint8), cv2.COLOR_BGR2RGB)
        return SwapResult(image=out.astype(np.uint8), generator=technique,
                          mask=mask)

    m3 = mask[..., None]
    out = warped.astype(np.float32) * m3 + target.astype(np.float32) * (1.0 - m3)
    # Round rather than truncate: a floor biases every blended pixel by about
    # half a grey level, an artifact perfectly correlated with the mask —
    # exactly the unintended signal `corpora/sbi.py` documents avoiding.
    return SwapResult(image=np.rint(np.clip(out, 0, 255)).astype(np.uint8),
                      generator=technique, mask=mask)


def _mouth_mask(shape: tuple[int, int], box: FaceBox) -> npt.NDArray[np.float32]:
    """An ellipse over the lower third of the face box.

    YuNet's landmarks 3 and 4 are believed to be the mouth corners, but
    `dfd.faces.FaceBox` records that the order is UNVERIFIED against real
    model output. So this derives the region from the BOX geometry, which is
    unambiguous, rather than from landmarks whose identity is not.
    """
    h, w = shape
    canvas = np.zeros((h, w), dtype=np.uint8)
    cx = box.x + box.w / 2.0
    cy = box.y + box.h * 0.72
    ax = max(1, int(box.w * 0.30))
    ay = max(1, int(box.h * 0.16))
    cv2.ellipse(canvas, (int(cx), int(cy)), (ax, ay), 0, 0, 360, 255, -1)
    k = max(3, int(0.06 * min(box.w, box.h)) | 1)
    return np.clip(cv2.GaussianBlur(canvas.astype(np.float32) / 255.0, (k, k), 0),
                   0.0, 1.0).astype(np.float32)


def rng_for(*parts: str, seed: int = 0) -> np.random.Generator:
    """A generator keyed on stable strings, reproducible ACROSS processes.

    `hash()` is salted per process unless PYTHONHASHSEED is set, so a corpus
    built with it would be reproducible within one run and silently different
    between runs — the worst of both, since a test calling it twice in one
    process still passes. Same reasoning as `corpora/sbi.py`.
    """
    digest = hashlib.sha256("|".join(parts).encode()).digest()[:8]
    return np.random.default_rng([seed, int.from_bytes(digest, "big")])


#: Generator labels whose SOURCE pixels are not a second FairFace photograph.
#: They reach `bench.protocol.logo_splits` as generators like the four
#: techniques above, but what distinguishes them is provenance rather than
#: blending: both composite with `WARP_HULL`, and differ only in where the
#: face being pasted came from.
#:
#: `SYNTH_SFHQ` pastes an SFHQ part-3 face — documented StyleGAN2 sampling,
#: CC0/MIT, no depicted real person — into a FairFace photograph. It is the
#: only fake in this project whose face pixels are genuine GENERATOR OUTPUT
#: sitting inside a real camera's imaging chain, which is the cell no corpus
#: here has ever filled: SFHQ used directly against FairFace reals separates
#: on colour alone, because the two halves arrive down different chains.
SYNTH_SFHQ = "synth_sfhq_stylegan2"
#: `SYNTH_CONTROL` is the measurement that makes `SYNTH_SFHQ` readable, and
#: neither may be reported without the other.
#:
#: SFHQ is 1024px and FairFace is 224px, so compositing one into the other
#: DOWNSAMPLES the face roughly threefold — and a 3x downsample is a low-pass
#: filter that destroys much of the high-frequency fingerprint an upsampling
#: detector (slot C) reads. A detector that separates `SYNTH_SFHQ` from real
#: might therefore be reading the generator, or might be reading the resample.
#: Those are different claims and one of them is worthless.
#:
#: So the control runs the couple's OWN FairFace source through the identical
#: path — upscaled to SFHQ's native size, then composited by the same
#: technique. Its face pixels are photographic; only the resampling history is
#: shared. **If a detector scores `SYNTH_SFHQ` and `SYNTH_CONTROL` alike, it
#: has found the resample and learnt nothing about generators.** A gap between
#: them is the generator fingerprint, and it is the only part of a number on
#: this corpus that means what it appears to mean.
SYNTH_CONTROL = "synth_control_resampled"

#: The side length `SYNTH_CONTROL` upscales a FairFace source to before
#: compositing. Equal to SFHQ part 3's native resolution, because the control
#: is only a control if the two paths differ in ONE thing.
SYNTH_NATIVE_SIZE = 1024


def rescale(frame: npt.NDArray[np.uint8], box: FaceBox, size: int,
            ) -> tuple[npt.NDArray[np.uint8], FaceBox]:
    """Resize a square frame to `size`, carrying its detection with it.

    The box is scaled rather than re-detected. Re-detecting would be the
    obvious alternative and is wrong here: the control must differ from
    `SYNTH_SFHQ` in resampling history alone, and a second detection would
    introduce a different face box as well, which is a second difference.

    Args:
        frame: RGB HWC uint8. Non-square input is resized anyway; the box
            scales per-axis, so the geometry stays consistent either way.
        box: its detection.
        size: target side length.

    Returns:
        The resized frame and its scaled box.
    """
    h, w = frame.shape[:2]
    out = cv2.resize(frame, (size, size), interpolation=cv2.INTER_CUBIC)
    sx, sy = size / float(w), size / float(h)
    return out, FaceBox(
        x=int(round(box.x * sx)), y=int(round(box.y * sy)),
        w=max(1, int(round(box.w * sx))), h=max(1, int(round(box.h * sy))),
        landmarks=box.landmarks * np.array([sx, sy]), score=box.score)
