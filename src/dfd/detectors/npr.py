"""Slot C — upsampling fingerprint (spec §6).

Physics: neural generators upsample via transposed convolution or
interpolate-then-convolve, which leaves a periodic residual structure that a
camera's optical chain does not produce. The NPR feature isolates that residual
by subtracting the image's own downsample-then-upsample reconstruction.

IMPORTANT: The exact feature definition below (stride=2 nearest-neighbour down/up)
is specified verbatim but is UNVERIFIED against the NPR paper (Tan et al., CVPR
2024, "Rethinking the Up-Sampling Operations in CNN-based Generative Network for
Generalizable Deepfake Detection"). This implementation must be verified against
that source before any published claim relies on the numerical values. The physical
property (upsampled content → near-zero residual) is what matters; the constants
remain valid either way.

Supply-chain control: weights are loaded with torch.load(..., weights_only=True)
by default, preventing arbitrary code execution from tampered models (spec §3A).
Full-module pickles must be explicitly enabled and logged as a warning.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from ..types import Modality, Observation, RawScore
from .base import (
    NO_OBSERVATIONS,
    OK,
    WEIGHTS_ABSENT,
    abstain,
    filter_by_quality_floor,
)
from .loading import load_model

logger = logging.getLogger(__name__)

# Downsampling stride for NPR feature. Matches Tan et al., CVPR 2024 spec.
DEFAULT_STRIDE = 2

# Expected number of output classes (real/fake binary classifier).
EXPECTED_NUM_CLASSES = 2


def npr_feature(
    img: npt.NDArray[np.uint8], stride: int = DEFAULT_STRIDE
) -> npt.NDArray[np.float32]:
    """Compute NPR (Neural-generated Periodic Residual) feature.

    The feature is the residual between the input image and its own
    downsample-then-upsample reconstruction using nearest-neighbour
    interpolation. This highlights the periodic artifacts left by neural
    generator upsampling, which natural images do not exhibit.

    Args:
        img: input image as uint8 HWC array
        stride: downsampling stride (default matches NPR paper)

    Returns:
        float32 residual array with same shape as input, in range [-1, 1]

    Raises:
        No exceptions raised; input is always valid numpy arrays.
    """
    x: npt.NDArray[np.float32] = img.astype(np.float32) / 255.0
    down = x[::stride, ::stride]
    up: npt.NDArray[np.float32] = np.repeat(
        np.repeat(down, stride, axis=0), stride, axis=1
    ).astype(np.float32)
    # Trim to match input shape (in case height or width not divisible by stride)
    up = up[: x.shape[0], : x.shape[1]]
    residual: npt.NDArray[np.float32] = (x - up).astype(np.float32)
    return residual


#: The four (dy, dx) sampling phases of a stride-2 grid, minus the one that
#: carries nothing. Nearest-neighbour upsampling REPLICATES the sampled
#: pixel, so `up[2i, 2j] == x[2i, 2j]` and the residual on the even phase is
#: identically zero for every image, real or generated. A statistic computed
#: there is a constant, not a feature.
INFORMATIVE_PHASES: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1))

#: Keeps the log-ratios finite when a phase's residual is exactly zero, which
#: happens on flat fields and on synthetic test images. A nan here would
#: reach the linear layer and poison every output silently.
_RATIO_EPS = 1e-6


#: The ROI was absent, so the detector does not know which pixels are the
#: face. Mirrors `dfd.detectors.blend.NO_ROI` and exists for the same reason:
#: scoring the whole frame instead silently reads mostly background. Distinct
#: from the quality reasons (about capture, not about localisation) and from
#: WEIGHTS_ABSENT (the detector never ran).
NO_ROI = "roi_absent"


class NPRStatsNet(nn.Module):
    """Slot C's model: phase statistics of the NPR residual, then a linear layer.

    **Why this shape and not a CNN.** The hardware ruling is CPU-only
    (docs/HANDOFF.md §1, correction 3), which makes "handcrafted features
    feeding a light model" the only detector this project can actually fit.
    This is that: 27 numbers describing how the upsampling residual is
    distributed across the stride-2 sampling phases, and a `Linear(27, 2)`.
    It trains in seconds and runs in milliseconds.

    **The physics it reads.** A generator that upsamples by replication or
    interpolation leaves content that reconstructs almost exactly under
    downsample-then-upsample, so its residual is small AND its residual is
    distributed differently across the three informative phases than a
    camera's optical chain leaves it. The ratios below are there to read the
    second property, which is scale-free — the absolute magnitudes are not,
    and that matters because resolution has already been measured as this
    project's most reliable shortcut (docs/HANDOFF.md §0).

    **Feature layout**, fixed because a fitted `state_dict` is meaningless
    without it:

        f[ 0: 9]  mean |residual|, channel-major over the three phases
        f[ 9:18]  standard deviation of the residual, same order
        f[18:27]  log-ratios of the phase means, per channel

    **The normalisation travels in the state_dict.** `feature_mean` and
    `feature_scale` are buffers, not fitter-side bookkeeping: a model that
    standardised during fitting and not at inference loads clean, scores
    confidently, and is wrong. `training/fit_npr.py` writes them.
    """

    #: Length of the PHASE block: per-channel mean and std over the three
    #: informative phases, plus their log-ratios. Scale-free by design in the
    #: ratio third, which is what makes it survive a resolution change.
    N_PHASE_FEATURES = 27

    #: Length of the SPECTRAL block appended 2026-09-24: per channel, three
    #: magnitude moments and a coarse three-band profile of the residual's
    #: 2-D spectrum, plus its peak. 10 numbers x 3 channels.
    N_SPECTRAL_FEATURES = 30

    #: Length of the feature vector `features` returns. A fitted state_dict
    #: is tied to it, so changing it invalidates every saved model.
    #:
    #: **Changed 2026-09-24, from 27 to 57, and nothing was invalidated**
    #: because no fitted model had ever been written — `assets/models/npr.pt`
    #: has been absent since this class was declared. Measured on 506 frames
    #: of the v-CIP capture corpus (106 sessions, real `inswapper_128` swaps
    #: of genuine capture frames, folds split on session), each block fitted
    #: on its own columns of one shared feature matrix:
    #:
    #:     phase block only (27)      AUC 0.861, caught  4.4% at zero FPR
    #:     spectral block only (30)   AUC 0.930, caught 11.8%
    #:     both (57)                  AUC 0.957, caught 45.6%
    #:
    #: AUC understates what the spectral block buys and the operating point
    #: does not, which is why both are quoted: the product decides at a
    #: false-alarm budget, not at a ranking.
    #:
    #: **Re-measured 2026-09-25 and these numbers replace an earlier set**
    #: (0.869/7.4% and 92.6%) that this comment carried until then. Those
    #: were taken before `training.fit_vcip.select_face` was corrected to the
    #: pipeline's largest-face rule and were never recomputed after it. The
    #: ranking of the three rows is unchanged; every figure in them moved.
    #: `bench/vcip_controls.json` is the committed measurement, and
    #: `python3 -m bench.vcip_controls` regenerates it.
    N_FEATURES = N_PHASE_FEATURES + N_SPECTRAL_FEATURES

    #: Declared for the type checker only. `register_buffer` assigns through
    #: `nn.Module.__setattr__`, so a reader of these attributes gets
    #: `Tensor | Module` from `nn.Module.__getattr__` and `forward`'s
    #: arithmetic on them does not type-check. These annotations bind no
    #: value -- the buffers are still created in `__init__`, and still move
    #: with `.to()` and appear in `state_dict()` as buffers must.
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(self.N_FEATURES))
        self.register_buffer("feature_scale", torch.ones(self.N_FEATURES))
        self.linear = nn.Linear(self.N_FEATURES, 2)

    def features(self, residual: torch.Tensor) -> torch.Tensor:
        """Phase statistics of an NPR residual batch.

        Args:
            residual: `(B, 3, H, W)` float tensor, as produced by
                `npr_feature` and permuted to channels-first.

        Returns:
            `(B, N_FEATURES)` float tensor. Always finite: an empty phase
            (an image smaller than the stride) and a flat field both
            contribute zeros rather than nan.
        """
        means: list[torch.Tensor] = []
        stds: list[torch.Tensor] = []
        for dy, dx in INFORMATIVE_PHASES:
            sub = residual[:, :, dy::2, dx::2].flatten(2)
            if sub.shape[-1] == 0:
                zeros = residual.new_zeros(residual.shape[0], residual.shape[1])
                means.append(zeros)
                stds.append(zeros)
                continue
            means.append(sub.abs().mean(-1))
            stds.append(sub.std(-1, unbiased=False))

        mean_stack = torch.stack(means, dim=2)          # (B, C, phase)
        std_stack = torch.stack(stds, dim=2)
        ratios = torch.stack([
            torch.log((mean_stack[:, :, a] + _RATIO_EPS)
                      / (mean_stack[:, :, b] + _RATIO_EPS))
            for a, b in ((0, 1), (0, 2), (1, 2))
        ], dim=2)
        phase = torch.cat([mean_stack.flatten(1), std_stack.flatten(1),
                           ratios.flatten(1)], dim=1)
        return torch.cat([phase, self._spectral(residual)], dim=1)

    def _spectral(self, residual: torch.Tensor) -> torch.Tensor:
        """Magnitude moments and a coarse spectral profile, per channel.

        **What this reads that the phase block cannot.** The phase block is
        per-phase mean, std and ratios: two residuals can match on all of
        those while distributing their energy across FREQUENCY completely
        differently. An upsampled generator patch is smooth where a camera's
        sensor noise is not, and that difference is the whole signal at low
        false-alarm budgets — where the phase block alone caught 4.4% of
        swapped sessions on the capture corpus and the two blocks together
        catch 45.6% (`bench/vcip_controls.json`, re-measured 2026-09-25;
        this line read 7.4% and 92.6% until then, from a run made before the
        fitter's face rule was corrected).

        Ten numbers per channel: mean, std and mean-absolute of the residual,
        its 90th and 99th absolute percentiles, three band means of the 2-D
        real FFT magnitude, and that spectrum's peak and mean.

        Returns:
            `(B, N_SPECTRAL_FEATURES)`, always finite. An image smaller than
            two pixels has no spectrum to take, and contributes zeros rather
            than nan — a nan here reaches the linear layer and leaves it as a
            confident score.
        """
        b, c = residual.shape[0], residual.shape[1]
        if residual.shape[-1] < 2 or residual.shape[-2] < 2:
            return residual.new_zeros(b, self.N_SPECTRAL_FEATURES)

        flat = residual.flatten(2)
        absflat = flat.abs()
        # `quantile` over the spatial axis; interpolation matches numpy's
        # default so the fitter and the detector agree to the last bit.
        q90 = absflat.quantile(0.90, dim=-1)
        q99 = absflat.quantile(0.99, dim=-1)

        spec = torch.fft.rfft2(residual).abs()
        h = spec.shape[-2]
        # Three bands over the vertical frequency axis. Coarse on purpose: a
        # fine profile is a resolution fingerprint, and resolution is this
        # project's most reliable shortcut.
        #
        # EVERY SPECTRAL TERM IS DIVIDED BY THE SPECTRUM'S OWN MEAN, which
        # makes it dimensionless. Raw FFT magnitude scales with the number of
        # pixels, so an un-normalised band mean is partly a measure of image
        # SIZE — precisely the shortcut this project keeps rediscovering, and
        # it broke `fit_npr`'s resolution-invariance test the moment it was
        # added. What survives the division is how energy is DISTRIBUTED
        # across frequency, which is the physical property worth reading.
        spec_mean = spec.flatten(2).mean(-1).clamp_min(_RATIO_EPS)
        lo = spec[:, :, :max(1, h // 4)].flatten(2).mean(-1) / spec_mean
        mid = (spec[:, :, max(1, h // 4):max(2, h // 2)].flatten(2).mean(-1)
               / spec_mean)
        tail = spec[:, :, max(2, h // 2):]
        hi = (tail.flatten(2).mean(-1) / spec_mean if tail.numel()
              else spec.new_zeros(b, c))
        peak = spec.flatten(2).amax(-1) / spec_mean

        stacked = torch.stack([
            flat.mean(-1), flat.std(-1, unbiased=False), absflat.mean(-1),
            q90, q99, lo, mid, hi, peak,
            # The one absolute term, in log space so it cannot dominate the
            # standardiser: residual energy per pixel, which is a property of
            # the imaging chain rather than of the image's size.
            torch.log(absflat.mean(-1).clamp_min(_RATIO_EPS)),
        ], dim=2)
        return torch.nan_to_num(stacked.flatten(1), nan=0.0,
                                posinf=0.0, neginf=0.0)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        """Two logits per sample, ordered (real, fake) to match `NPRDetector`.

        Args:
            residual: `(B, 3, H, W)` NPR residual batch.

        Returns:
            `(B, 2)` logits. `NPRDetector.score` softmaxes these and takes
            column 1 as P(fake).
        """
        f = (self.features(residual) - self.feature_mean) / self.feature_scale
        out: torch.Tensor = self.linear(f)
        return out


def _crop_to_roi_native(
    payload: npt.NDArray[np.uint8],
    roi: tuple[int, int, int, int] | None,
) -> npt.NDArray[np.uint8] | None:
    """Crop to the face ROI at its OWN resolution, or None.

    **Deliberately does not resize, and that is the opposite of
    `dfd.detectors.blend._crop_to_roi`.** The two differ because the
    quantities differ. Seam features normalise annulus GEOMETRY by the
    image's height and width, so they need a fixed scale to be comparable.
    The NPR residual IS a scale-dependent quantity — it is the difference
    between an image and its own stride-2 reconstruction — so resampling the
    crop rewrites the very fingerprint this slot reads.

    Measured on the v-CIP capture corpus (506 frames, 106 sessions, real
    `inswapper_128` swaps of genuine capture frames, folds split on session;
    re-measured 2026-09-25 by `bench/vcip_controls.py`, which varies the crop
    and holds the face-selection rule fixed — the run this replaces varied
    both at once). `inswapper_128` emits a 128x128 face and pastes it back
    upscaled; the median face in a 720x1280 capture frame is 237px across, so
    aligning to a 224 square DOWNSAMPLES it:

        aligned to 224   AUC 0.695, caught  1.5% at a zero false-alarm budget
        native ROI       AUC 0.957, caught 45.6% at the same budget

    (The figures here until 2026-09-25 — 0.924/51.5% and 0.992/89.7% — were
    taken before the fitter's face rule was corrected and were never
    recomputed. The conclusion they supported holds and the gap is wider than
    they showed; the numbers themselves were wrong.)

    The consequence for anyone fitting this slot: the fitter MUST crop the
    same way, or the weights describe features the detector never computes.

    Args:
        payload: the observation's image, HWC.
        roi: `(x, y, w, h)` in `payload`'s coordinates, or None.

    Returns:
        The cropped view, or None when `roi` is None or the clamped region
        is smaller than 2x2 — below which there is no spectrum to take.
    """
    if roi is None:
        return None
    x, y, w, h = roi
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1 = min(payload.shape[1], int(x) + int(w))
    y1 = min(payload.shape[0], int(y) + int(h))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    crop: npt.NDArray[np.uint8] = payload[y0:y1, x0:x1]
    return crop


@dataclass(frozen=True)
class NPRDetector:
    """Detector that reads upsampling fingerprints via NPR features.

    This detector is frozen to maintain registry identity invariant:
    once registered, it cannot mutate its name or configuration.
    Model weights are loaded via a module-level cache with staleness detection
    (mtime_ns, size) to avoid silent re-use of replaced files, and thread-safe
    to protect against concurrent first-load races.

    Supply-chain control: weights are loaded securely by default via
    torch.load(..., weights_only=True) combined with a model_factory that
    reconstructs the architecture and loads the state_dict into it. This
    prevents arbitrary code execution from tampered models (spec §3A).

    **Secure path:** Provide model_factory as a callable that returns an
    instantiated, uninitialized model (e.g., `lambda: torch.nn.Linear(3, 2)`).
    The weights file must be a state_dict. This path uses weights_only=True and
    produces no warnings.

    **Unsafe path:** Omit model_factory and set allow_unsafe_load=True to load
    full-module pickles. This permits arbitrary code execution and logs a warning.

    Note: A state_dict alone cannot become a model without knowing the architecture.
    Secure loading therefore requires the architecture to be known (via model_factory).
    This is a genuine constraint, not a limitation of this implementation.

    Attributes:
        weights_path: path to the detector's trained weights file
        model_factory: callable that returns an uninitialized model instance.
            If provided, enables the secure weights_only=True path. If None,
            only full-module pickles with allow_unsafe_load=True are accepted.
        name: unique identifier (read-only via property)
        slot: detector slot designation (read-only via property)
        version: detector version string (read-only via property)
        modalities: which modalities this detector supports (read-only via property)
        min_quality_band: minimum quality band for consultation (read-only via property)
        allow_unsafe_load: if True, load full-module pickles unsafely; logs warning
    """
    weights_path: str | Path
    model_factory: Callable[[], nn.Module] | None = None
    allow_unsafe_load: bool = False

    # Identity fields (read-only after frozen)
    name: str = "npr"
    slot: str = "C"
    version: str = "0.1.0"
    modalities: frozenset[Modality] = field(
        default_factory=lambda: frozenset({Modality.IMAGE, Modality.VIDEO})
    )
    #: **`"reject"`, and that is deliberate — this detector must see blurred
    #: faces.** Every other detector treats low quality as low reliability
    #: and refuses. For slot C, low quality is positively CORRELATED with the
    #: artefact: `inswapper_128` emits a 128x128 face pasted back upscaled,
    #: so a swap blurs, and a sharpness-based floor reads that blur as a bad
    #: capture. Measured on the v-CIP capture corpus, per band, held-out
    #: scores from folds split on session (re-measured 2026-09-25 by
    #: `bench.vcip_controls.per_band`):
    #:
    #:     band      n   genuine  swapped   AUC    caught at zero false alarms
    #:     reject   240       14      226   0.803   26.5%
    #:     medium   244      140      104   0.947   85.6%
    #:     low        8        2        6   1.000  100.0%
    #:     high      14       14        0     —       —
    #:
    #: **The argument for this default is the middle two columns, not the
    #: last two.** A floor above `reject` discards 226 of 336 swapped frames
    #: (67.3%) against 14 of 170 genuine — two thirds of the attacks thrown
    #: away before the detector runs, and almost none of the honest traffic.
    #: That is a security hole, not caution, and it holds whatever the
    #: per-band AUC turns out to be.
    #:
    #: **This table is not the one that stood here until 2026-09-25**, which
    #: read 0.992/94.2% for `reject` and 0.957/29.8% for `medium` — the two
    #: bands' figures very nearly the other way round. Those were taken
    #: before `training.fit_vcip.select_face` was corrected to the
    #: pipeline's largest-face rule. The composition columns are identical
    #: in both, which is why the default did not change.
    #:
    #: Reduced reliability at low quality is real and is expressed where it
    #: belongs: the per-band calibration curve (`dfd.calibration`), which
    #: exists for exactly this. The re-measured table now SHOWS that
    #: reliability drop directly — `reject` is the worst band, not the best.
    #:
    #: READ THE DENOMINATORS. `reject` holds 14 genuine frames and `low`
    #: holds 2, so their false-alarm columns rest on 14 and 2 negatives
    #: respectively; `low`'s 1.000 is eight frames and means nothing. `high`
    #: holds no swapped frame at all, so it has no AUC to report.
    min_quality_band: Literal["reject", "low", "medium", "high"] = "reject"

    def score(self, obs: Sequence[Observation]) -> RawScore:
        """Score observations using the NPR feature.

        Filters observations by quality floor, computes NPR features,
        runs them through the model, and returns a raw score or abstention.

        Args:
            obs: sequence of observations to score

        Returns:
            RawScore with score (or None if abstained) and reason, plus
            artifacts containing per-observation scores, max, and count

        Raises:
            ValueError: if model output has unexpected shape (not 2 classes)
            RuntimeError: if torch.load fails with weights_only=True and
                allow_unsafe_load is False (instructs user to convert to state_dict)
            Other exceptions from model inference propagate
        """
        # Handle empty observation list
        if not obs:
            logger.debug("score called with empty observation sequence; abstaining")
            return abstain(self.name, self.version, NO_OBSERVATIONS)

        weights_path = Path(self.weights_path)

        # Check if weights file exists
        if not weights_path.exists():
            logger.debug(
                "weights file not found: %s; abstaining with %s",
                weights_path,
                WEIGHTS_ABSENT,
            )
            return abstain(self.name, self.version, WEIGHTS_ABSENT)

        # Filter observations by quality floor (shared with every other
        # detector via detectors.base.filter_by_quality_floor).
        usable, reason = filter_by_quality_floor(obs, self.min_quality_band)
        if reason is not None:
            return abstain(self.name, self.version, reason)

        # Load model (or retrieve from cache) via the shared secure loader.
        try:
            model = load_model(weights_path, self.model_factory, self.allow_unsafe_load)
        except Exception as e:
            logger.error(
                "failed to load model from %s: %s",
                weights_path,
                type(e).__name__,
                exc_info=True,
            )
            raise

        # Compute NPR features and score
        try:
            maybe_crops = [_crop_to_roi_native(o.payload, o.roi)
                           for o in usable]
            if any(c is None for c in maybe_crops):
                # Abstain rather than fall back to the whole frame: a score
                # computed over background is indistinguishable, to every
                # consumer downstream, from one computed over a face.
                logger.debug("no usable ROI on at least one observation; "
                             "abstaining with %s", NO_ROI)
                return abstain(self.name, self.version, NO_ROI)
            crops = [c for c in maybe_crops if c is not None]

            # One at a time, NOT np.stack: these crops are at their native
            # resolutions and therefore have different shapes. Stacking them
            # raises, and resizing them to match would destroy the very
            # fingerprint this slot reads (see `_crop_to_roi_native`).
            with torch.no_grad():
                logits = torch.cat([
                    model(torch.from_numpy(npr_feature(c))
                          .permute(2, 0, 1)[None].float())
                    for c in crops
                ], dim=0)

            # Validate output shape (must be 2-class binary classifier)
            if logits.shape[1] != EXPECTED_NUM_CLASSES:
                raise ValueError(
                    f"expected {EXPECTED_NUM_CLASSES} output classes, "
                    f"got {logits.shape[1]} from model"
                )

            # Extract fake-class probabilities
            probs = torch.softmax(logits, dim=1)[:, 1]
            per_obs_scores = probs.cpu().numpy().tolist()

            # Aggregate: record all scores for transparency, use max (not mean)
            # to avoid diluting strong single-frame detections
            max_score = float(probs.max())
            mean_score = float(probs.mean())

            logger.debug(
                "computed scores (max=%f, mean=%f) from %d observations",
                max_score,
                mean_score,
                len(usable),
            )

            # Return the mean as the official score (per brief), but record
            # per-observation scores in artifacts for transparency and fusion
            return RawScore(
                detector=self.name,
                version=self.version,
                score=mean_score,
                abstained=False,
                reason=OK,
                artifacts={
                    "n_observations": len(usable),
                    "per_observation_scores": per_obs_scores,
                    "max_score": max_score,
                },
            )
        except Exception as e:
            logger.error(
                "failed to score observations: %s",
                type(e).__name__,
                exc_info=True,
            )
            raise
