"""The composition root: one file in, one audit record out (spec §5.1, §7.2).

Every other module in this package is a stage. This is the only place they are
wired together, and the only non-test caller of ingest, faces, quality,
calibration, fusion and audit.

What this path does TODAY is abstain, three times over: the YuNet weights are
absent, the detector weights are absent, and nothing has fitted a calibration
curve. Each cause is recorded separately — `stage_reasons` for the face stage,
the per-detector `reason` for the rest — because "insufficient evidence"
without a cause is not an audit trail.
"""
from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .audit import AuditRecord, build_audit_record
from .calibration import Calibrator
from .detectors.base import Registry, abstain
from .errors import InvalidInput
from .faces import DEFAULT_MODEL, FaceBox, detect_faces
from .fusion import fuse
from .ingest.image import load_image
from .ingest.video import DEFAULT_MAX_FRAMES, load_video
from .limits import DEFAULT_LIMITS, Limits
from .policy import DEFAULT_POLICY, Policy
from .quality import measure_quality
from .types import QUALITY_BANDS, Context, Evidence, Observation, Sample

logger = logging.getLogger(__name__)

#: The face stage's seam, kept a plain two-argument callable: `detect_faces` is
#: overloaded on a Literal keyword, and a Protocol reproducing those overloads
#: is brittle under mypy --strict for no gain. `_detect_with_reason` adapts it.
FaceDetectFn = Callable[[npt.NDArray[np.uint8], "str | Path"],
                        "tuple[list[FaceBox], str]"]

#: No observation carried a measured quality, so no band can be named. Not a
#: member of QUALITY_BANDS: it is the absence of a band, not a bad one.
UNMEASURED = "unmeasured"
#: The detector ran and found nothing. Distinct from `weights_absent`, which
#: means it never ran at all.
NO_FACE = "no_face"
#: A box that does not intersect the frame. Reported rather than measured,
#: because a clamped empty crop would make OpenCV raise.
DEGENERATE_BOX = "degenerate_box"
#: A detector raised instead of returning a RawScore. The detector is a plugin
#: and its failure must not become the system's failure (spec principles 5 and
#: 8), so it is recorded as that detector's abstention.
DETECTOR_ERROR = "detector_error"
NO_OBSERVATIONS = "no_observations"
MIXED = "mixed"
OK = "ok"


def _detect_with_reason(frame: npt.NDArray[np.uint8],
                        model_path: str | Path) -> tuple[list[FaceBox], str]:
    """Adapt `detect_faces` to the `FaceDetectFn` seam."""
    return detect_faces(frame, model_path, with_reason=True)


def _clamp_roi(shape: tuple[int, ...], box: FaceBox) -> tuple[int, int, int, int] | None:
    """Clamp a detection to the frame, or None if it does not intersect it.

    `measure_quality` slices `frame[y:y + h, x:x + w]` with no clamping, so a
    negative origin would silently slice from the far end of the array and an
    empty crop makes `cv2.cvtColor` raise. `faces.align` clamps for the same
    reason; this is that rule applied one stage earlier.
    """
    height, width = shape[:2]
    x0, y0 = max(0, box.x), max(0, box.y)
    x1, y1 = min(width, box.x + box.w), min(height, box.y + box.h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def _worst_band(observations: Sequence[Observation]) -> str:
    """The worst measured band across observations, or UNMEASURED.

    Worst by position in `types.QUALITY_BANDS` (`reject < low < medium < high`).
    Calibration conditions on the regime that held for the whole sample: the
    mean of `high` and `reject` is a band the sample never occupied, and the
    best band would calibrate a blurry sample as though it were sharp.
    """
    bands = [o.quality.band for o in observations if o.quality is not None]
    if not bands:
        return UNMEASURED
    return min(bands, key=QUALITY_BANDS.index)


def _aggregate(reasons: list[str]) -> str:
    """One sample-level reason from per-frame reasons."""
    if not reasons:
        return NO_OBSERVATIONS
    if OK in reasons:
        return OK
    unique = set(reasons)
    return reasons[0] if len(unique) == 1 else MIXED


def normalize(sample: Sample, *, detect: FaceDetectFn = _detect_with_reason,
              face_model: str | Path = DEFAULT_MODEL) -> tuple[Sample, dict[str, str]]:
    """Attach a face ROI and a quality measurement to every observation.

    Ingest deliberately emits `roi=None, quality=None` — decoding and analysis
    are separate stages. This is the analysis stage, and until it existed the
    only code performing it was the benchmark runner, which fabricated
    landmarks at 35% and 65% of frame width rather than detecting them.

    Args:
        sample: an ingested sample whose observations carry no quality.
        detect: face detection seam; must return (boxes, reason).
        face_model: path passed to `detect`.

    Returns:
        A pair of (sample with normalized observations, stage reasons). The
        reasons carry `faces` (`ok`, `no_face`, `weights_absent`,
        `degenerate_box`, `mixed` or `no_observations`), `frames_with_face`
        as "n/total", and `max_faces_in_frame`.

    Raises:
        No exceptions of its own; `detect` may raise (a corrupt model file
        makes OpenCV raise, which is deliberately not caught — treating
        corruption as absence would hide a deployment failure).
    """
    out: list[Observation] = []
    reasons: list[str] = []
    counts: list[int] = []

    for obs in sample.observations:
        boxes, reason = detect(obs.payload, face_model)
        counts.append(len(boxes))
        if not boxes:
            reasons.append(NO_FACE if reason == OK else reason)
            out.append(obs)
            continue
        box = max(boxes, key=lambda b: b.w * b.h)
        roi = _clamp_roi(obs.payload.shape, box)
        if roi is None:
            logger.warning("face box %s does not intersect the frame; not measured",
                           (box.x, box.y, box.w, box.h))
            reasons.append(DEGENERATE_BOX)
            out.append(obs)
            continue
        quality = measure_quality(obs.payload, roi, box.landmarks[:2])
        out.append(Observation(t=obs.t, payload=obs.payload, roi=roi,
                               quality=quality, source_id=obs.source_id))
        reasons.append(OK)

    with_face = sum(1 for r in reasons if r == OK)
    stage_reasons = {
        "faces": _aggregate(reasons),
        "frames_with_face": f"{with_face}/{len(sample.observations)}",
        "max_faces_in_frame": str(max(counts) if counts else 0),
    }
    logger.debug("normalized %d observations: %s", len(out), stage_reasons)
    return (Sample(sample_id=sample.sample_id, modality=sample.modality,
                   observations=tuple(out), context=sample.context),
            stage_reasons)


#: Extensions routed to each ingest adapter. An unlisted extension is refused
#: rather than guessed: `load_image` on a video returns the first frame with no
#: indication that the rest of the file was ignored.
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})


def _sha256(path: Path) -> str:
    """Hash the file in chunks, before anything decodes it.

    Raises:
        InvalidInput: if the file cannot be read. Without this translation a
            missing path raises OSError, which the CLI would report as an
            unexpected failure rather than as bad input.
    """
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise InvalidInput(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def _ingest(path: Path, context: Context, limits: Limits, max_frames: int,
            seed: int) -> Sample:
    """Route to an ingest adapter by extension.

    Raises:
        InvalidInput: if the extension is not one this package ingests, or if
            the adapter reports the file is undecodable.
        ResourceLimitExceeded: propagated from the adapters' header-first
            checks, deliberately untouched.
    """
    suffix = path.suffix.lower()
    if suffix not in IMAGE_SUFFIXES and suffix not in VIDEO_SUFFIXES:
        raise InvalidInput(
            f"unsupported file extension {suffix!r} for {path.name}; "
            f"images: {sorted(IMAGE_SUFFIXES)}, videos: {sorted(VIDEO_SUFFIXES)}")
    try:
        if suffix in IMAGE_SUFFIXES:
            return load_image(path, context, limits)
        # Keyword arguments, not positional: `max_frames` and `seed` are both
        # int, so a swap type-checks cleanly under mypy --strict and would
        # silently score every video from the wrong frame indices.
        return load_video(path, context, max_frames=max_frames, seed=seed,
                          limits=limits)
    except ValueError as exc:
        # Both adapters document a bare ValueError for an undecodable file or a
        # zero-frame video — one of errors.py's 19 un-migrated raise sites,
        # translated here at the boundary that needs it. Scoped to the adapter
        # call alone, NOT wrapped around the rest of the pipeline: a blanket
        # except ValueError would relabel genuine bugs as bad input.
        raise InvalidInput(f"could not decode {path.name}: {exc}") from exc


def decide(
    path: str | Path,
    *,
    registry: Registry,
    calibrators: Mapping[str, Calibrator] | None = None,
    policy: Policy = DEFAULT_POLICY,
    context: Context | None = None,
    limits: Limits = DEFAULT_LIMITS,
    detect: FaceDetectFn = _detect_with_reason,
    face_model: str | Path = DEFAULT_MODEL,
    max_frames: int = DEFAULT_MAX_FRAMES,
    seed: int = 0,
    created_at: str | None = None,
) -> AuditRecord:
    """Score one file and return its immutable audit record.

    The stages: hash, ingest (limits enforced from the header, before decode),
    normalize (faces and quality), score every registered detector, calibrate
    each raw score on the sample's worst measured band, fuse under `policy`,
    and record — with the same `policy` object, so the record's threshold is
    the one applied rather than a copy of it.

    `n_frames=1` is passed to `fuse` because every detector in this repo
    aggregates internally; passing the frame count would apply the ESS
    discount to already-aggregated evidence, which `fuse` documents as misuse.

    `ood_score` carries `FusedResult.disagreement`. P0 has no Mahalanobis or
    energy OOD head, and disagreement is the only OOD-shaped quantity that
    exists; the field name overstates what it holds. Revisit when P1 lands the
    real head.

    Args:
        path: file to score.
        registry: detectors to consult.
        calibrators: fitted calibrators by detector name. A detector with no
            entry calibrates through an unfitted `Calibrator`, which returns
            llr 0.0 and `uncalibrated_for_band` — the honest answer to "I was
            never calibrated in this regime".
        policy: thresholds to apply and to record.
        context: sample metadata; defaults to an empty `Context`.
        limits: decode limits, enforced before allocation.
        detect: face detection seam.
        face_model: path passed to `detect`.
        max_frames: frame cap for video ingest.
        seed: frame-selection seed for video ingest.
        created_at: ISO-8601 timestamp; injectable so two records describing
            the same decision are genuinely identical.

    Returns:
        An `AuditRecord`. A refusal produces no record: a refused input is not
        a decision.

    Raises:
        InvalidInput: unreadable file, unsupported extension, undecodable file,
            or a value `build_audit_record` refuses to record.
        ResourceLimitExceeded: the input exceeds a decode limit.
        cv2.error: from `detect_faces` when the face model file is present but
            corrupt. `faces.py` documents that deliberately — treating
            corruption as absence would hide a deployment failure — so it is
            not caught here either. Note the asymmetry with a detector's
            failure, which IS caught below: the face stage is one fixed,
            first-party model on which every later stage depends, while a
            detector is a replaceable plugin whose loss costs one slot.

        A detector that raises does NOT propagate: it is recorded as that
        detector's abstention with reason `detector_error` and the decision
        continues on the remaining detectors. Everything outside the
        per-detector call — ingest, `normalize`, `fuse`, calibration and
        `build_audit_record` — is first-party code whose exceptions are bugs
        and still propagate.
    """
    file_path = Path(path)
    input_sha256 = _sha256(file_path)
    sample = _ingest(file_path, context or Context(), limits, max_frames, seed)
    sample, stage_reasons = normalize(sample, detect=detect, face_model=face_model)
    band = _worst_band(sample.observations)

    fitted = dict(calibrators or {})
    evidence: list[Evidence] = []
    model_versions: dict[str, str] = {}
    for name in registry.names():
        detector = registry.get(name)
        started = time.perf_counter()
        try:
            raw = detector.score(sample.observations)
        except Exception:
            # A BROAD catch is correct HERE and nowhere else in this file.
            # A detector is a plugin: spec principle 5 makes detectors
            # perishable and hot-swappable, and principle 8 requires the
            # system to remain useful with every ML slot defeated. One corrupt
            # weights file must therefore cost one slot's evidence, not the
            # whole decision -- without this, a raising detector means no
            # audit record at all, no evidence from the healthy detectors, and
            # a traceback with exit 1. The failure surface is third-party
            # model code (torch, the weights file, the model's own forward
            # pass) and is not enumerable, which is exactly the condition a
            # narrow catch cannot meet. Contrast `_ingest`'s deliberate
            # `except ValueError`, scoped to the adapter call alone: there the
            # set of failures meaning "bad input" IS known and small, so
            # anything wider would relabel genuine bugs as bad input.
            # `Exception`, never a bare `except:` (gate-forbidden, and it
            # would swallow more): KeyboardInterrupt and SystemExit derive
            # from BaseException, not Exception, so Ctrl-C and sys.exit still
            # stop the process as the operator asked. The traceback is not
            # lost -- logger.exception records it at ERROR with exc_info.
            logger.exception(
                "detector %s raised while scoring; recording %s", name, DETECTOR_ERROR)
            raw = abstain(name, detector.version, DETECTOR_ERROR)
        logger.debug("detector %s scored in %.1f ms", name,
                     (time.perf_counter() - started) * 1000.0)
        model_versions[name] = detector.version
        calibrator = fitted.get(name) or Calibrator(name)
        evidence.append(calibrator.to_evidence(raw, band))

    fused = fuse(evidence, n_frames=1, policy=policy)
    logger.info("decided %s: verdict=%s band=%s contributing=%d",
                sample.sample_id, fused.verdict.value, band, fused.n_contributing)
    return build_audit_record(
        sample_id=sample.sample_id,
        input_sha256=input_sha256,
        verdict=fused.verdict,
        llr_total=fused.llr_total,
        posterior=fused.posterior,
        evidence=evidence,
        quality_band=band,
        ood_score=fused.disagreement,
        policy_version=policy.version,
        threshold=policy.fake_threshold,
        model_versions=model_versions,
        stage_reasons=stage_reasons,
        created_at=created_at,
    )
