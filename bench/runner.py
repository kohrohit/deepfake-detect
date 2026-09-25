"""Benchmark orchestration (spec §8, acceptance criteria 1, 5, 10).

Guards are enforced by default and can only be waived explicitly, because a
benchmark that is easy to run dirty will be run dirty — and every hygiene
failure flatters the result.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from dfd.quality import measure_quality
from dfd.types import Observation

from .guards import (
    GuardViolation,
    IdentityReport,
    ParityReport,
    check_compression_coverage,
    check_demographic_parity,
    check_identity_disjoint,
    check_subject_partition,
    check_threshold_provenance,
    check_uniform_preprocessing,
    check_video_level,
)
from .metrics import auc, bootstrap_ci_by_group, ece, tpr_at_fpr
from .protocol import UnsplittableCorpusError, logo_splits
from .robustness import robustness_sweep

logger = logging.getLogger(__name__)

#: The class name genuine records are counted under in
#: `DetectorResult.abstention_by_class`. Fakes are counted under their
#: generator id. A literal rather than None because this key reaches the
#: report JSON, where `null` reads as a label somebody forgot to set.
REAL_CLASS = "real"


@dataclass(frozen=True)
class RunConfig:
    seed: int = 0
    enforce_guards: bool = True
    fpr_targets: tuple[float, ...] = (0.01, 0.001)
    bootstrap_n: int = 200
    threshold_source: str = "validation"
    #: Spec §8.3 / acceptance criterion 9. Off by default because the sweep
    #: re-scores every record once per perturbation variant and is not free.
    robustness: bool = False

    #: sample_id -> identity embedding, for acceptance criterion 2. Supplied
    #: by the caller rather than computed here: embedding means running a
    #: face detector and a recogniser over every record, which is the
    #: corpus loader's job (it already holds the pixels and the detector) and
    #: not the orchestrator's. `dfd.embed.Embedder.embed` produces these.
    #: None means the criterion was NOT measured, which `RunRecord`
    #: distinguishes from "measured and clean" — the distinction the whole
    #: guard exists for.
    identity_embeddings: dict[str, np.ndarray] | None = None
    #: Cosine at or above which two crops are the same person. The default
    #: is `dfd.embed.DEFAULT_THRESHOLD`, measured on this project's faces.
    identity_threshold: float = 0.363
    #: Fraction of train x test pairs allowed to cross before it is called
    #: leakage. See `bench.guards.check_identity_disjoint`: unrelated faces
    #: cross at ~0.21% with this embedder, so a large split with 0.0 here
    #: fails for arithmetic rather than leakage.
    identity_max_false_match_rate: float = 0.0
    #: Cap on subjects compared by the corpus-level check. The comparison is
    #: quadratic in subjects, so a large corpus is sampled and the report
    #: says how many were compared.
    identity_max_subjects: int = 500
    #: Acceptance criterion 8. Off by default: the attack is a gradient loop
    #: per positive sample and costs far more than scoring does. Turning it
    #: on for a detector that exposes no differentiable target records why
    #: rather than silently measuring nothing.
    adversarial: bool = False
    #: L-inf budget for the attack, in [0, 1] pixel units.
    adversarial_eps: float = 0.03
    #: Acceptance criterion 11. Strata come from the records' `stratum`
    #: field; a corpus without one is not measured for parity, and says so.
    parity_max_fpr_ratio: float = 2.0
    #: The FPR at which per-stratum rates are compared. An aggregate FPR of
    #: 1% is the operating point the rest of this benchmark reports at, so
    #: parity is read at the same point rather than at a different one.
    parity_at_fpr: float = 0.01
    #: Genuine samples a stratum needs before its FPR is compared at all.
    #:
    #: By the rule of three, observing ZERO false positives in n genuine
    #: samples puts the 95% upper bound on that stratum's true FPR at about
    #: 3/n. At n=6 that bound is 50%, so a stratum can read 0.00 while its
    #: true rate is anything at all — and `check_demographic_parity`
    #: correctly calls a zero-FPR stratum an infinite ratio and raises. The
    #: guard is right; comparing it at n=6 is not. 150 puts the bound at 2%,
    #: the first point at which "this stratum's FPR is near the 1% operating
    #: point" is a claim the data can carry.
    #:
    #: Strata below this are EXCLUDED and counted in
    #: `RunRecord.parity_excluded_strata`, never silently merged: merging
    #: them into a neighbouring stratum is how a disparity gets averaged
    #: away.
    parity_min_genuine_per_stratum: int = 150


@dataclass(frozen=True)
class DetectorResult:
    detector: str
    auc: float
    auc_ci: tuple[float, float]
    tpr_at_1pct: float
    tpr_at_0p1pct: float
    ece: float
    adversarial_tpr_at_1pct: float | None
    abstention_rate: float
    p95_latency_ms: float
    n_samples: int
    #: Spec §8.3 / acceptance criterion 9: TPR@FPR=1% per perturbation
    #: variant emitted by `robustness_sweep` (clean, the JPEG quality curve,
    #: and the other perturbations including the two physical recapture
    #: paths). Empty when `RunConfig.robustness` is False, or on a LOGO
    #: fold — the sweep is never re-run per fold (see `_logo_results`), so
    #: an empty dict here means "not measured", never "measured as zero".
    tpr_by_perturbation: dict[str, float] = field(default_factory=dict)
    #: class -> (abstained, total), where class is "real" for genuine records
    #: and the generator id for fakes. Added 2026-09-24 because
    #: `abstention_rate` above is one number over the whole corpus and every
    #: metric beside it is computed over the records that did NOT abstain.
    #: When abstention is correlated with the label, those metrics are
    #: measured on a label-selected subsample and overstate what a reader
    #: takes them to mean. Measured on the swap corpus: `blend_seam` abstains
    #: on 74.3% of `swap_lowres_paste` fakes against 56.9% of reals, and
    #: `swap_lowres_paste` is the fold carrying the best AUC in the report.
    #: One aggregate rate cannot show that, and the skew is not a detail: it
    #: is the reason the fold looks strong.
    abstention_by_class: dict[str, tuple[int, int]] = field(default_factory=dict)


@dataclass(frozen=True)
class RunRecord:
    seed: int
    dataset_hash: str
    guards_enforced: bool
    model_versions: dict[str, str]
    identity_report: IdentityReport | None
    #: Whole-corpus metrics. THIS IS IN-DATASET PERFORMANCE, which measures
    #: memorisation, not field performance. Never report it as the headline.
    detector_results: dict[str, DetectorResult] = field(default_factory=dict)
    #: held-out generator -> detector -> metrics. Spec 8.1: the only number
    #: that predicts field performance. Empty when the corpus cannot be split.
    logo_results: dict[str, dict[str, DetectorResult]] = field(
        default_factory=dict)
    #: held-out generator -> count of fakes dropped by logo_splits for
    #: identity disjointness (Split.dropped_for_identity). A fold's AUC is
    #: computed over test_ids() only; without this a reader cannot tell a
    #: fold with 2 drops from one with 20 — both would print the same
    #: n_samples and AUC. Empty when logo_results is empty.
    logo_dropped: dict[str, int] = field(default_factory=dict)
    #: held-out generator -> the identity check for that fold. `identity_report`
    #: above is the WORST of these by crossing rate, so a reader who looks at
    #: only one number sees the fold most likely to be leaking rather than an
    #: average that hides it.
    identity_by_fold: dict[str, IdentityReport] = field(default_factory=dict)
    #: How criterion 2 was measured. "not_measured" (no embeddings
    #: supplied), "ok_corpus_only" (the declared subjects were checked
    #: against pixels, but the corpus carries one generator so there are no
    #: LOGO folds to certify), "ok" (both), or "violation" (measured and
    #: failed, with guards waived). An unmeasured criterion and a clean one
    #: must never read alike.
    identity_status: str = "not_measured"
    #: detector -> per-stratum error parity at `RunConfig.parity_at_fpr`.
    #: Acceptance criterion 11.
    parity_by_detector: dict[str, ParityReport] = field(default_factory=dict)
    #: Why `parity_by_detector` is empty, when it is: "no_strata" (the corpus
    #: carries no `stratum` field), "too_few_per_stratum" (fewer than two
    #: strata carry enough genuine samples to compare — see
    #: `RunConfig.parity_min_genuine_per_stratum`), "not_measurable" (no
    #: detector produced two labels' worth of finite scores), or "ok".
    parity_status: str = "no_strata"
    #: detector -> why its adversarial number is absent, when it is.
    #: "not_requested" (RunConfig.adversarial is False), "no_target" (the
    #: detector exposes no differentiable module to attack — every
    #: handcrafted-feature detector in this repo, by construction), or
    #: "weights_absent" (it has a target and no weights loaded into it). An
    #: adversarial TPR of None must never be readable as "the attack failed".
    adversarial_status: dict[str, str] = field(default_factory=dict)
    #: stratum -> genuine sample count, for each stratum excluded as too
    #: small to compare. Present even on an "ok" run: a parity result over
    #: three of eight strata is a different claim from one over all eight,
    #: and the excluded list is the only thing that says which it is.
    parity_excluded_strata: dict[str, int] = field(default_factory=dict)


def dataset_hash(records: list[dict]) -> str:
    """Content hash over identifying fields — stable, order-independent.

    Includes every field `check_uniform_preprocessing` can fail the run
    over (`face_detector`, `align`), not only the fields `logo_splits` and
    subject/label identity depend on. Two runs with different preprocessing
    aligned to the same sample/subject/source/generator/label/compression
    tuple would otherwise hash identically while producing different
    metrics, breaking the audit tie acceptance criterion 10 exists for.
    """
    keys = sorted(
        json.dumps({k: r.get(k) for k in
                    ("sample_id", "subject_id", "source_id", "generator",
                     "label", "compression", "face_detector", "align")},
                   sort_keys=True)
        for r in records)
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def _observation(record: dict) -> Observation:
    img = record["image"]
    h, w = img.shape[:2]
    lm = np.array([[w * 0.35, h * 0.4], [w * 0.65, h * 0.4]])
    q = measure_quality(img, (0, 0, w, h), lm)
    return Observation(t=0.0, payload=img, roi=(0, 0, w, h), quality=q,
                       source_id=record["source_id"])


def run_benchmark(records: list[dict], registry, config: RunConfig) -> RunRecord:
    if config.enforce_guards:
        # `groups` MUST identify the SOURCE VIDEO, never the sample id.
        # Passing sample_ids for both arguments makes this guard vacuous: its
        # only failure condition is groups[i] != sample_ids[i], so identical
        # lists can never raise. Records must carry a source-video field; for
        # image records each image is its own source, which must be recorded
        # explicitly rather than aliased to sample_id.
        check_video_level([r["sample_id"] for r in records],
                          [r["source_id"] for r in records])
        check_compression_coverage(records)
        check_uniform_preprocessing(records)
        check_threshold_provenance(config.threshold_source)

    labels = np.array([r["label"] for r in records], dtype=int)
    # The SOURCE video, not the sample id. bootstrap_ci_by_group resamples
    # over these, and resampling over frames rather than videos fabricates
    # precision: measured at 11.9x too narrow (group CI 0.751 vs row 0.063).
    # check_video_level enforces a 1:1 mapping while guards are on, but
    # enforce_guards=False is a supported path and that is exactly where an
    # honest interval matters most.
    groups = np.array([r["source_id"] for r in records])
    # "real" rather than the generator id for genuine records: `generator` is
    # None on those, and a None key serialises to `null` in the report JSON
    # where it reads as a missing label rather than as the genuine half.
    classes = np.array([r["generator"] or REAL_CLASS for r in records])
    observations = [_observation(r) for r in records]

    results: dict[str, DetectorResult] = {}
    versions: dict[str, str] = {}
    scores_by_detector: dict[str, np.ndarray] = {}

    for name in registry.names():
        det = registry.get(name)
        versions[name] = det.version

        scores: list[float] = []
        abstentions = 0
        latencies: list[float] = []

        for obs in observations:
            t0 = time.perf_counter()
            raw = det.score([obs])
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if raw.abstained or raw.score is None:
                abstentions += 1
                scores.append(np.nan)
            else:
                scores.append(float(raw.score))

        s = np.array(scores, dtype=float)
        scores_by_detector[name] = s

        tpr_by_perturbation: dict[str, float] = {}
        if config.robustness:
            variants: dict[str, list[float]] = {}
            for rec_in, obs in zip(records, observations, strict=True):
                for pname, pimg in robustness_sweep(rec_in["image"]).items():
                    # Deliberately reuses the CLEAN observation's `quality`
                    # rather than re-measuring it on the perturbed pixels.
                    # Measured across the full sweep at three fixture sizes
                    # (128x160, 240x320, 360x480): every perturbed variant
                    # bands identically to clean, so this is behaviourally
                    # equivalent today, not just convenient. That is a fact
                    # about `measure_quality`'s current thresholds, not a
                    # law — blur halves high-frequency energy and both
                    # recapture paths destroy roughly two thirds of it, yet
                    # none of that moves the band. If quality banding is
                    # ever made sensitive to these perturbations, this reuse
                    # must be revisited or a detector's quality floor will
                    # never trigger on a laundered image.
                    pobs = Observation(t=obs.t, payload=pimg, roi=obs.roi,
                                       quality=obs.quality,
                                       source_id=obs.source_id)
                    praw = det.score([pobs])
                    variants.setdefault(pname, []).append(
                        float(praw.score) if not praw.abstained
                        and praw.score is not None else np.nan)
            for pname, pscores in variants.items():
                ps = np.array(pscores, dtype=float)
                pv = np.isfinite(ps)
                tpr_by_perturbation[pname] = (
                    tpr_at_fpr(ps[pv], labels[pv], 0.01)
                    if pv.sum() and len(np.unique(labels[pv])) > 1
                    else float("nan"))

        results[name] = _detector_result(
            name, s, labels, groups, latencies, abstentions, config,
            classes=classes, tpr_by_perturbation=tpr_by_perturbation)

    adversarial_tprs, adversarial_status = _adversarial_results(
        registry, observations, labels, config)
    for name, value in adversarial_tprs.items():
        results[name] = dataclasses.replace(results[name],
                                            adversarial_tpr_at_1pct=value)

    splits = _logo_splits_or_none(records, config)
    logo_results, logo_dropped = _logo_results(
        records, registry, scores_by_detector, labels, groups, classes,
        config, splits)

    # Criteria 2 and 11. Both raise through their guards when they fail and
    # guards are enforced; with guards waived the failure is still measured
    # and recorded, because a waived guard must leave evidence behind rather
    # than a blank.
    try:
        identity_report, identity_by_fold, identity_status = _identity_reports(
            records, splits, config)
    except GuardViolation:
        if config.enforce_guards:
            raise
        # Recompute tolerating everything, purely to RECOVER THE NUMBERS. A
        # waived guard that leaves behind only the word "violation" tells a
        # reader that something failed and not how badly, which is the one
        # thing they need in order to decide whether to care.
        logger.warning("identity guard failed with guards waived; recording it")
        identity_report, identity_by_fold, _ = _identity_reports(
            records, splits,
            dataclasses.replace(config, identity_max_false_match_rate=1.0))
        identity_status = "violation"
    try:
        parity_by_detector, parity_status, parity_excluded = _parity_reports(
            records, scores_by_detector, labels, config)
    except GuardViolation:
        if config.enforce_guards:
            raise
        logger.warning("parity guard failed with guards waived; recording it")
        parity_by_detector, _, parity_excluded = _parity_reports(
            records, scores_by_detector, labels,
            dataclasses.replace(config, parity_max_fpr_ratio=float("inf")))
        parity_status = "violation"

    return RunRecord(seed=config.seed, dataset_hash=dataset_hash(records),
                     guards_enforced=config.enforce_guards,
                     model_versions=versions, identity_report=identity_report,
                     detector_results=results, logo_results=logo_results,
                     logo_dropped=logo_dropped,
                     identity_by_fold=identity_by_fold,
                     identity_status=identity_status,
                     parity_by_detector=parity_by_detector,
                     parity_status=parity_status,
                     parity_excluded_strata=parity_excluded,
                     adversarial_status=adversarial_status)


def _detector_result(name, s, labels, groups, latencies, abstentions,
                     config, classes=None,
                     tpr_by_perturbation=None) -> DetectorResult:
    """Metrics for one detector over one set of rows.

    Split out so a LOGO fold can reuse it verbatim: the fold differs only in
    which rows it passes, never in how the numbers are computed.

    `tpr_by_perturbation` defaults to empty: a LOGO fold calls this without
    it, because the robustness sweep is never re-run per fold (see
    `_logo_results`) and an empty dict there means "not measured", the same
    posture the fold already takes for `p95_latency_ms` (reported as NaN
    rather than a false zero).

    `classes` is the per-row class ("real", or the generator id) used for
    `DetectorResult.abstention_by_class`. It defaults to None so an existing
    caller keeps working, and None yields an EMPTY breakdown rather than a
    fabricated one — "not measured", never "nothing abstained".
    """
    n = len(s)
    valid = np.isfinite(s)
    base = {
        "detector": name,
        "adversarial_tpr_at_1pct": None,
        "abstention_rate": abstentions / max(1, n),
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "n_samples": n,
        "tpr_by_perturbation": dict(tpr_by_perturbation or {}),
        "abstention_by_class": _abstention_by_class(s, classes),
    }
    if valid.sum() == 0 or len(np.unique(labels[valid])) < 2:
        nan = float("nan")
        return DetectorResult(auc=nan, auc_ci=(nan, nan), tpr_at_1pct=nan,
                              tpr_at_0p1pct=nan, ece=nan, **base)
    return DetectorResult(
        auc=auc(s[valid], labels[valid]),
        auc_ci=bootstrap_ci_by_group(s[valid], labels[valid], groups[valid],
                                     auc, n=config.bootstrap_n,
                                     seed=config.seed),
        tpr_at_1pct=tpr_at_fpr(s[valid], labels[valid], 0.01),
        tpr_at_0p1pct=tpr_at_fpr(s[valid], labels[valid], 0.001),
        ece=ece(s[valid], labels[valid]),
        **base,
    )


def _abstention_by_class(s, classes) -> dict[str, tuple[int, int]]:
    """(abstained, total) per class, over the rows given.

    Counts EVERY row, so the totals sum to the corpus: a class missing from
    this mapping reads as "that class did not abstain", which is the failure
    this breakdown exists to prevent.
    """
    if classes is None:
        return {}
    out: dict[str, tuple[int, int]] = {}
    finite = np.isfinite(s)
    for cls in sorted({str(c) for c in classes}):
        rows = np.asarray(classes) == cls
        out[cls] = (int((~finite[rows]).sum()), int(rows.sum()))
    return out


#: A detector opts into acceptance criterion 8 by exposing this method. It
#: returns a differentiable `torch.nn.Module` mapping a (B, C, H, W) batch in
#: [0, 1] to two logits, and a callable turning an `Observation` into one row
#: of that batch — or None when the detector has no weights loaded. Declared
#: as a name rather than a Protocol because every detector in this repo today
#: is handcrafted features plus a linear model with no differentiable path
#: from pixels, and a Protocol nothing implements reads as an interface
#: somebody forgot to fill in rather than one that does not apply.
ADVERSARIAL_TARGET = "adversarial_target"


def _adversarial_results(registry, observations, labels, config):
    """Acceptance criterion 8 per detector — or a stated reason.

    Returns (tpr_by_detector, status_by_detector). A None TPR always comes
    with a reason: "the attack was not run" and "the attack succeeded
    completely" are opposite readings of the same missing number.
    """
    tprs: dict[str, float] = {}
    status: dict[str, str] = {}
    if not config.adversarial:
        return tprs, dict.fromkeys(registry.names(), "not_requested")

    import torch

    from .adversarial import adversarial_tpr

    for name in registry.names():
        det = registry.get(name)
        target = getattr(det, ADVERSARIAL_TARGET, None)
        if target is None:
            status[name] = "no_target"
            continue
        built = target()
        if built is None:
            status[name] = "weights_absent"
            continue
        module, to_row = built
        rows = [to_row(obs) for obs in observations]
        x = torch.stack([r for r in rows if r is not None])
        keep = np.array([r is not None for r in rows], dtype=bool)
        y = torch.as_tensor(labels[keep], dtype=torch.int64)
        if len(np.unique(labels[keep])) < 2:
            status[name] = "not_measurable"
            continue
        tprs[name] = float(adversarial_tpr(module, x, y,
                                           eps=config.adversarial_eps,
                                           fpr=0.01, seed=config.seed))
        status[name] = "ok"
    return tprs, status


def _logo_splits_or_none(records, config):
    """The LOGO folds, or None when this corpus cannot be split.

    Split out of `_logo_results` because acceptance criteria 1 and 2 both
    need the SAME folds: the identity guard certifies the very partition the
    metrics are computed over, and computing them from two separate calls
    would let a seed or a validation change make the certificate describe a
    different split from the one reported.

    A malformed corpus (bad label, straddling source, unattributed fake)
    raises plain ValueError from `_validate` and is deliberately NOT caught:
    it must propagate, not degrade.
    """
    try:
        return logo_splits(records, seed=config.seed)
    except UnsplittableCorpusError as exc:
        # A corpus with one generator, one subject, or no measurable fold.
        # Recorded rather than raised: the in-dataset numbers are still valid.
        logger.warning("LOGO unavailable for this corpus: %s", exc)
        return None


def _identity_reports(records, splits, config):
    """Acceptance criterion 2, measured per fold — or a stated reason.

    The folds are already identity-disjoint BY DECLARATION: `logo_splits`
    partitions on `subject_id`. This checks the declaration against pixels,
    which is the whole point — a corpus where one person enrolled twice
    carries two subject ids, and every split built on those ids looks clean
    while leaking a face. `corpora.sbi` and `training.fit_blend` both record
    that exact gap.

    Returns (worst_report, by_fold, status). The worst fold by crossing rate
    is surfaced as the headline so a reader taking one number takes the
    pessimistic one.
    """
    if config.identity_embeddings is None:
        return None, {}, "not_measured"

    # The corpus-level check first, because it needs no split and every
    # corpus this project holds carries ONE generator — so `logo_splits`
    # refuses them all and a fold-only implementation of this criterion can
    # never fire. See `check_subject_partition`.
    corpus = check_subject_partition(
        {r["sample_id"]: r["subject_id"] for r in records},
        config.identity_embeddings,
        threshold=config.identity_threshold,
        max_false_match_rate=config.identity_max_false_match_rate,
        max_subjects=config.identity_max_subjects,
        seed=config.seed,
    )
    if not splits:
        return corpus, {}, "ok_corpus_only"

    by_fold: dict[str, IdentityReport] = {}
    for split in splits:
        by_fold[split.held_out_generator] = check_identity_disjoint(
            [r["sample_id"] for r in split.train],
            list(split.test_ids()),
            config.identity_embeddings,
            threshold=config.identity_threshold,
            max_false_match_rate=config.identity_max_false_match_rate,
        )
    worst = max([corpus, *by_fold.values()],
                key=lambda r: (r.violation_rate, r.max_similarity))
    return worst, by_fold, "ok"


def _parity_reports(records, scores_by_detector, labels, config):
    """Acceptance criterion 11, per detector — or a stated reason.

    Read at `RunConfig.parity_at_fpr`, the same operating point the rest of
    the benchmark reports at: parity measured at a different threshold from
    the one a deployment would use is a number about neither.

    The threshold is taken from the GENUINE scores' quantile rather than
    swept, because that is what fixing an FPR means; with too few finite
    genuine scores to place a quantile the detector is skipped rather than
    compared at an invented threshold.
    """
    raw = [r.get("stratum") for r in records]
    if any(s is None for s in raw):
        return {}, "no_strata", {}
    strata = np.asarray([str(s) for s in raw])

    # Eligibility is decided on the CORPUS, once, not per detector: a
    # stratum that is too small to compare is too small whichever detector
    # scored it, and deciding per detector would let abstentions quietly
    # change which strata a run compares.
    genuine_counts = {s: int(((strata == s) & (labels == 0)).sum())
                      for s in sorted(set(strata.tolist()))}
    eligible = {s for s, n in genuine_counts.items()
                if n >= config.parity_min_genuine_per_stratum}
    excluded = {s: n for s, n in genuine_counts.items() if s not in eligible}
    if len(eligible) < 2:
        return {}, "too_few_per_stratum", excluded

    in_scope = np.isin(strata, list(eligible))
    out: dict[str, ParityReport] = {}
    for name, scores in scores_by_detector.items():
        finite = np.isfinite(scores) & in_scope
        genuine = finite & (labels == 0)
        if genuine.sum() < 2 or len(np.unique(labels[finite])) < 2:
            continue
        # The score above which `parity_at_fpr` of genuine samples fall.
        threshold = float(np.quantile(scores[genuine], 1.0 - config.parity_at_fpr))
        out[name] = check_demographic_parity(
            scores[finite], labels[finite], strata[finite],
            threshold=threshold, max_fpr_ratio=config.parity_max_fpr_ratio)
    return out, ("ok" if out else "not_measurable"), excluded


def _logo_results(
    records, registry, scores_by_detector, labels, groups, classes, config,
    splits,
) -> tuple[dict[str, dict[str, DetectorResult]], dict[str, int]]:
    """Per-held-out-generator metrics — spec 8.1, the number that predicts field
    performance.

    Detection is not re-run per fold: a detector's score for a record does not
    depend on which fold the record lands in, so folds slice the scores already
    computed.

    HONEST SCOPE. A full LOGO protocol trains on the fold's train side and
    tests on the held-out one. P0 detectors are not trained here, so what this
    computes is evaluation on the held-out generator's test rows. That is the
    number you report, and it becomes the full protocol once training or
    calibration fitting exists — at which point the fold's train side is also
    where the operating threshold must be frozen (spec 8.2 guard 5).

    Returns (logo_results, logo_dropped): the second maps held-out generator
    to the number of fakes `logo_splits` dropped for identity disjointness
    in that fold (Split.dropped_for_identity), so a shrunken test set is
    visible rather than indistinguishable from a clean one.
    """
    # A duplicate sample_id collapses in the `position` map below: one row
    # would be scored twice and another dropped from every fold with no
    # error, while `0 < n < len(records)` still holds so nothing downstream
    # notices. sample_id is the join key `logo_splits` and this function both
    # rely on; it must be unique before either is trusted.
    sample_ids = [r["sample_id"] for r in records]
    if len(set(sample_ids)) != len(sample_ids):
        seen: set = set()
        dupes = sorted({s for s in sample_ids
                        if s in seen or seen.add(s)})
        raise ValueError(
            "duplicate sample_id(s) would silently mis-slice LOGO folds: "
            f"{dupes}")

    if splits is None:
        return {}, {}

    position = {r["sample_id"]: i for i, r in enumerate(records)}
    out: dict[str, dict[str, DetectorResult]] = {}
    dropped: dict[str, int] = {}
    for split in splits:
        rows = np.array([position[sid] for sid in split.test_ids()], dtype=int)
        dropped[split.held_out_generator] = len(split.dropped_for_identity)
        fold_results = {}
        for name in registry.names():
            sliced = scores_by_detector[name][rows]
            # Fabricating 0.0 for both would be false: every row in the
            # fold WAS scored (by the main loop above), but which of those
            # scores are abstentions, and how long scoring took, are facts
            # about the fold's slice, not the whole corpus. Abstentions are
            # recovered from the slice itself; latency was never measured
            # per fold, so it is reported as unmeasured (NaN) rather than
            # printed as a false "0.0 ms" that reads as instantaneous.
            fold_abstentions = int((~np.isfinite(sliced)).sum())
            fold_results[name] = _detector_result(
                name, sliced, labels[rows], groups[rows],
                [float("nan")], fold_abstentions, config,
                classes=classes[rows])
        out[split.held_out_generator] = fold_results
    return out, dropped


def worst_logo_auc(record: RunRecord, detector: str) -> float:
    """The weakest held-out generator for one detector.

    Reported in preference to the mean, for the same reason spec 8.2 guard 3
    reports the worst compression cell: an average over generators hides the
    one an attacker will actually use.
    """
    folds = [f[detector].auc for f in record.logo_results.values()
             if detector in f]
    finite = [a for a in folds if a == a]
    return min(finite) if finite else float("nan")
