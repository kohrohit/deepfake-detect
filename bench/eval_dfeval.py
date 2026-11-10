"""Run the benchmark over Deepfake-Eval-2024, gated on an oracle.

    python3 -m bench.eval_dfeval \\
        --root ~/Desktop/agents/datasets/deepfake_eval_2024/images \\
        --scores baselines/ufd/dfeval.csv \\
        --out-md bench/dfeval_report.md --out-json bench/dfeval_report.json

**The oracle is why this file is different from its siblings.** The paper
that published this corpus reports NPR at AUC 0.53 on its image split, and
NPR is slot C of this repository. So unlike DF40 or the swap corpus, this
one ships with a way to check the HARNESS rather than only the detector: if
our slot C lands outside `ORACLE_LO`-`ORACLE_HI` on the face subset, the
harness has not been shown to measure anything here, and the run writes the
oracle block plus the corpus composition and exits non-zero. No detector
number, no control, no baseline. `--allow-oracle-failure` exists so the
failure itself can be investigated, and mirrors `bench.eval_swaps`'
`--allow-training-window`.

A fresh checkout fails that gate, and should: `assets/models/npr.pt` is
gitignored and absent, so slot C abstains on every row and its AUC is not a
number at all. "Slot C could not be measured" and "slot C disagrees with
the paper" are both answers of the same kind — the harness cannot be shown
to work — and both stop the report.

**There is no corpus-wide AUC in this report, for any detector.** Unlike
every other corpus here, this one is raw in-the-wild media and part of it
has no face at all (`corpora.dfeval` keeps those records rather than
dropping them). A face detector cannot score them, and a row it could not
score is not a row it got wrong. The field is `auc_face_subset`, beside
`n_face_subset`, so a reader cannot quote an aggregate that was never
computed; the rows outside it are accounted for in the abstention-by-class
table instead.

**Guards are not enforced here at all** — this entry point scores detectors
directly rather than through `bench.runner.run_benchmark`, the same posture
`enforce_guards=False` buys its siblings — and the two that would fail are
named in the waiver block rather than quietly satisfied with invented
labels.

**Leave-one-generator-out is refused and recorded as refused.** Every fake
carries one generator id, so `logo_splits` raises `UnsplittableCorpusError`
and `logo_verdict` returns the refusal sentence that goes where a LOGO
table would. A plain `ValueError` from `bench.protocol._validate` is NOT
caught: that means a malformed corpus, and it must propagate.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from corpora.dfeval import DFEVAL_GENERATOR, load_dfeval_images
from corpora.face_pool import DetectFn
from dfd.detectors.registry import default_registry
from dfd.faces import detect_faces

from .dfeval_controls import colour_features, jpeg_table_features, run_control
from .eval_df40 import _json_safe
from .metrics import auc, bootstrap_ci_by_group, kish_ess, tpr_at_fpr
from .protocol import UnsplittableCorpusError, logo_splits
from .runner import _observation
from .scorefile import aligned, load_score_file

logger = logging.getLogger(__name__)

#: The paper's published AUC for NPR on this corpus's image split.
PUBLISHED_NPR_AUC = 0.53

#: The detector the oracle is read off, by NAME. Slot C is identified by its
#: registry name rather than by its `slot` attribute because the published
#: number is for NPR specifically, not for "whatever occupies slot C today":
#: a future slot C with different physics would inherit a band argued for
#: somebody else's method.
ORACLE_DETECTOR = "npr"

#: The band inside which our slot C must land for anything else to be
#: reported. NOT a tolerance on a reproduction: our NPR is a
#: reimplementation whose constants are recorded as unverified against Tan
#: et al. (see `dfd/detectors/npr.py`), and the paper scored whole images
#: where we score a face ROI at its native resolution. A decimal match is
#: not expected and demanding one would block the harness over a non-bug.
#:
#: What this band IS for is catching gross errors, which have signatures:
#:
#:   below 0.40  the labels, the ids or the score direction are inverted.
#:               The published number is 0.53; landing under 0.40 is not a
#:               weaker NPR, it is a wired-up-backwards one.
#:   above 0.75  every published open-source model on this corpus sits at
#:               chance (UFD 0.56, DistilDIRE 0.52, NPR 0.53). A handcrafted
#:               27+30-feature head beating all of them by 20 points is the
#:               signature of a label leak or misaligned ids, not a
#:               breakthrough. This is the direction nobody checks, and it
#:               is therefore the one that ships.
#:
#: Between them, the difference is attributable to the ROI and
#: implementation differences named above, and the run proceeds.
ORACLE_LO = 0.40
ORACLE_HI = 0.75

#: Headings the two prose blocks render under. Named so a test can assert
#: each block is present without pinning its wording.
WAIVER_HEADING = "## Guards waived for this corpus"
SCOPE_HEADING = "## What this corpus can and cannot show"

_WAIVER = f"""{WAIVER_HEADING}

This entry point does not call `bench.runner.run_benchmark`, so no spec §8.2
guard runs over this corpus at all — the same posture `enforce_guards=False`
buys `bench/eval_df40.py`. Two of them would fail if they did run, and both
failures are properties of in-the-wild media rather than of this run:

- **`check_compression_coverage`.** Every record carries
  `compression=unknown`. The images arrive from 88 web domains at whatever
  quality each platform's re-encode pipeline produced, and nothing in the
  manifest says what that was. An invented `c23` would pass the guard by
  lying to it, which is the cheaper path and is what makes a benchmark
  flatter itself.
- **`check_identity_disjoint`.** In-the-wild media carries no identity
  labels: `corpora.dfeval` sets `subject_id` to the sample's own id because
  that is all it knows. Nothing is split here in any case — no fold is
  built, no model is fitted — so there is no split for the guard to certify.

Read everything below as a cross-corpus sanity check on unseen media, never
as a leave-one-generator-out result.
"""


def _scope_block(n_face: int, n_total: int, n_baselines: int) -> str:
    """The scope block, which carries the face-subset fraction it describes.

    A constant cannot state what fraction of THIS run's corpus could be
    scored, and that fraction is the single most load-bearing caveat on
    every number below it.
    """
    pct = (100.0 * n_face / n_total) if n_total else float("nan")
    headline = (f"{n_face} of {n_total} records ({pct:.1f}%)")
    if not n_baselines:
        baselines = "No baseline score file was ingested"
    elif n_baselines == 1:
        baselines = "One baseline score file was ingested"
    else:
        baselines = f"{n_baselines} baseline score files were ingested"
    return f"""{SCOPE_HEADING}

**This corpus cannot fold.** Deepfake-Eval-2024 publishes no per-item
generator attribution, so every fake here carries one id
(`{DFEVAL_GENERATOR}`) that says outright that the technique is unknown.
Leave-one-generator-out therefore has exactly one generator to hold out,
`bench.protocol.logo_splits` refuses the split, and that refusal is
recorded below where a LOGO table would be. Spec §8.1 calls the
held-out-generator number the only one that predicts field performance;
this corpus cannot produce it, and inventing generator ids from filenames
or domains would produce a number that looks like it.

**The headline is a face subset: {headline}.** The rest are in-the-wild
media with no detectable face at all — a screenshot, a caption card, a
non-facial frame — and `corpora.dfeval` keeps them rather than dropping
them, so they stay in the denominator. Every detector AUC below is
computed over the face subset
ALONE, and there is deliberately no corpus-wide AUC for any detector: a row
a face detector could not score is not a row it got wrong, and averaging
the two together produces a number with no operational meaning that will
nonetheless be quoted. The unscored rows are accounted for in the
abstention-by-class table instead, which is where a label-correlated
refusal to score would show up.

**The baselines are our reproductions, run in their own environments.**
{baselines} as `sample_id,score` rows; no baseline's code or weights enter
this tree (`bench.scorefile` says why). A discrepancy between a number here
and the one its paper published could be theirs, ours, or the environment's,
and nothing in this report can tell you which. Coverage travels beside every
baseline for the same reason.

**This is 2024 media.** The corpus was collected in 2024 and generators have
moved since; a detector that does well here has been shown to do well on
what was circulating then, which is a weaker claim than it reads as, and a
detector that does badly may be failing on a generation nobody deploys any
more.

**Nothing here compares this system to any vendor.** No commercial detector
is run, quoted or implied anywhere in this report.

"""


def score_detectors(records: Sequence[dict[str, Any]],
                    registry: Any) -> dict[str, dict[str, Any]]:
    """Score every registered detector over every record, in record order.

    Observations are built by `bench.runner._observation` rather than
    constructed here, deliberately. That function honours a record's own
    `roi` and `landmarks` — which this corpus, uniquely, carries, because
    its records are whole frames rather than face crops — and a second
    record-to-Observation path would be free to drift from the first. One
    function decides how a record becomes an Observation.

    Returns:
        `detector name -> {scores, abstained, reasons}`, each aligned
        index-for-index to `records`. `scores` is NaN exactly where
        `abstained` is True; `reasons` carries the detector's own stated
        reason for every row, abstained or not, so an abstention can be
        attributed (`weights_absent` reads very differently from
        `below_quality_floor`).
    """
    observations = [_observation(r) for r in records]
    out: dict[str, dict[str, Any]] = {}
    for name in registry.names():
        det = registry.get(name)
        scores: list[float] = []
        abstained: list[bool] = []
        reasons: list[str] = []
        for obs in observations:
            raw = det.score([obs])
            if raw.abstained or raw.score is None:
                scores.append(float("nan"))
                abstained.append(True)
            else:
                scores.append(float(raw.score))
                abstained.append(False)
            reasons.append(str(raw.reason))
        out[name] = {
            "scores": np.asarray(scores, dtype=float),
            "abstained": np.asarray(abstained, dtype=bool),
            "reasons": reasons,
            "version": str(det.version),
        }
    return out


def face_subset(records: Sequence[dict[str, Any]]) -> np.ndarray:
    """Boolean mask of the records a face was found in."""
    return np.asarray([bool(r["has_face"]) for r in records], dtype=bool)


def abstention_by_class(abstained: np.ndarray,
                        labels: np.ndarray) -> dict[str, tuple[int, int]]:
    """`(abstained, total)` for the reals and for the fakes.

    Deliberately simpler than `bench.runner._abstention_by_class`, which
    keys fakes by their generator id: this corpus has exactly one generator
    id, so that breakdown would be one bucket wearing a longer name.

    Both classes are always present as keys, counting every row given, so a
    missing class cannot read as "that class did not abstain" — which is the
    failure this breakdown exists to prevent. One aggregate rate cannot show
    label-correlated abstention, and label-correlated abstention means every
    metric above it was computed on a label-selected subsample.
    """
    a = np.asarray(abstained, dtype=bool)
    y = np.asarray(labels, dtype=int)
    out: dict[str, tuple[int, int]] = {}
    for key, value in (("real", 0), ("fake", 1)):
        rows = y == value
        out[key] = (int(a[rows].sum()), int(rows.sum()))
    return out


def check_oracle(npr_auc: float) -> tuple[bool, str]:
    """Does our slot C land where the corpus's own paper says it should.

    Args:
        npr_auc: slot C's AUC over the face subset. NaN is a failure, not a
            skip: "slot C could not be measured" and "slot C disagrees with
            the paper" are the same kind of answer — the harness has not
            been shown to work here — and both must stop the report.

    Returns:
        `(passed, message)`. The message always names the published number
        and both bounds, because it is read by whoever has to decide
        whether the run or the detector is broken.
    """
    band = (f"the band {ORACLE_LO:.2f}-{ORACLE_HI:.2f} around the published "
            f"NPR AUC of {PUBLISHED_NPR_AUC:.2f}")
    if not np.isfinite(npr_auc):
        return False, (
            f"oracle: {ORACLE_DETECTOR} produced no AUC at all over the face "
            f"subset (every row abstained, or one label was missing from "
            f"it), so it cannot be checked against {band}. On a fresh "
            f"checkout this is `assets/models/npr.pt` being absent.")
    if npr_auc < ORACLE_LO:
        return False, (
            f"oracle: {ORACLE_DETECTOR} scored {npr_auc:.3f} over the face "
            f"subset, below {ORACLE_LO:.2f} and outside {band}. That is not "
            f"a weaker NPR, it is a wired-up-backwards one: the labels, the "
            f"sample ids or the score direction are inverted somewhere "
            f"between the loader and the metric.")
    if npr_auc > ORACLE_HI:
        return False, (
            f"oracle: {ORACLE_DETECTOR} scored {npr_auc:.3f} over the face "
            f"subset, above {ORACLE_HI:.2f} and outside {band}. Every "
            f"published open-source model on this corpus sits at chance "
            f"(UFD 0.56, DistilDIRE 0.52, NPR 0.53); beating all of them is "
            f"the signature of a label leak or misaligned ids, not a "
            f"breakthrough.")
    return True, (
        f"oracle: {ORACLE_DETECTOR} scored {npr_auc:.3f} over the face "
        f"subset, inside {band}. The remaining difference is attributable "
        f"to the face-ROI scoring and to this project's NPR being an "
        f"unverified reimplementation; the run proceeds.")


def logo_verdict(records: Sequence[dict[str, Any]]) -> str:
    """The sentence that goes where a leave-one-generator-out table would.

    `UnsplittableCorpusError` is caught and recorded — that is the expected
    answer on this corpus and the whole point of this function. A plain
    `ValueError` from `bench.protocol._validate` is NOT caught: it means a
    malformed corpus (a straddling source, a label outside {0, 1}, an
    unattributed fake) and must propagate rather than be reported as a
    scope limit.
    """
    try:
        splits = logo_splits(list(records))
    except UnsplittableCorpusError as exc:
        return (
            "**Leave-one-generator-out cannot fold on this corpus, and no "
            "LOGO number is reported.** Deepfake-Eval-2024 publishes no "
            f"per-item generator attribution, so every fake carries the one "
            f"id `{DFEVAL_GENERATOR}` and there is no second generator to "
            f"hold out. `bench.protocol.logo_splits` refused the split: "
            f"{exc}. Spec §8.1 calls the held-out-generator number the only "
            "one that predicts field performance; this corpus cannot "
            "produce it, and fabricating generator ids from filenames or "
            "domains would produce a number that looks like it.")
    return (
        f"**This corpus now folds into {len(splits)} leave-one-generator-out "
        "split(s), and this report does not compute them.** That is a change "
        "in the corpus, not a scope limit: re-run it through "
        "`bench.runner.run_benchmark`, which owns the LOGO path, rather than "
        "quoting the face-subset numbers below as generalisation results.")


def _kish_by_domain(records: Sequence[dict[str, Any]], label: int) -> float:
    """Kish ESS over the web-domain sizes of one class.

    Domains, not rows: 88 domains' worth of information is what this corpus
    carries, and one domain contributing most of a class collapses this
    number toward 1. Reported beside every n so a reader is not invited to
    read rows as independent observations.
    """
    counts = Counter(str(r["domain"]) for r in records if r["label"] == label)
    if not counts:
        return float("nan")
    return kish_ess(list(counts.values()))


def _control_or_refusal(features: np.ndarray, labels: np.ndarray,
                        groups: np.ndarray, *, seed: int) -> dict[str, Any]:
    """`run_control`, with its refusal carried into the report as a string.

    A control exists to produce a legible verdict about the corpus, so its
    own "this corpus cannot be split that way" message belongs in the
    report rather than as a stack trace out of the entry point. The message
    travels verbatim — nothing is swallowed, and the key is `refused` so a
    reader cannot mistake it for a result.
    """
    try:
        return run_control(features, labels, groups, seed=seed)
    except ValueError as exc:
        logger.warning("control refused: %s", exc)
        return {"refused": str(exc)}


def _auc_block(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray,
               *, seed: int, prefix: str = "") -> dict[str, Any]:
    """AUC over the rows given, with a domain-grouped interval beside it.

    Never keyed `auc`: the caller is reporting a FACE SUBSET, and a field
    called `auc` on a corpus that is not all faces will be quoted by
    somebody who did not read the caveat.
    """
    nan = float("nan")
    usable = bool(scores.size) and len(np.unique(labels)) > 1
    counts = Counter(str(g) for g in groups.tolist())
    # One bootstrap, both ends. Calling it once per bound would resample a
    # thousand times for each, and -- since `seed` is fixed -- would spend
    # the second thousand recomputing exactly the first thousand.
    ci_lo, ci_hi = (bootstrap_ci_by_group(scores, labels, groups, auc,
                                          seed=seed) if usable else (nan, nan))
    return {
        f"{prefix}auc_face_subset": auc(scores, labels) if usable else nan,
        f"{prefix}auc_ci_lo": ci_lo,
        f"{prefix}auc_ci_hi": ci_hi,
        f"{prefix}tpr_at_1pct": (tpr_at_fpr(scores, labels, 0.01)
                                 if usable else nan),
        f"{prefix}tpr_at_0p1pct": (tpr_at_fpr(scores, labels, 0.001)
                                   if usable else nan),
        f"{prefix}n_scored_face_subset": int(scores.size),
        f"{prefix}kish_ess_face_subset": (kish_ess(list(counts.values()))
                                          if counts else nan),
    }


def _corpus_block(records: Sequence[dict[str, Any]], face: np.ndarray,
                  skipped: dict[str, int], args: argparse.Namespace,
                  ) -> dict[str, Any]:
    domains = Counter(str(r["domain"]) for r in records)
    return {
        "root": str(args.root),
        "records": len(records),
        "reals": int(sum(1 for r in records if r["label"] == 0)),
        "fakes": int(sum(1 for r in records if r["label"] == 1)),
        "n_face_subset": int(face.sum()),
        "n_no_face": int((~face).sum()),
        "domains": len(domains),
        "largest_domain": max(domains.values()) if domains else 0,
        "kish_ess_real": _kish_by_domain(records, 0),
        "kish_ess_fake": _kish_by_domain(records, 1),
        "skipped": dict(skipped),
        "seed": args.seed,
        "limit": args.limit,
        "generator": DFEVAL_GENERATOR,
    }


def _oracle_block(scored: dict[str, dict[str, Any]], labels: np.ndarray,
                  face: np.ndarray, *, allowed: bool) -> dict[str, Any]:
    entry = scored.get(ORACLE_DETECTOR)
    if entry is None:
        passed, message = False, (
            f"oracle: no detector named {ORACLE_DETECTOR!r} is registered, "
            f"so the published AUC of {PUBLISHED_NPR_AUC:.2f} on this "
            f"corpus cannot be checked against anything.")
        observed = float("nan")
        n_scored = 0
    else:
        scores = entry["scores"]
        valid = face & np.isfinite(scores)
        observed = (auc(scores[valid], labels[valid])
                    if valid.any() and len(np.unique(labels[valid])) > 1
                    else float("nan"))
        n_scored = int(valid.sum())
        passed, message = check_oracle(observed)
    return {
        "detector": ORACLE_DETECTOR,
        "published_auc": PUBLISHED_NPR_AUC,
        "lo": ORACLE_LO,
        "hi": ORACLE_HI,
        "auc_face_subset": observed,
        "n_face_subset": int(face.sum()),
        "n_scored_face_subset": n_scored,
        "passed": bool(passed),
        "message": message,
        "failure_allowed": bool(allowed),
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    """The report, leading with the waiver, then the scope, then the numbers.

    Hand-rendered rather than through `bench.report.render_markdown`, which
    renders a `RunRecord`: this entry point does not build one, because a
    `RunRecord` carries a corpus-wide `auc` field and this corpus has no
    honest value to put in it.
    """
    oracle = payload["oracle"]
    corpus = payload["corpus"]
    lines = [
        "# Deepfake-Eval-2024 — benchmark report",
        "",
        f"**Oracle: {'PASSED' if oracle['passed'] else 'FAILED'}.** "
        f"{oracle['message']}",
        "",
    ]
    if not oracle["passed"]:
        lines += [
            ("**`--allow-oracle-failure` was passed, so the run continued.** "
             "Every number below it was produced by a harness that has NOT "
             "been shown to measure anything on this corpus, and none of "
             "them may be quoted as a result — they exist to investigate "
             "the oracle failure above."
             if oracle["failure_allowed"] else
             "**Every other number is withheld.** A harness that cannot "
             "reproduce a published result on a published corpus has not "
             "been shown to measure anything, so what follows is the corpus "
             "composition and nothing else."),
            "",
        ]

    if oracle["passed"] or oracle["failure_allowed"]:
        lines += [_WAIVER.rstrip(), "", str(payload["scope"]).rstrip(), ""]

    lines += [
        "## Corpus",
        "",
        f"- {corpus['records']} records ({corpus['reals']} real, "
        f"{corpus['fakes']} fake) from {corpus['domains']} web domains",
        f"- face subset: {corpus['n_face_subset']} "
        f"({corpus['n_no_face']} records carry no detectable face)",
        f"- Kish ESS over domains: {corpus['kish_ess_real']:.2f} real, "
        f"{corpus['kish_ess_fake']:.2f} fake "
        f"(largest domain: {corpus['largest_domain']} records)",
        f"- skipped by the loader: {corpus['skipped'] or 'nothing'}",
        "",
    ]

    if "controls" in payload:
        lines += ["## Shortcut controls", "",
                  "Neither control measures a detector. They answer the "
                  "prior question: do this corpus's halves differ in some "
                  "trivial way that ANY model would find. A control that "
                  "fires makes every detector number below it unreadable.",
                  "",
                  "| control | AUC | 95% CI | null range | p | n fit | n test |",
                  "|---|---|---|---|---|---|---|"]
        for name, c in payload["controls"].items():
            if "refused" in c:
                lines.append(f"| {name} | refused | — | — | — | — | — |")
            else:
                lines.append(
                    f"| {name} | {c['auc']:.3f} | "
                    f"{c['ci_lo']:.3f}–{c['ci_hi']:.3f} | "
                    f"{c['null_lo']:.3f}–{c['null_hi']:.3f} | {c['p']:.3f} | "
                    f"{c['n_fit']} | {c['n_test']} |")
        for name, c in payload["controls"].items():
            if "refused" in c:
                lines += ["", f"`{name}` was refused: {c['refused']}"]
        lines.append("")

    if "detectors" in payload:
        lines += ["## Detectors", "",
                  "`auc_face_subset` is over the face subset ALONE. There is "
                  "no corpus-wide AUC here and quoting one is not possible: "
                  "see the scope block.",
                  "",
                  "| detector | version | AUC (face subset) | 95% CI | "
                  "TPR@1%FPR | n scored | Kish ESS | abstained real | "
                  "abstained fake |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for name, d in payload["detectors"].items():
            ar, tr = d["abstention_by_class"]["real"]
            af, tf = d["abstention_by_class"]["fake"]
            lines.append(
                f"| {name} | {d['version']} | "
                f"{_fmt(d['auc_face_subset'])} | "
                f"{_fmt(d['auc_ci_lo'])}–{_fmt(d['auc_ci_hi'])} | "
                f"{_fmt(d['tpr_at_1pct'])} | {d['n_scored_face_subset']} | "
                f"{_fmt(d['kish_ess_face_subset'], 2)} | {ar}/{tr} | "
                f"{af}/{tf} |")
        lines.append("")

    if payload.get("baselines"):
        lines += ["## Baselines", "",
                  "Ingested as `sample_id,score` rows. No baseline's code or "
                  "weights enter this tree; coverage is reported, never "
                  "thresholded.",
                  "",
                  "| baseline | coverage | covered | AUC (face subset) | "
                  "95% CI | n scored |",
                  "|---|---|---|---|---|---|"]
        for name, b in payload["baselines"].items():
            lines.append(
                f"| {name} | {b['coverage']:.1%} | {b['covered']}/{b['total']} "
                f"| {_fmt(b['auc_face_subset'])} | "
                f"{_fmt(b['auc_ci_lo'])}–{_fmt(b['auc_ci_hi'])} | "
                f"{b['n_scored_face_subset']} |")
        lines.append("")

    if "logo" in payload:
        lines += ["## Leave-one-generator-out", "", payload["logo"], ""]
    return "\n".join(lines) + "\n"


def _fmt(value: Any, places: int = 3) -> str:
    """A number, or `n/a` — never a bare `nan` a reader might quote."""
    if value is None:
        return "n/a"
    f = float(value)
    return "n/a" if f != f else f"{f:.{places}f}"


def _write(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    safe = _json_safe(payload)
    assert isinstance(safe, dict)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(safe, indent=2, allow_nan=False))
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(_render_markdown(payload))
    logger.info("wrote %s and %s", args.out_md, args.out_json)


def main(argv: Sequence[str] | None = None, *,
         detect: DetectFn = detect_faces, registry: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path,
                        help="the Deepfake-Eval-2024 image directory, with "
                             "its metadata.csv (see corpora.dfeval)")
    parser.add_argument("--limit", type=int, default=None,
                        help="manifest rows read, sampled with --seed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scores", type=Path, action="append", default=[],
                        help="a baseline's sample_id,score CSV. Repeatable; "
                             "each is reported with its coverage and none of "
                             "its code is imported (bench.scorefile)")
    parser.add_argument("--allow-oracle-failure", action="store_true",
                        help="report the rest of the run even though slot C "
                             "missed the published band. For investigating "
                             "the failure itself — every number produced "
                             "under this flag comes from a harness that has "
                             "not been shown to measure anything")
    parser.add_argument("--out-md", type=Path,
                        default=Path("bench/dfeval_report.md"))
    parser.add_argument("--out-json", type=Path,
                        default=Path("bench/dfeval_report.json"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    records, skipped = load_dfeval_images(
        args.root, limit=args.limit, seed=args.seed, detect=detect)
    if not records:
        logger.error("no usable records under %s (skipped: %s)",
                     args.root, skipped or "nothing")
        return 1
    present = {r["label"] for r in records}
    if len(present) < 2:
        # Every metric in this report is defined over both labels, the
        # oracle included. A table of `nan` invites a reader to re-run it
        # rather than to fix the corpus.
        logger.error("corpus has only label(s) %s; AUC and TPR@FPR are both "
                     "undefined over one label", sorted(present))
        return 1

    if registry is None:
        registry = default_registry()

    labels = np.asarray([r["label"] for r in records], dtype=int)
    domains = np.asarray([str(r["domain"]) for r in records])
    face = face_subset(records)

    scored = score_detectors(records, registry)

    oracle = _oracle_block(scored, labels, face,
                           allowed=args.allow_oracle_failure)
    payload: dict[str, Any] = {
        "oracle": oracle,
        "corpus": _corpus_block(records, face, skipped, args),
        "guards_enforced": False,
    }
    if not oracle["passed"]:
        logger.error("%s", oracle["message"])
        if not args.allow_oracle_failure:
            logger.error("writing the oracle block and the corpus "
                         "composition only; pass --allow-oracle-failure to "
                         "investigate the failure with the rest of the run")
            _write(args, payload)
            return 1
        logger.warning("--allow-oracle-failure: continuing with a harness "
                       "that has NOT been shown to measure anything here")

    bgr = [cv2.cvtColor(r["image"], cv2.COLOR_RGB2BGR) for r in records]
    controls: dict[str, Any] = {
        "colour": _control_or_refusal(colour_features(bgr), labels, domains,
                                      seed=args.seed),
    }
    # `record["path"]`, never a path re-derived from `sample_id`: the
    # JPEG-table control reads a file header, so the file a record came
    # from is part of what the record IS, and a join reconstructed from an
    # id format would fail silently into an honest-looking refusal when the
    # gated download's layout differs from the provisional one.
    paths = [Path(r["path"]) for r in records]
    controls["jpeg_table"] = _control_or_refusal(
        jpeg_table_features(paths), labels, domains, seed=args.seed)
    payload["controls"] = controls

    detectors: dict[str, Any] = {}
    for name, entry in scored.items():
        scores = entry["scores"]
        valid = face & np.isfinite(scores)
        detectors[name] = {
            "version": entry["version"],
            "n_face_subset": int(face.sum()),
            **_auc_block(scores[valid], labels[valid], domains[valid],
                         seed=args.seed),
            "abstention_by_class": abstention_by_class(entry["abstained"],
                                                       labels),
            "abstention_rate": float(entry["abstained"].mean()),
            "abstention_reasons": dict(Counter(
                r for r, a in zip(entry["reasons"], entry["abstained"],
                                  strict=True) if a)),
        }
    payload["detectors"] = detectors

    face_records = [r for r, keep in zip(records, face, strict=True) if keep]
    known = {str(r["sample_id"]) for r in records}
    baselines: dict[str, Any] = {}
    for path in args.scores:
        try:
            sf = load_score_file(path, known)
        except ValueError as exc:
            logger.error("baseline score file refused: %s", exc)
            return 1
        s, y, g = aligned(sf, face_records)
        baselines[sf.name] = {
            "coverage": sf.coverage,
            "covered": sf.covered,
            "total": sf.total,
            "n_face_subset": len(face_records),
            **_auc_block(s, y, g, seed=args.seed),
        }
    payload["baselines"] = baselines

    payload["logo"] = logo_verdict(records)
    payload["scope"] = _scope_block(int(face.sum()), len(records),
                                    len(baselines))
    _write(args, payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
