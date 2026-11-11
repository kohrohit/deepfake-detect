"""Phase 2 against Phase 1, on the worst LOGO fold, paired over the same rows.

WHY THIS IS NOT TWO NUMBERS SIDE BY SIDE. `bench/eval_gend.py` publishes an AUC and a grouped
bootstrap CI per fold. Reading two such runs and declaring the larger number a win is the error
this project has already paid for elsewhere: two independent intervals that happen not to overlap
is a weaker statement than it looks, and two that do overlap is not evidence of no difference.
Both runs score the SAME corpus, so the difference can be taken on the same rows and
bootstrapped by subject -- which answers "did the tuning do anything" rather than "is this
number bigger".

THE FOLD IS THE BASELINE'S WORST, NOT THE CANDIDATE'S. The plan's rule for Phase 2 is the worst
leave-one-generator-out fold, and a tuned run that collapsed a different family would otherwise
get to nominate an easier fold and call the swap an improvement.

WHAT A POSITIVE DELTA HERE STILL IS NOT. It is not a measurement against a generator -- every
fake in this corpus is a self-blend or a classical composite made in this repo -- and it is not
permission to decide. The 0.75 evidence floor is a separate gate (`src/dfd/service/evidence.py`),
and beating the frozen probe on a fold that sits at 0.738 does not by itself clear it.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from bench.metrics import auc, bootstrap_ci_by_group

logger = logging.getLogger(__name__)

#: Resamples for the paired delta. Matches `logo_refit`'s own bootstrap so
#: the two intervals in one report are not drawn from different budgets.
DEFAULT_BOOTSTRAP = 1000


def load_scorefile(path: str | Path) -> dict[str, Any]:
    """One run's per-record out-of-fold scores, as `eval_gend` writes them."""
    loaded: dict[str, Any] = json.loads(Path(path).read_text())
    if "folds" not in loaded:
        raise ValueError(f"{path} carries no `folds`; it is not a scorefile")
    return loaded


def paired_delta_ci(baseline: npt.NDArray[Any], candidate: npt.NDArray[Any],
                    labels: npt.NDArray[Any], groups: npt.NDArray[Any],
                    *, n: int = DEFAULT_BOOTSTRAP, seed: int = 0,
                    ) -> tuple[float, float, float]:
    """`(delta, ci_lo, ci_hi)` for `AUC(candidate) - AUC(baseline)`.

    PAIRED: the two score vectors are stacked into one array and resampled
    TOGETHER, by group, so every resample compares the two encoders on the
    same subjects. Resampling them independently would add a spread that
    exists only because two draws differ, and would put a visible interval
    around a difference of a vector with itself -- which
    `tests/bench/test_compare_lntuned.py` asserts is exactly zero.
    """
    stacked = np.column_stack([np.asarray(baseline, dtype=float),
                               np.asarray(candidate, dtype=float)])

    def _delta(scores: npt.NDArray[Any], y: npt.NDArray[Any]) -> float:
        return auc(scores[:, 1], y) - auc(scores[:, 0], y)

    delta = _delta(stacked, np.asarray(labels))
    lo, hi = bootstrap_ci_by_group(stacked, labels, groups, _delta,
                                   n=n, seed=seed)
    return float(delta), float(lo), float(hi)


def _aligned(baseline_rows: Sequence[dict[str, Any]],
             candidate_rows: Sequence[dict[str, Any]], *, fold: str,
             ) -> tuple[npt.NDArray[Any], ...]:
    """Both runs' scores for one fold, in one order, or a refusal.

    The join key is the sample id. Two runs over different records would
    compare one encoder on one corpus against another on another, and report
    the difference as the effect of tuning.
    """
    base = {str(r["sample_id"]): r for r in baseline_rows}
    cand = {str(r["sample_id"]): r for r in candidate_rows}
    if set(base) != set(cand):
        only_base, only_cand = len(set(base) - set(cand)), len(set(cand) - set(base))
        raise ValueError(
            f"fold {fold!r}: the two runs did not score the same records "
            f"({only_base} only in the baseline, {only_cand} only in the "
            f"candidate). A paired comparison needs one corpus.")
    ids = sorted(base)
    return (np.array([base[i]["score"] for i in ids], dtype=float),
            np.array([cand[i]["score"] for i in ids], dtype=float),
            np.array([base[i]["label"] for i in ids], dtype=int),
            np.array([str(base[i]["subject_id"]) for i in ids]))


def compare_scorefiles(baseline: dict[str, Any], candidate: dict[str, Any],
                       *, n_bootstrap: int = DEFAULT_BOOTSTRAP,
                       seed: int = 0) -> dict[str, Any]:
    """Per-fold AUCs for both runs, the paired delta, and the verdict.

    Raises:
        ValueError: if the two runs cover different folds or different
            records. Both are the same failure wearing two shapes -- a
            comparison of two different measurements — and neither is
            recoverable by dropping the mismatch, because which rows were
            dropped would then decide the answer.
    """
    if set(baseline["folds"]) != set(candidate["folds"]):
        raise ValueError(
            f"the two runs cover different folds: baseline "
            f"{sorted(baseline['folds'])}, candidate "
            f"{sorted(candidate['folds'])}")

    folds: dict[str, dict[str, float]] = {}
    for fold in sorted(baseline["folds"]):
        b, c, y, g = _aligned(baseline["folds"][fold], candidate["folds"][fold],
                              fold=fold)
        delta, lo, hi = paired_delta_ci(b, c, y, g, n=n_bootstrap, seed=seed)
        folds[fold] = {
            "baseline_auc": float(auc(b, y)), "candidate_auc": float(auc(c, y)),
            "delta": delta, "delta_ci_lo": lo, "delta_ci_hi": hi,
            "n": int(len(y)), "n_fake": int(y.sum()),
            "n_subjects": int(len(set(g.tolist()))),
        }

    worst = min(folds, key=lambda f: folds[f]["baseline_auc"])
    at_worst = folds[worst]
    # "Improved" means the point estimate moved up AND the paired interval
    # clears zero. A positive delta whose interval straddles zero is a run
    # that cannot be distinguished from noise, and this project has shipped
    # that mistake before.
    improved = at_worst["delta"] > 0 and at_worst["delta_ci_lo"] > 0
    return {
        "baseline_encoder": baseline.get("encoder"),
        "candidate_encoder": candidate.get("encoder"),
        "folds": folds,
        "worst_fold": worst,
        "worst_fold_note": (
            "the BASELINE's worst fold. The candidate does not get to "
            "nominate its own easiest one."),
        "decision": {
            "improved": bool(improved),
            "baseline_auc": at_worst["baseline_auc"],
            "candidate_auc": at_worst["candidate_auc"],
            "delta": at_worst["delta"],
            "delta_ci95": [at_worst["delta_ci_lo"], at_worst["delta_ci_hi"]],
            "paired": True,
            "n_bootstrap": n_bootstrap,
        },
        "still_not": (
            "Not a measurement against any generator: every fake in this "
            "corpus is a self-blend or a classical composite made in this "
            "repo. Not permission to decide either — the 0.75 evidence "
            "floor is a separate gate and this comparison does not address "
            "it."),
    }


def render_markdown(result: dict[str, Any]) -> str:
    """The comparison as a table, worst fold first and labelled as such."""
    worst = result["worst_fold"]
    d = result["decision"]
    lines = [
        "# Phase 2 against Phase 1 — LayerNorm-tuned versus frozen",
        "",
        f"Baseline encoder: `{result['baseline_encoder']}`  ",
        f"Candidate encoder: `{result['candidate_encoder']}`",
        "",
        f"**The fold that decides it is `{worst}`** — the baseline's worst, "
        f"not the candidate's.",
        "",
        f"    frozen  {d['baseline_auc']:.3f}",
        f"    tuned   {d['candidate_auc']:.3f}",
        f"    paired delta {d['delta']:+.3f}  "
        f"95% CI [{d['delta_ci95'][0]:+.3f}, {d['delta_ci95'][1]:+.3f}]",
        "",
        ("**Improved.**" if d["improved"] else "**Not improved.**")
        + " The delta is "
        + ("positive with a paired interval clear of zero."
           if d["improved"]
           else "not distinguishable from zero on a paired bootstrap by "
                "subject, so Phase 1's weights stay shipped and this is "
                "recorded as measured."),
        "",
        "| fold | frozen | tuned | paired delta | 95% CI | n | n fake | subjects |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for fold, f in sorted(result["folds"].items(),
                          key=lambda kv: kv[1]["baseline_auc"]):
        lines.append(
            f"| `{fold}`{' **(worst)**' if fold == worst else ''} "
            f"| {f['baseline_auc']:.3f} | {f['candidate_auc']:.3f} "
            f"| {f['delta']:+.3f} "
            f"| [{f['delta_ci_lo']:+.3f}, {f['delta_ci_hi']:+.3f}] "
            f"| {f['n']} | {f['n_fake']} | {f['n_subjects']} |")
    lines += ["", "## What this still does not say", "",
              result["still_not"], ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path,
                        help="the FROZEN run's scorefile "
                             "(bench/gend_report.scores.json)")
    parser.add_argument("--candidate", required=True, type=Path,
                        help="the TUNED run's scorefile")
    parser.add_argument("--out-json", type=Path,
                        default=Path("bench/lntuned_vs_frozen.json"))
    parser.add_argument("--out-md", type=Path,
                        default=Path("bench/lntuned_vs_frozen.md"))
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        result = compare_scorefiles(
            load_scorefile(args.baseline), load_scorefile(args.candidate),
            n_bootstrap=args.bootstrap, seed=args.seed)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True,
                                        allow_nan=False) + "\n")
    args.out_md.write_text(render_markdown(result))
    d = result["decision"]
    logger.info("worst fold %s: frozen %.4f -> tuned %.4f, paired delta "
                "%+.4f [%+.4f, %+.4f]; improved=%s",
                result["worst_fold"], d["baseline_auc"], d["candidate_auc"],
                d["delta"], d["delta_ci95"][0], d["delta_ci95"][1],
                d["improved"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
