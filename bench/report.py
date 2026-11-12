"""Markdown rendering of a benchmark run.

Deliberately prints no accuracy figure. At fraud base rates accuracy is a
number that always looks good and never means anything.
"""
from __future__ import annotations

from .runner import REAL_CLASS, RunRecord, worst_logo_auc

ADVERSARIAL_FLOOR = 0.10


def _f(x) -> str:
    return "n/a" if x is None or x != x else f"{x:.3f}"



def _rate(pair: tuple[int, int]) -> str:
    done, total = pair
    return f"{100.0 * done / total:.1f}% ({done}/{total})" if total else "n/a"


def _abstention_block(record, detectors, generators) -> list[str]:
    """Per-fold abstention, split into the genuine half and the held-out fakes.

    Every AUC in the table above is computed over the records that did NOT
    abstain. When abstention is correlated with the label, that AUC describes
    the survivors rather than the technique — and the survivors of a quality
    floor are the least degraded fakes, which is the direction that flatters
    a detector. Measured on the swap corpus 2026-09-24: `blend_seam` abstains
    on 74.3% of `swap_lowres_paste` fakes against 56.9% of reals, and
    `swap_lowres_paste` is the fold with the best AUC in that report.

    Rendered only when something was measured. An empty `abstention_by_class`
    means "not measured", and printing it as 0% would read as "nothing
    abstained" — the opposite claim.
    """
    measured = any(record.logo_results[g][name].abstention_by_class
                   for g in generators for name in detectors
                   if name in record.logo_results[g])
    if not measured:
        return []

    lines = ["### Abstention by class, per fold\n",
             "Every AUC above is computed over the records that did NOT "
             "abstain. Where these two columns differ, the fold's AUC "
             "describes the survivors rather than the technique — and the "
             "survivors of a quality floor are the least degraded fakes, "
             "which flatters the detector.\n",
             "| held out | detector | genuine | held-out fakes |",
             "|---|---|---|---|"]
    for g in generators:
        for name in detectors:
            if name not in record.logo_results[g]:
                continue
            by_class = record.logo_results[g][name].abstention_by_class
            if not by_class:
                continue
            # The held-out generator's own key, never an aggregate over the
            # fold's fakes: a fold's test side is that generator alone, and
            # naming the column after it is what lets a reader compare the
            # two numbers on the same row.
            lines.append(f"| {g} | {name} | "
                         f"{_rate(by_class.get(REAL_CLASS, (0, 0)))} | "
                         f"{_rate(by_class.get(g, (0, 0)))} |")
    lines.append("")
    return lines


def render_markdown(record: RunRecord) -> str:
    lines: list[str] = []
    lines.append("# Benchmark run\n")
    lines.append("## Reproducibility record\n")
    lines.append(f"- seed: `{record.seed}`")
    lines.append(f"- dataset hash: `{record.dataset_hash}`")
    lines.append(f"- guards enforced: `{record.guards_enforced}`")
    versions = ", ".join(f"{k}={v}" for k, v in sorted(record.model_versions.items()))
    lines.append(f"- model versions: `{versions}`")
    lines.append(f"- identity disjointness (criterion 2): `{record.identity_status}`")
    if record.identity_status == "violation":
        lines.append("  - **FAILED, and the run continued because guards are "
                     "waived.** The numbers below are the measurement, not a "
                     "pass.")
    if record.identity_report is not None:
        r = record.identity_report
        # The rate, never a recomputed denominator. `n_train * n_test` is the
        # pair count for a train/test check and NOT for the corpus-level
        # subject check, which compares n*(n-1)/2 pairs of subjects — the
        # same report type carries both, and multiplying the two sides
        # printed 15,625 for a comparison that made 7,750.
        lines.append(
            f"  - worst check: max cosine `{r.max_similarity:.4f}` at threshold "
            f"`{r.threshold}`, {r.violations} crossings "
            f"(`{r.violation_rate:.4%}` of compared pairs), tolerated "
            f"`{r.tolerated_rate:.4%}`, over {r.n_train} vs {r.n_test} ids")
        for gen, fold in sorted(record.identity_by_fold.items()):
            lines.append(
                f"  - held out {gen}: max cosine `{fold.max_similarity:.4f}`, "
                f"{fold.violations} crossings (`{fold.violation_rate:.4%}`)")
    elif record.identity_status == "not_measured":
        # An unmeasured criterion must never read like a passed one. Before
        # 2026-09-23 this line was absent entirely and `identity_report` was
        # hardcoded None, so a clean run and an unchecked one rendered alike.
        lines.append(
            "  - **No embeddings were supplied, so nothing was checked.** The "
            "splits are identity-disjoint by DECLARATION (`subject_id`) only; "
            "one person enrolled under two subject ids would sit on both "
            "sides and nothing here would see it.")
    if record.adversarial_status:
        states = sorted(set(record.adversarial_status.values()))
        lines.append(f"- adversarial robustness (criterion 8): `{', '.join(states)}`")
        for name, state in sorted(record.adversarial_status.items()):
            if state != "ok":
                lines.append(f"  - {name}: not attacked — `{state}`")
    lines.append(f"- demographic parity (criterion 11): `{record.parity_status}`")
    for name, pr in sorted(record.parity_by_detector.items()):
        rates = ", ".join(f"{k} {v:.4f}" for k, v in sorted(pr.fpr_by_stratum.items()))
        lines.append(f"  - {name}: FPR ratio `{pr.max_fpr_ratio:.2f}x` "
                     f"(ceiling `{pr.ceiling:.2f}x`) — {rates}")
    if record.parity_excluded_strata:
        excluded = ", ".join(f"{k} (n={v})"
                             for k, v in sorted(record.parity_excluded_strata.items()))
        lines.append(f"  - excluded as too small to compare: {excluded}")
    lines.append("")

    lines.append("## Leave-one-generator-out (spec §8.1)\n")
    if not record.logo_results:
        lines.append(
            "**Not computed for this corpus.** Without a held-out-generator "
            "number there is nothing here that predicts field performance; "
            "the table below measures memorisation only.\n")
    else:
        lines.append("Worst held-out generator per detector — the headline "
                     "number. Reported as the worst rather than the mean for "
                     "the same reason spec §8.2 guard 3 reports the worst "
                     "compression cell: an average hides the generator an "
                     "attacker will actually use.\n")
        detectors = sorted(record.detector_results)
        generators = sorted(record.logo_results)
        # Dropped-for-identity count sits beside each fold's AUC in the same
        # column header: `Split.dropped_for_identity` is a property of the
        # fold (subject/generator geometry), not of which detector scored
        # it, so it does not vary by row and does not need its own column.
        # Without it, n=23 does not say whether 2 or 20 records were
        # removed to reach that number.
        lines.append(
            "| detector | worst AUC | "
            + " | ".join(f"held out {g} (dropped {record.logo_dropped.get(g, 0)})"
                        for g in generators)
            + " |")
        lines.append("|---|---|" + "---|" * len(generators))
        for name in detectors:
            # `worst_logo_auc` already treats a detector missing from a
            # fold as absent rather than a KeyError (spec: a fold that
            # never scored a detector is unmeasured for it, not zero).
            # Building each row's cells must fail the same way, or a
            # bold worst-AUC renders and the row crashes building itself.
            cells = [_f(record.logo_results[g][name].auc)
                    if name in record.logo_results[g] else "n/a"
                    for g in generators]
            lines.append(f"| {name} | **{_f(worst_logo_auc(record, name))}** | "
                         + " | ".join(cells) + " |")
        lines.append("")
        lines.extend(_abstention_block(record, detectors, generators))

    lines.append("## In-dataset results — memorisation, not field performance\n")
    lines.append("These are computed over the whole corpus, with every "
                 "generator seen. Spec §8.1: in-dataset AUC measures "
                 "memorisation. Read the LOGO table above instead.\n")
    lines.append("| detector | AUC | 95% CI | TPR@FPR=1% | TPR@FPR=0.1% | "
                 "adversarial TPR@FPR=1% | ECE | abstained | p95 ms | n |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for name in sorted(record.detector_results):
        d = record.detector_results[name]
        lo, hi = d.auc_ci
        lines.append(
            f"| {d.detector} | {_f(d.auc)} | {_f(lo)}–{_f(hi)} | "
            f"{_f(d.tpr_at_1pct)} | {_f(d.tpr_at_0p1pct)} | "
            f"{_f(d.adversarial_tpr_at_1pct)} | {_f(d.ece)} | "
            f"{d.abstention_rate:.1%} | {d.p95_latency_ms:.1f} | {d.n_samples} |")
    lines.append("")

    any_rob = any(d.tpr_by_perturbation for d in record.detector_results.values())
    if any_rob:
        names = sorted({p for d in record.detector_results.values()
                        for p in d.tpr_by_perturbation})
        lines.append("## Robustness — TPR@FPR=1% under perturbation\n")
        lines.append("| detector | " + " | ".join(names) + " |")
        lines.append("|---" * (len(names) + 1) + "|")
        for name in sorted(record.detector_results):
            d = record.detector_results[name]
            row = " | ".join(_f(d.tpr_by_perturbation.get(p)) for p in names)
            lines.append(f"| {d.detector} | {row} |")
        lines.append("")
        lines.append("`screenshot_recapture` and `print_recapture` are the two "
                     "cheapest laundering steps available to an adversary; a "
                     "detector that collapses under them is not deployable "
                     "against the threat model in spec §3A.\n")

    demoted = [d for d in record.detector_results.values()
               if d.adversarial_tpr_at_1pct is not None
               and d.adversarial_tpr_at_1pct < ADVERSARIAL_FLOOR]
    if demoted:
        lines.append("## Adversarial demotion\n")
        lines.append(
            f"Adversarial TPR below {ADVERSARIAL_FLOOR:.0%} under white-box PGD. "
            "Per spec §3A.4 these are **evidence-only** and must not decide:\n")
        for d in demoted:
            lines.append(f"- `{d.detector}` — adversarial TPR@FPR=1% "
                         f"{_f(d.adversarial_tpr_at_1pct)}")
        lines.append("")

    return "\n".join(lines)
