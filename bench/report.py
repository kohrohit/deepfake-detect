"""Markdown rendering of a benchmark run.

Deliberately prints no accuracy figure. At fraud base rates accuracy is a
number that always looks good and never means anything.
"""
from __future__ import annotations

from .runner import RunRecord, worst_logo_auc

ADVERSARIAL_FLOOR = 0.10


def _f(x) -> str:
    return "n/a" if x is None or x != x else f"{x:.3f}"


def render_markdown(record: RunRecord) -> str:
    lines: list[str] = []
    lines.append("# Benchmark run\n")
    lines.append("## Reproducibility record\n")
    lines.append(f"- seed: `{record.seed}`")
    lines.append(f"- dataset hash: `{record.dataset_hash}`")
    lines.append(f"- guards enforced: `{record.guards_enforced}`")
    versions = ", ".join(f"{k}={v}" for k, v in sorted(record.model_versions.items()))
    lines.append(f"- model versions: `{versions}`")
    if record.identity_report is not None:
        r = record.identity_report
        lines.append(f"- identity disjointness: max cosine `{r.max_similarity:.4f}` "
                     f"at threshold `{r.threshold}`, {r.violations} violations")
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
