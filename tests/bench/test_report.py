from dataclasses import replace

from bench.report import render_markdown
from bench.runner import DetectorResult, RunRecord


def _record():
    return RunRecord(
        seed=7, dataset_hash="a" * 64, guards_enforced=True,
        model_versions={"synth_a": "synthetic-1"},
        identity_report=None,
        detector_results={
            "synth_a": DetectorResult(
                detector="synth_a", auc=0.83, auc_ci=(0.71, 0.92),
                tpr_at_1pct=0.42, tpr_at_0p1pct=0.19, ece=0.06,
                adversarial_tpr_at_1pct=0.03,
                abstention_rate=0.10, p95_latency_ms=42.0,
                n_samples=40),
        },
    )


def test_report_contains_the_reproducibility_record():
    md = render_markdown(_record())
    assert "seed" in md.lower()
    assert "a" * 64 in md


def test_report_shows_tpr_at_fpr_not_accuracy():
    md = render_markdown(_record())
    assert "TPR@FPR=1%" in md
    assert "accuracy" not in md.lower()


def test_report_shows_adversarial_column():
    md = render_markdown(_record())
    assert "adversarial" in md.lower()


def test_report_flags_a_detector_defeated_by_adversarial_attack():
    md = render_markdown(_record())
    assert "evidence-only" in md.lower()


def test_report_shows_confidence_intervals():
    md = render_markdown(_record())
    assert "0.71" in md and "0.92" in md


def _logo_record():
    base = _record()
    def _dr(auc):
        return DetectorResult(
            detector="synth_a", auc=auc, auc_ci=(auc - 0.1, auc + 0.1),
            tpr_at_1pct=0.2, tpr_at_0p1pct=0.1, ece=0.05,
            adversarial_tpr_at_1pct=0.03, abstention_rate=0.0,
            p95_latency_ms=1.0, n_samples=12)
    return replace(base, logo_results={
        "deepfacelive": {"synth_a": _dr(0.77)},
        "faceswap": {"synth_a": _dr(0.51)},
    })


def test_report_leads_with_the_worst_held_out_generator():
    """Spec §8.1. The mean of 0.77 and 0.51 is 0.64; reporting that would
    hide the generator an attacker would actually choose."""
    md = render_markdown(_logo_record())
    assert "0.510" in md
    assert "0.640" not in md
    assert md.index("Leave-one-generator-out") < md.index("In-dataset")


def test_report_labels_whole_corpus_numbers_as_memorisation():
    md = render_markdown(_logo_record())
    assert "memorisation" in md.lower()


def test_report_says_so_when_logo_was_not_computed():
    """A missing LOGO number must be stated, not left as a silent absence
    that reads as though the in-dataset table were the result."""
    md = render_markdown(_record())
    assert "not computed" in md.lower()
