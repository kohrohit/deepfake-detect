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


def test_report_shows_the_seed_value():
    """The previous test only checked the word "seed" appeared, which a
    renderer printing `- seed: \\`0\\`` — the wrong value — would still
    satisfy. Acceptance criterion 10 is the reproducibility ARTIFACT, so
    the actual seed value must be in it."""
    md = render_markdown(_record())
    assert "`7`" in md


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


def test_report_shows_the_logo_drop_count_beside_each_fold():
    """`Split.dropped_for_identity` is a property of the fold (subject and
    generator geometry), not of which detector scored it. Without
    rendering it, a reader cannot tell a fold that dropped 2 records from
    one that dropped 20 — both would show the same AUC and n_samples."""
    rec = replace(_logo_record(), logo_dropped={"deepfacelive": 3, "faceswap": 5})
    md = render_markdown(rec)
    assert "dropped 3" in md.lower()
    assert "dropped 5" in md.lower()


def test_report_says_so_when_logo_was_not_computed():
    """A missing LOGO number must be stated, not left as a silent absence
    that reads as though the in-dataset table were the result."""
    md = render_markdown(_record())
    assert "not computed" in md.lower()


def _logo_record_with_a_detector_missing_from_one_fold():
    """Two detectors overall, but one fold never scored `synth_b` — the
    shape `worst_logo_auc` already defends against (`if detector in f`).
    """
    base = _record()
    base = replace(base, detector_results={
        **base.detector_results,
        "synth_b": DetectorResult(
            detector="synth_b", auc=0.55, auc_ci=(0.45, 0.65),
            tpr_at_1pct=0.15, tpr_at_0p1pct=0.05, ece=0.04,
            adversarial_tpr_at_1pct=None, abstention_rate=0.0,
            p95_latency_ms=2.0, n_samples=40),
    })

    def _dr(auc):
        return DetectorResult(
            detector="synth_a", auc=auc, auc_ci=(auc - 0.1, auc + 0.1),
            tpr_at_1pct=0.2, tpr_at_0p1pct=0.1, ece=0.05,
            adversarial_tpr_at_1pct=0.03, abstention_rate=0.0,
            p95_latency_ms=1.0, n_samples=12)
    return replace(base, logo_results={
        "deepfacelive": {"synth_a": _dr(0.77)},           # synth_b absent
        "faceswap": {"synth_a": _dr(0.51), "synth_b": _dr(0.40)},
    })


def test_report_handles_a_detector_missing_from_one_logo_fold():
    """`worst_logo_auc` already treats a detector missing from a fold as
    unmeasured rather than raising `KeyError` (spec: a fold that never
    scored a detector is unmeasured for it, not zero). Building each row's
    cells must fail the same way, or a bold worst-AUC renders and the same
    row then crashes building itself, per the reviewer's finding."""
    md = render_markdown(_logo_record_with_a_detector_missing_from_one_fold())
    assert "synth_b" in md
    assert "n/a" in md.lower()


def _record_with_robustness():
    base = _record()
    dr = replace(base.detector_results["synth_a"],
                tpr_by_perturbation={"clean": 0.5, "jpeg_q10": 0.1,
                                     "screenshot_recapture": 0.2})
    return replace(base, detector_results={"synth_a": dr})


def test_report_shows_the_robustness_table_when_populated():
    """The table must carry the actual sweep column names and values, not
    just announce its own existence — a renderer that printed the heading
    with an empty body would still satisfy a weaker check."""
    md = render_markdown(_record_with_robustness())
    assert "Robustness" in md
    after_heading = md[md.index("## Robustness"):]
    lines = after_heading.splitlines()
    header_line = next(line for line in lines if line.startswith("| detector |"))
    for col in ("clean", "jpeg_q10", "screenshot_recapture"):
        assert col in header_line
    row_line = next(line for line in lines if line.startswith("| synth_a |"))
    assert "0.500" in row_line
    assert "0.100" in row_line
    assert "0.200" in row_line


def test_report_omits_the_robustness_table_when_nothing_was_measured():
    """The `any_rob` gate must close as well as open. Every detector in
    `_record()` has the default `tpr_by_perturbation == {}` (robustness was
    never run), so the section — heading, table, and caption — must not
    appear at all. A gate that never closes and prints an empty table would
    misreport an unmeasured sweep as one that ran and found nothing."""
    md = render_markdown(_record())
    assert "Robustness" not in md
    assert "screenshot_recapture" not in md


def test_identity_line_does_not_fabricate_a_pair_count():
    """`n_train * n_test` is the denominator for a train/test check and not
    for the corpus-level subject check, which compares n*(n-1)/2 subject
    pairs. The same report type carries both, so the renderer must print the
    RATE the guard computed rather than multiply the two sides itself."""
    from bench.guards import IdentityReport
    from bench.report import render_markdown
    from bench.runner import RunRecord

    rec = RunRecord(
        seed=0, dataset_hash="h", guards_enforced=False, model_versions={},
        identity_report=IdentityReport(
            n_train=125, n_test=125, max_similarity=0.99, violations=419,
            threshold=0.363, violation_rate=419 / 7750, tolerated_rate=1.0),
        identity_status="violation")

    md = render_markdown(rec)

    assert "419 crossings" in md
    assert "5.4065%" in md
    assert "15625" not in md, "printed a pair count nobody computed"


# --- Abstention by class ------------------------------------------------

def _skewed_logo_record():
    """A fold whose held-out fakes abstain far more often than reals do.

    The shape measured on the swap corpus 2026-09-24: `blend_seam` abstains
    on 74.3% of `swap_lowres_paste` fakes against 56.9% of reals, and that
    fold carries the best AUC in the report. Every metric beside it is
    computed over the survivors, so the AUC describes the quarter of that
    technique's fakes the quality floor let through.
    """
    base = _record()

    def _dr(auc, by_class):
        return DetectorResult(
            detector="synth_a", auc=auc, auc_ci=(auc - 0.1, auc + 0.1),
            tpr_at_1pct=0.2, tpr_at_0p1pct=0.1, ece=0.05,
            adversarial_tpr_at_1pct=0.03, abstention_rate=0.6,
            p95_latency_ms=1.0, n_samples=100,
            abstention_by_class=by_class)

    return replace(base, logo_results={
        "lowres": {"synth_a": _dr(0.93, {"real": (57, 100), "lowres": (76, 100)})},
        "poisson": {"synth_a": _dr(0.64, {"real": (57, 100), "poisson": (56, 100)})},
    })


def test_report_shows_abstention_per_class_beside_the_logo_folds():
    """An AUC computed over survivors means less when survival is
    label-correlated. The rate that produced it must render beside it."""
    md = render_markdown(_skewed_logo_record())
    assert "76.0%" in md          # the held-out fakes
    assert "57.0%" in md          # the reals they are compared against


def test_report_does_not_invent_an_abstention_breakdown_when_none_was_measured():
    """An empty `abstention_by_class` means "not measured". Rendering it as
    0% would read as "nothing abstained", which is the opposite claim."""
    md = render_markdown(_logo_record())
    assert "abstention by class" not in md.lower()
