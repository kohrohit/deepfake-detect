import numpy as np
import logging

import pytest
from bench.guards import GuardViolation
from bench.metrics import auc, bootstrap_ci_by_group
from bench.runner import (
    RunConfig, _observation, dataset_hash, run_benchmark, worst_logo_auc,
)
from dfd.detectors.base import Registry, SyntheticDetector


def _records(n=40):
    out = []
    for i in range(n):
        fake = i % 2 == 0
        out.append({
            "sample_id": f"s{i}",
            "subject_id": f"p{i}",
            # Each image is its own source. Recorded explicitly, never aliased
            # to sample_id: the moment video records arrive, several samples
            # share one source_id and the video-level guard must still bite.
            "source_id": f"src{i}",
            # TWO generators, because leave-one-generator-out is undefined
            # with one: holding out the only generator leaves nothing to
            # train on, and logo_splits refuses such a corpus outright.
            "generator": ["deepfacelive", "faceswap"][(i // 2) % 2] if fake else None,
            "label": 1 if fake else 0,
            "compression": ["c0", "c23", "c40"][i % 3],
            "face_detector": "yunet",
            "align": "v1",
            # Non-square and varying, and at least 128px on the short side.
            # Both matter. A uniform fixture SHAPE hid a crash through every
            # test in Task 14. And a 64x64 image measures quality band
            # "reject", below SyntheticDetector's "low" floor, so the whole
            # corpus abstains: AUC, CI, TPR and ECE all come back nan and the
            # runner's entire metric path goes untested while the suite looks
            # green.
            "image": np.random.default_rng(i).integers(
                0, 255, (128 + (i % 3) * 16, 160 + (i % 5) * 16, 3),
                dtype=np.uint8),
        })
    return out


def _registry():
    reg = Registry()
    reg.register(SyntheticDetector(name="synth_a", seed=1))
    reg.register(SyntheticDetector(name="synth_b", seed=2))
    return reg


def test_run_produces_one_result_per_detector():
    rec = run_benchmark(_records(), _registry(), RunConfig(seed=7))
    assert set(rec.detector_results) == {"synth_a", "synth_b"}


def test_run_records_seed_and_dataset_hash_for_reproducibility():
    """Spec acceptance criterion 10."""
    rec = run_benchmark(_records(), _registry(), RunConfig(seed=7))
    assert rec.seed == 7
    assert len(rec.dataset_hash) == 64


def test_dataset_hash_is_stable():
    a = _records()
    assert dataset_hash(a) == dataset_hash(a)


@pytest.mark.parametrize("field_name,value", [
    ("sample_id", "changed"),
    ("subject_id", "changed"),
    ("source_id", "changed"),
    ("generator", "changed"),
    ("label", 0),
    ("compression", "c99"),
    ("face_detector", "retinaface"),
    ("align", "v2"),
])
def test_dataset_hash_is_sensitive_to_every_identifying_field(field_name, value):
    """Mutating one field was one field's worth of evidence. The hash is what
    ties an audit record to the corpus it was computed on.

    `face_detector` and `align` are included because `check_uniform_preprocessing`
    treats either varying as grounds to abort the run — two runs aligned on
    every other field but differing here would otherwise hash identically
    while producing different metrics, and this test was previously unable
    to discover that: it asserts a completeness property while enumerating
    only the fields the implementation already covered.
    """
    a, b = _records(), _records()
    b[0][field_name] = value
    assert dataset_hash(a) != dataset_hash(b)


def test_run_records_model_versions():
    rec = run_benchmark(_records(), _registry(), RunConfig(seed=7))
    assert rec.model_versions["synth_a"] == "synthetic-1"


def test_run_records_latency_per_detector():
    """Spec acceptance criterion 5: recorded, not optimised.

    `>= 0.0` would be satisfied by a stub that never measures anything and
    returns 0.0. Scoring 40 records takes real time, so require it.
    """
    rec = run_benchmark(_records(), _registry(), RunConfig(seed=7))
    assert rec.detector_results["synth_a"].p95_latency_ms > 0.0


def test_guards_run_by_default_and_fail_the_run():
    recs = _records()
    for r in recs:
        r["compression"] = "c23"          # violates compression coverage
    with pytest.raises(GuardViolation, match="compression"):
        run_benchmark(recs, _registry(), RunConfig(seed=7))


def test_guards_can_be_waived_only_explicitly():
    recs = _records()
    for r in recs:
        r["compression"] = "c23"
    rec = run_benchmark(recs, _registry(), RunConfig(seed=7, enforce_guards=False))
    assert rec.guards_enforced is False


def test_a_detector_that_abstains_on_everything_reports_rate_one():
    """The previous form asserted only `0.0 <= rate <= 1.0`, which any value
    satisfies — while its own docstring named the property it failed to test."""
    reg = Registry()
    reg.register(SyntheticDetector(name="picky", seed=1, min_quality_band="high"))
    rec = run_benchmark(_records(), reg, RunConfig(seed=7, enforce_guards=False))
    result = rec.detector_results["picky"]
    assert result.abstention_rate == 1.0
    assert result.auc != result.auc          # nan: nothing was scored


def test_a_detector_that_abstains_on_nothing_reports_rate_zero():
    rec = run_benchmark(_records(), _registry(), RunConfig(seed=7))
    assert rec.detector_results["synth_a"].abstention_rate == 0.0


def test_logo_results_exist_for_every_generator():
    """Spec 8.1. Without this the harness reports only in-dataset AUC, which
    the spec describes as measuring memorisation."""
    rec = run_benchmark(_records(), _registry(), RunConfig(seed=7))
    assert set(rec.logo_results) == {"deepfacelive", "faceswap"}
    for folds in rec.logo_results.values():
        assert set(folds) == {"synth_a", "synth_b"}


def test_a_logo_fold_scores_only_its_held_out_generator():
    """The fold must be a strict subset of the corpus, or it is not held out
    at all — a fold silently scoring everything would report in-dataset
    numbers under a LOGO heading, which is worse than reporting neither."""
    records = _records()
    rec = run_benchmark(records, _registry(), RunConfig(seed=7))
    # Without this, an empty logo_results passes by never entering the loop.
    assert len(rec.logo_results) == 2
    for folds in rec.logo_results.values():
        n = folds["synth_a"].n_samples
        assert 0 < n < len(records)


def test_logo_and_in_dataset_numbers_are_reported_separately():
    """They must not be the same object or the same number by construction."""
    rec = run_benchmark(_records(), _registry(), RunConfig(seed=7))
    assert rec.detector_results["synth_a"].n_samples == 40
    # `all(...)` over an empty dict is True, so the count is asserted first.
    assert len(rec.logo_results) == 2
    assert all(f["synth_a"].n_samples < 40 for f in rec.logo_results.values())


def test_worst_logo_auc_takes_the_minimum_not_the_mean():
    """Spec 8.2 guard 3 reports the WORST compression cell for the same
    reason: an average over generators hides the one an attacker will use."""
    rec = run_benchmark(_records(), _registry(), RunConfig(seed=7))
    per_fold = [f["synth_a"].auc for f in rec.logo_results.values()]
    assert worst_logo_auc(rec, "synth_a") == min(per_fold)


def test_a_single_generator_corpus_reports_no_logo_rather_than_failing(caplog):
    """LOGO is undefined with one generator. The in-dataset numbers are still
    valid, so the run degrades rather than raising."""
    records = _records()
    for r in records:
        if r["label"] == 1:
            r["generator"] = "deepfacelive"
    with caplog.at_level(logging.WARNING):
        rec = run_benchmark(records, _registry(),
                            RunConfig(seed=7, enforce_guards=False))
    assert rec.logo_results == {}
    assert "LOGO unavailable" in caplog.text
    assert rec.detector_results["synth_a"].n_samples == 40


def test_a_malformed_corpus_raises_out_of_run_benchmark_rather_than_warning():
    """A straddling source_id, a label=2 record, an unattributed fake, or a
    real carrying a generator are corpus DEFECTS (bench/protocol.py raises
    plain ValueError for these), not a legitimately unsplittable corpus. They
    must propagate out of run_benchmark, not be downgraded to the same
    'LOGO unavailable' warning an unsplittable-but-valid corpus gets — those
    two situations must stay distinguishable to a caller."""
    records = _records()
    records[0]["label"] = 2
    with pytest.raises(ValueError, match="label"):
        run_benchmark(records, _registry(), RunConfig(seed=7, enforce_guards=False))


def test_an_unsplittable_corpus_still_warns_and_returns_empty_logo_results(caplog):
    """The single-subject case: well-formed, just too small to split. This
    must still degrade-and-warn, not raise, distinguishing it from the
    malformed-corpus case above."""
    records = _records()
    for r in records:
        r["subject_id"] = "only_subject"
    with caplog.at_level(logging.WARNING):
        rec = run_benchmark(records, _registry(),
                            RunConfig(seed=7, enforce_guards=False))
    assert rec.logo_results == {}
    assert "LOGO unavailable" in caplog.text


def test_run_is_reproducible_given_a_seed():
    a = run_benchmark(_records(), _registry(), RunConfig(seed=7))
    b = run_benchmark(_records(), _registry(), RunConfig(seed=7))
    assert (a.detector_results["synth_a"].auc
            == b.detector_results["synth_a"].auc)
    assert a.dataset_hash == b.dataset_hash
    # Folds are drawn from a seeded permutation; same seed, same folds.
    assert ({g: f["synth_a"].auc for g, f in a.logo_results.items()}
            == {g: f["synth_a"].auc for g, f in b.logo_results.items()})


def test_bootstrap_ci_is_computed_over_source_groups_not_rows():
    """The one wiring decision this task was corrected for: the `groups`
    array fed into `bootstrap_ci_by_group` must be `source_id`, never
    `sample_id`. Resampling rows instead of videos fabricates precision
    (measured at 11.9x too narrow elsewhere in this codebase).

    The stock `_records()` fixture gives every record its own `source_id`,
    so `sample_id` and `source_id` are the same partition there and a
    regression to `sample_id` would pass every other test in this file
    silently. This fixture pairs samples onto shared `source_id`s so the
    two groupings actually differ, and uses `enforce_guards=False` because
    `check_video_level` correctly refuses a corpus shaped this way — which
    is exactly the path an honest interval matters most on.
    """
    records = _records()
    for i, r in enumerate(records):
        r["source_id"] = f"grp{i // 2}"          # 20 groups of 2 samples
    # A source may carry only one (subject_id, generator) pair (bench/
    # protocol.py's straddle check, now enforced even when guards are off —
    # that check guards corpus validity, not evaluation hygiene). The stock
    # fixture alternates fake/real by index, so naive pairing straddles.
    # Align each pair's subject_id/generator/label onto one of its two
    # records, alternating which record wins so the corpus keeps its
    # overall 20 fake / 20 real balance.
    for g in range(len(records) // 2):
        i, j = 2 * g, 2 * g + 1
        src, dst = (records[i], records[j]) if g % 2 == 0 else (records[j], records[i])
        dst["subject_id"] = src["subject_id"]
        dst["generator"] = src["generator"]
        dst["label"] = src["label"]

    reg = _registry()
    rec = run_benchmark(records, reg, RunConfig(seed=7, enforce_guards=False))

    # Recompute the score array independently (same detector, same
    # observations) to derive an expectation from each candidate grouping,
    # rather than trusting the runner's own intermediate values.
    det = reg.get("synth_a")
    scores = [det.score([_observation(r)]).score for r in records]
    s = np.array(scores, dtype=float)
    labels = np.array([r["label"] for r in records], dtype=int)
    source_groups = np.array([r["source_id"] for r in records])
    sample_groups = np.array([r["sample_id"] for r in records])

    expected_by_source = bootstrap_ci_by_group(s, labels, source_groups, auc,
                                               n=RunConfig().bootstrap_n, seed=7)
    expected_by_row = bootstrap_ci_by_group(s, labels, sample_groups, auc,
                                            n=RunConfig().bootstrap_n, seed=7)

    assert rec.detector_results["synth_a"].auc_ci == expected_by_source
    assert rec.detector_results["synth_a"].auc_ci != expected_by_row


def test_logo_fold_reports_measured_abstention_not_fabricated_zero():
    """The first cut of the LOGO fold slicer passed `latencies=[]` and
    `abstentions=0` literally, so every fold claimed 0% abstained and 0.0ms
    regardless of what happened in that slice. A detector that abstains on
    everything must show `abstention_rate == 1.0` in every fold — recovered
    from the slice itself — and an explicitly unmeasured (NaN) p95 latency,
    never a false `0.0` that reads as instantaneous.
    """
    reg = Registry()
    reg.register(SyntheticDetector(name="picky", seed=1, min_quality_band="high"))
    rec = run_benchmark(_records(), reg, RunConfig(seed=7, enforce_guards=False))
    assert rec.logo_results          # sanity: LOGO was computed for this corpus
    for folds in rec.logo_results.values():
        result = folds["picky"]
        assert result.abstention_rate == 1.0
        assert result.p95_latency_ms != result.p95_latency_ms  # NaN


def _identity_conflict_record(sample_id, subject_id, source_id, label,
                              generator, compression, image_seed):
    return {
        "sample_id": sample_id, "subject_id": subject_id,
        "source_id": source_id, "generator": generator, "label": label,
        "compression": compression, "face_detector": "yunet", "align": "v1",
        "image": np.random.default_rng(image_seed).integers(
            0, 255, (128, 160, 3), dtype=np.uint8),
    }


def _identity_conflict_fixture():
    """Two subjects, each with a real record and fakes from BOTH generators.

    A subject with a real record never moves sides for generator reasons
    (only subjects with no real record do, per `logo_splits`'s
    Pareto-improving move). So for every fold, exactly one of each
    subject's two fakes disagrees with its fixed side and is dropped —
    2 drops per fold here, independently confirmed against
    `bench.protocol.logo_splits` directly before being encoded as this
    fixture's expected value.
    """
    return [
        _identity_conflict_record("s1", "target", "t_real", 0, None, "c0", 1),
        _identity_conflict_record("s2", "target", "t_df", 1, "deepfacelive", "c23", 2),
        _identity_conflict_record("s3", "target", "t_fs", 1, "faceswap", "c40", 3),
        _identity_conflict_record("s4", "other", "o_real", 0, None, "c0", 4),
        _identity_conflict_record("s5", "other", "o_df", 1, "deepfacelive", "c23", 5),
        _identity_conflict_record("s6", "other", "o_fs", 1, "faceswap", "c40", 6),
    ]


def test_logo_dropped_counts_are_reported_and_nonzero_when_drops_occur():
    """`logo_splits` drops identity-conflicted fakes rather than leaking
    them (`Split.dropped_for_identity`). Without carrying that count into
    `RunRecord`, a reader seeing a fold's `n_samples` cannot tell whether
    2 or 20 records were removed to reach it."""
    rec = run_benchmark(_identity_conflict_fixture(), _registry(), RunConfig(seed=7))
    assert rec.logo_dropped == {"deepfacelive": 2, "faceswap": 2}


def test_logo_dropped_is_zero_for_a_corpus_with_no_identity_conflicts():
    """The stock fixture drops nothing (each subject carries exactly one
    generator), so this is the counterpart to the test above: the count
    must be present and zero, not merely absent."""
    rec = run_benchmark(_records(), _registry(), RunConfig(seed=7))
    assert rec.logo_dropped == {"deepfacelive": 0, "faceswap": 0}


def test_duplicate_sample_id_raises_rather_than_silently_mis_slicing_folds():
    """The `position` map inside the LOGO fold slicer maps sample_id ->
    row index; a duplicate collapses to one entry, so one row would be
    scored twice and another dropped from every fold with no error — and
    `0 < n < len(records)` still holds, so the fold-subset test elsewhere
    in this file would not notice. This must raise instead of mis-slicing.
    """
    records = _records()
    records[1]["sample_id"] = records[0]["sample_id"]
    with pytest.raises(ValueError, match="duplicate sample_id"):
        run_benchmark(records, _registry(), RunConfig(seed=7, enforce_guards=False))
