import numpy as np
import logging

import pytest
from bench.guards import GuardViolation
from bench.runner import (
    RunConfig, dataset_hash, run_benchmark, worst_logo_auc,
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
])
def test_dataset_hash_is_sensitive_to_every_identifying_field(field_name, value):
    """Mutating one field was one field's worth of evidence. The hash is what
    ties an audit record to the corpus it was computed on."""
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


def test_run_is_reproducible_given_a_seed():
    a = run_benchmark(_records(), _registry(), RunConfig(seed=7))
    b = run_benchmark(_records(), _registry(), RunConfig(seed=7))
    assert (a.detector_results["synth_a"].auc
            == b.detector_results["synth_a"].auc)
    assert a.dataset_hash == b.dataset_hash
    # Folds are drawn from a seeded permutation; same seed, same folds.
    assert ({g: f["synth_a"].auc for g, f in a.logo_results.items()}
            == {g: f["synth_a"].auc for g, f in b.logo_results.items()})
