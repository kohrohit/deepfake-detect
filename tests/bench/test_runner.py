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


def _embeddings(records, *, leak: tuple[str, str] | None = None):
    """One orthogonal unit vector per sample — different people by construction.

    `leak` makes two named samples the same person, which is what the guard
    must catch and what a declared-subject-id split cannot see.
    """
    n = len(records)
    eye = np.eye(n * 2)
    emb = {r["sample_id"]: eye[i] for i, r in enumerate(records)}
    if leak is not None:
        emb[leak[0]] = emb[leak[1]]
    return emb


def test_identity_criterion_is_unmeasured_without_embeddings():
    """An unmeasured criterion and a clean one must never read alike.

    This was the state of the repo until 2026-09-23: `identity_report` was
    hardcoded None, which a reader could take either way.
    """
    rec = run_benchmark(_records(), _registry(), RunConfig(seed=7))
    assert rec.identity_report is None
    assert rec.identity_status == "not_measured"


def test_identity_criterion_is_measured_when_embeddings_are_supplied():
    records = _records()
    rec = run_benchmark(records, _registry(),
                        RunConfig(seed=7, identity_embeddings=_embeddings(records)))

    assert rec.identity_status == "ok"
    assert rec.identity_report is not None
    assert rec.identity_report.violations == 0
    # Per fold, not one aggregate: a corpus with two generators has two folds.
    assert set(rec.identity_by_fold) == {"deepfacelive", "faceswap"}
    # And it actually compared pairs — a report over zero pairs proves nothing.
    assert rec.identity_report.n_train > 0 and rec.identity_report.n_test > 0


def test_identity_leakage_across_a_fold_raises_while_guards_are_enforced():
    """The declared split is clean; the FACES are not. That is the whole point."""
    records = _records()
    cfg = RunConfig(seed=7)
    from bench.runner import _logo_splits_or_none

    splits = _logo_splits_or_none(records, cfg)
    split = splits[0]
    train_id = split.train[0]["sample_id"]
    test_id = split.test_ids()[0]
    emb = _embeddings(records, leak=(train_id, test_id))

    with pytest.raises(GuardViolation, match="identity leakage"):
        run_benchmark(records, _registry(),
                      RunConfig(seed=7, identity_embeddings=emb))


def test_identity_leakage_is_recorded_rather_than_raised_when_guards_are_waived():
    """A waived guard must leave evidence, not a blank that reads as clean."""
    records = _records()
    cfg = RunConfig(seed=7)
    from bench.runner import _logo_splits_or_none

    split = _logo_splits_or_none(records, cfg)[0]
    emb = _embeddings(records,
                      leak=(split.train[0]["sample_id"], split.test_ids()[0]))

    rec = run_benchmark(records, _registry(),
                        RunConfig(seed=7, enforce_guards=False,
                                  identity_embeddings=emb))
    assert rec.identity_status == "violation"
    assert rec.identity_report is None


def test_identity_headline_is_the_worst_fold_not_the_average():
    """A reader who takes one number must take the pessimistic one.

    The leak is placed in a pair that crosses train/test in ONE fold only, so
    the folds carry different rates and a headline taking the best fold, the
    mean, or an arbitrary fold all differ from taking the worst. Leaking a
    pair that crosses in both folds would make every one of those choices
    agree, and the test would pass for an implementation that picked any of
    them.
    """
    records = _records()
    cfg = RunConfig(seed=7)
    from bench.runner import _logo_splits_or_none

    a, b = _logo_splits_or_none(records, cfg)
    train_a, test_a = {r["sample_id"] for r in a.train}, set(a.test_ids())
    train_b, test_b = {r["sample_id"] for r in b.train}, set(b.test_ids())
    # Same SIDE in fold b, not merely a different side-assignment: the leak
    # is symmetric, so a pair that merely swaps sides still crosses b's
    # train x test product and both folds would score alike.
    def _same_side_in_b(x, y):
        return (x in train_b and y in train_b) or (x in test_b and y in test_b)

    pair = next((x, y) for x in sorted(train_a) for y in sorted(test_a)
                if _same_side_in_b(x, y))
    emb = _embeddings(records, leak=pair)

    rec = run_benchmark(records, _registry(),
                        RunConfig(seed=7, identity_embeddings=emb,
                                  identity_max_false_match_rate=1.0))

    rates = [r.violation_rate for r in rec.identity_by_fold.values()]
    assert max(rates) > min(rates), "fixture must separate the folds"
    assert rec.identity_report is not None
    assert rec.identity_report.violation_rate == pytest.approx(max(rates))


def test_parity_is_not_measured_on_a_corpus_without_strata():
    rec = run_benchmark(_records(), _registry(), RunConfig(seed=7))
    assert rec.parity_by_detector == {}
    assert rec.parity_status == "no_strata"


def test_parity_excludes_strata_too_small_to_compare():
    """Rule of three: zero false positives in 6 genuine samples bounds the true
    FPR at ~50%, so comparing that stratum's 0.00 against another's 0.14 fires
    the guard on arithmetic rather than on bias."""
    records = _records()
    for i, r in enumerate(records):
        r["stratum"] = ["East Asian", "White", "Black"][(i // 2) % 3]

    rec = run_benchmark(records, _registry(), RunConfig(seed=7))

    assert rec.parity_status == "too_few_per_stratum"
    assert rec.parity_by_detector == {}
    # And it says which strata and how small, rather than going quiet.
    assert set(rec.parity_excluded_strata) == {"East Asian", "White", "Black"}
    assert all(0 < n < 150 for n in rec.parity_excluded_strata.values())


def test_parity_is_measured_when_the_strata_are_large_enough():
    """Acceptance criterion 11. FairFace sessions carry age/gender/race.

    Read at a 20% operating point, not the default 1%. That is not a
    convenience: at a 1% FPR over a 40-row fixture the threshold sits above
    every genuine score in one stratum, so both rates round to zero and the
    ratio is arithmetic rather than measurement — the same effect
    `parity_min_genuine_per_stratum` exists to keep out of real runs. A unit
    test cannot conjure the hundreds of genuine samples per stratum a 1%
    comparison needs, so it moves the operating point instead of pretending.
    """
    records = _records()
    for i, r in enumerate(records):
        r["stratum"] = ["East Asian", "White"][(i // 2) % 2]

    rec = run_benchmark(records, _registry(),
                        RunConfig(seed=7, parity_at_fpr=0.2,
                                  parity_min_genuine_per_stratum=5,
                                  parity_max_fpr_ratio=10.0))

    assert rec.parity_status == "ok"
    assert set(rec.parity_by_detector) == {"synth_a", "synth_b"}
    report = rec.parity_by_detector["synth_a"]
    assert set(report.fpr_by_stratum) == {"East Asian", "White"}
    assert report.ceiling == pytest.approx(10.0)
    assert rec.parity_excluded_strata == {}
    # Rates were actually computed, not defaulted: at a 20% operating point
    # over 20 genuine samples a stratum cannot legitimately be empty.
    assert all(0.0 <= v <= 1.0 for v in report.fpr_by_stratum.values())
    assert max(report.fpr_by_stratum.values()) > 0.0


def test_parity_violation_raises_while_guards_are_enforced():
    """One stratum rejected far more often than another must stop the run."""
    from dfd.types import RawScore

    records = _records()
    for i, r in enumerate(records):
        r["stratum"] = "A" if (i // 2) % 2 else "B"
        # The detector cannot see a stratum, so the fixture encodes it in the
        # payload's height and the stub reads that.
        r["image"] = np.random.default_rng(i).integers(
            0, 255, (128 if r["stratum"] == "A" else 144,
                     161 if r["label"] == 1 else 160, 3), dtype=np.uint8)

    class _Biased:
        name = "biased"
        version = "0.0.1"

        def score(self, observations):
            obs = observations[0]
            biased_up = obs.payload.shape[0] == 144
            # Fakes score 1.0 — above both genuine clusters — so the quantile
            # over ALL scores lands in a different place from the quantile
            # over genuine scores alone. Fixing an FPR means reading the
            # genuine distribution; a threshold taken from the mixture is a
            # different and much looser operating point, and this fixture is
            # what makes that substitution visible.
            if obs.payload.shape[1] == 161:
                return RawScore(detector="biased", version="0.0.1", score=1.0,
                                abstained=False, reason="ok")
            return RawScore(detector="biased", version="0.0.1",
                            score=0.99 if biased_up else 0.01,
                            abstained=False, reason="ok")

    reg = Registry()
    reg.register(_Biased())
    # Read at the MEDIAN genuine score. The stub emits two values, so with
    # ten genuine rows per stratum the median falls exactly between them:
    # every genuine row of one stratum is above the threshold and none of the
    # other's is. That is a 1.0-vs-0.0 disparity, which is what this guard is
    # for, and it is arithmetic rather than luck — the assertion cannot pass
    # for a run that computed nothing.
    with pytest.raises(GuardViolation, match="disparity"):
        run_benchmark(records, reg,
                      RunConfig(seed=7, parity_at_fpr=0.5,
                                parity_max_fpr_ratio=1.5,
                                parity_min_genuine_per_stratum=5))


def test_adversarial_is_not_requested_by_default_and_says_so():
    """A None adversarial TPR must never read as 'the attack succeeded'."""
    rec = run_benchmark(_records(), _registry(), RunConfig(seed=7))
    assert set(rec.adversarial_status.values()) == {"not_requested"}
    assert all(r.adversarial_tpr_at_1pct is None
               for r in rec.detector_results.values())


def test_adversarial_reports_no_target_for_a_handcrafted_detector():
    """Every detector in this repo is features plus a linear model: there is
    no differentiable path from pixels, and that is a property to state
    rather than a measurement to fake."""
    rec = run_benchmark(_records(), _registry(), RunConfig(seed=7, adversarial=True))
    assert set(rec.adversarial_status.values()) == {"no_target"}


def test_adversarial_reports_weights_absent_separately_from_no_target():
    """A detector that COULD be attacked but has no weights is a different
    state from one that never could be."""
    class _Unloaded(SyntheticDetector):
        def adversarial_target(self):
            return None

    reg = Registry()
    reg.register(_Unloaded(name="unloaded", seed=1))
    rec = run_benchmark(_records(), reg, RunConfig(seed=7, adversarial=True))
    assert rec.adversarial_status == {"unloaded": "weights_absent"}


def test_adversarial_tpr_is_measured_for_a_detector_that_exposes_a_target():
    """Acceptance criterion 8, end to end over the real PGD loop."""
    import torch

    class _Tiny(torch.nn.Module):
        """Two logits from the mean pixel — differentiable, and attackable."""

        def forward(self, x):
            m = x.mean(dim=(1, 2, 3), keepdim=False) * 10.0
            return torch.stack([-m, m], dim=1)

    class _Attackable(SyntheticDetector):
        def adversarial_target(self):
            def to_row(obs):
                img = obs.payload.astype("float32") / 255.0
                t = torch.from_numpy(img).permute(2, 0, 1)
                return torch.nn.functional.interpolate(
                    t[None], size=(16, 16), mode="bilinear")[0]
            return _Tiny(), to_row

    reg = Registry()
    reg.register(_Attackable(name="attackable", seed=1))
    rec = run_benchmark(_records(), reg,
                        RunConfig(seed=7, adversarial=True, adversarial_eps=0.1))

    assert rec.adversarial_status == {"attackable": "ok"}
    tpr = rec.detector_results["attackable"].adversarial_tpr_at_1pct
    assert tpr is not None
    assert 0.0 <= tpr <= 1.0

    # And the attack DID something. STRICTLY less, not `<=`: an
    # implementation that ignores the configured budget and always attacks
    # at eps=0 returns the clean number, which passes any non-strict
    # comparison. Measured on this stub: eps=0 gives 0.05, eps>=0.03 gives
    # 0.00, so the gap is real and the assertion can fail.
    clean = run_benchmark(_records(), reg,
                          RunConfig(seed=7, adversarial=True, adversarial_eps=0.0))
    clean_tpr = clean.detector_results["attackable"].adversarial_tpr_at_1pct
    assert clean_tpr is not None
    assert tpr < clean_tpr, (
        f"attack at eps=0.1 gave {tpr}, no better than the clean {clean_tpr}")
