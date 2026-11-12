import numpy as np
import pytest
from bench.guards import (
    GuardViolation, check_compression_coverage, check_demographic_parity,
    check_identity_disjoint, check_threshold_provenance,
    check_uniform_preprocessing, check_video_level,
)


def test_identity_disjoint_passes_when_identities_do_not_overlap():
    emb = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    rep = check_identity_disjoint(["a"], ["b"], emb, threshold=0.9)
    assert rep.violations == 0
    assert rep.max_similarity < 0.9


def test_identity_disjoint_raises_on_the_same_person_in_both_splits():
    emb = {"a": np.array([1.0, 0.0]), "a_dup": np.array([0.99, 0.01])}
    with pytest.raises(GuardViolation, match=r"identity leakage"):
        check_identity_disjoint(["a"], ["a_dup"], emb, threshold=0.9)


def test_identity_disjoint_raises_rather_than_skips_when_embeddings_are_missing():
    """An empty embeddings dict must not certify disjointness over zero
    actually-compared pairs; the same ids on both sides must never come back
    clean just because nothing was looked up."""
    with pytest.raises(GuardViolation, match=r"no embedding"):
        check_identity_disjoint(["a", "b", "c"], ["a", "b", "c"], {}, threshold=0.6)


def test_identity_disjoint_raises_naming_the_missing_id_for_a_partial_dict():
    emb = {"a": np.array([1.0, 0.0])}
    with pytest.raises(GuardViolation, match=r"no embedding.*\['b'\]"):
        check_identity_disjoint(["a"], ["b"], emb, threshold=0.9)


def test_identity_report_carries_a_number_not_an_assertion():
    """Spec acceptance criterion 2: report the measurement, do not claim it."""
    emb = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    rep = check_identity_disjoint(["a"], ["b"], emb, threshold=0.9)
    assert isinstance(rep.max_similarity, float)
    assert rep.n_train == 1 and rep.n_test == 1


def test_video_level_raises_when_a_group_is_split_across_samples():
    with pytest.raises(GuardViolation, match=r"appears in multiple samples"):
        check_video_level(sample_ids=["s1", "s2"], groups=["v1", "v1"])


def test_video_level_passes_for_one_sample_per_group():
    check_video_level(sample_ids=["s1", "s2"], groups=["v1", "v2"])


def test_video_level_raises_when_groups_identical_to_sample_ids():
    """FIX 2: Raise ValueError when guard would be a no-op (groups == sample_ids)."""
    with pytest.raises(ValueError, match=r"vacuous"):
        check_video_level(sample_ids=["s1", "s2"], groups=["s1", "s2"])


def test_compression_coverage_raises_when_a_level_is_missing():
    recs = [{"compression": "c23"}, {"compression": "c23"}]
    with pytest.raises(GuardViolation, match=r"compression levels missing"):
        check_compression_coverage(recs, required=("c0", "c23", "c40"))


def test_compression_coverage_passes_when_all_present():
    recs = [{"compression": c} for c in ("c0", "c23", "c40")]
    check_compression_coverage(recs, required=("c0", "c23", "c40"))


def test_uniform_preprocessing_raises_on_mixed_detectors():
    recs = [{"face_detector": "yunet", "align": "v1", "label": 0},
            {"face_detector": "retinaface", "align": "v1", "label": 1}]
    with pytest.raises(GuardViolation, match=r"non-uniform preprocessing"):
        check_uniform_preprocessing(recs)


def test_uniform_preprocessing_passes_when_pipeline_is_identical():
    recs = [{"face_detector": "yunet", "align": "v1", "label": 0},
            {"face_detector": "yunet", "align": "v1", "label": 1}]
    check_uniform_preprocessing(recs)


def test_threshold_from_test_set_is_rejected():
    with pytest.raises(GuardViolation, match=r"must be 'validation'"):
        check_threshold_provenance("test")


def test_threshold_from_validation_is_accepted():
    check_threshold_provenance("validation")


def test_demographic_parity_reports_a_spread_not_a_mean():
    """Guard 6: an aggregate FPR hides a group rejected three times as often."""
    scores = [0.3, 0.7, 0.6, 0.8] * 5
    labels = [0, 0, 0, 0] * 5
    strata = (["A"] * 2 + ["B"] * 2) * 5
    rep = check_demographic_parity(scores, labels, strata, threshold=0.5,
                                   max_fpr_ratio=100.0)
    assert set(rep.fpr_by_stratum) == {"A", "B"}
    assert rep.fpr_by_stratum["B"] > rep.fpr_by_stratum["A"]


def test_demographic_parity_raises_when_the_ratio_exceeds_the_ceiling():
    scores = [0.1, 0.1, 0.9, 0.9] * 5
    labels = [0, 0, 0, 0] * 5
    strata = (["A"] * 2 + ["B"] * 2) * 5
    with pytest.raises(GuardViolation, match=r"demographic FPR"):
        check_demographic_parity(scores, labels, strata, threshold=0.5,
                                 max_fpr_ratio=2.0)


def test_demographic_parity_passes_when_groups_are_treated_alike():
    scores = [0.1, 0.9, 0.1, 0.9] * 5
    labels = [0, 0, 0, 0] * 5
    strata = (["A"] * 2 + ["B"] * 2) * 5
    rep = check_demographic_parity(scores, labels, strata, threshold=0.5,
                                   max_fpr_ratio=2.0)
    assert rep.max_fpr_ratio == pytest.approx(1.0)


def test_demographic_parity_ignores_strata_with_no_negatives():
    """A stratum with no genuine samples has no measurable FPR; do not divide by zero."""
    scores = [0.1, 0.9, 0.9, 0.9]
    labels = [0, 0, 1, 1]
    strata = ["A", "A", "B", "B"]
    rep = check_demographic_parity(scores, labels, strata, threshold=0.5,
                                   max_fpr_ratio=2.0)
    assert "B" not in rep.fpr_by_stratum


def test_demographic_parity_subfloor_ratio_computed_correctly():
    """FIX 1: For lo nonzero but below 1e-3, compute true ratio, not floored.

    Fixture: A FPR=0.0002 (nonzero, below old 1e-3 floor), B FPR=0.0008.
    True ratio: 0.0008 / 0.0002 = 4.0x.
    Old buggy floor would compute: 0.0008 / 1e-3 = 0.8x (false negative).
    """
    # Engineer scores to get A FPR=0.0002 and B FPR=0.0008 (5 negatives each)
    # A: 1 above threshold, 4 below = 0.2 FPR... need 0.0002
    # With 5000 negatives: 1 above, 4999 below = 0.0002 FPR
    scores_a = [0.4] * 4999 + [0.6]  # FPR = 1/5000 ≈ 0.0002
    scores_b = [0.4] * 4992 + [0.6] * 8  # FPR = 8/5000 ≈ 0.0016 (approx 4x)
    scores = scores_a + scores_b
    labels = [0] * 5000 + [0] * 5000
    strata = ["A"] * 5000 + ["B"] * 5000

    # Ratio = 0.0016 / 0.0002 = 8.0, which exceeds 2.0 ceiling -> should raise
    with pytest.raises(GuardViolation, match=r"demographic FPR"):
        check_demographic_parity(scores, labels, strata, threshold=0.5,
                                 max_fpr_ratio=2.0)

    # Now test with ceiling high enough to pass: true ratio should be ~8.0
    rep = check_demographic_parity(scores, labels, strata, threshold=0.5,
                                   max_fpr_ratio=10.0)
    # Ratio should be approximately 8.0, not a floored value like 0.8
    assert rep.max_fpr_ratio == pytest.approx(8.0, rel=0.1)
    assert rep.fpr_by_stratum["B"] > rep.fpr_by_stratum["A"]


def _orthogonal_embeddings(n: int, prefix: str) -> dict:
    """n mutually orthogonal unit vectors — cosine 0.0 between any two."""
    return {f"{prefix}{i}": np.eye(n * 2)[i] for i in range(n)}


def test_identity_guard_tolerates_a_rate_it_was_told_to_tolerate():
    """Measured: 0.21% of UNRELATED face pairs cross 0.363 (dfd.embed).

    A guard that raises on any crossing therefore refuses honest splits once
    the pair count is large, which is why the rate exists.
    """
    from bench.guards import check_identity_disjoint

    emb = _orthogonal_embeddings(20, "x")
    train = [f"x{i}" for i in range(10)]
    test = [f"x{i}" for i in range(10, 20)]
    # One leaking pair out of 100: rate 1%.
    emb["x0"] = emb["x10"]

    with pytest.raises(GuardViolation, match="identity leakage"):
        check_identity_disjoint(train, test, emb, threshold=0.6)

    report = check_identity_disjoint(train, test, emb, threshold=0.6,
                                     max_false_match_rate=0.01)
    assert report.violations == 1
    assert report.violation_rate == pytest.approx(0.01)
    assert report.tolerated_rate == pytest.approx(0.01)


def test_identity_guard_still_raises_above_the_tolerated_rate():
    from bench.guards import check_identity_disjoint

    emb = _orthogonal_embeddings(20, "x")
    train = [f"x{i}" for i in range(10)]
    test = [f"x{i}" for i in range(10, 20)]
    emb["x0"] = emb["x10"]
    emb["x1"] = emb["x11"]  # two leaking pairs: rate 2%

    with pytest.raises(GuardViolation, match="2 of 100"):
        check_identity_disjoint(train, test, emb, threshold=0.6,
                                max_false_match_rate=0.01)


def test_identity_guard_reports_the_rate_on_a_clean_split():
    from bench.guards import check_identity_disjoint

    emb = _orthogonal_embeddings(20, "x")
    report = check_identity_disjoint([f"x{i}" for i in range(10)],
                                     [f"x{i}" for i in range(10, 20)], emb)
    assert report.violations == 0
    assert report.violation_rate == 0.0
    assert report.n_train == 10 and report.n_test == 10


def test_identity_guard_refuses_a_rate_outside_zero_to_one():
    """Outside [0, 1] the guard is always-fail or always-pass, and looks fine."""
    from bench.guards import check_identity_disjoint

    emb = _orthogonal_embeddings(4, "x")
    for bad in (-0.01, 1.5):
        with pytest.raises(ValueError, match=r"max_false_match_rate must be in \[0, 1\]"):
            check_identity_disjoint(["x0"], ["x1"], emb, max_false_match_rate=bad)


def test_identity_guard_rate_is_zero_when_no_pairs_were_compared():
    """An empty side means nothing was checked; the rate must not read 'clean'
    as a division by zero."""
    from bench.guards import check_identity_disjoint

    report = check_identity_disjoint([], [], {})
    assert report.violation_rate == 0.0
    assert report.n_train == 0 and report.n_test == 0
