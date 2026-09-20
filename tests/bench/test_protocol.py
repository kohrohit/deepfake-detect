import pytest

from bench.protocol import Split, logo_splits  # noqa: F401 -- Split is part of the public API this test exercises


def rec(sample_id, subject_id, generator, label, source_id=None):
    return {"sample_id": sample_id, "subject_id": subject_id,
            "source_id": source_id or sample_id,
            "generator": generator, "label": label}


# Six real subjects, and five fake subjects each faked by two or three
# generators. The repeated subjects are the point: a splitter that assigns
# fakes by generator alone leaks all five.
RECORDS = [rec(f"r{i}", f"p{i}", None, 0) for i in range(1, 7)] + [
    rec("f1", "pA", "deepfacelive", 1), rec("f2", "pA", "faceswap", 1),
    rec("f3", "pB", "deepfacelive", 1), rec("f4", "pB", "stylegan", 1),
    rec("f5", "pC", "faceswap", 1), rec("f6", "pC", "stylegan", 1),
    rec("f7", "pD", "deepfacelive", 1), rec("f8", "pD", "faceswap", 1),
    rec("f9", "pD", "stylegan", 1),
    rec("f10", "pE", "deepfacelive", 1), rec("f11", "pE", "faceswap", 1),
    rec("f12", "pE", "stylegan", 1),
]

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]


def test_one_split_per_generator():
    splits = logo_splits(RECORDS)
    assert {s.held_out_generator for s in splits} == {
        "deepfacelive", "faceswap", "stylegan"}
    assert len(splits) == 3


@pytest.mark.parametrize("seed", SEEDS)
def test_held_out_generator_never_appears_in_train(seed):
    for s in logo_splits(RECORDS, seed=seed):
        gens = {r["generator"] for r in s.train if r["label"] == 1}
        assert s.held_out_generator not in gens


@pytest.mark.parametrize("seed", SEEDS)
def test_held_out_generator_is_the_only_fake_generator_in_test(seed):
    for s in logo_splits(RECORDS, seed=seed):
        gens = {r["generator"] for r in s.test if r["label"] == 1}
        assert gens == {s.held_out_generator}


@pytest.mark.parametrize("seed", SEEDS)
def test_both_sides_carry_reals_and_fakes(seed):
    """No test reals means no FPR; no test fakes means no TPR."""
    for s in logo_splits(RECORDS, seed=seed):
        for side in (s.train, s.test):
            assert any(r["label"] == 0 for r in side)
            assert any(r["label"] == 1 for r in side)


@pytest.mark.parametrize("seed", SEEDS)
def test_subjects_are_disjoint_across_the_split_including_fake_subjects(seed):
    """Spec §8.2 guard 1. RECORDS fakes each subject with several
    generators, so a by-generator-only assignment fails this."""
    for s in logo_splits(RECORDS, seed=seed):
        tr = {r["subject_id"] for r in s.train}
        te = {r["subject_id"] for r in s.test}
        assert tr.isdisjoint(te), f"{s.held_out_generator}: leaked {tr & te}"


@pytest.mark.parametrize("seed", SEEDS)
def test_no_source_video_straddles_the_split(seed):
    """Spec §8.2 guard 2: a source video belongs wholly to one side."""
    for s in logo_splits(RECORDS, seed=seed):
        tr = {r["source_id"] for r in s.train}
        te = {r["source_id"] for r in s.test}
        assert tr.isdisjoint(te)


@pytest.mark.parametrize("seed", SEEDS)
def test_every_record_is_placed_or_reported_dropped(seed):
    """Nothing vanishes silently."""
    for s in logo_splits(RECORDS, seed=seed):
        placed = s.train_ids() + s.test_ids() + s.dropped_ids()
        assert sorted(placed) == sorted(r["sample_id"] for r in RECORDS)
        assert len(placed) == len(set(placed))


def test_dropped_records_are_exactly_the_identity_conflicts():
    """Each drop is a fake whose generator wants the side its subject is not
    on. On this corpus that is a large fraction, which is why it is counted."""
    for s in logo_splits(RECORDS, seed=0):
        test_subjects = {r["subject_id"] for r in s.test}
        for r in s.dropped_for_identity:
            assert r["label"] == 1
            wants_test = r["generator"] == s.held_out_generator
            assert wants_test != (r["subject_id"] in test_subjects)
        assert len(s.dropped_for_identity) > 0


def test_split_is_deterministic_given_a_seed():
    a = logo_splits(RECORDS, seed=5)
    b = logo_splits(RECORDS, seed=5)
    assert [s.test_ids() for s in a] == [s.test_ids() for s in b]


def test_seed_actually_changes_the_partition():
    """Guards against an implementation that accepts `seed` and ignores it."""
    by_seed = {tuple(tuple(s.test_ids()) for s in logo_splits(RECORDS, seed=k))
               for k in SEEDS}
    assert len(by_seed) > 1


def test_records_without_a_generator_key_are_rejected():
    bad = [{"sample_id": "x", "subject_id": "p", "source_id": "x", "label": 1}]
    with pytest.raises(KeyError, match="missing required"):
        logo_splits(bad)


def test_records_without_a_source_id_are_rejected():
    bad = [{"sample_id": "x", "subject_id": "p", "generator": "g", "label": 1}]
    with pytest.raises(KeyError, match="missing required"):
        logo_splits(bad)


def test_a_fake_with_no_generator_is_rejected():
    """An unattributed fake would silently join the training side of EVERY
    split, which is the one place it can never be measured."""
    bad = [rec("r1", "p1", None, 0), rec("f1", "pA", None, 1)]
    with pytest.raises(ValueError, match="unattributed fake"):
        logo_splits(bad)


def test_a_corpus_with_one_subject_is_rejected():
    """Better to refuse than to emit a split with nothing on one side."""
    bad = [rec("r1", "pA", None, 0), rec("f1", "pA", "deepfacelive", 1)]
    with pytest.raises(ValueError, match="needs at least 2"):
        logo_splits(bad)


def test_a_single_generator_corpus_is_rejected():
    """Holding out the only generator leaves nothing to train on."""
    bad = [rec("r1", "p1", None, 0), rec("r2", "p2", None, 0),
           rec("f1", "pA", "deepfacelive", 1), rec("f2", "pB", "deepfacelive", 1)]
    with pytest.raises(ValueError, match="train fakes"):
        logo_splits(bad)


def test_a_source_carrying_two_subjects_is_rejected():
    bad = [rec("a", "p1", None, 0, source_id="v"),
           rec("b", "p2", None, 0, source_id="v"),
           rec("f", "pA", "deepfacelive", 1)]
    with pytest.raises(ValueError, match="more than one subject/generator"):
        logo_splits(bad)
