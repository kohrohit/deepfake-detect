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


# rec()'s source_id defaults to sample_id, so every record in RECORDS is
# already its own source: no two of them ever share one to straddle. That
# makes test_no_source_video_straddles_the_split above pass even against a
# splitter that assigns per record instead of per subject -- there is
# nothing in RECORDS for such a bug to straddle. This fixture gives one
# subject three frames of a single source video, so guard 2's actual
# dimension (several samples, one video) is exercised.
VIDEO_RECORDS = RECORDS + [
    rec("v1", "pF", "deepfacelive", 1, source_id="videoF"),
    rec("v2", "pF", "deepfacelive", 1, source_id="videoF"),
    rec("v3", "pF", "deepfacelive", 1, source_id="videoF"),
]


@pytest.mark.parametrize("seed", SEEDS)
def test_a_multi_frame_source_video_lands_wholly_on_one_side(seed):
    """Guard 2, exercised for real: three frames of ONE video, not three
    single-frame videos each of which is its own source."""
    for s in logo_splits(VIDEO_RECORDS, seed=seed):
        train_frames = {r["sample_id"] for r in s.train
                         if r["source_id"] == "videoF"}
        test_frames = {r["sample_id"] for r in s.test
                        if r["source_id"] == "videoF"}
        assert not (train_frames and test_frames), (
            f"{s.held_out_generator}: video 'videoF' straddled "
            f"train={train_frames} test={test_frames}")
        assert train_frames | test_frames == {"v1", "v2", "v3"}


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
    """Holding out the only generator leaves nothing to train on.

    `match` names the "has no train fakes" phrase specifically, not the
    bare substring "train fakes": `_require_measurable`'s message always
    embeds the full counts dict, and "train fakes" is one of that dict's
    keys regardless of which count is actually zero, so the bare substring
    would match any failure of this function, not just this one.
    """
    bad = [rec("r1", "p1", None, 0), rec("r2", "p2", None, 0),
           rec("f1", "pA", "deepfacelive", 1), rec("f2", "pB", "deepfacelive", 1)]
    with pytest.raises(ValueError, match="has no train fakes"):
        logo_splits(bad)


def test_a_source_carrying_two_subjects_is_rejected():
    bad = [rec("a", "p1", None, 0, source_id="v"),
           rec("b", "p2", None, 0, source_id="v"),
           rec("f", "pA", "deepfacelive", 1)]
    with pytest.raises(ValueError, match="more than one subject/generator"):
        logo_splits(bad)


def test_a_label_outside_zero_or_one_is_rejected():
    bad = [rec("r1", "p1", None, 0),
           {"sample_id": "x", "subject_id": "pA", "source_id": "x",
            "generator": "deepfacelive", "label": 2}]
    with pytest.raises(ValueError, match="label"):
        logo_splits(bad)


def test_a_real_record_carrying_a_generator_is_rejected():
    """The schema says generator=None for reals. Without this check a real
    with a generator string passes validation and then contributes a
    distinct (subject, generator) pair, which can trigger a confusing
    'more than one subject/generator pair' rejection for an unrelated
    reason."""
    bad = [rec("r1", "p1", "deepfacelive", 0), rec("r2", "p2", None, 0)]
    with pytest.raises(ValueError, match="generator"):
        logo_splits(bad)


# Minimal corpus with a valid, zero-drop, identity-disjoint split for BOTH
# folds: g1: train {p1, pB} / test {p2, pA}; g2: train {p1, pA} / test
# {p2, pB}. pA and pB are real-less, single-generator subjects -- each is
# exactly the case the two vanishing-subject moves in logo_splits exist for.
REVIEWER_CORPUS = [
    rec("r1", "p1", None, 0), rec("r2", "p2", None, 0),
    rec("fA", "pA", "g1", 1), rec("fB", "pB", "g2", 1),
]


@pytest.mark.parametrize("seed", SEEDS)
def test_reviewer_corpus_splits_at_every_seed_with_zero_drops(seed):
    """Before the symmetric move (Finding 1), this corpus raised ValueError
    ('no test fakes') at every seed 0-7: the one-directional fix could push
    a real-less subject off test but had no mirror move able to push one
    onto it, so no seed's random partition could ever produce a fold with
    both a train fake and a test fake."""
    for s in logo_splits(REVIEWER_CORPUS, seed=seed):
        assert s.dropped_for_identity == []


@pytest.mark.parametrize("seed", SEEDS)
def test_a_subject_with_no_matching_generator_is_kept_on_train(seed):
    """Move 1: a real-less subject whose fakes never match the held-out
    generator would drop everything if left on test, so it is moved to (or
    kept on) train regardless of where the random partition first put it."""
    for s in logo_splits(REVIEWER_CORPUS, seed=seed):
        no_match = "pB" if s.held_out_generator == "g1" else "pA"
        assert no_match in s.train_subjects
        assert no_match not in s.test_subjects


@pytest.mark.parametrize("seed", SEEDS)
def test_a_subject_whose_only_generator_is_held_out_is_moved_to_test(seed):
    """Move 2, the mirror of the above: a real-less subject whose entire
    generator set is exactly the held-out generator would drop everything
    if left on train, so it is moved to (or kept on) test regardless of
    where the random partition first put it."""
    for s in logo_splits(REVIEWER_CORPUS, seed=seed):
        only_match = "pA" if s.held_out_generator == "g1" else "pB"
        assert only_match in s.test_subjects
        assert only_match not in s.train_subjects
