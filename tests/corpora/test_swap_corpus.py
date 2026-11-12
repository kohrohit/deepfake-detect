"""The corpus builder: couples, strata, and the properties LOGO depends on.

Hermetic — a synthetic session tree and an injected detector, so no FairFace
and no weight file.
"""
import json

import cv2
import numpy as np
import pytest
from corpora.swap_corpus import CROP_SIZE, build_swap_corpus
from corpora.swaps import TECHNIQUES
from dfd.faces import FaceBox

RACES = ["White", "Black", "Indian", "East Asian"]


def _sessions(tmp_path, n=8, *, demographics=True):
    root = tmp_path / "sessions"
    root.mkdir()
    for i in range(n):
        folder = root / f"ff-{i:04d}"
        folder.mkdir()
        img = np.random.default_rng(i).integers(40, 215, (200, 200, 3), np.uint8)
        cv2.imwrite(str(folder / "frame_00.jpg"), img)
        meta = {"session_id": folder.name, "swapped": False}
        if demographics:
            # Two races x two genders, so pairing has partners in each.
            meta["demographics"] = {"race": RACES[(i // 4) % 2],
                                    "gender": "Male" if i % 2 else "Female",
                                    "age": "20-29"}
        (folder / "results.json").write_text(json.dumps(meta))
    return root


def _detect(frame):
    lms = np.array([[75.0, 85.0], [125.0, 85.0], [100.0, 110.0],
                    [80.0, 135.0], [120.0, 135.0]])
    return [FaceBox(x=60, y=60, w=80, h=80, landmarks=lms, score=0.9)]


def test_each_couple_yields_two_reals_and_one_fake_per_technique(tmp_path):
    records, skipped = build_swap_corpus(_sessions(tmp_path, 8), detect=_detect)

    couples = {r["subject_id"] for r in records}
    assert couples, f"no couples built; skipped {skipped}"
    for couple in couples:
        rows = [r for r in records if r["subject_id"] == couple]
        assert sum(1 for r in rows if r["label"] == 0) == 2
        fakes = {r["generator"] for r in rows if r["label"] == 1}
        assert fakes == set(TECHNIQUES), f"{couple} produced {fakes}"


def test_the_corpus_carries_enough_generators_for_logo(tmp_path):
    """The whole point. Every other corpus here has one generator, so
    `logo_splits` refuses it and the criterion 3 number has never existed."""
    from bench.protocol import logo_splits

    records, _ = build_swap_corpus(_sessions(tmp_path, 8), detect=_detect)
    splits = logo_splits(records, seed=0)

    assert len(splits) == len(TECHNIQUES)
    assert {s.held_out_generator for s in splits} == set(TECHNIQUES)
    for split in splits:
        assert split.train and split.test_ids()


def test_a_couple_never_straddles_a_split(tmp_path):
    """A swap carries the SOURCE's identity in the TARGET's photograph, so
    filing either person alone would put one identity on both sides."""
    from bench.protocol import logo_splits

    records, _ = build_swap_corpus(_sessions(tmp_path, 8), detect=_detect)
    by_id = {r["sample_id"]: r["subject_id"] for r in records}

    for split in logo_splits(records, seed=0):
        train = {by_id[r["sample_id"]] for r in split.train}
        test = {by_id[s] for s in split.test_ids()}
        assert not (train & test), f"couple on both sides of {split.held_out_generator}"


def test_both_members_of_a_couple_share_one_stratum(tmp_path):
    """Criterion 11 needs an unambiguous stratum per record, and pairing
    across strata would make a couple's stratum a coin flip."""
    records, _ = build_swap_corpus(_sessions(tmp_path, 8), detect=_detect)

    for couple in {r["subject_id"] for r in records}:
        strata = {r["stratum"] for r in records if r["subject_id"] == couple}
        assert len(strata) == 1, f"{couple} spans {strata}"


def test_a_session_without_demographics_is_counted_not_silently_dropped(tmp_path):
    """It would otherwise shrink criterion 11's denominator invisibly."""
    records, skipped = build_swap_corpus(
        _sessions(tmp_path, 8, demographics=False), detect=_detect)

    assert records == []
    assert skipped["no_stratum"] == 8


def test_every_record_is_cropped_to_the_same_size(tmp_path):
    """A preprocessing difference correlated with the label is the defect this
    whole corpus exists to avoid."""
    records, _ = build_swap_corpus(_sessions(tmp_path, 8), detect=_detect)

    assert records
    for r in records:
        assert r["image"].shape == (CROP_SIZE, CROP_SIZE, 3)
        assert r["image"].dtype == np.uint8
        assert r["face_detector"] == "yunet" and r["align"] == "v1"


def test_source_id_is_not_aliased_to_subject_id(tmp_path):
    """`check_video_level` refuses a corpus where groups equal sample_ids, and
    a subject-level source_id would make six records look like one source."""
    records, _ = build_swap_corpus(_sessions(tmp_path, 8), detect=_detect)

    assert len({r["source_id"] for r in records}) == len(records)


def test_the_same_seed_builds_the_same_corpus(tmp_path):
    root = _sessions(tmp_path, 8)
    a, _ = build_swap_corpus(root, detect=_detect, seed=3)
    b, _ = build_swap_corpus(root, detect=_detect, seed=3)

    assert [r["sample_id"] for r in a] == [r["sample_id"] for r in b]
    assert all(np.array_equal(x["image"], y["image"]) for x, y in zip(a, b))


def test_a_different_seed_pairs_people_differently(tmp_path):
    root = _sessions(tmp_path, 16)
    a, _ = build_swap_corpus(root, detect=_detect, seed=1)
    b, _ = build_swap_corpus(root, detect=_detect, seed=2)

    assert {r["subject_id"] for r in a} != {r["subject_id"] for r in b}


def test_a_missing_root_says_so_rather_than_returning_an_empty_corpus(tmp_path):
    with pytest.raises(FileNotFoundError, match="no FairFace session directory"):
        build_swap_corpus(tmp_path / "nope", detect=_detect)


def test_a_subset_of_techniques_still_folds(tmp_path):
    """A caller wanting a two-generator corpus must get one that LOGO accepts."""
    from bench.protocol import logo_splits

    records, _ = build_swap_corpus(_sessions(tmp_path, 8), detect=_detect,
                                   techniques=TECHNIQUES[:2])

    assert {r["generator"] for r in records if r["label"] == 1} == set(TECHNIQUES[:2])
    assert len(logo_splits(records, seed=0)) == 2


def test_a_face_that_will_not_detect_is_counted(tmp_path):
    records, skipped = build_swap_corpus(_sessions(tmp_path, 8),
                                         detect=lambda f: [])
    assert records == []
    assert skipped["no_face"] == 8


def test_offset_skips_the_first_photographs(tmp_path):
    """`blend_seam` was fitted on the first 10,000 FairFace sessions, so a
    corpus built without an offset evaluates it on its own training data —
    which is how a worst-generator AUC of 0.923 was reported once before
    anyone checked."""
    root = _sessions(tmp_path, 16)

    early, _ = build_swap_corpus(root, limit=8, detect=_detect)
    late, _ = build_swap_corpus(root, limit=8, offset=8, detect=_detect)

    def people(records):
        return {p for r in records for p in r["subject_id"].split("+")}

    assert people(early) and people(late)
    assert not (people(early) & people(late)), "offset did not move the window"
    assert people(late) <= {f"ff-{i:04d}" for i in range(8, 16)}


# --- Synthetic-source fakes, and their matched control ------------------

def _synth_pool(tmp_path, n=4):
    """Stand-in SFHQ images: 1024px square, as part 3 actually ships."""
    d = tmp_path / "sfhq"
    d.mkdir()
    out = []
    for i in range(n):
        img = np.random.default_rng(100 + i).integers(
            40, 215, (1024, 1024, 3), np.uint8)
        p = d / f"SFHQ_pt3_{i:08d}.jpg"
        cv2.imwrite(str(p), img)
        out.append(p)
    return out


def test_synthetic_sources_add_both_the_fake_and_its_control(tmp_path):
    """Neither label may appear without the other.

    `SYNTH_SFHQ` alone cannot be read: SFHQ is 1024px and FairFace is 224px,
    so the composite downsamples ~3x and a detector separating it from real
    may be reading the generator OR the resample. `SYNTH_CONTROL` runs a
    photographic face through the identical path, so the gap between them is
    the only interpretable quantity.
    """
    from corpora.swaps import SYNTH_CONTROL, SYNTH_SFHQ

    records, _ = build_swap_corpus(
        _sessions(tmp_path, 8), detect=_detect,
        synthetic_sources=_synth_pool(tmp_path))

    generators = {r["generator"] for r in records if r["label"] == 1}
    assert SYNTH_SFHQ in generators
    assert SYNTH_CONTROL in generators
    # One of each per couple, exactly as the four techniques get one each.
    couples = {r["subject_id"] for r in records}
    for label in (SYNTH_SFHQ, SYNTH_CONTROL):
        made = [r for r in records if r["generator"] == label]
        assert {r["subject_id"] for r in made} == couples


def test_the_control_is_not_emitted_without_synthetic_sources(tmp_path):
    """A control with nothing to control for is a fake with no purpose, and
    it would enter LOGO as a generator on its own."""
    from corpora.swaps import SYNTH_CONTROL, SYNTH_SFHQ

    records, _ = build_swap_corpus(_sessions(tmp_path, 8), detect=_detect)
    generators = {r["generator"] for r in records if r["label"] == 1}
    assert SYNTH_SFHQ not in generators
    assert SYNTH_CONTROL not in generators
    assert generators == set(TECHNIQUES)


def test_synthetic_fakes_are_labelled_fake_and_carry_the_couple(tmp_path):
    """They must split with their couple. A synthetic face has no real
    identity, but the PHOTOGRAPH it lands in does, and filing the record
    anywhere else puts that identity on both sides of a split."""
    from corpora.swaps import SYNTH_CONTROL, SYNTH_SFHQ

    records, _ = build_swap_corpus(
        _sessions(tmp_path, 8), detect=_detect,
        synthetic_sources=_synth_pool(tmp_path))
    made = [r for r in records
            if r["generator"] in (SYNTH_SFHQ, SYNTH_CONTROL)]
    assert made
    for r in made:
        assert r["label"] == 1
        assert "+" in r["subject_id"]
        assert r["stratum"]
        assert r["image"].shape == (CROP_SIZE, CROP_SIZE, 3)


def test_the_synthetic_pair_differs_only_in_source_provenance(tmp_path):
    """The whole design rests on this: both go through the same compositing
    technique into the same target, so the ONLY difference is where the
    pasted face came from. If the two images were identical the control
    would be vacuous; if they differed in geometry it would not be a
    control."""
    from corpora.swaps import SYNTH_CONTROL, SYNTH_SFHQ

    records, _ = build_swap_corpus(
        _sessions(tmp_path, 8), detect=_detect,
        synthetic_sources=_synth_pool(tmp_path))
    by_couple = {}
    for r in records:
        if r["generator"] in (SYNTH_SFHQ, SYNTH_CONTROL):
            by_couple.setdefault(r["subject_id"], {})[r["generator"]] = r

    pairs = [v for v in by_couple.values() if len(v) == 2]
    assert pairs, "no couple produced both halves of the pair"
    for v in pairs:
        a, b = v[SYNTH_SFHQ]["image"], v[SYNTH_CONTROL]["image"]
        assert a.shape == b.shape
        assert not np.array_equal(a, b)


def test_an_unusable_synthetic_pool_emits_neither_label_and_says_why(tmp_path):
    """A control emitted without its fake would enter LOGO as a generator in
    its own right, and a reader would take it for a result."""
    from corpora.swap_corpus import SYNTH_UNUSABLE
    from corpora.swaps import SYNTH_CONTROL, SYNTH_SFHQ

    bad = tmp_path / "bad"
    bad.mkdir()
    junk = bad / "not-an-image.jpg"
    junk.write_bytes(b"this is not a JPEG")

    records, skipped = build_swap_corpus(
        _sessions(tmp_path, 8), detect=_detect, synthetic_sources=[junk])

    generators = {r["generator"] for r in records if r["label"] == 1}
    assert SYNTH_SFHQ not in generators
    assert SYNTH_CONTROL not in generators
    assert skipped.get(SYNTH_UNUSABLE, 0) > 0
