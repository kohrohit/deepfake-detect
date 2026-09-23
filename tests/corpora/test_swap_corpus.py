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
