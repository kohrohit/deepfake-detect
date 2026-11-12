"""load_df40_records turns the ungated DF40 test split into benchmark records.

Every assertion here exists because the alternative silently flatters a
number: a guessed generator label would fabricate a LOGO axis this data
cannot support, a duplicated crop would let one image count many times, and
a real carrying a generator would be rejected by `bench.protocol` only after
the benchmark had already been run.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from bench.protocol import UnsplittableCorpusError, logo_splits
from bench.protocol import _validate as validate_corpus
from corpora.df40 import (
    ALIGN,
    DF40_GENERATOR,
    DUPLICATE,
    FACE_DETECTOR,
    NO_FACE,
    UNKNOWN_COMPRESSION,
    load_df40_records,
)
from dfd.faces import FaceBox


def _box() -> FaceBox:
    lms = np.array([[30.0, 40.0], [60.0, 40.0], [45.0, 55.0],
                    [33.0, 70.0], [57.0, 70.0]])
    return FaceBox(x=20, y=25, w=50, h=60, landmarks=lms, score=0.99)


def _detect_one(_frame):
    return [_box()]


def _write(root: Path, sub: str, names: list[str], *, seed: int = 0) -> None:
    d = root / sub
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for name in names:
        img = rng.integers(0, 255, (120, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(d / name), img)


def _corpus(root: Path, n_fake: int = 2, n_real: int = 2) -> None:
    _write(root, "fake", [f"f{i}.png" for i in range(n_fake)], seed=1)
    _corpus_reals(root, n_real)


def _corpus_reals(root: Path, n_real: int = 2) -> None:
    _write(root, "real", [f"r{i}.png" for i in range(n_real)], seed=2)


def test_fakes_are_labelled_1_and_reals_0(tmp_path: Path) -> None:
    _corpus(tmp_path)
    records, _ = load_df40_records(tmp_path, detect=_detect_one)
    by_label = {r["sample_id"]: r["label"] for r in records}
    assert sorted(k for k, v in by_label.items() if v == 1) == [
        "fake/f0.png", "fake/f1.png"]
    assert sorted(k for k, v in by_label.items() if v == 0) == [
        "real/r0.png", "real/r1.png"]


def test_reals_carry_no_generator_and_fakes_carry_the_unknown_mixture(
        tmp_path: Path) -> None:
    _corpus(tmp_path)
    records, _ = load_df40_records(tmp_path, detect=_detect_one)
    assert {r["generator"] for r in records if r["label"] == 1} == {
        DF40_GENERATOR}
    assert {r["generator"] for r in records if r["label"] == 0} == {None}


def test_the_corpus_cannot_support_a_leave_one_generator_out_split(
        tmp_path: Path) -> None:
    """The repackaging has no per-technique label, so LOGO must be refused.

    Not an incidental property: this is the reason the loader emits ONE
    generator value rather than a guess derived from filenames.
    """
    _corpus(tmp_path, n_fake=4, n_real=4)
    records, _ = load_df40_records(tmp_path, detect=_detect_one)
    with pytest.raises(UnsplittableCorpusError):
        logo_splits(records)


def test_every_record_declares_the_same_preprocessing(tmp_path: Path) -> None:
    _corpus(tmp_path)
    records, _ = load_df40_records(tmp_path, detect=_detect_one)
    assert {r["face_detector"] for r in records} == {FACE_DETECTOR}
    assert {r["align"] for r in records} == {ALIGN}


def test_compression_is_recorded_as_unknown_not_guessed(tmp_path: Path) -> None:
    _corpus(tmp_path)
    records, _ = load_df40_records(tmp_path, detect=_detect_one)
    assert {r["compression"] for r in records} == {UNKNOWN_COMPRESSION}


def test_an_image_is_no_longer_its_own_source(tmp_path: Path) -> None:
    """Replaced 2026-09-23 (was `test_each_image_is_its_own_source_and_subject`).

    That contract is what made every interval over this corpus too narrow:
    it declared 1,601 frames to be 1,601 independent sources when 999 of
    them come from one filename family. `source_id` is now the family.
    """
    _corpus(tmp_path)
    records, _ = load_df40_records(tmp_path, detect=_detect_one)
    for r in records:
        assert r["source_id"] != r["sample_id"]
        assert r["subject_id"] == r["source_id"]


def test_the_image_is_the_aligned_crop_not_the_source_frame(
        tmp_path: Path) -> None:
    _corpus(tmp_path, n_fake=1, n_real=1)
    records, _ = load_df40_records(tmp_path, detect=_detect_one, size=224)
    assert all(r["image"].shape == (224, 224, 3) for r in records)
    assert all(r["image"].dtype == np.uint8 for r in records)


def test_an_image_with_no_detected_face_is_counted_not_dropped_silently(
        tmp_path: Path) -> None:
    _corpus(tmp_path, n_fake=2, n_real=2)
    records, skipped = load_df40_records(tmp_path, detect=lambda _f: [])
    assert records == []
    assert skipped[NO_FACE] == 4


def test_a_crop_identical_to_one_already_taken_is_counted_as_duplicate(
        tmp_path: Path) -> None:
    """One DF40 image repeated must not vote twice.

    The capture corpus shipped 979 copies of a single demo asset; the same
    defect in an evaluation set inflates n and narrows every interval.
    """
    d = tmp_path / "fake"
    d.mkdir(parents=True)
    img = np.random.default_rng(3).integers(0, 255, (120, 100, 3),
                                            dtype=np.uint8)
    cv2.imwrite(str(d / "a.png"), img)
    cv2.imwrite(str(d / "b.png"), img)
    _write(tmp_path, "real", ["r0.png"], seed=2)
    records, skipped = load_df40_records(tmp_path, detect=_detect_one)
    assert sorted(r["sample_id"] for r in records) == ["fake/a.png",
                                                       "real/r0.png"]
    assert skipped[DUPLICATE] == 1


def test_a_limit_takes_a_balanced_sample(tmp_path: Path) -> None:
    _corpus(tmp_path, n_fake=10, n_real=10)
    records, _ = load_df40_records(tmp_path, detect=_detect_one, limit=6)
    assert len(records) == 6
    assert sum(r["label"] for r in records) == 3


def test_the_same_seed_takes_the_same_sample_and_a_different_seed_does_not(
        tmp_path: Path) -> None:
    _corpus(tmp_path, n_fake=20, n_real=20)
    a, _ = load_df40_records(tmp_path, detect=_detect_one, limit=8, seed=0)
    b, _ = load_df40_records(tmp_path, detect=_detect_one, limit=8, seed=0)
    c, _ = load_df40_records(tmp_path, detect=_detect_one, limit=8, seed=1)
    ids = lambda rs: sorted(r["sample_id"] for r in rs)  # noqa: E731
    assert ids(a) == ids(b)
    assert ids(a) != ids(c)


def test_a_root_without_both_label_directories_is_refused(
        tmp_path: Path) -> None:
    _write(tmp_path, "fake", ["f0.png"])
    with pytest.raises(FileNotFoundError, match="real"):
        load_df40_records(tmp_path, detect=_detect_one)


def test_two_images_differing_only_in_extension_get_distinct_ids(
        tmp_path: Path) -> None:
    """DF40's test/fake holds both `0.png` and `0.jpg`, and they differ.

    A stem-based id collides on those 20 pairs. `bench.runner` catches the
    collision and refuses the run — which is the good case; the bad case is a
    downstream consumer that de-duplicates by id and silently drops half of a
    filename family.
    """
    d = tmp_path / "fake"
    d.mkdir(parents=True)
    rng = np.random.default_rng(7)
    cv2.imwrite(str(d / "0.png"),
                rng.integers(0, 255, (120, 100, 3), dtype=np.uint8))
    cv2.imwrite(str(d / "0.jpg"),
                rng.integers(0, 255, (120, 100, 3), dtype=np.uint8))
    _write(tmp_path, "real", ["r0.png"], seed=2)
    records, _ = load_df40_records(tmp_path, detect=_detect_one)
    ids = sorted(r["sample_id"] for r in records if r["label"] == 1)
    assert ids == ["fake/0.jpg", "fake/0.png"]


# --- source grouping -------------------------------------------------------
#
# Added 2026-09-23, after measuring what this corpus's rows are actually worth.
# Treating each image as its own source (what this loader did until now) says
# the test split's 1,601 fakes are 1,601 independent observations. They are
# not: 999 of them share the filename family `000000_x_0_*`, and the Kish
# effective sample size over filename families is **2.4**. Every confidence
# interval computed over rows was therefore roughly sqrt(1601/2.4) ~ 26x too
# narrow on the fake side.

def test_frames_of_one_family_share_one_source_id(tmp_path: Path) -> None:
    _write(tmp_path, "fake", ["000000_x_0_100.png", "000000_x_0_101.png"], seed=1)
    _write(tmp_path, "real", ["00000000_10.png", "00000000_11.png"], seed=2)
    records, _ = load_df40_records(tmp_path, detect=_detect_one)
    by_id = {r["sample_id"]: r["source_id"] for r in records}
    assert len(by_id) == 4, "sample ids must stay distinct"
    assert by_id["fake/000000_x_0_100.png"] == "fake/000000_x_0"
    assert by_id["fake/000000_x_0_101.png"] == "fake/000000_x_0"
    assert by_id["real/00000000_10.png"] == "real/00000000"
    assert by_id["real/00000000_11.png"] == "real/00000000"
    assert by_id["fake/000000_x_0_100.png"] != by_id["real/00000000_10.png"]


def test_two_different_families_do_not_merge(tmp_path: Path) -> None:
    """The over-merging direction is safe but not free.

    Without this, a `source_group` that collapsed EVERY name into one bucket
    would satisfy `test_frames_of_one_family_share_one_source_id` — merging
    is what that test asks for. Merging distinct families throws away real
    independence, so the split must survive too.
    """
    _write(tmp_path, "fake", ["000000_x_0_100.png", "000001_x_0_100.png"], seed=1)
    _corpus_reals(tmp_path)
    records, _ = load_df40_records(tmp_path, detect=_detect_one)
    fakes = {r["source_id"] for r in records if r["label"] == 1}
    assert len(fakes) == 2, fakes


def test_one_family_name_under_both_labels_is_two_sources(tmp_path: Path) -> None:
    """`bench.protocol` refuses a source carrying two generator values.

    DF40's two halves use overlapping filename families — 226 names appear
    under both `fake/` and `real/` in the test split — so a group id built
    from the name alone would straddle the label boundary and take the whole
    run down with it.
    """
    _write(tmp_path, "fake", ["000_1.png", "000_2.png"], seed=1)
    _write(tmp_path, "real", ["000_1.png", "000_2.png"], seed=2)
    records, _ = load_df40_records(tmp_path, detect=_detect_one)
    assert len({r["source_id"] for r in records}) == 2
    validate_corpus(records)


def test_a_parenthesised_index_is_a_frame_index_too(tmp_path: Path) -> None:
    _write(tmp_path, "fake", ["7 (1).png", "7 (2).png"], seed=1)
    _corpus_reals(tmp_path)
    records, _ = load_df40_records(tmp_path, detect=_detect_one)
    fakes = {r["sample_id"]: r["source_id"] for r in records if r["label"] == 1}
    # The VALUE, not just its uniqueness: a `source_group` that recognised no
    # index at all would drop both names into the `unnumbered` bucket and
    # satisfy a uniqueness-only assertion while stripping nothing.
    assert set(fakes.values()) == {"fake/7"}, fakes


def test_a_name_with_no_frame_index_falls_into_one_bucket(tmp_path: Path) -> None:
    """Conservative on purpose: unparseable names merge rather than split.

    `00042.png` and `00099.png` carry no separator, so nothing in the name
    says whether they are two frames of one video or two unrelated stills.
    Merging them understates the evidence; splitting them would overstate it,
    and only one of those two errors flatters the benchmark.
    """
    _write(tmp_path, "fake", ["00042.png", "00099.png"], seed=1)
    _corpus_reals(tmp_path)
    records, _ = load_df40_records(tmp_path, detect=_detect_one)
    fakes = {r["source_id"] for r in records if r["label"] == 1}
    assert len(fakes) == 1, fakes


def test_digits_with_no_separator_are_not_read_as_a_frame_index(tmp_path: Path) -> None:
    """`shot7.png` is not frame 7 of a video called `shot`.

    Without this, dropping the separator requirement from `_FRAME_INDEX`
    would split two unparseable names into two source groups — the unsafe
    direction, because it claims independence the names do not establish.
    """
    _write(tmp_path, "fake", ["shot7.png", "clip9.png"], seed=1)
    _corpus_reals(tmp_path)
    records, _ = load_df40_records(tmp_path, detect=_detect_one)
    fakes = {r["source_id"] for r in records if r["label"] == 1}
    assert fakes == {"fake/unnumbered"}, fakes


def test_the_subject_tracks_the_source_so_the_protocol_accepts_the_corpus(
        tmp_path: Path) -> None:
    """A source that straddles subjects is refused by `bench.protocol`.

    Grouping source_id while leaving subject_id per-image would make every
    multi-frame family straddle, and `validate_corpus` would refuse the whole
    run — so the two must move together.
    """
    _write(tmp_path, "fake", ["000000_x_0_100.png", "000000_x_0_101.png"], seed=1)
    _write(tmp_path, "real", ["00000000_10.png", "00000000_11.png"], seed=2)
    records, _ = load_df40_records(tmp_path, detect=_detect_one)
    validate_corpus(records)  # raises if a source straddles
    for r in records:
        assert r["subject_id"] == r["source_id"]
