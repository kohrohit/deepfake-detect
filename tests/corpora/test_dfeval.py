from __future__ import annotations

import numpy as np
import pytest

from corpora.dfeval import (
    CONTRADICTORY_DUPLICATE,  # noqa: F401 -- part of the module's public interface
    DFEVAL_GENERATOR,
    DUPLICATE,
    load_dfeval_images,
)


def test_records_satisfy_bench_protocol_validation(dfeval_root, stub_detect):
    # _validate is strict about three things this loader can get wrong:
    # labels outside {0,1}, an unattributed fake, and a real carrying a
    # generator. Catching them here beats catching them inside a fold.
    from bench.protocol import _validate

    records, _ = load_dfeval_images(dfeval_root, detect=stub_detect)
    _validate(records)


def test_every_fake_carries_the_single_unknown_mixture_generator(
        dfeval_root, stub_detect):
    records, _ = load_dfeval_images(dfeval_root, detect=stub_detect)
    fakes = [r for r in records if r["label"] == 1]
    assert fakes
    assert {r["generator"] for r in fakes} == {DFEVAL_GENERATOR}
    assert all(r["generator"] is None for r in records if r["label"] == 0)


def test_logo_refuses_this_corpus_gracefully(dfeval_root, stub_detect):
    # The whole point of one honest generator id rather than None: holding
    # out the only generator leaves no training fakes, so logo_splits raises
    # UnsplittableCorpusError -- which a caller may catch and record -- and
    # NOT the plain ValueError that means a malformed corpus.
    from bench.protocol import UnsplittableCorpusError, logo_splits

    records, _ = load_dfeval_images(dfeval_root, detect=stub_detect)
    with pytest.raises(UnsplittableCorpusError):
        logo_splits(records)


def test_the_domain_is_carried_as_the_resampling_family(dfeval_root, stub_detect):
    records, _ = load_dfeval_images(dfeval_root, detect=stub_detect)
    assert all(r["domain"] for r in records)
    assert len({r["domain"] for r in records}) > 1


def test_identical_pixels_are_deduplicated_and_counted(dfeval_root_with_dupe,
                                                       stub_detect):
    # This repo has already read 58 distinct images as a pool of 1,088. An
    # in-the-wild corpus assembled from social media is the place that
    # happens again: one image circulates across many domains.
    records, skipped = load_dfeval_images(dfeval_root_with_dupe,
                                          detect=stub_detect)
    assert skipped[DUPLICATE] == 1
    arrays = [r["image"].tobytes() for r in records]
    assert len(set(arrays)) == len(arrays)


def test_a_duplicate_spanning_both_labels_is_refused_not_dropped(
        dfeval_root_with_contradiction, stub_detect):
    # The same pixels labelled real in one place and fake in another is a
    # labelling contradiction. Dropping one silently picks a winner; the
    # corpus has to be refused and the hash named.
    with pytest.raises(ValueError, match="contradictory"):
        load_dfeval_images(dfeval_root_with_contradiction, detect=stub_detect)


def test_images_with_no_face_are_KEPT_and_flagged(dfeval_root_mixed, stub_detect_sometimes):
    # Dropping them would hide the gap §6 of the spec exists to measure:
    # this corpus is in-the-wild AI media and not all of it is facial, while
    # every detector here needs a face. A loader that filtered them would
    # make the abstention accounting impossible and quietly flatter us.
    records, skipped = load_dfeval_images(dfeval_root_mixed,
                                          detect=stub_detect_sometimes)
    assert any(not r["has_face"] for r in records)
    assert all(r["roi"] is None for r in records if not r["has_face"])
    assert "no_face" not in skipped


def test_a_faced_record_carries_a_real_roi_and_real_landmarks(dfeval_root,
                                                              stub_detect):
    records, _ = load_dfeval_images(dfeval_root, detect=stub_detect)
    faced = [r for r in records if r["has_face"]]
    assert faced
    for r in faced:
        x, y, w, h = r["roi"]
        assert w >= 2 and h >= 2
        assert np.asarray(r["landmarks"]).shape[1] == 2


def test_preprocessing_fields_are_uniform(dfeval_root, stub_detect):
    # check_uniform_preprocessing fails a corpus whose face_detector or
    # align varies, and it is right to: applying different preprocessing by
    # label is itself a giveaway the model will happily learn.
    from bench.guards import check_uniform_preprocessing

    records, _ = load_dfeval_images(dfeval_root, detect=stub_detect)
    check_uniform_preprocessing(records)


def test_compression_is_recorded_unknown_not_invented(dfeval_root, stub_detect):
    # corpora.df40 and corpora.sfhq set the precedent: an invented c23 label
    # passes check_compression_coverage by lying to it.
    records, _ = load_dfeval_images(dfeval_root, detect=stub_detect)
    assert {r["compression"] for r in records} == {"unknown"}


def test_limit_is_applied_deterministically(dfeval_root, stub_detect):
    a, _ = load_dfeval_images(dfeval_root, limit=4, seed=0, detect=stub_detect)
    b, _ = load_dfeval_images(dfeval_root, limit=4, seed=0, detect=stub_detect)
    assert [r["sample_id"] for r in a] == [r["sample_id"] for r in b]
    assert len(a) <= 4


def test_a_missing_root_names_the_gate_rather_than_raising_a_traceback(tmp_path):
    with pytest.raises(FileNotFoundError, match="huggingface.co/datasets"):
        load_dfeval_images(tmp_path / "absent")


def test_a_renamed_manifest_column_names_the_layout_not_a_KeyError(
        dfeval_root, stub_detect):
    # The single most likely way the gated download differs from the layout
    # `_manifest_rows` guessed at. A bare `KeyError: 'domain'` out of a
    # comprehension reads as a bug in this loader; the refusal has to say
    # which column is missing, which ones the file has, and that Task 7 is
    # where the layout gets reconciled.
    manifest = dfeval_root / "metadata.csv"
    manifest.write_text(manifest.read_text().replace("domain", "site", 1))
    with pytest.raises(ValueError, match="missing the column"):
        load_dfeval_images(dfeval_root, detect=stub_detect)


def test_every_record_carries_the_file_it_was_decoded_from(dfeval_root,
                                                           stub_detect):
    # `bench.dfeval_controls.jpeg_table_features` reads a quantisation
    # table out of the file HEADER, so the path is part of what a record
    # is here. A consumer re-deriving it from `sample_id` would couple
    # itself to that string format and, when the gated download's real id
    # scheme differs, would fail into a legitimate-looking "no JPEG table
    # on this corpus" refusal rather than an error.
    import csv

    records, _ = load_dfeval_images(dfeval_root, detect=stub_detect)
    with (dfeval_root / "metadata.csv").open(newline="") as fh:
        filenames = {row["item_id"]: row["filename"]
                     for row in csv.DictReader(fh)}
    assert records
    for r in records:
        assert r["path"].is_file()
        assert r["path"] == dfeval_root / filenames[r["sample_id"].split("/")[1]]
