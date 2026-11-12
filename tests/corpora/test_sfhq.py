"""`load_sfhq_records` turns SFHQ part 3 into label-1 benchmark records.

SFHQ is the first fake supply this project may fit a SHIPPING detector on, so
the assertions here are about the claims that licence rests on and about the
two defects that have already cost this repo a corpus: a duplicate image
entering twice under two ids (docs/HANDOFF.md §0, the capture corpus), and a
label the data does not actually carry (`corpora.df40`'s refused generator).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from bench.protocol import _validate as validate_corpus
from corpora.sfhq import (
    ALIGN,
    DUPLICATE,
    FACE_DETECTOR,
    NO_FACE,
    SFHQ_COMPRESSION,
    SFHQ_PT3_GENERATOR,
    load_sfhq_records,
)
from dfd.faces import FaceBox


def _box() -> FaceBox:
    lms = np.array([[30.0, 40.0], [60.0, 40.0], [45.0, 55.0],
                    [33.0, 70.0], [57.0, 70.0]])
    return FaceBox(x=20, y=25, w=50, h=60, landmarks=lms, score=0.99)


def _detect_one(_frame):
    return [_box()]


def _write(root: Path, names: list[str], *, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for name in names:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        img = rng.integers(0, 255, (120, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(p), img)


def test_every_record_is_a_fake_carrying_the_documented_generator(
        tmp_path: Path) -> None:
    """The generator id is a fact about part 3, not a filename guess.

    `corpora.df40` refuses to read a generator off filenames because a
    guessed one fabricates a leave-one-generator-out axis. SFHQ part 3 is
    documented as pure StyleGAN2 sampling, so this label is the opposite
    case and the distinction is the whole reason it may be used.
    """
    _write(tmp_path, ["a.jpg", "b.jpg"])
    records, _ = load_sfhq_records(tmp_path, detect=_detect_one)
    assert [r["label"] for r in records] == [1, 1]
    assert {r["generator"] for r in records} == {SFHQ_PT3_GENERATOR}


def test_the_nested_sample_folder_is_found(tmp_path: Path) -> None:
    """The archive nests `a small samples (750 images)/` beside the full set.

    A non-recursive scan of the archive root loads nothing at all from it,
    and an empty corpus is not an error anyone notices downstream — every
    metric over zero records is undefined rather than wrong.
    """
    _write(tmp_path, ["a small samples (750 images)/x.jpg", "top.jpg"])
    records, _ = load_sfhq_records(tmp_path, detect=_detect_one)
    assert {r["sample_id"] for r in records} == {
        "sfhq/a small samples (750 images)/x.jpg", "sfhq/top.jpg"}


def test_a_repeated_image_enters_the_corpus_once_and_is_counted(
        tmp_path: Path) -> None:
    """The defect that cost this project its capture corpus, in one test.

    The copy is made at the BYTE level, which is what dedup by content hash
    can catch and all it can catch. Re-encoding the image through
    imread/imwrite produces a visually identical file whose aligned crop
    hashes differently, and it would pass — the same near-duplicate gap
    `corpora.face_pool` documents. Writing the test the other way would have
    claimed a guard this code does not have.
    """
    _write(tmp_path, ["a.jpg"])
    shutil.copyfile(tmp_path / "a.jpg", tmp_path / "copy.jpg")
    records, skipped = load_sfhq_records(tmp_path, detect=_detect_one)
    assert len(records) == 1
    assert skipped == {DUPLICATE: 1}


def test_an_image_with_no_face_is_counted_not_dropped_silently(
        tmp_path: Path) -> None:
    _write(tmp_path, ["a.jpg", "b.jpg"])
    records, skipped = load_sfhq_records(tmp_path, detect=lambda _f: [])
    assert records == []
    assert skipped == {NO_FACE: 2}


def test_compression_is_recorded_as_unknown_not_guessed(tmp_path: Path) -> None:
    _write(tmp_path, ["a.jpg"])
    records, _ = load_sfhq_records(tmp_path, detect=_detect_one)
    assert records[0]["compression"] == SFHQ_COMPRESSION == "unknown"


def test_preprocessing_matches_every_other_corpus(tmp_path: Path) -> None:
    """Crops from two corpora are only comparable if made the same way."""
    _write(tmp_path, ["a.jpg"])
    records, _ = load_sfhq_records(tmp_path, detect=_detect_one, size=224)
    assert records[0]["face_detector"] == FACE_DETECTOR
    assert records[0]["align"] == ALIGN
    assert records[0]["image"].shape == (224, 224, 3)
    assert records[0]["image"].dtype == np.uint8


def test_the_protocol_accepts_a_mixed_corpus(tmp_path: Path) -> None:
    """These records must sit beside reals without tripping the validator."""
    _write(tmp_path, ["a.jpg", "b.jpg"])
    records, _ = load_sfhq_records(tmp_path, detect=_detect_one)
    reals = [{"sample_id": f"real/{i}", "source_id": f"real/{i}",
              "subject_id": f"real/{i}", "generator": None, "label": 0}
             for i in range(2)]
    validate_corpus(records + reals)


def test_a_limit_is_deterministic_under_a_seed(tmp_path: Path) -> None:
    _write(tmp_path, [f"{i:03d}.jpg" for i in range(20)])
    a, _ = load_sfhq_records(tmp_path, limit=5, seed=1, detect=_detect_one)
    b, _ = load_sfhq_records(tmp_path, limit=5, seed=1, detect=_detect_one)
    c, _ = load_sfhq_records(tmp_path, limit=5, seed=2, detect=_detect_one)
    ids = lambda rs: [r["sample_id"] for r in rs]  # noqa: E731
    assert len(a) == 5
    assert ids(a) == ids(b)
    assert ids(a) != ids(c)


def test_a_root_that_is_not_a_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not a directory"):
        load_sfhq_records(tmp_path / "missing", detect=_detect_one)


def test_a_directory_with_no_images_is_refused(tmp_path: Path) -> None:
    """Silently returning zero records makes every later metric undefined."""
    (tmp_path / "notes.txt").write_text("nothing here")
    with pytest.raises(FileNotFoundError, match="no image files"):
        load_sfhq_records(tmp_path, detect=_detect_one)
