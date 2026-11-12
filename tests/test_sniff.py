"""`sniff_suffix` names a byte stream only when it is certain."""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from dfd.ingest.sniff import sniff_suffix
from dfd.pipeline import IMAGE_SUFFIXES, VIDEO_SUFFIXES


def _encoded(fmt: str) -> bytes:
    img = Image.fromarray(
        np.random.default_rng(0).integers(0, 255, (32, 32, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.mark.parametrize("fmt,want", [("JPEG", ".jpg"), ("PNG", ".png"),
                                      ("BMP", ".bmp"), ("WEBP", ".webp")])
def test_it_names_real_encoded_images(fmt, want):
    # Real encoder output, not hand-written headers: a signature that only
    # matches bytes I wrote myself proves nothing about Pillow's.
    assert sniff_suffix(_encoded(fmt)) == want


def test_every_suffix_it_returns_is_one_the_pipeline_routes():
    # A suffix the dispatch does not know would turn a missing filename into
    # the same error by a longer road.
    known = IMAGE_SUFFIXES | VIDEO_SUFFIXES
    for fmt in ("JPEG", "PNG", "BMP", "WEBP"):
        assert sniff_suffix(_encoded(fmt)) in known


def test_riff_without_a_known_type_is_not_guessed():
    # WEBP and AVI share the RIFF prefix; a third RIFF type is neither.
    assert sniff_suffix(b"RIFF\x00\x00\x00\x00WAVEfmt ") is None


def test_avi_and_webp_are_told_apart_at_offset_8():
    assert sniff_suffix(b"RIFF\x24\x00\x00\x00AVI LIST") == ".avi"
    assert sniff_suffix(b"RIFF\x24\x00\x00\x00WEBPVP8 ") == ".webp"


def test_quicktime_and_mp4_are_told_apart_by_brand():
    assert sniff_suffix(b"\x00\x00\x00\x14ftypqt  \x00\x00\x02\x00") == ".mov"
    assert sniff_suffix(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00") == ".mp4"


def test_ebml_without_a_doctype_is_not_guessed():
    # Matroska and WebM share a magic number and differ only in a DocType
    # string that is not at a fixed offset.
    assert sniff_suffix(b"\x1a\x45\xdf\xa3" + b"\x00" * 40) is None
    assert sniff_suffix(b"\x1a\x45\xdf\xa3" + b"\x00" * 20 + b"webm") == ".webm"
    assert sniff_suffix(b"\x1a\x45\xdf\xa3" + b"\x00" * 20
                        + b"matroska") == ".mkv"


@pytest.mark.parametrize("data", [
    b"", b"short", b"not an image at all, just prose",
    b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n", b"\x00" * 64,
])
def test_an_unrecognised_stream_is_never_guessed(data):
    # Guessing wrong is strictly worse than not guessing: it routes the file
    # to a decoder that fails confusingly, or silently reads one frame of a
    # video as an image. None lets the existing clear error fire.
    assert sniff_suffix(data) is None
