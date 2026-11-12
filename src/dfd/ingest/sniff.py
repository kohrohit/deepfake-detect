"""Name a byte stream by what it IS, for callers that did not name it.

WHY THIS EXISTS. `dfd.pipeline` routes to an ingest adapter by file
EXTENSION, deliberately: `load_image` on a video returns the first frame with
no indication the rest was ignored, so an unlisted extension is refused
rather than guessed. That is right for a file on disk, which always has a
name. It is wrong for `POST /api/analyze`, where the caller has bytes and
often no filename at all — and where the resulting error ("unsupported file
extension ''") describes a problem the caller did not cause and cannot read.

WHAT THIS IS NOT. It is not a replacement for the extension rule and it never
overrides a caller who DID name their file: a request for `x.png` is served
as a PNG whatever the bytes look like, because the dispatch-by-extension
contract is what the rest of the package is tested against. This only fills
in a name that was missing.

CONSERVATIVE BY CONSTRUCTION. Every branch here returns a suffix only on an
unambiguous signature, and `None` otherwise — so an unrecognised stream still
reaches the existing, clear extension error instead of being guessed into the
wrong adapter. Guessing wrong is strictly worse than not guessing: it routes
a file to a decoder that will either fail confusingly or silently read one
frame of it.
"""
from __future__ import annotations

#: How much of the stream a signature may live in. EBML's DocType sits a few
#: dozen bytes in; nothing here needs more than this.
SNIFF_BYTES = 4096


def sniff_suffix(data: bytes) -> str | None:
    """The file extension these bytes deserve, or `None` if unsure.

    Returns a value from `dfd.pipeline`'s IMAGE_SUFFIXES or VIDEO_SUFFIXES,
    with the leading dot, so the caller can append it to a filename and let
    the ordinary dispatch run unchanged.
    """
    head = data[:SNIFF_BYTES]
    if len(head) < 12:
        return None

    if head.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if head.startswith(b"BM"):
        return ".bmp"

    # RIFF containers carry their type at offset 8, so WEBP and AVI share a
    # prefix and are told apart only there.
    if head.startswith(b"RIFF"):
        kind = head[8:12]
        if kind == b"WEBP":
            return ".webp"
        if kind == b"AVI ":
            return ".avi"
        return None

    # ISO base media: 'ftyp' at offset 4, with the brand deciding. QuickTime
    # brands are .mov; everything else in this family is served as .mp4,
    # which is what the video adapter opens either way.
    if head[4:8] == b"ftyp":
        return ".mov" if head[8:12] in (b"qt  ", b"qt\x20\x20") else ".mp4"

    # EBML covers both Matroska and WebM; the DocType string is the only
    # thing that separates them, and it is not at a fixed offset.
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        if b"webm" in head:
            return ".webm"
        if b"matroska" in head:
            return ".mkv"
        return None

    return None
