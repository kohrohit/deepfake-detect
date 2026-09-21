"""Resource limits enforced at every decode boundary (spec §3A, §10).

Media arrives from an adversary. A crafted image can allocate gigabytes before
any detection logic runs, and a denial-of-service against a fraud control is a
fraud enabler: it forces the fallback path.

Two things this module refuses to do, because both are the usual way this
control is built and neither works:

- It does not treat file size as a proxy for decoded size. A 12,000x12,000
  uniform PNG occupies 161 KB on disk and 0.40 GB decoded; defeating a size
  limit is what a decompression bomb IS.
- It does not inspect a decoded array's shape. By the time an array has a
  shape the memory is already committed.

Dimensions come from the image header instead, before any decode.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .errors import InvalidInput, ResourceLimitExceeded

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Limits:
    # 8K RGB decodes to ~100MB; beyond this nothing legitimate in v-CIP arrives.
    max_pixels: int = 7680 * 4320
    max_file_bytes: int = 256 * 1024 * 1024
    max_frames: int = 10_000
    max_duration_s: float = 1800.0


DEFAULT_LIMITS = Limits()


def check_file_size(path: str | Path, limits: Limits = DEFAULT_LIMITS) -> None:
    """Raise unless the file exists and is within `limits.max_file_bytes`.

    This is a cheap first gate against a merely huge file. It is NOT a defence
    against a decompression bomb — see the module docstring.

    Raises:
        InvalidInput: the path does not exist or is not a regular file.
        ResourceLimitExceeded: the file is larger than the configured maximum.
    """
    p = Path(path)
    if not p.is_file():
        raise InvalidInput(f"not a readable file: {p}")
    size = p.stat().st_size
    if size > limits.max_file_bytes:
        raise ResourceLimitExceeded(
            f"file {p.name} is {size} bytes, exceeds limit {limits.max_file_bytes}")


def check_frame_dims(width: int, height: int,
                     limits: Limits = DEFAULT_LIMITS) -> None:
    """Raise unless the frame dimensions are positive and within the pixel cap.

    The cap is on the PRODUCT, not on either side: a 1 x 10**9 strip carries
    the same allocation as a square bomb and has no large dimension.

    Raises:
        InvalidInput: a dimension is zero or negative.
        ResourceLimitExceeded: width * height exceeds `limits.max_pixels`.
    """
    if width <= 0 or height <= 0:
        raise InvalidInput(
            f"frame dimensions must be positive, got {width}x{height}")
    if width * height > limits.max_pixels:
        raise ResourceLimitExceeded(
            f"frame {width}x{height} = {width * height} pixels, "
            f"exceeds limit {limits.max_pixels}")


def probe_image_dims(path: str | Path) -> tuple[int, int]:
    """Read (width, height) from the image header without decoding it.

    Pillow's own bomb guard is absorbed here rather than allowed to surface:
    its threshold is looser than ours, so our limit should be the one that
    speaks, and its warning would otherwise pollute output.

    Raises:
        InvalidInput: the header cannot be read.
        ResourceLimitExceeded: Pillow refused the image as a bomb outright.
    """
    p = Path(path)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(p) as im:
                return int(im.width), int(im.height)
    except Image.DecompressionBombError as exc:
        raise ResourceLimitExceeded(
            f"{p.name} rejected as a decompression bomb: {exc}") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidInput(f"could not read image header for {p}: {exc}") from exc


def check_image_before_decode(path: str | Path,
                              limits: Limits = DEFAULT_LIMITS) -> tuple[int, int]:
    """Gate an image file before a single pixel is decoded.

    Returns:
        The header (width, height), so callers need not read it twice.

    Raises:
        InvalidInput: unreadable path or unreadable header.
        ResourceLimitExceeded: file too large, or too many pixels.
    """
    check_file_size(path, limits)
    width, height = probe_image_dims(path)
    check_frame_dims(width, height, limits)
    return width, height
