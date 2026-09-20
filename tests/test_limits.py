import dataclasses

import cv2
import numpy as np
import pytest

from dfd.errors import DfdError, InvalidInput, ResourceLimitExceeded
from dfd.limits import (
    DEFAULT_LIMITS, Limits, check_file_size, check_frame_dims,
    check_image_before_decode, probe_image_dims,
)
from dfd.ingest.image import load_image
from dfd.types import Context


def _png(path, width, height):
    """A uniform image: large in pixels, tiny on disk. That gap is the attack."""
    cv2.imwrite(str(path), np.zeros((height, width), dtype=np.uint8))
    return path


@pytest.fixture
def context():
    return Context(label=0)


def test_limits_are_a_frozen_value_object():
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_LIMITS.max_pixels = 1


def test_defaults_are_the_documented_values():
    """Asserting only `> 0` would let max_pixels drift to 1 unnoticed."""
    assert DEFAULT_LIMITS.max_pixels == 7680 * 4320
    assert DEFAULT_LIMITS.max_file_bytes == 256 * 1024 * 1024
    assert DEFAULT_LIMITS.max_frames == 10_000
    assert DEFAULT_LIMITS.max_duration_s == 1800.0


def test_oversized_file_is_rejected(tmp_path):
    p = tmp_path / "big.bin"
    p.write_bytes(b"0" * 2048)
    with pytest.raises(ResourceLimitExceeded, match="exceeds limit 1024"):
        check_file_size(p, Limits(max_file_bytes=1024))


def test_file_within_limit_passes(tmp_path):
    p = tmp_path / "ok.bin"
    p.write_bytes(b"0" * 100)
    assert check_file_size(p, Limits(max_file_bytes=1024)) is None


def test_missing_file_is_invalid_input_not_a_resource_limit(tmp_path):
    """`DfdError` alone cannot tell these apart — ResourceLimitExceeded is one."""
    with pytest.raises(InvalidInput, match="not a readable file"):
        check_file_size(tmp_path / "nope.bin", DEFAULT_LIMITS)


@pytest.mark.parametrize("width,height", [
    (50000, 50000),        # the square bomb
    (1, 10 ** 9),          # a degenerate strip: same pixel count, no large side
    (10 ** 9, 1),          # and its transpose
    (7681, 4320),          # one pixel over the cap
])
def test_oversized_dimensions_are_rejected(width, height):
    """Shape is parametrised deliberately. A check that compares each side
    against a maximum instead of the product passes the strips."""
    with pytest.raises(ResourceLimitExceeded, match="exceeds limit"):
        check_frame_dims(width, height, DEFAULT_LIMITS)


@pytest.mark.parametrize("width,height", [(1920, 1080), (7680, 4320), (1, 1)])
def test_dimensions_within_the_cap_pass(width, height):
    assert check_frame_dims(width, height, DEFAULT_LIMITS) is None


@pytest.mark.parametrize("width,height", [(0, 100), (100, 0), (-1, 100), (100, -1)])
def test_non_positive_dimensions_are_invalid_input(width, height):
    with pytest.raises(InvalidInput, match="must be positive"):
        check_frame_dims(width, height, DEFAULT_LIMITS)


def test_probe_reads_dimensions_from_the_header(tmp_path):
    _png(tmp_path / "a.png", 640, 480)
    assert probe_image_dims(tmp_path / "a.png") == (640, 480)


def test_probe_reads_a_bomb_without_decoding_it(tmp_path):
    """12000x12000 is 144M pixels. If this decoded, it would allocate ~0.4GB."""
    _png(tmp_path / "bomb.png", 12000, 12000)
    assert probe_image_dims(tmp_path / "bomb.png") == (12000, 12000)


def test_probe_emits_no_warnings_on_a_bomb(tmp_path, recwarn):
    """Pillow's own DecompressionBombWarning must be absorbed, not leaked:
    this project requires pristine test output."""
    _png(tmp_path / "bomb.png", 12000, 12000)
    probe_image_dims(tmp_path / "bomb.png")
    assert [w.category.__name__ for w in recwarn] == []


def test_probe_rejects_a_file_that_is_not_an_image(tmp_path):
    p = tmp_path / "junk.png"
    p.write_bytes(b"not an image")
    with pytest.raises(InvalidInput, match="could not read image header"):
        probe_image_dims(p)


def test_a_bomb_passes_the_file_size_check_and_is_still_rejected(tmp_path):
    """The measurement that justifies header probing: this file is ~0.15MB,
    far under the 256MB default, and decodes to ~0.4GB."""
    p = _png(tmp_path / "bomb.png", 12000, 12000)
    assert p.stat().st_size < DEFAULT_LIMITS.max_file_bytes
    assert check_file_size(p, DEFAULT_LIMITS) is None
    with pytest.raises(ResourceLimitExceeded, match="exceeds limit"):
        check_image_before_decode(p, DEFAULT_LIMITS)


def test_the_check_runs_before_any_decode(tmp_path, monkeypatch):
    """The whole point. If cv2.imread is reached, the allocation already
    happened and the limit is decorative."""
    p = _png(tmp_path / "bomb.png", 12000, 12000)

    def _boom(*args, **kwargs):
        raise AssertionError("cv2.imread was called — decode preceded the check")

    monkeypatch.setattr(cv2, "imread", _boom)
    with pytest.raises(ResourceLimitExceeded):
        check_image_before_decode(p, DEFAULT_LIMITS)


def test_load_image_rejects_a_bomb_before_decoding_it(tmp_path, monkeypatch, context):
    """The guard must be WIRED IN. Calling the checker directly in every test
    would let the loader enforce nothing while the suite stayed green."""
    p = _png(tmp_path / "bomb.png", 12000, 12000)

    def _boom(*args, **kwargs):
        raise AssertionError("cv2.imread was called — decode preceded the check")

    monkeypatch.setattr(cv2, "imread", _boom)
    with pytest.raises(ResourceLimitExceeded):
        load_image(p, context)


def test_load_image_still_loads_a_normal_image(tmp_path, context):
    _png(tmp_path / "ok.png", 64, 48)
    sample = load_image(tmp_path / "ok.png", context)
    assert sample.observations[0].payload.shape == (48, 64, 3)


def test_resource_limit_exceeded_is_a_dfd_error():
    assert issubclass(ResourceLimitExceeded, DfdError)
    assert issubclass(InvalidInput, DfdError)
