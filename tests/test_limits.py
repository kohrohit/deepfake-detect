import dataclasses

import cv2
import numpy as np
import PIL.Image
import pytest

import dfd.ingest.video as video_mod
from dfd.errors import DfdError, InvalidInput, ResourceLimitExceeded
from dfd.limits import (
    DEFAULT_LIMITS, Limits, check_file_size, check_frame_dims,
    check_image_before_decode, probe_image_dims,
)
from dfd.ingest.image import load_image
from dfd.ingest.video import load_video
from dfd.types import Context


def _png(path, width, height):
    """A uniform image: large in pixels, tiny on disk. That gap is the attack."""
    cv2.imwrite(str(path), np.zeros((height, width), dtype=np.uint8))
    return path


def _mp4(path, width, height, n_frames=30, fps=10.0):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(n_frames):
        vw.write(np.full((height, width, 3), i * 8 % 255, dtype=np.uint8))
    vw.release()
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
    happened and the limit is decorative.

    Also fences PIL.Image.Image.load: `probe_image_dims` uses Pillow, not
    cv2, to read the header. A cv2.imread patch alone proves nothing about
    a regression where the header reader itself decodes pixels (e.g. an
    added `im.load()` call) — that would allocate the full bomb while
    passing every test that only watches cv2.
    """
    p = _png(tmp_path / "bomb.png", 12000, 12000)

    def _boom_cv2(*args, **kwargs):
        raise AssertionError("cv2.imread was called — decode preceded the check")

    def _boom_pil(*args, **kwargs):
        raise AssertionError("Image.load was called — decode preceded the check")

    monkeypatch.setattr(cv2, "imread", _boom_cv2)
    monkeypatch.setattr(PIL.Image.Image, "load", _boom_pil)
    with pytest.raises(ResourceLimitExceeded):
        check_image_before_decode(p, DEFAULT_LIMITS)


def test_probe_does_not_decode_pixels(tmp_path, monkeypatch):
    """probe_image_dims must read the header only. A regression that calls
    `im.load()` (or otherwise forces pixel access) would allocate ~144MB on
    this fixture without tripping any size or dimension check — invisible
    unless the decode call itself is fenced directly, independent of any
    ResourceLimitExceeded assertion."""
    p = _png(tmp_path / "bomb.png", 12000, 12000)

    def _boom(*args, **kwargs):
        raise AssertionError("Image.load was called — probe decoded pixels")

    monkeypatch.setattr(PIL.Image.Image, "load", _boom)
    assert probe_image_dims(p) == (12000, 12000)


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


# --- Video path: the same header-before-decode guards, mirrored. ---
#
# `load_image`'s guard is proven wired above by monkeypatching cv2.imread.
# Without an equivalent proof for `load_video`, `check_file_size`, the
# `max_frames` clamp, the frame-dimension gate, and the duration gate could
# each be silently deleted and the rest of the suite would stay green — the
# exact hazard this task exists to close.


def test_load_video_rejects_a_missing_path(tmp_path):
    with pytest.raises(InvalidInput, match="not a readable file"):
        load_video(tmp_path / "nope.mp4", Context())


def test_load_video_rejects_an_oversized_file_before_opening(tmp_path, monkeypatch):
    """check_file_size must run before the container is ever opened."""
    p = tmp_path / "big.mp4"
    p.write_bytes(b"0" * 2048)

    def _boom(*args, **kwargs):
        raise AssertionError(
            "cv2.VideoCapture was constructed — open preceded the size check")

    monkeypatch.setattr(cv2, "VideoCapture", _boom)
    with pytest.raises(ResourceLimitExceeded, match="exceeds limit 1024"):
        load_video(p, Context(), limits=Limits(max_file_bytes=1024))


def test_load_video_clamps_max_frames_to_the_limit(tmp_path):
    """A caller cannot reintroduce unbounded extraction via `max_frames`:
    `limits.max_frames` wins. Without the clamp, total=30 <= requested k=100
    would return all 30 frames via sample_indices' total<=k branch."""
    p = _mp4(tmp_path / "a.mp4", 64, 48, n_frames=30)
    sample = load_video(p, Context(), max_frames=100, seed=1,
                         limits=Limits(max_frames=3))
    assert len(sample.observations) <= 3


def test_load_video_rejects_declared_oversized_dimensions_before_reading_frames(
        tmp_path, monkeypatch):
    """The container declares frame width/height at open time, before any
    cap.read(). check_frame_dims must run against those declared dimensions
    before the decode loop starts, not after a frame is decoded."""
    p = tmp_path / "fake.mp4"
    p.write_bytes(b"0" * 100)

    def _boom_read(self):
        raise AssertionError("cap.read() was called — decode preceded the dimension check")

    props = {cv2.CAP_PROP_FRAME_WIDTH: 50000.0, cv2.CAP_PROP_FRAME_HEIGHT: 50000.0}
    monkeypatch.setattr(cv2.VideoCapture, "isOpened", lambda self: True)
    monkeypatch.setattr(cv2.VideoCapture, "get", lambda self, prop: props.get(prop, 0.0))
    monkeypatch.setattr(cv2.VideoCapture, "read", _boom_read)
    monkeypatch.setattr(cv2.VideoCapture, "release", lambda self: None)

    with pytest.raises(ResourceLimitExceeded, match="exceeds limit"):
        load_video(p, Context(), limits=DEFAULT_LIMITS)


def test_load_video_rejects_a_container_reporting_zero_dimensions(tmp_path, monkeypatch):
    """A container declaring 0x0 must be refused, not read unguarded."""
    p = tmp_path / "fake.mp4"
    p.write_bytes(b"0" * 100)

    monkeypatch.setattr(cv2.VideoCapture, "isOpened", lambda self: True)
    monkeypatch.setattr(cv2.VideoCapture, "get", lambda self, prop: 0.0)
    monkeypatch.setattr(cv2.VideoCapture, "read", lambda self: (False, None))
    monkeypatch.setattr(cv2.VideoCapture, "release", lambda self: None)

    with pytest.raises(InvalidInput, match="must be positive"):
        load_video(p, Context(), limits=DEFAULT_LIMITS)


def test_load_video_rejects_a_container_declaring_excessive_duration(tmp_path, monkeypatch):
    """Declared duration (frame_count / fps) must be checked before decode.
    A well-compressed multi-hour file passes the byte-size cap easily; the
    duration gate is what still refuses it, before a single frame decodes."""
    p = tmp_path / "fake.mp4"
    p.write_bytes(b"0" * 100)

    def _boom_read(self):
        raise AssertionError("cap.read() was called — decode preceded the duration check")

    props = {
        cv2.CAP_PROP_FRAME_WIDTH: 64.0,
        cv2.CAP_PROP_FRAME_HEIGHT: 48.0,
        cv2.CAP_PROP_FRAME_COUNT: 10_000_000.0,
        cv2.CAP_PROP_FPS: 30.0,
    }
    monkeypatch.setattr(cv2.VideoCapture, "isOpened", lambda self: True)
    monkeypatch.setattr(cv2.VideoCapture, "get", lambda self, prop: props.get(prop, 0.0))
    monkeypatch.setattr(cv2.VideoCapture, "read", _boom_read)
    monkeypatch.setattr(cv2.VideoCapture, "release", lambda self: None)

    with pytest.raises(ResourceLimitExceeded, match="exceeds limit"):
        load_video(p, Context(), limits=DEFAULT_LIMITS)


@pytest.mark.parametrize("fps,expect_gate_fires", [
    (100.0, False),        # real, usable fps: 60000/100 = 600s, well under the cap
    (1e-9, True),          # finite and positive -- no fix needed here, but the
                            # boundary ("small positive still gates correctly")
                            # is worth pinning rather than leaving implicit
    (-30.0, True),         # negative: truthy, and `> 0` alone would reject it
    (float("nan"), True),  # NaN: truthy, and comparisons with NaN are all False
    (float("inf"), True),  # infinite: truthy AND `> 0` is True -- division
                            # collapses duration to 0.0, which never exceeds
                            # the limit unless finiteness is checked too
])
def test_load_video_duration_gate_treats_fps_as_a_domain_not_a_truthiness_check(
        tmp_path, monkeypatch, fps, expect_gate_fires):
    """fps is adversary-controlled container metadata read from the
    container header. Three distinct degenerate values must all fall back
    to DEFAULT_FPS instead of being used directly:

    - `0.0` and negative values are truthy, so a bare `cap.get(...) or
      DEFAULT_FPS` only catches exact zero, not negative -- and negative
      fps yields a negative duration that can never exceed a positive limit.
    - NaN is truthy and also survives a plain `> 0` check by luck of being
      compared to anything and getting False either way; `duration_s > limit`
      is silently always False.
    - Infinity is truthy AND satisfies a plain `> 0` check (it IS positive),
      so a fix that checks only `> 0` and not finiteness lets it through:
      `total / inf` collapses to `0.0`, which also never exceeds the limit.

    The fallback must therefore be "finite and positive", not just
    "positive". The positive (100.0) and small-positive (1e-9) cases pin
    that the fix is a domain check, not a fallback taken unconditionally:
    with a real, usable fps the gate must NOT fire (100.0) or must still
    fire correctly because the duration genuinely is enormous (1e-9)."""
    p = tmp_path / "fake.mp4"
    p.write_bytes(b"0" * 100)

    props = {
        cv2.CAP_PROP_FRAME_WIDTH: 64.0,
        cv2.CAP_PROP_FRAME_HEIGHT: 48.0,
        cv2.CAP_PROP_FRAME_COUNT: 60_000.0,  # at DEFAULT_FPS=25.0 this is 2400s, over the cap
        cv2.CAP_PROP_FPS: fps,
    }
    monkeypatch.setattr(cv2.VideoCapture, "isOpened", lambda self: True)
    monkeypatch.setattr(cv2.VideoCapture, "get", lambda self, prop: props.get(prop, 0.0))
    monkeypatch.setattr(cv2.VideoCapture, "read", lambda self: (False, None))
    monkeypatch.setattr(cv2.VideoCapture, "release", lambda self: None)

    if expect_gate_fires:
        with pytest.raises(ResourceLimitExceeded, match="exceeds limit"):
            load_video(p, Context(), limits=DEFAULT_LIMITS)
    else:
        # Gate did not fire; falls through to the (mocked, frameless) read
        # loop, which is the observable proof that no ResourceLimitExceeded
        # was raised for this fps.
        with pytest.raises(ValueError, match="could not decode any frames"):
            load_video(p, Context(), limits=DEFAULT_LIMITS)


def test_load_video_stops_decoding_once_it_has_the_frames_it_needs(tmp_path, monkeypatch):
    """The max_frames clamp must bound decode work, not merely retained
    observations: a loop that keeps reading to the end of the stream after
    the quota is met still pays the full decode cost of a crafted long
    video. Forces sample_indices to want only frames {0,1,2} out of 200 and
    counts cap.read() calls."""
    p = _mp4(tmp_path / "a.mp4", 8, 8, n_frames=200, fps=100.0)
    monkeypatch.setattr(video_mod, "sample_indices", lambda total, k, seed: [0, 1, 2])

    real_read = cv2.VideoCapture.read
    read_calls = {"n": 0}

    def counting_read(self):
        read_calls["n"] += 1
        return real_read(self)

    monkeypatch.setattr(cv2.VideoCapture, "read", counting_read)
    sample = load_video(p, Context(), max_frames=50, seed=1)
    assert len(sample.observations) == 3
    assert read_calls["n"] <= 5
