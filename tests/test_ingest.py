from unittest import mock

import cv2
import numpy as np
import pytest
from dfd.ingest.image import load_image
from dfd.ingest.video import load_video, sample_indices
from dfd.types import Context, Modality


@pytest.fixture
def jpg(tmp_path):
    p = tmp_path / "a.jpg"
    img = np.full((256, 256, 3), 120, dtype=np.uint8)
    cv2.imwrite(str(p), img)
    return p


@pytest.fixture
def mp4(tmp_path):
    p = tmp_path / "a.mp4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (128, 128))
    for i in range(30):
        vw.write(np.full((128, 128, 3), i * 8 % 255, dtype=np.uint8))
    vw.release()
    return p


def test_image_yields_exactly_one_observation(jpg):
    s = load_image(jpg, Context(label=0))
    assert s.modality is Modality.IMAGE
    assert len(s.observations) == 1


def test_image_observation_source_id_is_the_sample_id(jpg):
    """Video-level aggregation groups by source_id; an image is its own group."""
    s = load_image(jpg, Context())
    assert s.observations[0].source_id == s.sample_id


def test_video_yields_at_most_max_frames(mp4):
    s = load_video(mp4, Context(), max_frames=5, seed=42)
    assert s.modality is Modality.VIDEO
    assert 1 <= len(s.observations) <= 5


def test_video_all_observations_share_one_source_id(mp4):
    s = load_video(mp4, Context(), max_frames=5, seed=42)
    assert len({o.source_id for o in s.observations}) == 1


def test_frame_sampling_is_deterministic_given_a_seed():
    a = sample_indices(total=100, k=7, seed=42)
    b = sample_indices(total=100, k=7, seed=42)
    c = sample_indices(total=100, k=7, seed=43)
    assert a == b
    assert a != c


def test_frame_sampling_never_exceeds_available_frames():
    assert sample_indices(total=3, k=10, seed=1) == [0, 1, 2]


def test_video_timestamps_increase(mp4):
    s = load_video(mp4, Context(), max_frames=5, seed=1)
    ts = [o.t for o in s.observations]
    assert len(ts) > 1
    assert all(b > a for a, b in zip(ts, ts[1:]))


def test_ingest_decodes_but_does_not_analyse(jpg, mp4):
    """Ingest must not do face detection or quality measurement.

    Keeping analysis a separate stage is what lets the face detector be
    swapped (the licence-driven move from InsightFace to YuNet). If ingest
    ever populates these, that swap becomes a rewrite.
    """
    for s in (load_image(jpg, Context()), load_video(mp4, Context(), 3, 1)):
        for o in s.observations:
            assert o.roi is None
            assert o.quality is None


def test_sampling_returns_exactly_k_at_the_tight_boundary():
    """total = k+1 is where bin collisions would appear first if binning changed."""
    assert len(sample_indices(total=8, k=7, seed=1)) == 7
    assert len(sample_indices(total=101, k=100, seed=2)) == 100


def test_video_with_unusable_frame_count_still_yields_observations(mp4):
    """When frame-count metadata is unreliable, fall back to sequential read.

    Mocks cv2.VideoCapture.get to return 0 for CAP_PROP_FRAME_COUNT, simulating
    a codec that does not report frame count reliably.
    """
    with mock.patch.object(cv2.VideoCapture, 'get') as mock_get:
        def get_side_effect(prop):
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 0  # Simulate unusable metadata
            elif prop == cv2.CAP_PROP_FPS:
                return 10.0
            return 0
        mock_get.side_effect = get_side_effect

        # This should not raise; it should fall back to sequential read
        s = load_video(mp4, Context(), max_frames=5, seed=42)
        assert len(s.observations) > 0
        assert len(s.observations) <= 5
        assert all(o.source_id == s.sample_id for o in s.observations)
