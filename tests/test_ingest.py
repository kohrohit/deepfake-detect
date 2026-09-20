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
    assert len(s.observations) <= 5


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
    assert ts == sorted(ts)
