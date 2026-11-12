from __future__ import annotations

import numpy as np

from dfd.detectors.clip_vit import CLIPVisualTower
from training.clip_features import FeatureCache


def _img(v: int) -> np.ndarray:
    return np.full((40, 40, 3), v, dtype=np.uint8)


def test_second_pass_is_all_hits(tmp_path):
    tower = CLIPVisualTower.tiny().eval()
    cache = FeatureCache(tmp_path, encoder_id="tiny")
    images = [_img(10), _img(200), _img(90)]
    first = cache.embed_all(tower, images)
    assert cache.stats == {"hits": 0, "misses": 3}

    again = FeatureCache(tmp_path, encoder_id="tiny")
    second = again.embed_all(tower, images)
    assert again.stats == {"hits": 3, "misses": 0}
    assert np.array_equal(first, second)


def test_identical_pixels_share_one_entry(tmp_path):
    # 979 copies of one demo asset are already known to live in this
    # project's capture corpus. Hashing content, not paths, means they cost
    # one forward pass -- and means the cache cannot be fooled into serving
    # one image's embedding for another's.
    cache = FeatureCache(tmp_path, encoder_id="tiny")
    assert cache.key(_img(50)) == cache.key(_img(50))
    assert cache.key(_img(50)) != cache.key(_img(51))


def test_a_different_encoder_id_is_a_different_cache(tmp_path):
    a = FeatureCache(tmp_path, encoder_id="clip-l14")
    b = FeatureCache(tmp_path, encoder_id="clip-l14-lntuned")
    assert a.key(_img(7)) != b.key(_img(7))


def test_a_preprocessing_change_invalidates_every_entry(tmp_path, monkeypatch):
    before = FeatureCache(tmp_path, encoder_id="tiny").key(_img(7))
    monkeypatch.setattr("training.clip_features.PREPROCESS_VERSION", "changed")
    after = FeatureCache(tmp_path, encoder_id="tiny").key(_img(7))
    assert before != after


def test_batching_does_not_change_the_vectors(tmp_path):
    tower = CLIPVisualTower.tiny().eval()
    images = [_img(v) for v in range(10, 100, 10)]
    one = FeatureCache(tmp_path / "a", encoder_id="t").embed_all(
        tower, images, batch_size=1)
    many = FeatureCache(tmp_path / "b", encoder_id="t").embed_all(
        tower, images, batch_size=4)
    assert np.allclose(one, many, atol=1e-5)


def test_a_half_cached_list_keeps_its_order(tmp_path):
    # The failure this forbids: misses are computed in one batch and
    # appended, so a cache that filled 2 of 5 rows would return the 3
    # computed vectors followed by the 2 cached ones. Every caller zips this
    # against a label vector, and a matrix permuted against its labels fits
    # beautifully and means nothing.
    tower = CLIPVisualTower.tiny().eval()
    images = [_img(v) for v in (11, 22, 33, 44, 55)]

    warm = FeatureCache(tmp_path, encoder_id="tiny")
    warm.embed_all(tower, [images[1], images[3]])

    mixed = FeatureCache(tmp_path, encoder_id="tiny")
    got = mixed.embed_all(tower, images)
    assert mixed.stats == {"hits": 2, "misses": 3}

    fresh = FeatureCache(tmp_path / "fresh", encoder_id="tiny")
    assert np.allclose(got, fresh.embed_all(tower, images), atol=1e-6)


def test_duplicate_images_each_get_their_own_row(tmp_path):
    # The 979-copies case. One cache entry is the point; one ROW would be a
    # bug, because the caller's label vector has an entry per observation.
    tower = CLIPVisualTower.tiny().eval()
    images = [_img(70), _img(70), _img(80), _img(70)]
    cache = FeatureCache(tmp_path, encoder_id="tiny")
    got = cache.embed_all(tower, images)
    assert got.shape[0] == 4
    assert np.array_equal(got[0], got[1]) and np.array_equal(got[0], got[3])
    assert not np.array_equal(got[0], got[2])
