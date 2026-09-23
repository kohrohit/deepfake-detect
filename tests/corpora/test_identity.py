"""Embedding a loaded corpus for acceptance criterion 2.

Hermetic: the embedder and the detector are both injected, so nothing here
touches a weight file.
"""
import numpy as np
import pytest
from corpora.identity import embeddings_for_records
from dfd.embed import EMBEDDING_DIM
from dfd.faces import FaceBox


def _records(n=4):
    return [{"sample_id": f"s{i}",
             "image": np.random.default_rng(i).integers(
                 0, 255, (224, 224, 3), dtype=np.uint8)}
            for i in range(n)]


def _box():
    lms = np.array([[80.0, 90.0], [140.0, 90.0], [110.0, 120.0],
                    [85.0, 150.0], [135.0, 150.0]])
    return FaceBox(x=40, y=40, w=140, h=140, landmarks=lms, score=0.9)


class _Embedder:
    """Returns a distinct unit vector per call, or None on demand."""

    def __init__(self, fail_on=()):
        self.fail_on = set(fail_on)
        self.calls = 0

    def embed(self, frame, box):
        self.calls += 1
        if self.calls in self.fail_on:
            return None
        v = np.zeros(EMBEDDING_DIM, np.float32)
        v[self.calls % EMBEDDING_DIM] = 1.0
        return v


def test_every_record_that_embeds_is_keyed_by_sample_id():
    records = _records()
    emb, skipped = embeddings_for_records(
        records, embedder=_Embedder(), detect=lambda f: [_box()])

    assert set(emb) == {"s0", "s1", "s2", "s3"}
    assert skipped == {}
    assert all(v.shape == (EMBEDDING_DIM,) for v in emb.values())


def test_a_crop_that_will_not_redetect_is_counted_not_silently_dropped():
    """check_identity_disjoint refuses to certify a split holding an id it
    cannot compare, so the caller must be able to see how many that is."""
    records = _records()
    emb, skipped = embeddings_for_records(
        records, embedder=_Embedder(), detect=lambda f: [])

    assert emb == {}
    assert skipped == {"no_face": 4}


def test_a_refused_embedding_is_counted_under_its_own_reason():
    records = _records()
    emb, skipped = embeddings_for_records(
        records, embedder=_Embedder(fail_on={2}), detect=lambda f: [_box()])

    assert len(emb) == 3
    assert skipped == {"no_embedding": 1}


def test_a_degenerate_box_is_counted_separately_from_no_face():
    """A detection that misses the frame is a different failure from no
    detection at all, and they are fixed differently."""
    off = FaceBox(x=-10_000, y=-10_000, w=10, h=10,
                  landmarks=np.zeros((5, 2)), score=0.9)
    emb, skipped = embeddings_for_records(
        _records(2), embedder=_Embedder(), detect=lambda f: [off])

    assert emb == {}
    assert skipped == {"degenerate_box": 2}


def test_absent_weights_report_a_reason_rather_than_raising(monkeypatch):
    """A deployment without the recogniser must still run the benchmark, with
    criterion 2 reported as unmeasured."""
    import corpora.identity as mod

    monkeypatch.setattr(mod, "load_embedder", lambda: (None, "weights_absent"))
    emb, skipped = embeddings_for_records(_records(3), detect=lambda f: [_box()])

    assert emb == {}
    assert skipped == {"weights_absent": 3}


def test_the_highest_scoring_face_is_the_one_embedded():
    """A crop with two detections must embed the confident one, not the first."""
    seen = {}
    low = FaceBox(x=0, y=0, w=10, h=10, landmarks=np.zeros((5, 2)), score=0.2)
    high = _box()

    class _Watch(_Embedder):
        def embed(self, frame, box):
            seen["score"] = box.score
            return super().embed(frame, box)

    embeddings_for_records(_records(1), embedder=_Watch(),
                           detect=lambda f: [low, high])
    assert seen["score"] == pytest.approx(0.9)
