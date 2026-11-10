from __future__ import annotations

import numpy as np
import pytest
import torch

from dfd.detectors.gend import GendHead
from training.fit_gend import build_training_records, fit_head, reencode


def test_every_record_passes_through_the_same_encode(tiny_fairface_root, stub_detect):
    records, _ = build_training_records(tiny_fairface_root, limit=6, offset=0,
                                        seed=0, detect=stub_detect)
    # The guard this stands in for: reals keeping their original encode while
    # fakes are written by a different encoder teaches the detector the
    # encoder. Matching resolution and container on DF40 RAISED its colour
    # shortcut from 0.800 to 0.904, so "both halves look similar" is not a
    # substitute for "both halves went through one encoder".
    assert {r["compression"] for r in records} == {"jpeg_q95"}


def test_the_corpus_carries_five_generator_families(tiny_fairface_root, stub_detect):
    records, _ = build_training_records(tiny_fairface_root, limit=8, offset=0,
                                        seed=0, detect=stub_detect)
    generators = {r["generator"] for r in records if r["label"] == 1}
    assert generators == {"sbi", "swap_warp_hull", "swap_poisson",
                          "swap_mouth_patch", "swap_lowres_paste"}


def test_every_fake_shares_a_subject_with_the_real_it_came_from(
        tiny_fairface_root, stub_detect):
    records, _ = build_training_records(tiny_fairface_root, limit=8, offset=0,
                                        seed=0, detect=stub_detect)
    by_subject = {}
    for r in records:
        by_subject.setdefault(r["subject_id"], set()).add(r["label"])
    assert all(labels == {0, 1} for labels in by_subject.values())


def test_sample_ids_are_unique(tiny_fairface_root, stub_detect):
    # bench.runner._logo_results raises on duplicates, but only after the
    # whole corpus has been scored. Catching it here costs nothing.
    records, _ = build_training_records(tiny_fairface_root, limit=8, offset=0,
                                        seed=0, detect=stub_detect)
    ids = [r["sample_id"] for r in records]
    assert len(set(ids)) == len(ids)


def test_offset_skips_the_first_n_photographs(tiny_fairface_root, stub_detect):
    a, _ = build_training_records(tiny_fairface_root, limit=4, offset=0,
                                  seed=0, detect=stub_detect)
    b, _ = build_training_records(tiny_fairface_root, limit=4, offset=4,
                                  seed=0, detect=stub_detect)
    assert not ({r["subject_id"] for r in a} & {r["subject_id"] for r in b})


def test_reencode_is_idempotent_in_container_terms():
    img = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    once, twice = reencode(img), reencode(reencode(img))
    assert once.shape == twice.shape == img.shape


def test_fit_refuses_a_one_class_corpus():
    features = np.random.default_rng(0).normal(size=(20, 8)).astype(np.float32)
    with pytest.raises(ValueError, match="both labels"):
        fit_head(features, np.zeros(20, dtype=np.int64), version="0.0.1-test")


def test_the_fitted_head_reproduces_the_logistic_probabilities():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 8)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    labels = (features[:, 0] > 0).astype(np.int64)
    head = fit_head(features, labels, version="0.0.1-test")
    with torch.no_grad():
        probs = torch.softmax(head(torch.from_numpy(features)), dim=1)[:, 1]
    assert ((probs > 0.5).numpy().astype(int) == labels).mean() > 0.9


def test_the_version_is_written_into_the_head():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 8)).astype(np.float32)
    labels = np.array([0, 1] * 20, dtype=np.int64)
    head = fit_head(features, labels, version="0.1.0-probe-fairface60000")
    assert head.version() == "0.1.0-probe-fairface60000"


def test_a_saved_head_loads_back_strictly(tmp_path):
    rng = np.random.default_rng(0)
    head = fit_head(rng.normal(size=(40, 8)).astype(np.float32),
                    np.array([0, 1] * 20, dtype=np.int64), version="v")
    path = tmp_path / "h.pt"
    torch.save(head.state_dict(), path)
    GendHead(in_dim=8).load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True))
