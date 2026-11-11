from __future__ import annotations

import numpy as np
import pytest
import torch

from dfd.detectors.clip_vit import CLIPVisualTower
from dfd.detectors.gend import GendHead, embed
from training.clip_features import FeatureCache
from training.fit_gend import (
    ENCODER_ID_LNTUNED,
    _embed_for_training,
    build_training_records,
    encoder_cache_id,
    fit_head,
    reencode,
    tune_layernorms,
)


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


# --- Phase 2: LayerNorm tuning (plan Task 13) -------------------------------


def _tuning_records(n: int = 12) -> list[dict]:
    """Paired reals and fakes over `n` subjects, in the record shape.

    Synthetic rather than `build_training_records` because what these tests
    check -- which parameters move, what the artefact loads as -- does not
    depend on the corpus being real, and a real one costs a face detector
    pass per test.
    """
    rng = np.random.default_rng(0)
    records = []
    for i in range(n):
        base = rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)
        for label in (0, 1):
            img = base.copy()
            if label:
                img[12:36, 12:36] = 255 - img[12:36, 12:36]
            records.append({
                "sample_id": f"s{i}/{label}", "subject_id": f"s{i}",
                "label": label, "image": img, "generator": "sbi" if label else None,
            })
    return records


def test_tuning_touches_layernorms_and_nothing_else():
    # 102,400 of ~304M parameters on the real tower. A tuning run that moved
    # anything else would be fine-tuning CLIP, which is a different licence
    # question and a different compute budget from the one Task 6 measured.
    tower = CLIPVisualTower.tiny()
    before = {k: v.clone() for k, v in tower.state_dict().items()}
    tuned, _, _ = tune_layernorms(tower, _tuning_records(), epochs=1, lr=1e-2)
    ln = set(tuned.layernorm_parameter_names())
    after = tuned.state_dict()
    assert any(not torch.equal(after[k], before[k]) for k in ln)
    assert all(torch.equal(after[k], before[k]) for k in after if k not in ln)


def test_the_tuned_encoder_loads_through_the_same_strict_factory(tmp_path):
    # The plan's reason for writing a COMPLETE state_dict rather than a
    # delta: a delta needs a partial, non-strict load, and a non-strict load
    # is how half a model gets loaded without complaining.
    tuned, _, _ = tune_layernorms(CLIPVisualTower.tiny(), _tuning_records(),
                                  epochs=1, lr=1e-2)
    path = tmp_path / "e.pt"
    torch.save(tuned.state_dict(), path)
    CLIPVisualTower.tiny().load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True))


def test_a_tuned_encoder_gets_its_own_feature_cache_id(tmp_path):
    # Sharing the frozen encoder's cache would serve embeddings from the
    # wrong model and the only symptom would be a disappointing AUC.
    img = np.random.default_rng(0).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    frozen = FeatureCache(tmp_path, encoder_id=encoder_cache_id(
        "assets/models/clip_vit_l14_visual.pt"))
    tuned = FeatureCache(tmp_path, encoder_id=ENCODER_ID_LNTUNED)
    assert frozen.encoder_id != tuned.encoder_id
    assert frozen.key(img) != tuned.key(img)


def test_the_tuned_head_carries_the_tuned_version():
    # bench/eval_swaps.py refuses a corpus inside a detector's training
    # window and keys that refusal on this string. A tuned head declaring the
    # frozen probe's version would be evaluated under the wrong window.
    _, head, _ = tune_layernorms(CLIPVisualTower.tiny(), _tuning_records(),
                                 epochs=1, lr=1e-2,
                                 version="0.2.0-lntuned-test")
    assert head.version() == "0.2.0-lntuned-test"


def test_tuning_refuses_a_one_class_corpus():
    records = [r for r in _tuning_records() if r["label"] == 0]
    with pytest.raises(ValueError, match="both labels"):
        tune_layernorms(CLIPVisualTower.tiny(), records, epochs=1, lr=1e-2)


def test_tuning_keeps_the_best_holdout_epoch_not_the_last():
    # Early stopping, implemented as "keep the best epoch". Without it a run
    # that overfits in epoch 5 ships epoch 5, and the overnight cost of the
    # run makes "just run it again with fewer epochs" an expensive answer.
    _, _, trace = tune_layernorms(CLIPVisualTower.tiny(), _tuning_records(),
                                  epochs=3, lr=1e-2, seed=0)
    assert len(trace.holdout_auc) == 3
    scored = [a for a in trace.holdout_auc if a is not None]
    assert trace.selected_epoch == trace.holdout_auc.index(max(scored)) + 1
    assert trace.selected_auc == max(scored)


def test_the_trace_counts_the_parameters_that_were_allowed_to_move():
    # The claim "0.03% of the encoder" is checkable in the artefact rather
    # than only in the plan's prose.
    tower = CLIPVisualTower.tiny()
    _, _, trace = tune_layernorms(tower, _tuning_records(), epochs=1, lr=1e-2)
    expected = sum(
        p.numel() for n, p in tower.named_parameters()
        if n in set(tower.layernorm_parameter_names()))
    assert trace.n_tuned_parameters == expected
    assert trace.n_frozen_parameters == (
        sum(p.numel() for p in tower.parameters()) - expected)


def test_the_tuning_forward_matches_the_detectors_own_embed():
    # The tuner cannot reuse `gend.embed` -- it is wrapped in no_grad and a
    # detached embedding cannot be back-propagated through -- so the same
    # arithmetic exists twice. A tuner that normalised differently from the
    # detector would fit coefficients the detector never sees, and every
    # symptom of that is "the AUC is lower than the fit reported".
    tower = CLIPVisualTower.tiny()
    tower.eval()
    crops = [r["image"] for r in _tuning_records(3)]
    with torch.no_grad():
        mine = _embed_for_training(tower, crops)
    assert torch.allclose(mine, embed(tower, crops), atol=0, rtol=0)


def test_the_layernorms_are_not_weight_decayed():
    # AdamW's default weight_decay=0.01 pulls a LayerNorm gain toward zero,
    # and a gain of zero deletes the channel. The LayerNorms ARE the fitted
    # set here, so the default would quietly shrink the whole fit. The head
    # keeps the default: it stands in for the L2 that Phase 1's
    # LogisticRegression(C=1.0) applies to the same 1,538 numbers.
    tower = CLIPVisualTower.tiny()
    captured = {}
    real = torch.optim.AdamW

    def _spy(groups, **kw):
        captured["groups"] = groups
        captured["kw"] = kw
        return real(groups, **kw)

    torch.optim.AdamW = _spy
    try:
        tune_layernorms(tower, _tuning_records(), epochs=1, lr=1e-3)
    finally:
        torch.optim.AdamW = real

    groups = captured["groups"]
    assert len(groups) == 2
    assert groups[0]["weight_decay"] == 0.0
    ln = set(tower.layernorm_parameter_names())
    tuned_ids = {id(p) for n, p in tower.named_parameters() if n in ln}
    assert {id(p) for p in groups[0]["params"]} == tuned_ids
    # The head's group is left to inherit torch's default, so the read-out is
    # regularised in Phase 2 as LogisticRegression(C=1.0) regularises it in
    # Phase 1. AdamW fills the inherited value in on construction, so this
    # reads the EFFECTIVE decay, not the dict as it was passed.
    assert groups[1]["weight_decay"] > 0


def test_the_returned_weights_are_the_best_epoch_not_the_last(monkeypatch):
    # The trace bookkeeping can say "epoch 1" while the tower carries epoch
    # 3's weights, and nothing downstream would notice: the artefact loads,
    # scores, and is simply worse than the number reported beside it. This
    # pins the WEIGHTS, by scripting the holdout AUC to peak on epoch 1 and
    # asserting the result is bit-identical to a one-epoch run at the same
    # seed.
    import training.fit_gend as fg

    one, _, _ = tune_layernorms(CLIPVisualTower.tiny(), _tuning_records(),
                                epochs=1, lr=1e-2, seed=0)
    scripted = iter([0.9, 0.4, 0.4])
    monkeypatch.setattr(fg, "roc_auc_score", lambda *a, **k: next(scripted))
    three, _, trace = tune_layernorms(CLIPVisualTower.tiny(), _tuning_records(),
                                      epochs=3, lr=1e-2, seed=0)

    assert trace.selected_epoch == 1 and trace.selected_auc == 0.9
    ln = one.layernorm_parameter_names()
    a, b = one.state_dict(), three.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in ln)
