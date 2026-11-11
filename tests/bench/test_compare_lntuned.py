"""Tuned versus frozen on the worst LOGO fold, paired over the same records.

Two AUCs with overlapping bootstrap CIs are not a comparison. Both runs score
the SAME corpus, so the difference can be tested on the same rows rather than
on two independent intervals -- which is the difference between "the tuned
number is bigger" and "the tuning did something".
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from bench.compare_lntuned import compare_scorefiles, load_scorefile, paired_delta_ci
from bench.eval_gend import logo_fold_scores


def _corpus(n=400, seed=0):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 16)).astype(np.float32)
    generators = np.array(["real"] * (n // 2) + ["g1"] * (n // 4) + ["g2"] * (n // 4))
    labels = (generators != "real").astype(np.int64)
    features[labels == 1, 0] += 6.0
    groups = np.array([f"c{i // 2}" for i in range(n)])
    samples = np.array([f"s{i}" for i in range(n)])
    return features, labels, generators, groups, samples


def test_fold_scores_agree_with_the_aggregates_they_are_sliced_from():
    # The refactor's only real risk: logo_refit's published AUCs must be the
    # AUCs of the per-record scores now written beside them, or the report
    # and its evidence describe different runs.
    from bench.eval_gend import logo_refit
    from bench.metrics import auc
    features, labels, generators, groups, _ = _corpus()
    folds = logo_refit(features, labels, generators, groups, seed=0)
    scored = logo_fold_scores(features, labels, generators, groups, seed=0)
    for held, agg in folds.items():
        got = auc(scored[held]["probs"], scored[held]["y"])
        assert got == pytest.approx(agg["auc"])
        assert len(scored[held]["rows"]) == agg["n"]


def test_a_paired_delta_of_a_vector_with_itself_is_exactly_zero():
    # The sanity check that catches an unpaired implementation: resampling
    # two independent times would put a nonzero spread around zero.
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 50)
    s = rng.normal(size=100) + y
    g = np.array([f"c{i // 2}" for i in range(100)])
    delta, lo, hi = paired_delta_ci(s, s, y, g, n=200, seed=0)
    assert delta == 0.0 and lo == 0.0 and hi == 0.0


def test_a_real_improvement_lands_with_a_ci_clear_of_zero():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 200)
    g = np.array([f"c{i // 2}" for i in range(400)])
    weak = rng.normal(size=400) + 0.3 * y
    strong = rng.normal(size=400) + 2.0 * y
    delta, lo, hi = paired_delta_ci(weak, strong, y, g, n=400, seed=0)
    assert delta > 0 and lo > 0


def test_comparing_two_different_corpora_is_refused(tmp_path):
    # The join key is the sample id. Two runs over different records would
    # compare a tuned encoder on one corpus against a frozen one on another
    # and report the difference as the effect of tuning.
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    rows = [{"sample_id": "s0", "subject_id": "c0", "generator": "g1",
             "label": 1, "score": 0.7}]
    a.write_text(json.dumps({"encoder": "frozen.pt", "folds": {"g1": rows}}))
    b.write_text(json.dumps({"encoder": "tuned.pt", "folds": {"g1": [
        {**rows[0], "sample_id": "s9"}]}}))
    with pytest.raises(ValueError, match="same records"):
        compare_scorefiles(load_scorefile(a), load_scorefile(b))


def test_the_comparison_names_the_worst_fold_of_the_frozen_baseline(tmp_path):
    # The plan's rule: the number that decides Phase 2 is the worst fold, and
    # the baseline's worst fold is the one to track -- picking the tuned
    # run's own worst fold would let a run that collapsed a different family
    # look like an improvement.
    def _rows(prefix, scores):
        return [{"sample_id": f"{prefix}{i}", "subject_id": f"c{i // 2}",
                 "generator": prefix, "label": i % 2, "score": s}
                for i, s in enumerate(scores)]
    hard = [0.5] * 40               # every row tied: AUC 0.5, exactly
    easy = [0.1, 0.9] * 20          # separable: even rows label 0, odd label 1
    frozen = {"encoder": "frozen.pt",
              "folds": {"g1": _rows("g1", hard), "g2": _rows("g2", easy)}}
    tuned = {"encoder": "tuned.pt",
             "folds": {"g1": _rows("g1", easy), "g2": _rows("g2", hard)}}
    (tmp_path / "f.json").write_text(json.dumps(frozen))
    (tmp_path / "t.json").write_text(json.dumps(tuned))
    out = compare_scorefiles(load_scorefile(tmp_path / "f.json"),
                             load_scorefile(tmp_path / "t.json"))
    assert out["worst_fold"] == "g1"
    assert out["decision"]["improved"] is True


def test_an_undefined_interval_is_recorded_rather_than_crashing(tmp_path,
                                                                monkeypatch):
    # bootstrap_ci_by_group legitimately returns (nan, nan) when every
    # resample was degenerate. allow_nan=False alone would turn that into a
    # crash on the LAST step of a nine-hour chain, losing the comparison
    # instead of recording that one interval was undefined.
    import bench.compare_lntuned as cl

    monkeypatch.setattr(cl, "bootstrap_ci_by_group",
                        lambda *a, **k: (float("nan"), float("nan")))
    rows = [{"sample_id": f"s{i}", "subject_id": f"c{i // 2}",
             "generator": "g1", "label": i % 2, "score": 0.1 + 0.8 * (i % 2)}
            for i in range(8)]
    rows2 = [{**r, "generator": "g2"} for r in rows]
    for name, gen in (("a", rows), ("b", rows2)):
        (tmp_path / f"{name}.json").write_text(json.dumps(
            {"encoder": f"{name}.pt", "folds": {"g1": gen}}))
    rc = cl.main(["--baseline", str(tmp_path / "a.json"),
                  "--candidate", str(tmp_path / "b.json"),
                  "--out-json", str(tmp_path / "o.json"),
                  "--out-md", str(tmp_path / "o.md")])
    assert rc == 0
    written = json.loads((tmp_path / "o.json").read_text())
    assert written["decision"]["delta_ci95"] == [None, None]
    assert "undefined" in (tmp_path / "o.md").read_text()
