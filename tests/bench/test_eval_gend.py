"""Slot D's measurement harness.

The three `kish_ess` tests the plan listed here are NOT duplicated: that
function and its four tests already landed with the benchmark harness
(`bench/metrics.py`, `tests/bench/test_metrics.py`), and a second copy of a
test is a second thing to keep in step.
"""
from __future__ import annotations

import numpy as np
import pytest

from bench.eval_gend import colour_features, logo_refit


def test_logo_refits_per_fold_and_never_trains_on_the_held_out_family():
    rng = np.random.default_rng(0)
    n = 400
    features = rng.normal(size=(n, 16)).astype(np.float32)
    generators = np.array(["real"] * 200 + ["g1"] * 100 + ["g2"] * 100)
    labels = (generators != "real").astype(np.int64)
    # g1 is separable on axis 0, g2 on axis 1 and on nothing else: a fitter
    # that saw g2 would ace the g2 fold, and one that did not cannot.
    features[generators == "g1", 0] += 6.0
    features[generators == "g2", 1] += 6.0
    groups = np.arange(n) // 2
    folds = logo_refit(features, labels, generators, groups, seed=0)
    assert set(folds) == {"g1", "g2"}
    assert folds["g1"]["auc"] < 0.75
    assert folds["g2"]["auc"] < 0.75


def test_logo_reports_a_fold_it_can_actually_separate():
    rng = np.random.default_rng(1)
    n = 400
    features = rng.normal(size=(n, 16)).astype(np.float32)
    generators = np.array(["real"] * 200 + ["g1"] * 100 + ["g2"] * 100)
    labels = (generators != "real").astype(np.int64)
    features[labels == 1, 0] += 6.0        # one shared direction
    groups = np.arange(n) // 2
    folds = logo_refit(features, labels, generators, groups, seed=0)
    assert folds["g1"]["auc"] > 0.9 and folds["g2"]["auc"] > 0.9


def test_logo_refuses_a_single_generator_corpus():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 4)).astype(np.float32)
    generators = np.array(["real"] * 20 + ["g1"] * 20)
    labels = (generators != "real").astype(np.int64)
    with pytest.raises(ValueError, match="one generator"):
        logo_refit(features, labels, generators, np.arange(40), seed=0)


def test_the_colour_control_finds_a_planted_colour_shortcut():
    # The control must be able to PASS, or reporting it proves nothing. Lab
    # colour means alone separated DF40's halves at 0.843.
    rng = np.random.default_rng(0)
    reals = [rng.integers(0, 120, (32, 32, 3), dtype=np.uint8) for _ in range(40)]
    fakes = [rng.integers(135, 255, (32, 32, 3), dtype=np.uint8) for _ in range(40)]
    x = colour_features(reals + fakes)
    y = np.array([0] * 40 + [1] * 40)
    from sklearn.linear_model import LogisticRegression

    from bench.metrics import auc
    probs = LogisticRegression(max_iter=1000).fit(x, y).predict_proba(x)[:, 1]
    assert auc(probs, y) > 0.95


def test_the_colour_control_is_blind_to_a_corpus_with_no_colour_shortcut():
    rng = np.random.default_rng(0)
    both = [rng.integers(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(80)]
    x = colour_features(both)
    y = np.array([0] * 40 + [1] * 40)
    from sklearn.linear_model import LogisticRegression

    from bench.metrics import auc
    probs = LogisticRegression(max_iter=1000).fit(x, y).predict_proba(x)[:, 1]
    assert 0.3 < auc(probs, y) < 0.7
