import json
from corpora.rd_cache import load_rd_cache, aggregate_is_max_like


def _write(root, name, verdict, score, models):
    d = root / name
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps(
        {"verdict": verdict, "score": score, "models": models}))


def test_loads_results_and_skips_quota_file(tmp_path):
    _write(tmp_path, "aaa", "MANIPULATED", 0.99,
           [{"name": "m1", "verdict": "MANIPULATED", "score": 0.99}])
    (tmp_path / "quota.json").write_text(json.dumps({"month": "2026-08", "count": 27}))
    out = load_rd_cache(tmp_path)
    assert len(out) == 1
    assert out[0].verdict == "MANIPULATED"


def test_exposes_per_model_scores(tmp_path):
    _write(tmp_path, "aaa", "MANIPULATED", 0.98,
           [{"name": "m1", "verdict": "MANIPULATED", "score": 0.99},
            {"name": "m2", "verdict": "AUTHENTIC", "score": 0.01}])
    r = load_rd_cache(tmp_path)[0]
    assert r.model_scores == {"m1": 0.99, "m2": 0.01}
    assert r.n_manipulated == 1 and r.n_models == 2


def test_detects_max_like_aggregation(tmp_path):
    """Reproduces the spec §1.2 finding: the aggregate tracks the maximum."""
    for i, (v, s, top) in enumerate([("MANIPULATED", 0.99, 0.99),
                                     ("MANIPULATED", 0.98, 0.99),
                                     ("MANIPULATED", 0.96, 0.99)]):
        _write(tmp_path, f"d{i}", v, s,
               [{"name": "m1", "verdict": "MANIPULATED", "score": top},
                {"name": "m2", "verdict": "AUTHENTIC", "score": 0.01}])
    results = load_rd_cache(tmp_path)
    assert aggregate_is_max_like(results, tolerance=0.2) is True


def test_a_mean_tracking_ensemble_is_not_max_like(tmp_path):
    """The negative case. Without it a constant `return True` passes, and this
    function launders one of the three headline findings in the spec rather
    than testing it: RD's aggregate tracking the MAX is why its false-positive
    rate approximates the UNION of its members' FPRs."""
    for i in range(3):
        _write(tmp_path, f"m{i}", "MANIPULATED", 0.50,
               [{"name": "m1", "verdict": "MANIPULATED", "score": 0.99},
                {"name": "m2", "verdict": "AUTHENTIC", "score": 0.01}])
    results = load_rd_cache(tmp_path)
    assert aggregate_is_max_like(results, tolerance=0.2) is False


def test_tolerance_is_load_bearing(tmp_path):
    """Same corpus, two tolerances, two answers — so `tolerance` cannot be
    silently ignored."""
    for i in range(3):
        _write(tmp_path, f"m{i}", "MANIPULATED", 0.50,
               [{"name": "m1", "verdict": "MANIPULATED", "score": 0.99},
                {"name": "m2", "verdict": "AUTHENTIC", "score": 0.01}])
    results = load_rd_cache(tmp_path)
    assert aggregate_is_max_like(results, tolerance=0.4) is False
    assert aggregate_is_max_like(results, tolerance=0.6) is True


def test_an_empty_corpus_is_not_reported_as_max_like(tmp_path):
    assert aggregate_is_max_like(load_rd_cache(tmp_path)) is False


def test_missing_models_key_does_not_crash(tmp_path):
    d = tmp_path / "bbb"
    d.mkdir()
    (d / "result.json").write_text(json.dumps({"verdict": "AUTHENTIC", "score": 0.01}))
    out = load_rd_cache(tmp_path)
    assert out[0].model_scores == {}


def test_wrong_shape_result_is_skipped_not_fatal(tmp_path):
    """A result.json that parses fine but is a list, not an object, must not
    take the rest of the 24-entry cache down with it."""
    d = tmp_path / "listshaped"
    d.mkdir()
    (d / "result.json").write_text(json.dumps([1, 2, 3]))
    _write(tmp_path, "ok", "AUTHENTIC", 0.01,
           [{"name": "m1", "verdict": "AUTHENTIC", "score": 0.01}])
    out = load_rd_cache(tmp_path)
    assert [r.cache_key for r in out] == ["ok"]


def test_non_numeric_score_is_skipped_not_fatal(tmp_path):
    d = tmp_path / "badscore"
    d.mkdir()
    (d / "result.json").write_text(json.dumps({
        "verdict": "MANIPULATED", "score": "high",
        "models": [{"name": "m1", "verdict": "MANIPULATED", "score": 0.99}],
    }))
    _write(tmp_path, "ok", "AUTHENTIC", 0.01,
           [{"name": "m1", "verdict": "AUTHENTIC", "score": 0.01}])
    out = load_rd_cache(tmp_path)
    assert [r.cache_key for r in out] == ["ok"]
