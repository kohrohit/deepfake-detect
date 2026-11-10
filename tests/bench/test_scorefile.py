from __future__ import annotations

import numpy as np  # noqa: F401 -- brief's verbatim import; `aligned` returns ndarrays
import pytest

from bench.scorefile import aligned, load_score_file


def _write(tmp_path, text, name="ufd.csv"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_loads_scores_and_reports_coverage(tmp_path):
    p = _write(tmp_path, "sample_id,score\na,0.1\nb,0.9\n")
    sf = load_score_file(p, known_ids={"a", "b", "c", "d"})
    assert sf.scores == {"a": 0.1, "b": 0.9}
    assert (sf.covered, sf.total) == (2, 4)
    assert sf.coverage == pytest.approx(0.5)
    assert sf.missing == ("c", "d")


def test_the_name_defaults_to_the_file_stem(tmp_path):
    sf = load_score_file(_write(tmp_path, "sample_id,score\na,0.1\n"),
                         known_ids={"a"})
    assert sf.name == "ufd"


def test_an_unknown_sample_id_is_refused_and_named(tmp_path):
    # A baseline scoring a corpus it was not given is a configuration error,
    # not a low coverage number, and the two must not look alike.
    p = _write(tmp_path, "sample_id,score\na,0.1\nzzz,0.9\n")
    with pytest.raises(ValueError, match="zzz"):
        load_score_file(p, known_ids={"a"})


def test_a_duplicate_sample_id_is_refused(tmp_path):
    # One row silently overwriting another changes the denominator
    # invisibly: coverage still reads 100% and one image was scored twice.
    p = _write(tmp_path, "sample_id,score\na,0.1\na,0.9\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_score_file(p, known_ids={"a"})


def test_a_non_finite_score_is_refused(tmp_path):
    p = _write(tmp_path, "sample_id,score\na,nan\n")
    with pytest.raises(ValueError, match="not finite"):
        load_score_file(p, known_ids={"a"})


def test_scores_outside_zero_one_are_accepted(tmp_path):
    # Deliberate. AUC is rank-based, so a baseline emitting logits is as
    # usable as one emitting probabilities, and rejecting logits would
    # silently exclude the baselines most likely to be honest about what
    # their model outputs. Calibration is NOT claimed for these and the
    # report must not print an ECE for them.
    sf = load_score_file(_write(tmp_path, "sample_id,score\na,-3.2\nb,7.9\n"),
                         known_ids={"a", "b"})
    assert sf.scores["a"] == pytest.approx(-3.2)


def test_an_empty_file_is_refused_rather_than_read_as_total_abstention(tmp_path):
    p = _write(tmp_path, "sample_id,score\n")
    with pytest.raises(ValueError, match="no rows"):
        load_score_file(p, known_ids={"a"})


def test_a_missing_header_is_refused(tmp_path):
    p = _write(tmp_path, "a,0.1\nb,0.9\n")
    with pytest.raises(ValueError, match="header"):
        load_score_file(p, known_ids={"a", "b"})


def test_aligned_returns_the_covered_subset_in_record_order(tmp_path):
    records = [
        {"sample_id": "a", "label": 0, "domain": "x.com"},
        {"sample_id": "b", "label": 1, "domain": "y.com"},
        {"sample_id": "c", "label": 1, "domain": "x.com"},
    ]
    sf = load_score_file(_write(tmp_path, "sample_id,score\nc,0.7\na,0.2\n"),
                         known_ids={"a", "b", "c"})
    scores, labels, groups = aligned(sf, records)
    assert list(scores) == [0.2, 0.7]
    assert list(labels) == [0, 1]
    assert list(groups) == ["x.com", "x.com"]
