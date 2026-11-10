"""The Deepfake-Eval-2024 entry point, and the oracle it is gated on.

The tests that matter most here are the ones keeping a number this project
did not choose attached to the harness that produced it. The published
result for NPR on this corpus's image split is AUC 0.53, and NPR is slot C
of this repository: if our slot C lands outside the band around it, the
harness has not been shown to measure anything and no other number may be
reported. The three `check_oracle` tests cover both directions of failure,
including the one nobody checks -- a result that is too good.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from bench.eval_dfeval import (
    ORACLE_HI,
    ORACLE_LO,
    SCOPE_HEADING,
    WAIVER_HEADING,
    abstention_by_class,
    check_oracle,
    face_subset,
    logo_verdict,
)


def test_the_oracle_passes_in_the_published_neighbourhood():
    ok, msg = check_oracle(0.53)
    assert ok and "0.53" in msg


def test_the_oracle_fails_an_inverted_or_broken_result():
    # Below 0.40 on a corpus where the published number is 0.53 means the
    # labels, the ids or the score direction are wrong -- not that our NPR
    # is a slightly different NPR.
    ok, msg = check_oracle(0.22)
    assert not ok and "0.40" in msg


def test_the_oracle_fails_a_result_that_is_too_good():
    # Above 0.75 where every published open-source model sits at chance is
    # not good news; it is the signature of a label leak or misaligned ids.
    # This is the direction nobody checks, and it is the one that ships.
    ok, msg = check_oracle(0.93)
    assert not ok and "0.75" in msg


def test_the_oracle_boundaries_are_inclusive_and_stated():
    assert check_oracle(ORACLE_LO)[0]
    assert check_oracle(ORACLE_HI)[0]


def test_logo_is_refused_and_the_refusal_is_the_recorded_answer(dfeval_root,
                                                                stub_detect):
    from corpora.dfeval import load_dfeval_images

    records, _ = load_dfeval_images(dfeval_root, detect=stub_detect)
    verdict = logo_verdict(records)
    assert "cannot fold" in verdict.lower()
    assert "generator" in verdict.lower()


def test_logo_verdict_records_an_unfoldable_corpus_and_propagates_a_broken_one(
        monkeypatch):
    # `UnsplittableCorpusError` SUBCLASSES ValueError (bench/protocol.py),
    # and a plain ValueError out of `bench.protocol._validate` means
    # something else entirely: a straddling source, a label outside {0, 1},
    # an unattributed fake, a real carrying a generator -- a corpus that is
    # broken, not one that merely cannot fold. That subclass relationship is
    # the trap, and it is why the catch in `logo_verdict` is as specific as
    # it looks: widening it to `except ValueError` would answer "this corpus
    # cannot fold" for a corpus nobody should be reporting on at all, and
    # every other test in this file would still pass.
    #
    # Both directions are asserted, so a catch that is too wide and a catch
    # that is too narrow each fail here. Checking only the propagation would
    # pass against a catch that was deleted; checking only the refusal would
    # pass against `except Exception`.
    from bench import eval_dfeval
    from bench.protocol import UnsplittableCorpusError

    records = [{"sample_id": "dfeval/x"}]

    def _unsplittable(_records, *_args, **_kwargs):
        raise UnsplittableCorpusError("no second generator to hold out")

    monkeypatch.setattr(eval_dfeval, "logo_splits", _unsplittable)
    assert "cannot fold" in eval_dfeval.logo_verdict(records).lower()

    def _malformed(_records, *_args, **_kwargs):
        raise ValueError("malformed")

    monkeypatch.setattr(eval_dfeval, "logo_splits", _malformed)
    with pytest.raises(ValueError, match="malformed"):
        eval_dfeval.logo_verdict(records)


def test_face_subset_selects_only_faced_records():
    records = [{"has_face": True}, {"has_face": False}, {"has_face": True}]
    assert list(face_subset(records)) == [True, False, True]


def test_abstention_by_class_separates_reals_from_fakes():
    # One aggregate rate cannot show label-correlated abstention, and
    # label-correlated abstention means every metric above it was computed
    # on a label-selected subsample. Slot A abstained on 74.3% of one
    # generator's fakes against 56.9% of reals while that fold carried its
    # best AUC.
    abstained = np.array([True, False, True, True, False, False])
    labels = np.array([0, 0, 0, 1, 1, 1])
    out = abstention_by_class(abstained, labels)
    assert out["real"] == (2, 3)
    assert out["fake"] == (1, 3)


def test_the_report_carries_the_waiver_and_scope_blocks(tmp_path, dfeval_root,
                                                        stub_detect, stub_registry):
    from bench.eval_dfeval import main

    rc = main(["--root", str(dfeval_root), "--out-json", str(tmp_path / "r.json"),
               "--out-md", str(tmp_path / "r.md"), "--allow-oracle-failure"],
              detect=stub_detect, registry=stub_registry)
    assert rc == 0
    md = (tmp_path / "r.md").read_text()
    assert WAIVER_HEADING in md
    assert SCOPE_HEADING in md
    assert "cannot fold" in md.lower()


def test_a_failing_oracle_stops_every_other_number(tmp_path, dfeval_root,
                                                   stub_detect, inverted_registry):
    # The gate's whole purpose: a harness that cannot reproduce a published
    # result on a published corpus has not been shown to measure anything,
    # so it reports the failure and nothing else.
    from bench.eval_dfeval import main

    rc = main(["--root", str(dfeval_root), "--out-json", str(tmp_path / "r.json"),
               "--out-md", str(tmp_path / "r.md")],
              detect=stub_detect, registry=inverted_registry)
    assert rc != 0
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["oracle"]["passed"] is False
    assert "detectors" not in report


def test_no_single_corpus_wide_auc_is_reported(tmp_path, dfeval_root,
                                               stub_detect, stub_registry):
    # For a face detector on a corpus that is not all faces there is no
    # honest aggregate, and printing one invites it to be quoted.
    from bench.eval_dfeval import main

    main(["--root", str(dfeval_root), "--out-json", str(tmp_path / "r.json"),
          "--out-md", str(tmp_path / "r.md"), "--allow-oracle-failure"],
         detect=stub_detect, registry=stub_registry)
    report = json.loads((tmp_path / "r.json").read_text())
    for entry in report["detectors"].values():
        assert "auc" not in entry
        assert "auc_face_subset" in entry
        assert "n_face_subset" in entry


def test_kish_ess_is_reported_beside_every_n(tmp_path, dfeval_root,
                                             stub_detect, stub_registry):
    from bench.eval_dfeval import main

    main(["--root", str(dfeval_root), "--out-json", str(tmp_path / "r.json"),
          "--out-md", str(tmp_path / "r.md"), "--allow-oracle-failure"],
         detect=stub_detect, registry=stub_registry)
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["corpus"]["kish_ess_real"] > 0
    assert report["corpus"]["kish_ess_fake"] > 0


def test_a_baseline_score_file_is_reported_with_its_coverage(
        tmp_path, dfeval_root, stub_detect, stub_registry):
    from bench.eval_dfeval import main
    from corpora.dfeval import load_dfeval_images

    records, _ = load_dfeval_images(dfeval_root, detect=stub_detect)
    csv = tmp_path / "fake_baseline.csv"
    rows = "\n".join(f"{r['sample_id']},0.5" for r in records[:2])
    csv.write_text("sample_id,score\n" + rows + "\n")
    main(["--root", str(dfeval_root), "--out-json", str(tmp_path / "r.json"),
          "--out-md", str(tmp_path / "r.md"), "--scores", str(csv),
          "--allow-oracle-failure"], detect=stub_detect, registry=stub_registry)
    report = json.loads((tmp_path / "r.json").read_text())
    entry = report["baselines"]["fake_baseline"]
    assert entry["coverage"] < 1.0
    assert entry["covered"] == 2
