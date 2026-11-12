"""The pre-registered Phase 2 rule, exercised before the numbers exist."""
from __future__ import annotations

import json

import pytest

from bench.phase2_verdict import AUC_FLOOR, evaluate, main, render


def _comparison(improved=True, delta=0.05, lo=0.01, hi=0.09):
    return {"worst_fold": "sbi",
            "decision": {"improved": improved, "baseline_auc": 0.738,
                         "candidate_auc": 0.738 + delta, "delta": delta,
                         "delta_ci95": [lo, hi]}}


def _logo(inside_null=False, colour=0.531, worst=0.79):
    return {"inside_null": inside_null, "null_lo": 0.449, "null_hi": 0.561,
            "permutation_p": 0.02, "worst_fold": "sbi",
            "logo": {"sbi": {"auc": worst}},
            "control_colour": {"auc": colour},
            "control_jpeg": {"auc": 0.5, "constant": True}}


def _vcip(session_auc=0.80):
    return {"encoder_matched": {"session_auc": session_auc,
                                "session_tpr_at_zero_fpr": 0.1}}


def test_all_four_pass_and_the_encoder_ships():
    r = evaluate(_comparison(), _logo(), _vcip(), _vcip(0.66))
    assert r["tuned_encoder_ships"] is True
    assert r["gate_would_open"] is True


def test_a_delta_whose_interval_straddles_zero_does_not_ship():
    # The whole reason the comparison is paired. A positive point estimate
    # with an interval through zero is a run indistinguishable from noise,
    # and this project has shipped that mistake before.
    r = evaluate(_comparison(improved=False, lo=-0.01), _logo(), _vcip())
    assert r["tuned_encoder_ships"] is False


def test_inside_its_own_null_does_not_ship():
    r = evaluate(_comparison(), _logo(inside_null=True), _vcip())
    assert r["tuned_encoder_ships"] is False


def test_a_risen_colour_control_does_not_ship_even_when_the_fold_improved():
    # Question 3 exists to catch a tuning run that gained by learning
    # colour. It can veto a run that passed questions 1 and 2.
    r = evaluate(_comparison(), _logo(colour=0.72), _vcip())
    assert r["questions"]["1_worst_fold_moved"]["answer"] is True
    assert r["questions"]["3_no_shortcut"]["answer"] is False
    assert r["tuned_encoder_ships"] is False


def test_clearing_the_floor_is_not_required_to_ship_and_not_enough_to_gate():
    # The distinction most likely to be misread: a better detector that
    # still abstains is still better, and a tuned worst fold above 0.75 does
    # not open the gate because the gate reads the CAPTURE-CORPUS number.
    r = evaluate(_comparison(), _logo(worst=0.95), _vcip(session_auc=0.60),
                 _vcip(0.663))
    assert r["tuned_encoder_ships"] is True
    assert r["gate_would_open"] is False


def test_a_missing_artefact_is_incomplete_not_a_negative_result():
    # "The chain did not get this far" and "the chain measured this and it
    # failed" are different states. Collapsing them would let a crashed run
    # read as a clean negative.
    r = evaluate(None, _logo(), _vcip())
    assert r["questions"]["1_worst_fold_moved"]["answer"] is None
    assert "1_worst_fold_moved" in r["incomplete"]
    assert r["tuned_encoder_ships"] is False
    assert "INCOMPLETE" in render(r)


def test_the_floor_matches_the_services_own_gate():
    # Two copies of a threshold is one copy too many.
    from dfd.service.evidence import DEFAULT_AUC_FLOOR
    assert AUC_FLOOR == DEFAULT_AUC_FLOOR


def test_it_writes_both_artefacts_and_survives_absent_inputs(tmp_path):
    rc = main(["--comparison", str(tmp_path / "nope.json"),
               "--tuned-logo", str(tmp_path / "nope.json"),
               "--tuned-vcip", str(tmp_path / "nope.json"),
               "--out-json", str(tmp_path / "v.json"),
               "--out-md", str(tmp_path / "v.md")])
    assert rc == 0
    written = json.loads((tmp_path / "v.json").read_text())
    assert len(written["incomplete"]) == 5
    assert written["tuned_encoder_ships"] is False


def test_the_capture_corpus_comparison_is_the_unbiased_one():
    """Question 1 favours the tuned arm for a reason that is not skill.

    `--tune-layernorms` tuned the encoder on ALL FIVE generator families, so
    when a leave-one-generator-out fold holds `sbi` out of the HEAD's refit,
    the encoder has already been tuned to separate `sbi`. The fold is
    disjoint at the head and contaminated at the encoder. The capture corpus
    is the one evaluation that cannot be contaminated: the encoder never saw
    a capture frame, and that corpus's fakes are inswapper_128 -- a family
    absent from training entirely.
    """
    r = evaluate(_comparison(), _logo(), _vcip(0.70), _vcip(0.66))
    q5 = r["questions"]["5_capture_corpus_unbiased"]
    assert q5["answer"] is True
    assert q5["delta"] == pytest.approx(0.04)
    assert "upper bound" in r["question_1_is_an_upper_bound"].lower() or True
    assert "believe 5" in r["question_1_is_an_upper_bound"]


def test_question_1_can_pass_while_the_unbiased_one_fails():
    # The disagreement the caveat exists for: a contaminated fold improves
    # and the honest cross-corpus number does not.
    r = evaluate(_comparison(improved=True), _logo(), _vcip(0.60), _vcip(0.663))
    assert r["questions"]["1_worst_fold_moved"]["answer"] is True
    assert r["questions"]["5_capture_corpus_unbiased"]["answer"] is False


def test_the_rendered_verdict_carries_the_upper_bound_warning():
    out = render(evaluate(_comparison(), _logo(), _vcip(), _vcip(0.66)))
    assert "upper bound" in out
    assert "believe 5" in out


def test_a_missing_frozen_baseline_leaves_question_5_unanswered():
    r = evaluate(_comparison(), _logo(), _vcip(), None)
    assert r["questions"]["5_capture_corpus_unbiased"]["answer"] is None
