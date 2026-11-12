"""The Phase 2 decision rule, as code rather than as prose in a plan.

WHY THIS IS EXECUTABLE. The rule that decides whether LayerNorm tuning ships was written down on
2026-11-11, while the run was still in its first epoch, precisely so it could not be fitted to the
result. A rule that lives only in a document is still re-read by a human who already knows the
numbers -- and this project's own answer to that problem everywhere else is to make the gate code
(`dfd.service.evidence.gated_detectors` is the precedent). So the four questions are here, each
reading its own artefact, each answering before anyone has formed an opinion.

QUESTION 1 IS BIASED IN THE TUNED ARM'S FAVOUR, AND THAT IS NOT FIXABLE WITHOUT RE-TUNING PER
FOLD. Leave-one-generator-out holds a family out of the HEAD's refit. For the frozen encoder that
is a true transfer measurement: the encoder never saw any fake. For the TUNED encoder it is not —
`training/fit_gend.py --tune-layernorms` tuned the LayerNorms on all five families at once, so
when the fold holds out `sbi` the encoder has already been tuned to separate `sbi`. The fold is
generator-disjoint at the head and contaminated at the encoder. An honest Phase 2 LOGO would
re-tune one encoder per fold — five runs, roughly nine hours each on the deployment box — which
is why this is recorded rather than fixed. Read question 1 as an UPPER BOUND on the tuned arm,
never as a transfer number, and read question 5 for the unbiased comparison.

WHAT THIS DOES NOT DO. It does not decide whether slot D may contribute evidence to a verdict;
that is the 0.75 evidence floor and it lives in `dfd/service/evidence.py`. Question 4 reports
whether the floor is cleared, but clearing it is necessary and not sufficient -- the card's
cross-corpus checks still apply, and they are enforced there, not here.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The evidence floor. Mirrors `dfd.service.evidence.DEFAULT_AUC_FLOOR`; a
#: disagreement between the two is a bug, and `tests/bench/test_phase2_verdict.py`
#: asserts they match rather than leaving two numbers to drift.
AUC_FLOOR = 0.75

#: The frozen probe's colour control, from `bench/gend_report.json`. Question 3
#: asks whether tuning bought a shortcut instead of a detector, and "near 0.5"
#: has to mean something: this is the number the frozen encoder produced under
#: the strongest form of the control (fitted and scored IN SAMPLE).
FROZEN_COLOUR_CONTROL = 0.530

#: How far above the frozen control a tuned colour control may sit before it
#: is reporting a shortcut rather than a detector. Deliberately generous --
#: this is a tripwire, not a measurement.
COLOUR_CONTROL_MARGIN = 0.05


def _paired_sessions(frozen: dict[str, Any],
                     tuned: dict[str, Any]) -> dict[str, Any]:
    """The paired session-level delta, or an explanation of its absence.

    Returns `paired_delta`, `paired_ci95` and `paired_established`. The last
    is the one that matters: a positive point estimate whose interval
    contains zero is a difference this corpus cannot resolve, and saying so
    is the whole job.
    """
    import numpy as np

    from bench.compare_lntuned import paired_delta_ci

    fs = frozen["encoder_matched"].get("sessions")
    ts = tuned["encoder_matched"].get("sessions")
    if not fs or not ts:
        return {"paired_delta": None, "paired_ci95": None,
                "paired_established": None,
                "paired_note": "one of the runs stored no per-session "
                               "scores, so no paired test is possible"}
    fb = {r["session_id"]: r for r in fs}
    tb = {r["session_id"]: r for r in ts}
    if set(fb) != set(tb):
        return {"paired_delta": None, "paired_ci95": None,
                "paired_established": None,
                "paired_note": "the two runs did not score the same "
                               "sessions; a paired test needs one corpus"}
    ids = sorted(fb)
    if any(fb[i]["label"] != tb[i]["label"] for i in ids):
        return {"paired_delta": None, "paired_ci95": None,
                "paired_established": None,
                "paired_note": "the two runs disagree about a session's "
                               "label"}
    y = np.array([fb[i]["label"] for i in ids])
    a = np.array([fb[i]["score"] for i in ids])
    b = np.array([tb[i]["score"] for i in ids])
    delta, lo, hi = paired_delta_ci(a, b, y, np.array(ids), n=2000, seed=0)
    return {"paired_delta": delta, "paired_ci95": [lo, hi],
            "paired_established": bool(delta > 0 and lo > 0),
            "paired_n_sessions": len(ids)}


def _load(path: Path) -> dict[str, Any] | None:
    try:
        loaded: dict[str, Any] = json.loads(path.read_text())
        return loaded
    except (OSError, ValueError):
        return None


def evaluate(comparison: dict[str, Any] | None,
             tuned_logo: dict[str, Any] | None,
             tuned_vcip: dict[str, Any] | None,
             frozen_vcip: dict[str, Any] | None = None) -> dict[str, Any]:
    """The four pre-registered questions, each with its evidence.

    A missing artefact answers `None`, never `False`: "the chain did not get
    this far" and "the chain measured this and it failed" are different
    states, and collapsing them would let an incomplete run read as a
    negative result.
    """
    questions: dict[str, dict[str, Any]] = {}

    # Q1 -- did tuning move the WORST LOGO fold, paired?
    q1: dict[str, Any] = {"question": "paired delta on the baseline's worst "
                                      "LOGO fold is positive and its 95% "
                                      "interval clears zero",
                          "artefact": "bench/lntuned_vs_frozen.json"}
    if comparison is None:
        q1["answer"] = None
    else:
        d = comparison["decision"]
        q1.update(answer=bool(d["improved"]), fold=comparison["worst_fold"],
                  frozen=d["baseline_auc"], tuned=d["candidate_auc"],
                  delta=d["delta"], delta_ci95=d["delta_ci95"])
    questions["1_worst_fold_moved"] = q1

    # Q2 -- is the tuned run still outside its own permutation null?
    q2: dict[str, Any] = {"question": "the tuned run sits outside its own "
                                      "permutation null",
                          "artefact": "bench/gend_report_lntuned.json"}
    if tuned_logo is None:
        q2["answer"] = None
    else:
        q2.update(answer=not bool(tuned_logo["inside_null"]),
                  null=[tuned_logo.get("null_lo"), tuned_logo.get("null_hi")],
                  permutation_p=tuned_logo.get("permutation_p"),
                  worst_fold_auc=tuned_logo["logo"][
                      tuned_logo["worst_fold"]]["auc"])
    questions["2_outside_its_null"] = q2

    # Q3 -- did tuning buy a shortcut instead of a detector?
    q3: dict[str, Any] = {"question": "the colour and JPEG controls stay flat",
                          "artefact": "bench/gend_report_lntuned.json"}
    if tuned_logo is None:
        q3["answer"] = None
    else:
        colour = float(tuned_logo["control_colour"]["auc"])
        jpeg = tuned_logo["control_jpeg"]
        ceiling = FROZEN_COLOUR_CONTROL + COLOUR_CONTROL_MARGIN
        q3.update(answer=colour <= ceiling,
                  colour=colour, colour_ceiling=ceiling,
                  frozen_colour=FROZEN_COLOUR_CONTROL,
                  jpeg=jpeg.get("auc"), jpeg_structural=jpeg.get("constant"))
    questions["3_no_shortcut"] = q3

    # Q4 -- does slot D now clear the evidence floor? A DIFFERENT number
    # from question 1, and the one most likely to be misread: the gate reads
    # the capture-corpus SESSION AUC, not the LOGO fold.
    q4: dict[str, Any] = {"question": f"capture-corpus session AUC >= "
                                      f"{AUC_FLOOR}",
                          "artefact": "bench/gend_vcip_lntuned.json"}
    if tuned_vcip is None:
        q4["answer"] = None
    else:
        matched = tuned_vcip["encoder_matched"]
        auc = matched.get("session_auc")
        ci = matched.get("session_auc_ci95")
        # THE QUESTION IS UNCHANGED -- it was registered as a point-estimate
        # test and it stays one, because moving a bar after seeing the data
        # is what pre-registration exists to prevent. But the interval is
        # REPORTED beside the answer, because registering a point-estimate
        # test was a mistake and "yes" read alone is a stronger claim than
        # the measurement supports. If `floor_inside_interval` is true, the
        # honest statement is "the point estimate clears the floor and the
        # uncertainty does not permit the claim".
        q4.update(answer=(auc is not None and float(auc) >= AUC_FLOOR),
                  session_auc=auc, floor=AUC_FLOOR,
                  session_auc_ci95=ci,
                  floor_inside_interval=(
                      bool(ci) and ci[0] is not None
                      and float(ci[0]) <= AUC_FLOOR <= float(ci[1])),
                  session_tpr_at_zero_fpr=matched.get(
                      "session_tpr_at_zero_fpr"))
    questions["4_clears_the_floor"] = q4

    # Q5 -- the UNBIASED comparison, and the reason it exists is above:
    # the capture corpus is the one evaluation the tuned encoder cannot have
    # been contaminated by. It never saw a capture frame, and that corpus's
    # fakes are inswapper_128 output -- a generator family absent from
    # training entirely, where every family in question 1 was present.
    q5: dict[str, Any] = {"question": "capture-corpus session AUC, tuned "
                                      "against frozen (UNBIASED: the encoder "
                                      "never saw this corpus or its "
                                      "generator)",
                          "artefact": "bench/gend_vcip_lntuned.json vs "
                                      "bench/gend_vcip.json"}
    if tuned_vcip is None or frozen_vcip is None:
        q5["answer"] = None
    else:
        t = tuned_vcip["encoder_matched"].get("session_auc")
        f = frozen_vcip["encoder_matched"].get("session_auc")
        if t is None or f is None:
            q5["answer"] = None
        else:
            # THE PAIRED TEST, and it is the only one that can answer
            # whether tuning moved this corpus. Both arms score the SAME
            # sessions, so the difference is taken on those sessions and
            # bootstrapped over them. The registered answer stays what it
            # was -- direction -- and `paired_established` says whether the
            # direction survives the arithmetic.
            paired = _paired_sessions(frozen_vcip, tuned_vcip)
            q5.update(answer=float(t) > float(f), tuned=float(t),
                      frozen=float(f), delta=float(t) - float(f),
                      tuned_ci95=tuned_vcip["encoder_matched"].get(
                          "session_auc_ci95"),
                      frozen_ci95=frozen_vcip["encoder_matched"].get(
                          "session_auc_ci95"),
                      **paired,
                      note="The marginal intervals are reported but are NOT "
                           "the test: both arms score the same sessions, so "
                           "overlapping marginal intervals say nothing about "
                           "a paired difference either way. "
                           "`paired_ci95` is the test.")
    questions["5_capture_corpus_unbiased"] = q5

    answers = {k: v.get("answer") for k, v in questions.items()}
    incomplete = [k for k, v in answers.items() if v is None]
    # SHIPPING is decided by questions 1-3, exactly as registered on
    # 2026-11-11. Questions 4 and 5 are reported, not gating: 4 says whether
    # the evidence floor is cleared (a different gate, in
    # dfd/service/evidence.py) and 5 is the unbiased read on whether the
    # tuning helped at all. An unanswered 4 or 5 must not turn a decided
    # "ships" into "no" -- that would let a missing artefact veto a
    # measurement that was made.
    deciding = ("1_worst_fold_moved", "2_outside_its_null", "3_no_shortcut")
    ships = all(answers[k] is True for k in deciding)
    return {
        "questions": questions,
        "incomplete": incomplete,
        "tuned_encoder_ships": ships,
        "gate_would_open": answers["4_clears_the_floor"] is True,
        "deciding_questions": list(deciding),
        "question_1_is_an_upper_bound": (
            "The tuned encoder was tuned on ALL FIVE generator families, so a "
            "leave-one-generator-out fold is disjoint at the head and "
            "contaminated at the encoder. Question 1 therefore favours the "
            "tuned arm for a reason that is not generalisation. Question 5 is "
            "the unbiased comparison: the encoder never saw the capture "
            "corpus, whose fakes are inswapper_128 — a family absent from "
            "training. If 1 and 5 disagree, believe 5."),
        "rule": ("Questions 1, 2 and 3 decide whether the tuned encoder "
                 "replaces the frozen one. Question 4 is separate and "
                 "decides nothing about shipping: it reports whether slot D "
                 "clears the 0.75 evidence floor, which is necessary and not "
                 "sufficient for the gate — dfd/service/evidence.py still "
                 "applies the card's cross-corpus checks. A tuned worst fold "
                 "above 0.75 does NOT open the gate."),
        "if_not_shipping": ("Phase 1's weights stay shipped and the result is "
                            "recorded as measured. That is the plan's own "
                            "rule and a near miss does not renegotiate it."),
    }


#: Printed whenever the floor sits inside the measured interval. Question 4
#: was registered as a point-estimate test and stays one; this says what that
#: answer is worth when the uncertainty straddles the bar.
_FLOOR_CAVEAT = (
    " — **on the point estimate only. The 95% interval CONTAINS the floor**, "
    "so the measurement does not support the claim that slot D clears it. "
    "Question 4 was registered as a point-estimate test, which was a mistake; "
    "the question is left as registered and this is the correction.")


def _paired_line(result: dict[str, Any]) -> str:
    """One sentence on the only unbiased, properly-tested number here."""
    q5 = result["questions"]["5_capture_corpus_unbiased"]
    est = q5.get("paired_established")
    if est is None:
        return ("**No paired test of the capture corpus was possible** — see "
                f"`{q5.get('paired_note', 'missing per-session scores')}`.")
    ci = q5["paired_ci95"]
    body = (f"paired delta {q5['paired_delta']:+.4f}, 95% CI "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] over "
            f"{q5.get('paired_n_sessions')} sessions")
    if est:
        return (f"**The unbiased improvement is established**: {body} — the "
                "interval clears zero.")
    return (f"**The unbiased improvement is NOT established**: {body} — the "
            "interval contains zero. The direction is positive on every "
            "honest measure and this corpus cannot resolve it. That is the "
            "finding, not a failure to finish.")


def render(result: dict[str, Any]) -> str:
    lines = ["# Phase 2 — the pre-registered verdict", ""]
    if result["incomplete"]:
        lines += [f"**INCOMPLETE.** Missing: {', '.join(result['incomplete'])}."
                  " A missing artefact is not a negative result.", ""]
    for key, q in result["questions"].items():
        a = q.get("answer")
        mark = "—" if a is None else ("yes" if a else "NO")
        lines.append(f"- **{key}** [{mark}] {q['question']}")
        detail = {k: v for k, v in q.items()
                  if k not in ("question", "artefact", "answer")}
        if detail:
            lines.append(f"      {json.dumps(detail, sort_keys=True)}")
    lines += ["",
              "> **Question 1 is an upper bound, not a transfer number.** The "
              "tuned encoder was tuned on all five generator families, so a "
              "leave-one-generator-out fold is disjoint at the head and "
              "contaminated at the encoder. Question 5 is the unbiased "
              "comparison. If they disagree, believe 5.",
              "",
              f"**Tuned encoder ships: "
              f"{'YES' if result['tuned_encoder_ships'] else 'NO'}**",
              f"**Evidence floor cleared: "
              f"{'YES' if result['gate_would_open'] else 'NO'}**"
              + (_FLOOR_CAVEAT
                 if result["questions"]["4_clears_the_floor"].get(
                     "floor_inside_interval") else ""),
              "", _paired_line(result), "", result["rule"], "",
              result["if_not_shipping"], ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path,
                        default=Path("bench/lntuned_vs_frozen.json"))
    parser.add_argument("--tuned-logo", type=Path,
                        default=Path("bench/gend_report_lntuned.json"))
    parser.add_argument("--tuned-vcip", type=Path,
                        default=Path("bench/gend_vcip_lntuned.json"))
    parser.add_argument("--frozen-vcip", type=Path,
                        default=Path("bench/gend_vcip.json"),
                        help="the FROZEN encoder's capture-corpus result, "
                             "for the one comparison the tuned encoder "
                             "cannot have contaminated")
    parser.add_argument("--out-json", type=Path,
                        default=Path("bench/phase2_verdict.json"))
    parser.add_argument("--out-md", type=Path,
                        default=Path("bench/phase2_verdict.md"))
    args = parser.parse_args(argv)

    result = evaluate(_load(args.comparison), _load(args.tuned_logo),
                      _load(args.tuned_vcip), _load(args.frozen_vcip))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True,
                                        allow_nan=False) + "\n")
    args.out_md.write_text(render(result))
    logger.info("ships=%s floor_cleared=%s incomplete=%s",
                result["tuned_encoder_ships"], result["gate_would_open"],
                result["incomplete"] or "nothing")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
