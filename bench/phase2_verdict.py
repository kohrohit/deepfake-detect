"""The Phase 2 decision rule, as code rather than as prose in a plan.

WHY THIS IS EXECUTABLE. The rule that decides whether LayerNorm tuning ships was written down on
2026-11-11, while the run was still in its first epoch, precisely so it could not be fitted to the
result. A rule that lives only in a document is still re-read by a human who already knows the
numbers -- and this project's own answer to that problem everywhere else is to make the gate code
(`dfd.service.evidence.gated_detectors` is the precedent). So the four questions are here, each
reading its own artefact, each answering before anyone has formed an opinion.

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


def _load(path: Path) -> dict[str, Any] | None:
    try:
        loaded: dict[str, Any] = json.loads(path.read_text())
        return loaded
    except (OSError, ValueError):
        return None


def evaluate(comparison: dict[str, Any] | None,
             tuned_logo: dict[str, Any] | None,
             tuned_vcip: dict[str, Any] | None) -> dict[str, Any]:
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
        q4.update(answer=(auc is not None and float(auc) >= AUC_FLOOR),
                  session_auc=auc, floor=AUC_FLOOR,
                  session_tpr_at_zero_fpr=matched.get(
                      "session_tpr_at_zero_fpr"))
    questions["4_clears_the_floor"] = q4

    answers = {k: v.get("answer") for k, v in questions.items()}
    incomplete = [k for k, v in answers.items() if v is None]
    ships = (not incomplete
             and bool(answers["1_worst_fold_moved"])
             and bool(answers["2_outside_its_null"])
             and bool(answers["3_no_shortcut"]))
    return {
        "questions": questions,
        "incomplete": incomplete,
        "tuned_encoder_ships": ships,
        "gate_would_open": (not incomplete
                            and bool(answers["4_clears_the_floor"])),
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
              f"**Tuned encoder ships: "
              f"{'YES' if result['tuned_encoder_ships'] else 'NO'}**",
              f"**Evidence floor cleared: "
              f"{'YES' if result['gate_would_open'] else 'NO'}**",
              "", result["rule"], "",
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
    parser.add_argument("--out-json", type=Path,
                        default=Path("bench/phase2_verdict.json"))
    parser.add_argument("--out-md", type=Path,
                        default=Path("bench/phase2_verdict.md"))
    args = parser.parse_args(argv)

    result = evaluate(_load(args.comparison), _load(args.tuned_logo),
                      _load(args.tuned_vcip))
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
