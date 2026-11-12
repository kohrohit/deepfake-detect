"""The product-shaped view of a decision: a verdict, and the opinions behind it.

Spec §5.3. The audit record is the thing of record and does not change shape
for anyone; this is a projection of it for a caller who wants a verdict, a
score, and the per-detector list that makes the score auditable at a glance.

**Every detector appears, including the silent ones.** A four-model ensemble
in which three abstained is not a one-model system, and a response that
omitted the abstentions would read as though the remaining model had agreed
with something. Today that matters more than it will later: on this
deployment every detector is gated out of deciding (`dfd/service/evidence.py`),
so the abstentions ARE the result, and their `reason` is the whole content of
the answer.

Pure: no registry, no clock, no I/O. It takes a finished record — the
dataclass, or the JSON form the store reads back — and returns a dict.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from dfd.audit import AuditRecord

#: Internal verdict → the product vocabulary. Both of the "cannot say"
#: verdicts map to UNKNOWN, and neither maps to AUTHENTIC: "I have never seen
#: anything like this" and "this is fine" are different statements, and
#: collapsing the first into the second is how an unfamiliar generator gets
#: waved through.
VERDICT_NAMES = {
    "real": "AUTHENTIC",
    "fake": "MANIPULATED",
    "insufficient_evidence": "UNKNOWN",
    "out_of_distribution": "UNKNOWN",
}

#: What an unrecognised verdict string becomes. Fails toward "cannot say".
DEFAULT_VERDICT_NAME = "UNKNOWN"


def _as_mapping(record: AuditRecord | Mapping[str, Any]) -> Mapping[str, Any]:
    """The record as plain data, whichever form the caller holds.

    The route serving this reads finished records out of SQLite as JSON and
    never reconstructs the dataclass, so accepting only `AuditRecord` would
    produce a function that passes its tests and cannot serve its own route.
    """
    if isinstance(record, AuditRecord):
        return json.loads(record.to_json())  # type: ignore[no-any-return]
    return record


def _verdict_name(raw: Any) -> str:
    value = getattr(raw, "value", raw)
    return VERDICT_NAMES.get(str(value), DEFAULT_VERDICT_NAME)


def _sign(value: float) -> int:
    if value > 0:
        return 1
    return -1 if value < 0 else 0


def _dissenters(evidence: list[Mapping[str, Any]], llr_total: float) -> list[str]:
    """Non-abstaining detectors that point away from the fused result.

    An abstention is never dissent. Silence is not disagreement, and counting
    it as such would make a correctly-gated deployment — where everything is
    silent — look maximally conflicted.

    A fused total of exactly 0.0 has no direction to disagree with, so the
    sign test cannot be applied. When that zero was MADE of non-zero opinions
    (+2 and -2 cancelling), it is not consensus: it is the most conflicted
    record this system can produce, and every contributor to it is named.
    """
    speaking = [e for e in evidence
                if not e.get("abstained") and float(e.get("llr") or 0.0) != 0.0]
    direction = _sign(llr_total)
    if direction == 0:
        return [str(e["detector"]) for e in speaking]
    return [str(e["detector"]) for e in speaking
            if _sign(float(e["llr"])) != direction]


def rd_response(record: AuditRecord | Mapping[str, Any]) -> dict[str, Any]:
    """Project a finished record onto the product contract.

    Args:
        record: a finished `AuditRecord`, or the JSON form of one.

    Returns:
        `verdict` in {AUTHENTIC, MANIPULATED, UNKNOWN}; `score`, the recorded
        posterior and never a recomputation of it; `models`, one entry per
        detector with its raw score, whether it abstained and why;
        `dissenting_models`; `quality_band`; and `schema_version`.
    """
    data = _as_mapping(record)
    evidence = list(data.get("evidence") or ())
    llr_total = float(data.get("llr_total") or 0.0)

    models = [{
        "name": str(e.get("detector")),
        # The detector's OWN direction, not the fused one. `none` where it
        # abstained or contributed nothing, because an abstention has no
        # opinion to report and inventing AUTHENTIC for it would be a
        # fabricated vote.
        "verdict": _model_verdict(e),
        # Reported even for an abstention. On this deployment slot D runs,
        # produces a score and is gated out of deciding; dropping the score
        # would hide the one number showing the detector worked at all.
        "score": e.get("raw_score"),
        "abstained": bool(e.get("abstained")),
        "reason": str(e.get("reason") or ""),
    } for e in evidence]

    return {
        "verdict": _verdict_name(data.get("verdict")),
        "score": float(data.get("posterior") or 0.0),
        "models": models,
        "dissenting_models": _dissenters(evidence, llr_total),
        "quality_band": data.get("quality_band"),
        "schema_version": data.get("schema_version"),
    }


def _model_verdict(evidence: Mapping[str, Any]) -> str | None:
    if evidence.get("abstained"):
        return None
    llr = float(evidence.get("llr") or 0.0)
    if llr > 0:
        return "MANIPULATED"
    return "AUTHENTIC" if llr < 0 else None


#: The simplest question a caller can ask: is this a deepfake, and how sure
#: are you. `rd_response` answers it with four detectors and their reasons;
#: this answers it with three fields, for an integrator who wants a boolean.
SIMPLE_SCHEMA_VERSION = 1


def simple_response(record: AuditRecord | Mapping[str, Any]) -> dict[str, Any]:
    """`{is_deepfake, confidence, reason}` — the three-field view.

    THE ONE RULE THAT MATTERS HERE: **an abstention has no confidence.**
    When the system cannot say, `is_deepfake` is `null` and `confidence` is
    `null` — never `0.5`. A 0.5 would be a number, and a caller who
    thresholds numbers would extract a verdict from a refusal to give one;
    that is precisely the failure this service's evidence gate exists to
    prevent, and it would be undone by one convenient default in a
    convenience format. `null` cannot be thresholded.

    CONFIDENCE IS IN THE ANSWER GIVEN, not in "fake". For MANIPULATED it is
    the posterior P(fake); for AUTHENTIC it is P(authentic) = 1 - posterior.
    A caller who wants a single P(fake) axis should use `rd_response`'s
    `score`, which is the recorded posterior and never a recomputation.

    `why_no_answer` names every detector that declined and why, so a null
    answer is distinguishable from a server fault without a second request.
    On this deployment that list IS the answer: no detector is measured above
    the 0.75 evidence floor, so all of them decline.
    """
    rd = rd_response(record)
    verdict = rd["verdict"]
    score = float(rd["score"])

    if verdict == "MANIPULATED":
        is_deepfake: bool | None = True
        confidence: float | None = score
    elif verdict == "AUTHENTIC":
        is_deepfake, confidence = False, 1.0 - score
    else:
        is_deepfake, confidence = None, None

    silent = [{"name": m["name"], "reason": m["reason"]}
              for m in rd["models"] if m["abstained"]]
    deciding = [m for m in rd["models"] if not m["abstained"]]

    return {
        "is_deepfake": is_deepfake,
        "confidence": confidence,
        "verdict": verdict,
        "reason": ("decided" if is_deepfake is not None
                   else "insufficient_evidence"),
        "detectors_consulted": len(rd["models"]),
        "detectors_that_decided": len(deciding),
        "why_no_answer": silent if is_deepfake is None else [],
        "quality_band": rd["quality_band"],
        "schema_version": SIMPLE_SCHEMA_VERSION,
    }
