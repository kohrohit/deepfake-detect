"""The RD-shaped product contract: a verdict, and the opinions behind it.

`rd_response` is a pure function over a finished record — no registry, no
clock, no I/O — so every case here is a dict in and a dict out.
"""
from __future__ import annotations

from typing import Any

import pytest

from dfd.audit import AuditRecord
from dfd.service.contract import rd_response
from dfd.types import Verdict


def _evidence(detector: str, *, llr: float = 0.0, raw: float | None = 0.5,
              abstained: bool = False, reason: str = "ok") -> dict[str, Any]:
    return {"detector": detector, "detector_version": "t-1", "llr": llr,
            "raw_score": raw, "uncertainty": 0.0, "abstained": abstained,
            "reason": reason, "artifacts": {}}


def _record(*, verdict: Verdict = Verdict.REAL, posterior: float = 0.5,
            llr_total: float = 0.0,
            evidence: tuple[dict[str, Any], ...] = ()) -> AuditRecord:
    return AuditRecord(
        schema_version="2", sample_id="s", input_sha256="0" * 64,
        verdict=verdict.value, llr_total=llr_total, posterior=posterior,
        evidence=evidence, quality_band="medium", ood_score=0.0,
        policy_version="p0-default-v0", threshold=1.0,
        model_versions={}, created_at="2026-11-11T00:00:00+00:00",
        stage_reasons={})


def _record_with_abstentions() -> AuditRecord:
    return _record(
        verdict=Verdict.INSUFFICIENT_EVIDENCE, llr_total=0.0,
        evidence=(
            _evidence("blend_seam", abstained=True, raw=None,
                      reason="below_quality_floor"),
            _evidence("effnet_b4", abstained=True, raw=None,
                      reason="weights_absent"),
            _evidence("gend_clip", abstained=True, raw=0.266,
                      reason="uncalibrated_for_band"),
            _evidence("npr", abstained=True, raw=0.901,
                      reason="uncalibrated_for_band"),
        ))


def test_maps_the_three_verdicts_onto_the_product_vocabulary():
    assert rd_response(_record(verdict=Verdict.REAL))["verdict"] == "AUTHENTIC"
    assert rd_response(_record(verdict=Verdict.FAKE))["verdict"] == "MANIPULATED"
    for v in (Verdict.INSUFFICIENT_EVIDENCE, Verdict.OUT_OF_DISTRIBUTION):
        assert rd_response(_record(verdict=v))["verdict"] == "UNKNOWN"


def test_abstaining_detectors_appear_with_their_reason_not_omitted():
    # Omitting them would turn a four-model ensemble with three abstentions
    # into what looks like a one-model system that agreed with itself.
    out = rd_response(_record_with_abstentions())
    assert {m["name"] for m in out["models"]} == {"blend_seam", "effnet_b4",
                                                  "gend_clip", "npr"}
    assert any(m["abstained"] and m["reason"] == "weights_absent"
               for m in out["models"])


def test_an_abstaining_detector_is_never_a_dissenter():
    # Silence is not disagreement. Counting it as dissent would make a
    # correctly-gated deployment look maximally conflicted.
    out = rd_response(_record_with_abstentions())
    assert all(name not in out["dissenting_models"]
               for name in ("npr", "effnet_b4"))


def test_an_abstention_carrying_a_raw_score_still_reports_it():
    # This is today's deployment: slot D runs, produces 0.266, and is gated
    # out of deciding. A contract that dropped the score would hide the one
    # number showing the detector worked.
    out = rd_response(_record_with_abstentions())
    gend = next(m for m in out["models"] if m["name"] == "gend_clip")
    assert gend["abstained"] is True
    assert gend["score"] == pytest.approx(0.266)
    assert gend["reason"] == "uncalibrated_for_band"


def test_the_score_is_the_recorded_posterior_and_nothing_recomputed():
    rec = _record(posterior=0.8123)
    assert rd_response(rec)["score"] == pytest.approx(0.8123)


def test_out_of_distribution_is_not_flattened_into_authentic():
    # The honest answer for an unfamiliar generator. Mapping it to AUTHENTIC
    # would turn "I have never seen anything like this" into "this is fine".
    out = rd_response(_record(verdict=Verdict.OUT_OF_DISTRIBUTION))
    assert out["verdict"] == "UNKNOWN"


def test_a_model_pointing_the_other_way_is_named_as_dissenting():
    rec = _record(verdict=Verdict.FAKE, llr_total=3.0, evidence=(
        _evidence("npr", llr=4.0), _evidence("blend_seam", llr=-1.0)))
    assert rd_response(rec)["dissenting_models"] == ["blend_seam"]


def test_a_fused_zero_made_of_cancelling_opinions_is_not_reported_as_consensus():
    # +2 and -2 fuse to 0.0, which reads as "no information" and is in fact
    # two detectors contradicting each other outright. Reporting no
    # dissenters there would describe the most conflicted record this system
    # can produce as the calmest.
    rec = _record(verdict=Verdict.INSUFFICIENT_EVIDENCE, llr_total=0.0,
                  evidence=(_evidence("npr", llr=2.0),
                            _evidence("blend_seam", llr=-2.0)))
    assert set(rd_response(rec)["dissenting_models"]) == {"npr", "blend_seam"}


def test_a_record_that_is_all_silence_has_no_dissenters():
    assert rd_response(_record_with_abstentions())["dissenting_models"] == []


def test_the_quality_band_and_schema_version_travel_with_the_verdict():
    # A verdict on a `reject`-band frame is a different claim from the same
    # verdict on a clean one, and the band is what tells them apart.
    out = rd_response(_record())
    assert out["quality_band"] == "medium"
    assert out["schema_version"] == "2"


def test_the_stored_json_form_of_a_record_is_accepted_too():
    # The service reads finished records back out of SQLite as JSON, never
    # as an AuditRecord. A contract that only accepted the dataclass would
    # work in tests and not on the route it exists for.
    import json
    rec = _record(verdict=Verdict.FAKE, posterior=0.9)
    assert rd_response(json.loads(rec.to_json())) == rd_response(rec)
