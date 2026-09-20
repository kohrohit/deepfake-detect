import dataclasses
import json

import numpy as np
import pytest

from dfd.audit import build_audit_record, record_digest
from dfd.errors import DfdError, InvalidInput
from dfd.types import Evidence, Verdict

FIXED_TIME = "2026-09-20T10:00:00+00:00"


def _ev(name, llr, abstained=False, reason="ok"):
    return Evidence(detector=name, detector_version="1.0", llr=llr,
                    raw_score=0.5, uncertainty=0.0,
                    abstained=abstained, reason=reason)


def _record(**kw):
    base = dict(
        sample_id="s1", input_sha256="a" * 64, verdict=Verdict.FAKE,
        llr_total=3.2, posterior=0.96,
        evidence=[_ev("npr", 2.0), _ev("sbi", 1.2)],
        quality_band="high", ood_score=0.1,
        policy_version="policy-1", threshold=1.0,
        model_versions={"npr": "0.1.0", "sbi": "0.1.0"},
        created_at=FIXED_TIME,
    )
    base.update(kw)
    return build_audit_record(**base)


def test_record_carries_every_field_a_regulator_would_ask_for():
    r = _record()
    for name in ("sample_id", "input_sha256", "verdict", "llr_total",
                 "posterior", "policy_version", "threshold", "model_versions",
                 "quality_band", "ood_score", "created_at", "schema_version"):
        assert getattr(r, name) is not None, name


def test_rebinding_a_field_raises():
    with pytest.raises(dataclasses.FrozenInstanceError):
        _record().verdict = Verdict.REAL


def test_model_versions_cannot_be_mutated_in_place():
    """`frozen=True` is shallow: a plain dict field stays writable and
    mutating it changes the digest, so the record is not immutable at all."""
    r = _record()
    with pytest.raises(TypeError):
        r.model_versions["npr"] = "tampered"


def test_evidence_rows_cannot_be_appended_to():
    r = _record()
    with pytest.raises(AttributeError):
        r.evidence.append({"detector": "ghost"})


def test_per_detector_llrs_are_preserved_including_abstentions():
    """An abstention is evidence about the system, not an absence of evidence."""
    r = _record(evidence=[_ev("npr", 2.0),
                          _ev("sbi", 0.0, abstained=True,
                              reason="weights_absent")])
    got = {e["detector"]: e for e in r.evidence}
    assert got["sbi"]["abstained"] is True
    assert got["sbi"]["reason"] == "weights_absent"
    assert got["npr"]["llr"] == 2.0


def test_to_json_round_trips():
    d = json.loads(_record().to_json())
    assert d["sample_id"] == "s1"
    assert d["verdict"] == "fake"
    assert len(d["evidence"]) == 2


def test_digest_is_stable_for_identical_records():
    assert record_digest(_record()) == record_digest(_record())


@pytest.mark.parametrize("field,value", [
    ("verdict", Verdict.REAL),
    ("llr_total", 3.3),
    ("posterior", 0.95),
    ("sample_id", "s2"),
    ("input_sha256", "b" * 64),
    ("quality_band", "low"),
    ("ood_score", 0.2),
    ("policy_version", "policy-2"),
    ("threshold", 1.5),
    ("model_versions", {"npr": "0.2.0", "sbi": "0.1.0"}),
    ("created_at", "1999-01-01T00:00:00+00:00"),
])
def test_digest_changes_when_any_field_changes(field, value):
    """Tamper-evidence, field by field. `created_at` is in this list
    deliberately: excluding it makes backdating a decision invisible."""
    assert record_digest(_record()) != record_digest(_record(**{field: value}))


def test_digest_changes_when_evidence_changes():
    assert record_digest(_record()) != record_digest(
        _record(evidence=[_ev("npr", 9.9), _ev("sbi", 1.2)]))


def test_a_value_that_cannot_be_serialised_is_refused_not_stringified():
    """`json.dumps(default=str)` never raises, so an image passed by mistake
    would be absorbed into the record instead of rejected."""
    with pytest.raises((InvalidInput, TypeError)):
        _record(model_versions={"npr": np.zeros((4, 4), dtype=np.uint8)}).to_json()


def test_the_record_references_its_input_by_hash_only():
    d = json.loads(_record().to_json())
    assert d["input_sha256"] == "a" * 64
    assert not any(k.startswith("image") or k.endswith("bytes") for k in d)


def test_rejects_a_malformed_input_hash():
    with pytest.raises(InvalidInput, match="64 lowercase hex"):
        _record(input_sha256="not-a-hash")


def test_rejects_an_uppercase_hash():
    """Case matters: the same digest in two cases would give two records."""
    with pytest.raises(InvalidInput, match="64 lowercase hex"):
        _record(input_sha256="A" * 64)


def test_rejects_an_empty_sample_id():
    with pytest.raises(InvalidInput, match="sample_id"):
        _record(sample_id="")


def test_invalid_input_is_a_dfd_error():
    """One catchable root for every error this package raises."""
    assert issubclass(InvalidInput, DfdError)
