"""Immutable per-decision audit record (spec §7.2).

Makes a rejection defensible: what was decided, by which model versions, on
what evidence, under which policy. References the input by SHA-256 and never
carries image bytes — the record is retained far longer than the media, and
BFSI face data is sensitive personal data under India's DPDP Act.

Immutable means immutable in depth: `frozen=True` alone leaves dict and list
fields writable, and mutating one changes the digest, which would let a record
be altered after the fact and re-digested to match.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .errors import InvalidInput
from .types import Evidence, Verdict

logger = logging.getLogger(__name__)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
AUDIT_SCHEMA_VERSION = "1"

# The only leaf types a record is allowed to carry. Anything else (bytes,
# a numpy array, an arbitrary object) is refused at build time, not merely
# at serialisation time: a record that never accepted the value cannot leak
# it via a repr, a log line, or a pickle taken before to_json() is called.
_SERIALISABLE_LEAF_TYPES = (str, int, float, bool, type(None))


def _freeze(value: Any) -> Any:
    """Deep-freeze the containers a record holds."""
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _validate_serialisable(value: Any, *, field: str) -> None:
    """Reject a value this record cannot honestly carry, before it is ever
    assigned to a field. Mirrors `to_json`'s refusal to stringify, but at
    the build boundary rather than the serialisation boundary."""
    if isinstance(value, Mapping):
        for k, v in value.items():
            _validate_serialisable(v, field=f"{field}[{k!r}]")
        return
    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _validate_serialisable(v, field=f"{field}[{i}]")
        return
    if not isinstance(value, _SERIALISABLE_LEAF_TYPES):
        raise InvalidInput(
            f"{field} holds a {type(value).__name__}, which this record "
            "cannot serialise; only str, int, float, bool, and None are "
            "accepted")


def _validate_evidence(e: Evidence) -> None:
    """Reject an `Evidence` row this record cannot honestly carry, before the
    row is built from it.

    `Evidence` is a plain dataclass with no runtime type enforcement, and the
    row-building loop below copies `detector`, `detector_version`, and
    `reason` straight into the row. Without this check, a caller could
    construct `Evidence(..., reason=b"...")`, `build_audit_record` would
    succeed and the record would hold raw bytes in memory, and only a later
    `to_json()` call would raise -- the identical build-succeeds/serialise-
    fails split `_validate_serialisable` closes for `model_versions`,
    relocated to a field that helper is never called on.
    """
    if not isinstance(e.detector, str):
        raise InvalidInput(
            f"evidence.detector must be a string, got {type(e.detector).__name__}")
    if not isinstance(e.detector_version, str):
        raise InvalidInput(
            "evidence.detector_version must be a string, got "
            f"{type(e.detector_version).__name__}")
    if not isinstance(e.reason, str):
        raise InvalidInput(
            f"evidence.reason must be a string, got {type(e.reason).__name__}")
    if not isinstance(e.llr, (int, float)):
        raise InvalidInput(
            f"evidence.llr must be numeric, got {type(e.llr).__name__}")
    if e.raw_score is not None and not isinstance(e.raw_score, (int, float)):
        raise InvalidInput(
            "evidence.raw_score must be numeric or None, got "
            f"{type(e.raw_score).__name__}")


def _thaw(value: Any) -> Any:
    """Plain-Python view for serialisation. No `default=` fallback: a value
    this cannot render must raise rather than be silently stringified."""
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


@dataclass(frozen=True)
class AuditRecord:
    schema_version: str
    sample_id: str
    input_sha256: str
    verdict: str
    llr_total: float
    posterior: float
    evidence: tuple
    quality_band: str
    ood_score: float
    policy_version: str
    threshold: float
    model_versions: Mapping[str, str]
    created_at: str

    def to_json(self) -> str:
        """Serialise deterministically (sorted keys) so digests are comparable.

        `dataclasses.asdict` is deliberately not used: it deep-copies every
        field value, and a `MappingProxyType` cannot be deep-copied.

        `allow_nan=False`: the default (`True`) emits the bare tokens `NaN` /
        `Infinity`, which Python's own parser accepts but no conforming JSON
        parser does. For a record whose entire purpose is to be re-read by
        someone else's tooling, silently emitting non-conformant JSON is the
        same wrong failure direction as `default=str` — just for numbers
        instead of objects.

        Raises:
            TypeError: if any field holds a value JSON cannot represent.
            InvalidInput: if any field holds NaN or +/-Infinity.
        """
        payload = {f.name: _thaw(getattr(self, f.name)) for f in fields(self)}
        try:
            return json.dumps(payload, sort_keys=True, allow_nan=False)
        except ValueError as e:
            raise InvalidInput(
                "record contains a non-finite float (NaN or Infinity), "
                "which is not valid JSON: " + str(e)) from e


def build_audit_record(
    sample_id: str,
    input_sha256: str,
    verdict: Verdict,
    llr_total: float,
    posterior: float,
    evidence: Sequence[Evidence],
    quality_band: str,
    ood_score: float,
    policy_version: str,
    threshold: float,
    model_versions: Mapping[str, str],
    created_at: str | None = None,
) -> AuditRecord:
    """Build an immutable decision record.

    `created_at` is injectable so that two records describing the same decision
    are genuinely identical. It is covered by `record_digest`: a timestamp
    outside the digest makes backdating a decision invisible.

    Every string-typed field is validated to actually be a string rather than
    relying on the field happening to hold a scalar: `mypy --strict` does not
    enforce types at runtime, and a caller passing e.g. a list for
    `quality_band` would otherwise be silently accepted, stay mutable, and
    change the digest on later mutation — the same defect the brief calls out
    for `model_versions`, just for a field `_freeze` was never applied to.

    Raises:
        InvalidInput: if `sample_id`, `quality_band`, `policy_version`, or
            `created_at` is not a string (or `sample_id` is empty); if
            `input_sha256` is not a lowercase 64-character hex digest; if
            `created_at` is not a valid ISO-8601 timestamp; if
            `model_versions` holds a value this record cannot serialise; or
            if any evidence row's `detector`, `detector_version`, or `reason`
            is not a string, or its `llr`/`raw_score` is not numeric.
    """
    if not isinstance(sample_id, str) or not sample_id:
        raise InvalidInput(f"sample_id must be a non-empty string, got {sample_id!r}")
    if not isinstance(input_sha256, str) or not _SHA256_RE.match(input_sha256):
        raise InvalidInput(
            "input_sha256 must be 64 lowercase hex characters, got "
            f"{input_sha256!r}")
    if not isinstance(quality_band, str):
        raise InvalidInput(
            f"quality_band must be a string, got {type(quality_band).__name__}")
    if not isinstance(policy_version, str):
        raise InvalidInput(
            f"policy_version must be a string, got {type(policy_version).__name__}")

    _validate_serialisable(model_versions, field="model_versions")
    for e in evidence:
        _validate_evidence(e)

    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    elif not isinstance(created_at, str):
        raise InvalidInput(
            f"created_at must be an ISO-8601 string, got {type(created_at).__name__}")
    else:
        try:
            datetime.fromisoformat(created_at)
        except ValueError as e:
            raise InvalidInput(
                f"created_at must be a valid ISO-8601 timestamp, got {created_at!r}"
            ) from e

    rows = tuple(
        MappingProxyType({
            "detector": e.detector,
            "version": e.detector_version,
            "llr": float(e.llr),
            "raw_score": None if e.raw_score is None else float(e.raw_score),
            "abstained": bool(e.abstained),
            "reason": e.reason,
        })
        for e in evidence
    )
    record = AuditRecord(
        schema_version=AUDIT_SCHEMA_VERSION,
        sample_id=sample_id,
        input_sha256=input_sha256,
        verdict=verdict.value if isinstance(verdict, Verdict) else str(verdict),
        llr_total=float(llr_total),
        posterior=float(posterior),
        evidence=rows,
        quality_band=quality_band,
        ood_score=float(ood_score),
        policy_version=policy_version,
        threshold=float(threshold),
        model_versions=_freeze(dict(model_versions)),
        created_at=created_at,
    )
    logger.info("audit record built: sample=%s verdict=%s detectors=%d",
                sample_id, record.verdict, len(rows))
    return record


def record_digest(record: AuditRecord) -> str:
    """Tamper-evident digest over the whole record, timestamp included."""
    return hashlib.sha256(record.to_json().encode()).hexdigest()
