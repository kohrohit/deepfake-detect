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


def _freeze(value: Any) -> Any:
    """Deep-freeze the containers a record holds."""
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


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

        Raises:
            TypeError: if any field holds a value JSON cannot represent.
        """
        payload = {f.name: _thaw(getattr(self, f.name)) for f in fields(self)}
        return json.dumps(payload, sort_keys=True)


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

    Raises:
        InvalidInput: if `sample_id` is empty or `input_sha256` is not a
            lowercase 64-character hex digest.
    """
    if not sample_id:
        raise InvalidInput("sample_id must be a non-empty string")
    if not _SHA256_RE.match(input_sha256 or ""):
        raise InvalidInput(
            "input_sha256 must be 64 lowercase hex characters, got "
            f"{input_sha256!r}")

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
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
    )
    logger.info("audit record built: sample=%s verdict=%s detectors=%d",
                sample_id, record.verdict, len(rows))
    return record


def record_digest(record: AuditRecord) -> str:
    """Tamper-evident digest over the whole record, timestamp included."""
    return hashlib.sha256(record.to_json().encode()).hexdigest()
