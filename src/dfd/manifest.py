"""Asset provenance manifest and the release gate that enforces it (spec §11).

Fails closed: an asset with no record is treated as non-commercial.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class NonCommercialAsset(Exception):
    """Raised when a release bundle references an asset not cleared for commercial use."""


@dataclass(frozen=True)
class AssetRecord:
    asset_id: str
    source: str
    license: str
    commercial_use: bool
    evidence_url: str
    date_checked: str
    checked_by: str


def load_manifest(path: str | Path) -> dict[str, AssetRecord]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    out: dict[str, AssetRecord] = {}
    for asset_id, rec in (data.get("assets") or {}).items():
        out[asset_id] = AssetRecord(
            asset_id=asset_id,
            source=rec["source"],
            license=rec["license"],
            commercial_use=bool(rec["commercial_use"]),
            evidence_url=rec["evidence_url"],
            date_checked=rec["date_checked"],
            checked_by=rec["checked_by"],
        )
    return out


def assert_release_clean(manifest: dict[str, AssetRecord], asset_ids: list[str]) -> None:
    """Raise if any asset is unregistered or not cleared for commercial use."""
    bad: list[str] = []
    for aid in asset_ids:
        rec = manifest.get(aid)
        if rec is None or not rec.commercial_use:
            bad.append(aid)
    if bad:
        raise NonCommercialAsset(
            "assets not cleared for commercial release: " + ", ".join(sorted(bad))
        )
