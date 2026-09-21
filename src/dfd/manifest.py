"""Asset provenance manifest and the release gate that enforces it (spec §11).

Fails closed: an asset with no record is treated as non-commercial.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .errors import DfdError


class NonCommercialAsset(DfdError):
    """Raised when a release bundle references an asset not cleared for commercial use.

    Joins the `DfdError` hierarchy introduced by Task 20: this fires from the
    same `assert_all_assets_registered` call as `AssetScanEmpty` and
    `DuplicateAssetClaim`, so `except DfdError` must catch all three or it
    silently misses the one that fires on an actual licensing violation.
    """


@dataclass(frozen=True)
class AssetRecord:
    asset_id: str
    source: str
    license: str
    commercial_use: bool
    evidence_url: str
    date_checked: str
    checked_by: str
    # Repo-relative paths this id covers. A filename is not a reliable proxy
    # for a logical asset id (vendor-named files collide — "model.safetensors"
    # says nothing about which model), so the manifest declares the mapping
    # explicitly instead of it being inferred. Defaults to empty so entries
    # written before this field existed keep loading.
    files: tuple[str, ...] = ()


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
            files=tuple(rec.get("files") or ()),
        )
    return out


def assert_release_clean(manifest: dict[str, AssetRecord], asset_ids: list[str]) -> None:
    """Raise if any asset is unregistered or not cleared for commercial use.

    The two causes are reported distinctly: an asset absent from the
    manifest has never been evaluated at all, which is a different fault
    from one that was evaluated and found non-commercial. Collapsing them
    into one message tells a reader debugging a red gate a licensing story
    about a file that may simply be missing a manifest entry.
    """
    unregistered = [a for a in asset_ids if manifest.get(a) is None]
    non_commercial = [a for a in asset_ids
                      if manifest.get(a) is not None
                      and not manifest[a].commercial_use]
    if unregistered or non_commercial:
        parts = []
        if unregistered:
            parts.append("unregistered assets (absent from the manifest): "
                         + ", ".join(sorted(unregistered)))
        if non_commercial:
            parts.append("assets not cleared for commercial release: "
                         + ", ".join(sorted(non_commercial)))
        raise NonCommercialAsset("; ".join(parts))
