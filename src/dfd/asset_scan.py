"""Enumerate assets on disk so the release gate cannot pass vacuously.

`assert_release_clean` judges only the ids it is handed, so handing it nothing
returns cleanly. That is a gate guaranteeing nothing. This module supplies the
list from the filesystem instead of from a human's memory.

Discovery returns repo-relative *paths*, not filename stems. A stem is not a
reliable proxy for a logical asset id: vendor-named files collide (a
HuggingFace export is routinely `model.safetensors` regardless of which model
it is), and deduplicating by stem through a set would let one registered file
vouch for an unrelated, unregistered file that happens to share a name. The
manifest instead declares, per id, which paths it covers (`AssetRecord.files`),
and this module matches discovered paths against that declaration.
"""
from __future__ import annotations

import os
from pathlib import Path

from .errors import DfdError
from .manifest import AssetRecord, assert_release_clean, load_manifest

# Extensions that carry model weights or dataset payloads.
ASSET_SUFFIXES = (".onnx", ".pt", ".pth", ".safetensors", ".tflite", ".bin", ".npz")


def discover_assets(root: str | Path) -> list[str]:
    """Repo-relative paths of every weight-like file under `root`.

    Walks with `followlinks=False` and skips symlinked files explicitly: a
    symlink would otherwise admit an id named after the link rather than its
    target, and a symlinked directory cycle would hang a scan that is
    supposed to fail closed, not hang open.
    """
    root = Path(root)
    found: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            path = Path(dirpath) / name
            if path.is_symlink():
                continue
            if path.suffix.lower() in ASSET_SUFFIXES:
                found.append(path.relative_to(root).as_posix())
    return sorted(found)


class AssetScanEmpty(DfdError):
    """The scan found no assets, so it can certify nothing.

    Joins the `DfdError` hierarchy introduced by Task 20.
    """


class DuplicateAssetClaim(DfdError):
    """More than one manifest entry declares the same file.

    Two records for one file mean at least one of them is wrong — a human
    recorded that file's provenance twice, possibly with different licence
    verdicts. Silently keeping the stricter record would hide that authoring
    error rather than surface it, and would make the gate's verdict depend on
    a reconciliation rule nobody reading the manifest can see. This is raised
    even when the two records happen to agree today: agreement is
    coincidental, not a guarantee, and a later edit to one copy without the
    other would then diverge undetected.

    Joins the `DfdError` hierarchy introduced by Task 20.
    """


def _build_claims(manifest: dict[str, AssetRecord]) -> dict[str, str]:
    """Map each manifest-declared file path to the single id that claims it.

    Raises DuplicateAssetClaim if any path is declared by more than one id.
    This runs over every declaration in the manifest, not just paths that
    happen to be discovered on disk right now: a manifest with a duplicate
    claim is malformed regardless of what is currently checked out.
    """
    claimants: dict[str, list[str]] = {}
    for asset_id, rec in manifest.items():
        for f in rec.files:
            claimants.setdefault(f, []).append(asset_id)
    duplicates = {f: ids for f, ids in claimants.items() if len(ids) > 1}
    if duplicates:
        parts = [f"{f!r} claimed by {', '.join(sorted(ids))}"
                 for f, ids in sorted(duplicates.items())]
        raise DuplicateAssetClaim(
            "manifest declares the same file under more than one id: "
            + "; ".join(parts))
    return {f: ids[0] for f, ids in claimants.items()}


def _resolve_ids(manifest: dict[str, AssetRecord], discovered: list[str]) -> list[str]:
    """Map each discovered path to the manifest id whose `files` claims it.

    A path no entry claims is left as-is: it cannot equal a real asset id (ids
    are logical names, not paths), so handing it to `assert_release_clean`
    makes that function's own `manifest.get(aid) is None` branch report it as
    unregistered — by the path that was actually found on disk, which is more
    useful than a bare stem when two files share one.
    """
    claims = _build_claims(manifest)
    return [claims.get(path, path) for path in discovered]


def assert_all_assets_registered(root: str | Path,
                                 manifest_path: str | Path,
                                 allow_empty: bool = False) -> None:
    """Raise unless every asset on disk is registered and commercially clear.

    Raises:
        AssetScanEmpty: nothing was discovered and `allow_empty` is False. An
            empty scan passing `assert_release_clean` would return cleanly
            having examined no files — the vacuity this module exists to
            remove, one level up. On a fresh checkout this is the normal case,
            because weight files are gitignored, so the caller must opt into
            it deliberately rather than inherit it by silence.
        DuplicateAssetClaim: the manifest declares the same file under more
            than one id — an authoring error that must be fixed by hand, not
            silently resolved in either direction.
        NonCommercialAsset: a discovered path is claimed by no manifest entry,
            or claimed by one without commercial clearance.
    """
    discovered = discover_assets(root)
    if not discovered and not allow_empty:
        raise AssetScanEmpty(
            f"asset scan found no assets under {root} matching "
            f"{', '.join(ASSET_SUFFIXES)}; a gate that examined nothing cannot "
            "certify a release. Provide the assets, or pass allow_empty=True "
            "to record that this environment deliberately has none.")
    manifest = load_manifest(manifest_path)
    assert_release_clean(manifest, _resolve_ids(manifest, discovered))
