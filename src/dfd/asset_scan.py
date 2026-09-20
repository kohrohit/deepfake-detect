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


class AssetScanEmpty(Exception):
    """The scan found no assets, so it can certify nothing.

    Joins the `DfdError` hierarchy when Task 20 introduces it.
    """


def _resolve_ids(manifest: dict[str, AssetRecord], discovered: list[str]) -> list[str]:
    """Map each discovered path to the manifest id whose `files` claims it.

    A path no entry claims is left as-is: it cannot equal a real asset id (ids
    are logical names, not paths), so handing it to `assert_release_clean`
    makes that function's own `manifest.get(aid) is None` branch report it as
    unregistered — by the path that was actually found on disk, which is more
    useful than a bare stem when two files share one.
    """
    claims: dict[str, str] = {}
    for asset_id, rec in manifest.items():
        for f in rec.files:
            claims[f] = asset_id
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
