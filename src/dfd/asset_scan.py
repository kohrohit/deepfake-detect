"""Enumerate assets on disk so the release gate cannot pass vacuously.

`assert_release_clean` judges only the ids it is handed, so handing it nothing
returns cleanly. That is a gate guaranteeing nothing. This module supplies the
list from the filesystem instead of from a human's memory.
"""
from __future__ import annotations

from pathlib import Path

from .manifest import assert_release_clean, load_manifest

# Extensions that carry model weights or dataset payloads.
ASSET_SUFFIXES = (".onnx", ".pt", ".pth", ".safetensors", ".tflite", ".bin", ".npz")


def discover_assets(root: str | Path) -> list[str]:
    """Asset ids (filename stems) for every weight-like file under `root`."""
    found: set[str] = set()
    for path in Path(root).rglob("*"):
        if path.is_file() and path.suffix.lower() in ASSET_SUFFIXES:
            found.add(path.stem)
    return sorted(found)


class AssetScanEmpty(Exception):
    """The scan found no assets, so it can certify nothing.

    Joins the `DfdError` hierarchy when Task 20 introduces it.
    """


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
        NonCommercialAsset: a discovered asset is unregistered, or registered
            without commercial clearance.
    """
    discovered = discover_assets(root)
    if not discovered and not allow_empty:
        raise AssetScanEmpty(
            f"asset scan found no assets under {root} matching "
            f"{', '.join(ASSET_SUFFIXES)}; a gate that examined nothing cannot "
            "certify a release. Provide the assets, or pass allow_empty=True "
            "to record that this environment deliberately has none.")
    assert_release_clean(load_manifest(manifest_path), discovered)
