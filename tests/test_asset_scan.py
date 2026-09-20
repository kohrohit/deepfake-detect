import pytest
from dfd.asset_scan import (
    ASSET_SUFFIXES, AssetScanEmpty, assert_all_assets_registered,
    discover_assets,
)
from dfd.manifest import NonCommercialAsset, assert_release_clean, load_manifest

MANIFEST = """
assets:
  good_weights:
    source: "s"
    license: "MIT"
    commercial_use: true
    evidence_url: "u"
    date_checked: "2026-09-20"
    checked_by: "k"
"""


def _tree(tmp_path, *names):
    (tmp_path / "assets" / "models").mkdir(parents=True, exist_ok=True)
    for n in names:
        (tmp_path / "assets" / "models" / n).write_bytes(b"x")
    return tmp_path


def test_discovers_weight_files_by_suffix(tmp_path):
    _tree(tmp_path, "good_weights.onnx", "notes.txt")
    found = discover_assets(tmp_path)
    assert "good_weights" in found
    assert "notes" not in found


def test_every_declared_suffix_is_discovered(tmp_path):
    names = [f"a{i}{s}" for i, s in enumerate(sorted(ASSET_SUFFIXES))]
    _tree(tmp_path, *names)
    found = set(discover_assets(tmp_path))
    assert len(found) == len(ASSET_SUFFIXES)


def test_passes_when_every_discovered_asset_is_registered(tmp_path):
    _tree(tmp_path, "good_weights.onnx")
    mp = tmp_path / "manifest.yaml"
    mp.write_text(MANIFEST)
    assert assert_all_assets_registered(tmp_path, mp) is None


def test_raises_on_an_asset_present_on_disk_but_absent_from_the_manifest(tmp_path):
    """The whole point: a weight file nobody registered must fail the build."""
    _tree(tmp_path, "good_weights.onnx", "sneaky_weights.pt")
    mp = tmp_path / "manifest.yaml"
    mp.write_text(MANIFEST)
    with pytest.raises(NonCommercialAsset, match="sneaky_weights"):
        assert_all_assets_registered(tmp_path, mp)


def test_an_empty_scan_fails_the_gate(tmp_path):
    """The defect this task exists to remove, at the level it actually bites.

    Handing `assert_release_clean` an empty list returns cleanly, so a scan
    that finds nothing would certify a clean release having examined no files.
    On a fresh checkout that is the normal case: weight files are gitignored.
    """
    (tmp_path / "assets").mkdir()
    mp = tmp_path / "manifest.yaml"
    mp.write_text(MANIFEST)
    with pytest.raises(AssetScanEmpty, match="found no assets"):
        assert_all_assets_registered(tmp_path, mp)


def test_the_empty_scan_message_names_where_it_looked(tmp_path):
    """A gate that fails must say enough to be fixed or waived deliberately."""
    (tmp_path / "assets").mkdir()
    mp = tmp_path / "manifest.yaml"
    mp.write_text(MANIFEST)
    with pytest.raises(AssetScanEmpty) as exc:
        assert_all_assets_registered(tmp_path, mp)
    assert str(tmp_path) in str(exc.value)
    assert ".onnx" in str(exc.value)


def test_an_empty_scan_can_be_waived_only_explicitly(tmp_path):
    (tmp_path / "assets").mkdir()
    mp = tmp_path / "manifest.yaml"
    mp.write_text(MANIFEST)
    assert assert_all_assets_registered(tmp_path, mp, allow_empty=True) is None


def test_empty_tree_discovers_nothing(tmp_path):
    (tmp_path / "assets").mkdir()
    assert discover_assets(tmp_path) == []


def test_unregistered_and_non_commercial_are_reported_distinctly(tmp_path):
    """"I have never heard of this file" is not "this file's licence forbids
    commercial use", and a reader debugging a red gate should not be told a
    licensing story about a file that is merely absent from the manifest."""
    mp = tmp_path / "manifest.yaml"
    mp.write_text(MANIFEST)
    with pytest.raises(NonCommercialAsset) as exc:
        assert_release_clean(load_manifest(mp), ["nobody_registered_this"])
    message = str(exc.value)
    assert "unregistered" in message.lower()
    assert "nobody_registered_this" in message
