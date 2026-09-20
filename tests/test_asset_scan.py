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
    files:
      - "assets/models/good_weights.onnx"
"""


def _tree(tmp_path, *names):
    (tmp_path / "assets" / "models").mkdir(parents=True, exist_ok=True)
    for n in names:
        (tmp_path / "assets" / "models" / n).write_bytes(b"x")
    return tmp_path


def test_discovers_weight_files_by_suffix(tmp_path):
    _tree(tmp_path, "good_weights.onnx", "notes.txt")
    found = discover_assets(tmp_path)
    assert "assets/models/good_weights.onnx" in found
    assert not any("notes" in f for f in found)


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


def test_stem_sharing_files_are_not_conflated(tmp_path):
    """Two files that share a filename stem must be judged independently.

    `assets/models/model.onnx` is registered; `assets/models/model.pt` is
    not. Discovery by path (not stem) must still catch the unregistered one
    rather than letting the registered file's clearance vouch for it.
    """
    _tree(tmp_path, "model.onnx", "model.pt")
    mp = tmp_path / "manifest.yaml"
    mp.write_text("""
assets:
  registered_model:
    source: "s"
    license: "MIT"
    commercial_use: true
    evidence_url: "u"
    date_checked: "2026-09-20"
    checked_by: "k"
    files:
      - "assets/models/model.onnx"
""")
    with pytest.raises(NonCommercialAsset, match="model.pt"):
        assert_all_assets_registered(tmp_path, mp)


def test_registered_asset_whose_filename_differs_from_its_id_passes(tmp_path):
    """A logical id need not match its filename; the manifest's `files` list
    is what makes the connection, not name coincidence."""
    _tree(tmp_path, "face_detection_yunet_2023mar.onnx")
    mp = tmp_path / "manifest.yaml"
    mp.write_text("""
assets:
  yunet_face_detector:
    source: "s"
    license: "MIT"
    commercial_use: true
    evidence_url: "u"
    date_checked: "2026-09-20"
    checked_by: "k"
    files:
      - "assets/models/face_detection_yunet_2023mar.onnx"
""")
    assert assert_all_assets_registered(tmp_path, mp) is None


def test_file_claimed_by_no_entry_is_unregistered(tmp_path):
    _tree(tmp_path, "orphan.onnx")
    mp = tmp_path / "manifest.yaml"
    mp.write_text(MANIFEST)  # declares good_weights, not orphan
    with pytest.raises(NonCommercialAsset, match="orphan"):
        assert_all_assets_registered(tmp_path, mp)


def test_symlinked_file_is_not_admitted_as_an_asset(tmp_path):
    """A symlink must not be admitted under the name of the link itself —
    that would let a scan report an id nobody could ever register correctly,
    since the manifest declares real paths, not link paths."""
    (tmp_path / "assets" / "models").mkdir(parents=True)
    target = tmp_path / "payload.dat"
    target.write_bytes(b"x")
    link = tmp_path / "assets" / "models" / "linked.onnx"
    link.symlink_to(target)
    assert discover_assets(tmp_path) == []
