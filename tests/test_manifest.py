import pytest
from dfd.manifest import AssetRecord, load_manifest, assert_release_clean, NonCommercialAsset

MANIFEST = """
assets:
  yunet_face_detector:
    source: "https://github.com/opencv/opencv_zoo"
    license: "MIT"
    commercial_use: true
    evidence_url: "https://github.com/opencv/opencv_zoo/blob/main/LICENSE"
    date_checked: "2026-09-20"
    checked_by: "kohrohit@gmail.com"
  ffpp_xception_weights:
    source: "DeepfakeBench"
    license: "research-only"
    commercial_use: false
    evidence_url: "https://github.com/SCLBD/DeepfakeBench"
    date_checked: "2026-09-20"
    checked_by: "kohrohit@gmail.com"
"""


def test_load_manifest_parses_records(tmp_path):
    p = tmp_path / "manifest.yaml"
    p.write_text(MANIFEST)
    m = load_manifest(p)
    assert m["yunet_face_detector"].commercial_use is True
    assert m["ffpp_xception_weights"].commercial_use is False


def test_release_gate_passes_on_clean_assets(tmp_path):
    p = tmp_path / "manifest.yaml"
    p.write_text(MANIFEST)
    m = load_manifest(p)
    assert_release_clean(m, ["yunet_face_detector"]) is None


def test_release_gate_blocks_non_commercial_asset(tmp_path):
    p = tmp_path / "manifest.yaml"
    p.write_text(MANIFEST)
    m = load_manifest(p)
    with pytest.raises(NonCommercialAsset) as exc:
        assert_release_clean(m, ["yunet_face_detector", "ffpp_xception_weights"])
    assert "ffpp_xception_weights" in str(exc.value)


def test_unregistered_asset_is_an_error_not_a_pass(tmp_path):
    """An asset with no manifest record must fail closed."""
    p = tmp_path / "manifest.yaml"
    p.write_text(MANIFEST)
    m = load_manifest(p)
    with pytest.raises(NonCommercialAsset):
        assert_release_clean(m, ["some_weights_nobody_registered"])
