"""export_fairface turns FairFace parquet shards into the capture layout.

The point of exporting rather than adding a second corpus reader: every stage
downstream — `load_capture_sessions`, `build_face_pool` with its real face
detection, deduplication and ROI clamping, `build_sbi_corpus` — is already
written and tested against that layout. A parallel FairFace path would be a
second, less-tested copy of all of it.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from corpora.captures import load_capture_sessions
from training.export_fairface import export_fairface


def _jpeg(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    img = Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _shard(path: Path, payloads: list[bytes]) -> None:
    table = pa.table({
        "image": [{"bytes": b, "path": f"{i}.jpg"}
                  for i, b in enumerate(payloads)],
        "age": [3] * len(payloads),
        "gender": [0] * len(payloads),
        "race": [1] * len(payloads),
        "service_test": [True] * len(payloads),
    })
    pq.write_table(table, path)


@pytest.fixture()
def shards(tmp_path: Path) -> tuple[Path, list[bytes]]:
    src = tmp_path / "0.25"
    src.mkdir()
    payloads = [_jpeg(i) for i in range(4)]
    _shard(src / "train-00000-of-00001.parquet", payloads)
    return src, payloads


def test_frames_are_written_verbatim_never_re_encoded(
        shards: tuple[Path, list[bytes]], tmp_path: Path) -> None:
    """The parquet holds the ORIGINAL JPEG bytes, so they are copied, not decoded.

    Re-encoding would stack a second generation of JPEG quantisation onto every
    real face while the pseudo-fake made from it in `corpora.sbi` is blended
    from the decoded pixels. The seam detector reads high-frequency residuals,
    so a systematic compression difference between the real and fake halves of
    the corpus is exactly the kind of shortcut it would learn instead of the
    seam — and the resulting AUC would look excellent and mean nothing.
    """
    src, payloads = shards
    export_fairface(src, tmp_path / "out", limit=4)

    written = sorted((tmp_path / "out").glob("*/frame_00.jpg"))
    assert len(written) == 4
    assert [p.read_bytes() for p in written] == payloads


def test_every_exported_session_is_labelled_genuine(
        shards: tuple[Path, list[bytes]], tmp_path: Path) -> None:
    """FairFace is real faces only. A swapped=True row here would be a fake with
    no swap in it, and `build_sbi_corpus` would refuse the crop as
    evaluation-only — silently shrinking the pool instead of failing."""
    src, _ = shards
    export_fairface(src, tmp_path / "out", limit=4)

    sessions = load_capture_sessions(tmp_path / "out")
    assert len(sessions) == 4
    assert all(not s.swapped for s in sessions)
    assert all(s.frame_count == 1 for s in sessions)


def test_limit_caps_the_export(shards: tuple[Path, list[bytes]],
                               tmp_path: Path) -> None:
    src, _ = shards
    report = export_fairface(src, tmp_path / "out", limit=2)
    assert report["exported"] == 2
    assert len(list((tmp_path / "out").glob("*/results.json"))) == 2


def test_provenance_and_demographics_travel_with_every_frame(
        shards: tuple[Path, list[bytes]], tmp_path: Path) -> None:
    """Licence provenance because `assets/manifest.yaml` fails closed on
    anything unattributed, and demographics because acceptance criterion 11
    (demographic parity) has a guard built and no labelled corpus to run it
    against. FairFace carries age, gender and race per row; dropping them here
    would throw away the only thing that could close that criterion."""
    src, _ = shards
    export_fairface(src, tmp_path / "out", limit=1)

    record = json.loads(next((tmp_path / "out").glob("*/results.json")).read_text())
    assert record["provenance"]["dataset"] == "FairFace"
    assert record["provenance"]["license"] == "CC BY 4.0"
    # The fixture stores age=3, gender=0, race=1. Decoded against the dataset
    # card's class_label orderings (verified at source 2026-09-22) that is
    # 20-29 / Male / Indian -- NOT 3-9, which is index 1. The off-by-one is
    # worth a comment because a demographic-parity slice computed on shifted
    # age bands would be wrong in a way no test downstream could detect.
    assert record["demographics"] == {
        "age": "20-29", "gender": "Male", "race": "Indian"}
