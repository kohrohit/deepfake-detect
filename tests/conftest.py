"""Pytest configuration and fixtures."""
import csv
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

# Add project root to sys.path so bench module can be imported
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dfd.detectors.base import Registry, SyntheticDetector  # noqa: E402
from dfd.faces import FaceBox  # noqa: E402
from dfd.types import RawScore  # noqa: E402

# --- corpora.dfeval fixtures -------------------------------------------
#
# Live here, at the top level, rather than in tests/corpora/conftest.py:
# Task 5 (tests/bench/) needs the same dfeval_root and stub_detect, and a
# fixture copied into a second conftest is a fixture that drifts.

#: Mean pixel intensity above which `stub_detect_sometimes` reports no face.
#: Fixture images are built either side of it: ~75 for a "dark" image a face
#: is found in, ~230 for a "bright" one that stands in for non-facial
#: in-the-wild content (a screenshot, a caption card, a blank frame).
_DFEVAL_NO_FACE_MEAN_THRESHOLD = 200.0


def _dfeval_face_box() -> FaceBox:
    lms = np.array([[30.0, 40.0], [60.0, 40.0], [45.0, 55.0],
                    [33.0, 70.0], [57.0, 70.0]])
    return FaceBox(x=20, y=25, w=50, h=60, landmarks=lms, score=0.99)


def _write_dfeval_corpus(root: Path, rows: list[dict[str, Any]]) -> None:
    """Write a `corpora.dfeval` fixture: images, plus the manifest it reads.

    `rows` is a list of {item_id, filename, label, domain, image} dicts.
    `image` is the array written to `filename`, so callers can give two
    rows the exact same array to build a duplicate or a contradiction.
    """
    root.mkdir(parents=True, exist_ok=True)
    with (root / "metadata.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["item_id", "filename", "label", "domain"])
        writer.writeheader()
        for row in rows:
            path = root / row["filename"]
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), row["image"])
            writer.writerow({"item_id": row["item_id"],
                              "filename": row["filename"],
                              "label": row["label"],
                              "domain": row["domain"]})


@pytest.fixture
def stub_detect():
    """Always finds the one fixed face."""
    def _detect(_frame):
        return [_dfeval_face_box()]
    return _detect


@pytest.fixture
def stub_detect_sometimes():
    """Finds the fixed face, except in images bright enough to be the
    non-facial content an in-the-wild corpus actually contains."""
    def _detect(frame):
        if frame.mean() > _DFEVAL_NO_FACE_MEAN_THRESHOLD:
            return []
        return [_dfeval_face_box()]
    return _detect


@pytest.fixture
def dfeval_root(tmp_path: Path) -> Path:
    """Six images, three domains, both labels -- the ordinary case."""
    root = tmp_path / "dfeval"
    rng = np.random.default_rng(0)
    domains = ["x.com", "reddit.com", "tiktok.com"]
    rows = []
    for i, label in enumerate(["real", "real", "real", "fake", "fake", "fake"]):
        rows.append({
            "item_id": f"{label}{i}",
            "filename": f"{label}{i}.jpg",
            "label": label,
            "domain": domains[i % len(domains)],
            "image": rng.integers(0, 150, (120, 100, 3), dtype=np.uint8),
        })
    _write_dfeval_corpus(root, rows)
    return root


@pytest.fixture
def dfeval_root_with_dupe(tmp_path: Path) -> Path:
    """One image's exact decoded pixels, repeated under a second name and
    the SAME label -- the ordinary in-the-wild duplicate."""
    root = tmp_path / "dfeval_dupe"
    rng = np.random.default_rng(1)
    shared = rng.integers(0, 150, (120, 100, 3), dtype=np.uint8)
    rows = [
        {"item_id": "r1", "filename": "a.jpg", "label": "real",
         "domain": "x.com", "image": shared},
        {"item_id": "r2", "filename": "b.jpg", "label": "real",
         "domain": "x.com", "image": rng.integers(0, 150, (120, 100, 3), dtype=np.uint8)},
        {"item_id": "r3", "filename": "c.jpg", "label": "real",
         "domain": "reddit.com", "image": shared},
        {"item_id": "f1", "filename": "d.jpg", "label": "fake",
         "domain": "reddit.com", "image": rng.integers(0, 150, (120, 100, 3), dtype=np.uint8)},
    ]
    _write_dfeval_corpus(root, rows)
    return root


@pytest.fixture
def dfeval_root_with_contradiction(tmp_path: Path) -> Path:
    """One image's exact decoded pixels, repeated under the OPPOSITE label."""
    root = tmp_path / "dfeval_contradiction"
    rng = np.random.default_rng(2)
    shared = rng.integers(0, 150, (120, 100, 3), dtype=np.uint8)
    rows = [
        {"item_id": "r1", "filename": "a.jpg", "label": "real",
         "domain": "x.com", "image": shared},
        {"item_id": "f1", "filename": "b.jpg", "label": "fake",
         "domain": "x.com", "image": shared},
    ]
    _write_dfeval_corpus(root, rows)
    return root


@pytest.fixture
def dfeval_root_mixed(tmp_path: Path) -> Path:
    """Both labels, and some images bright enough that `stub_detect_sometimes`
    finds no face in them -- the in-the-wild case this corpus exists for."""
    root = tmp_path / "dfeval_mixed"
    rng = np.random.default_rng(3)
    rows = [
        {"item_id": "r1", "filename": "a.jpg", "label": "real", "domain": "x.com",
         "image": rng.integers(0, 150, (120, 100, 3), dtype=np.uint8)},
        {"item_id": "r2", "filename": "b.jpg", "label": "real", "domain": "reddit.com",
         "image": rng.integers(210, 250, (120, 100, 3), dtype=np.uint8)},
        {"item_id": "f1", "filename": "c.jpg", "label": "fake", "domain": "x.com",
         "image": rng.integers(0, 150, (120, 100, 3), dtype=np.uint8)},
        {"item_id": "f2", "filename": "d.jpg", "label": "fake", "domain": "reddit.com",
         "image": rng.integers(210, 250, (120, 100, 3), dtype=np.uint8)},
    ]
    _write_dfeval_corpus(root, rows)
    return root


# --- bench.eval_dfeval fixtures ----------------------------------------
#
# Both registries hold a detector literally named `npr`, because
# `bench.eval_dfeval` identifies slot C by name: the oracle gate exists to
# check OUR slot C against the number the corpus's own paper published for
# NPR, and a registry without one leaves the gate with nothing to measure.
# Neither stand-in loads a weight file, which is the point -- `npr.pt` is
# gitignored and absent from every fresh checkout.

#: What `_FlatNPR` returns for every observation. A CONSTANT, so its AUC is
#: exactly 0.5 under `bench.metrics.auc`'s tie-averaging -- inside the
#: oracle band without depending on how a fixture corpus happens to be
#: ordered or on what its pixels contain.
_FLAT_NPR_SCORE = 0.5


class _FlatNPR:
    """Slot C stand-in that scores everything alike: AUC 0.5, in band."""

    name = "npr"
    slot = "C"
    version = "stub-flat-1"
    modalities = frozenset()
    min_quality_band = "reject"

    def score(self, obs):
        return RawScore(detector=self.name, version=self.version,
                        score=_FLAT_NPR_SCORE, abstained=False, reason="ok")


class _InvertedNPR:
    """Slot C stand-in wired backwards: high on reals, low on fakes.

    It reads the label off the fixture's own `source_id` rather than off
    the pixels. That is cheating, deliberately and in the WRONG direction:
    it reproduces an AUC of 0.0 the way inverted labels, misaligned ids or
    a flipped score sign would, which is what `ORACLE_LO` exists to catch,
    and it does so without depending on record order.
    """

    name = "npr"
    slot = "C"
    version = "stub-inverted-1"
    modalities = frozenset()
    min_quality_band = "reject"

    def score(self, obs):
        fake = all("fake" in o.source_id for o in obs)
        return RawScore(detector=self.name, version=self.version,
                        score=0.0 if fake else 1.0, abstained=False,
                        reason="ok")


@pytest.fixture
def stub_registry() -> Registry:
    """A `npr` whose AUC lands inside the oracle band, plus one other slot."""
    registry = Registry()
    registry.register(SyntheticDetector(name="synthetic"))
    registry.register(_FlatNPR())
    return registry


@pytest.fixture
def inverted_registry() -> Registry:
    """The same, with an `npr` landing below `ORACLE_LO`."""
    registry = Registry()
    registry.register(SyntheticDetector(name="synthetic"))
    registry.register(_InvertedNPR())
    return registry
