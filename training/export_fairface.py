"""Export FairFace parquet shards into the capture-corpus layout on disk.

Not part of the installed package: this is corpus preparation, not inference,
and nothing in src/dfd/ may depend on it.

WHY THIS EXISTS. Measured 2026-09-22 (docs/HANDOFF.md §0), the project's own
capture corpus yields **19** distinct genuine face crops — 442 session folders
holding 58 distinct images, 979 of whose 1088 frame files are one demo asset
replayed. Self-blending needs only REAL faces, so the binding constraint is a
licence-clean supply of them. FairFace is 97,698 real faces, already cropped
and aligned to 224x224 (exactly `face_pool.DEFAULT_CROP_SIZE`, and exactly the
resolution `dfd.detectors.blend`'s seam features assume), ungated, under
CC BY 4.0 — which permits commercial use with attribution, unlike every
research dataset in docs/EULA-ACCESS.md.

WHY EXPORT RATHER THAN ADD A READER. Everything downstream of the capture
layout is already written and tested: `load_capture_sessions`,
`build_face_pool` (real YuNet detection, content deduplication, ROI clamping),
and `build_sbi_corpus`. Writing FairFace into that layout reuses all of it. A
parallel FairFace reader would be a second, less-tested copy of the same
pipeline, and would need its own answer to the landmark question — which is
the trap `bench/runner.py` fell into when it fabricated eye positions at 35%
and 65% of frame width rather than detecting them.

Run:
    python3 -m training.export_fairface \\
        --parquet-dir /path/to/fairface/0.25 \\
        --out /path/to/fairface_sessions \\
        --limit 20000

Then treat the output exactly as a capture corpus:
    load_capture_sessions(out) -> build_face_pool(...) -> build_sbi_corpus(...)

ATTRIBUTION IS A LICENCE CONDITION, not a courtesy. CC BY 4.0 requires it, so
every exported session records its provenance in `results.json` and the
dataset is registered in `assets/manifest.yaml`. Do not strip either.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

#: FairFace's class_label orderings, from the dataset card. Stored as
#: integers in the parquet; decoded here so the exported record is readable
#: and so acceptance criterion 11 has real group names to slice on.
AGE_LABELS = ("0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59",
              "60-69", "more than 70")
GENDER_LABELS = ("Male", "Female")
RACE_LABELS = ("East Asian", "Indian", "Black", "White", "Middle Eastern",
               "Latino_Hispanic", "Southeast Asian")

LICENSE = "CC BY 4.0"
DATASET = "FairFace"
HOMEPAGE = "https://github.com/joojs/fairface"


def _label(labels: tuple[str, ...], index: object) -> str:
    """Decode a class-label index, without inventing a value for a bad one.

    Returns the literal index as a string when it is out of range rather than
    clamping to a neighbour: a demographic slice computed over a silently
    mislabelled group is worse than one that obviously has an "11" in it.
    """
    if isinstance(index, int) and 0 <= index < len(labels):
        return labels[index]
    return str(index)


def export_fairface(parquet_dir: str | Path, out_root: str | Path, *,
                    limit: int | None = None,
                    split: str = "train") -> dict[str, int]:
    """Write FairFace rows as one-frame capture sessions.

    Args:
        parquet_dir: directory holding the shards, e.g. the dataset's `0.25`
            config directory.
        out_root: directory to create the session folders in.
        limit: stop after this many rows; None exports every row found.
        split: shard filename prefix to read — "train" or "validation".

    Returns:
        `{"exported": n}`.

    The JPEG bytes are copied VERBATIM out of the parquet. FairFace stores the
    original encoded files, so decoding and re-encoding would add a second
    generation of JPEG quantisation to every real face while the pseudo-fake
    `corpora.sbi` makes from it is blended from decoded pixels. A seam detector
    reading high-frequency residuals would learn that compression difference in
    preference to the seam, and score beautifully on a corpus-wide artefact.
    """
    src = Path(parquet_dir)
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    shards = sorted(src.glob(f"{split}-*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"no {split}-*.parquet shards under {src}. Point --parquet-dir at "
            "the config directory (e.g. .../fairface/0.25), not the dataset root.")

    exported = 0
    for shard in shards:
        handle = pq.ParquetFile(shard)
        for group in range(handle.num_row_groups):
            for row in handle.read_row_group(group).to_pylist():
                if limit is not None and exported >= limit:
                    logger.info("exported %d sessions to %s", exported, out)
                    return {"exported": exported}
                session_id = f"fairface-{split}-{exported:06d}"
                folder = out / session_id
                folder.mkdir(exist_ok=True)
                (folder / "frame_00.jpg").write_bytes(row["image"]["bytes"])
                (folder / "results.json").write_text(json.dumps({
                    "session_id": session_id,
                    "folder": f"{out.name}/{session_id}",
                    "swapped": False,
                    "frame_count": 1,
                    "decision": {"approved": True, "reason": "fairface_real_face"},
                    # Named so nobody mistakes it for a liveness measurement:
                    # no scanner ran, and these are stills, not captures.
                    "scan": {"verdict": None, "source": "not-scanned"},
                    "demographics": {
                        "age": _label(AGE_LABELS, row["age"]),
                        "gender": _label(GENDER_LABELS, row["gender"]),
                        "race": _label(RACE_LABELS, row["race"]),
                    },
                    "provenance": {
                        "dataset": DATASET, "license": LICENSE,
                        "url": HOMEPAGE, "split": split,
                        "shard": shard.name,
                    },
                }, indent=2, sort_keys=True))
                exported += 1

    logger.info("exported %d sessions to %s", exported, out)
    return {"exported": exported}


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default="train")
    args = parser.parse_args(argv)
    report = export_fairface(args.parquet_dir, args.out,
                             limit=args.limit, split=args.split)
    logger.info("done: %s", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
