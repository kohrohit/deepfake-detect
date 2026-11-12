"""Run the benchmark over the licence-clean FairFace swap corpus.

    python3 -m bench.eval_swaps \\
        --root ~/Desktop/agents/datasets/fairface_sessions \\
        --offset 10000 --limit 3000 --identity \\
        --out-md bench/swap_corpus_report.md \\
        --out-json bench/swap_corpus_report.json

**Why this file exists.** `bench/swap_corpus_report.md` — the first
leave-one-generator-out result this project has ever produced — was written
by a script that was never committed. The headline number could not be
reproduced from the repository, which is the same defect as an unpinned
dependency: the artifact is there and the thing that made it is not.

**The offset is not optional, and this script enforces it.**
`corpora.swap_corpus.build_swap_corpus` documents the rule and states that it
cannot check it: the training window is a property of the WEIGHTS, not of the
corpus. This entry point knows which weights are loaded, so here the rule is
checked. It has already produced one wrong headline — a worst-generator AUC
of 0.923 that was memorisation (see `TRAINING_WINDOWS`).

**What this corpus can and cannot show** travels with every report it writes,
in `_SCOPE` below. Read `corpora/swaps.py` before quoting any number from it.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from corpora.face_pool import DetectFn
from corpora.identity import embeddings_for_records
from corpora.swap_corpus import build_swap_corpus
from dfd.detectors.registry import default_registry
from dfd.faces import detect_faces

from .eval_df40 import DEFAULT_IDENTITY_FMR, _json_safe
from .report import render_markdown
from .runner import RunConfig, run_benchmark

logger = logging.getLogger(__name__)

#: detector version string -> the number of FairFace sessions, in
#: `sorted()` order, that version was FITTED on. A corpus evaluating one of
#: these must start at or beyond that many sessions.
#:
#: Keyed on the VERSION, never on the detector name: a future `blend_seam`
#: refitted on something else carries a different version string and must not
#: inherit this window. A version absent from this mapping is not gated —
#: "no declared window" means the detector was not fitted on this corpus, and
#: guessing a window for it would refuse valid runs.
#:
#: `0.2.0-fairface10k`: `training/fit_blend.py` over the first 10,000
#: sessions, selected with `sorted(root.glob("*/results.json"))`. A corpus
#: built below that offset drew every real record — and every photograph its
#: fakes were composited from — out of this detector's own training set, and
#: reported a worst-generator AUC of 0.923 on 2026-09-24. That is
#: memorisation. Rebuilt at offset 10000 the same run gives 0.527.
TRAINING_WINDOWS: dict[str, int] = {
    "0.2.0-fairface10k": 10000,
}

#: The heading the scope block renders under, named so a test can assert the
#: block is present without pinning its prose.
SCOPE_HEADING = "## What this corpus can and cannot show"

_SCOPE = f"""{SCOPE_HEADING}

**These are classical compositing swaps — warp, mask, blend — and they are
not generator output.** A detector that beats them has not been shown to beat
FSGAN, InSwapper or any diffusion pipeline, and no number below may be
reported as though it had. What this corpus CAN do:

- **Refute a detector.** One that cannot find a hard-blended composite
  boundary will not find a subtle one.
- **Support leave-one-generator-out across these four techniques**, which is
  a real generalisation test between real techniques even though all four are
  classical. It is the only corpus in this project LOGO can fold at all.

Both halves are FairFace photographs, so the two sides share an imaging
chain and nothing here can learn "smooth means fake" — the defect that
invalidated every fit against SFHQ. What is measured is generalisation across
TECHNIQUE, never across corpus.

**Guards are waived.** Two of the five spec §8.2 guards fail on this corpus
and both failures are measured rather than hidden: identity (a few FairFace
couples are genuinely the same person or near-twins) and demographic parity
(the per-stratum FPR ratio exceeds its ceiling). The numbers are in the
reproducibility record above. Compression coverage is waived too — these
crops are PNG-encoded in memory and never re-compressed, so the corpus
carries one honest `compression=none` level rather than an invented c0/c23/c40
spread.

"""


def _check_training_window(registry, offset: int) -> None:
    """Refuse a corpus drawn from a loaded detector's own training set.

    Raises:
        SystemExit: naming the detector, its version, the window and the
            offset given. A refusal that says only "contaminated" leaves the
            caller to guess which of several detectors caused it.
    """
    for name in registry.names():
        version = registry.get(name).version
        window = TRAINING_WINDOWS.get(version)
        if window is not None and offset < window:
            raise SystemExit(
                f"--offset {offset} draws FairFace sessions that {name} "
                f"{version} was fitted on (its training window is the first "
                f"{window} sessions). Every real record, and every "
                f"photograph its fakes are composited from, would come out "
                f"of the detector's own training set; this has already "
                f"produced one wrong headline. Use --offset {window} or "
                f"higher, or pass --allow-training-window if you are "
                f"deliberately measuring the contamination.")


def main(argv: Sequence[str] | None = None, *,
         detect: DetectFn = detect_faces, registry=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path,
                        help="the FairFace session directory, one folder per "
                             "photograph (training/export_fairface.py)")
    parser.add_argument("--offset", required=True, type=int,
                        help="photographs to SKIP, in sorted order. Required "
                             "rather than defaulted: the right value is a "
                             "property of the weights being evaluated, and a "
                             "default of 0 is the contaminated one")
    parser.add_argument("--limit", type=int, default=None,
                        help="photographs read after the offset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-training-window", action="store_true",
                        help="run even though the offset falls inside a "
                             "loaded detector's training window. For "
                             "deliberately measuring contamination")
    parser.add_argument("--out-md", type=Path,
                        default=Path("bench/swap_corpus_report.md"))
    parser.add_argument("--out-json", type=Path,
                        default=Path("bench/swap_corpus_report.json"))
    parser.add_argument("--synthetic-sources", type=Path, default=None,
                        help="directory of GENERATOR-OUTPUT face images "
                             "(SFHQ part 3). Adds two further generators per "
                             "couple: the synthetic composite and its "
                             "matched resampling control, which are only "
                             "meaningful read together — see "
                             "corpora.swap_corpus.build_swap_corpus")
    parser.add_argument("--blend-weights", type=Path, default=None,
                        help="override the blend_seam weights path")
    parser.add_argument("--identity", action="store_true",
                        help="embed every crop and certify the LOGO folds "
                             "identity-disjoint (acceptance criterion 2). "
                             "Needs the SFace weights — ops/fetch-assets.sh")
    parser.add_argument("--identity-max-false-match-rate", type=float,
                        default=DEFAULT_IDENTITY_FMR,
                        help="crossing rate attributable to the embedder "
                             "rather than to leakage; see dfd.embed")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    if registry is None:
        registry = (default_registry(blend_weights=args.blend_weights)
                    if args.blend_weights is not None else default_registry())

    # BEFORE building the corpus, which takes minutes: a refusal that arrives
    # after the work is a refusal the caller learns to pass --allow past.
    if not args.allow_training_window:
        _check_training_window(registry, args.offset)

    synthetic: list[Path] = []
    if args.synthetic_sources is not None:
        synthetic = sorted(
            p for p in args.synthetic_sources.rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not synthetic:
            logger.error("--synthetic-sources %s holds no images",
                         args.synthetic_sources)
            return 1
        logger.info("synthetic source pool: %d images", len(synthetic))

    records, skipped = build_swap_corpus(
        args.root, limit=args.limit, offset=args.offset, seed=args.seed,
        detect=detect, synthetic_sources=synthetic or None)
    if not records:
        logger.error("no usable records under %s at offset %d (skipped: %s)",
                     args.root, args.offset, skipped or "nothing")
        return 1
    labels = {r["label"] for r in records}
    if len(labels) < 2:
        logger.error("corpus has only label(s) %s; AUC, TPR@FPR and ECE are "
                     "all undefined over one label", sorted(labels))
        return 1

    embeddings: dict[str, Any] | None = None
    identity_skipped: dict[str, int] = {}
    if args.identity:
        embeddings, identity_skipped = embeddings_for_records(
            records, detect=detect)
        if not embeddings:
            logger.error("--identity was requested and nothing could be "
                         "embedded (%s). Run ops/fetch-assets.sh.",
                         identity_skipped or "no reason recorded")
            return 1
        before = len(records)
        records = [r for r in records if r["sample_id"] in embeddings]
        if len(records) < before:
            logger.warning("identity: dropped %d of %d records with no "
                           "embedding (%s)", before - len(records), before,
                           identity_skipped)

    record = run_benchmark(
        records, registry,
        RunConfig(seed=args.seed, enforce_guards=False,
                  identity_embeddings=embeddings,
                  identity_max_false_match_rate=(
                      args.identity_max_false_match_rate)))

    md = render_markdown(record)
    md = md.replace("## Leave-one-generator-out",
                    _SCOPE + "## Leave-one-generator-out", 1)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md)

    payload = _json_safe(dataclasses.asdict(record))
    assert isinstance(payload, dict)
    payload["corpus"] = {
        "root": str(args.root),
        "records": len(records),
        "fakes": sum(r["label"] for r in records),
        "skipped": skipped,
        "seed": args.seed,
        "limit": args.limit,
        "offset": args.offset,
        "training_window_allowed": bool(args.allow_training_window),
        "synthetic_sources": len(synthetic),
        "identity_requested": bool(args.identity),
        "identity_skipped": identity_skipped,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, allow_nan=False))

    logger.info("wrote %s and %s", args.out_md, args.out_json)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
