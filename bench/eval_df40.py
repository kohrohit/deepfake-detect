"""Run the benchmark over the ungated DF40 eval subset.

    python3 -m bench.eval_df40 \\
        --root ~/Desktop/agents/datasets/df40_eval/extracted/test \\
        --out-md bench/df40_report.md --out-json bench/df40_report.json

Every number this produces is GUARD-WAIVED, and the report says so at the
top. Two of the five spec §8.2 guards cannot pass on this corpus and neither
failure is repairable from the data:

- Guard 3 (compression coverage) wants c0/c23/c40. The repackaging arrives
  at one unlabelled compression level (`corpora.df40.UNKNOWN_COMPRESSION`),
  so the worst-compression cell the spec asks for does not exist here.
- Leave-one-generator-out needs several generators. This copy carries no
  per-technique label, so there is exactly one generator id and `logo_splits`
  refuses the split. Without a held-out-generator number, what remains is
  in-dataset AUC over an unseen corpus — better than nothing, and not the
  number spec §8.1 calls the only one that predicts field performance.

Guards are therefore waived explicitly rather than quietly satisfied with
invented labels. Inventing them is the cheaper path and is what makes a
benchmark flatter itself.
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

import numpy as np

from corpora.df40 import load_df40_records
from corpora.face_pool import DetectFn
from corpora.identity import embeddings_for_records
from dfd.detectors.registry import default_registry
from dfd.faces import detect_faces

from .report import render_markdown
from .runner import RunConfig, run_benchmark

logger = logging.getLogger(__name__)

#: Crossing rate attributed to the embedder rather than to leakage. Measured
#: 2026-09-23 over 124,251 unrelated FairFace pairs: 0.21% cross SFace's
#: threshold of 0.363. Set as the DEFAULT for this corpus only — it is a
#: property of the embedder and the faces, and another corpus must measure
#: its own rather than inherit this one.
DEFAULT_IDENTITY_FMR = 0.0021

#: The heading the waiver block renders under. Named so a test can assert the
#: block is present without pinning its prose.
WAIVER_HEADING = "## Guards waived for this corpus"

_WAIVER = f"""{WAIVER_HEADING}

Two of the five spec §8.2 guards cannot pass over the DF40 eval subset, so
this run was made with `enforce_guards=False`. Both failures are properties
of the data, not of the run:

- **Compression coverage.** Every record carries `compression=unknown`: the
  repackaging does not say what quality its images were encoded at, so the
  worst-compression cell cannot be reported.
- **Generator attribution.** Every fake carries one generator id
  (`df40_unknown_mixture`). DF40 spans 40 techniques; this copy carries no
  per-technique label, so leave-one-generator-out is refused rather than
  faked from filename families.

- **Video-level sampling (guard 2).** `corpora.df40` groups frames of one
  filename family into one source (2026-09-23). That is the honest
  grouping, and it is exactly what guard 2 forbids: the guard wants one
  sample per source, and this corpus has up to 999. The interval below is
  computed over sources rather than rows because of it.

Read the in-dataset table below as a cross-corpus sanity check — the corpus
is unseen, which the capture corpus was not — and never as a LOGO result.

**The interval is what changed most.** The fake half of the test split is
1,601 images in 45 filename families, the largest holding 999 of them — a
Kish effective sample size of 2.4. Resampling rows, as every earlier run of
this report did, reported a precision the data does not have.

- **No signal is measurable here at all (2026-09-23).** Fit a model on
  SHUFFLED training labels — one that has learnt nothing by construction —
  and report it against this corpus: across four training pairs those models
  score 0.229–0.780. Every AUC this project has reported on this corpus sits
  inside that null, including an inverted 0.316 whose grouped interval
  excluded chance and which is now retracted (p=0.33). The cause is the
  corpus: its halves arrive down different imaging chains and 62% of its
  fakes are one family, so almost any direction in feature space separates
  them somewhat, in one direction or the other. The interval below resamples
  the EVALUATION corpus and is silent about the variance contributed by the
  FIT. **So read the table below as neither support NOR refutation.** Pair
  any number taken from it with `bench.metrics.permutation_null`, or do not
  report it.
"""


def _json_safe(value: object) -> object:
    # bool BEFORE int: `isinstance(True, int)` is True in Python, so an
    # unguarded int branch serialises `guards_enforced` as `0`. A reader
    # scanning a report for `false` would not find it.
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        f = float(value)
        # JSON has no NaN. `nan` here means "not measurable", and an invalid
        # bare NaN token in a report file is how that becomes a parse error
        # for whoever reads it next (see commit 582678b).
        return None if f != f or f in (float("inf"), float("-inf")) else f
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def main(argv: Sequence[str] | None = None, *,
         detect: DetectFn = detect_faces, registry=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path,
                        help="directory holding fake/ and real/")
    parser.add_argument("--limit", type=int, default=None,
                        help="total records, split evenly between labels")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-md", type=Path,
                        default=Path("bench/df40_report.md"))
    parser.add_argument("--out-json", type=Path,
                        default=Path("bench/df40_report.json"))
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

    records, skipped = load_df40_records(
        args.root, limit=args.limit, seed=args.seed, detect=detect)
    if not records:
        logger.error("no usable records under %s (skipped: %s)",
                     args.root, skipped or "nothing")
        return 1
    labels = {r["label"] for r in records}
    if len(labels) < 2:
        # Every metric in the report is defined over both labels. Emitting a
        # table of `nan` invites a reader to re-run it rather than to fix the
        # corpus, so fail loudly here instead.
        logger.error("corpus has only label(s) %s; AUC, TPR@FPR and ECE are "
                     "all undefined over one label", sorted(labels))
        return 1

    if registry is None:
        registry = (default_registry(blend_weights=args.blend_weights)
                    if args.blend_weights is not None else default_registry())

    embeddings: dict[str, Any] | None = None
    identity_skipped: dict[str, int] = {}
    if args.identity:
        embeddings, identity_skipped = embeddings_for_records(
            records, detect=detect)
        if not embeddings:
            # Refuse rather than run: `--identity` is a request for a
            # certificate, and producing a report that silently says
            # "not_measured" is how a caller comes to believe it has one.
            logger.error("--identity was requested and nothing could be "
                         "embedded (%s). Run ops/fetch-assets.sh.",
                         identity_skipped or "no reason recorded")
            return 1
        # A crop that would not re-detect has no embedding, and
        # check_identity_disjoint refuses to certify a split containing an
        # id it cannot compare. Drop those records from the run rather than
        # from the certificate: a smaller corpus honestly certified beats a
        # whole one certified over the part that happened to embed.
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
                    _WAIVER + "\n## Leave-one-generator-out", 1)
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
        "identity_requested": bool(args.identity),
        "identity_skipped": identity_skipped,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, allow_nan=False))

    logger.info("wrote %s and %s", args.out_md, args.out_json)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
