"""Ingests a third-party baseline's scores; never its code.

A baseline detector (UFD, DistilDIRE, the DeepfakeBench family) runs in its
own environment and hands back `sample_id,score` rows. This module reads
that CSV and nothing else — no baseline import, download, or execution
happens anywhere downstream of here. Two reasons that boundary is load-bearing
rather than tidy:

- Dependency containment. `tests/test_ci_gates.py` pins `requirements-dev.txt`
  and `requirements-floor.txt` with exact `==` versions across both CI legs.
  Each of those baselines carries its own dependency tree (and its own pinned,
  often conflicting, torch/numpy/opencv versions); installing even one would
  make that pinning unenforceable on the next bump, for every future baseline
  added the same way.
- Licence containment. Several obtainable baseline weights are NonCommercial
  licensed, and `assets/manifest.yaml` fails closed on exactly that: unregistered
  means non-commercial (spec §11). This product is to be commercialised. A
  weight file that never enters the tree cannot be the file that blocks a sale,
  so this protocol measures baselines by their output, never by vendoring them.

The unknown-id refusal is about the FILE, but its usual cause is upstream of
the file. `known_ids` is built from the records a loader actually returned,
and `corpora.dfeval` drops duplicate, unreadable and unlabelled rows (and
`--limit` drops more) — so a wrapper that correctly emitted one row per
manifest `item_id` will be refused here for scoring ids this run no longer
holds. That is still a refusal rather than a silent accept, because silently
accepting would hide a genuinely mismatched corpus; the message names the
dedup so the reader does not go looking for the bug in their baseline.

Coverage is reported, never thresholded. There is deliberately no number below
which a baseline is dropped from a report — dropping one is a judgement the
reader makes with `ScoreFile.coverage` in hand, not one this module makes for
them by silently discarding low-coverage rows.

Scores outside [0, 1] are accepted on purpose. `bench.metrics.auc` is rank-based,
so a baseline that emits logits is exactly as usable as one that emits
calibrated probabilities, and rejecting logits would silently exclude the
baselines most likely to be honest about what their model actually outputs.
The corollary: calibration is NOT claimed for an ingested score file, so no
ECE may be computed from one — `bench.metrics.ece` requires its input in
[0, 1], and a logit-valued score file would either fail that check or, worse,
pass it by coincidence and report a meaningless number. Only AUC-family
statistics are valid on a `ScoreFile`.
"""
from __future__ import annotations

import csv
import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: The only header this protocol accepts. Anything else is a format the
#: writer did not intend for this module, and guessing at a remapping would
#: hide that mistake rather than surface it.
EXPECTED_HEADER = ["sample_id", "score"]


@dataclass(frozen=True)
class ScoreFile:
    """One baseline's scores over (a subset of) a known corpus.

    `coverage` is `covered / total` and is informational only — see the
    module docstring on why nothing here enforces a floor on it.

    Attributes:
        name: Identifies the baseline in reports; defaults to the score
            file's stem when not given explicitly.
        scores: `sample_id -> score`, over exactly the rows the file covered.
            Values may be outside [0, 1]; see the module docstring.
        covered: `len(scores)`.
        total: `len(known_ids)` the file was checked against.
        coverage: `covered / total`.
        missing: The known ids the file did not score, sorted.
    """

    name: str
    scores: dict[str, float]
    covered: int
    total: int
    coverage: float
    missing: tuple[str, ...]


def load_score_file(
    path: str | Path, known_ids: Collection[str], *, name: str | None = None
) -> ScoreFile:
    """Reads a baseline's `sample_id,score` CSV and checks it against a corpus.

    Every failure mode here is a refusal, not a silent degrade: a baseline
    that scored the wrong corpus, scored one sample twice, or emitted a
    non-finite score is a configuration error, and a report built on top of
    one would misattribute the defect to the baseline's accuracy rather than
    to the file that produced it.

    Args:
        path: Path to the CSV file. Must have the exact header
            `sample_id,score`, at least one data row, and every `sample_id`
            drawn from `known_ids`, each appearing at most once.
        known_ids: The full set of sample ids the baseline could have scored
            (typically a corpus's `sample_id` column). Defines `total` and
            `missing`; an id in the file but not in this collection is refused.
        name: Overrides the baseline name used in reports. Defaults to the
            file's stem (`ufd.csv` -> `"ufd"`).

    Returns:
        A `ScoreFile` over the covered subset of `known_ids`.

    Raises:
        ValueError: If the file has no header row, the header is not exactly
            `sample_id,score`, there are no data rows (an empty score file
            must not be read as total abstention), a row's `sample_id` is not
            in `known_ids`, a `sample_id` appears more than once, a score does
            not parse as a float, or a score is not finite (NaN or inf). Every
            message names the offending row's `sample_id` or value.
    """
    p = Path(path)
    known = set(known_ids)

    with p.open(newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{p}: empty file, no header row found") from exc
        if header != EXPECTED_HEADER:
            raise ValueError(
                f"{p}: expected header {EXPECTED_HEADER!r}, got {header!r}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError(
            f"{p}: header but no rows; an empty score file would otherwise "
            "be read as total abstention rather than refused"
        )

    scores: dict[str, float] = {}
    for row in rows:
        if len(row) != 2:
            raise ValueError(f"{p}: row {row!r} does not have exactly 2 fields")
        sample_id, raw_score = row
        if sample_id not in known:
            raise ValueError(
                f"{p}: sample_id {sample_id!r} is not in the known corpus "
                f"of {len(known)} ids; a baseline scoring a sample it was "
                "not given is a configuration error, not low coverage. The "
                "likeliest cause is not the baseline: the corpus loaded "
                "here is SMALLER than the manifest the baseline was run "
                "against. `corpora.dfeval` drops duplicate, unreadable and "
                "unlabelled rows before this set is built (duplicates are "
                "EXPECTED on in-the-wild media, where one image circulates "
                "across many domains), and `--limit` drops more, while a "
                "baseline wrapper emits one row per manifest `item_id`. "
                "Re-run the baseline over the ids this run actually loaded, "
                "or drop `--limit`, rather than editing the score file."
            )
        if sample_id in scores:
            raise ValueError(
                f"{p}: duplicate sample_id {sample_id!r}; one row silently "
                "overwriting another would change the coverage denominator "
                "invisibly"
            )
        try:
            value = float(raw_score)
        except ValueError as exc:
            raise ValueError(
                f"{p}: sample_id {sample_id!r} has unparseable score "
                f"{raw_score!r}"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                f"{p}: sample_id {sample_id!r} has score {raw_score!r} which "
                "is not finite"
            )
        scores[sample_id] = value

    covered = len(scores)
    total = len(known)
    missing = tuple(sorted(known - scores.keys()))
    coverage = covered / total if total else float("nan")

    return ScoreFile(
        name=name if name is not None else p.stem,
        scores=scores,
        covered=covered,
        total=total,
        coverage=coverage,
        missing=missing,
    )


def aligned(
    sf: ScoreFile, records: Sequence[dict]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Projects `records` onto the subset a score file actually covered.

    A record the score file did not cover is omitted from all three returned
    arrays — not zero-filled, not NaN-filled — because either fill would
    quietly hand the baseline credit or blame for samples it never scored.

    Args:
        sf: The loaded score file.
        records: Corpus records in report order, each with at least
            `sample_id`, `label`, and `domain` keys. `domain` is the
            resampling family a grouped bootstrap (see
            `bench.metrics.bootstrap_ci_by_group`) resamples over.

    Returns:
        `(scores, labels, groups)`, three arrays of equal length, aligned
        index-for-index, over the covered subset, in `records` order.
    """
    out_scores: list[float] = []
    out_labels: list[int] = []
    out_groups: list[object] = []
    for r in records:
        sample_id = r["sample_id"]
        if sample_id in sf.scores:
            out_scores.append(sf.scores[sample_id])
            out_labels.append(r["label"])
            out_groups.append(r["domain"])
    return (
        np.asarray(out_scores, dtype=float),
        np.asarray(out_labels, dtype=int),
        np.asarray(out_groups),
    )
