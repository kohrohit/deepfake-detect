"""Leave-one-generator-out split construction (spec §8.1).

In-dataset AUC measures memorisation. LOGO measures what happens when a
generator the model has never seen walks through the door — which, in the
field, is every generator eventually.

Two spec §8.2 guards are structural here rather than checked after the fact:
identity disjointness (guard 1) and video-level integrity (guard 2). A split
that leaks a subject or straddles a source video produces a number that cannot
be repaired downstream, so this module refuses to emit one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

REQUIRED_KEYS = ("sample_id", "subject_id", "source_id", "generator", "label")


class UnsplittableCorpusError(ValueError):
    """Raised when a well-formed corpus cannot support a LOGO split.

    A corpus can be internally consistent — every record valid, every source
    unstraddled, every fake attributed — and still have too few subjects or
    generators, or too little of one label, to build even one identity-disjoint
    fold. That is a legitimate degrade-and-warn condition, not a defect, and
    is kept as its own subclass of ValueError specifically so a caller can
    catch it without also catching the plain ValueError that `_validate`
    raises for a malformed corpus (a straddling source, a label outside
    {0, 1}, an unattributed fake, or a real carrying a generator) — those
    must propagate, not degrade.
    """


@dataclass(frozen=True)
class Split:
    """One fold: every fake in `test` comes from `held_out_generator`.

    `train_subjects`/`test_subjects` are the partition actually used to
    build this fold (after both vanishing-subject moves below), not merely
    subjects observed in `train`/`test` — the two coincide for every
    subject, but making the partition itself a field means a caller (or a
    test) can ask "which side is this subject on" without re-deriving it
    from placed records, which is exactly the derivation that went wrong
    once already.
    """

    held_out_generator: str
    train: list[dict[str, Any]]
    test: list[dict[str, Any]]
    train_subjects: frozenset[str]
    test_subjects: frozenset[str]
    dropped_for_identity: list[dict[str, Any]] = field(default_factory=list)

    def test_ids(self) -> list[str]:
        return [r["sample_id"] for r in self.test]

    def train_ids(self) -> list[str]:
        return [r["sample_id"] for r in self.train]

    def dropped_ids(self) -> list[str]:
        return [r["sample_id"] for r in self.dropped_for_identity]


def _validate(records: list[dict[str, Any]]) -> None:
    for r in records:
        missing = [k for k in REQUIRED_KEYS if k not in r]
        if missing:
            raise KeyError(
                f"record {r.get('sample_id')!r} is missing required "
                f"key(s) {missing}")
        if r["label"] not in (0, 1):
            raise ValueError(
                f"record {r['sample_id']!r} has label {r['label']!r}; only "
                "0 (real) and 1 (fake) are valid")
        if r["label"] == 1 and not r["generator"]:
            raise ValueError(
                f"fake record {r['sample_id']!r} has no generator; an "
                "unattributed fake would join the training side of every split")
        if r["label"] == 0 and r["generator"] is not None:
            raise ValueError(
                f"real record {r['sample_id']!r} carries generator "
                f"{r['generator']!r}; reals must have generator=None, or "
                "they contribute a phantom (subject, generator) pair that "
                "can trigger an unrelated source-straddle rejection")

    by_source: dict[str, set[tuple[Any, Any]]] = {}
    for r in records:
        by_source.setdefault(r["source_id"], set()).add(
            (r["subject_id"], r["generator"]))
    for src, pairs in sorted(by_source.items()):
        if len(pairs) > 1:
            raise ValueError(
                f"source {src!r} carries more than one subject/generator pair: "
                f"{sorted(map(str, pairs))}. A source that straddles cannot be "
                "assigned to one side of a split")


def _require_measurable(
    held_out: str, train: list[dict[str, Any]], test: list[dict[str, Any]]
) -> None:
    counts = {
        "train reals": sum(1 for r in train if r["label"] == 0),
        "test reals": sum(1 for r in test if r["label"] == 0),
        "train fakes": sum(1 for r in train if r["label"] == 1),
        "test fakes": sum(1 for r in test if r["label"] == 1),
    }
    empty = sorted(k for k, v in counts.items() if v == 0)
    if empty:
        raise UnsplittableCorpusError(
            f"split holding out {held_out!r} has no {' and no '.join(empty)} "
            f"(counts={counts}); without test reals there is no FPR to measure "
            "and without test fakes there is no TPR")


def logo_splits(records: list[dict[str, Any]], seed: int = 0) -> list[Split]:
    """One split per generator, identity-disjoint and video-whole.

    Subjects — not records — are partitioned, so a subject faked by several
    generators cannot appear on both sides. Fakes whose generator wants one
    side while their subject sits on the other are dropped and reported in
    `Split.dropped_for_identity` rather than silently leaked.
    """
    _validate(records)

    subjects = sorted({r["subject_id"] for r in records})
    if len(subjects) < 2:
        raise UnsplittableCorpusError(
            f"corpus has {len(subjects)} distinct subject(s); an "
            "identity-disjoint split needs at least 2")

    rng = np.random.default_rng(seed)
    shuffled = [subjects[i] for i in rng.permutation(len(subjects))]
    cut = max(1, len(shuffled) // 2)
    train_subjects = set(shuffled[:cut])

    has_real: dict[str, bool] = dict.fromkeys(subjects, False)
    fake_generators: dict[str, set[str]] = {s: set() for s in subjects}
    for r in records:
        if r["label"] == 0:
            has_real[r["subject_id"]] = True
        else:
            fake_generators[r["subject_id"]].add(r["generator"])

    # Reals never move for generator reasons (no fold-specific logic below
    # touches them), so if the random partition happens to put every real
    # subject on one side, every fold built from it has no reals on the
    # other and _require_measurable rejects every fold identically and
    # needlessly. With at least 2 real subjects a real-carrying split
    # always exists; deterministically move the alphabetically-first real
    # subject on the over-represented side to balance it.
    real_subjects = sorted(s for s in subjects if has_real[s])
    if len(real_subjects) >= 2:
        train_reals = [s for s in real_subjects if s in train_subjects]
        test_reals = [s for s in real_subjects if s not in train_subjects]
        if not train_reals:
            train_subjects.add(test_reals[0])
        elif not test_reals:
            train_subjects.discard(train_reals[0])

    generators = sorted({r["generator"] for r in records if r["label"] == 1})

    splits: list[Split] = []
    for g in generators:
        # A subject with no real record has no anchor: it is visible in a
        # fold only through fakes that place there. Two mirror-image moves
        # keep every such subject visible instead of letting it vanish:
        #
        #  - test-slated, but none of its fakes are attributed to g: every
        #    fake wants train (none match g), so all would be dropped and
        #    the subject would disappear from test rather than register as
        #    the identity conflict it is. Move it to train, where its fakes
        #    place cleanly.
        #  - train-slated, but its entire generator set is exactly {g}:
        #    every fake wants test (all match g), so all would be dropped
        #    and the subject would disappear from train. Move it to test,
        #    where its fakes place cleanly.
        #
        # Both moves are Pareto-improving: they add zero drops on the side
        # a subject moves to (its fakes agree unanimously with that side)
        # and remove nothing that was placeable on the side it leaves.
        train_subjects_g = set(train_subjects)
        for subj in subjects:
            if has_real[subj]:
                continue
            gens = fake_generators[subj]
            if subj not in train_subjects and g not in gens:
                train_subjects_g.add(subj)
            elif subj in train_subjects and gens == {g}:
                train_subjects_g.discard(subj)

        test_subjects_g = frozenset(subjects) - train_subjects_g

        train: list[dict[str, Any]] = []
        test: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = []
        for r in records:
            subject_side_is_test = r["subject_id"] not in train_subjects_g
            if r["label"] == 0:
                (test if subject_side_is_test else train).append(r)
                continue
            belongs_in_test = r["generator"] == g
            if belongs_in_test == subject_side_is_test:
                (test if belongs_in_test else train).append(r)
            else:
                dropped.append(r)
        _require_measurable(g, train, test)
        splits.append(Split(
            g, train, test, frozenset(train_subjects_g), test_subjects_g,
            dropped))
    return splits
