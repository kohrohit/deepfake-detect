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


@dataclass(frozen=True)
class Split:
    """One fold: every fake in `test` comes from `held_out_generator`."""

    held_out_generator: str
    train: list[dict[str, Any]]
    test: list[dict[str, Any]]
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
        if r["label"] == 1 and not r["generator"]:
            raise ValueError(
                f"fake record {r['sample_id']!r} has no generator; an "
                "unattributed fake would join the training side of every split")

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
        raise ValueError(
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
        raise ValueError(
            f"corpus has {len(subjects)} distinct subject(s); an "
            "identity-disjoint split needs at least 2")

    rng = np.random.default_rng(seed)
    shuffled = [subjects[i] for i in rng.permutation(len(subjects))]
    cut = max(1, len(shuffled) // 2)
    train_subjects = set(shuffled[:cut])

    has_real: dict[str, bool] = {s: False for s in subjects}
    fake_generators: dict[str, set[str]] = {s: set() for s in subjects}
    for r in records:
        if r["label"] == 0:
            has_real[r["subject_id"]] = True
        else:
            fake_generators[r["subject_id"]].add(r["generator"])

    generators = sorted({r["generator"] for r in records if r["label"] == 1})

    splits: list[Split] = []
    for g in generators:
        # A test-side subject with no real record and no fake attributed to
        # g cannot place anything on test: every one of its fakes wants
        # train (none match g), so all would be dropped and the subject
        # would vanish from the fold instead of registering as the identity
        # conflict it is. Keep such a subject on train, where its fakes
        # place cleanly, rather than dropping it into invisibility.
        train_subjects_g = set(train_subjects)
        for subj in subjects:
            if (subj not in train_subjects_g
                    and not has_real[subj]
                    and g not in fake_generators[subj]):
                train_subjects_g.add(subj)

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
        splits.append(Split(g, train, test, dropped))
    return splits
