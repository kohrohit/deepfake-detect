"""Shortcut controls: can a corpus measure a detector at all.

These two functions do not measure a detector. They answer a prior question
that this project has twice answered wrong by not asking it: do a corpus's
real and fake halves differ in some trivial way that ANY model would find,
making every detector number fitted on it unreadable regardless of how good
the detector is.

**The two measured precedents that make this module exist.**

DF40: Lab colour means and standard deviations alone — no face, no learned
feature, six numbers per image — separated that corpus's real and fake
halves at AUC 0.843 on a val-fit/test-report split (`bench/eval_df40.py`,
`bench/df40_report.json`). Every detector number ever reported on DF40 was
therefore unreadable: a model that looked at paint rather than faces did
nearly as well as any detector could.

The v-CIP capture corpus (`bench/vcip_controls.py`): one byte of the JPEG
luma quantisation table labels 99.2% of it, because the swap pipeline
re-encodes at OpenCV quality 95 and the capture path never re-encodes at
all. A model fitted on the header tables alone — no pixels read — reached
frame AUC 0.985, ABOVE the detector's own held-out 0.957.

**Why the null has to be a refit, not a reshuffle of scores.**
`bench.metrics.permutation_null` takes a callable that refits the model on
shuffled labels and returns the transfer statistic; its own docstring is
explicit that permuting an already-fitted model's scores tests a different
and much weaker hypothesis. A reshuffled-score null only asks "is this
ranking consistent", which a corpus with no shortcut would already satisfy —
it would not answer the question this module exists to ask, which is
whether the labels carry information the fit could have memorised by
accident.

**Why the split has to be group-disjoint.**
`run_control` fits on one set of groups and reports on another. A curve
fitted and reported on the same rows has already seen its own exam: it
would make every control fire on every corpus, which is indistinguishable
from a control that never fires at all and therefore worthless as a gate.

**Why the split also has to be class-aware.** On this module's actual
corpus — Deepfake-Eval-2024, grouped by the 88 originating web domains —
a domain that only hosts deepfake content, or a stock-photo domain that
only contributes reals, is not an edge case; it is what an in-the-wild
corpus collected from 88 domains looks like. A split that shuffles groups
blind to their label composition will, on real data, eventually land every
real-only domain on one side and leave the other side with no reals at
all — and the failure then surfaces as `LogisticRegression`'s own exception
from inside a control whose entire job is to produce a legible verdict
about the corpus, not to hand that verdict back as somebody else's
stack trace. `_group_disjoint_split` places groups greedily so each side
accumulates both classes, and refuses with its OWN "both labels" message,
naming what it saw, when no assignment of the given groups can do that.

**Why a non-JPEG gets NaN, not zero.**
`jpeg_table_features` reads the quantisation table from the file HEADER —
PIL's `quantization` attribute, populated by its own JPEG-marker parser on
`Image.open` without decoding a single pixel (Python's stdlib has no DQT
parser; the alternative is walking `0xFFDB` segment markers by hand, which
is what PIL already does). A PNG carries no such table. Returning zeros for
it would assert a feature value that does not exist — and zero is itself a
valid low-quality quantisation entry, so it would not even fail loudly. NaN
lets the caller exclude the row and count it, which is the only honest
thing to do with a measurement that was never taken.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image
from sklearn.linear_model import LogisticRegression

from .metrics import auc, bootstrap_ci_by_group, permutation_null, permutation_p

#: Luma quantisation tables are the 8x8 DCT block, zig-zag flattened by PIL
#: into 64 entries. Used only as the width of an all-NaN row when NOT ONE
#: path in a batch carries a table, so the return shape is still (N, K)
#: rather than (N, 0).
DEFAULT_TABLE_WIDTH = 64


def colour_features(images: Sequence[np.ndarray]) -> np.ndarray:
    """Per-channel Lab mean and standard deviation: `(N, 6)`.

    No face detection, no learned feature — the six numbers DF40 fell to at
    AUC 0.843.

    **`images` are RGB arrays**, the convention the corpus loaders in
    `corpora/` hand out (`corpora.dfeval` stores `cv2.cvtColor(..., BGR2RGB)`
    as a record's `image`), and the conversion happens HERE, one image at a
    time. A caller converting a whole corpus to BGR first in order to satisfy
    a BGR contract would hold the entire decoded corpus twice, in full, to
    produce six numbers per image — which is what this function's previous
    contract cost its only caller.

    Column order: L mean, a mean, b mean, L std, a std, b std.
    """
    rows: list[npt.NDArray[np.float64]] = []
    for image in images:
        lab = cv2.cvtColor(np.ascontiguousarray(image),
                           cv2.COLOR_RGB2LAB).astype(np.float64)
        flat = lab.reshape(-1, 3)
        rows.append(np.concatenate([flat.mean(axis=0), flat.std(axis=0)]))
    return np.asarray(rows, dtype=np.float64)


def _luma_table(path: Path) -> tuple[int, ...] | None:
    """The lowest-indexed quantisation table in `path`'s header, or None.

    PIL assigns table 0 to luma on every baseline encoder this project has
    produced or captured; `min(tables)` rather than a bare `tables[0]` in
    case some writer ever numbers them otherwise. None, not an empty tuple,
    for a file with no quantisation tables at all — a PNG, or anything else
    PIL did not have to parse a DQT segment to open.
    """
    with Image.open(path) as image:
        tables = getattr(image, "quantization", None)
        if not tables:
            return None
        key = min(tables)
        return tuple(int(v) for v in tables[key])


def jpeg_table_features(paths: Sequence[Path]) -> np.ndarray:
    """The luma quantisation table read from each file's header: `(N, K)`.

    Read from the header alone — never by re-encoding, never by inspecting
    decoded pixels, because either of those would measure something the
    corpus's own pipeline did not actually write. A path with no table (a
    PNG) gets a row of NaN, excluded and counted by the caller rather than
    zero-filled: zero is itself a legal quantisation entry and would read as
    a measurement that was never taken.

    `K` is the shortest table actually found, truncating longer ones, the
    same defence `bench.vcip_controls.encoder_shortcut` uses against a
    writer that emits a table of unexpected length; `DEFAULT_TABLE_WIDTH`
    only when NOT ONE path carries a table, so the shape stays `(N, K)`.
    """
    tables = [_luma_table(path) for path in paths]
    widths = [len(t) for t in tables if t is not None]
    width = min(widths) if widths else DEFAULT_TABLE_WIDTH
    rows = [np.full(width, np.nan) if table is None
            else np.asarray(table[:width], dtype=np.float64)
            for table in tables]
    return np.asarray(rows, dtype=np.float64)


def _group_disjoint_split(groups: npt.NDArray[np.str_],
                          labels: npt.NDArray[np.int_], *, seed: int,
                          ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """A fit/test partition of ROWS that never splits a group across them,
    and that guarantees both labels on BOTH sides.

    Groups, not rows, are shuffled and assigned, because the question is
    whether the shortcut generalises ACROSS groups — splitting rows within
    a group would let the fit half and the test half share whatever is
    constant within a group (a session, a source image, a web domain),
    which is exactly the kind of leakage a shortcut control has to be
    immune to.

    The assignment is class-aware rather than blind, because a blind split
    can place every group of a class on one side. Groups are visited in a
    seeded shuffle order; a group whose rows carry only ONE class goes to
    whichever side currently holds FEWER rows of that class (greedily
    balancing each class across sides, independently); a group whose rows
    carry both classes falls back to whichever side currently holds fewer
    groups (plain size balance, since it cannot unbalance either class by
    much either way). This does not search for an optimal split — it does
    not need to — it only needs to not produce a one-class side when some
    assignment of these groups could avoid it.

    Raises:
        ValueError: if fewer than two groups are given, or if NO assignment
            of them can give both sides both classes (every group pure in
            the same class with too few groups to go around, e.g. one real
            domain and one fake domain and nothing else). The message names
            the side that came up short and the group counts per class, so
            the caller does not have to re-derive what was seen.
    """
    uniq = np.unique(groups)
    if uniq.size < 2:
        msg = f"run_control needs at least two groups to split, got {uniq.size}"
        raise ValueError(msg)

    classes = sorted(int(c) for c in np.unique(labels))
    order = np.random.default_rng(seed).permutation(uniq.size)
    shuffled = uniq[order]

    class_rows_on_side: dict[int, dict[str, int]] = {
        c: {"fit": 0, "test": 0} for c in classes}
    groups_on_side: dict[str, int] = {"fit": 0, "test": 0}
    side_of: dict[Any, str] = {}

    for group in shuffled:
        rows = labels[groups == group]
        present = sorted(int(c) for c in np.unique(rows))
        if len(present) == 1:
            c = present[0]
            tally = class_rows_on_side[c]
            side = "fit" if tally["fit"] <= tally["test"] else "test"
        else:
            side = "fit" if groups_on_side["fit"] <= groups_on_side["test"] else "test"
        side_of[group] = side
        groups_on_side[side] += 1
        for c in present:
            class_rows_on_side[c][side] += int((rows == c).sum())

    fit_mask = np.array([side_of[g] == "fit" for g in groups])
    test_mask = ~fit_mask

    fit_classes = {int(c) for c in np.unique(labels[fit_mask])}
    test_classes = {int(c) for c in np.unique(labels[test_mask])}
    all_classes = set(classes)
    if fit_classes != all_classes or test_classes != all_classes:
        groups_per_class = {
            c: int(sum(1 for grp in uniq if c in np.unique(labels[groups == grp])))
            for c in classes}
        short = [side for side, seen in
                (("fit", fit_classes), ("test", test_classes)) if seen != all_classes]
        msg = (f"run_control needs both labels on both sides of the "
               f"group-disjoint split, but no assignment of {uniq.size} "
               f"groups achieves that: side(s) {short} came up missing a "
               f"class; groups per class: {groups_per_class}")
        raise ValueError(msg)

    return fit_mask, test_mask


def run_control(features: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                *, seed: int, n_null: int = 24) -> dict[str, Any]:
    """Fit a logistic shortcut model on one group half, report on the other.

    All-NaN rows (a non-JPEG under `jpeg_table_features`) are excluded from
    both halves and counted by the shrunk `n_fit` / `n_test` — never
    zero-filled, which would assert a measurement that was never taken.

    The null comes from `bench.metrics.permutation_null`: `n_null` refits of
    a fresh `LogisticRegression` on SHUFFLED fit-half labels, each re-scored
    against the same, unshuffled test half. `permutation_p` against
    `centre=0.5` says how often that null beats the observed AUC; the
    corresponding `bootstrap_ci_by_group` interval says how precise the
    observed AUC is. Both travel together because a control that clears the
    null by a hair, with a wide bootstrap interval, is not the same finding
    as one that clears it by a mile.

    Returns:
        `{auc, ci_lo, ci_hi, null_lo, null_hi, p, n_fit, n_test}`.

    Raises:
        ValueError: if `labels` carries only one class, before or after
            dropping all-NaN rows — shuffling a one-class vector permutes to
            itself, so the "null" would be `n_null` copies of the observed
            statistic and every corpus would appear to fire. Also raised,
            with the same "both labels" wording, if `_group_disjoint_split`
            cannot place both classes on both sides of ANY assignment of the
            given groups (see that function).
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    g = np.asarray(groups)
    if x.shape[0] != y.shape[0] or x.shape[0] != g.shape[0]:
        msg = (f"shape mismatch: features {x.shape}, labels {y.shape}, "
               f"groups {g.shape}")
        raise ValueError(msg)
    if len(np.unique(y)) < 2:
        msg = f"run_control needs both labels, got classes {np.unique(y).tolist()}"
        raise ValueError(msg)

    usable = ~np.all(np.isnan(x), axis=1)
    x, y, g = x[usable], y[usable], g[usable]
    if len(np.unique(y)) < 2:
        msg = ("run_control needs both labels after dropping all-NaN rows, "
               f"got classes {np.unique(y).tolist()}")
        raise ValueError(msg)

    fit_mask, test_mask = _group_disjoint_split(g, y, seed=seed)
    x_fit, y_fit = x[fit_mask], y[fit_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    model = LogisticRegression(max_iter=1000).fit(x_fit, y_fit)
    test_scores = model.predict_proba(x_test)[:, 1]
    observed = auc(test_scores, y_test)
    ci_lo, ci_hi = bootstrap_ci_by_group(test_scores, y_test, g[test_mask],
                                         auc, seed=seed)

    def refit_and_score(shuffled_fit_labels: np.ndarray) -> float:
        refit = LogisticRegression(max_iter=1000).fit(x_fit, shuffled_fit_labels)
        return auc(refit.predict_proba(x_test)[:, 1], y_test)

    null = permutation_null(refit_and_score, y_fit, n=n_null, seed=seed)
    p = permutation_p(observed, null, centre=0.5)

    return {
        "auc": float(observed),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "null_lo": float(np.nanmin(null)),
        "null_hi": float(np.nanmax(null)),
        "p": float(p),
        "n_fit": int(fit_mask.sum()),
        "n_test": int(test_mask.sum()),
    }
