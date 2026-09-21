"""The corpus builder's job is split discipline, not image processing."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from corpora.face_pool import FaceCrop
from corpora.sbi import SBI_GENERATOR, EvaluationOnlySessionError, build_sbi_corpus
from dfd.faces import FaceBox
from dfd.types import Modality, Quality


def _crop(session_id: str, frame_index: int = 0, swapped: bool = False) -> FaceCrop:
    lms = np.array([[70.0, 80.0], [110.0, 80.0], [90.0, 100.0],
                    [75.0, 125.0], [105.0, 125.0]])
    # hashlib, not `len(session_id) * 1000 + frame_index`: that shape
    # collapses distinct session ids of the same length onto the same seed
    # (e.g. every "s0".."s9" fixture in this file shared one image) -- the
    # exact defect fixed in tests/test_fit_blend.py's `_seed_for_session`,
    # which this mirrors. Harmless for the assertions in this file (none
    # compare image content across different subjects), but the same shape
    # is worth not carrying forward.
    seed = int.from_bytes(
        hashlib.sha256(f"{session_id}:{frame_index}".encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    return FaceCrop(
        session_id=session_id,
        frame_index=frame_index,
        image=rng.integers(40, 210, (224, 224, 3), dtype=np.uint8),
        box=FaceBox(x=55, y=55, w=70, h=90, landmarks=lms, score=0.99),
        quality=Quality(inter_ocular_px=40.0, blur_var=120.0, yaw_deg=0.0,
                        pitch_deg=0.0, exposure=0.5, band="high"),
        swapped=swapped,
    )


def test_each_crop_yields_one_real_and_one_fake() -> None:
    samples = build_sbi_corpus([_crop("s1"), _crop("s2")])
    assert len(samples) == 4
    labels = sorted(s.context.label for s in samples)
    assert labels == [0, 0, 1, 1]


def test_reals_carry_no_generator_and_fakes_carry_sbi() -> None:
    """bench/protocol.py:78-86 rejects a corpus that gets this wrong."""
    samples = build_sbi_corpus([_crop("s1")])
    reals = [s for s in samples if s.context.label == 0]
    fakes = [s for s in samples if s.context.label == 1]
    assert all(s.context.generator is None for s in reals)
    assert all(s.context.generator == SBI_GENERATOR for s in fakes)


def test_a_real_and_its_own_fake_share_a_subject_but_not_a_source() -> None:
    """bench/protocol.py:92-96 requires one (subject, generator) pair per source.

    Sharing a source_id would make that source carry both (subj, None) and
    (subj, "sbi") and the corpus would be rejected. Sharing subject_id is
    required in the other direction, so identity-disjoint splitting keeps a
    face and its own pseudo-fake on the same side of every fold.
    """
    samples = build_sbi_corpus([_crop("s1")])
    real = next(s for s in samples if s.context.label == 0)
    fake = next(s for s in samples if s.context.label == 1)
    assert real.context.subject_id == fake.context.subject_id == "s1"
    assert real.observations[0].source_id != fake.observations[0].source_id


def test_a_swapped_session_is_refused_outright() -> None:
    """The 7 swapped sessions are the evaluation set. Blending them would
    train on the only labelled fraud this project has."""
    with pytest.raises(EvaluationOnlySessionError, match="s_fraud"):
        build_sbi_corpus([_crop("ok"), _crop("s_fraud", swapped=True)])


def test_the_corpus_is_reproducible_for_a_given_seed() -> None:
    a = build_sbi_corpus([_crop("s1"), _crop("s2")], seed=5)
    b = build_sbi_corpus([_crop("s1"), _crop("s2")], seed=5)
    for x, y in zip(a, b):
        assert x.sample_id == y.sample_id
        assert np.array_equal(x.observations[0].payload, y.observations[0].payload)


def test_reproducibility_survives_a_different_hash_seed() -> None:
    """Same seed, fresh interpreter, randomised PYTHONHASHSEED: same bytes.

    Within one process, `hash()` and `hashlib` are indistinguishable here, so
    the test above passes either way. This one is the guard that matters —
    str hashing is salted per process, and a corpus seeded from it would be
    irreproducible between runs while every in-process test stayed green.
    """
    import os
    import subprocess
    import sys
    import textwrap

    repo_root = Path(__file__).resolve().parents[2]
    script = textwrap.dedent("""
        import numpy as np
        from corpora.face_pool import FaceCrop
        from corpora.sbi import build_sbi_corpus
        from dfd.faces import FaceBox
        from dfd.types import Quality
        lms = np.array([[70., 80.], [110., 80.], [90., 100.],
                        [75., 125.], [105., 125.]])
        img = np.random.default_rng(11).integers(40, 210, (224, 224, 3),
                                                 dtype=np.uint8)
        crop = FaceCrop(session_id="s1", frame_index=0, image=img,
                        box=FaceBox(x=55, y=55, w=70, h=90, landmarks=lms,
                                    score=0.99),
                        quality=Quality(inter_ocular_px=40.0, blur_var=120.0,
                                        yaw_deg=0.0, pitch_deg=0.0,
                                        exposure=0.5, band="high"),
                        swapped=False)
        s = build_sbi_corpus([crop], seed=5)
        fake = next(x for x in s if x.context.label == 1)
        print(int(fake.observations[0].payload.astype(np.int64).sum()))
    """)
    outs = set()
    for hashseed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hashseed}
        r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                           text=True, env=env, cwd=str(repo_root))
        assert r.returncode == 0, r.stderr
        outs.add(r.stdout.strip())
    assert len(outs) == 1, f"corpus changed with PYTHONHASHSEED: {outs}"


def test_a_different_seed_changes_the_fakes_but_not_the_reals() -> None:
    a = build_sbi_corpus([_crop("s1")], seed=1)
    b = build_sbi_corpus([_crop("s1")], seed=2)
    real_a = next(s for s in a if s.context.label == 0)
    real_b = next(s for s in b if s.context.label == 0)
    fake_a = next(s for s in a if s.context.label == 1)
    fake_b = next(s for s in b if s.context.label == 1)
    assert np.array_equal(real_a.observations[0].payload,
                          real_b.observations[0].payload)
    assert not np.array_equal(fake_a.observations[0].payload,
                              fake_b.observations[0].payload)


def test_samples_are_images_carrying_the_crops_quality() -> None:
    samples = build_sbi_corpus([_crop("s1")])
    for s in samples:
        assert s.modality is Modality.IMAGE
        assert len(s.observations) == 1
        assert s.observations[0].quality is not None
        assert s.observations[0].quality.band == "high"


def test_a_single_generator_corpus_cannot_be_logo_split() -> None:
    """Holding out the only generator leaves nothing to train on.

    This is the protocol working, not failing:
    `bench.protocol._require_measurable` demands non-empty train fakes for
    every fold, and a fold that holds out generator g sends every
    g-labelled fake to test by definition. With a single generator, g is
    the corpus's entire fake population, so train fakes is provably always
    empty -- confirmed at subject counts 2..20 and at `logo_splits` seeds
    0..19 for this exact corpus (see task-3-report.md). A self-blend-only
    corpus therefore offers the LOGO protocol zero folds, not the "single
    fold" an earlier draft of this module claimed; a second licence-clean
    generator family is a precondition for running the LOGO benchmark at
    all (docs/HANDOFF.md §4).
    """
    from bench.protocol import UnsplittableCorpusError, logo_splits
    crops = [_crop(f"s{i}") for i in range(6)]
    samples = build_sbi_corpus(crops)
    records = [
        {"sample_id": s.sample_id,
         "subject_id": s.context.subject_id,
         "source_id": s.observations[0].source_id,
         "generator": s.context.generator,
         "label": s.context.label}
        for s in samples
    ]
    with pytest.raises(UnsplittableCorpusError, match="no train fakes"):
        logo_splits(records)


def test_the_id_scheme_satisfies_the_protocol_validator() -> None:
    """The end-to-end contract: this builder's ids make a corpus splittable,
    proven by adding a second generator and getting real folds.

    A single-generator corpus can never itself demonstrate the id scheme is
    `_validate`-clean via a successful split (see the test above), so this
    adds a second generator's fake rows by hand for half the subjects --
    distinct source_id, same subject_id, label=1 -- and shows `_validate`
    (which `logo_splits` runs first) accepts the mix and `logo_splits`
    produces one fold per generator. A straddling source or a mislabelled
    real would raise before any split were built, which is exactly what
    this test would catch.
    """
    from bench.protocol import logo_splits
    crops = [_crop(f"s{i}") for i in range(6)]
    samples = build_sbi_corpus(crops)
    records = [
        {"sample_id": s.sample_id,
         "subject_id": s.context.subject_id,
         "source_id": s.observations[0].source_id,
         "generator": s.context.generator,
         "label": s.context.label}
        for s in samples
    ]
    # A second generator's fakes for half the subjects -- enough for both
    # generators to have train and test representation at the default seed.
    records += [
        {"sample_id": f"s{i}-other", "subject_id": f"s{i}",
         "source_id": f"s{i}:other", "generator": "other", "label": 1}
        for i in range(3)
    ]
    splits = logo_splits(records)
    assert {s.held_out_generator for s in splits} == {SBI_GENERATOR, "other"}
    assert len(splits) == 2


def test_the_fake_still_differs_when_the_original_box_sits_outside_the_crop() -> None:
    """Regression guard for `_crop_box`. Do not delete as redundant: the
    brief that specified this module claimed the `_crop_box` mutation (using
    `crop.box` instead of the crop-space box) would fail
    `test_a_different_seed_changes_the_fakes_but_not_the_reals`, and it does
    not -- checked by running the mutation, not assumed. Every `_crop()`
    fixture's box (`x=55, y=55, w=70, h=90`) already fits inside its 224x224
    image whether read as crop-space or original-frame coordinates, because
    no fixture ever gives the "original frame" a different size or position
    than the crop, so `_crop_box`'s remap and the identity mapping coincide
    and no test built only from `_crop()` can tell them apart. This test
    exists because that gap needed a fixture that doesn't coincide: a box at
    (800, 600), standing in for a real original-frame box on a much larger
    source frame, far outside the 224x224 canvas. A real `FaceCrop.box` is in
    the ORIGINAL frame's coordinates, which can sit nowhere near the aligned
    crop; blending under it directly renders no mask at all, so the fake
    would come out byte-identical to the real -- verified by mutation (see
    task-3-report.md) that this fails without `_crop_box`.
    """
    lms = np.array([[820.0, 630.0], [860.0, 630.0], [840.0, 650.0],
                    [825.0, 675.0], [855.0, 675.0]])
    rng = np.random.default_rng(99)
    crop = FaceCrop(
        session_id="s9", frame_index=0,
        image=rng.integers(40, 210, (224, 224, 3), dtype=np.uint8),
        box=FaceBox(x=800, y=600, w=70, h=90, landmarks=lms, score=0.99),
        quality=Quality(inter_ocular_px=40.0, blur_var=120.0, yaw_deg=0.0,
                        pitch_deg=0.0, exposure=0.5, band="high"),
        swapped=False)
    samples = build_sbi_corpus([crop])
    real = next(s for s in samples if s.context.label == 0)
    fake = next(s for s in samples if s.context.label == 1)
    assert not np.array_equal(real.observations[0].payload,
                              fake.observations[0].payload)
