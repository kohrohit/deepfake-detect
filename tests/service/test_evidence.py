"""The evidence gate: a detector decides only if it was measured to work.

This is the mechanism that keeps the service honest without anyone having to
remember to be. Every detector in the default registry must appear in the
card, and a detector whose measured AUC does not clear the floor never gets a
calibration curve — so it contributes llr 0.0 and the verdict stays
`insufficient_evidence`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dfd.detectors.registry import default_registry
from dfd.service.evidence import (
    DEFAULT_AUC_FLOOR,
    EVIDENCE_CARD_PATH,
    CardError,
    gated_detectors,
    load_card,
)


def _card(tmp_path: Path, detectors: dict) -> Path:
    path = tmp_path / "card.json"
    path.write_text(json.dumps({
        "format_version": 1,
        "generated_at": "2026-09-23T00:00:00Z",
        "auc_floor": DEFAULT_AUC_FLOOR,
        "detectors": detectors,
    }))
    return path


def test_a_detector_above_the_floor_is_allowed_to_decide(
        tmp_path: Path) -> None:
    card = load_card(_card(tmp_path, {
        "good": {"auc": 0.91, "corpus": "df40_eval_subset",
                 "trained_on": "fairface_corpus", "note": ""}}))
    assert gated_detectors(card) == {"good"}


def test_a_detector_below_the_floor_is_not_allowed_to_decide(
        tmp_path: Path) -> None:
    card = load_card(_card(tmp_path, {
        "blend_seam": {"auc": 0.289, "corpus": "df40_eval_subset", "note": ""}}))
    assert gated_detectors(card) == set()


def test_a_detector_at_chance_is_not_allowed_to_decide(
        tmp_path: Path) -> None:
    """0.52 is the measured ViT. Above 0.5 is not the same as working."""
    card = load_card(_card(tmp_path, {
        "vit": {"auc": 0.5214, "corpus": "df40_eval_subset", "note": ""}}))
    assert gated_detectors(card) == set()


def test_a_detector_that_was_never_measured_is_not_allowed_to_decide(
        tmp_path: Path) -> None:
    """Unmeasured must fail closed, exactly like the asset manifest."""
    card = load_card(_card(tmp_path, {
        "npr": {"auc": None, "corpus": None, "note": "no weights"}}))
    assert gated_detectors(card) == set()


def test_the_floor_can_be_raised_but_the_card_cannot_lower_it_silently(
        tmp_path: Path) -> None:
    """A card that ships its own lower floor must not weaken the caller's."""
    path = tmp_path / "card.json"
    path.write_text(json.dumps({
        "format_version": 1, "generated_at": "2026-09-23T00:00:00Z",
        "auc_floor": 0.30,
        "detectors": {"blend_seam": {"auc": 0.40, "corpus": "x", "note": ""}},
    }))
    card = load_card(path)
    assert gated_detectors(card, floor=DEFAULT_AUC_FLOOR) == set()


def test_an_unreadable_card_is_an_error_not_an_empty_gate(
        tmp_path: Path) -> None:
    """An empty gate and a broken card look identical downstream: nothing
    decides. Only an exception tells the operator which one happened."""
    path = tmp_path / "card.json"
    path.write_text("{not json")
    with pytest.raises(CardError):
        load_card(path)


def test_a_card_from_a_future_format_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "card.json"
    path.write_text(json.dumps({"format_version": 99, "detectors": {}}))
    with pytest.raises(CardError, match="format_version"):
        load_card(path)


def test_a_missing_card_is_an_error_not_an_empty_gate(tmp_path: Path) -> None:
    with pytest.raises(CardError):
        load_card(tmp_path / "absent.json")


def test_the_committed_card_covers_every_detector_in_the_registry() -> None:
    """The gate that stops a new detector shipping unmeasured.

    Adding a detector to `default_registry` without measuring it fails here,
    which is the only moment anyone is guaranteed to be looking.
    """
    card = load_card(EVIDENCE_CARD_PATH)
    assert set(default_registry().names()) <= set(card["detectors"])


def test_no_detector_currently_clears_the_floor() -> None:
    """A fact about this deployment, asserted so it cannot change unnoticed.

    If a detector is ever measured above the floor, this test fails and
    whoever raised it must come here and say so deliberately — which is the
    point at which the service starts issuing real verdicts.
    """
    assert gated_detectors(load_card(EVIDENCE_CARD_PATH)) == set()


def test_a_detector_measured_on_the_corpus_it_trained_on_cannot_decide(
        tmp_path: Path) -> None:
    """In-dataset AUC measures memorisation. It must not open the gate.

    This is the loophole a high number closes over: fit on a corpus's val
    split, measure on its test split, score 0.93, and the floor waves it
    through. Spec §8.1 calls in-dataset AUC memorisation for exactly this
    reason, and the gate has to encode that rather than trust whoever fills
    in the card.
    """
    card = load_card(_card(tmp_path, {
        "memoriser": {"auc": 0.93, "corpus": "df40_eval_subset",
                      "trained_on": "df40_eval_subset", "note": ""}}))
    assert gated_detectors(card) == set()


def test_a_detector_measured_on_a_corpus_it_did_not_train_on_can_decide(
        tmp_path: Path) -> None:
    card = load_card(_card(tmp_path, {
        "honest": {"auc": 0.93, "corpus": "df40_eval_subset",
                   "trained_on": "fairface_corpus", "note": ""}}))
    assert gated_detectors(card) == {"honest"}


def test_a_detector_that_does_not_say_what_it_trained_on_cannot_decide(
        tmp_path: Path) -> None:
    """Silence about provenance is not evidence of disjointness."""
    card = load_card(_card(tmp_path, {
        "vague": {"auc": 0.93, "corpus": "df40_eval_subset", "note": ""}}))
    assert gated_detectors(card) == set()


def test_the_committed_card_declares_training_provenance_for_every_detector(
        ) -> None:
    card = load_card(EVIDENCE_CARD_PATH)
    for name, entry in card["detectors"].items():
        assert "trained_on" in entry, f"{name} does not say what it trained on"
