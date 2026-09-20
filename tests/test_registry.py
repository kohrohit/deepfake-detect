import numpy as np
import pytest
from dataclasses import FrozenInstanceError

from dfd.detectors.base import (
    BELOW_FLOOR,
    NO_OBSERVATIONS,
    NO_QUALITY,
    SyntheticDetector,
    abstain,
    filter_by_quality_floor,
)
from dfd.detectors.registry import Registry
from dfd.types import Modality, Observation, Quality


def _obs(band="high") -> Observation:
    q = Quality(inter_ocular_px=100, blur_var=200, yaw_deg=0,
                pitch_deg=0, exposure=0.5, band=band)
    return Observation(t=0.0, payload=np.zeros((64, 64, 3), np.uint8),
                       roi=(0, 0, 64, 64), quality=q, source_id="s1")


def test_abstain_helper_produces_zero_information_rawscore():
    r = abstain("npr", "0.1.0", "weights_absent")
    assert r.abstained and r.score is None and r.reason == "weights_absent"


def test_synthetic_detector_is_deterministic():
    """Must fail if detector returns constant, so verify both determinism AND variation."""
    d = SyntheticDetector(name="synth", seed=7)

    # Same payload with same seed produces same score (determinism)
    a = d.score([_obs()])
    b = d.score([_obs()])
    assert a.score == b.score, "Same payload should produce same score"
    assert a.score is not None, "Score should not be None"

    # Different payloads produce different scores (not constant)
    obs_different = Observation(
        t=0.0,
        payload=np.ones((64, 64, 3), np.uint8),  # Different from zeros
        roi=(0, 0, 64, 64),
        quality=Quality(inter_ocular_px=100, blur_var=200, yaw_deg=0,
                       pitch_deg=0, exposure=0.5, band="high"),
        source_id="s2"
    )
    c = d.score([obs_different])
    assert c.score is not None, "Score should not be None for different payload"
    assert a.score != c.score, "Different payloads must produce different scores"


def test_detector_abstains_below_its_quality_floor():
    """Detector below its quality floor reports below_quality_floor, not quality_not_measured."""
    d = SyntheticDetector(name="synth", seed=7, min_quality_band="high")
    r = d.score([_obs(band="low")])
    assert r.abstained, "Should abstain when below quality floor"
    assert r.reason == "below_quality_floor", f"Expected 'below_quality_floor', got '{r.reason}'"
    assert r.score is None, "Abstained score should be None"


def test_detector_abstains_when_quality_not_measured():
    """Detector abstains with quality_not_measured when obs has quality=None."""
    d = SyntheticDetector(name="synth", seed=7, min_quality_band="low")

    # Observation with quality=None
    obs_no_quality = Observation(
        t=0.0,
        payload=np.zeros((64, 64, 3), np.uint8),
        roi=(0, 0, 64, 64),
        quality=None,  # No quality measured
        source_id="s1"
    )
    r = d.score([obs_no_quality])
    assert r.abstained, "Should abstain when quality not measured"
    assert r.reason == "quality_not_measured", f"Expected 'quality_not_measured', got '{r.reason}'"
    assert r.score is None, "Abstained score should be None"


def test_detector_abstains_reasons_are_distinct():
    """Verify below_quality_floor and quality_not_measured are distinct outcomes."""
    d = SyntheticDetector(name="synth", seed=7, min_quality_band="high")

    # Case 1: quality measured but below floor
    r_below_floor = d.score([_obs(band="low")])

    # Case 2: quality not measured
    obs_no_quality = Observation(
        t=0.0,
        payload=np.zeros((64, 64, 3), np.uint8),
        roi=(0, 0, 64, 64),
        quality=None,
        source_id="s1"
    )
    r_no_quality = d.score([obs_no_quality])

    # Both abstain but for different reasons
    assert r_below_floor.abstained and r_below_floor.reason == "below_quality_floor"
    assert r_no_quality.abstained and r_no_quality.reason == "quality_not_measured"
    assert r_below_floor.reason != r_no_quality.reason


def test_registry_round_trips():
    """Must fail against stub registry that stores nothing."""
    reg = Registry()
    d1 = SyntheticDetector(name="synth1", seed=1)
    d2 = SyntheticDetector(name="synth2", seed=2)

    reg.register(d1)
    reg.register(d2)

    # Identity check: exact same object retrieved
    assert reg.get("synth1") is d1, "Should retrieve exact same detector object"
    assert reg.get("synth2") is d2, "Should retrieve exact same detector object"

    # Names list check: both detectors present and sorted
    names = reg.names()
    assert "synth1" in names, "synth1 should be in names list"
    assert "synth2" in names, "synth2 should be in names list"
    assert len(names) == 2, "Should have exactly 2 detectors"
    assert names == ["synth1", "synth2"], "Names should be sorted"


def test_registry_rejects_duplicate_names():
    reg = Registry()
    reg.register(SyntheticDetector(name="synth", seed=1))
    with pytest.raises(ValueError):
        reg.register(SyntheticDetector(name="synth", seed=2))


def test_subset_selection_is_seeded_and_reproducible():
    reg = Registry()
    for i in range(5):
        reg.register(SyntheticDetector(name=f"d{i}", seed=i))

    # Deterministic assertion: verify exact expected subsets for each seed
    # (prevents flaking when selecting 3 of 5 items)
    a = reg.select_subset(k=3, seed=99)
    b = reg.select_subset(k=3, seed=99)
    c = reg.select_subset(k=3, seed=100)

    # Primary assertion: exact expected subsets (verified empirically)
    assert a == ["d2", "d3", "d4"], f"Expected ['d2', 'd3', 'd4'], got {a}"
    assert b == ["d2", "d3", "d4"], f"Expected ['d2', 'd3', 'd4'], got {b}"
    assert c == ["d0", "d2", "d3"], f"Expected ['d0', 'd2', 'd3'], got {c}"

    # Secondary assertion: same seed gives same subset, different seeds differ
    assert a == b, "Same seed should produce identical subsets"
    assert a != c, "Different seeds should produce different subsets"


def test_subset_selection_caps_at_registry_size():
    reg = Registry()
    reg.register(SyntheticDetector(name="only", seed=1))
    assert reg.select_subset(k=10, seed=1) == ["only"]


def test_synthetic_detector_is_frozen():
    """FIX 1: SyntheticDetector must be frozen to protect registry identity invariant."""
    d = SyntheticDetector(name="synth", seed=7)
    with pytest.raises(FrozenInstanceError):
        d.name = "mutated"


def test_detector_abstains_reasons_mixed_batch():
    """FIX 2: Mixed batch (some quality=None, some below floor) should report below_quality_floor.

    The rule: if ANY observation carries measured quality that failed the floor,
    report below_quality_floor (more informative). Only when NO observation
    carries quality at all is it quality_not_measured.
    """
    d = SyntheticDetector(name="synth", seed=7, min_quality_band="high")

    obs_no_quality = Observation(
        t=0.0, payload=np.zeros((64, 64, 3), np.uint8),
        roi=(0, 0, 64, 64), quality=None, source_id="s1"
    )
    obs_low = _obs(band="low")

    # Order 1: unmeasured first, then below-floor
    r1 = d.score([obs_no_quality, obs_low])
    assert r1.abstained and r1.reason == "below_quality_floor", \
        "Should report below_quality_floor (measured failure trumps unmeasured)"

    # Order 2: below-floor first, then unmeasured
    r2 = d.score([obs_low, obs_no_quality])
    assert r2.abstained and r2.reason == "below_quality_floor", \
        "Should report below_quality_floor (same rule regardless of order)"


def test_synthetic_detector_no_observations_vs_quality_not_measured():
    """FIX 6: Distinguish no observations at all from quality not measured."""
    d = SyntheticDetector(name="synth", seed=7)

    # Empty list
    r_empty = d.score([])
    assert r_empty.abstained and r_empty.reason == "no_observations"

    # Non-empty but all have quality=None
    obs_no_quality = Observation(
        t=0.0, payload=np.zeros((64, 64, 3), np.uint8),
        roi=(0, 0, 64, 64), quality=None, source_id="s1"
    )
    r_no_quality = d.score([obs_no_quality])
    assert r_no_quality.abstained and r_no_quality.reason == "quality_not_measured"

    # Reasons are distinct
    assert r_empty.reason != r_no_quality.reason


def test_score_duplicates_accumulate_not_annihilate():
    """FIX 3: Duplicates must accumulate (addition) not cancel (XOR).

    XOR is self-inverse, so identical frames contribute nothing, which is
    catastrophic for video (static scenes, frozen injected streams). Addition
    makes duplicates accumulate.
    """
    d = SyntheticDetector(name="synth", seed=7)
    obs = _obs()

    # Score with one copy
    r_single = d.score([obs])
    assert r_single.score is not None
    assert r_single.score != 0.0, "Non-degenerate payload should not score 0.0"

    # Score with two identical copies
    r_double = d.score([obs, obs])
    assert r_double.score is not None
    assert r_double.score != 0.0, "Duplicate payload should not score 0.0"

    # Must be different (addition, not annihilation)
    assert r_single.score != r_double.score, \
        "Duplicate frames must produce different scores (accumulation, not cancellation)"


def test_registry_identity_invariant_cannot_be_broken():
    """FIX 1: Registry key stability: detector name is snapshotted at registration.

    Even if someone mutated a detector's name after registration (which is now
    impossible because it's frozen), the registry would still work correctly
    because it captured the name at registration time. This test verifies the
    snapshot mechanism.
    """
    reg = Registry()
    d = SyntheticDetector(name="original", seed=1)
    reg.register(d)

    # Detector is frozen, so mutation is impossible
    with pytest.raises(FrozenInstanceError):
        d.name = "mutated"

    # Registry still returns the correct object under the original name
    assert reg.get("original") is d
    assert "original" in reg.names()


def test_filter_by_quality_floor_covers_all_four_shapes():
    """The shared quality filter (dfd.detectors.base.filter_by_quality_floor)
    now backs SyntheticDetector, NPRDetector, and EffNetDetector. Exercise
    all four outcome shapes directly, plus both orderings of the mixed case,
    to pin the order-independence guarantee at the source instead of only
    through each detector that happens to call it.
    """
    # Shape 1: empty list -> NO_OBSERVATIONS
    usable, reason = filter_by_quality_floor([], "high")
    assert usable == [] and reason == NO_OBSERVATIONS

    obs_low = _obs(band="low")
    obs_none = Observation(
        t=0.0, payload=np.zeros((64, 64, 3), np.uint8),
        roi=(0, 0, 64, 64), quality=None, source_id="s1",
    )

    # Shape 2: all quality None -> NO_QUALITY
    usable, reason = filter_by_quality_floor([obs_none, obs_none], "high")
    assert usable == [] and reason == NO_QUALITY

    # Shape 3: all measured-below-floor -> BELOW_FLOOR
    usable, reason = filter_by_quality_floor([obs_low, obs_low], "high")
    assert usable == [] and reason == BELOW_FLOOR

    # Shape 4: mixed (some None, some below floor) -> BELOW_FLOOR,
    # in BOTH orderings (the order-dependent bug class this exists to kill).
    usable, reason = filter_by_quality_floor([obs_none, obs_low], "high")
    assert usable == [] and reason == BELOW_FLOOR, (
        f"[none, low] gave {reason}, expected {BELOW_FLOOR}"
    )
    usable, reason = filter_by_quality_floor([obs_low, obs_none], "high")
    assert usable == [] and reason == BELOW_FLOOR, (
        f"[low, none] gave {reason}, expected {BELOW_FLOOR}"
    )


def test_select_subset_rejects_negative_k():
    """FIX 5: select_subset must validate k >= 0."""
    reg = Registry()
    reg.register(SyntheticDetector(name="d1", seed=1))

    with pytest.raises(ValueError, match="k must be >= 0"):
        reg.select_subset(k=-1, seed=99)
