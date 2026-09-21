import math
from dataclasses import FrozenInstanceError

import pytest

from dfd.errors import InvalidInput
from dfd.policy import DEFAULT_POLICY, Policy


def test_default_policy_matches_the_thresholds_fusion_has_always_applied():
    assert DEFAULT_POLICY.fake_threshold == 1.0
    assert DEFAULT_POLICY.real_threshold == -1.0
    assert DEFAULT_POLICY.disagreement_ood == 3.0
    assert DEFAULT_POLICY.version == "p0-default-v0"


def test_policy_is_frozen():
    """A policy that can be edited after a record cites it makes the record a lie.

    `FrozenInstanceError`, not bare `Exception`: this repo's own handoff §6
    names "`pytest.raises` with no discriminating `match=`" as one of its
    recurring defects, and `pytest.raises(Exception)` is the same defect with
    the type widened instead of the message dropped — an AttributeError, a
    TypeError, or a typo raising NameError would all satisfy it.
    """
    with pytest.raises(FrozenInstanceError):
        DEFAULT_POLICY.fake_threshold = 2.0  # type: ignore[misc]


def test_real_threshold_must_sit_below_fake_threshold():
    with pytest.raises(InvalidInput, match="real_threshold"):
        Policy(fake_threshold=1.0, real_threshold=1.0)


def test_non_finite_thresholds_are_refused():
    """NaN compares false against everything, so a NaN threshold silently makes
    every verdict INSUFFICIENT_EVIDENCE instead of failing."""
    with pytest.raises(InvalidInput, match="fake_threshold"):
        Policy(fake_threshold=math.nan)


def test_disagreement_trigger_must_be_positive():
    with pytest.raises(InvalidInput, match="disagreement_ood"):
        Policy(disagreement_ood=0.0)


def test_version_must_be_a_non_empty_string():
    """policy_version is what an auditor uses to reconstruct the decision."""
    with pytest.raises(InvalidInput, match="version"):
        Policy(version="")
