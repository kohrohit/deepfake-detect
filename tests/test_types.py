import numpy as np
import pytest
from dfd.types import (
    Modality, Verdict, Quality, Observation, Context, Sample, RawScore, Evidence,
    QUALITY_BANDS,
)


def _frame() -> np.ndarray:
    return np.zeros((64, 64, 3), dtype=np.uint8)


def test_quality_band_is_explicit_not_derived():
    q = Quality(inter_ocular_px=80.0, blur_var=120.0, yaw_deg=5.0,
                pitch_deg=2.0, exposure=0.5, band="high")
    assert q.band == "high"


def test_observation_carries_source_id_for_video_level_aggregation():
    obs = Observation(t=0.0, payload=_frame(), roi=(0, 0, 64, 64),
                      quality=None, source_id="vid_001")
    assert obs.source_id == "vid_001"


def test_sample_is_immutable():
    s = Sample(sample_id="s1", modality=Modality.IMAGE, observations=(), context=Context())
    with pytest.raises(Exception):
        s.sample_id = "s2"


def test_abstained_raw_score_has_no_score():
    r = RawScore(detector="npr", version="0.1.0", score=None,
                 abstained=True, reason="weights_absent")
    assert r.abstained and r.score is None


def test_zero_llr_means_no_information():
    e = Evidence(detector="npr", detector_version="0.1.0", llr=0.0, raw_score=None,
                 uncertainty=0.0, abstained=True, reason="below_quality_floor")
    assert e.llr == 0.0


def test_four_verdicts_exist():
    assert {v.value for v in Verdict} == {
        "real", "fake", "insufficient_evidence", "out_of_distribution"}


def test_quality_bands_are_ordered_worst_to_best():
    """Order is semantic: meets_floor() indexes into this tuple."""
    assert QUALITY_BANDS == ("reject", "low", "medium", "high")
    assert QUALITY_BANDS.index("reject") < QUALITY_BANDS.index("low")
    assert QUALITY_BANDS.index("low") < QUALITY_BANDS.index("medium")
    assert QUALITY_BANDS.index("medium") < QUALITY_BANDS.index("high")
