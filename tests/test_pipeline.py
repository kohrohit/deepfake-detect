import hashlib
import json
from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from dfd.calibration import Calibrator
from dfd.detectors.base import Registry, SyntheticDetector
from dfd.errors import InvalidInput, ResourceLimitExceeded
from dfd.faces import FaceBox
from dfd.limits import Limits
from dfd.pipeline import (DEGENERATE_BOX, DETECTOR_ERROR, NO_FACE,
                          NO_OBSERVATIONS, UNMEASURED, _worst_band, decide,
                          normalize)
from dfd.policy import Policy
from dfd.types import (Context, Modality, Observation, Quality, RawScore,
                       Sample, Verdict)


def _noise(size=256, seed=0):
    """High-variance image: a flat fill has blur_var ~0 and always bands 'reject'."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (size, size, 3), dtype=np.uint8)


def _box(x=10, y=10, w=200, h=200, iod=100.0):
    lm = np.array([[x + 20.0, y + 60.0], [x + 20.0 + iod, y + 60.0],
                   [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    return FaceBox(x=x, y=y, w=w, h=h, landmarks=lm, score=0.99)


def _sample(frames=1, seed=0):
    obs = tuple(
        Observation(t=float(i), payload=_noise(seed=seed + i), roi=None,
                    quality=None, source_id="s1")
        for i in range(frames))
    return Sample(sample_id="s1", modality=Modality.IMAGE, observations=obs,
                  context=Context())


def _detector(boxes, reason="ok"):
    return lambda frame, model_path: (list(boxes), reason)


def test_absent_face_weights_are_recorded_not_swallowed():
    """This is the state of the repo today: the YuNet ONNX is gitignored."""
    s, reasons = normalize(_sample(), detect=_detector([], reason="weights_absent"))
    assert reasons["faces"] == "weights_absent"
    assert s.observations[0].quality is None


def test_a_frame_with_no_face_is_distinguished_from_absent_weights():
    """Same verdict, completely different remedy."""
    _, reasons = normalize(_sample(), detect=_detector([], reason="ok"))
    assert reasons["faces"] == NO_FACE


def test_quality_is_attached_when_a_face_is_found():
    s, reasons = normalize(_sample(), detect=_detector([_box()]))
    assert reasons["faces"] == "ok"
    q = s.observations[0].quality
    assert q is not None and q.band in ("low", "medium", "high")
    assert s.observations[0].roi == (10, 10, 200, 200)


def test_inter_ocular_distance_comes_from_the_landmarks():
    s, _ = normalize(_sample(), detect=_detector([_box(iod=100.0)]))
    assert s.observations[0].quality.inter_ocular_px == pytest.approx(100.0)


def test_the_largest_face_is_the_one_measured():
    """v-CIP is single-subject; the subject is the big face, not the bystander."""
    small, large = _box(x=0, y=0, w=20, h=20), _box(x=30, y=30, w=180, h=180)
    s, _ = normalize(_sample(), detect=_detector([small, large]))
    assert s.observations[0].roi == (30, 30, 180, 180)


def test_a_box_running_past_the_frame_edge_is_clamped_not_crashed():
    """measure_quality slices without clamping; a negative start silently
    slices from the far end and an empty crop raises inside OpenCV."""
    s, reasons = normalize(_sample(), detect=_detector([_box(x=200, y=200, w=400, h=400)]))
    assert reasons["faces"] == "ok"
    x, y, w, h = s.observations[0].roi
    assert x + w <= 256 and y + h <= 256


def test_a_box_with_a_negative_origin_is_clamped_to_the_frame():
    """The exact defect this task exists to prevent: `measure_quality` slices
    frame[y:y+h, x:x+w] with no clamp of its own, so a negative x or y must
    never reach it. Every other fixture in this file uses a non-negative
    origin; this is the one that exercises `max(0, box.x)` / `max(0, box.y)`.
    """
    s, reasons = normalize(_sample(),
                           detect=_detector([_box(x=-20, y=-10, w=300, h=200)]))
    assert reasons["faces"] == "ok"
    roi = s.observations[0].roi
    assert roi == (0, 0, 256, 190)
    x, y, w, h = roi
    assert x >= 0 and y >= 0
    assert x + w <= 256 and y + h <= 256


def test_a_box_entirely_outside_the_frame_is_reported_not_measured():
    s, reasons = normalize(_sample(), detect=_detector([_box(x=300, y=300, w=50, h=50)]))
    assert reasons["faces"] == DEGENERATE_BOX
    assert s.observations[0].quality is None


def test_face_counts_are_recorded_so_a_crowd_is_visible():
    _, reasons = normalize(_sample(frames=2), detect=_detector([_box(), _box(x=40)]))
    assert reasons["frames_with_face"] == "2/2"
    assert reasons["max_faces_in_frame"] == "2"


def test_mixed_outcomes_across_frames_are_not_reported_as_success():
    calls = {"n": 0}

    def flaky(frame, model_path):
        calls["n"] += 1
        return ([], "weights_absent") if calls["n"] == 1 else ([], "ok")

    _, reasons = normalize(_sample(frames=2), detect=flaky)
    assert reasons["faces"] == "mixed"


def test_worst_band_is_worst_not_first_and_not_best():
    """Calibration conditions on the regime that held for the whole sample."""
    def obs(band):
        return Observation(t=0.0, payload=_noise(), roi=None, source_id="s",
                           quality=Quality(inter_ocular_px=100.0, blur_var=200.0,
                                           yaw_deg=0.0, pitch_deg=0.0,
                                           exposure=0.5, band=band))
    assert _worst_band([obs("high"), obs("low")]) == "low"
    assert _worst_band([obs("low"), obs("high")]) == "low"
    assert _worst_band([obs("medium"), obs("reject"), obs("high")]) == "reject"


def test_worst_band_of_nothing_measured_is_unmeasured():
    assert _worst_band(_sample().observations) == UNMEASURED


def test_a_sample_with_no_observations_reports_no_observations():
    """NO_OBSERVATIONS is named in normalize's own docstring as a value
    callers may see; an empty observation list is the edge case that
    produces it, and nothing else in this file constructs one."""
    empty = _sample(frames=0)
    assert empty.observations == ()
    _, reasons = normalize(empty, detect=_detector([]))
    assert reasons["faces"] == NO_OBSERVATIONS
    assert reasons["frames_with_face"] == "0/0"
    assert reasons["max_faces_in_frame"] == "0"


FIXED_TIME = "2026-09-21T10:00:00+00:00"


@pytest.fixture
def png(tmp_path):
    p = tmp_path / "subject.png"
    cv2.imwrite(str(p), _noise())
    return p


@dataclass(frozen=True)
class _FixedDetector:
    """Returns a chosen score without needing weights or quality."""
    name: str = "fixed"
    version: str = "test-1"
    modalities: frozenset = frozenset({Modality.IMAGE})
    min_quality_band: str = "low"
    value: float = 0.99

    def score(self, obs):
        return RawScore(detector=self.name, version=self.version, score=self.value,
                        abstained=False, reason="ok")


def _registry(*detectors):
    r = Registry()
    for d in detectors:
        r.register(d)
    return r


def _fitted_calibrator(name="fixed", band="high"):
    """Separable training data so a 0.99 score earns a strongly positive llr."""
    scores = [0.95 + 0.001 * i for i in range(20)] + [0.01 * i for i in range(20)]
    labels = [1] * 20 + [0] * 20
    return Calibrator(name).fit(scores, labels, [band] * 40)


def test_the_path_runs_end_to_end_and_abstains_for_stated_reasons(png):
    """Today's real behaviour: no face weights, no detector weights, no
    calibration. The value is that the record names all three separately."""
    record = decide(png, registry=_registry(SyntheticDetector(name="synthetic")),
                    detect=_detector([], reason="weights_absent"),
                    created_at=FIXED_TIME)
    assert record.verdict == Verdict.INSUFFICIENT_EVIDENCE.value
    assert record.stage_reasons["faces"] == "weights_absent"
    assert record.quality_band == UNMEASURED
    assert [e["reason"] for e in record.evidence] != []


def test_the_path_can_actually_reach_a_verdict(png):
    """THE load-bearing test. Every other test here passes on a pipeline that
    always abstains; this is the only one that does not. Without it, 'correctly
    abstaining' and 'broken in a way abstention hides' are indistinguishable."""
    record = decide(
        png,
        registry=_registry(_FixedDetector()),
        calibrators={"fixed": _fitted_calibrator()},
        detect=_detector([_box()]),
        created_at=FIXED_TIME,
    )
    assert record.quality_band == "high", "the injected face must band high"
    assert record.verdict == Verdict.FAKE.value
    assert record.llr_total > 1.0
    assert record.evidence[0]["abstained"] is False


def test_input_sha256_is_the_hash_of_the_file(png):
    expected = hashlib.sha256(png.read_bytes()).hexdigest()
    record = decide(png, registry=_registry(SyntheticDetector(name="s")),
                    detect=_detector([]), created_at=FIXED_TIME)
    assert record.input_sha256 == expected


def test_the_recorded_threshold_is_the_one_applied(png):
    strict = Policy(fake_threshold=4.0, real_threshold=-4.0, version="strict-v1")
    record = decide(png, registry=_registry(SyntheticDetector(name="s")),
                    detect=_detector([]), policy=strict, created_at=FIXED_TIME)
    assert record.threshold == 4.0
    assert record.policy_version == "strict-v1"


def test_the_policy_recorded_is_the_one_fuse_actually_applied(png):
    """`record.threshold` alone cannot catch `fuse` being called under a
    different policy than the one recorded: with an all-abstained detector
    (as above) llr_total is 0.0, which is INSUFFICIENT_EVIDENCE under both a
    lenient and a strict policy, so a `policy` argument dropped on the way
    into `fuse` would go unnoticed. This uses evidence with llr_total ~1.53
    (see test_the_path_can_actually_reach_a_verdict): FAKE under
    DEFAULT_POLICY's threshold of 1.0, but still INSUFFICIENT_EVIDENCE under
    a stricter threshold of 4.0. If `fuse` silently used DEFAULT_POLICY while
    the record claimed strict-v1, the verdict would betray it."""
    strict = Policy(fake_threshold=4.0, real_threshold=-4.0, version="strict-v1")
    record = decide(
        png,
        registry=_registry(_FixedDetector()),
        calibrators={"fixed": _fitted_calibrator()},
        detect=_detector([_box()]),
        policy=strict,
        created_at=FIXED_TIME,
    )
    assert record.verdict == Verdict.INSUFFICIENT_EVIDENCE.value


def test_an_unknown_extension_is_refused_by_name(png):
    other = png.with_suffix(".xyz")
    other.write_bytes(png.read_bytes())
    with pytest.raises(InvalidInput, match=r"\.xyz"):
        decide(other, registry=_registry(SyntheticDetector(name="s")))


def test_a_missing_file_is_a_dfd_error_not_an_oserror(tmp_path):
    with pytest.raises(InvalidInput, match="missing"):
        decide(tmp_path / "missing.png", registry=_registry(SyntheticDetector(name="s")))


def test_an_undecodable_file_is_invalid_input_not_a_bare_valueerror(tmp_path):
    """`b"not an image"` alone exercises the wrong branch: Pillow cannot even
    identify its header, so `check_image_before_decode` raises its own
    InvalidInput ("could not read image header...") before `_ingest`'s
    ValueError translation is ever reached. To reach the branch this test
    names, the header must be valid (so the limits gate passes and cv2 is
    asked to decode) while the pixel data is not: keep the PNG signature and
    IHDR chunk but cut the file right after the IDAT chunk header, dropping
    the compressed pixel data cv2 needs."""
    p = tmp_path / "broken.png"
    good = tmp_path / "good.png"
    cv2.imwrite(str(good), _noise())
    data = good.read_bytes()
    p.write_bytes(data[: data.find(b"IDAT") + 8])
    with pytest.raises(InvalidInput, match="decode"):
        decide(p, registry=_registry(SyntheticDetector(name="s")))


def test_the_decode_bomb_defence_is_reachable_through_decide(png):
    """Task 21's limits were exercised by their own unit tests and nothing
    else. This is the caller that makes them real."""
    with pytest.raises(ResourceLimitExceeded):
        decide(png, registry=_registry(SyntheticDetector(name="s")),
               limits=Limits(max_pixels=16))


def test_model_versions_name_every_registered_detector(png):
    record = decide(png, registry=_registry(SyntheticDetector(name="a"),
                                            SyntheticDetector(name="b")),
                    detect=_detector([]), created_at=FIXED_TIME)
    assert set(record.model_versions) == {"a", "b"}


def test_the_record_is_json_serialisable_end_to_end(png):
    record = decide(png, registry=_registry(SyntheticDetector(name="s")),
                    detect=_detector([]), created_at=FIXED_TIME)
    assert json.loads(record.to_json())["schema_version"] == "2"


@pytest.fixture
def mp4(tmp_path):
    """A real, decodable 30-frame clip (the working pattern from test_ingest).

    The video branch of `_ingest` is the one line of `pipeline.py` that no test
    reached, so a synthetic Sample will not do: the point is to execute
    `load_video` through `decide` with the arguments `decide` actually passes.
    """
    p = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (256, 256))
    for i in range(30):
        vw.write(_noise(seed=i))
    vw.release()
    return p


def test_the_video_path_runs_end_to_end_and_honours_max_frames(mp4):
    """`decide` passed `max_frames` and `seed` POSITIONALLY into `load_video`,
    where they are adjacent ints — a swap type-checks under mypy --strict and
    changes which frames are scored. `max_frames=2` must yield exactly two
    observations; under the swap it becomes `max_frames=0`, `load_video` finds
    no frames to keep and raises, and `_ingest` relabels that as an
    undecodable file. Asserting the count, not merely that a record exists, is
    what makes the swap visible.
    """
    record = decide(mp4, registry=_registry(SyntheticDetector(name="s")),
                    detect=_detector([_box()]), max_frames=2, seed=0,
                    created_at=FIXED_TIME)
    assert record.sample_id == "clip"
    assert record.stage_reasons["faces"] == "ok"
    assert record.stage_reasons["frames_with_face"] == "2/2"


def test_the_video_path_carries_every_frame_when_under_the_cap(mp4):
    """A multi-frame sample reaches the record: the clip holds 30 frames and
    the default cap is 32, so every frame is sampled and counted."""
    record = decide(mp4, registry=_registry(SyntheticDetector(name="s")),
                    detect=_detector([_box()]), created_at=FIXED_TIME)
    assert record.stage_reasons["frames_with_face"] == "30/30"
    assert record.stage_reasons["max_faces_in_frame"] == "1"


def test_the_video_seed_reaches_the_frame_sampler(mp4):
    """`seed` is the other half of the swapped pair. Two seeds must select
    different frames, which a payload-hashing detector turns into different
    scores; if `seed` never arrived, both runs would score identically."""
    def score_with(seed):
        record = decide(mp4, registry=_registry(SyntheticDetector(name="s", seed=0)),
                        detect=_detector([_box()]), max_frames=4, seed=seed,
                        created_at=FIXED_TIME)
        return record.evidence[0]["raw_score"]

    a, b = score_with(0), score_with(7)
    assert a is not None and b is not None, "the detector must not abstain here"
    assert a != b, "different seeds must sample different frames"


@dataclass(frozen=True)
class _RaisingDetector:
    """A detector whose weights file is present but corrupt.

    `npr.py` and `effnet.py` both log a load failure and then `raise`, so this
    is the shape of a real production failure, not an invented one.
    """
    name: str = "broken"
    version: str = "test-1"
    modalities: frozenset = frozenset({Modality.IMAGE})
    min_quality_band: str = "low"

    def score(self, obs):
        raise RuntimeError("corrupt checkpoint: cannot deserialise weights")


def test_a_raising_detector_becomes_an_abstention_not_an_outage(png, caplog):
    """Spec principle 8 — the system must remain useful with every ML slot
    defeated. Unisolated, one corrupt weights file on a production box means
    no audit record at all, no evidence from the healthy detector, and a
    traceback with exit 1. The failing slot must cost its own evidence and
    nothing else: the record is still produced, the broken detector abstains
    with `detector_error`, and the healthy detector's evidence still carries
    the decision to a real verdict.
    """
    with caplog.at_level("ERROR"):
        record = decide(png,
                        registry=_registry(_RaisingDetector(), _FixedDetector()),
                        calibrators={"fixed": _fitted_calibrator()},
                        detect=_detector([_box()]), created_at=FIXED_TIME)

    rows = {row["detector"]: row for row in record.evidence}
    assert set(rows) == {"broken", "fixed"}
    assert rows["broken"]["abstained"] is True
    # The LITERAL, not the imported constant: the reason string travels in the
    # audit record and downstream consumers match on it, so a test comparing
    # the constant to itself could not notice it being renamed.
    assert rows["broken"]["reason"] == "detector_error" == DETECTOR_ERROR
    assert rows["broken"]["llr"] == 0.0
    assert rows["fixed"]["abstained"] is False, \
        "the healthy detector's evidence must survive its neighbour's failure"
    assert record.verdict == Verdict.FAKE.value
    assert record.model_versions["broken"] == "test-1"
    # The traceback must not be swallowed: logger.exception carries exc_info.
    assert "corrupt checkpoint" in caplog.text
    assert "RuntimeError" in caplog.text


def test_every_detector_failing_still_produces_a_record(png):
    """The degenerate case of the same rule: a decision with no usable
    evidence is `insufficient_evidence` WITH a record naming why, not an
    exception with no record at all."""
    record = decide(png, registry=_registry(_RaisingDetector()),
                    detect=_detector([_box()]), created_at=FIXED_TIME)
    assert record.verdict == Verdict.INSUFFICIENT_EVIDENCE.value
    assert [row["reason"] for row in record.evidence] == ["detector_error"]
