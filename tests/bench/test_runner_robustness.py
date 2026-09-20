import numpy as np
from bench.robustness import JPEG_QUALITIES, robustness_sweep
from bench.runner import RunConfig, run_benchmark
from dfd.detectors.base import Registry, SyntheticDetector


def _records(n=30):
    out = []
    for i in range(n):
        fake = i % 2 == 0
        out.append({
            "sample_id": f"s{i}", "subject_id": f"p{i}",
            # The runner reads source_id (spec §8.2 guard 2); an image is its
            # own source and that is recorded, never aliased to sample_id.
            "source_id": f"src{i}",
            # Two generators: leave-one-generator-out is undefined with one.
            "generator": ["deepfacelive", "faceswap"][(i // 2) % 2] if fake else None,
            "label": 1 if fake else 0,
            "compression": ["c0", "c23", "c40"][i % 3],
            "face_detector": "yunet", "align": "v1",
            # At least 128px on the short side, and non-square. A 64x64 image
            # measures quality band "reject", below SyntheticDetector's floor,
            # so the whole corpus abstains and every number comes back nan
            # while the suite still passes.
            "image": np.random.default_rng(i).integers(
                0, 255, (128 + (i % 3) * 16, 160 + (i % 5) * 16, 3),
                dtype=np.uint8),
        })
    return out


def _registry():
    reg = Registry()
    reg.register(SyntheticDetector(name="synth_a", seed=1))
    return reg


def test_robustness_is_off_by_default():
    rec = run_benchmark(_records(), _registry(), RunConfig(seed=1))
    assert rec.detector_results["synth_a"].tpr_by_perturbation == {}


def test_robustness_reports_one_entry_per_sweep_variant():
    """Keyed on what `robustness_sweep` actually emits, not on PERTURBATIONS.
    Those differ deliberately: the sweep expands `jpeg` into one entry per
    quality in JPEG_QUALITIES, because spec §8.3 asks for a curve and a single
    quality cannot show where a detector falls off."""
    rec = run_benchmark(_records(), _registry(),
                        RunConfig(seed=1, robustness=True))
    got = rec.detector_results["synth_a"].tpr_by_perturbation
    expected = set(robustness_sweep(_records(1)[0]["image"]))
    assert set(got) == expected
    assert "clean" in got


def test_the_whole_jpeg_quality_curve_is_measured():
    """A single JPEG point would let a detector look robust at q=90 while
    collapsing at q=10, which is the regime real uploads live in."""
    rec = run_benchmark(_records(), _registry(),
                        RunConfig(seed=1, robustness=True))
    got = rec.detector_results["synth_a"].tpr_by_perturbation
    for quality in JPEG_QUALITIES:
        assert f"jpeg_q{quality}" in got


def test_physical_recapture_paths_are_measured():
    """Spec acceptance criterion 9 — the reason this task exists.

    Key presence alone is not measurement: a corpus that abstains everywhere
    produces every key with a nan value. Require real numbers.
    """
    rec = run_benchmark(_records(), _registry(),
                        RunConfig(seed=1, robustness=True))
    got = rec.detector_results["synth_a"].tpr_by_perturbation
    for name in ("screenshot_recapture", "print_recapture"):
        assert name in got
        assert got[name] == got[name], f"{name} is nan — nothing was measured"
        assert 0.0 <= got[name] <= 1.0


def test_the_clean_baseline_is_measured_too():
    """Without it the perturbed numbers have nothing to be compared against."""
    rec = run_benchmark(_records(), _registry(),
                        RunConfig(seed=1, robustness=True))
    clean = rec.detector_results["synth_a"].tpr_by_perturbation["clean"]
    assert clean == clean


def test_robustness_run_is_reproducible():
    a = run_benchmark(_records(), _registry(), RunConfig(seed=1, robustness=True))
    b = run_benchmark(_records(), _registry(), RunConfig(seed=1, robustness=True))
    assert (a.detector_results["synth_a"].tpr_by_perturbation
            == b.detector_results["synth_a"].tpr_by_perturbation)
