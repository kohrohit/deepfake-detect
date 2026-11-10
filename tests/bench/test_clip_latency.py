from bench.clip_latency import measure
from dfd.detectors.clip_vit import CLIPVisualTower


def test_measure_reports_a_distribution_not_one_number():
    out = measure(CLIPVisualTower.tiny().eval(), n=4, batch_size=2, seed=0)
    assert out["n"] == 4 and out["batch_size"] == 2
    assert out["p95_ms"] >= out["p50_ms"] > 0.0
    assert out["threads"] >= 1


def test_the_first_pass_is_reported_separately_not_averaged_in():
    # torch's first forward allocates, dispatches and warms caches. Folding
    # it into the mean makes a latency that describes process startup, and
    # the service forwards thousands of images per process. It is reported
    # rather than discarded, because cold start is a real number -- just not
    # the one a steady-state p95 is claiming to be.
    out = measure(CLIPVisualTower.tiny().eval(), n=6, batch_size=1, seed=0)
    assert out["n"] == 6
    assert out["warmup_ms"] > 0.0
    assert out["p50_ms"] <= out["p95_ms"]
