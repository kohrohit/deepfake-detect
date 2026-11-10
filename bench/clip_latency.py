"""What one CLIP ViT-L/14 forward pass costs on the hardware that serves it.

WHY THIS IS ITS OWN TASK. Spec §8 records the encoder's CPU cost as "unmeasured and must be
benchmarked before any latency claim". The deployment is 8 vCPU, 15 GB and no GPU, and the
comparison that makes the question urgent is arithmetic: the ViT-Base/16 profiled for this
project is ~86M parameters and ~17.6 GFLOPs at 224px; ViT-L/14 is ~304M and ~81 GFLOPs -- about
3.5x the parameters and 4.5x the compute. A number extrapolated from the smaller model is a guess,
and this file exists so nobody has to make one.

IT ALSO DECIDES PHASE 2. LayerNorm tuning (Task 13 of the slot D plan) needs a BACKWARD pass
through the frozen encoder, which costs roughly three times a forward pass per image per step.
Whether that is an overnight job on a reduced corpus or is blocked on the GPU decision is
multiplication, once this number exists.

THE WARM-UP IS REPORTED, NOT DISCARDED. torch's first forward allocates, dispatches and warms
caches, and folding it into the mean produces a "latency" that mostly describes process startup.
The service forwards thousands of images per process, so steady state is the number that matters
-- but cold start is real too, and a reader deciding about a lambda-shaped deployment needs it.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dfd.detectors.clip_vit import CLIPVisualTower
from dfd.detectors.loading import load_model

logger = logging.getLogger(__name__)


def measure(tower: nn.Module, *, n: int, batch_size: int,
            seed: int = 0) -> dict[str, float]:
    """Per-image milliseconds over `n` images, in batches of `batch_size`.

    Returns mean, p50 and p95 per image, the warm-up batch's own per-image
    cost, and `torch.get_num_threads()` -- which belongs in the result
    because the same binary on the same box reports a different number under
    a different thread count, and a latency without it is unreproducible.
    """
    g = torch.Generator().manual_seed(seed)
    size = int(getattr(tower, "image_size", 224))
    batch = torch.randn(batch_size, 3, size, size, generator=g)

    start = time.perf_counter()
    with torch.no_grad():
        tower(batch)
    warmup_ms = (time.perf_counter() - start) * 1000.0 / batch_size

    per_image: list[float] = []
    done = 0
    while done < n:
        take = min(batch_size, n - done)
        chunk = batch[:take]
        start = time.perf_counter()
        with torch.no_grad():
            tower(chunk)
        per_image.append((time.perf_counter() - start) * 1000.0 / take)
        done += take

    times = np.asarray(per_image, dtype=np.float64)
    return {
        "n": n,
        "batch_size": batch_size,
        "mean_ms": float(times.mean()),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "warmup_ms": float(warmup_ms),
        "threads": int(torch.get_num_threads()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path,
                        default=Path("assets/models/clip_vit_l14_visual.pt"))
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path("bench/clip_latency.json"))
    parser.add_argument("--quantize", action="store_true",
                        help="dynamic INT8 over nn.Linear. Measured 1.52x on "
                             "the ViT-Base profiled for this project; whether "
                             "it holds at ViT-L/14 scale is what this flag is "
                             "for, and the quantised encoder is a DIFFERENT "
                             "encoder -- nothing fitted on the fp32 one may "
                             "be served by it without re-measuring")
    args = parser.parse_args(argv)

    tower = load_model(args.encoder, model_factory=CLIPVisualTower.vit_l14)
    tower.eval()
    if args.quantize:
        # torch.ao.quantization, not the torch.quantization alias: the
        # alias is the deprecated spelling and the deployment runs a
        # newer torch than this workstation.
        tower = torch.ao.quantization.quantize_dynamic(
            tower, {nn.Linear}, dtype=torch.qint8)

    result = measure(tower, n=args.n, batch_size=args.batch_size,
                     seed=args.seed)
    result["quantized"] = args.quantize
    result["encoder"] = str(args.encoder)
    # The Phase 2 arithmetic, carried beside the number so it cannot drift
    # from it. A LayerNorm-tuning step is forward + backward; ~3x forward is
    # the standard rule of thumb and is labelled as such, not measured here.
    result["implied_ln_tuning_step_ms"] = result["mean_ms"] * 3.0
    result["implied_ln_tuning_step_note"] = (
        "forward x 3, the usual rule of thumb for forward+backward. NOT "
        "measured. Multiply by the training corpus size for one epoch.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True,
                                   allow_nan=False) + "\n")
    logger.info("%s batch %d, %d threads: mean %.0f ms/image, p95 %.0f ms, "
                "cold start %.0f ms", "INT8" if args.quantize else "fp32",
                args.batch_size, result["threads"], result["mean_ms"],
                result["p95_ms"], result["warmup_ms"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
