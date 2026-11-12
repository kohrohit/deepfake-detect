# CLIP ViT-L/14 forward cost, measured on the deployment

Measured 2026-11-11 on the deployment instance itself — 8 vCPU, 15 GB, no GPU, torch 2.15.0+cpu,
Python 3.14.4, `torch.get_num_threads() == 4`. Spec §8 recorded this cost as "unmeasured and must
be benchmarked before any latency claim"; it is no longer unmeasured. `n=32`, batch size 1, one
discarded warm-up batch reported separately.

```
forward, fp32, batch 1:            276 ms/image   (p50 276, p95 277, cold start 281)
forward, dynamic INT8, batch 1:    177 ms/image   (p50 177, p95 177, cold start 181)
implied LN-tuning step (~3x fwd):  829 ms/image   superseded — see below
MEASURED LN-tuning step:           617 ms/image   batch 8, 4 threads, n=2 steps after warm-up
one epoch over 9,886 records:      1.70 hours     (the implied figure said 2.3 h)
```

**The rule of thumb was measured on 2026-11-11, when Phase 2 actually ran, and it was 26% too
pessimistic.** A real step — forward, cross-entropy, backward, AdamW — costs **617 ms/image** at
batch 8, against the 829 ms that forward-times-three implied. The reason is the thing that defines
Phase 2: only 102,400 of 303,966,208 parameters carry a gradient, so the backward pass computes
input gradients through the stack but weight gradients for almost nothing. At 8 threads the same
step is 604 ms — a 2% gain, which is why the run was left at the 4-thread configuration every
other number in this file was measured under.

INT8 is **1.57x**, which is within a hair of the 1.52x the spec recorded for the ViT-Base profiled
earlier — the speedup survives the 3.5x jump in parameters.

## The ruling on Phase 2

**Phase 2 (LayerNorm tuning, Task 13) is runnable here. It is NOT blocked on the GPU decision**,
at the corpus size slot D is actually fitted on.

One epoch over the 9,887-record training corpus is **2.3 hours** of fp32 forward-equivalent work,
so a 5-epoch tuning run is an overnight job on this box. That conclusion is conditional on the
corpus, and the condition is the whole finding:

| training corpus | one epoch |
|---|---|
| 9,887 records (`--limit 2000`, what was fitted) | 2.3 h |
| ~49,000 records (`--limit 10000`) | 11.3 h |
| ~300,000 records (`--limit 60000`, the plan's original) | 69.1 h |

At the plan's original 60,000-photograph window, one epoch is **69 hours** and Phase 2 would have
been blocked on hardware this project does not have. It is affordable only because the window is
small, which is the same arithmetic that set the window in the first place.

## What this does not say

- **INT8 is not a free upgrade and is not used by anything.** A quantised encoder is a DIFFERENT
  encoder: nothing fitted on the fp32 embeddings may be served by it without re-measuring the
  detector end to end. torch also now emits a deprecation warning for the quantized tensor
  creation path (`torch.quantize_per_tensor` and friends, "will be removed in a future PyTorch
  release"), so this is a measurement, not a plan.
- ~~**The 3x for a backward pass is a rule of thumb, not a measurement.**~~ **Measured
  2026-11-11** when Phase 2 was attempted, exactly as this line asked: 617 ms/image, 26% under the
  estimate. The forward numbers above are still `n=32`; the step number is `n=2` timed steps of 8
  images and is therefore a far weaker measurement than they are — enough to size a run, not
  enough to quote as a latency.
- **This is encoder cost alone** — no face detection, no decode, no the rest of the pipeline.
- **A local number is not this number.** The same harness on the workstation this was developed on
  (12 vCPU, torch 2.4.1+cu121) reports **1,637 ms/image**, roughly 6x slower, because that is a
  CUDA build running a CPU path. Any latency quoted for this product must come from the
  deployment's own box, which is what this file is.
