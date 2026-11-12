# Phase 2 against Phase 1 — LayerNorm-tuned versus frozen

Baseline encoder: `clip_vit_l14_visual.pt`  
Candidate encoder: `clip_vit_l14_lntuned.pt`

**The fold that decides it is `sbi`** — the baseline's worst, not the candidate's.

    frozen  0.738
    tuned   0.999
    paired delta +0.261  95% CI [+0.248, +0.273]

**Improved.** The delta is positive with a paired interval clear of zero.

| fold | frozen | tuned | paired delta | 95% CI | n | n fake | subjects |
|---|---|---|---|---|---|---|---|
| `sbi` **(worst)** | 0.738 | 0.999 | +0.261 | [+0.248, +0.273] | 2988 | 997 | 500 |
| `swap_mouth_patch` | 0.984 | 0.995 | +0.011 | [+0.004, +0.018] | 2487 | 496 | 500 |
| `swap_poisson` | 0.995 | 0.995 | +0.000 | [-0.003, +0.004] | 2482 | 491 | 500 |
| `swap_warp_hull` | 0.998 | 0.998 | +0.000 | [-0.002, +0.002] | 2475 | 484 | 500 |
| `swap_lowres_paste` | 0.999 | 1.000 | +0.001 | [+0.000, +0.002] | 2475 | 484 | 500 |

## What this still does not say

Not a measurement against any generator: every fake in this corpus is a self-blend or a classical composite made in this repo. Not permission to decide either — the 0.75 evidence floor is a separate gate and this comparison does not address it.
