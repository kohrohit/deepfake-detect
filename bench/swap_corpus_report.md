# Benchmark run

## Reproducibility record

- seed: `0`
- dataset hash: `ba60232ea3a1e9dcf5e01a3014033e22b8ff6e40e114b3bd7b58549277359eaa`
- guards enforced: `False`
- model versions: `blend_seam=0.2.0-fairface10k, effnet_b4=0.1.0, npr=0.1.0`
- identity disjointness (criterion 2): `violation`
  - **FAILED, and the run continued because guards are waived.** The numbers below are the measurement, not a pass.
  - worst check: max cosine `0.7445` at threshold `0.363`, 320 crossings (`0.2565%` of compared pairs), tolerated `100.0000%`, over 500 vs 500 ids
  - held out swap_lowres_paste: max cosine `0.7997`, 19337 crossings (`0.2357%`)
  - held out swap_mouth_patch: max cosine `0.7997`, 17090 crossings (`0.2077%`)
  - held out swap_poisson: max cosine `0.7997`, 18383 crossings (`0.2241%`)
  - held out swap_warp_hull: max cosine `0.7997`, 19494 crossings (`0.2375%`)
- adversarial robustness (criterion 8): `not_requested`
  - blend_seam: not attacked — `not_requested`
  - effnet_b4: not attacked — `not_requested`
  - npr: not attacked — `not_requested`
- demographic parity (criterion 11): `violation`
  - blend_seam: FPR ratio `infx` (ceiling `infx`) — Black|Female 0.0000, Black|Male 0.0000, East Asian|Female 0.0260, East Asian|Male 0.0250, Indian|Female 0.0106, Indian|Male 0.0110, Latino_Hispanic|Female 0.0074, Latino_Hispanic|Male 0.0102, Middle Eastern|Male 0.0000, Southeast Asian|Female 0.0000, Southeast Asian|Male 0.0441, White|Female 0.0070, White|Male 0.0085
  - excluded as too small to compare: Middle Eastern|Female (n=102)

## Leave-one-generator-out (spec §8.1)

Worst held-out generator per detector — the headline number. Reported as the worst rather than the mean for the same reason spec §8.2 guard 3 reports the worst compression cell: an average hides the generator an attacker will actually use.

| detector | worst AUC | held out swap_lowres_paste (dropped 2939) | held out swap_mouth_patch (dropped 2938) | held out swap_poisson (dropped 2943) | held out swap_warp_hull (dropped 2937) |
|---|---|---|---|---|---|
| blend_seam | **0.537** | 0.923 | 0.537 | 0.663 | 0.645 |
| effnet_b4 | **n/a** | n/a | n/a | n/a | n/a |
| npr | **n/a** | n/a | n/a | n/a | n/a |

## In-dataset results — memorisation, not field performance

These are computed over the whole corpus, with every generator seen. Spec §8.1: in-dataset AUC measures memorisation. Read the LOGO table above instead.

| detector | AUC | 95% CI | TPR@FPR=1% | TPR@FPR=0.1% | adversarial TPR@FPR=1% | ECE | abstained | p95 ms | n |
|---|---|---|---|---|---|---|---|---|---|
| blend_seam | 0.653 | 0.634–0.671 | 0.093 | 0.005 | n/a | 0.288 | 60.5% | 8.0 | 8852 |
| effnet_b4 | n/a | n/a–n/a | n/a | n/a | n/a | n/a | 100.0% | 0.0 | 8852 |
| npr | n/a | n/a–n/a | n/a | n/a | n/a | n/a | 100.0% | 0.0 | 8852 |
