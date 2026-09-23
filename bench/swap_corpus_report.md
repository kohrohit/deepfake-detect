# Benchmark run

## Reproducibility record

- seed: `0`
- dataset hash: `0dc880aab0057f39ed8a167493216b9945b4ef199feebc84acb37e46b77073ef`
- guards enforced: `False`
- model versions: `blend_seam=0.2.0-fairface10k, effnet_b4=0.1.0, npr=0.1.0`
- identity disjointness (criterion 2): `violation`
  - **FAILED, and the run continued because guards are waived.** The numbers below are the measurement, not a pass.
  - worst check: max cosine `0.6765` at threshold `0.363`, 383 crossings (`0.3070%` of compared pairs), tolerated `100.0000%`, over 500 vs 500 ids
  - held out swap_lowres_paste: max cosine `0.9688`, 19791 crossings (`0.2426%`)
  - held out swap_mouth_patch: max cosine `0.9688`, 16165 crossings (`0.1977%`)
  - held out swap_poisson: max cosine `0.9688`, 18216 crossings (`0.2232%`)
  - held out swap_warp_hull: max cosine `0.9688`, 19603 crossings (`0.2401%`)
- adversarial robustness (criterion 8): `not_requested`
  - blend_seam: not attacked — `not_requested`
  - effnet_b4: not attacked — `not_requested`
  - npr: not attacked — `not_requested`
- demographic parity (criterion 11): `violation`
  - blend_seam: FPR ratio `infx` (ceiling `infx`) — Black|Female 0.0093, Black|Male 0.0000, East Asian|Female 0.0000, East Asian|Male 0.0247, Indian|Female 0.0000, Indian|Male 0.0000, Latino_Hispanic|Female 0.0000, Latino_Hispanic|Male 0.0192, Middle Eastern|Male 0.0095, Southeast Asian|Female 0.0118, Southeast Asian|Male 0.0286, White|Female 0.0000, White|Male 0.0305
  - excluded as too small to compare: Middle Eastern|Female (n=86)

## Leave-one-generator-out (spec §8.1)

Worst held-out generator per detector — the headline number. Reported as the worst rather than the mean for the same reason spec §8.2 guard 3 reports the worst compression cell: an average hides the generator an attacker will actually use.

| detector | worst AUC | held out swap_lowres_paste (dropped 2923) | held out swap_mouth_patch (dropped 2926) | held out swap_poisson (dropped 2924) | held out swap_warp_hull (dropped 2924) |
|---|---|---|---|---|---|
| blend_seam | **0.527** | 0.928 | 0.527 | 0.640 | 0.633 |
| effnet_b4 | **n/a** | n/a | n/a | n/a | n/a |
| npr | **n/a** | n/a | n/a | n/a | n/a |

## In-dataset results — memorisation, not field performance

These are computed over the whole corpus, with every generator seen. Spec §8.1: in-dataset AUC measures memorisation. Read the LOGO table above instead.

| detector | AUC | 95% CI | TPR@FPR=1% | TPR@FPR=0.1% | adversarial TPR@FPR=1% | ECE | abstained | p95 ms | n |
|---|---|---|---|---|---|---|---|---|---|
| blend_seam | 0.661 | 0.644–0.678 | 0.076 | 0.006 | n/a | 0.271 | 59.8% | 7.0 | 8819 |
| effnet_b4 | n/a | n/a–n/a | n/a | n/a | n/a | n/a | 100.0% | 0.0 | 8819 |
| npr | n/a | n/a–n/a | n/a | n/a | n/a | n/a | 100.0% | 0.0 | 8819 |
