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

## What this corpus can and cannot show

**These are classical compositing swaps — warp, mask, blend — and they are
not generator output.** A detector that beats them has not been shown to beat
FSGAN, InSwapper or any diffusion pipeline, and no number below may be
reported as though it had. What this corpus CAN do:

- **Refute a detector.** One that cannot find a hard-blended composite
  boundary will not find a subtle one.
- **Support leave-one-generator-out across these four techniques**, which is
  a real generalisation test between real techniques even though all four are
  classical. It is the only corpus in this project LOGO can fold at all.

Both halves are FairFace photographs, so the two sides share an imaging
chain and nothing here can learn "smooth means fake" — the defect that
invalidated every fit against SFHQ. What is measured is generalisation across
TECHNIQUE, never across corpus.

**Guards are waived.** Two of the five spec §8.2 guards fail on this corpus
and both failures are measured rather than hidden: identity (a few FairFace
couples are genuinely the same person or near-twins) and demographic parity
(the per-stratum FPR ratio exceeds its ceiling). The numbers are in the
reproducibility record above. Compression coverage is waived too — these
crops are PNG-encoded in memory and never re-compressed, so the corpus
carries one honest `compression=none` level rather than an invented c0/c23/c40
spread.

## Leave-one-generator-out (spec §8.1)

Worst held-out generator per detector — the headline number. Reported as the worst rather than the mean for the same reason spec §8.2 guard 3 reports the worst compression cell: an average hides the generator an attacker will actually use.

| detector | worst AUC | held out swap_lowres_paste (dropped 2923) | held out swap_mouth_patch (dropped 2926) | held out swap_poisson (dropped 2924) | held out swap_warp_hull (dropped 2924) |
|---|---|---|---|---|---|
| blend_seam | **0.527** | 0.928 | 0.527 | 0.640 | 0.633 |
| effnet_b4 | **n/a** | n/a | n/a | n/a | n/a |
| npr | **n/a** | n/a | n/a | n/a | n/a |

### Abstention by class, per fold

Every AUC above is computed over the records that did NOT abstain. Where these two columns differ, the fold's AUC describes the survivors rather than the technique — and the survivors of a quality floor are the least degraded fakes, which flatters the detector.

| held out | detector | genuine | held-out fakes |
|---|---|---|---|
| swap_lowres_paste | blend_seam | 56.8% (848/1493) | 75.6% (549/726) |
| swap_lowres_paste | effnet_b4 | 100.0% (1493/1493) | 100.0% (726/726) |
| swap_lowres_paste | npr | 100.0% (1493/1493) | 100.0% (726/726) |
| swap_mouth_patch | blend_seam | 56.8% (848/1493) | 57.4% (427/744) |
| swap_mouth_patch | effnet_b4 | 100.0% (1493/1493) | 100.0% (744/744) |
| swap_mouth_patch | npr | 100.0% (1493/1493) | 100.0% (744/744) |
| swap_poisson | blend_seam | 56.8% (848/1493) | 58.4% (426/730) |
| swap_poisson | effnet_b4 | 100.0% (1493/1493) | 100.0% (730/730) |
| swap_poisson | npr | 100.0% (1493/1493) | 100.0% (730/730) |
| swap_warp_hull | blend_seam | 56.8% (848/1493) | 59.7% (436/730) |
| swap_warp_hull | effnet_b4 | 100.0% (1493/1493) | 100.0% (730/730) |
| swap_warp_hull | npr | 100.0% (1493/1493) | 100.0% (730/730) |

## In-dataset results — memorisation, not field performance

These are computed over the whole corpus, with every generator seen. Spec §8.1: in-dataset AUC measures memorisation. Read the LOGO table above instead.

| detector | AUC | 95% CI | TPR@FPR=1% | TPR@FPR=0.1% | adversarial TPR@FPR=1% | ECE | abstained | p95 ms | n |
|---|---|---|---|---|---|---|---|---|---|
| blend_seam | 0.661 | 0.644–0.678 | 0.076 | 0.006 | n/a | 0.271 | 59.8% | 7.3 | 8819 |
| effnet_b4 | n/a | n/a–n/a | n/a | n/a | n/a | n/a | 100.0% | 0.0 | 8819 |
| npr | n/a | n/a–n/a | n/a | n/a | n/a | n/a | 100.0% | 0.0 | 8819 |
