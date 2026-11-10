# Slot D (`gend_clip`) — leave-one-generator-out

Measured 2026-11-10 on 9907 records from 2000 photographs at offset 10000, 1001 couples.

## Corpus

- 9907 records, 5918 fake, 1001 couples
- families: {'real': 3989, 'swap_warp_hull': 973, 'swap_poisson': 984, 'swap_mouth_patch': 994, 'swap_lowres_paste': 970, 'sbi': 1997}
- skipped: {'no_face': 69}
- Kish ESS, fakes by family: **4.47** of 5 families
- Kish ESS, records by couple: **996.44** of 1001 couples

## Shortcut controls — read these before the folds

- **colour (Lab means and stds)**: AUC 0.530 — fitted and scored IN SAMPLE, which is the strongest form of this control: if even an in-sample shortcut model cannot reach the detector, the detector is not reading the shortcut.
- **JPEG quantisation**: AUC 0.500 — the feature is identical on every record, so there is nothing for a control to separate. Not a measured 0.5 -- a structural one.

## Folds

| held-out generator | AUC | 95% CI (by couple) | n | n fake |
|---|---|---|---|---|
| sbi | 0.738 | 0.726–0.751 | 2988 | 997 |
| swap_lowres_paste | 0.999 | 0.997–1.000 | 2475 | 484 |
| swap_mouth_patch | 0.984 | 0.977–0.991 | 2487 | 496 |
| swap_poisson | 0.995 | 0.992–0.998 | 2482 | 491 |
| swap_warp_hull | 0.998 | 0.996–0.999 | 2475 | 484 |

## The headline is the worst fold

**Worst fold: `sbi` at AUC 0.738** (95% CI 0.726–0.751).

The best fold, `swap_lowres_paste` at 0.999, is explicitly distrusted: it is the family most like the ones fitted on, and quoting it would describe this corpus's internal similarity rather than slot D's transfer.

## The null that judges it

Permutation null for the worst fold over 48 shuffles of the TRAINING labels: **0.449–0.561**, two-sided p = **0.02**.

**Outside the null, below the floor.** The worst fold 0.738 is outside the null 0.449–0.561 (p = 0.02), so there is signal — but it is under the 0.75 evidence floor, so slot D does not open the gate.

## What this corpus can and cannot show

- **These are classical composites, not generator output.** Self-blends and four OpenCV
  compositing techniques. A detector that beats them has NOT been shown to beat FSGAN, SimSwap,
  or any diffusion face. `corpora/swaps.py` is the authority on what each technique does; read it
  before quoting a number from here.
- **Every number is memorisation unless `--offset` is at or beyond the fitted detector's training
  window.** `bench/eval_swaps.py: TRAINING_WINDOWS` records the windows. This has produced a
  fabricated headline once already (blend_seam, a worst-generator 0.923 that became 0.527 when
  rebuilt at the right offset).
- **A fold is both generator-disjoint and group-disjoint.** Holding out a generator does not hold
  out the REALS: every couple contributes its reals to every family's fold. So each fold also
  splits couples, and tests only on couples the fit never saw. A fold that tests on training
  reals is half contaminated and reads as a strong fold.
- **The permutation null, not the point estimate, decides whether there is a result.** Every
  number this project has reported on DF40 sits inside the null a model that learnt nothing
  produces there.
