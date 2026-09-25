# Benchmark run

## Reproducibility record

- seed: `0`
- dataset hash: `ffd39424542b9ebd02cc85080e8534d84f2dee9f3b3d0c9c1cc37651fca77257`
- guards enforced: `False`
- model versions: `blend_seam=0.2.0-fairface10k, effnet_b4=0.1.0, npr=0.1.0`
- identity disjointness (criterion 2): `violation`
  - **FAILED, and the run continued because guards are waived.** The numbers below are the measurement, not a pass.
  - worst check: max cosine `0.9908` at threshold `0.363`, 419 crossings (`5.4065%` of compared pairs), tolerated `100.0000%`, over 125 vs 125 ids
- adversarial robustness (criterion 8): `not_requested`
  - blend_seam: not attacked — `not_requested`
  - effnet_b4: not attacked — `not_requested`
  - npr: not attacked — `not_requested`
- demographic parity (criterion 11): `no_strata`

## Guards waived for this corpus

Two of the five spec §8.2 guards cannot pass over the DF40 eval subset, so
this run was made with `enforce_guards=False`. Both failures are properties
of the data, not of the run:

- **Compression coverage.** Every record carries `compression=unknown`: the
  repackaging does not say what quality its images were encoded at, so the
  worst-compression cell cannot be reported.
- **Generator attribution.** Every fake carries one generator id
  (`df40_unknown_mixture`). DF40 spans 40 techniques; this copy carries no
  per-technique label, so leave-one-generator-out is refused rather than
  faked from filename families.

- **Video-level sampling (guard 2).** `corpora.df40` groups frames of one
  filename family into one source (2026-09-23). That is the honest
  grouping, and it is exactly what guard 2 forbids: the guard wants one
  sample per source, and this corpus has up to 999. The interval below is
  computed over sources rather than rows because of it.

Read the in-dataset table below as a cross-corpus sanity check — the corpus
is unseen, which the capture corpus was not — and never as a LOGO result.

**The interval is what changed most.** The fake half of the test split is
1,601 images in 45 filename families, the largest holding 999 of them — a
Kish effective sample size of 2.4. Resampling rows, as every earlier run of
this report did, reported a precision the data does not have.

- **No signal is measurable here at all (2026-09-23).** Fit a model on
  SHUFFLED training labels — one that has learnt nothing by construction —
  and report it against this corpus: across four training pairs those models
  score 0.229–0.780. Every AUC this project has reported on this corpus sits
  inside that null, including an inverted 0.316 whose grouped interval
  excluded chance and which is now retracted (p=0.33). The cause is the
  corpus: its halves arrive down different imaging chains and 62% of its
  fakes are one family, so almost any direction in feature space separates
  them somewhat, in one direction or the other. The interval below resamples
  the EVALUATION corpus and is silent about the variance contributed by the
  FIT. **So read the table below as neither support NOR refutation.** Pair
  any number taken from it with `bench.metrics.permutation_null`, or do not
  report it.

## Leave-one-generator-out (spec §8.1)

**Not computed for this corpus.** Without a held-out-generator number there is nothing here that predicts field performance; the table below measures memorisation only.

## In-dataset results — memorisation, not field performance

These are computed over the whole corpus, with every generator seen. Spec §8.1: in-dataset AUC measures memorisation. Read the LOGO table above instead.

| detector | AUC | 95% CI | TPR@FPR=1% | TPR@FPR=0.1% | adversarial TPR@FPR=1% | ECE | abstained | p95 ms | n |
|---|---|---|---|---|---|---|---|---|---|
| blend_seam | 0.289 | 0.230–0.670 | 0.000 | 0.000 | n/a | 0.485 | 15.4% | 8.4 | 3207 |
| effnet_b4 | n/a | n/a–n/a | n/a | n/a | n/a | n/a | 100.0% | 0.0 | 3207 |
| npr | n/a | n/a–n/a | n/a | n/a | n/a | n/a | 100.0% | 0.0 | 3207 |
