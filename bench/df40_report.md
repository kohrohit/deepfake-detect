# Benchmark run

## Reproducibility record

- seed: `0`
- dataset hash: `44c5f339dabb1e4eddada249ab121d2ba0f3c13ce73a6e8f9272ad2980029aa0`
- guards enforced: `False`
- model versions: `blend_seam=0.2.0-fairface10k, effnet_b4=0.1.0, npr=0.1.0`

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

Read the in-dataset table below as a cross-corpus sanity check — the corpus
is unseen, which the capture corpus was not — and never as a LOGO result.

## Leave-one-generator-out (spec §8.1)

**Not computed for this corpus.** Without a held-out-generator number there is nothing here that predicts field performance; the table below measures memorisation only.

## In-dataset results — memorisation, not field performance

These are computed over the whole corpus, with every generator seen. Spec §8.1: in-dataset AUC measures memorisation. Read the LOGO table above instead.

| detector | AUC | 95% CI | TPR@FPR=1% | TPR@FPR=0.1% | adversarial TPR@FPR=1% | ECE | abstained | p95 ms | n |
|---|---|---|---|---|---|---|---|---|---|
| blend_seam | 0.289 | 0.271–0.308 | 0.000 | 0.000 | n/a | 0.485 | 15.4% | 9.1 | 3207 |
| effnet_b4 | n/a | n/a–n/a | n/a | n/a | n/a | n/a | 100.0% | 0.0 | 3207 |
| npr | n/a | n/a–n/a | n/a | n/a | n/a | n/a | 100.0% | 0.0 | 3207 |
