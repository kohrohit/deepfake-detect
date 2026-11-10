# DistilDIRE — run instructions

Runs entirely outside this repository, in its own checkout and its own Python environment. This
file only documents how; see `baselines/README.md` for why the code and weights never enter this
tree.

## Upstream

- **Paper**: Lim, Lee, Kim, Etzioni, "DistilDIRE: A Small, Fast, Cheap and Lightweight Diffusion
  Synthesized Deepfake Detection," 2024. arXiv:2406.00856.
- **Repository**: <https://github.com/miraflow/DistilDIRE>. Confirmed to exist and reachable
  this session (not archived, per `https://api.github.com/repos/miraflow/DistilDIRE`).
- **Commit observed at verification time (2026-11-10)**:
  `5a18afb351e11f1bd8ae64338e274200a1c701c9` (the `main` branch HEAD, read from
  `https://api.github.com/repos/miraflow/DistilDIRE/commits/main`; commit dated 2024-08-02,
  "Merge branch 'main' of https://github.com/miraflow/DistilDIRE"). No commits since, and no
  GitHub Releases exist on this repository (checked this session: zero releases). **This is the
  commit this session observed, not a pin this repo enforces** — record whatever commit your own
  clone actually checks out.
- **Licence**: **NonCommercial.** GitHub's own licence detector reports no SPDX-recognised
  `LICENSE` file (`license: null` from the repos API), but the README's own "Licenses" section,
  read directly from
  `https://raw.githubusercontent.com/miraflow/DistilDIRE/main/README.md` this session, states
  verbatim: "This work is licensed under the Creative Commons Attribution-NonCommercial 4.0
  license. You are free to read, share, and modify this code as long as you keep the original
  author attribution and non-commercial license." This is the licence this project records for
  DistilDIRE — CC BY-NC-4.0, stated in prose rather than a standalone `LICENSE` file. It governs
  why this baseline is internal-evaluation-only without exception (`baselines/README.md`).

## Environment

The upstream README does not provide a `requirements.txt`, a conda environment file, or a pinned
Python version (checked by reading the README and the repo's root file listing this session: only
`Dockerfile`, `DockerfileOcto`, `README.md`, `compute_dire_eps.sh`, `custommodel.py`, `dataset/`,
`distil.png`, `down_diffusiondb.sh`, `guided_diffusion/`, `networks/`, `server.py`, `test.py`,
`train.py`, `train.sh`, `utils/`). It states only "Minimum requirements: 1 GPU, 10GB VRAM" and
ships a `Dockerfile` as the implied reproducible environment.

UNVERIFIED — read at source before use: the exact contents of `Dockerfile` (base image, pinned
package versions) were not read in this session; read it in the checkout before building the
environment. Build a dedicated venv or container for this baseline; never install any of this
into this repo's own environment (`baselines/README.md`, dependency containment).

## Weights

Two things are needed, and their status differs:

1. **The ADM diffusion model** (used to reconstruct DIRE-equivalent features). The README gives
   a direct download:
   `https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_diffusion.pt`, with a
   `wget ... -O models/512x512-adm.pt` snippet. **This upstream README is internally
   inconsistent about this file**, read directly this session: its "Note" section says the repo
   "runs on ADM diffusion model (256x256, unconditional)," while the only download link it gives
   is the `512x512_diffusion.pt` checkpoint. This project did not resolve that discrepancy — do
   not guess which resolution the code path actually expects; check `guided_diffusion/` and
   `compute_dire_eps.sh` in the checkout for the resolution the model config actually loads
   before downloading either file.
2. **The DistilDIRE classifier checkpoint itself** (CelebA-HQ or ImageNet, 224×224). The README's
   own table lists both, but **both links are empty placeholders** — literally `[link]()` in the
   raw markdown, read directly this session (`https://raw.githubusercontent.com/miraflow/DistilDIRE/main/README.md`).
   No GitHub Release exists either (checked this session). **No working download link for a
   pretrained DistilDIRE checkpoint was found anywhere in this repository as of the commit
   above.** UNVERIFIED — read at source before use: check whether the authors have since fixed
   these links, published a release, or respond to a request; absent that, this baseline may
   only be reproducible by retraining it yourself via the README's `Train` section, which is a
   materially different (and much larger) undertaking than running a provided checkpoint.

## What the native script does NOT do, verified by reading it

`test.py` and `utils/trainer.py` (fetched and read directly this session) run inference through
`Trainer.validate(gather=False, save=True, save_name=...)`. That method accumulates `y_pred`
(sigmoid of the model's logit) and `y_true` across the whole validation loader, computes
`accuracy_score`, `average_precision_score`, and `precision_score`, and — when `save=True` —
writes **only those three aggregate numbers plus the real/fake counts** to the results file. The
per-item file `path` is available in the training loop (`for data, path in tqdm(self.val_loader, ...)`
inside `train()`) but is **not** threaded through `validate()`'s accumulation at all, and no
per-image score or path is ever written anywhere by the upstream code as checked this session.

**Conclusion: producing the `sample_id,score` CSV this project needs requires a thin wrapper**,
written and run entirely inside this baseline's own checkout and environment — never inside this
repository. It must, at minimum:

1. Reuse the dataset construction `test.py` uses (`TMIMGOnlyDataset(cfg.dataset_test_root,
   istrain=False)`) or an equivalent loader, over the Deepfake-Eval-2024 image split laid out in
   the `mydataset/test/images/{fakes,reals}` structure the README's "Data Preparation" section
   describes.
2. Build the `DataLoader` with `shuffle=False` (as `test.py` already does) so each prediction can
   be matched back to the image it came from by loader order, or modify the loop to carry the
   `path` value through to the output (the variable already exists in the per-batch tuple, as in
   `train()` above — `validate()` simply never looks at it).
3. Call `trainer.load_networks(cfg.pretrained_weights)` with the checkpoint located per the
   Weights section above, then run the forward pass the same way `validate()` does:
   `self.output['logit'].sigmoid()`.
4. Map each file path to this corpus's `sample_id`, the same way described in
   `baselines/ufd/run.md` step 5: `sample_id = f"dfeval/{item_id}"` from `corpora/dfeval.py`,
   using that corpus's own `metadata.csv` manifest as the path → `item_id` lookup, carried into
   the baseline's environment rather than reinvented from the filename.
5. Write `sample_id,score` rows, one per scored image. The sigmoid output is a valid monotone
   score (so is the pre-sigmoid logit, `self.output['logit']` — `bench.metrics.auc` is
   rank-based and accepts either).

No such wrapper exists in this repository and none should be committed here; this section
documents what it must do, not a script to run.

## The figure to check a reproduction against

**AUC 0.52** on Deepfake-Eval-2024's image split, per
`docs/superpowers/specs/2026-11-10-benchmark-harness-design.md` §3 (sourced there from the paper
and the dataset card; not independently re-derived from the paper PDF in this session). Per the
honesty clause in `baselines/README.md`: a discrepancy against 0.52 could be the paper's split
definition, the ADM-resolution ambiguity above, this wrapper, or the environment — report it, do
not explain it away. Given that no working pretrained checkpoint was found for this baseline (see
Weights above), the first and most likely outcome of attempting this reproduction is that it
cannot be run at all without contacting the authors or retraining from scratch — that is itself
the honest status to report, not a result to paper over by substituting a different checkpoint
silently.
