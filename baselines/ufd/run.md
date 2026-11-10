# UFD (Universal Fake Image Detector) — run instructions

Runs entirely outside this repository, in its own checkout and its own Python environment. This
file only documents how; see `baselines/README.md` for why the code and weights never enter this
tree.

## Upstream

- **Paper**: Ojha, Li, Lee, "Towards Universal Fake Image Detectors that Generalize Across
  Generative Models," CVPR 2023. arXiv:2302.10174.
- **Repository**: <https://github.com/WisconsinAIVision/UniversalFakeDetect>. The paper's own
  links (project page and the README on this repo) point at
  `https://github.com/Yuheng-Li/UniversalFakeDetect`; GitHub's API resolves that to the same
  repository under its current name, `WisconsinAIVision/UniversalFakeDetect` (`full_name` /
  `html_url` read from `https://api.github.com/repos/Yuheng-Li/UniversalFakeDetect` this
  session, 2026-11-10). Same repo, not a fork — clone from either URL.
- **Commit observed at verification time (2026-11-10)**:
  `030495aea3300a8b54c0ec37ec7fe1dd7e63c619` (the `main` branch HEAD, read from
  `https://api.github.com/repos/Yuheng-Li/UniversalFakeDetect/commits/main`; message "Update
  README.md", authored 2026-08-19 by Utkarsh Ojha). **This is the commit this session observed,
  not a pin this repo enforces** — `main` can move. Whoever runs this baseline should record
  the commit their own clone actually checks out, in their own notes, since that is the
  provenance record a reproduction stands on.
- **Licence**: MIT. Read directly at
  `https://raw.githubusercontent.com/Yuheng-Li/UniversalFakeDetect/main/LICENSE` this session:
  the file reads, verbatim, "MIT License / Copyright (c) 2025 Wisconsin AI and Vision Lab
  (WAIV)" followed by the standard MIT grant text. GitHub's own licence detector
  (`https://api.github.com/repos/Yuheng-Li/UniversalFakeDetect`, field `license`) independently
  reports `spdx_id: MIT`, agreeing with the file. Permissive — this is an eval-only posture for
  the reasons in `baselines/README.md` regardless, since no weights from this repo ever enter
  this product.

## Environment

The upstream README's own install step is only:

```
pip install torch torchvision
```

No `requirements.txt`, no pinned versions, no conda environment file exists in the repository
(checked by listing its root tree this session — only `LICENSE`, `README.md`, `data/`,
`dataset_paths.py`, `models/`, `networks/`, `options/`, `pretrained_weights/`, `resources/`,
`test.sh`, `train.py`, `validate.py`). The CLIP backbone (`models/clip/`) is **vendored in the
repo**, not pip-installed from `openai/CLIP` — confirmed by listing that directory
(`clip.py`, `model.py`, `simple_tokenizer.py`, `bpe_simple_vocab_16e6.txt.gz`). The vendored
tokenizer is the standard OpenAI CLIP code, which commonly needs `ftfy` and `regex` at import
time; the upstream README does not list them.

UNVERIFIED — read at source before use: exact torch/torchvision versions, CUDA version, and
whether `ftfy`/`regex` are actually required by this vendored copy (inspect
`models/clip/simple_tokenizer.py`'s imports in the checkout before building the environment).
Build a dedicated venv or container for this baseline; never install any of this into this
repo's own environment (`baselines/README.md`, dependency containment).

## Weights

`pretrained_weights/fc_weights.pth`, committed directly inside the repository itself (not an
external download) — confirmed via the GitHub contents API this session: a 4,083-byte file at
that path, consistent with a linear probe fitted over a frozen CLIP ViT-L/14 tower (the paper's
`--fix_backbone` training mode: only the final linear layer is trained). No separate weight
licence question beyond the repository's own MIT licence above — this is the only artefact this
baseline needs beyond the vendored CLIP backbone.

## What the native script does NOT do, verified by reading it

`validate.py` (fetched and read directly this session) only ever returns **aggregate** AP and
accuracy for a dataset; `y_pred` (the per-image sigmoid output) is computed inside `validate()`
but discarded after the aggregate metrics are derived — it is never written per-image, per-path.
The per-image file paths do exist in order: `RealFakeDataset.total_list` is built from
`real_list + fake_list` and the `DataLoader` is constructed with `shuffle=False`, so
`total_list[i]` and the i-th prediction line up.

**The CLI also silently subsamples.** `RealFakeDataset.read_path` takes `--max_sample` (CLI
default 1000) and, if that value exceeds either class's count, resets it to **100** rather than
to the full count, then does `random.shuffle` on each class list before truncating to
`max_sample` per class. On Deepfake-Eval-2024 (767 real / 1,208 fake) the default flags will
neither score the whole corpus nor produce a reproducible-by-inspection ordering. `set_seed()`
(`SEED = 0`) is called before each dataset is built, so the shuffle is deterministic given the
same underlying file list order — but this script was never designed to emit a full-coverage,
per-item score file, and partial coverage from the native `--max_sample` must not be mistaken for
the partial coverage `bench/scorefile.py` reports honestly (a baseline that chose not to score
every image, vs. one that was never asked to).

**Conclusion: producing the `sample_id,score` CSV this project needs requires a thin wrapper
script, written and run entirely inside this baseline's own checkout and environment — never
inside this repository.** It must, at minimum:

1. Build the image list the same way `RealFakeDataset.read_path` does with `data_mode="ours"`
   (no `must_contain` filter — every file under the given directories), but **without** the
   `max_sample` truncation — pass every image.
2. Keep the list order fixed (no shuffle) so each path can be zipped 1:1 with its prediction.
3. Load the model exactly as `validate.py`'s `__main__` block does:
   `model = get_model("CLIP:ViT-L/14")`; `model.fc.load_state_dict(torch.load("pretrained_weights/fc_weights.pth", map_location="cpu"))`;
   `model.eval(); model.cuda()`.
4. Run the same preprocessing `RealFakeDataset.__getitem__` applies (`CenterCrop(224)`,
   `ToTensor`, CLIP mean/std) and the same forward UFD's `validate()` uses:
   `model(in_tens).sigmoid()`.
5. Map each file path to this corpus's `sample_id`. `corpora/dfeval.py` emits
   `sample_id = f"dfeval/{item_id}"`, where `item_id` is a column in that corpus's own
   `metadata.csv` manifest (see `corpora/dfeval.py`, `_manifest_rows`/`_MANIFEST_NAME`) — carry
   that same manifest (or an equivalent path → `item_id` lookup) into the baseline's environment
   so the wrapper can emit the correct `sample_id` per image rather than inventing one from the
   filename.

   **`corpora/dfeval._manifest_rows` is the authority on all of this, not this file.** The
   `metadata.csv` name, the `item_id` column and the `f"dfeval/{item_id}"` id format are repeated
   here only so a wrapper author can read one page; that function holds the real definition and is
   the only place in the repo that knows the corpus layout. **That layout is provisional** — it was
   written blind, against a gated download nobody here has yet seen — and Task 7 reconciles it
   against the real archive. **When Task 7 changes `_manifest_rows`, this file must be revised with
   it**, or a wrapper built from these instructions will emit ids the harness refuses as unknown.
6. Write `sample_id,score` rows, one per scored image, `score` = the sigmoid output above (or
   the pre-sigmoid logit — `bench.metrics.auc` is rank-based, either is accepted).

No such wrapper exists in this repository and none should be committed here; this section
documents what it must do, not a script to run.

## The figure to check a reproduction against

**AUC 0.56** on Deepfake-Eval-2024's image split, per
`docs/superpowers/specs/2026-11-10-benchmark-harness-design.md` §3 (sourced there from the paper
and the dataset card; not independently re-derived from the paper PDF in this session). Per the
honesty clause in `baselines/README.md`: a discrepancy against 0.56 could be the paper's split
definition, this wrapper, or the environment — report it, do not explain it away.
