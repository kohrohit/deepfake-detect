# Slot D — `gend_clip` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put one detector into the registry that measures above the 0.75 evidence floor on a
corpus capable of carrying a number, so the deployed service stops returning
`insufficient_evidence` for every input.

**Architecture:** A CLIP ViT-L/14 vision tower, reimplemented in-repo in pure torch so its
`state_dict` keys mirror OpenAI's own and no new runtime dependency is taken, loads from a
hash-pinned MIT weight file through `dfd.detectors.loading.load_model`'s secure path. Its
L2-normalised 768-d class-token embedding feeds a `Linear(768, 2)` head fitted here, from scratch,
on self-blends and in-repo composites of FairFace photographs. Embeddings are cached to disk once,
which is what makes a per-fold-refit LOGO, a permutation null and a grouped bootstrap affordable on
a CPU-only box. Phase 1 fits the head with the encoder frozen; Phase 2 (LayerNorm tuning, ~102K
parameters) is gated behind a measured decision about compute and writes a tuned encoder file, so
no loading machinery changes between the two.

**Tech Stack:** Python 3.10+, torch (already pinned), numpy, opencv, scikit-learn. **No new
runtime dependency** — see Task 1's ruling.

**Spec:** `docs/superpowers/specs/2026-11-10-own-model-detector-design.md` (and its parent,
`docs/superpowers/specs/2026-09-20-deepfake-detection-design.md`, which remains the authority).

---

## Global Constraints

Every task's requirements implicitly include all of these. Values are copied from the spec.

- **The evidence floor is `AUC 0.75`**, measured cross-corpus. It is the existing
  `dfd.service.evidence.DEFAULT_AUC_FLOOR`, not a new goalpost. A card cannot lower it.
- **A number is admissible only with all five of spec §7**: a permutation null beside it; intervals
  resampled over families, not rows; Kish effective sample size reported; a leave-one-generator-out
  fold reported per fold with the *best* fold explicitly distrusted; and a shortcut control
  (colour statistics and the JPEG header) reported beside it.
- **No FaceForensics++-derived weights, ever.** No fine-tuning from `yermandy/GenD_CLIP_L_14` and
  no initialising from it. The LayerNorm parameters and the head are fitted here from scratch.
- **Backbone is OpenAI CLIP ViT-L/14 (MIT).** Not DINOv3 — its terms are Meta's and unread.
- **Training consumes only**: FairFace (CC BY 4.0), SFHQ part 3 (CC0/MIT), and composites this repo
  makes from them. **No generator weight file participates in training.**
- **`vkyc-server/captures` and the existing `inswapper_128` swaps of them are EVALUATION ONLY**, and
  even that is flagged for legal review. Nothing in this plan may put them on a training path.
- **Every asset slot D touches needs a `commercial_use: true` entry in `assets/manifest.yaml`**
  citing a licence text read at its origin. The gate fails closed; unregistered means
  non-commercial. This is an acceptance criterion, not hygiene.
- **FairFace attribution and the CLIP MIT notice travel with anything shipped.**
  `training/export_fairface.py` already writes dataset/licence/URL into every exported session; it
  must not be stripped.
- **Weights load through `dfd.detectors.loading.load_model`'s secure path**: `weights_only=True`, a
  plain `state_dict`, an explicit `model_factory`. A full pickle is arbitrary code execution from a
  weights file and is not acceptable in a deployed service.
- **Inference is CPU-only** on the deployed instance (8 vCPU, 15 GB, no GPU). No latency claim
  without a measurement on that box.
- **No new runtime dependency.** `requirements-dev.txt` and `requirements-floor.txt` are exactly
  `==`-pinned and `tests/test_ci_gates.py` enforces that every pin sits inside its `pyproject.toml`
  range. Adding one means bumping both files and re-running all four gates in a clean venv.
- **Repo gates, every commit:** `ruff`, `mypy --strict`, `python3 -m pytest -q`, coverage ≥ 85%,
  and the asset gate. Current baseline: 928 tests, 92.8% coverage.
- **No claim of parity with Reality Defender** appears in any artefact this plan produces.
- The service binds `127.0.0.1` and has no authentication. Nothing here exposes a port.

---

## File Structure

**Created:**

| Path | Responsibility |
|---|---|
| `src/dfd/detectors/clip_vit.py` | The ViT-L/14 vision tower in pure torch, plus CLIP's preprocessing constants. Knows nothing about deepfakes. |
| `src/dfd/detectors/gend.py` | Slot D: `GendHead` (the fitted part) and `GendClipDetector` (the `Detector` protocol implementation). |
| `training/convert_clip.py` | One-time: OpenAI's distributed archive → a plain `state_dict` our tower loads, with a numeric check against the reference implementation. Never imported by `src/`. |
| `training/clip_features.py` | Content-addressed embedding cache. One forward pass per image, ever. |
| `training/fit_gend.py` | Builds the training corpus, fits the head, writes `assets/models/gend_clip.pt` and a report. |
| `bench/eval_gend.py` | The measurement: per-fold-refit LOGO, permutation null, family bootstrap, Kish ESS, shortcut control. Writes `bench/gend_report.json` and `.md`. |
| `bench/eval_gend_vcip.py` | The in-domain evaluation on the capture corpus and its `inswapper_128` swaps. Scoring only; fits nothing. |
| `src/dfd/service/contract.py` | `AuditRecord` → the RD-shaped `{verdict, score, models[], dissenting_models}` response. Pure function, no I/O. |
| `tests/test_clip_vit.py`, `tests/test_gend.py`, `tests/test_convert_clip.py`, `tests/test_clip_features.py`, `tests/test_fit_gend.py`, `tests/bench/test_eval_gend.py`, `tests/bench/test_eval_gend_vcip.py`, `tests/service/test_contract.py` | One test file per unit above. |

**Modified:**

| Path | Change |
|---|---|
| `src/dfd/detectors/registry.py` | Register slot D; `DEFAULT_GEND_WEIGHTS`, `DEFAULT_CLIP_WEIGHTS`. |
| `src/dfd/service/app.py` | `ServiceConfig` carries the two new weight paths. |
| `src/dfd/service/api.py` | `GET /api/submissions/<id>?format=rd`, and `POST /api/analyze`. |
| `bench/metrics.py` | `kish_ess`. |
| `bench/eval_swaps.py` | `TRAINING_WINDOWS` entry for slot D's version. |
| `training/fit_calibration.py` | A second record source, so calibration need not come from DF40. |
| `assets/manifest.yaml` | `clip_vit_l14_visual`, `gend_clip_weights`, `fairface_swap_corpus`, `vkyc_capture_corpus`, `vkyc_inswapper_swaps`. |
| `ops/fetch-assets.sh` | The CLIP encoder, hash-pinned. |
| `ops/deploy-ec2.sh` | Ship the encoder and the head; say what the transfer now costs. |
| `bench/evidence_card.json` | Slot D's measurement, with its caveat. |
| `tests/test_registry.py` | Three declared detectors become four. |
| `docs/HANDOFF.md` | A §0 resume block. |
| `docs/superpowers/specs/2026-11-10-own-model-detector-design.md` | §2 item 2 still describes the pre-commercialisation corpus §6 replaced. Fixed in Task 0. |

---

### Task 0: The spec contradicts itself about the corpus — **DONE 2026-11-10**

Applied and committed before the plan was handed over; the steps are kept so the reasoning is
readable. §2 item 2 read "A paired real/fake corpus built from `vkyc-server/captures`". §6.1 and §6.4
replaced that: FairFace is the training supply and the captures are evaluation only. An executor
reading the scope list before §6 would build the thing the design forbids.

**Files:**
- Modify: `docs/superpowers/specs/2026-11-10-own-model-detector-design.md:46-48`

- [x] **Step 1: Replace §2 item 2 with the §6 position**

```markdown
2. **A paired real/fake corpus** built from FairFace crops, where every fake is generated from a
   specific real crop and re-encoded identically. The v-CIP captures are evaluation only (§6.4).
```

- [x] **Step 2: Check no other scope line still points at the captures for training**

Run: `grep -n 'captures' docs/superpowers/specs/2026-11-10-own-model-detector-design.md`
Expected: every remaining hit is in §6.4, §9.1 or §9.2 and says *evaluation*.
Verified: eleven hits, all in §4.2, §6 and §9, none of them a training instruction.

- [x] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-11-10-own-model-detector-design.md
git commit -m "docs: the spec's scope list still named the corpus its own §6 forbids"
```

---

### Task 1: The CLIP ViT-L/14 vision tower, in this repo — **DONE 2026-11-10** (`ae3a518`)

**Ruling, recorded here because it will be questioned:** the tower is reimplemented rather than
imported. `open_clip_torch` and `transformers` would each add a dependency tree (huggingface_hub,
timm, regex, ftfy) to a product whose two requirements files are exactly `==`-pinned and whose
`tests/test_ci_gates.py` enforces that pinning, and neither is typed well enough for
`mypy --strict`. The tower is ~180 lines of standard, well-understood code; writing it keeps
`model_factory` honest (the secure load path *requires* us to own the architecture), and the module
names below mirror OpenAI's exactly, so the converter in Task 2 is nearly the identity map and its
correctness is checkable key by key.

**ViT-L/14, from OpenAI's config:** patch 14, width 1024, 24 layers, 16 heads, output 768, image
224 (so 16x16 = 256 patches + 1 class token = 257 positions). The activation is **QuickGELU**
(`x * sigmoid(1.702x)`), not `nn.GELU` — every OpenAI CLIP checkpoint was trained with it, and
substituting `nn.GELU` produces plausible-looking embeddings that are wrong.

**Files:**
- Create: `src/dfd/detectors/clip_vit.py`
- Test: `tests/test_clip_vit.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class CLIPVisualTower(nn.Module)` with
    `__init__(self, *, image_size: int = 224, patch_size: int = 14, width: int = 1024, layers: int = 24, heads: int = 16, output_dim: int = 768)`
    and `forward(self, x: torch.Tensor) -> torch.Tensor` taking `(B, 3, H, W)` normalised float and
    returning `(B, output_dim)`.
  - `CLIPVisualTower.vit_l14() -> CLIPVisualTower` — a `classmethod` taking no arguments, suitable
    as `load_model`'s `model_factory`.
  - `CLIPVisualTower.tiny() -> CLIPVisualTower` — `image_size=32, patch_size=16, width=32,
    layers=2, heads=2, output_dim=8`. Exists so every test in this plan runs in milliseconds with
    no 1.2 GB file present.
  - `CLIPVisualTower.layernorm_parameter_names(self) -> list[str]` — sorted `state_dict` keys of
    every LayerNorm weight and bias. Phase 2 tunes exactly these.
  - `CLIP_MEAN: tuple[float, float, float]`, `CLIP_STD: tuple[float, float, float]`
  - `def preprocess(crop: npt.NDArray[np.uint8], size: int = 224) -> torch.Tensor` — RGB HWC uint8
    in, `(1, 3, size, size)` normalised float32 out. Bicubic resize, CLIP's mean/std.
  - `PREPROCESS_VERSION: str = "clip-224-bicubic-v1"` — travels into the feature cache key.

- [x] **Step 1: Write the failing tests**

```python
"""The CLIP ViT-L/14 vision tower. Tests run on `tiny()`; the 1.2 GB file is never needed here."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from dfd.detectors.clip_vit import (
    CLIP_MEAN,
    CLIP_STD,
    PREPROCESS_VERSION,
    CLIPVisualTower,
    preprocess,
)


def test_tiny_tower_maps_a_batch_to_its_output_dim():
    tower = CLIPVisualTower.tiny()
    out = tower(torch.zeros(2, 3, 32, 32))
    assert out.shape == (2, 8)


def test_the_tower_is_deterministic_in_eval_mode():
    tower = CLIPVisualTower.tiny().eval()
    x = torch.linspace(-1.0, 1.0, 2 * 3 * 32 * 32).reshape(2, 3, 32, 32)
    with torch.no_grad():
        first, second = tower(x), tower(x)
    assert torch.equal(first, second)


def test_state_dict_keys_mirror_openai_clip_visual_names():
    # The converter in training/convert_clip.py strips a "visual." prefix and
    # nothing else. If these names drift, that map silently stops being the
    # identity and the weights land in the wrong tensors.
    keys = set(CLIPVisualTower.tiny().state_dict())
    assert "conv1.weight" in keys
    assert "class_embedding" in keys
    assert "positional_embedding" in keys
    assert "ln_pre.weight" in keys and "ln_post.bias" in keys
    assert "proj" in keys
    for i in range(2):
        for suffix in ("ln_1.weight", "ln_2.bias",
                       "attn.in_proj_weight", "attn.out_proj.bias",
                       "mlp.c_fc.weight", "mlp.c_proj.bias"):
            assert f"transformer.resblocks.{i}.{suffix}" in keys


def test_vit_l14_declares_the_published_shapes():
    tower = CLIPVisualTower.vit_l14()
    sd = tower.state_dict()
    assert sd["conv1.weight"].shape == (1024, 3, 14, 14)
    assert sd["positional_embedding"].shape == (257, 1024)
    assert sd["proj"].shape == (1024, 768)
    assert len([k for k in sd if k.endswith("attn.in_proj_weight")]) == 24


def test_layernorm_parameters_are_the_small_tunable_set():
    tower = CLIPVisualTower.vit_l14()
    names = tower.layernorm_parameter_names()
    sd = tower.state_dict()
    total = sum(int(sd[n].numel()) for n in names)
    # 24 blocks x 2 LayerNorms x (1024 weight + 1024 bias), plus ln_pre and
    # ln_post. The spec quotes GenD's "~91K"; this is the number OUR tower
    # actually exposes and it is the one to report.
    assert total == 102_400
    assert all(n.endswith(".weight") or n.endswith(".bias") for n in names)
    assert names == sorted(names)


def test_quickgelu_is_used_and_is_not_nn_gelu():
    # QuickGELU(x) = x * sigmoid(1.702x). At x = 1 that is 0.845...; nn.GELU
    # gives 0.8413. A tower built with the wrong activation still runs and
    # still produces 768 numbers, so only a numeric check catches it.
    from dfd.detectors.clip_vit import QuickGELU

    got = QuickGELU()(torch.tensor([1.0]))
    assert got.item() == pytest.approx(1.0 * torch.sigmoid(torch.tensor(1.702)).item())
    assert got.item() != pytest.approx(torch.nn.GELU()(torch.tensor([1.0])).item(), abs=1e-4)


def test_preprocess_normalises_with_clips_own_constants():
    crop = np.full((50, 40, 3), 255, dtype=np.uint8)
    t = preprocess(crop, size=32)
    assert t.shape == (1, 3, 32, 32)
    assert t.dtype == torch.float32
    expected = [(1.0 - m) / s for m, s in zip(CLIP_MEAN, CLIP_STD, strict=True)]
    for channel, want in enumerate(expected):
        assert t[0, channel].mean().item() == pytest.approx(want, abs=1e-4)


def test_preprocess_version_is_declared():
    # The feature cache keys on it: a change to resize or normalisation that
    # did not change this string would serve stale embeddings forever.
    assert PREPROCESS_VERSION
```

- [x] **Step 2: Run them and watch them fail**

Run: `python3 -m pytest tests/test_clip_vit.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'dfd.detectors.clip_vit'`.

- [x] **Step 3: Write the tower**

```python
"""OpenAI CLIP's ViT vision tower, reimplemented so this repo owns the architecture.

WHY THIS IS HERE AND NOT A DEPENDENCY. `loading.load_model`'s secure path needs a
`model_factory` -- a callable returning an uninitialised module -- because a state_dict alone
cannot become a model without the architecture. Owning the architecture is therefore a
precondition for loading the encoder securely, not a preference. `open_clip_torch` and
`transformers` would each add a dependency tree to two exactly-pinned requirements files
(`tests/test_ci_gates.py` enforces the pinning) and neither types cleanly under `mypy --strict`.

THE MODULE NAMES BELOW ARE NOT ARBITRARY. They mirror OpenAI's own `VisionTransformer`, so the
published checkpoint's `visual.*` keys map onto this module's `state_dict` by stripping one
prefix. `training/convert_clip.py` asserts that map is total in both directions; if these names
drift, that assertion is the thing that catches it.

UNVERIFIED UNTIL TASK 2 RUNS. The constants here (QuickGELU, pre-norm residual order, the
`ln_post(x[:, 0]) @ proj` read-out) are written from the published architecture. The check that
they are RIGHT is numeric and lives in `training/convert_clip.py --verify`, which compares this
tower against OpenAI's own distributed module on the same input. No claim here rests on the
numbers until that has run.
"""
from __future__ import annotations

import logging
from collections import OrderedDict

import cv2
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

#: CLIP's image normalisation. Not ImageNet's -- they differ in the third
#: decimal and using the wrong one shifts every embedding slightly, which
#: shows up as a mysteriously weak linear probe rather than as an error.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

#: Keyed into the feature cache. Bump it whenever resize, colour order or
#: normalisation changes, or the cache will serve embeddings computed under
#: the old rule for ever.
PREPROCESS_VERSION = "clip-224-bicubic-v1"

#: ViT-L/14 at 224px, from OpenAI's config.
VIT_L14 = {"image_size": 224, "patch_size": 14, "width": 1024,
           "layers": 24, "heads": 16, "output_dim": 768}


class QuickGELU(nn.Module):
    """`x * sigmoid(1.702x)`, which is what CLIP was trained with.

    `nn.GELU` is a different function (they differ by ~0.004 at x=1) and
    substituting it produces a tower that runs, returns the right shapes,
    and is quietly wrong everywhere.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """Pre-norm attention then pre-norm MLP, CLIP's layout and CLIP's names."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(width, width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(width * 4, width)),
        ]))
        self.ln_2 = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.ln_2(x))


class CLIPVisualTower(nn.Module):
    """CLIP's image encoder: patches, a class token, a transformer, a projection.

    `forward` returns the class token after `ln_post` and the projection --
    the 768-d image embedding CLIP itself publishes. It is NOT L2-normalised
    here: normalisation is slot D's read-out decision (`dfd.detectors.gend`),
    and burying it in the encoder would make the encoder's output disagree
    with every other CLIP implementation for no reason.
    """

    def __init__(self, *, image_size: int = 224, patch_size: int = 14,
                 width: int = 1024, layers: int = 24, heads: int = 16,
                 output_dim: int = 768) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError(
                f"image_size {image_size} is not a whole number of "
                f"{patch_size}px patches")
        self.image_size = image_size
        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size,
                               stride=patch_size, bias=False)
        scale = width ** -0.5
        grid = (image_size // patch_size) ** 2
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(grid + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.Sequential(  # type: ignore[assignment]
            *[ResidualAttentionBlock(width, heads) for _ in range(layers)])
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    @classmethod
    def vit_l14(cls) -> CLIPVisualTower:
        """The published ViT-L/14. Suitable as `load_model`'s model_factory."""
        return cls(**VIT_L14)  # type: ignore[arg-type]

    @classmethod
    def tiny(cls) -> CLIPVisualTower:
        """A 32px, 2-layer tower for tests. Same code path, no weight file."""
        return cls(image_size=32, patch_size=16, width=32, layers=2,
                   heads=2, output_dim=8)

    def layernorm_parameter_names(self) -> list[str]:
        """Sorted state_dict keys of every LayerNorm weight and bias.

        Phase 2 tunes exactly these and nothing else. Derived from the module
        tree rather than from a hardcoded list, so a layer-count change cannot
        leave the set stale.
        """
        return sorted(
            f"{name}.{suffix}"
            for name, module in self.named_modules()
            if isinstance(module, nn.LayerNorm)
            for suffix in ("weight", "bias"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)                                  # (B, W, g, g)
        x = x.flatten(2).permute(0, 2, 1)                  # (B, g*g, W)
        cls = self.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1) + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x).permute(1, 0, 2)                # LND, as CLIP does
        x = self.transformer.resblocks(x)
        x = x.permute(1, 0, 2)                             # NLD
        return self.ln_post(x[:, 0]) @ self.proj


def preprocess(crop: npt.NDArray[np.uint8], size: int = 224) -> torch.Tensor:
    """RGB HWC uint8 face crop -> `(1, 3, size, size)` normalised float32.

    Bicubic, because that is what CLIP's own preprocessing uses, and the
    encoder is sensitive to the resampling kernel in a way the handcrafted
    slots are not. No centre crop: the input is already a face crop, and
    cropping it again would cut the jawline this detector needs to see.
    """
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)
    x = resized.astype(np.float32) / 255.0
    mean = np.asarray(CLIP_MEAN, dtype=np.float32)
    std = np.asarray(CLIP_STD, dtype=np.float32)
    x = (x - mean) / std
    return torch.from_numpy(x).permute(2, 0, 1)[None].contiguous()
```

- [x] **Step 4: Run the tests**

Run: `python3 -m pytest tests/test_clip_vit.py -q`
Expected: 8 passed.

- [x] **Step 5: Run the gates**

Run: `ruff check src bench corpora training && mypy --config-file mypy.ini && python3 -m pytest -q`
Expected: clean. `mypy --strict` will complain about `self.transformer.resblocks` assignment on a
bare `nn.Module`; if the `type: ignore[assignment]` above is not the right code, use whatever mypy
names — do not widen to a bare `type: ignore`.

- [x] **Step 6: Commit**

```bash
git add src/dfd/detectors/clip_vit.py tests/test_clip_vit.py
git commit -m "feat: CLIP's vision tower, owned here so it can load securely"
```

**One deviation from the code above, applied 2026-11-10.** Step 5 predicted mypy would object to
`self.transformer.resblocks` and told the executor to use whatever error code mypy names. It names
a different problem than the plan expected: the `# type: ignore[assignment]` is reported *unused*,
and the error lands on the **call** instead — `"Tensor" not callable`, because
`nn.Module.__getattr__` reads an attribute assigned onto a bare `nn.Module` back as
`Tensor | Module`. Silencing that at the call site would have put an ignore on the hot path of
every forward pass. Instead a four-line `_Transformer(nn.Module)` holder declares
`resblocks: nn.Sequential`, which produces **the identical `transformer.resblocks.*` state_dict
key path** — the thing Task 2's converter depends on, and what
`test_state_dict_keys_mirror_openai_clip_visual_names` pins. Three further `no-any-return` errors
(the MLP call, the `@ proj` read-out, the block-stack call) are fixed with annotated locals, which
is the idiom `dfd/detectors/npr.py:302` already uses. No ignore comment survives in the file.

**And a trap for the next task.** `mypy --config-file mypy.ini` run by hand reported *Success*
while `tests/test_ci_gates.py::test_mypy_strict_actually_passes` failed on the same tree — the
hand run was serving a stale incremental cache. The test shells out fresh and is the authority.
Run `rm -rf .mypy_cache` before believing a hand-run clean.

**Measured after Step 6:** 1,001 tests pass (993 before this task, +8), coverage 93.03% against the
85% gate, `ruff check src bench corpora training` clean, `mypy --config-file mypy.ini` clean from a
cold cache. The asset gate fails locally on `npr_capture_head` — that is its designed fail-closed
behaviour on a box that has the gitignored weight files, it is identical with this task's files
stashed, and CI discovers zero assets and passes.

---

### Task 2: Convert and pin the encoder, and register it — **DONE 2026-11-10** (`1efbfbf`)

OpenAI distributes ViT-L/14 as a TorchScript archive at a URL whose penultimate path segment **is**
the file's SHA-256 — their own `clip/clip.py` verifies exactly that, so the hash is published
rather than something we assert about a file we happened to download. The archive is also the
reference implementation, which makes it both the source of the weights and the oracle for whether
Task 1's tower is right.

Converted output: `assets/models/clip_vit_l14_visual.pt`, a plain `state_dict` of
`CLIPVisualTower.vit_l14()` — float32, ~1.2 GB.

**Files:**
- Create: `training/convert_clip.py`
- Create: `tests/test_convert_clip.py`
- Modify: `ops/fetch-assets.sh`
- Modify: `assets/manifest.yaml`

**Interfaces:**
- Consumes: `CLIPVisualTower.vit_l14`, `CLIPVisualTower.tiny` (Task 1).
- Produces:
  - `def visual_state_dict(full: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]` — strips
    the `visual.` prefix and **raises** if the result is not an exact key-and-shape match for
    `CLIPVisualTower.vit_l14().state_dict()`.
  - `def main(argv: Sequence[str] | None = None) -> int` — `--archive`, `--out`, `--verify`.

- [x] **Step 1: Write the failing tests**

```python
"""The converter's job is to be total. These tests run on `tiny()`."""
from __future__ import annotations

import pytest
import torch

from dfd.detectors.clip_vit import CLIPVisualTower
from training.convert_clip import MissingKeys, UnexpectedKeys, visual_state_dict


def _tiny_full() -> dict[str, torch.Tensor]:
    """A checkpoint shaped like CLIP's: a visual tower under a prefix, plus
    text-tower keys the converter must drop."""
    sd = {f"visual.{k}": v for k, v in CLIPVisualTower.tiny().state_dict().items()}
    sd["token_embedding.weight"] = torch.zeros(49408, 8)
    sd["logit_scale"] = torch.tensor(4.6052)
    return sd


def test_conversion_keeps_every_visual_key_and_drops_the_rest(monkeypatch):
    monkeypatch.setattr("training.convert_clip.TOWER_FACTORY", CLIPVisualTower.tiny)
    out = visual_state_dict(_tiny_full())
    assert set(out) == set(CLIPVisualTower.tiny().state_dict())
    assert not any(k.startswith("visual.") for k in out)
    assert "logit_scale" not in out


def test_converted_weights_load_strictly_into_the_tower(monkeypatch):
    monkeypatch.setattr("training.convert_clip.TOWER_FACTORY", CLIPVisualTower.tiny)
    tower = CLIPVisualTower.tiny()
    tower.load_state_dict(visual_state_dict(_tiny_full()))  # strict=True default


def test_a_missing_visual_key_is_refused_not_defaulted(monkeypatch):
    monkeypatch.setattr("training.convert_clip.TOWER_FACTORY", CLIPVisualTower.tiny)
    full = _tiny_full()
    del full["visual.ln_post.bias"]
    with pytest.raises(MissingKeys, match="ln_post.bias"):
        visual_state_dict(full)


def test_an_extra_visual_key_is_refused_not_ignored(monkeypatch):
    monkeypatch.setattr("training.convert_clip.TOWER_FACTORY", CLIPVisualTower.tiny)
    full = _tiny_full()
    full["visual.transformer.resblocks.7.ln_1.weight"] = torch.zeros(32)
    with pytest.raises(UnexpectedKeys, match="resblocks.7"):
        visual_state_dict(full)


def test_a_shape_mismatch_is_refused(monkeypatch):
    # The failure this catches: a 336px or a ViT-B checkpoint handed to a
    # ViT-L factory. Keys match; shapes do not; load_state_dict's own message
    # arrives 40 tensors deep.
    monkeypatch.setattr("training.convert_clip.TOWER_FACTORY", CLIPVisualTower.tiny)
    full = _tiny_full()
    full["visual.positional_embedding"] = torch.zeros(99, 32)
    with pytest.raises(ValueError, match="positional_embedding"):
        visual_state_dict(full)
```

- [x] **Step 2: Run them and watch them fail**

Run: `python3 -m pytest tests/test_convert_clip.py -q`
Expected: `ModuleNotFoundError: No module named 'training.convert_clip'`.

- [x] **Step 3: Write the converter**

```python
"""OpenAI's distributed CLIP archive -> a plain state_dict this repo can load securely.

Run once per machine, offline:

    ./ops/fetch-assets.sh                       # fetches and hash-verifies the archive
    python3 -m training.convert_clip \\
        --archive assets/models/ViT-L-14.pt \\
        --out assets/models/clip_vit_l14_visual.pt \\
        --verify

WHY A CONVERSION STEP AT ALL. The published file is a TorchScript archive. It cannot be read by
`torch.load(..., weights_only=True)`, which is the only load the deployed service is allowed to
perform (`dfd/detectors/loading.py`). Reading it is therefore a one-time offline act, done here,
in `training/` -- which `src/dfd/` must never import -- against a hash this project did not choose:
OpenAI's own `clip/clip.py` verifies the download against the SHA-256 embedded in its URL, so the
pin in `ops/fetch-assets.sh` is their published digest, not ours.

WHAT `--verify` DOES, and why it is worth the minutes. The archive is also a working reference
implementation. `--verify` runs both it and `dfd.detectors.clip_vit.CLIPVisualTower` on the same
fixed input and refuses to write unless they agree to 1e-4. That is the ONLY check that the
activation, the residual order and the read-out in our tower are right; key coverage proves the
weights landed somewhere, not that they landed in the right place.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from dfd.detectors.clip_vit import CLIPVisualTower

logger = logging.getLogger(__name__)

#: The factory the conversion targets. A module-level name so tests can point
#: it at `tiny()` instead of instantiating 304M parameters per assertion.
TOWER_FACTORY = CLIPVisualTower.vit_l14

VISUAL_PREFIX = "visual."

#: Agreement required between our tower and the reference implementation.
#: Loose enough for float32 reassociation across two different attention
#: implementations, far tighter than any wrong activation or residual order.
VERIFY_ATOL = 1e-4


class MissingKeys(ValueError):
    """The archive has no weights for part of the tower.

    Its own type because the remedy differs from `UnexpectedKeys`: missing
    means the wrong checkpoint (a text-only export, a partial download),
    unexpected means the wrong SIZE of checkpoint.
    """


class UnexpectedKeys(ValueError):
    """The archive carries visual weights this tower has nowhere to put."""


def visual_state_dict(full: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The vision tower's weights, under this repo's names, or an exception.

    Total in both directions by construction: every key of the target tower
    must be present, and every `visual.` key in the archive must be wanted.
    Silently dropping either side is how a converter ships a tower that is
    half random and scores like a weak detector instead of like a bug.
    """
    want = TOWER_FACTORY().state_dict()
    got = {k[len(VISUAL_PREFIX):]: v for k, v in full.items()
           if k.startswith(VISUAL_PREFIX)}

    missing = sorted(set(want) - set(got))
    if missing:
        raise MissingKeys(
            f"the archive has no weights for {len(missing)} tower "
            f"parameter(s): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    extra = sorted(set(got) - set(want))
    if extra:
        raise UnexpectedKeys(
            f"the archive carries {len(extra)} visual parameter(s) this tower "
            f"has nowhere to put: {extra[:5]}{' ...' if len(extra) > 5 else ''}. "
            f"A ViT-L/14 checkpoint is expected; a different depth or input "
            f"resolution lands here.")
    bad = sorted(k for k in want if tuple(got[k].shape) != tuple(want[k].shape))
    if bad:
        detail = ", ".join(f"{k}: archive {tuple(got[k].shape)} vs tower "
                           f"{tuple(want[k].shape)}" for k in bad[:3])
        raise ValueError(f"shape mismatch on {len(bad)} parameter(s): {detail}")
    return {k: got[k].detach().clone().float() for k in want}


def _reference_embedding(archive: Path, x: torch.Tensor) -> torch.Tensor:
    """Run OpenAI's own module on `x`. Offline, hash-pinned, never in src/."""
    model = torch.jit.load(str(archive), map_location="cpu").eval()
    with torch.no_grad():
        return model.encode_image(x).float()


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out", type=Path,
                        default=Path("assets/models/clip_vit_l14_visual.pt"))
    parser.add_argument("--verify", action="store_true",
                        help="compare this repo's tower against the archive's "
                             "own module before writing. Costs two forward "
                             "passes and is the only check that the "
                             "architecture is right")
    args = parser.parse_args(argv)

    archive = torch.jit.load(str(args.archive), map_location="cpu")
    sd = visual_state_dict(archive.state_dict())
    tower = TOWER_FACTORY()
    tower.load_state_dict(sd)
    tower.eval()

    if args.verify:
        g = torch.Generator().manual_seed(0)
        x = torch.randn(2, 3, tower.image_size, tower.image_size, generator=g)
        with torch.no_grad():
            ours = tower(x)
        theirs = _reference_embedding(args.archive, x)
        delta = float((ours - theirs).abs().max())
        if delta > VERIFY_ATOL:
            logger.error(
                "this repo's tower disagrees with the reference by %.3e "
                "(tolerance %.0e). Do NOT ship this: the weights are real "
                "but the architecture reading them is wrong. Suspect the "
                "activation (QuickGELU, not nn.GELU), the pre/post-norm "
                "residual order, or the ln_post/proj read-out.", delta,
                VERIFY_ATOL)
            return 1
        logger.info("verified against the reference implementation: max "
                    "abs difference %.3e over %d embeddings", delta, len(x))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, args.out)
    total = sum(int(t.numel()) for t in sd.values())
    logger.info("wrote %s: %d tensors, %d parameters", args.out, len(sd), total)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

- [x] **Step 4: Run the tests**

Run: `python3 -m pytest tests/test_convert_clip.py -q`
Expected: 5 passed.

- [x] **Step 5: Pin the archive in `ops/fetch-assets.sh`**

Add to the `ASSETS` array, keeping the `id|path|sha256|url` shape. The digest is the URL's
penultimate segment, which is how OpenAI's own client verifies it:

```bash
  "clip_vit_l14_archive|ViT-L-14.pt|b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836|https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
```

And below the array, a comment the next reader needs:

```bash
# clip_vit_l14_archive is ~890MB and is the INPUT to a conversion, not a
# runtime asset: `python3 -m training.convert_clip --archive
# assets/models/ViT-L-14.pt --out assets/models/clip_vit_l14_visual.pt
# --verify` turns it into the plain state_dict the service loads. The
# archive may be deleted afterwards; the converted file may not.
```

- [x] **Step 6: Verify the pin against the real file**

Run:
```bash
./ops/fetch-assets.sh && sha256sum assets/models/ViT-L-14.pt
```
Expected: `OK clip_vit_l14_archive`, and the digest equals the URL segment. **If the URL 404s or
the digest differs, stop and report it — do not substitute an unpinned mirror.** The fallback, if
and only if that happens, is HuggingFace `openai/clip-vit-large-patch14`'s `pytorch_model.bin`
(a plain state_dict, so `torch.load(weights_only=True)` reads it and `--verify` is unavailable),
pinned to a hash measured locally and recorded as measured-here rather than published.

- [x] **Step 7: Convert and verify**

Run:
```bash
python3 -m training.convert_clip --archive assets/models/ViT-L-14.pt \
    --out assets/models/clip_vit_l14_visual.pt --verify
```
Expected: `verified against the reference implementation: max abs difference <1e-4`, then
`wrote ... 303,966,208 parameters` — that figure is what a `CLIPVisualTower.vit_l14()` built from
this plan's Task 1 actually holds, measured, so a different number means a different
architecture and not a rounding difference.
**If verification fails, Task 1 is wrong and must be fixed before anything else proceeds.**

- [x] **Step 8: Register both files in `assets/manifest.yaml`**

```yaml
  clip_vit_l14_visual:
    # The CONVERTED vision tower (training/convert_clip.py), plus the archive
    # it was converted from. One id covers both: they are the same weights in
    # two containers, and splitting them would invite two different licence
    # verdicts on one set of numbers.
    source: "OpenAI CLIP ViT-L/14 — https://github.com/openai/CLIP, weights at https://openaipublic.azureedge.net/clip/models/b8cca.../ViT-L-14.pt (sha256 is the URL segment, verified by ops/fetch-assets.sh)"
    license: "MIT"
    # Read at source: github.com/openai/CLIP/blob/main/LICENSE is MIT, and the
    # model card distributes the weights under that repository's terms.
    # Attribution is the only real obligation and it travels with the product
    # (see the spec's §9.1 conditions). NOT the GenD checkpoint, which is
    # FF++-trained and is never fetched by anything in this repo.
    commercial_use: true
    evidence_url: "https://github.com/openai/CLIP/blob/main/LICENSE"
    date_checked: "2026-11-10"
    checked_by: "rohit.kohli@scoreme.in"
    files:
      - "assets/models/ViT-L-14.pt"
      - "assets/models/clip_vit_l14_visual.pt"
```

- [x] **Step 9: Run the asset gate and the full suite**

Run: `python3 -m pytest -q && python3 -c "from dfd.asset_scan import *; print(discover_assets('.'))"`
Expected: the two new paths appear and the gate is green. (Use whatever invocation
`tests/test_asset_scan.py` uses for the release check.)

- [x] **Step 10: Commit**

```bash
git add training/convert_clip.py tests/test_convert_clip.py ops/fetch-assets.sh assets/manifest.yaml
git commit -m "feat: the encoder is pinned to OpenAI's own digest, and verified against their module"
```

**TASK 1 IS VERIFIED.** `--verify` reports **max abs difference 2.116e-06** between
`CLIPVisualTower.vit_l14()` and OpenAI's own module on the same two inputs, against a 1e-4
tolerance, cosine similarity 1.0 on both. QuickGELU, the pre-norm residual order and the
`ln_post / proj` read-out are right. The written file holds **303,966,208 parameters in 296
tensors** — the exact figure Step 7 predicted, so the architecture is the one the plan costed and
not merely one that runs. `ops/fetch-assets.sh` reports `OK clip_vit_l14_archive` and
`sha256sum` returns `b8cca3fd...a03836`, identical to the URL's penultimate segment. The MIT
licence was fetched and read at `raw.githubusercontent.com/openai/CLIP/main/LICENSE` before
`commercial_use: true` was written, not inferred.

**One deviation, and it was necessary.** Step 3's `_reference_embedding` — `torch.jit.load(...,
map_location="cpu")` then `encode_image` — **cannot work on this box, or on the deployment**.
OpenAI traced the archive on a GPU in float16, so `torch.device("cuda:0")` and `dtype=5` (half)
are frozen into the graph as *constants*, which `map_location` does not touch. The first run
raised `RuntimeError: Found no NVIDIA driver` from inside the TorchScript interpreter; patching
past that alone then raised `Input type (torch.FloatTensor) and weight type (torch.HalfTensor)`
from the first convolution. Routing around it via `model.visual` does not help — that graph
carries its own `cuda:0` constant for the class-token broadcast.

The fix is the one OpenAI's own `clip/clip.py` applies on a CPU load, and is now
`load_reference`: trace a one-line holder to obtain a CPU device constant and a float32 dtype
constant, then `copyAttributes` them over every `prim::Constant` whose value starts with `cuda`
and every `aten::to` whose dtype argument is 5. It rewrites the graph and never the weights.
`main` now reads the 890 MB archive **once** and uses the same object for both the state_dict and
the oracle, rather than loading it twice.

**Two tests were added beyond the plan's five** (`test_convert_clip.py`, 7 total). The float patch
is asserted on the **graph constant**, not on an output dtype: a toy trace carries a `Half(...)`
annotation on its output value that the interpreter honours regardless of the constant, so a dtype
assertion passes on the real archive and fails on a toy. The device patch cannot be exercised at
all without a CUDA box — a `cuda` constant is untraceable here — so its test asserts only that the
walk is safe on a graph with nothing to patch, and says so in its own comment. `--verify` against
the real file is the only thing that proves either rewrite, and it has now run.

**Also changed, unlisted:** `ops/fetch-assets.sh`'s closing line said "both are permissive and
commercially cleared" when two assets were pinned. Three are now. It says "all".

**Measured after Step 10:** 1,008 tests pass (1,001 before this task, +7), coverage 93.03%, ruff
clean, `mypy --config-file mypy.ini` clean from a cold cache. The asset gate discovers both new
paths, registers and clears both, and still fails locally on `npr_capture_head` alone — unchanged,
pre-existing, and by that entry's own design.

---

### Task 3: Slot D, and the registry it joins — **DONE 2026-11-10** (`c5287f6`)

**Files:**
- Create: `src/dfd/detectors/gend.py`
- Create: `tests/test_gend.py`
- Modify: `src/dfd/detectors/registry.py`
- Modify: `tests/test_registry.py:303-337`
- Modify: `src/dfd/service/app.py:46-50`

**Interfaces:**
- Consumes: `CLIPVisualTower`, `preprocess`, `PREPROCESS_VERSION` (Task 1);
  `dfd.detectors.loading.load_model`; `dfd.detectors.base.{abstain, filter_by_quality_floor,
  NO_OBSERVATIONS, WEIGHTS_ABSENT, OK}`.
- Produces:
  - `class GendHead(nn.Module)` — `linear: nn.Linear(768, 2)` plus a fixed-width
    `version_bytes` buffer. `__init__(self, in_dim: int = 768)`.
    `GendHead.version(self) -> str`, `GendHead.set_version(self, v: str) -> None`.
  - `VERSION_BYTES: int = 32`
  - `class GendClipDetector` — frozen dataclass, fields
    `weights_path: str | Path`, `encoder_path: str | Path`, `name: str = "gend_clip"`,
    `slot: str = "D"`, `modalities`, `min_quality_band`.
  - `def embed(tower: nn.Module, crops: Sequence[npt.NDArray[np.uint8]]) -> torch.Tensor` —
    `(N, 768)` L2-normalised. Shared with `training/clip_features.py` so the fitter and the
    detector cannot drift apart.
  - `NO_ROI: str = "roi_absent"`, `ENCODER_ABSENT: str = "encoder_absent"`

- [x] **Step 1: Write the failing tests**

```python
"""Slot D. Every test uses a tiny tower; none needs the 1.2 GB encoder."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from dfd.detectors.clip_vit import CLIPVisualTower
from dfd.detectors.gend import (
    ENCODER_ABSENT,
    NO_ROI,
    VERSION_BYTES,
    GendClipDetector,
    GendHead,
    embed,
)
from dfd.quality import measure_quality
from dfd.types import Modality, Observation


def _obs(value: int = 128, size: int = 64) -> Observation:
    img = np.full((size, size, 3), value, dtype=np.uint8)
    lm = np.array([[size * 0.35, size * 0.4], [size * 0.65, size * 0.4]])
    return Observation(t=0.0, payload=img, roi=(0, 0, size, size),
                       quality=measure_quality(img, (0, 0, size, size), lm),
                       source_id="s1")


def _fitted(tmp_path, in_dim: int = 8, version: str = "0.1.0-test"):
    head = GendHead(in_dim=in_dim)
    head.set_version(version)
    path = tmp_path / "gend_clip.pt"
    torch.save(head.state_dict(), path)
    return path


def _encoder(tmp_path):
    path = tmp_path / "clip.pt"
    torch.save(CLIPVisualTower.tiny().state_dict(), path)
    return path


def test_embeddings_are_l2_normalised():
    tower = CLIPVisualTower.tiny().eval()
    crops = [np.full((40, 40, 3), v, dtype=np.uint8) for v in (10, 200)]
    out = embed(tower, crops)
    assert out.shape == (2, 8)
    assert torch.allclose(out.norm(dim=1), torch.ones(2), atol=1e-5)


def test_the_version_travels_inside_the_state_dict(tmp_path):
    head = GendHead(in_dim=8)
    head.set_version("0.1.0-probe-fairface60000")
    reloaded = GendHead(in_dim=8)
    reloaded.load_state_dict(head.state_dict())  # strict
    assert reloaded.version() == "0.1.0-probe-fairface60000"


def test_a_version_longer_than_the_buffer_is_refused():
    # Silently truncating would produce a weights file whose declared
    # training window is a PREFIX of the real one, which is exactly how the
    # eval_swaps offset guard gets bypassed.
    head = GendHead(in_dim=8)
    with pytest.raises(ValueError, match="longer than"):
        head.set_version("x" * (VERSION_BYTES + 1))


def test_abstains_when_the_head_is_absent(tmp_path):
    d = GendClipDetector(weights_path=tmp_path / "nope.pt",
                         encoder_path=_encoder(tmp_path))
    raw = d.score([_obs()])
    assert raw.abstained and raw.reason == "weights_absent" and raw.score is None


def test_abstains_when_the_encoder_is_absent_and_says_which(tmp_path):
    # Distinct from weights_absent: "fit the head" and "run
    # ops/fetch-assets.sh" are different remedies, and one reason for both
    # sends every operator down the wrong one.
    d = GendClipDetector(weights_path=_fitted(tmp_path),
                         encoder_path=tmp_path / "nope.pt")
    raw = d.score([_obs()])
    assert raw.abstained and raw.reason == ENCODER_ABSENT


def test_abstains_on_no_observations(tmp_path):
    d = GendClipDetector(weights_path=_fitted(tmp_path),
                         encoder_path=_encoder(tmp_path))
    assert d.score([]).reason == "no_observations"


def test_abstains_rather_than_scoring_the_whole_frame_without_a_roi(tmp_path):
    d = GendClipDetector(weights_path=_fitted(tmp_path),
                         encoder_path=_encoder(tmp_path))
    obs = _obs()
    raw = d.score([Observation(t=0.0, payload=obs.payload, roi=None,
                               quality=obs.quality, source_id="s1")])
    assert raw.abstained and raw.reason == NO_ROI


def test_scores_in_the_unit_interval_and_records_its_denominator(tmp_path):
    d = GendClipDetector(weights_path=_fitted(tmp_path),
                         encoder_path=_encoder(tmp_path))
    raw = d.score([_obs(100), _obs(150)])
    assert not raw.abstained
    assert 0.0 <= raw.score <= 1.0
    assert raw.artifacts["n_observations"] == 2
    assert len(raw.artifacts["per_observation_scores"]) == 2
    assert "max_score" in raw.artifacts


def test_version_is_read_from_the_weights_file_not_hardcoded(tmp_path):
    d = GendClipDetector(weights_path=_fitted(tmp_path, version="9.9.9-custom"),
                         encoder_path=_encoder(tmp_path))
    assert d.version == "9.9.9-custom"


def test_version_falls_back_to_the_code_version_with_no_weights(tmp_path):
    d = GendClipDetector(weights_path=tmp_path / "nope.pt",
                         encoder_path=tmp_path / "nope.pt")
    assert d.version  # a detector that cannot say what it is, is unauditable
    assert d.score([_obs()]).version == d.version


def test_the_detector_is_frozen_so_the_registry_identity_holds(tmp_path):
    d = GendClipDetector(weights_path=_fitted(tmp_path),
                         encoder_path=_encoder(tmp_path))
    with pytest.raises(Exception):
        d.name = "other"  # type: ignore[misc]


def test_declares_image_and_video_modalities(tmp_path):
    d = GendClipDetector(weights_path=tmp_path / "x.pt",
                         encoder_path=tmp_path / "y.pt")
    assert Modality.IMAGE in d.modalities
```

Add to `tests/test_registry.py` (and update the two existing three-detector tests to four):

```python
def test_default_registry_now_composes_four_distinct_physics() -> None:
    reg = default_registry()
    assert reg.names() == ["blend_seam", "effnet_b4", "gend_clip", "npr"]
    assert sorted(reg.get(n).slot for n in reg.names()) == ["A", "C", "D", "E"]


def test_gend_clip_abstains_when_its_weights_are_absent(tmp_path) -> None:
    reg = default_registry(gend_weights=tmp_path / "absent.pt",
                           clip_weights=tmp_path / "absent-encoder.pt")
    raw = reg.get("gend_clip").score([])
    assert raw.abstained
```

- [x] **Step 2: Run them and watch them fail**

Run: `python3 -m pytest tests/test_gend.py tests/test_registry.py -q`
Expected: `ModuleNotFoundError: No module named 'dfd.detectors.gend'`, and the two registry tests
fail on the three-detector assertions.

- [x] **Step 3: Write the detector**

```python
"""Slot D — learned appearance read from a general-purpose encoder (spec §5.1).

Physics: not a handcrafted artefact at all. CLIP's image embedding is a general visual
representation, and the claim GenD (WACV 2026) makes -- 0.912-0.916 mean AUROC over 14 datasets
-- is that a LINEAR read-out of it separates generated faces from photographed ones better than
any single artefact detector generalises. That is a different failure mode from slots A, C and E,
which is the whole point of giving it its own slot letter.

WHAT IS OURS AND WHAT IS NOT. The encoder is OpenAI's CLIP ViT-L/14 under MIT
(`assets/manifest.yaml: clip_vit_l14_visual`). Everything fitted -- the head here, and in Phase 2
the LayerNorm parameters -- is fitted in this repo on FairFace-derived material and inherits CC BY
4.0. GenD's own released weights are trained on FaceForensics++ and are NEVER fetched, loaded or
initialised from: non-commercial research terms make a model fitted on them derived data, which is
the same bar that disqualified `sbi_effnetb4_weights` in this repo's manifest.

TWO WEIGHT FILES, TWO ABSENCES. The encoder and the head are separate assets with separate
licences and separate remedies, so they abstain with separate reasons.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from ..types import Modality, Observation, RawScore
from .base import (
    NO_OBSERVATIONS,
    OK,
    WEIGHTS_ABSENT,
    abstain,
    filter_by_quality_floor,
)
from .clip_vit import CLIPVisualTower, preprocess
from .loading import load_model

logger = logging.getLogger(__name__)

#: The embedding width `GendHead` reads. CLIP ViT-L/14's projected output.
EMBED_DIM = 768

#: Fixed width of the version buffer. Fixed, because `load_state_dict` is
#: strict about buffer shapes and a variable-length one could not be loaded
#: at all. 32 bytes holds "0.1.0-probe-fairface60000" with room.
VERSION_BYTES = 32

#: The observation carried no usable ROI. Mirrors `blend.NO_ROI` and
#: `npr.NO_ROI` deliberately -- one condition, one name across the codebase.
NO_ROI = "roi_absent"

#: The HEAD is present but the encoder file is not. Distinct from
#: WEIGHTS_ABSENT because the remedies differ: fit the head, versus run
#: ops/fetch-assets.sh and training/convert_clip.py.
ENCODER_ABSENT = "encoder_absent"

#: Reported when there is no weights file to read a version from. Describes
#: the CODE -- the preprocessing and the read-out -- never the coefficients.
CODE_VERSION = "0.1.0-unfitted"


class GendHead(nn.Module):
    """Everything this project fits in Phase 1: one linear read-out.

    `Linear(768, 2)` over an L2-normalised CLIP embedding, which is 1,538
    numbers. The encoder stays frozen and is not part of this file, so the
    shipped artefact's licence is FairFace's and the encoder's stays MIT.

    THE VERSION IS A BUFFER, NOT METADATA BESIDE THE FILE. `bench/eval_swaps.py`
    refuses a corpus drawn from a detector's own training window, and it keys
    that refusal on the detector's VERSION string. A version living in a
    sidecar file is a version that can be separated from the coefficients it
    describes, and the first thing anyone does with a weights file is copy it
    somewhere else.
    """

    version_bytes: torch.Tensor

    def __init__(self, in_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, 2)
        self.register_buffer("version_bytes",
                             torch.zeros(VERSION_BYTES, dtype=torch.uint8))

    def set_version(self, value: str) -> None:
        """Record the fitted version. Refuses to truncate.

        Raises:
            ValueError: if the encoded version does not fit. Truncating would
                leave a file declaring a training window that is a prefix of
                the real one -- which reads as a smaller window and lets a
                contaminated evaluation corpus past the offset guard.
        """
        raw = value.encode("utf-8")
        if len(raw) > VERSION_BYTES:
            raise ValueError(
                f"version {value!r} is {len(raw)} bytes, longer than the "
                f"{VERSION_BYTES}-byte buffer; it would be truncated")
        buf = torch.zeros(VERSION_BYTES, dtype=torch.uint8)
        buf[:len(raw)] = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        self.version_bytes = buf

    def version(self) -> str:
        """The recorded version, or "" when none was set."""
        return bytes(self.version_bytes.tolist()).rstrip(b"\x00").decode("utf-8")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def embed(tower: nn.Module, crops: Sequence[npt.NDArray[np.uint8]]) -> torch.Tensor:
    """L2-normalised embeddings of face crops, `(N, EMBED_DIM)`.

    Shared by the detector and `training/clip_features.py` on purpose: a
    fitter that normalised and a detector that did not would produce a model
    that loads clean, scores confidently and is wrong -- the defect
    `NPRStatsNet` already carries a comment about.
    """
    batch = torch.cat([preprocess(c, size=tower.image_size) for c in crops])
    with torch.no_grad():
        out = tower(batch)
    return out / out.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _crop_to_roi(payload: npt.NDArray[np.uint8],
                 roi: tuple[int, int, int, int] | None,
                 ) -> npt.NDArray[np.uint8] | None:
    """Clamp to the face box, or None.

    No resize here -- `preprocess` does it bicubically at the encoder's own
    input size. Unlike slot C, resampling is not destructive for this slot
    (the encoder resizes to 224 regardless), but it must happen exactly once
    and with CLIP's kernel.
    """
    if roi is None:
        return None
    h, w = payload.shape[:2]
    x, y, rw, rh = roi
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(w, int(x) + int(rw)), min(h, int(y) + int(rh))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return np.ascontiguousarray(payload[y0:y1, x0:x1])


@lru_cache(maxsize=8)
def _version_of(path: str, fingerprint: tuple[int, int]) -> str:
    """The version recorded inside the head at `path`, or CODE_VERSION.

    `fingerprint` is (mtime_ns, size) and is never read: it is in the cache
    key so refitting invalidates the entry. Caching on the path alone would
    pin the first version ever read for the life of the process, and the
    fitter writes to a FIXED path -- so stale would be the normal case.
    Never raises: this is read on the abstention path too.
    """
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        buf = sd["version_bytes"]
        return bytes(buf.tolist()).rstrip(b"\x00").decode("utf-8") or CODE_VERSION
    except Exception:
        logger.debug("cannot read a version from %s", path, exc_info=True)
        return CODE_VERSION


@dataclass(frozen=True)
class GendClipDetector:
    """Slot D. Frozen, so the registry's name -> detector map cannot go stale."""

    weights_path: str | Path
    encoder_path: str | Path
    name: str = "gend_clip"
    slot: str = "D"
    modalities: frozenset[Modality] = field(
        default_factory=lambda: frozenset({Modality.IMAGE, Modality.VIDEO}))
    #: PROVISIONAL, and it must not stay that way. Spec §5.1 requires this
    #: floor to be MEASURED, as slot C's "reject" was: the per-band table on
    #: the capture corpus (Task 8) decides it. "low" is the starting value
    #: because an appearance read-out has no reason to want a blurred face --
    #: but slot C had no reason either, and the measurement said otherwise.
    min_quality_band: Literal["reject", "low", "medium", "high"] = "low"

    @property
    def version(self) -> str:
        p = Path(self.weights_path)
        try:
            st = p.stat()
        except OSError:
            return CODE_VERSION
        return _version_of(str(p), (st.st_mtime_ns, st.st_size))

    def score(self, obs: Sequence[Observation]) -> RawScore:
        """Embed every usable face crop, read the head, return mean P(fake).

        Mean is the official score and max rides in artifacts, matching
        `NPRDetector`: a reader comparing two slots on one record must not
        have to discover they aggregate differently.
        """
        if not obs:
            return abstain(self.name, self.version, NO_OBSERVATIONS)
        if not Path(self.weights_path).exists():
            return abstain(self.name, self.version, WEIGHTS_ABSENT)
        if not Path(self.encoder_path).exists():
            logger.debug("encoder missing at %s; abstaining", self.encoder_path)
            return abstain(self.name, self.version, ENCODER_ABSENT)

        usable, reason = filter_by_quality_floor(obs, self.min_quality_band)
        if reason is not None:
            return abstain(self.name, self.version, reason)

        maybe = [_crop_to_roi(o.payload, o.roi) for o in usable]
        if any(c is None for c in maybe):
            return abstain(self.name, self.version, NO_ROI)
        crops = [c for c in maybe if c is not None]

        tower = load_model(self.encoder_path,
                           model_factory=CLIPVisualTower.vit_l14)
        head = load_model(self.weights_path, model_factory=GendHead)
        with torch.no_grad():
            probs = torch.softmax(head(embed(tower, crops)), dim=1)[:, 1]

        per_obs = probs.cpu().numpy().tolist()
        return RawScore(
            detector=self.name, version=self.version,
            score=float(probs.mean()), abstained=False, reason=OK,
            artifacts={"n_observations": len(usable),
                       "per_observation_scores": per_obs,
                       "max_score": float(probs.max())})
```

- [x] **Step 4: Wire it into `src/dfd/detectors/registry.py`**

```python
DEFAULT_GEND_WEIGHTS = Path("assets/models/gend_clip.pt")
DEFAULT_CLIP_WEIGHTS = Path("assets/models/clip_vit_l14_visual.pt")
```

Add the two keyword arguments to `default_registry` and, after the three existing registrations:

```python
    # Slot D. TWO paths because the encoder (MIT, 304M parameters, fetched
    # and hash-pinned) and the head (fitted here, 1,538 numbers, CC BY 4.0 by
    # inheritance) are different assets with different licences, and the
    # manifest records them separately.
    registry.register(GendClipDetector(weights_path=gend_weights,
                                       encoder_path=clip_weights))
```

Update the docstring: "Three distinct physics" becomes four, and name slot D's.

- [x] **Step 5: Add the paths to `ServiceConfig` in `src/dfd/service/app.py`**

```python
    gend_weights: Path = Path(DEFAULT_GEND_WEIGHTS)
    clip_weights: Path = Path(DEFAULT_CLIP_WEIGHTS)
```

and pass them through wherever `default_registry` is called there.

- [x] **Step 6: Run the tests**

Run: `python3 -m pytest tests/test_gend.py tests/test_registry.py tests/service -q`
Expected: all pass.

- [x] **Step 7: Run the gates**

Run: `ruff check src bench corpora training && mypy --config-file mypy.ini && python3 -m pytest -q`
Expected: clean, test count up by ~14.

- [x] **Step 8: Commit**

```bash
git add src/dfd/detectors/gend.py src/dfd/detectors/registry.py src/dfd/service/app.py \
        tests/test_gend.py tests/test_registry.py
git commit -m "feat: slot D exists, abstains honestly, and says which of its two files is missing"
```

**Three things the plan's file list missed, all of them gates doing their job.** Step 1 named
`tests/test_registry.py` as the only test to update. Three others assert the registry's
composition and all three went red on a four-detector registry:

1. `tests/service/test_app.py::test_the_registry_carries_the_three_declared_slots` — renamed to
   `..._four_...` and updated.
2. `tests/test_cli.py::test_the_record_carries_one_evidence_row_per_default_detector` — the CLI
   emits one evidence row per registered detector, so four rows now.
3. `tests/service/test_evidence.py::test_the_committed_card_covers_every_detector_in_the_registry`
   — **the important one**. It refuses a detector that is in the registry and absent from
   `bench/evidence_card.json`, which is exactly slot D's position. A `gend_clip` entry was added
   in the shape `effnet_b4` and `npr` already use for an unmeasured detector: every numeric field
   `null`, and a note recording that the encoder is present and verified, that the head is not
   fitted, that the detector therefore abstains with `weights_absent`, and that **Task 7** is what
   fills this in. `test_no_detector_currently_clears_the_floor` still passes, which is the
   assertion that slot D has not quietly started deciding anything.

**Two deviations in the code.**

*The test fixture could not reach the code it was testing.* Step 1's `_obs` built a flat grey
patch and derived its quality with `measure_quality`, which returns `blur_var=0.0` and band
`reject` on a constant image. Against slot D's `min_quality_band="low"` that abstains with
`below_quality_floor` before any crop, so two tests — the ROI refusal and the only scoring test —
asserted nothing about slot D. The fixture now STATES its `Quality`, which is the idiom
`tests/test_npr.py` already uses for exactly this reason. The detector's floor was not weakened to
make a test pass; it is still the provisional `"low"` that Task 8 must measure.

*The detector could not be constructed against a tiny tower.* Step 3 hardcoded
`model_factory=CLIPVisualTower.vit_l14` and `GendHead`, while Step 1's fixtures write a `tiny()`
encoder and an 8-dim head — so the one test that loads both files failed on 296 shape mismatches.
`encoder_factory` and `head_factory` are now dataclass fields defaulting to the shipped pair,
which is what `NPRDetector.model_factory` already does and what `loading.load_model`'s secure path
wants anyway. Phase 2's tuned encoder gets a seam to enter through at no extra cost.

Also: `embed` reads `tower.image_size` through `getattr(..., 224)` rather than attribute access,
since its parameter is typed `nn.Module`; and two `no-any-return` errors needed annotated locals,
the same idiom as Tasks 1 and 2.

**Measured after Step 8:** 1,021 tests pass (1,008 before this task, +13 — twelve in
`test_gend.py` and one new registry test), coverage 93.15%, ruff clean, mypy clean from a cold
cache. `default_registry().names()` is now
`["blend_seam", "effnet_b4", "gend_clip", "npr"]` across slots A, C, D, E, and slot D abstains
with `weights_absent` on this machine because the head does not exist yet.

---

### Task 4: The feature cache, which is what makes the measurement affordable — **DONE 2026-11-11** (`2cc88c8`)

One ViT-L/14 forward pass is ~81 GFLOPs. Every number spec §7 demands — a permutation null (12
refits), a grouped bootstrap (1000 resamples), a per-fold-refit LOGO (6 folds) — would re-run the
encoder if the embeddings were not cached. Cached, each of those is a logistic regression over a
matrix that is already in memory, and the whole validation suite costs one pass over the corpus.

**Files:**
- Create: `training/clip_features.py`
- Create: `tests/test_clip_features.py`

**Interfaces:**
- Consumes: `embed` (Task 3), `PREPROCESS_VERSION` (Task 1).
- Produces:
  - `class FeatureCache` — `__init__(self, root: str | Path, *, encoder_id: str)`.
    - `def key(self, image: npt.NDArray[np.uint8]) -> str` — sha256 of the contiguous bytes, the
      shape, `PREPROCESS_VERSION` and `encoder_id`.
    - `def get(self, image) -> npt.NDArray[np.float32] | None`
    - `def put(self, image, vector: npt.NDArray[np.float32]) -> None`
    - `def embed_all(self, tower, images: Sequence[...], *, batch_size: int = 8) -> npt.NDArray[np.float32]`
      — `(N, 768)`, cached rows served from disk, the rest computed and stored.
    - `stats: dict[str, int]` — `{"hits": .., "misses": ..}`, reported by every caller.

- [x] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import numpy as np

from dfd.detectors.clip_vit import CLIPVisualTower
from training.clip_features import FeatureCache


def _img(v: int) -> np.ndarray:
    return np.full((40, 40, 3), v, dtype=np.uint8)


def test_second_pass_is_all_hits(tmp_path):
    tower = CLIPVisualTower.tiny().eval()
    cache = FeatureCache(tmp_path, encoder_id="tiny")
    images = [_img(10), _img(200), _img(90)]
    first = cache.embed_all(tower, images)
    assert cache.stats == {"hits": 0, "misses": 3}

    again = FeatureCache(tmp_path, encoder_id="tiny")
    second = again.embed_all(tower, images)
    assert again.stats == {"hits": 3, "misses": 0}
    assert np.array_equal(first, second)


def test_identical_pixels_share_one_entry(tmp_path):
    # 979 copies of one demo asset are already known to live in this
    # project's capture corpus. Hashing content, not paths, means they cost
    # one forward pass -- and means the cache cannot be fooled into serving
    # one image's embedding for another's.
    cache = FeatureCache(tmp_path, encoder_id="tiny")
    assert cache.key(_img(50)) == cache.key(_img(50))
    assert cache.key(_img(50)) != cache.key(_img(51))


def test_a_different_encoder_id_is_a_different_cache(tmp_path):
    a = FeatureCache(tmp_path, encoder_id="clip-l14")
    b = FeatureCache(tmp_path, encoder_id="clip-l14-lntuned")
    assert a.key(_img(7)) != b.key(_img(7))


def test_a_preprocessing_change_invalidates_every_entry(tmp_path, monkeypatch):
    before = FeatureCache(tmp_path, encoder_id="tiny").key(_img(7))
    monkeypatch.setattr("training.clip_features.PREPROCESS_VERSION", "changed")
    after = FeatureCache(tmp_path, encoder_id="tiny").key(_img(7))
    assert before != after


def test_batching_does_not_change_the_vectors(tmp_path):
    tower = CLIPVisualTower.tiny().eval()
    images = [_img(v) for v in range(10, 100, 10)]
    one = FeatureCache(tmp_path / "a", encoder_id="t").embed_all(
        tower, images, batch_size=1)
    many = FeatureCache(tmp_path / "b", encoder_id="t").embed_all(
        tower, images, batch_size=4)
    assert np.allclose(one, many, atol=1e-5)
```

- [x] **Step 2: Run them and watch them fail**

Run: `python3 -m pytest tests/test_clip_features.py -q`
Expected: `ModuleNotFoundError: No module named 'training.clip_features'`.

- [x] **Step 3: Write the cache**

Write `training/clip_features.py` implementing exactly the interface above. Required properties,
each of which a test above pins:

- the key is `sha256(contiguous bytes + repr(shape) + PREPROCESS_VERSION + encoder_id)`, so content
  addresses the entry and both the preprocessing and the encoder are part of the address;
- one `.npy` per key under `root/<first two hex chars>/<key>.npy`, so a corpus of 200k embeddings
  does not put 200k entries in one directory;
- `embed_all` preserves input order, calls `embed()` (Task 3) in batches for misses only, and
  writes each miss as it goes, so an interrupted run resumes;
- `stats` counts hits and misses and is logged at INFO by every caller.

Module docstring must say why this exists (the arithmetic at the head of this task) and that the
cache directory belongs **outside** the repo — `~/Desktop/agents/datasets/clip_cache/`, like every
other corpus here — because a few hundred thousand `.npy` files inside a git worktree is a mistake
nobody undoes.

- [x] **Step 4: Run the tests**

Run: `python3 -m pytest tests/test_clip_features.py -q`
Expected: 5 passed.

- [x] **Step 5: Commit**

```bash
git add training/clip_features.py tests/test_clip_features.py
git commit -m "feat: embed once, measure many times"
```

**The module as the plan specified it did not do the thing its own docstring claims.** Step 3
requires the docstring to say that content addressing is why 979 byte-identical copies cost one
forward pass. With `embed_all` written as a flat list of pending indices, that is only true
ACROSS calls: inside one call all 979 are misses on disk, so all 979 are embedded, and the cache
helps on the second run only. `pending` is now a `dict[key, list[int]]`, one forward pass per
DISTINCT image, copied to every position that asked for it.

**And duplicates were not even returning equal rows.** Two byte-identical images embedded in the
same batch disagreed by ~6e-08 depending on where in the batch each sat — a batched matmul
reassociates differently per row position. Harmless as a number, but it meant the on-disk entry
for a repeated image was whichever copy happened to be written last, so a first pass and a second
pass over the same corpus returned slightly different matrices. Deduplicating removes the
ambiguity: one vector per distinct image, written once, identical everywhere it appears.

**Two tests added beyond the plan's five** (7 total), covering the two ways this module can
silently corrupt a measurement rather than fail:

- `test_a_half_cached_list_keeps_its_order` — misses are computed in batches and written back by
  index. A version that appended results instead would return the computed rows followed by the
  cached ones. Every caller zips this matrix against a label vector, and a matrix permuted against
  its labels fits beautifully and means nothing.
- `test_duplicate_images_each_get_their_own_row` — one cache ENTRY for 979 copies is the point;
  one ROW would be a bug, because the caller's labels have an entry per observation.

`embed_all` also now refuses to return a matrix shorter than its input rather than filtering the
unfilled rows away, for the same reason.

**One clarification, recorded because the number will be read:** `stats["misses"]` counts INPUTS
not served from disk, not forward passes. A call holding 979 copies reports 979 misses and runs
the encoder once. The plan's tests assert `stats` by exact dict equality, so a third counter could
not be added without rewriting them; the docstring says it instead.

**Measured after Step 5:** 1,028 tests pass (1,021 before this task, +7), coverage 93.15%, ruff
clean, `mypy --config-file mypy.ini` clean from a cold cache.

---

### Task 5: Fit the head — **DONE 2026-11-11** (`5c2cbb0`)

**The corpus, concretely.** Reals and fakes both come from the FairFace session directory that
`training/export_fairface.py` writes (one folder per photograph, `frame_00.jpg` plus a
`results.json` carrying dataset, licence, URL and demographics). Two builders already exist and
both are used:

| Family | Builder | Count per photograph |
|---|---|---|
| `real` | `corpora.face_pool.build_face_pool` over `corpora.captures.load_capture_sessions` | 1 |
| `sbi` | `corpora.sbi.build_sbi_corpus` | 1 fake |
| `swap_warp_hull`, `swap_poisson`, `swap_mouth_patch`, `swap_lowres_paste` | `corpora.swap_corpus.build_swap_corpus` | 1 fake each per couple |

That is **five generator families**, all licence-clean, which is what lets LOGO fold five ways in
Task 7. SFHQ is deliberately **not** in the training mix: it saturates (measured — 500 images
reach the same number as 204,716) and it cannot be paired, so it is a held-out control, not a
training family.

**Pairing and re-encoding are not optional** (spec §6.3). Every fake is made from a specific real
crop, and both halves pass through the identical encode. One JPEG header byte labels 99.2% of the
capture corpus; a detector handed reals at one encode and fakes at another learns the encoder.

**Files:**
- Create: `training/fit_gend.py`
- Create: `tests/test_fit_gend.py`
- Modify: `bench/eval_swaps.py` (`TRAINING_WINDOWS`)

**Interfaces:**
- Consumes: `FeatureCache` (Task 4), `GendHead`, `embed` (Task 3), `CLIPVisualTower`,
  `build_face_pool`, `build_sbi_corpus`, `build_swap_corpus`, `load_capture_sessions`.
- Produces:
  - `def build_training_records(root, *, limit, offset, seed, detect) -> tuple[list[dict], dict[str, int]]`
    — bench-runner-shaped records (`sample_id`, `source_id`, `subject_id`, `stratum`, `generator`,
    `label`, `compression`, `face_detector`, `align`, `image`), reals and all five fake families,
    every record re-encoded identically.
  - `REENCODE_QUALITY: int = 95` and `def reencode(img) -> npt.NDArray[np.uint8]`
  - `def fit_head(features, labels, *, version: str) -> GendHead`
  - `def main(argv=None, *, detect=detect_faces) -> int` — flags `--root`, `--limit`, `--offset`,
    `--cache`, `--encoder`, `--out`, `--report`, `--seed`, `--holdout`.

- [x] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import numpy as np
import pytest
import torch

from dfd.detectors.gend import GendHead
from training.fit_gend import build_training_records, fit_head, reencode


def test_every_record_passes_through_the_same_encode(tiny_fairface_root, stub_detect):
    records, _ = build_training_records(tiny_fairface_root, limit=6, offset=0,
                                        seed=0, detect=stub_detect)
    # The guard this stands in for: reals keeping their original encode while
    # fakes are written by a different encoder teaches the detector the
    # encoder. Matching resolution and container on DF40 RAISED its colour
    # shortcut from 0.800 to 0.904, so "both halves look similar" is not a
    # substitute for "both halves went through one encoder".
    assert {r["compression"] for r in records} == {"jpeg_q95"}


def test_the_corpus_carries_five_generator_families(tiny_fairface_root, stub_detect):
    records, _ = build_training_records(tiny_fairface_root, limit=8, offset=0,
                                        seed=0, detect=stub_detect)
    generators = {r["generator"] for r in records if r["label"] == 1}
    assert generators == {"sbi", "swap_warp_hull", "swap_poisson",
                          "swap_mouth_patch", "swap_lowres_paste"}


def test_every_fake_shares_a_subject_with_the_real_it_came_from(
        tiny_fairface_root, stub_detect):
    records, _ = build_training_records(tiny_fairface_root, limit=8, offset=0,
                                        seed=0, detect=stub_detect)
    by_subject = {}
    for r in records:
        by_subject.setdefault(r["subject_id"], set()).add(r["label"])
    assert all(labels == {0, 1} for labels in by_subject.values())


def test_sample_ids_are_unique(tiny_fairface_root, stub_detect):
    # bench.runner._logo_results raises on duplicates, but only after the
    # whole corpus has been scored. Catching it here costs nothing.
    records, _ = build_training_records(tiny_fairface_root, limit=8, offset=0,
                                        seed=0, detect=stub_detect)
    ids = [r["sample_id"] for r in records]
    assert len(set(ids)) == len(ids)


def test_offset_skips_the_first_n_photographs(tiny_fairface_root, stub_detect):
    a, _ = build_training_records(tiny_fairface_root, limit=4, offset=0,
                                  seed=0, detect=stub_detect)
    b, _ = build_training_records(tiny_fairface_root, limit=4, offset=4,
                                  seed=0, detect=stub_detect)
    assert not ({r["subject_id"] for r in a} & {r["subject_id"] for r in b})


def test_reencode_is_idempotent_in_container_terms():
    img = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    once, twice = reencode(img), reencode(reencode(img))
    assert once.shape == twice.shape == img.shape


def test_fit_refuses_a_one_class_corpus():
    features = np.random.default_rng(0).normal(size=(20, 8)).astype(np.float32)
    with pytest.raises(ValueError, match="both labels"):
        fit_head(features, np.zeros(20, dtype=np.int64), version="0.0.1-test")


def test_the_fitted_head_reproduces_the_logistic_probabilities():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 8)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    labels = (features[:, 0] > 0).astype(np.int64)
    head = fit_head(features, labels, version="0.0.1-test")
    with torch.no_grad():
        probs = torch.softmax(head(torch.from_numpy(features)), dim=1)[:, 1]
    assert ((probs > 0.5).numpy().astype(int) == labels).mean() > 0.9


def test_the_version_is_written_into_the_head():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 8)).astype(np.float32)
    labels = np.array([0, 1] * 20, dtype=np.int64)
    head = fit_head(features, labels, version="0.1.0-probe-fairface60000")
    assert head.version() == "0.1.0-probe-fairface60000"


def test_a_saved_head_loads_back_strictly(tmp_path):
    rng = np.random.default_rng(0)
    head = fit_head(rng.normal(size=(40, 8)).astype(np.float32),
                    np.array([0, 1] * 20, dtype=np.int64), version="v")
    path = tmp_path / "h.pt"
    torch.save(head.state_dict(), path)
    GendHead(in_dim=8).load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True))
```

`tiny_fairface_root` and `stub_detect` are fixtures: a `tmp_path` directory of N session folders,
each holding a synthetic `frame_00.jpg` with a drawn face-like blob and a `results.json` in
`training/export_fairface.py`'s format, and a detector returning one fixed box per frame. Follow
the pattern already in `tests/corpora/test_swap_corpus.py` and `tests/test_fit_vcip.py` rather
than inventing a third. **Put both in `tests/conftest.py`**, not in this test module: Task 9 uses
the same two, and a fixture copied into a second file is a fixture that drifts.

- [x] **Step 2: Run them and watch them fail**

Run: `python3 -m pytest tests/test_fit_gend.py -q`
Expected: `ModuleNotFoundError: No module named 'training.fit_gend'`.

- [x] **Step 3: Write the fitter**

Key implementation points, each load-bearing:

```python
#: Both halves of every pair go through this, once. The value matches the
#: existing swap pipeline's OpenCV quality 95 so that a future corpus mixing
#: the two does not acquire a header shortcut at the join.
REENCODE_QUALITY = 95


def reencode(img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Encode to JPEG at a fixed quality and decode back.

    This is not compression for storage -- the arrays stay in memory. It is
    for making the encode history of reals and fakes IDENTICAL. 99.2% of this
    project's capture corpus is labelled by one byte of the JPEG header
    (bench/vcip_controls.py), which is what happens when the two halves of a
    corpus are written by different encoders.
    """
    ok, buf = cv2.imencode(".jpg", img[:, :, ::-1],
                           [int(cv2.IMWRITE_JPEG_QUALITY), REENCODE_QUALITY])
    if not ok:
        raise ValueError("JPEG encode failed")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1]
```

```python
def fit_head(features, labels, *, version: str) -> GendHead:
    """Logistic regression on cached embeddings, copied into a torch Linear.

    sklearn rather than a torch training loop because the fitted object is
    1,538 numbers over an L2-normalised 768-d input: it is a convex problem
    with a closed enough solution that an optimiser's schedule, seed and
    stopping rule would be three more things to defend and nothing to gain.
    The coefficients are copied into `Linear(768, 2)` so the DETECTOR's
    forward path is torch end to end and `loading.load_model`'s secure path
    applies unchanged.

    The two-row layout reproduces sklearn's probabilities exactly:
    softmax([-w.x/2 - b/2, +w.x/2 + b/2])[1] == sigmoid(w.x + b).
    """
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("training corpus must contain both labels, got "
                         f"{sorted(set(labels.tolist()))}")
    clf = LogisticRegression(max_iter=2000).fit(features, labels)
    w = np.asarray(clf.coef_[0], dtype=np.float32)
    b = float(clf.intercept_[0])
    head = GendHead(in_dim=features.shape[1])
    with torch.no_grad():
        head.linear.weight.copy_(torch.from_numpy(np.stack([-w / 2, w / 2])))
        head.linear.bias.copy_(torch.tensor([-b / 2, b / 2]))
    head.set_version(version)
    return head
```

`main` must:
1. build records at `--offset 0 --limit N`, re-encoding both halves;
2. embed every record's image through `FeatureCache` (reporting hits/misses);
3. split by `subject_id` — **couples, never rows**, since a swap puts one identity in another's
   photograph and `build_swap_corpus` already files both under one pair id;
4. fit on the train side only;
5. compose the version as `f"0.1.0-probe-fairface{n_photographs}"` and write it into the head;
6. `torch.save(head.state_dict(), args.out)`;
7. write `bench/gend_fit.json` carrying: record and family counts, skip tally, cache hits/misses,
   split sizes, the number of photographs consumed (this is the training window), the held-out AUC,
   and the version string. Reuse `_json_safe` from `training/fit_blend.py` (import it rather than
   copying it — a third copy of that function is one too many).

- [x] **Step 4: Run the tests**

Run: `python3 -m pytest tests/test_fit_gend.py -q`
Expected: 10 passed.

- [x] **Step 5: Declare the training window in `bench/eval_swaps.py`**

```python
#: `0.1.0-probe-fairface60000`: `training/fit_gend.py` over the first 60,000
#: photographs, selected with `sorted(root.glob("*/results.json"))`. The same
#: contamination that gave blend_seam a fabricated 0.923 applies identically
#: here and is worse, because this detector's fakes are composited from the
#: same photographs.
TRAINING_WINDOWS = {
    "0.2.0-fairface10k": 10_000,
    "0.1.0-probe-fairface60000": 60_000,
}
```

Use whatever photograph count the run actually consumed. **The number in the version string, the
number in this map and the number in the report must be the same number.**

- [x] **Step 6: Fit for real**

Run:
```bash
python3 -m training.fit_gend \
    --root ~/Desktop/agents/datasets/fairface_sessions \
    --limit 60000 --offset 0 \
    --cache ~/Desktop/agents/datasets/clip_cache \
    --encoder assets/models/clip_vit_l14_visual.pt \
    --out assets/models/gend_clip.pt --report bench/gend_fit.json
```
This is the long one: it is one encoder forward pass per record and the records are
~6x the photograph count. **Run it in the background and record the wall-clock time** — Task 6
needs it, and it is also the honest answer to "how long does a refit cost".

Expected: a held-out AUC printed, `assets/models/gend_clip.pt` written (~8 KB),
`bench/gend_fit.json` written. **An in-domain held-out AUC is not a result** — it is a smoke test
that the pipeline is wired. Do not put it in the evidence card and do not quote it anywhere.

- [x] **Step 7: Register the fitted weights in `assets/manifest.yaml`**

```yaml
  gend_clip_weights:
    source: "Fitted in this repo by training/fit_gend.py: a Linear(768, 2) over frozen OpenAI CLIP ViT-L/14 embeddings of self-blended and composited FairFace crops. No generator weight file, no licensed deepfake dataset, and nothing derived from FaceForensics++ contributes to them."
    license: "CC BY 4.0 (inherited from fairface_corpus) — attribution required wherever these coefficients ship. The encoder they read is separately MIT (clip_vit_l14_visual)."
    commercial_use: true
    owner_attestation: "No capture-corpus material reaches these coefficients: training/fit_gend.py reads only the FairFace session directory, and corpora/sbi.py hard-refuses any swapped session with EvaluationOnlySessionError."
    evidence_url: "docs/EULA-ACCESS.md"
    date_checked: "2026-11-10"
    checked_by: "rohit.kohli@scoreme.in"
    files:
      - "assets/models/gend_clip.pt"
```

- [x] **Step 8: Commit**

```bash
git add training/fit_gend.py tests/test_fit_gend.py bench/eval_swaps.py \
        assets/manifest.yaml bench/gend_fit.json
git commit -m "feat: slot D has coefficients, and a declared training window that can refuse a corpus"
```

**`--limit 60000` is not reachable and the window is 2,000 instead.** Measured before committing
any hours: one ViT-L/14 forward pass is ~0.6 images/second on the development workstation, and the
corpus is ~5 records per photograph, so 60,000 photographs is ~300,000 forward passes — about
**139 hours**. 2,000 was chosen for three reasons, in order: it fits one run; it sits INSIDE
`blend_seam`'s 10,000-photograph window, so the standing "evaluate at offset >= 10000" rule
already covers slot D and no second discipline is needed; and a 768-parameter linear probe over
~8,000 training rows is well determined — this project's own SFHQ finding (500 images reach what
204,716 reach) is the evidence that more data saturates fast.

**The fit was moved to the deployment instance, and that is the lesson.** Started locally, it ran
at ~0.4 images/second and projected to ~7 hours. The same job on the 8-vCPU EC2 box took **35
minutes** — the workstation's torch is a CUDA build running a CPU path and is ~6x slower (Task 6
measured this: 1,637 ms/image local against 276 ms on the instance). Every long job in this plan
belongs there.

**Measured corpus:** 9,886 records, 5,903 fake, 1,001 couples, 78 photographs skipped `no_face`.
Families: real 3,983 / sbi 1,995 / swap_mouth_patch 991 / swap_poisson 979 / swap_warp_hull 969 /
swap_lowres_paste 969. In-domain held-out AUC **0.928** over 1,975 rows — which is a SMOKE TEST
that the pipeline is wired and is recorded as such in `bench/gend_fit.json`; it is not in the
evidence card and is not slot D's performance.

**Task 4's cache earned its place here:** 9,886 records produced only **7,895 forward passes**,
because each photograph's self-blend-real and its couple-real are byte-identical after `reencode`
and address the same cache entry.

**Three deviations.**

*Subject ids are couple ids for the self-blends too.* `corpora.sbi` files a sample under the
photograph; `corpora.swap_corpus` files one under the COUPLE. Left alone, one photograph carries
two different `subject_id`s and a subject-disjoint split will happily put its self-blend pair in
train and its couple in test — the same face, at the same alignment, on both sides. That is the
contamination this project has already paid for once. Couples are disjoint, so
`build_training_records` adopts the couple id for both families and an unpaired photograph becomes
`solo/<id>`.

*`_split_by_subject` is local to this module* rather than reusing `fit_blend.split_by_subject`,
which takes `Sample` objects and not bench-runner records. `_json_safe` IS imported from
`fit_blend`, as Step 3 requires.

*A cross-environment difference worth recording:* the instance's OpenCV 5.0 skips **78**
photographs for `no_face` where the workstation's OpenCV 4.x skips **77**. One record. The corpus
is therefore not bit-reproducible across the two boxes, and the number that ships is the
instance's.

**Measured after Step 8:** 1,045 tests pass, coverage 93.15%, ruff and mypy clean. The head loads
strictly into a default `GendHead` (1,538 parameters, version `0.1.0-probe-fairface2000`), the
asset gate clears `gend_clip_weights`, and the CLI now reports slot D as
`uncalibrated_for_band` rather than `weights_absent` — it scores; it has no calibration yet, which
is Task 9.

---

### Task 6: Measure the forward pass on the deployment's own hardware — **DONE 2026-11-11** (`47a4f41`)

Spec §8: "CLIP ViT-L/14 forward cost on that box is **unmeasured and must be benchmarked before
any latency claim**." This task produces that number, and it also decides Phase 2: LayerNorm
tuning needs a backward pass through the frozen encoder, which costs roughly three times a forward
pass per image per step, and whether that is an overnight job or an impossible one is arithmetic
nobody has done yet.

**Files:**
- Create: `bench/clip_latency.py`
- Create: `tests/bench/test_clip_latency.py`
- Modify: `bench/gend_report.md` is not written yet; this writes `bench/clip_latency.json`.

**Interfaces:**
- Produces: `def measure(tower, *, n: int, batch_size: int, seed: int) -> dict[str, float]`
  returning `{"n", "batch_size", "mean_ms", "p50_ms", "p95_ms", "warmup_ms", "threads"}` — all
  per-image milliseconds except `n`, `batch_size` and `threads`; `def main(argv=None) -> int`.

- [x] **Step 1: Write the failing test**

```python
from dfd.detectors.clip_vit import CLIPVisualTower
from bench.clip_latency import measure


def test_measure_reports_a_distribution_not_one_number():
    out = measure(CLIPVisualTower.tiny().eval(), n=4, batch_size=2, seed=0)
    assert out["n"] == 4 and out["batch_size"] == 2
    assert out["p95_ms"] >= out["p50_ms"] > 0.0
    assert out["threads"] >= 1


def test_the_first_pass_is_reported_separately_not_averaged_in():
    # torch's first forward allocates, dispatches and warms caches. Folding
    # it into the mean makes a latency that describes process startup, and
    # the service forwards thousands of images per process. It is reported
    # rather than discarded, because cold start is a real number -- just not
    # the one a steady-state p95 is claiming to be.
    out = measure(CLIPVisualTower.tiny().eval(), n=6, batch_size=1, seed=0)
    assert out["n"] == 6
    assert out["warmup_ms"] > 0.0
    assert out["p50_ms"] <= out["p95_ms"]
```

- [x] **Step 2: Run it and watch it fail**

Run: `python3 -m pytest tests/bench/test_clip_latency.py -q`
Expected: `ModuleNotFoundError: No module named 'bench.clip_latency'`.

- [x] **Step 3: Write it**

`measure` runs one discarded warm-up batch, then `n` timed forwards under `torch.no_grad()`,
reporting mean/p50/p95 per image and `torch.get_num_threads()`. `main` adds `--encoder`,
`--n`, `--batch-size`, `--out`, and `--quantize` (dynamic INT8 over `nn.Linear`, which measured
1.52x on the ViT-Base profiled here). Docstring records the comparison the spec gives: ViT-Base/16
is ~86M parameters and ~17.6 GFLOPs at 224px, ViT-L/14 is ~304M and ~81 GFLOPs — about 3.5x the
parameters and 4.5x the compute.

- [x] **Step 4: Run the tests, then measure on the EC2 instance**

```bash
python3 -m pytest tests/bench/test_clip_latency.py -q
# on the instance (ops/deploy-ec2.sh gets the code there):
python3 -m bench.clip_latency --encoder assets/models/clip_vit_l14_visual.pt \
    --n 32 --batch-size 1 --out bench/clip_latency.json
python3 -m bench.clip_latency --encoder assets/models/clip_vit_l14_visual.pt \
    --n 32 --batch-size 1 --quantize --out bench/clip_latency_int8.json
```

- [x] **Step 5: Do the Phase 2 arithmetic and write it down**

In `bench/clip_latency.json`'s accompanying note, record:

```
forward, fp32, batch 1:            <measured> ms/image
forward, dynamic INT8, batch 1:    <measured> ms/image
implied LN-tuning step (~3x fwd):  <measured x 3> ms/image
one epoch over N training images:  <that x N> hours
```

Then state the ruling plainly: whether Phase 2 (Task 13) is an overnight CPU job on a reduced
corpus, or is blocked on the GPU decision that is item 7 of `docs/HANDOFF.md`'s owner-action list.
**Do not guess this in advance** — the whole point of the task is that the number decides.

- [x] **Step 6: Commit**

```bash
git add bench/clip_latency.py tests/bench/test_clip_latency.py bench/clip_latency*.json
git commit -m "feat: the encoder's CPU cost is measured, and it decides whether Phase 2 can run here"
```

**Measured on the deployment instance itself**, not extrapolated: 8 vCPU, 15 GB, no GPU, torch
2.15.0+cpu, Python 3.14.4, `torch.get_num_threads() == 4`, `n=32`, batch 1.

```
forward, fp32, batch 1:            276 ms/image   (p50 276, p95 277, cold start 281)
forward, dynamic INT8, batch 1:    177 ms/image   (p50 177, p95 177, cold start 181)
implied LN-tuning step (~3x fwd):  829 ms/image   NOT MEASURED - rule of thumb
one epoch over 9,886 images:       2.3 hours
```

INT8 is **1.57x**, within a hair of the 1.52x the spec recorded for ViT-Base — the speedup
survives the 3.5x parameter jump. The full write-up, including what the number does NOT say, is
`bench/clip_latency.md`.

**The ruling the plan said not to guess: Phase 2 IS runnable here, and is NOT blocked on the GPU
decision** — at the corpus size slot D is actually fitted on. One epoch over 9,886 records is 2.3
hours, so a 5-epoch run is an overnight job. The condition is the whole finding: at the plan's
original 60,000-photograph window (~300,000 records) one epoch is **69 hours** and Phase 2 would
have been blocked on hardware this project does not have. It is affordable only because the
window is small.

**A finding that changes how everything after this runs.** The same harness on the development
workstation (12 vCPU, torch 2.4.1+cu121) reports **1,637 ms/image** — roughly **6x slower than
the 8-vCPU deployment**, because the local build is a CUDA build running a CPU path. Every long
job in this plan therefore belongs on the instance, not here. Task 5's fit was started locally,
measured at ~0.4 images/second, and moved; see that task's note.

**Two deviations.** `torch.ao.quantization.quantize_dynamic`, not the `torch.quantization` alias
the plan wrote — the alias is deprecated and the deployment runs a newer torch than this
workstation. And `bench/clip_latency.md` was added beside the JSON, because Step 5 asks for an
arithmetic note and a `.json` is the wrong container for a ruling.

**Worth knowing before trusting INT8 later:** torch now warns that
`torch.quantize_per_tensor` and the other quantized tensor creation functions "are deprecated and
will be removed in a future PyTorch release". The 1.57x is a measurement, not a plan.

---

### Task 7: The measurement — honest LOGO, a null beside it, and a control that survives it — **DONE 2026-11-11** (`70bc99b`)

**This is the task the whole plan exists for.** Everything before it is plumbing.

**What makes this LOGO different from `bench/runner.py`'s.** `_logo_results` slices
already-computed scores, because P0's detectors are not trained in the harness; its own docstring
calls that "honest scope" and says it becomes the full protocol once fitting exists. Fitting now
exists, and the cached features make a per-fold refit cost milliseconds. So `bench/eval_gend.py`
runs the **full** protocol: for each held-out generator, fit the head on the other families' fakes
plus the reals, score the held-out family, report that fold.

**The five admissibility conditions** (spec §7), each a named output of this module:
1. `permutation_null` — refit on shuffled labels, `bench.metrics.permutation_null`, with
   `permutation_p` beside it. A number inside the null is not evidence.
2. Intervals from `bench.metrics.bootstrap_ci_by_group`, resampled over **families** (the couple
   id), never rows.
3. Kish effective sample size, reported per class.
4. LOGO per fold, with the **best** fold explicitly distrusted in the prose.
5. A shortcut control: a model fitted on Lab colour means alone, and one on the JPEG luma
   quantisation table alone. If either reaches the detector's AUC, the detector has measured
   nothing.

**Files:**
- Create: `bench/eval_gend.py`
- Create: `tests/bench/test_eval_gend.py`
- Modify: `bench/metrics.py` (add `kish_ess`)
- Modify: `tests/bench/test_metrics.py`

**Interfaces:**
- Consumes: `build_training_records` (Task 5), `FeatureCache` (Task 4), `fit_head` (Task 5),
  `bench.metrics.{auc, bootstrap_ci_by_group, permutation_null, permutation_p}`,
  `bench.protocol.logo_splits`.
- Produces:
  - `bench.metrics.kish_ess(counts: Sequence[int] | np.ndarray) -> float` — `(Σn)² / Σn²`.
  - `def logo_refit(features, labels, generators, groups, *, seed) -> dict[str, dict[str, float]]`
    — held-out generator → `{auc, n, n_fake, ci_lo, ci_hi}`.
  - `def colour_features(images) -> npt.NDArray[np.float32]` — per-channel Lab means and stds.
  - `def jpeg_table_features(images) -> npt.NDArray[np.float32]` — the first luma quantisation
    coefficients, the statistic that labels 99.2% of the capture corpus.
  - `def main(argv=None, *, detect=detect_faces) -> int`.

- [x] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import numpy as np
import pytest

from bench.eval_gend import colour_features, logo_refit
from bench.metrics import kish_ess


def test_kish_ess_of_equal_families_is_the_family_count():
    assert kish_ess([10, 10, 10, 10]) == pytest.approx(4.0)


def test_kish_ess_collapses_when_one_family_dominates():
    # DF40's measured SHAPE, not its exact counts: 1601 fakes fell into 45
    # filename families with 999 in the largest, for a Kish ESS of 2.4. The
    # array below is that shape at a smaller total and lands at 2.45 -- the
    # point being that a corpus worth ~2.4 fakes cannot carry an interval
    # however many rows it has.
    counts = [999] + [13] * 44
    assert kish_ess(counts) < 3.0


def test_kish_ess_refuses_an_empty_corpus():
    with pytest.raises(ValueError, match="at least one"):
        kish_ess([])


def test_logo_refits_per_fold_and_never_trains_on_the_held_out_family():
    rng = np.random.default_rng(0)
    n = 400
    features = rng.normal(size=(n, 16)).astype(np.float32)
    generators = np.array(["real"] * 200 + ["g1"] * 100 + ["g2"] * 100)
    labels = (generators != "real").astype(np.int64)
    # g1 is separable on axis 0, g2 on axis 1 and on nothing else: a fitter
    # that saw g2 would ace the g2 fold, and one that did not cannot.
    features[generators == "g1", 0] += 6.0
    features[generators == "g2", 1] += 6.0
    groups = np.arange(n) // 2
    folds = logo_refit(features, labels, generators, groups, seed=0)
    assert set(folds) == {"g1", "g2"}
    assert folds["g1"]["auc"] < 0.75
    assert folds["g2"]["auc"] < 0.75


def test_logo_reports_a_fold_it_can_actually_separate():
    rng = np.random.default_rng(1)
    n = 400
    features = rng.normal(size=(n, 16)).astype(np.float32)
    generators = np.array(["real"] * 200 + ["g1"] * 100 + ["g2"] * 100)
    labels = (generators != "real").astype(np.int64)
    features[labels == 1, 0] += 6.0        # one shared direction
    groups = np.arange(n) // 2
    folds = logo_refit(features, labels, generators, groups, seed=0)
    assert folds["g1"]["auc"] > 0.9 and folds["g2"]["auc"] > 0.9


def test_logo_refuses_a_single_generator_corpus():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 4)).astype(np.float32)
    generators = np.array(["real"] * 20 + ["g1"] * 20)
    labels = (generators != "real").astype(np.int64)
    with pytest.raises(ValueError, match="one generator"):
        logo_refit(features, labels, generators, np.arange(40), seed=0)


def test_the_colour_control_finds_a_planted_colour_shortcut():
    # The control must be able to PASS, or reporting it proves nothing. Lab
    # colour means alone separated DF40's halves at 0.843.
    rng = np.random.default_rng(0)
    reals = [rng.integers(0, 120, (32, 32, 3), dtype=np.uint8) for _ in range(40)]
    fakes = [rng.integers(135, 255, (32, 32, 3), dtype=np.uint8) for _ in range(40)]
    x = colour_features(reals + fakes)
    y = np.array([0] * 40 + [1] * 40)
    from sklearn.linear_model import LogisticRegression
    from bench.metrics import auc
    probs = LogisticRegression(max_iter=1000).fit(x, y).predict_proba(x)[:, 1]
    assert auc(probs, y) > 0.95


def test_the_colour_control_is_blind_to_a_corpus_with_no_colour_shortcut():
    rng = np.random.default_rng(0)
    both = [rng.integers(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(80)]
    x = colour_features(both)
    y = np.array([0] * 40 + [1] * 40)
    from sklearn.linear_model import LogisticRegression
    from bench.metrics import auc
    probs = LogisticRegression(max_iter=1000).fit(x, y).predict_proba(x)[:, 1]
    assert 0.3 < auc(probs, y) < 0.7
```

- [x] **Step 2: Run them and watch them fail**

Run: `python3 -m pytest tests/bench/test_eval_gend.py -q`
Expected: `ModuleNotFoundError: No module named 'bench.eval_gend'` and
`ImportError: cannot import name 'kish_ess'`.

- [x] **Step 3: Add `kish_ess` to `bench/metrics.py`**

```python
def kish_ess(counts: Sequence[int] | np.ndarray) -> float:
    """Kish's effective sample size over group sizes: (sum n)^2 / sum n^2.

    Equals the group count when groups are equal, and collapses toward 1 when
    one group dominates. Measured on DF40: 1601 fakes in 45 filename families
    with 999 in the largest gives 2.4, which is why no interval over that
    corpus meant what it appeared to. Report it beside every n, or the reader
    is invited to read rows as independent observations.

    Raises:
        ValueError: on an empty or all-zero input. Returning 0.0 would read
            as a measured answer rather than as nothing to measure.
    """
    n = np.asarray(counts, dtype=float)
    if n.size == 0 or not n.sum():
        raise ValueError("kish_ess needs at least one non-empty group")
    return float(n.sum() ** 2 / np.square(n).sum())
```

With two tests in `tests/bench/test_metrics.py` mirroring the two above.

- [x] **Step 4: Write `bench/eval_gend.py`**

Module docstring states what the corpus can and cannot show, in the style of
`bench/eval_swaps.py`'s `_SCOPE` block — in particular that these are **classical composites, not
generator output**, so a detector that beats them has not been shown to beat FSGAN; and that the
corpus must be built at `--offset >= ` the fitted detector's training window or every number is
memorisation.

`logo_refit` must:
- raise `ValueError("... one generator ...")` when fewer than two fake families are present
  (`bench.protocol.logo_splits` refuses too, but after the corpus is built);
- for each held-out family, fit on reals + the *other* families, score the held-out family's fakes
  against the held-out **reals** (partition the reals by group so the fold's reals are also unseen
  — a fold that tests on training reals is half contaminated and reads as a strong fold);
- return per-fold AUC, n, n_fake and a family-resampled CI from `bootstrap_ci_by_group`.

`main` must write `bench/gend_report.json` and `bench/gend_report.md` carrying, in this order:
the corpus composition and skip tally; Kish ESS per class; the two shortcut controls; the per-fold
LOGO table with the worst fold as the headline and a sentence distrusting the best;
the permutation null range and the two-sided p; and the scope block. A card that records only good
news is not a report card.

- [x] **Step 5: Run the tests**

Run: `python3 -m pytest tests/bench/test_eval_gend.py tests/bench/test_metrics.py -q`
Expected: all pass.

- [x] **Step 6: Run the measurement**

```bash
python3 -m bench.eval_gend \
    --root ~/Desktop/agents/datasets/fairface_sessions \
    --offset 60000 --limit 8000 \
    --cache ~/Desktop/agents/datasets/clip_cache \
    --encoder assets/models/clip_vit_l14_visual.pt \
    --synthetic-sources ~/Desktop/agents/datasets/sfhq_part3 \
    --out-json bench/gend_report.json --out-md bench/gend_report.md
```

The offset **must** be at or beyond the training window Task 5 recorded. A run at a smaller offset
is the mistake that produced a fabricated 0.923 once already.

- [x] **Step 7: Read the result before believing it**

Check, in this order, and stop at the first failure:
1. Is the worst LOGO fold's AUC outside the permutation null, with p below 0.08 at n=12? If not,
   **there is no result** — raise the shuffle count, and if it still sits inside the null, record
   that and stop.
2. Does either shortcut control reach the detector's AUC? If so the number is the shortcut's.
3. Is the Kish ESS large enough that the interval means anything?
4. Only then: is the worst fold at or above 0.75?

- [x] **Step 8: Commit, whatever the answer**

```bash
git add bench/eval_gend.py bench/metrics.py tests/bench/test_eval_gend.py \
        tests/bench/test_metrics.py bench/gend_report.json bench/gend_report.md
git commit -m "feat: slot D gets a leave-one-generator-out number, and the null that judges it"
```

A negative result is committed with the same message discipline as a positive one. This project
has retracted a headline before, and the retraction is in the history where it belongs.

## THE RESULT

Measured on the deployment instance over photographs 10,000–11,999 — outside every declared
training window — 9,907 records, 5,918 fake, 1,001 couples, 69 skipped `no_face`.

| held-out generator | AUC | 95% CI (by couple) | n | n fake |
|---|---|---|---|---|
| **sbi** | **0.738** | 0.726–0.751 | 2,988 | 997 |
| swap_mouth_patch | 0.984 | 0.977–0.991 | 2,487 | 496 |
| swap_poisson | 0.995 | 0.992–0.998 | 2,482 | 491 |
| swap_warp_hull | 0.998 | 0.996–0.999 | 2,475 | 484 |
| swap_lowres_paste | 0.999 | 0.997–1.000 | 2,475 | 484 |

**Headline: the worst fold, `sbi`, at 0.738.** Permutation null over 48 shuffles of the TRAINING
labels: **0.449–0.561**, two-sided **p = 0.02**.

Step 7's checklist, in its order:

1. **Outside the null?** Yes. 0.738 against 0.449–0.561. At the plan's 12 shuffles p was 0.0769,
   which is the resolution FLOOR at n=12 and says only "none of 12 came close" — so the null was
   re-run at 48, which the cache made cost 1m57s (9,907 hits, 0 misses). p = **0.02**. There is
   signal.
2. **Does either control reach it?** No. Colour (Lab means and stds, fitted and scored IN SAMPLE,
   the strongest form) reaches **0.530**. The JPEG quantisation control is **0.500 and structural,
   not measured** — every record passes through one encoder at q95, which is exactly what
   `reencode` exists to arrange, so there is nothing for it to separate. Reported as structural
   rather than dressed up as a searched-for-and-not-found 0.5.
3. **Is the Kish ESS large enough?** Yes, and this is the first corpus in this project where it
   is: **4.47 of 5** fake families, **996.44 of 1,001** couples. Compare DF40's 2.4. The intervals
   mean what they appear to.
4. **Is the worst fold at or above 0.75?** **No — 0.738, and its CI 0.726–0.751 barely touches
   the floor.** Slot D does not open the evidence gate.

**So: a real but insufficient result.** Signal that survives a null, clean controls, honest
intervals, and a headline **0.012 under the floor it had to clear**. The four swap folds at
0.984–0.999 are explicitly distrusted and are not the headline: holding out one of four nearly
identical OpenCV compositing techniques still leaves three of them in the fit, so those folds
describe the corpus's internal similarity, not transfer. `sbi` is the only family unlike the
others, and it is the only fold that tests anything.

**One deviation, in the plan's favour.** Step 3 asks for `kish_ess` to be added to
`bench/metrics.py`; it was already there, with four tests, from the benchmark-harness plan. The
three duplicate tests Step 1 listed were NOT written, and `tests/bench/test_eval_gend.py` says so
at the top.

**One design point the plan left implicit, and it matters.** Holding out a generator does not hold
out the REALS — every couple contributes its reals to every family's fold. So each fold is split
by couple as well, and is scored only on couples the fit never saw. Without that, every fold tests
on its own training reals and reads stronger than it is.

---

### Task 8: The in-domain evaluation, and the quality floor it decides — **DONE 2026-11-11** (`19f4eac`)

The capture corpus is the only material that looks like live v-CIP traffic, and a detector trained
on Flickr photographs faces a real domain shift towards it. **Evaluation only**: this module fits
nothing, and the biometric-consent question (spec §9.1) stays open regardless of what it measures.

**Files:**
- Create: `bench/eval_gend_vcip.py`
- Create: `tests/bench/test_eval_gend_vcip.py`
- Modify: `assets/manifest.yaml`
- Modify: `src/dfd/detectors/gend.py` (the measured `min_quality_band`)

**Interfaces:**
- Consumes: `GendClipDetector.score` (Task 3), `training.fit_vcip.{_frames, select_face}`,
  `bench.vcip_controls`'s matched re-encode, `bench.metrics.{auc, tpr_at_fpr, kish_ess}`.
- Produces:
  - `def session_scores(scores, labels, sessions) -> tuple[np.ndarray, np.ndarray, np.ndarray]` —
    mean score per session, its label, and the session ids in sorted order. Raises `ValueError`
    on a session whose frames carry conflicting labels.
  - `def per_band(scores, labels, bands) -> dict[str, dict[str, float | int | None]]` — band →
    `{n, n_genuine, n_swapped, auc, tpr_at_zero_fpr}`, with `auc: None` wherever one class is
    absent.
  - `def main(argv=None, *, detect=detect_faces) -> int` — `--captures`, `--encoder`,
    `--weights`, `--out`.

- [x] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import numpy as np
import pytest

from bench.eval_gend_vcip import per_band, session_scores


def test_this_module_fits_nothing(monkeypatch):
    # The guard that matters more than any number this module produces: a
    # capture-corpus fit is the one thing spec §6.4 forbids outright, and it
    # is forbidden for a reason no measurement can overturn. Importing a
    # fitter here would be the first step toward one.
    import bench.eval_gend_vcip as m

    assert not hasattr(m, "fit_head")
    assert not hasattr(m, "fit_gend")
    source = __import__("inspect").getsource(m)
    assert "LogisticRegression" not in source
    assert "training.fit_gend" not in source


def test_per_band_table_reports_its_denominators():
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    labels = np.array([0, 1, 0, 1, 0, 1])
    bands = np.array(["reject", "reject", "medium", "medium", "low", "low"])
    table = per_band(scores, labels, bands)
    assert set(table) == {"reject", "medium", "low"}
    for band, row in table.items():
        assert set(row) >= {"n", "n_genuine", "n_swapped", "auc",
                            "tpr_at_zero_fpr"}
        assert row["n"] == row["n_genuine"] + row["n_swapped"]


def test_a_band_with_one_class_reports_no_auc_rather_than_a_number():
    # `high` held 14 genuine frames and no swapped one when slot C was
    # measured on this corpus. An AUC there would be invented, and a
    # fabricated 1.0 in a per-band table is exactly the kind of number that
    # gets quoted later without its denominator.
    scores = np.array([0.1, 0.2, 0.3, 0.9])
    labels = np.array([0, 0, 0, 1])
    bands = np.array(["high", "high", "high", "medium"])
    table = per_band(scores, labels, bands)
    assert table["high"]["auc"] is None
    assert table["high"]["n_swapped"] == 0
    assert table["medium"]["auc"] is None  # one class on that side too


def test_sessions_are_the_unit_not_frames():
    # Frames of one session share the subject, the camera, the lighting and
    # the encode. Scoring them as independent rows is how 506 frames read as
    # 506 observations when they carry about 106.
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    labels = np.array([1, 1, 0, 0])
    sessions = np.array(["s1", "s1", "s2", "s2"])
    s, y, ids = session_scores(scores, labels, sessions)
    assert list(ids) == ["s1", "s2"]
    assert len(s) == len(y) == 2


def test_a_session_with_conflicting_frame_labels_is_refused():
    # A session is genuine or swapped; frames of it disagreeing means the
    # corpus loader mislabelled something, and averaging that away would
    # produce a plausible number from a broken input.
    with pytest.raises(ValueError, match="conflicting labels"):
        session_scores(np.array([0.9, 0.1]), np.array([1, 0]),
                       np.array(["s1", "s1"]))


def test_the_encoder_shortcut_control_runs_and_is_reported(capture_pair_fixture):
    # 99.2% of this corpus is labelled by one JPEG header byte: the swap
    # pipeline re-encodes at OpenCV quality 95 and the capture path never
    # re-encodes at all. The control re-encodes the GENUINE half alone, at
    # the quality that reproduces the swapped half's luma table EXACTLY --
    # `bench.vcip_controls` finds it by search and refuses if no quality
    # matches or if more than one table survives. Both numbers are reported;
    # the controlled one is the one to quote.
    from bench.eval_gend_vcip import main

    rc = main(["--captures", str(capture_pair_fixture.root),
               "--encoder", str(capture_pair_fixture.encoder),
               "--weights", str(capture_pair_fixture.head),
               "--out", str(capture_pair_fixture.out)])
    assert rc == 0
    report = json.loads(capture_pair_fixture.out.read_text())
    assert "uncontrolled" in report and "encoder_matched" in report
    assert report["encoder_matched"]["quality"] is not None
```

`capture_pair_fixture` builds a handful of synthetic genuine/swapped session folders plus a tiny
encoder and head on `tmp_path`, in the shape `tests/test_fit_vcip.py` already uses. Do not invent
a second fixture style for the same corpus.

- [x] **Step 2: Run them and watch them fail**

Run: `python3 -m pytest tests/bench/test_eval_gend_vcip.py -q`

- [x] **Step 3: Write it**

Reuse, do not reimplement: `training.fit_vcip._frames` and `select_face` for frame selection
(the largest-face rule — getting this wrong is what made every slot C number stale once), and
`bench.vcip_controls`'s matched re-encode for the control. Score through
`GendClipDetector.score` so the measured path is the deployed path, not a reimplementation of it.
Report: session-level AUC (sessions, never frames — frames of one session share everything), TPR
at a zero false-alarm budget, the per-band table with denominators, and the same numbers after the
encoder shortcut is removed. **Quote the controlled pair.**

- [x] **Step 4: Run it**

```bash
python3 -m bench.eval_gend_vcip --captures ~/Desktop/agents/fraud_gff/deepfake_detection/captures \
    --encoder assets/models/clip_vit_l14_visual.pt \
    --weights assets/models/gend_clip.pt --out bench/gend_vcip.json
```

- [x] **Step 5: Set the quality floor from the table, not from taste**

Replace slot D's provisional `min_quality_band` with the measured value and paste the per-band
table into the attribute's comment, in the form `NPRDetector.min_quality_band` already uses —
including the denominators and an explicit sentence about which columns the argument rests on. If
the table says a floor above `reject` discards a large share of swapped frames, the floor goes
down, whatever the per-band AUCs say.

- [x] **Step 6: Register the two evaluation corpora**

```yaml
  vkyc_capture_corpus:
    source: "v-CIP session captures at ~/Desktop/agents/fraud_gff/deepfake_detection/captures"
    license: "UNRESOLVED — owner attests rights in the recordings; no legal review has addressed biometric processing. EVALUATION ONLY. Never a training input (spec §6.4)."
    commercial_use: false
    owner_attestation: "2026-09-21: the capture corpus is available with rights. Not independently verified. DPDP Act / GDPR Art. 9 treat biometric data as sensitive and a licence in the recording does not answer that."
    evidence_url: "docs/EULA-ACCESS.md"
    date_checked: "2026-11-10"
    checked_by: "rohit.kohli@scoreme.in"
  vkyc_inswapper_swaps:
    source: "inswapper_128 swaps of vkyc_capture_corpus frames, generated in this project"
    license: "research-only — InsightFace's distributed models are non-commercial and inswapper_128 was withdrawn from distribution. EVALUATION ONLY; never distributed, and nothing fitted on them ships."
    commercial_use: false
    evidence_url: "assets/manifest.yaml: npr_capture_head"
    date_checked: "2026-11-10"
    checked_by: "rohit.kohli@scoreme.in"
```

- [x] **Step 7: Commit**

```bash
git add bench/eval_gend_vcip.py tests/bench/test_eval_gend_vcip.py \
        bench/gend_vcip.json assets/manifest.yaml src/dfd/detectors/gend.py
git commit -m "feat: slot D meets the traffic it is for, and its quality floor stops being a guess"
```

**Step 4's `--captures` path is the wrong corpus, and it matters.** The plan names
`~/Desktop/agents/fraud_gff/deepfake_detection/captures`: 42 sessions, 142 frames, and really 58
distinct images with 19 genuine crops. The corpus every measurement since 2026-09-24 actually uses
is `/home/rohit/Desktop/vkyc-server/captures` — **146 session folders, 554 frames, 140 sessions,
218 genuine / 336 swapped**. Both exist on this machine. The run used the second, and
`assets/manifest.yaml: vkyc_capture_corpus` now records which one it is and why, so the next
reader does not have to rediscover it.

## THE RESULT

509 frames scored over 125 sessions (58 genuine / 67 swapped), 45 frames skipped `no_face`. Kish
ESS of frames by session: **106.9** of 125.

| | frame AUC | session AUC | session TPR @ 0% FPR |
|---|---|---|---|
| uncontrolled | 0.678 | 0.664 | 0.000 |
| **encoder-matched (q95)** | **0.674** | **0.663** | **0.015** |

**The encoder shortcut is not what slot D is reading, and that is the finding.** The two rows are
the same number — 0.664 against 0.663. One JPEG header byte labels 99.2% of this corpus, and slot
C's uncontrolled figures were inflated by exactly that. Slot D's survives the control untouched.
It is a weak number honestly, rather than a strong number dishonestly.

**0.663 is well under the 0.75 floor**, and the TPR at a zero false-alarm budget is 1.5% — slot D
catches essentially nothing on live-shaped traffic at an operationally usable threshold. Together
with Task 7's 0.738 worst fold, slot D does not open the evidence gate on either corpus.

## Step 5: the floor is `reject`, and the table says so

| band | n | genuine | swapped | frame AUC | TPR@0%FPR |
|---|---|---|---|---|---|
| reject | 232 | 14 | 218 | 0.541 | 0.165 |
| medium | 238 | 139 | 99 | 0.573 | 0.000 |
| low | 12 | 2 | 10 | 0.700 | 0.400 |
| high | 27 | 27 | 0 | none | none |

**The argument rests on the denominator columns, not the AUC column.** A floor above `reject`
discards **218 of 328 swapped frames (66.5%)** against 14 of 182 genuine (7.7%) — two thirds of
the attacks thrown away before the detector runs, and almost none of the honest traffic. That is a
security hole, not caution. The provisional `"low"` is replaced by `"reject"`, which is also slot
C's floor, measured from the same corpus and for the same reason. `high` has no swapped frame at
all so it has no AUC; `low`'s 0.700 is the best cell in the table and rests on 12 frames.

**One fix the run forced.** YuNet reports face boxes that run off the edge of a frame, and
`dfd.quality.measure_quality` crops without clamping — so the first real run died on
`cv2.error: (-215:Assertion failed) !_src.empty()` out of `cvtColor`. `_clamped_roi` clips the box
to the frame and counts a frame with nothing left under `roi_outside_frame`. One unmeasurable
frame is a skip with a reason, not a failed evaluation of the other 553. (`GendClipDetector`
already clamps its own crop; `measure_quality` is called before it and does not.)

**Two small deviations.** `matched_encoder` in `bench/vcip_controls.py` could not be reused whole —
it is slot-C specific, building `NPRStatsNet` features — so this module reuses its MECHANICS
(`quant_table`, `find_matching_quality`, `_encode`) and scores through `GendClipDetector.score`,
which is what Step 3 asks for: the measured path is the deployed path. And `main` takes
`encoder_factory`/`head_factory` so the test can drive the whole entry point on a tiny tower.

**Measured after Step 7:** 1,051 tests pass (1,045 before, +6), coverage 93.19%, ruff and mypy
clean from a cold cache.

---

### Task 9: Calibration, from a corpus that is not DF40 — **DONE 2026-11-11** (`f5bf68e`)

A detector with no calibrator returns `llr 0.0` and `uncalibrated_for_band` — the honest answer to
"I was never calibrated in this regime", and also a detector that cannot move a verdict no matter
what the gate says. `training/fit_calibration.py` currently reads DF40 and only DF40, which this
project has ruled can neither support nor refute anything.

**Files:**
- Modify: `training/fit_calibration.py`
- Modify: `tests/test_fit_calibration.py`

- [x] **Step 1: Write the failing test**

```python
import json

import pytest

from training.fit_calibration import main


def test_calibration_can_be_fitted_from_the_swap_corpus(
        tmp_path, tiny_fairface_root, stub_detect):
    # The point: the one corpus this project has ruled unable to measure
    # anything must not be the only thing that can produce a calibration
    # curve for a detector allowed to decide.
    rc = main(["--corpus-kind", "swaps", "--root", str(tiny_fairface_root),
               "--offset", "0", "--limit", "8",
               "--out", str(tmp_path / "cal.json"),
               "--report", str(tmp_path / "rep.json")], detect=stub_detect)
    assert rc == 0
    assert json.loads((tmp_path / "cal.json").read_text())


def test_the_report_records_which_corpus_the_curve_came_from(
        tmp_path, tiny_fairface_root, stub_detect):
    # Two curves for one detector, fitted on two corpora, are two different
    # claims about what a score means. A report that does not say which is
    # which cannot be read a month later.
    main(["--corpus-kind", "swaps", "--root", str(tiny_fairface_root),
          "--offset", "0", "--limit", "8",
          "--out", str(tmp_path / "cal.json"),
          "--report", str(tmp_path / "rep.json")], detect=stub_detect)
    assert json.loads((tmp_path / "rep.json").read_text())["corpus_kind"] == "swaps"


def test_an_unknown_corpus_kind_is_refused_by_the_parser():
    with pytest.raises(SystemExit):
        main(["--corpus-kind", "df40-but-ok", "--corpus", "x"])
```

- [x] **Step 2: Run and watch fail**

Run: `python3 -m pytest tests/test_fit_calibration.py -q`

- [x] **Step 3: Add the second source**

Add `--corpus-kind {df40,swaps}` (default `df40`, so every existing invocation is unchanged).
`swaps` takes `--root/--offset/--limit` and loads via `training.fit_gend.build_training_records`,
mapping each record to the `{image, source_id, label}` shape `_score`/`_observe` already expect.
`split_by_source` is unchanged and still splits whole sources onto one side. Record the corpus kind
in the report so a reader knows which curve they are looking at.

- [x] **Step 4: Fit on the held-out window and check band coverage**

```bash
python3 -m training.fit_calibration --corpus-kind swaps \
    --root ~/Desktop/agents/datasets/fairface_sessions --offset 60000 --limit 4000 \
    --out assets/models/calibration.json --report bench/calibration_report.json
```

Then read `report["detectors"]["gend_clip"]["bands"]`. **If the fitted bands do not cover the bands
the capture corpus actually produces** (Task 8's table says what those are), say so in the report
rather than shipping a curve that abstains at runtime on the only band that matters.

- [x] **Step 5: Commit**

```bash
git add training/fit_calibration.py tests/test_fit_calibration.py \
        assets/models/calibration.json bench/calibration_report.json
git commit -m "feat: a calibration curve that does not come from the corpus we cannot trust"
```

**What Step 4 decided.** `gend_clip`'s fitted bands are `medium` and `reject`, covering 470 of the
509 capture frames Task 8 measured — 92.3%. `reject` alone carries 218 of that corpus's 328 swapped
frames, and it is covered only because this run used the floor Task 8 measured. The first run, on
an EC2 checkout still carrying the provisional `min_quality_band = "low"`, fitted `medium` only and
covered 46.8% — a curve that would have abstained with `uncalibrated_for_band` on nearly half of
live-shaped traffic, and on the half where the attacks are. The stale-floor artefacts were deleted
rather than kept beside the good ones.

Uncovered: `high` (27 capture frames, no swapped one — it could not be fitted on any corpus) and
`low` (12 frames; the fit split held 13 rows there, under `MIN_FIT_SAMPLES`). Both return `llr 0.0`,
which is the honest answer and is visible in the report rather than inferred.

**A defect this task found in the reporting, not in the curve.** The report carried a single `bands`
field built from `sorted(set(f_bands))` — the bands the fit split *saw*. `Calibrator.fit` drops a
band under `MIN_FIT_SAMPLES` or with one class, so that field claimed curves that do not exist, on
the exact field this step tells a reader to consult. `bands_fitted` and `bands_seen` are now
separate, and `Calibrator.fitted_bands` is the source for the first.

**`npr`'s curve changed corpus.** `assets/models/calibration.json` previously held an `npr` curve
fitted by `training/fit_vcip.py` on the capture corpus, conditioned on a 66.4% base rate that file
itself warns about. It is now the swaps-fitted one (holdout AUC 0.546, ECE 0.297). The weaker
coefficients are the honest ones: a curve fitted where the detector cannot separate should produce
small llrs, and the capture-fitted curve's confidence came from a corpus one JPEG header byte
labels for 99.2% of rows.

---

### Task 10: The evidence card, and watching the gate open (or not) — **DONE 2026-11-11** (`bcfe62d`)

The gate needs four things true before slot D may decide: a measured `auc` at or above 0.75, a
`trained_on` that is not the `corpus`, and **both** names present as ids in `assets/manifest.yaml`.

**Files:**
- Modify: `bench/evidence_card.json`
- Modify: `tests/service/test_evidence.py`

- [x] **Step 1: Write the entry**

Fill it from Task 7's and Task 8's reports. Headline AUC is the **worst LOGO fold** or the
controlled in-domain number — never the in-dataset figure, which measures memorisation.

```json
    "gend_clip": {
      "auc": null,
      "auc_ci95": [null, null],
      "corpus": "vkyc_capture_corpus",
      "trained_on": "fairface_corpus",
      "trained_on_note": "self-blended (corpora/sbi.py) and composited (corpora/swaps.py) FairFace crops; the first 60,000 photographs",
      "evidence": "bench/gend_report.md, bench/gend_vcip.json",
      "measured_on": "2026-11-__",
      "weights": "Linear(768, 2) over frozen CLIP ViT-L/14; encoder MIT, head CC BY 4.0",
      "note": "<the worst fold, the permutation null range and p, the Kish ESS, what the two shortcut controls reached, and the sentence distrusting the best fold. If the number is inside the null, say that here and leave auc null — an auc field filled from a number inside its own null is the defect this card exists to prevent.>"
    }
```

- [x] **Step 2: Add a test that the card and the manifest agree**

```python
def test_every_corpus_name_in_the_card_is_a_manifest_id():
    # gated_detectors already enforces this at runtime and logs a refusal.
    # A test makes the card's authoring error visible at commit time instead
    # of as a silent abstention on a deployed box.
    card = load_card()
    known = registered_assets()
    for name, entry in card["detectors"].items():
        for field in ("corpus", "trained_on"):
            if entry.get(field) is not None:
                assert entry[field] in known, f"{name}.{field}"
```

- [x] **Step 3: Check what the gate now does**

```bash
python3 - <<'PY'
from dfd.service.evidence import gated_detectors, load_card
print(sorted(gated_detectors(load_card())))
PY
```

Expected: `['gend_clip']` if the measurement cleared the floor, `[]` otherwise. **Both are correct
outcomes of this plan.** If it is `[]`, the service keeps abstaining, which spec §9.2 says ships
safely, and the honest product is a human-review flagger.

- [x] **Step 4: Start the service and read a real verdict**

```bash
python3 -m dfd.service --evidence bench/evidence_card.json \
    --calibration assets/models/calibration.json &
curl -sS -X POST --data-binary @some_face.jpg 'http://127.0.0.1:8077/api/scan?filename=x.jpg'
curl -sS http://127.0.0.1:8077/api/submissions/<id> | python3 -m json.tool
```

Expected: the startup WARN lines for slots A, C and E remain (correct — they are unmeasured or
below the floor), and slot D either appears in the record with a non-zero `llr` or carries an
explicit abstention reason. **Paste the actual output into the commit message.**

- [x] **Step 5: Commit**

```bash
git add bench/evidence_card.json tests/service/test_evidence.py
git commit -m "feat: the card records slot D, and the gate decides without being asked twice"
```

**The gate stayed shut, which this plan says is a correct outcome.** `gated_detectors(load_card())`
returns `[]`. Slot D's headline is the controlled in-domain number — 0.663 session AUC over 125
capture sessions — because that is the number measured on `vkyc_capture_corpus`, the corpus the
card names. The worst LOGO fold, 0.738, sits in the note beside its permutation null (0.449–0.561,
48 shuffles, p = 0.020): outside the null, so there is real signal, and still 0.012 under the
floor. Both numbers are below 0.75, so the choice of headline does not change the verdict. The
in-dataset 0.928 appears only to be ruled out.

**Step 2 added no test.** `tests/service/test_evidence.py` already carries
`test_the_committed_card_names_only_registered_assets`, which asserts exactly what this step's
snippet asserts, over the same two fields with the same `None` skip. A second copy would be
duplication, not coverage.

**What Step 4 showed that the gate alone did not.** Slot D *ran*: encoder and head loaded, a face
was found, and it produced `raw_score 0.266`. It then abstained with `uncalibrated_for_band` on a
`reject`-band frame — not because Task 9 failed to fit that band, which it did fit, but because the
service discarded the curve at startup: *"discarded the calibration curve for gend_clip: it has
not been measured above the evidence floor."* A fitted curve is not permission to speak; permission
comes from measured cross-corpus AUC. Task 9's work is therefore correct and currently inert, and
it becomes live the moment a measurement clears the floor — not before.

On the same frame `blend_seam` abstained with `below_quality_floor` where slot D ran. That is
Task 8's measured floor visible in production: slot D's floor is `reject` because anything higher
discards two thirds of the swapped frames.

---

### Task 11: The product surface — the RD-shaped contract — **DONE 2026-11-11** (`f0632a6`)

Spec §5.3: an upload endpoint returning overall verdict and score **plus the per-detector list**,
with slots A, C and E present as abstentions carrying their raw scores. A single number is weaker
than a number plus the four opinions behind it, and `fuse` already escalates hard disagreement to
`OUT_OF_DISTRIBUTION`, which is the right answer for an input unlike anything measured.

**Files:**
- Create: `src/dfd/service/contract.py`
- Create: `tests/service/test_contract.py`
- Modify: `src/dfd/service/api.py`
- Modify: `tests/service/test_api.py`

**Interfaces:**
- Produces: `def rd_response(record: AuditRecord) -> dict[str, Any]` returning
  `{"verdict": "AUTHENTIC"|"MANIPULATED"|"UNKNOWN", "score": float, "models": [{"name", "verdict",
  "score", "abstained", "reason"}], "dissenting_models": [str], "quality_band": str,
  "schema_version": str}`.

- [x] **Step 1: Write the failing tests**

```python
def test_maps_the_three_verdicts_onto_the_product_vocabulary():
    assert rd_response(_record(verdict=Verdict.REAL))["verdict"] == "AUTHENTIC"
    assert rd_response(_record(verdict=Verdict.FAKE))["verdict"] == "MANIPULATED"
    for v in (Verdict.INSUFFICIENT_EVIDENCE, Verdict.OUT_OF_DISTRIBUTION):
        assert rd_response(_record(verdict=v))["verdict"] == "UNKNOWN"


def test_abstaining_detectors_appear_with_their_reason_not_omitted():
    # Omitting them would turn a four-model ensemble with three abstentions
    # into what looks like a one-model system that agreed with itself.
    out = rd_response(_record_with_abstentions())
    assert {m["name"] for m in out["models"]} == {"blend_seam", "effnet_b4",
                                                  "gend_clip", "npr"}
    assert any(m["abstained"] and m["reason"] == "weights_absent"
               for m in out["models"])


def test_an_abstaining_detector_is_never_a_dissenter():
    # Silence is not disagreement. Counting it as dissent would make a
    # correctly-gated deployment look maximally conflicted.
    out = rd_response(_record_with_abstentions())
    assert all(name not in out["dissenting_models"]
               for name in ("npr", "effnet_b4"))


def test_the_score_is_the_recorded_posterior_and_nothing_recomputed():
    rec = _record(posterior=0.8123)
    assert rd_response(rec)["score"] == pytest.approx(0.8123)


def test_out_of_distribution_is_not_flattened_into_authentic():
    # The honest answer for an unfamiliar generator. Mapping it to AUTHENTIC
    # would turn "I have never seen anything like this" into "this is fine".
    out = rd_response(_record(verdict=Verdict.OUT_OF_DISTRIBUTION))
    assert out["verdict"] == "UNKNOWN"
```

And in `tests/service/test_api.py`:

```python
def test_rd_format_is_served_for_a_scored_submission(server):
    sid = server.worker.submit_bytes(_JPEG_BYTES, filename="f.jpg", source="api")
    server.worker.process_one()
    status, body = _get(server, f"/api/submissions/{sid}?format=rd")
    assert status == 200
    assert set(body) >= {"verdict", "score", "models", "dissenting_models"}
    assert body["verdict"] in {"AUTHENTIC", "MANIPULATED", "UNKNOWN"}


def test_the_plain_format_is_unchanged_by_the_new_parameter(server):
    # Everything already reading /api/submissions/<id> must keep seeing the
    # audit record. A format parameter that quietly reshapes the default
    # response is a breaking change wearing an opt-in's clothes.
    sid = server.worker.submit_bytes(_JPEG_BYTES, filename="f.jpg", source="api")
    server.worker.process_one()
    status, body = _get(server, f"/api/submissions/{sid}")
    assert status == 200 and "record" in body and "models" not in body


def test_rd_format_on_an_unscored_submission_says_so_rather_than_guessing(server):
    # A queued submission has no verdict. Returning UNKNOWN would be
    # indistinguishable from "scored, and nothing could be concluded", which
    # is a completely different statement about the same image.
    sid = server.worker.submit_bytes(_JPEG_BYTES, filename="f.jpg", source="api")
    status, body = _get(server, f"/api/submissions/{sid}?format=rd")
    assert status == 409
    assert body["status"] == "queued"


def test_rd_format_on_an_unknown_id_is_still_a_404(server):
    status, _ = _get(server, "/api/submissions/nope?format=rd")
    assert status == 404


def test_analyze_scores_synchronously_and_returns_the_contract(server):
    status, body = _post(server, "/api/analyze?filename=f.jpg", _JPEG_BYTES)
    assert status == 200
    assert set(body) >= {"verdict", "score", "models", "dissenting_models"}


def test_analyze_enforces_the_same_upload_limit_as_scan(server):
    status, body = _post(server, "/api/analyze?filename=f.jpg",
                         b"x" * (server.max_upload_bytes + 1))
    assert status == 413


def test_analyze_refuses_chunked_like_scan_does(server):
    # Reused, not re-derived: an /api/analyze that reads a chunked body as
    # length 0 accepts an EMPTY submission and reports a decode failure about
    # content the caller never finished sending.
    status, _ = _post(server, "/api/analyze", b"", chunked=True)
    assert status == 411
```

`_JPEG_BYTES` is a small valid JPEG built once in the module (OpenCV-encoded from a synthetic
face-like array); `_post` is the existing helper's POST sibling — add it beside `_get` rather than
inlining `urllib` calls in seven tests.

- [x] **Step 2: Run and watch fail**

Run: `python3 -m pytest tests/service -q`

- [x] **Step 3: Write `contract.py` and wire the routes**

`rd_response` is a pure function over `AuditRecord` — no I/O, no registry, no clock — so it is
testable without a service. `dissenting_models` are the non-abstaining detectors whose per-detector
LLR sign disagrees with the fused verdict's direction; abstentions never count as dissent.

In `api.py`: `?format=rd` on the existing `/api/submissions/<id>` route, returning 409 with the
current status when the record is not ready; and `POST /api/analyze`, which is `/api/scan`'s body
handling (the chunked refusal, the Content-Length check, the size limit, the short-read check —
**reuse that code, do not re-derive it**) followed by a synchronous `decide` inside the existing
concurrency slot. Document in the module docstring that `/api/analyze` holds a worker slot for the
duration of a CPU forward pass through a 304M-parameter encoder, with Task 6's measured number, so
nobody deploys it behind a 2-second timeout by accident.

- [x] **Step 4: Run the tests and gates**

Run: `python3 -m pytest -q && ruff check src bench corpora training && mypy --config-file mypy.ini`

- [x] **Step 5: Commit**

```bash
git add src/dfd/service/contract.py src/dfd/service/api.py \
        tests/service/test_contract.py tests/service/test_api.py
git commit -m "feat: the product contract, with the three abstentions still visible in it"
```

**What the contract actually says on this deployment.** Live, `POST /api/analyze` with a real
FairFace frame: `verdict UNKNOWN`, `score 0.5`, `quality_band reject`, `dissenting_models []`, and
four models all abstaining — `blend_seam` `below_quality_floor`, `effnet_b4` `weights_absent`,
`gend_clip` `uncalibrated_for_band` carrying `raw_score 0.2558`, `npr` the same carrying `0.8964`.
The abstentions are not padding around the answer; they *are* the answer, which is why they carry
their raw scores.

**Two judgement calls the plan left open.**
- A fused `llr_total` of exactly 0.0 has no sign, so the dissent test cannot be applied. Where that
  zero was *made of* non-zero opinions (+2 and −2 cancelling), it is not consensus — it is the most
  conflicted record this system can produce, and every contributor is named. An all-silent record
  still reports no dissenters, because silence is not disagreement.
- `?format=` with an unrecognised value is a 400 rather than an ignored parameter. Answering
  `?format=rd2` with the audit record would read to a caller as "this build serves rd2".

**One correction to the plan's interface.** It specifies `rd_response(record: AuditRecord)`, but the
route serving it reads finished records back out of SQLite as JSON and never reconstructs the
dataclass. A dataclass-only signature would have passed every test in this task and failed the one
route it exists for, so it accepts both forms and a test pins that they agree.

**Measured cost of the synchronous route**: 6.3 s end to end for a single frame on this
workstation. `api.py`'s docstring now carries Task 6's per-face numbers (276 ms on the deployment
instance, 1,637 ms here) so nobody puts `/api/analyze` behind a 2-second gateway timeout.

---

### Task 12: Deploy, and write down what the next person needs — **DONE 2026-11-11** (`3ed1600`)

**Files:**
- Modify: `ops/deploy-ec2.sh`
- Modify: `docs/SERVICE.md`
- Modify: `docs/HANDOFF.md` (a new §0 block)
- Modify: `README.md`

- [x] **Step 1: Teach the deploy script about the new weights**

`assets/models/clip_vit_l14_visual.pt` is ~1.2 GB, which dwarfs everything the script currently
ships and inverts its own stated reasoning about `dima806` (328 MB "would be 89% of the transfer
for a detector that contributes nothing"). Update the WHAT IS DELIBERATELY NOT SHIPPED block
honestly: the encoder **is** shipped because slot D cannot score without it, `dima806` still is
not, and a fresh instance should prefer `ops/fetch-assets.sh` plus `training/convert_clip.py` run
**on the instance** over pushing 1.2 GB up a home uplink. Add a `--fetch-remote` path that does
exactly that, and keep the rsync route for an instance with no outbound access.

- [x] **Step 2: Deploy and verify end to end**

```bash
./ops/deploy-ec2.sh --host <host> --key key.pem --dry-run
./ops/deploy-ec2.sh --host <host> --key key.pem
ssh -i key.pem -L 8077:127.0.0.1:8077 ubuntu@<host>
curl -sS http://127.0.0.1:8077/health
curl -sS -X POST --data-binary @face.jpg 'http://127.0.0.1:8077/api/analyze?filename=face.jpg'
```

Expected: the contract shape from Task 11, with a real per-model list. **Record the wall-clock
latency of that call** and check it against Task 6's measurement on the same box.

- [x] **Step 3: Write the §0 resume block in `docs/HANDOFF.md`**

Following the house style — headline first, the number and its control, then what is NOT true.
It must say: what slot D measured on each corpus; whether the gate opened; what the permutation
null and the two shortcut controls did; what the quality-floor table said; the measured CPU
latency; whether Phase 2 is runnable here or blocked on the GPU decision; and that the
biometric-consent blocker (owner-action item 6) is untouched by all of it.

- [x] **Step 4: Update `docs/SERVICE.md` and `README.md`**

The endpoint list gains `/api/analyze` and `?format=rd`. The README's evidence-card paragraph
gains slot D. **No parity claim appears in either** (spec §4.3).

- [x] **Step 5: Commit**

```bash
git add ops/deploy-ec2.sh docs/SERVICE.md docs/HANDOFF.md README.md
git commit -m "feat: slot D reaches the instance, and the handoff says what it actually did"
```

**Deployed and verified.** `uptime_s` went 108,887.4 → 21.2, which is the only check worth making
here: this service answers HTTP 200 from whatever process is already running, so a stale deploy and
a good one are indistinguishable from outside. `/api/analyze` on the instance, one real FairFace
frame, returned the Task 11 contract with all four detectors abstaining and **2.119 s** wall clock.
Against Task 6's 276 ms for the encoder alone, the CLIP forward is ~13% of that call and face
detection plus the other three detectors are the rest — consistent, not a contradiction.
`gend_clip`'s raw score was byte-identical to the local run on the same file.

**`--fetch-remote` added.** It excludes the 1.2 GB encoder and the 890 MB OpenAI archive from the
rsync and runs `ops/fetch-assets.sh` + `training/convert_clip.py --verify` on the instance instead.
Both steps skip when their output already exists, so a redeploy does not re-download. The rsync
route stays the default for an instance with no outbound access. `--delete-excluded` remains
deliberately absent, so an encoder already on the instance survives an excluded sync — which this
deploy confirmed.

**The not-shipped block was carrying arithmetic the tree had outgrown.** It argued dima806's 328 MB
"would be 89% of the transfer for a detector that contributes nothing". With slot D's encoder
present that share is closer to 10%. Both statements were true when written; the block now says so
rather than quietly keeping the stale one.

**The handoff block is deliberately NOT numbered §0.** Nineteen places in `docs/HANDOFF.md` say
"§0" meaning the 2026-09-25 resume block, and four say "§0a" meaning the one below it. Renumbering
to put the new block at §0 silently repoints every one of them — I did it, caught the breakage by
chasing one quoted line (`§0a`'s "7 positives is not a training set", which lives in the block that
would have moved), and reverted the whole cascade. The diff to that file is now purely additive:
109 insertions, 0 deletions, every existing anchor intact. A document this size has its section
numbers as load-bearing references, not as ordering.

---

### Task 13 (Phase 2, gated on Task 6): LayerNorm tuning — **IN FLIGHT 2026-11-11**

**Gate check: Task 6 said run it.** `bench/clip_latency.md`'s ruling is "Phase 2 IS runnable here
and is NOT blocked on the GPU decision", so this task is not blocked on owner-action item 7.

**Status.** Steps 1–4 are done and committed (`35e93bb`, `a428878`); Step 5 — re-measure and
re-report — is waiting on a 5-epoch run live on the deployment instance since 18:12 UTC. Eight
tests cover the fitter, and the 102,400-parameter / 0.034% claim is verified against the real
ViT-L/14, not only against the tiny test tower. A chain script on the instance
(`~/dfd/lntune_chain.sh`) fires Tasks 7, 8 and 9 against the tuned pair the moment the fit exits,
writing `bench/gend_report_lntuned.{json,md}`, `bench/gend_vcip_lntuned.json` and
`bench/calibration_report_lntuned.json`. **No Phase 1 artefact is overwritten by any of it** —
including `assets/models/calibration.json`, whose default the chain redirects.

**The 3x rule of thumb was measured before the run, as this plan's own Task 6 write-up asked.**
A real step — forward, cross-entropy, backward, AdamW, LayerNorms only — is **617 ms/image** at
batch 8 on the instance, 26% under the 829 ms that forward-times-three implied, because only
102,400 of 303,966,208 parameters carry a weight gradient. One epoch is **1.70 h**, not 2.3 h, so
the 5-epoch run is ~9.3 h including per-epoch holdout scoring. 8 threads buys 2% and was declined
to keep the run in the configuration every other number in `bench/clip_latency.md` was measured
under.

**Three deviations from the reference code below, each because transcribing it faithfully would
have shipped a defect** (the same lesson as Task 9's):

1. `tune_layernorms` returns a third value, a `TuningTrace`. The 2-tuple below loses the per-epoch
   holdout curve, which is the only evidence that early stopping did anything; a 9-hour run whose
   shape cannot be inspected afterwards is a run somebody repeats.
2. `ENCODER_ID_LNTUNED` is derived from a shared `encoder_cache_id()` helper rather than being the
   bare string `"clip-l14-lntuned"`. The id scheme already in the code is `"clip-l14:<filename>"`,
   and the constant below would have been a second, conflicting authority for the same key.
3. `--out` no longer defaults to the SHIPPED head. Under `--tune-layernorms` it defaults to
   `assets/models/gend_clip_lntuned.pt`, because one overnight run finishing with the old default
   would have overwritten the only slot D artefact that is currently measured.

**And one defect this task's own work surfaced, now gated.** A `logger.info` here carried seven
format placeholders and six arguments. `logging` catches that `TypeError` itself and prints a
traceback to stderr **without raising**, so the run continued, every test passed, and the only
symptom was a log line that never appeared. `PLE1205`/`PLE1206` joined the ruff gate; both were
clean across `src`, `bench`, `corpora`, `training` and `tests` when added.


**Do not start this task until Task 6's arithmetic is written down and says it can run.** If it
says otherwise, this task is blocked on owner-action item 7 ("Decide GPU or no GPU") and the plan
is complete without it: Phase 1 ships a measured detector, and `bench/clip_latency.json` is the
evidence for why Phase 2 waits.

**The design, so it is not hand-waved when it does run.** Tuning touches
`CLIPVisualTower.layernorm_parameter_names()` and nothing else — 102,400 parameters out of
~304M, 0.03%. The fitted output is written as a **complete tuned encoder state_dict** at
`assets/models/clip_vit_l14_lntuned.pt`, not as a delta file. That is deliberate: a complete
state_dict loads through `loading.load_model`'s strict secure path with the existing
`CLIPVisualTower.vit_l14` factory and **no loading machinery changes at all** — the detector simply
has its `encoder_path` pointed at the tuned file. A delta would need a partial, non-strict load,
which is the kind of special case that eventually loads half a model without complaining.

**Files:**
- Modify: `training/fit_gend.py` (`--tune-layernorms`, `--epochs`, `--lr`, `--encoder-out`)
- Modify: `tests/test_fit_gend.py`
- Modify: `assets/manifest.yaml` (`gend_clip_encoder_lntuned` — dual provenance: MIT weights with
  CC BY 4.0 fitted modifications, so the stricter attribution condition governs)
- Modify: `bench/eval_swaps.py` (a `TRAINING_WINDOWS` entry for the new version string)

**Interfaces:**
- Produces:
  - `def tune_layernorms(tower: CLIPVisualTower, records: Sequence[dict], *, epochs: int, lr: float, batch_size: int = 16, seed: int = 0) -> tuple[CLIPVisualTower, GendHead]`
    — takes the Task 5 records (pixels, because a cached embedding cannot be back-propagated
    through), returns the tuned tower and the head trained jointly with it.
  - `ENCODER_ID_LNTUNED: str = "clip-l14-lntuned"` — the `FeatureCache` id for the tuned encoder.

- [ ] **Step 1: Write the failing tests**

```python
def test_tuning_touches_layernorms_and_nothing_else():
    before = {k: v.clone() for k, v in CLIPVisualTower.tiny().state_dict().items()}
    tuned = tune_layernorms(CLIPVisualTower.tiny(), features_loader, epochs=1, lr=1e-3)
    ln = set(tuned.layernorm_parameter_names())
    after = tuned.state_dict()
    assert any(not torch.equal(after[k], before[k]) for k in ln)
    assert all(torch.equal(after[k], before[k]) for k in after if k not in ln)


def test_the_tuned_encoder_loads_through_the_same_strict_factory(tmp_path):
    tuned = tune_layernorms(CLIPVisualTower.tiny(), ..., epochs=1, lr=1e-3)
    torch.save(tuned.state_dict(), tmp_path / "e.pt")
    CLIPVisualTower.tiny().load_state_dict(
        torch.load(tmp_path / "e.pt", map_location="cpu", weights_only=True))


def test_a_tuned_encoder_gets_its_own_feature_cache_id():
    # Sharing the frozen encoder's cache would serve embeddings from the
    # wrong model and the only symptom would be a disappointing AUC.
    assert FeatureCache(root, encoder_id="clip-l14").key(img) != \
           FeatureCache(root, encoder_id="clip-l14-lntuned").key(img)
```

- [ ] **Step 2–5: run them failing, implement, re-measure, re-report**

Implementation: freeze every parameter, unfreeze the LayerNorm set, train head and LayerNorms
jointly with AdamW over the Task 5 corpus, early-stop on the subject-disjoint holdout. Then
**re-run Tasks 7, 8 and 9 unchanged against the tuned encoder** — a new version string, a new
`TRAINING_WINDOWS` entry, a new feature-cache id, a fresh permutation null, fresh controls. The
comparison to report is tuned versus frozen on the **worst LOGO fold**, not on the mean and not
in-dataset. If LN-tuning does not beat the frozen probe on that number, Phase 1's weights stay
shipped and the result is recorded as measured.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: the LayerNorms are tuned, and the worst fold says whether it was worth it"
```

---

## Self-Review

**Spec coverage.** §2 item 1 → Tasks 1, 3, 13. §2 item 2 (as corrected in Task 0) → Task 5.
§2 item 3 (two or more generators) → Task 5 supplies five. §2 item 4 (a measurement with a
permutation null, into the evidence card) → Tasks 7, 8, 10. §2 item 5 (calibration per band) →
Task 9. §2 item 6 (an image upload endpoint returning the per-model list) → Task 11. §2 item 7
(`commercial_use: true` per asset, licence read at origin) → Tasks 2, 5, 8. §4.1's "method not
checkpoint" → Tasks 1 and 2 fetch only OpenAI's MIT weights, and nothing in the plan touches
`yermandy/GenD_CLIP_L_14`. §4.2 → Task 13's 102,400-parameter set. §4.3 → the Global Constraints
and Task 12 step 4. §5.1 (secure loading, measured quality floor, hash-pinned encoder) → Tasks 2,
3, 8. §5.2 (nothing else changes) → held: `fuse`, `Calibrator`, `decide`, `build_audit_record`,
the policy and the worker are untouched. §5.3 → Task 11. §6.1–6.3 → Task 5. §6.4 → Task 8. §7's
five conditions → Task 7, each a named output. §8 (CPU benchmark before any latency claim) →
Task 6. §9.1 → the manifest entries in Tasks 2, 5, 8 record every blocker including the
unresolvable one. §9.2's six failure modes → Task 7 step 7 (inside the null), Task 8 (domain
shift, seam transfer), Task 7's LOGO (families too similar), SFHQ held out as a control (Task 7),
Task 6 (latency). §10 → Task 10.

**Known gaps, stated rather than hidden:**

1. **The spec's §6.2 table does not mention `corpora/swaps.py`.** It lists self-blends, SFHQ and
   the inswapper swaps. The four composite families landed on 2026-11-09, after the design was
   drafted, and they are licence-clean FairFace derivatives — which is why this plan trains on
   five families rather than one. The spec is not wrong, it is behind; Task 5 is where the
   difference lives and it should be folded back into §6.2 when the plan is executed.
2. **The GPU question is unresolved and this plan does not resolve it.** Task 6 turns it from an
   opinion into arithmetic, which is as far as engineering can take it.
3. **The biometric-consent blocker is untouched.** No task here can close it, and Task 8 says so
   in the manifest entry rather than quietly proceeding.
4. **`--verify` in Task 2 depends on OpenAI's archive still being served at the pinned digest.**
   If it is not, the fallback is named (HuggingFace, hash measured locally) but the architecture
   oracle is lost and Task 1's constants become unverified. That would be worth saying out loud in
   the evidence card.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-11-10-slot-d-gend-clip.md`.
