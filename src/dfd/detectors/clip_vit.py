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
        mlp_out: torch.Tensor = self.mlp(self.ln_2(x))
        return x + mlp_out


class _Transformer(nn.Module):
    """Holds the block stack at CLIP's own `transformer.resblocks` path.

    A bare `nn.Module` with the attribute assigned onto it would produce
    the same state_dict keys, but reads back through
    `nn.Module.__getattr__` as `Tensor | Module` -- so `mypy --strict`
    rejects calling it, and the only way through is an ignore on the
    call site. Declaring the type here keeps the key path and the
    annotation honest at once.
    """

    resblocks: nn.Sequential

    def __init__(self, width: int, heads: int, layers: int) -> None:
        super().__init__()
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads) for _ in range(layers)])


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
        self.transformer = _Transformer(width, heads, layers)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    @classmethod
    def vit_l14(cls) -> CLIPVisualTower:
        """The published ViT-L/14. Suitable as `load_model`'s model_factory."""
        return cls(**VIT_L14)

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
        encoded: torch.Tensor = self.transformer.resblocks(x)
        encoded = encoded.permute(1, 0, 2)                 # NLD
        pooled: torch.Tensor = self.ln_post(encoded[:, 0])
        projected: torch.Tensor = pooled @ self.proj
        return projected


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
