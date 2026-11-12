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
