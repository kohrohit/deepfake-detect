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


def test_patch_float_rewrites_the_half_dtype_constant_to_float32():
    # The archive was traced in float16 and casts to half inside its graph,
    # which on a CPU raises "Input type (FloatTensor) and weight type
    # (HalfTensor) should be the same" from the first convolution.
    #
    # This asserts on the GRAPH, not on an output dtype, and deliberately so:
    # a toy trace carries a `Half(...)` annotation on its output value that
    # the interpreter honours regardless of the constant, so a dtype
    # assertion here would fail while the identical rewrite works on the real
    # archive. What patch_float promises is that the constant is rewritten.
    # That the rewrite is SUFFICIENT is what `--verify` proves, against the
    # 890 MB file, and nothing smaller can prove it.
    from training.convert_clip import HALF_DTYPE, patch_float

    traced = torch.jit.trace(lambda t: t.half(), example_inputs=(torch.ones(2),))

    def dtype_constant():
        node = traced.graph.findNode("aten::to")
        arg = list(node.inputs())[1].node()
        return getattr(arg, arg.kindOf("value"))("value")

    assert dtype_constant() == HALF_DTYPE

    holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
    float_node = list(holder.graph.findNode("aten::to").inputs())[1].node()
    patch_float(traced, float_node)

    assert dtype_constant() != HALF_DTYPE


def test_patch_device_leaves_a_cpu_only_graph_alone():
    # A cuda constant cannot be traced on this box, so this does NOT prove the
    # cuda rewrite works -- `--verify` against the real archive is what proves
    # that. What it does prove is that the walk is safe on a graph with
    # nothing to patch, which is the case every re-run hits.
    from training.convert_clip import patch_device

    traced = torch.jit.trace(lambda t: t * 2, example_inputs=(torch.ones(2),))
    before = traced(torch.ones(2)).clone()

    holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device("cpu")),
                             example_inputs=[])
    device_node = [n for n in holder.graph.findAllNodes("prim::Constant")
                   if "Device" in repr(n)][-1]
    patch_device(traced, device_node)

    assert torch.equal(traced(torch.ones(2)), before)
