"""OpenAI's distributed CLIP archive -> a plain state_dict this repo can load securely.

Run once per machine, offline:

    ./ops/fetch-assets.sh                       # fetches and hash-verifies the archive
    python3 -m training.convert_clip \
        --archive assets/models/ViT-L-14.pt \
        --out assets/models/clip_vit_l14_visual.pt \
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

WHY THE ARCHIVE HAS TO BE REWRITTEN BEFORE IT WILL RUN. OpenAI traced it on a GPU in float16, so
`torch.device("cuda:0")` and `dtype=5` (half) are frozen into the graph as constants -- not as
anything `map_location="cpu"` can redirect. On a CPU-only box it raises "Found no NVIDIA driver"
from inside the TorchScript interpreter, and then, once that is fixed, a Half/Float type error
from the first convolution. `load_reference` below rewrites exactly those two kinds of constant,
which is what OpenAI's own `clip/clip.py` does on a CPU load (MIT, same repository as the
LICENSE the manifest cites). It touches the graph, never the weights: the state_dict read out of
the patched module is bit-identical to the one read before patching, apart from the float32 cast
this conversion performs anyway.
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

#: float16 in TorchScript's dtype enumeration. The archive was traced
#: in half precision on a GPU and carries this constant throughout.
HALF_DTYPE = 5


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


def _node_value(node: torch._C.Node, key: str) -> object:
    """A TorchScript constant's value, whatever type it was stored under."""
    return getattr(node, node.kindOf(key))(key)


def _graphs(module: object) -> list[torch._C.Graph]:
    """Every graph attached to a scripted module or method.

    `hasattr(module, "graph")` can itself raise inside TorchScript, which is
    why this is a try and not a conditional.
    """
    try:
        graphs = [module.graph] if hasattr(module, "graph") else []
    except RuntimeError:
        graphs = []
    if hasattr(module, "forward1"):
        graphs.append(module.forward1.graph)
    return graphs


def patch_device(module: object, device_node: torch._C.Node) -> None:
    """Repoint every frozen `cuda:*` constant at `device_node`'s device."""
    for graph in _graphs(module):
        for node in graph.findAllNodes("prim::Constant"):
            if ("value" in node.attributeNames()
                    and str(_node_value(node, "value")).startswith("cuda")):
                node.copyAttributes(device_node)


def patch_float(module: object, float_node: torch._C.Node) -> None:
    """Rewrite every `aten::to(..., half)` in the graph to float32.

    `5` is float16 in TorchScript's dtype enumeration. The dtype can be the
    second or the third argument to `aten::to` depending on the overload the
    trace captured, so both positions are checked.
    """
    for graph in _graphs(module):
        for node in graph.findAllNodes("aten::to"):
            inputs = list(node.inputs())
            for i in (1, 2):
                if _node_value(inputs[i].node(), "value") == HALF_DTYPE:
                    inputs[i].node().copyAttributes(float_node)


def load_reference(archive: Path) -> torch.jit.ScriptModule:
    """OpenAI's own module, rewritten to run on this CPU. Never in src/.

    Returns a module whose `encode_image` works and whose `state_dict` is the
    published one. Both callers want the same object, so the 890 MB archive is
    read once.
    """
    model = torch.jit.load(str(archive), map_location="cpu").eval()

    holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device("cpu")),
                             example_inputs=[])
    device_node = [n for n in holder.graph.findAllNodes("prim::Constant")
                   if "Device" in repr(n)][-1]
    model.apply(lambda m: patch_device(m, device_node))
    patch_device(model.encode_image, device_node)

    float_holder = torch.jit.trace(lambda: torch.ones([]).float(),
                                   example_inputs=[])
    float_node = list(float_holder.graph.findNode("aten::to").inputs())[1].node()
    model.apply(lambda m: patch_float(m, float_node))
    patch_float(model.encode_image, float_node)
    model.float()
    return model


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

    reference = load_reference(args.archive)
    sd = visual_state_dict(reference.state_dict())
    tower = TOWER_FACTORY()
    tower.load_state_dict(sd)
    tower.eval()

    if args.verify:
        g = torch.Generator().manual_seed(0)
        x = torch.randn(2, 3, tower.image_size, tower.image_size, generator=g)
        with torch.no_grad():
            ours = tower(x)
            theirs = reference.encode_image(x).float()
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
