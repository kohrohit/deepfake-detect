"""White-box adversarial baseline (spec §3A, acceptance criterion 8).

The threat model assumes the adversary holds our weights. Under that assumption
a gradient attack against any differentiable detector is not a risk — it is the
expected case. Measuring it is what turns "state-sponsored threat model" from a
sentence in a document into a number in a report.

A detector whose adversarial TPR collapses is not thereby useless. It is
demoted from decider to evidence contributor (spec §3A.4).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .metrics import tpr_at_fpr


def pgd_attack(model, x: torch.Tensor, y: torch.Tensor, eps: float = 0.03,
               alpha: float = 0.01, steps: int = 10,
               seed: int = 0) -> torch.Tensor:
    """Projected gradient descent within an L-inf ball, clamped to [0, 1].

    The random start is what distinguishes PGD from iterative FGSM, and it is
    drawn from a local `torch.Generator` so that attacking does not perturb the
    global RNG state the rest of the suite draws from.
    """
    x = x.detach()
    if eps == 0.0:
        return x.clone()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    start = (torch.rand(x.shape, generator=generator) * 2.0 - 1.0) * eps
    adv = torch.clamp(x + start.to(x.device), 0.0, 1.0).detach()

    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        (grad,) = torch.autograd.grad(loss, adv)
        with torch.no_grad():
            adv = adv + alpha * grad.sign()
            adv = torch.clamp(adv, x - eps, x + eps)
            adv = torch.clamp(adv, 0.0, 1.0)
        adv = adv.detach()
    return adv


def adversarial_tpr(model, x: torch.Tensor, y: torch.Tensor, eps: float,
                    fpr: float = 0.01, alpha: float | None = None,
                    steps: int = 10, seed: int = 0) -> float:
    """TPR@FPR after attacking only the positives (the adversary's goal).

    Negatives are left clean: a fraudster wants fakes to read as real, not the
    reverse, and attacking negatives too would move the threshold and
    understate the detector.
    """
    alpha = alpha if alpha is not None else max(eps / 4.0, 1e-4)
    pos = y == 1
    adv = x.clone()
    if eps > 0 and bool(pos.any()):
        adv[pos] = pgd_attack(model, x[pos], y[pos], eps=eps, alpha=alpha,
                              steps=steps, seed=seed)
    with torch.no_grad():
        scores = torch.softmax(model(adv), dim=1)[:, 1].cpu().numpy()
    return tpr_at_fpr(scores, y.cpu().numpy(), fpr)
