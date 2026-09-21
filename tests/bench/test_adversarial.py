import pytest
import torch
import torch.nn as nn

from bench.adversarial import adversarial_tpr, pgd_attack


class TinyModel(nn.Module):
    """A trivially attackable linear detector: mean pixel above 0.5 => fake."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 8 * 8, 2)
        with torch.no_grad():
            self.fc.weight.copy_(torch.cat([
                -torch.ones(1, 192) / 192.0,
                torch.ones(1, 192) / 192.0]))
            self.fc.bias.copy_(torch.tensor([0.5, -0.5]))

    def forward(self, x):
        return self.fc(x.flatten(1))


@pytest.fixture
def model():
    m = TinyModel()
    m.eval()
    return m


def _batch(value, n=8):
    return torch.full((n, 3, 8, 8), value, dtype=torch.float32)


def _labelled():
    """Negatives at 0.45, positives at 0.55 — close enough that eps=0.2
    genuinely flips the ranking. A wider gap makes the attack unmeasurable."""
    x = torch.cat([_batch(0.45, 16), _batch(0.55, 16)])
    y = torch.cat([torch.zeros(16, dtype=torch.long),
                   torch.ones(16, dtype=torch.long)])
    return x, y


def test_pgd_output_stays_within_the_epsilon_ball(model):
    x = _batch(0.5)
    adv = pgd_attack(model, x, torch.ones(8, dtype=torch.long),
                     eps=0.03, alpha=0.01, steps=5)
    assert torch.max(torch.abs(adv - x)).item() <= 0.03 + 1e-6


def test_pgd_output_stays_in_valid_pixel_range(model):
    """y=0 at x=0.99: ascending the loss on class 0 drives pixels upward,
    so the iterate walks toward 1.0 and actually exercises the in-loop
    clamp rather than only the one-off clamp on the random start."""
    x = _batch(0.99)
    adv = pgd_attack(model, x, torch.zeros(8, dtype=torch.long),
                     eps=0.1, alpha=0.02, steps=5)
    assert adv.min().item() >= 0.0 and adv.max().item() <= 1.0


def test_pgd_output_stays_in_valid_pixel_range_at_the_lower_bound(model):
    """Mirrors the case above: y=1 at x=0.01 drives pixels downward toward
    0.0, pinning the other side of the in-loop clamp the same way."""
    x = _batch(0.01)
    adv = pgd_attack(model, x, torch.ones(8, dtype=torch.long),
                     eps=0.1, alpha=0.02, steps=5)
    assert adv.min().item() >= 0.0 and adv.max().item() <= 1.0


def test_pgd_actually_moves_the_input(model):
    """A no-op attack must not be able to reach the later assertions."""
    x = _batch(0.5)
    adv = pgd_attack(model, x, torch.ones(8, dtype=torch.long),
                     eps=0.1, alpha=0.02, steps=5)
    assert not torch.equal(adv, x)


def test_pgd_reduces_confidence_on_the_true_class(model):
    x = _batch(0.9)
    y = torch.ones(8, dtype=torch.long)
    before = torch.softmax(model(x), 1)[:, 1].mean().item()
    after = torch.softmax(model(pgd_attack(model, x, y, eps=0.2, alpha=0.05,
                                           steps=20)), 1)[:, 1].mean().item()
    assert after < before - 0.05


def test_pgd_is_deterministic_given_a_seed(model):
    x, y = _batch(0.7), torch.ones(8, dtype=torch.long)
    kw = dict(eps=0.1, alpha=0.02, steps=5)
    assert torch.equal(pgd_attack(model, x, y, seed=3, **kw),
                       pgd_attack(model, x, y, seed=3, **kw))


def test_pgd_seed_is_load_bearing(model):
    """Without a random start the seed is dead code and the determinism
    test above passes for any implementation, seeded or not."""
    x, y = _batch(0.7), torch.ones(8, dtype=torch.long)
    kw = dict(eps=0.1, alpha=0.02, steps=5)
    assert not torch.equal(pgd_attack(model, x, y, seed=3, **kw),
                           pgd_attack(model, x, y, seed=999, **kw))


def test_pgd_step_size_is_proportional_to_alpha(model):
    """`alpha` is a documented parameter; nothing else pins it. A single
    step from a shared random start differs by exactly the alpha
    difference on the pixels the projection does not clamp."""
    x, y = _batch(0.5), torch.ones(8, dtype=torch.long)
    kw = dict(eps=0.4, steps=1, seed=11)
    a = pgd_attack(model, x, y, alpha=0.01, **kw)
    b = pgd_attack(model, x, y, alpha=0.03, **kw)
    assert (a - b).abs().max().item() == pytest.approx(0.02, abs=1e-6)


def test_pgd_does_not_disturb_global_torch_rng(model):
    """The attack must not reseed the RNG every other test draws from."""
    torch.manual_seed(1234)
    expected = torch.randn(4)
    torch.manual_seed(1234)
    pgd_attack(model, _batch(0.5), torch.ones(8, dtype=torch.long),
               eps=0.1, alpha=0.02, steps=5, seed=7)
    assert torch.equal(torch.randn(4), expected)


def test_clean_tpr_is_perfect_on_this_fixture(model):
    """Anchors the collapse below: without this, 'attacked == 0.0' could
    mean the detector never worked."""
    x, y = _labelled()
    assert adversarial_tpr(model, x, y, eps=0.0, fpr=0.1) == 1.0


def test_attack_collapses_tpr_to_zero(model):
    """Acceptance criterion 8. Exact values, not `attacked <= clean` — that
    inequality is satisfied by an attack that does nothing at all."""
    x, y = _labelled()
    assert adversarial_tpr(model, x, y, eps=0.2, fpr=0.1) == 0.0


def test_adversarial_tpr_does_not_mutate_the_callers_input_in_place(model):
    """`adversarial_tpr` must not corrupt the caller's tensor, whatever it
    does internally. This does NOT establish that only positives are
    attacked — `adv = x.clone()` inside `adversarial_tpr` guarantees `x`
    itself is untouched regardless of which rows the attack later writes
    into `adv`. See test_only_the_positives_are_attacked for that."""
    x, y = _labelled()
    before = x.clone()
    adversarial_tpr(model, x, y, eps=0.2, fpr=0.1)
    assert torch.equal(x, before)


def test_only_the_positives_are_attacked(model):
    """The adversary wants fakes to read as real, not the reverse; attacking
    negatives too would understate the detector by moving the threshold, and
    spec §3A.4's demotion decision rides on that number. Spies on the scores
    the model is actually evaluated on, not on the caller's tensor — `x` is
    never mutated in place regardless of which rows get attacked, so
    asserting on `x` cannot distinguish positives-only from whole-batch."""
    seen = []

    class Spy(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, z):
            seen.append(z.detach().clone())
            return self.inner(z)

    x, y = _labelled()
    adversarial_tpr(Spy(model), x, y, eps=0.2, fpr=0.1)
    scored = seen[-1]  # the final whole-batch scoring pass
    assert torch.equal(scored[y == 0], x[y == 0])
    assert not torch.equal(scored[y == 1], x[y == 1])


def test_zero_epsilon_is_the_identity(model):
    x = _batch(0.5)
    assert torch.equal(
        pgd_attack(model, x, torch.ones(8, dtype=torch.long), eps=0.0), x)
