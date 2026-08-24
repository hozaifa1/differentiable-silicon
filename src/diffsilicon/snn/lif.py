"""T4's engine: a surrogate-gradient LIF network whose HARDWARE is differentiable.

The point of this module is not the classifier. It is that the five numbers the
device hands up -- beta, g_min, g_max, th_th, sig_w -- enter the network as real
tensors with `requires_grad=True`, so `torch.autograd.grad(L, phi)` returns a
genuine dL/dphi that then crosses the wire into JAX. Nothing here is a stand-in
for the device; the device is upstream and this is what consumes it.

Weight mapping, reparameterised so the gradient reaches sig_w:

    W~_ij = g_min + (g_max - g_min) * sigmoid(W_ij) + sig_w * eps_ij,  eps ~ N(0,1)

Neuron: LIF with soft reset. The spike is a Heaviside forward and the DERIVATIVE
OF A STATED SMOOTH RELAXATION backward -- see `soft_heaviside`. Bounded, so it does
not blow up when the threshold itself moves during co-design.

Two networks live here.

`LIFNet` is the D1 two-layer classifier. It is kept because the liveness tests,
the gradient checks and the cheap Tier A path all run on it, and because it is
the smallest thing that exercises the PyTorch <-> JAX boundary.

`snn.lsnn.LSNNNet` is the D3 recalibration: the SAME device coupling wired into
the thesis' verified LSNN topology -- delayed input synapses, a recurrent hidden
layer split between plain LIF and adaptive LIF neurons, a low-pass readout
filter. It lives in its own module and reuses `LIFNet._weights` unchanged.

`synthetic_batch` remains for tests that must not touch the dataset; the real
task is `diffsilicon.snn.ecg`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# float64 throughout. The five hyperparameters span beta ~ 0.6 down to
# g_min ~ 2.6e-5, and in float32 the loss differences that a gradient check probes
# land at machine epsilon and come back quantised to powers of two. Doubling the
# cost of a 2-layer LIF on 32 samples is not a consideration.
DTYPE = torch.float64

__all__ = [
    "SurrGradSpike",
    "soft_heaviside",
    "LIFNet",
    "synthetic_batch",
    "balanced_ce",
    "PHI_KEYS",
]

# Order is load-bearing: it fixes the layout of every cotangent crossing the wire.
PHI_KEYS = ("beta", "g_min", "g_max", "th_th", "sig_w")


SURR_K = 2.0


def soft_heaviside(u):
    """The smooth relaxation of the spike, stated explicitly rather than implied.

        H_k(u) = 1/2 * (1 + k u / (1 + k|u|)),   H_k'(u) = (k/2) / (1 + k|u|)^2

    This matters because it makes the surrogate gradient DERIVED rather than
    asserted: the backward pass below is exactly H_k', so the training gradient is
    the true gradient of a network that is written down, not a plausible-looking
    stand-in for the gradient of a network that has none. It also gives the
    organizers' `check-gradients` something it can legitimately verify -- finite
    differences of a Heaviside are either exactly zero or one spike-flip large, so
    a hard forward pass cannot be gradient-checked at all, by anyone.
    """
    return 0.5 * (1.0 + SURR_K * u / (1.0 + SURR_K * u.abs()))


class SurrGradSpike(torch.autograd.Function):
    """Heaviside forward, soft-Heaviside derivative backward."""

    k = SURR_K

    @staticmethod
    def forward(ctx, u):
        ctx.save_for_backward(u)
        return (u > 0.0).to(u.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (u,) = ctx.saved_tensors
        return grad_out * (0.5 * SurrGradSpike.k) / (1.0 + SurrGradSpike.k * u.abs()) ** 2


spike_fn = SurrGradSpike.apply


class LIFNet(torch.nn.Module):
    """Two-layer LIF classifier parameterised by the five device numbers.

    beta, g_min, g_max, th_th and sig_w arrive as tensors, not floats. Every one
    of them is on the autograd graph, which is what makes the VJP endpoint real.
    """

    def __init__(self, n_in: int, n_hidden: int, n_out: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w1 = torch.nn.Parameter(torch.randn(n_in, n_hidden, generator=g, dtype=DTYPE) * 0.5)
        self.w2 = torch.nn.Parameter(torch.randn(n_hidden, n_out, generator=g, dtype=DTYPE) * 0.5)
        self.n_out = n_out

    @staticmethod
    def _weights(w_logit, g_min, g_max, sig_w, noise):
        """Physical conductance weights, normalised by the conductance window.

        Note what this does NOT do: g_min and g_max cancel exactly, so
        dL/dg_min and dL/dg_max are ~1e-13 rather than merely small. That is
        correct, not a detached gradient. A synapse programmable anywhere in
        [g_min, g_max] has the same usable weight range whatever the window is;
        the window earns its place through sig_w -- which is already defined
        relative to it, as dg/dV_th * sigma_Vth / (g_max - g_min) -- and through
        th_th, which goes as 1/g_max. Adding a direct dependence here would
        double-count the memory window, and it would inflate every gradient
        attribution in the writeup.
        """
        span = g_max - g_min
        w_phys = g_min + span * torch.sigmoid(w_logit) + sig_w * span * noise
        return (w_phys - g_min) / span

    def forward(self, x, phi: dict[str, torch.Tensor], seed: int = 0, smooth: bool = False):
        """x: (B, T, n_in) spike trains. Returns (logits, spike_count).

        `smooth=True` replaces the Heaviside with the soft-Heaviside it is the
        relaxation of, making the whole network differentiable in the ordinary
        sense. Used for gradient checking, never for reported results.
        """
        fire = soft_heaviside if smooth else spike_fn
        beta, g_min, g_max = phi["beta"], phi["g_min"], phi["g_max"]
        th_th, sig_w = phi["th_th"], phi["sig_w"]

        g = torch.Generator().manual_seed(seed)
        n1 = torch.randn(self.w1.shape, generator=g, dtype=DTYPE)
        n2 = torch.randn(self.w2.shape, generator=g, dtype=DTYPE)
        w1 = self._weights(self.w1, g_min, g_max, sig_w, n1)
        w2 = self._weights(self.w2, g_min, g_max, sig_w, n2)

        B, T, _ = x.shape
        # th_th is spikes-to-fire at max weight; the membrane is measured in the
        # same units, so the threshold the neuron compares against is th_th itself.
        u1 = torch.zeros(B, w1.shape[1], dtype=DTYPE)
        u2 = torch.zeros(B, self.n_out, dtype=DTYPE)
        out = torch.zeros(B, self.n_out, dtype=DTYPE)
        spikes = x.new_zeros(())

        for t in range(T):
            i1 = x[:, t] @ w1
            u1 = beta * u1 + i1
            s1 = fire(u1 - th_th)
            u1 = u1 - s1 * th_th  # soft reset
            spikes = spikes + s1.sum()

            i2 = s1 @ w2
            u2 = beta * u2 + i2
            s2 = fire(u2 - th_th)
            u2 = u2 - s2 * th_th
            spikes = spikes + s2.sum()
            out = out + u2

        return out / T, spikes / (B * T)


def balanced_ce(logits, targets, n_classes: int):
    """Class-balanced cross-entropy: the objective actually optimised.

    MIT-BIH inter-patient DS2 is ~90% class N. Plain cross-entropy would be
    optimised by predicting N and nothing else, and every reported improvement
    would be noise around that fixed point.
    """
    counts = torch.bincount(targets, minlength=n_classes).to(logits.dtype)
    weight = torch.where(counts > 0, counts.sum() / (n_classes * counts.clamp(min=1)),
                         torch.zeros_like(counts))
    return F.cross_entropy(logits, targets, weight=weight)


def synthetic_batch(batch: int = 32, T: int = 24, n_in: int = 16, n_classes: int = 4, seed: int = 0):
    """A deliberately imbalanced, linearly separable spike dataset.

    Stands in for MIT-BIH DS1/DS2 (which lands on D4) so the PyTorch <-> JAX
    boundary and the balanced-CE objective are testable without a download. Class
    frequencies are skewed 8:2:1:1 for the same reason the real problem is.
    """
    g = torch.Generator().manual_seed(seed)
    freq = torch.tensor([8.0, 2.0, 1.0, 1.0])[:n_classes]
    y = torch.multinomial(freq / freq.sum(), batch, replacement=True, generator=g)
    rate = torch.full((batch, n_in), 0.10)
    for c in range(n_classes):
        lo, hi = c * n_in // n_classes, (c + 1) * n_in // n_classes
        rate[y == c, lo:hi] = 0.55
    x = (torch.rand(batch, T, n_in, generator=g) < rate.unsqueeze(1)).to(DTYPE)
    return x, y
