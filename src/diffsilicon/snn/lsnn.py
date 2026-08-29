"""The thesis LSNN, wired to the device.

This is the D3 recalibration of T4. The network is no longer a two-layer LIF
classifier invented for this project; it is the architecture the thesis already
validated on this exact dataset, with the FeFET put where the FeFET goes.

Every number in `THESIS_LSNN` is read off the thesis' own deployed
configuration -- `ecg detection/post_quantize.py` and the training command its
`README.md` and `METHODOLOGY.md` record:

    num_in=3, num_lif=100, num_alif=60, num_out=4, max_delay=10, refractory=5,
    dt=5.556e-4, tau_mem=tau_lp=11.11e-3, Ca=5556e-9, Ra=100e3, batch=64

They are NOT re-tuned here. The point of the recalibration is that the optimiser
now moves a DEVICE underneath a network whose architecture is already known to
work, instead of moving both at once and being unable to say which one earned
the result.

Reference: Yuan et al., Nat. Commun. 14:3695 (2023), whose LSNN architecture,
neuron parameters and dataset the thesis adopts; the LSNN itself is Bellec et
al., NeurIPS 2018.
"""

from __future__ import annotations

import math

import torch

from .lif import DTYPE, LIFNet, soft_heaviside, spike_fn

__all__ = ["THESIS_LSNN", "ADAPT_BETA", "DelayedLinear", "LSNNNet", "decay"]


THESIS_LSNN = {
    "n_in": 3,
    "n_lif": 100,
    "n_alif": 60,
    "n_out": 4,
    "max_delay": 10,
    "refractory": 5,
    "dt_s": 5.556e-4,
    "tau_mem_s": 11.11e-3,
    "tau_lp_s": 11.11e-3,
    "tau_adapt_s": 5556e-9 * 100e3,  # Ca * Ra = 0.5556 s
    "spike_target_hz": 20.0,
    "spike_lambda": 5e-6,
    "reference": "Yuan et al., Nat. Commun. 14:3695 (2023); LSNN: Bellec et al., NeurIPS 2018",
}

# Adaptation strength of the ALIF sub-population, in units of the threshold.
#
# This is the ONE coefficient of the port that is a choice rather than a
# transcription, and it is called out here rather than buried. The thesis'
# ALIFVO2 implements adaptation as a real MOSFET leak current: a spike charges
# `va` through a pMOS, and `va` gates an nMOS whose current is subtracted from
# the membrane drive. Those equations are in volts and amps against Vdd = 5 V and
# v_threshold = 3.6 V. This network is in normalised spikes-to-fire units,
# because that is what the device hands it -- th_th, not a gate voltage -- so the
# MOSFET constants have no meaning on this side of the boundary and copying the
# numbers across would be worse than useless.
#
# What IS carried over is the mechanism and its time constant: spike-driven,
# subtractive, slow (tau = Ca*Ra = 0.5556 s, from the thesis' own components).
# The form is the canonical LSNN adaptive threshold of Bellec et al. 2018 -- the
# paper the thesis' own model.py cites for this layer, and which the VO2/FeFET
# adaptation circuit is a physical realisation of.
ADAPT_BETA = 1.8


def decay(dt_s: float, tau_s: float) -> float:
    """exp(-dt/tau): the discrete decay of a first-order RC over one timestep."""
    return math.exp(-dt_s / tau_s)


class DelayedLinear(torch.nn.Module):
    """Fully-connected layer where each connection has its own synaptic delay.

    The thesis' `DelayedLinear`, kept in structure: one delay per connection,
    drawn once at construction from `randint(max_delay)` and then FIXED, with the
    weight for that connection living only in its own delay slot. Ten
    independently programmed taps per connection is an architectural claim of the
    baseline -- it is what makes the synapse array ten times larger -- so the tap
    count is part of what is being aligned to and is not a knob this project
    turns.

    Implementation note: rather than materialise an (in, out, max_delay) weight
    and mask it every timestep, the mask is applied once per forward call and the
    result contracted as (B, in) @ (in, out*max_delay) -> (B, out, max_delay).
    That is the thesis' `einsum_bi_ijk_bjk`, and being a single gemm it is the
    reason a 1116-step sequence is affordable at all.

    `diag_disconnect` removes self-connections, as it does on the recurrent layer
    there: a neuron feeding its own delayed spike straight back is an adaptation
    mechanism, and this network already has an explicit one.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        max_delay: int = 10,
        diag_disconnect: bool = False,
        seed: int = 0,
    ):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        # Same initialisation scale as the thesis' custom_reset_parameters:
        # normal(0, 1/sqrt(fan_in)).
        self.w = torch.nn.Parameter(
            torch.randn(n_in, n_out, generator=g, dtype=DTYPE) / math.sqrt(n_in)
        )
        delays = torch.randint(max_delay, size=(n_in, n_out), generator=g)
        # (n_in, n_out, max_delay), one-hot over the delay axis. A buffer, not a
        # parameter: the delays are drawn once and never trained.
        mask = torch.zeros(n_in, n_out, max_delay, dtype=DTYPE)
        mask.scatter_(2, delays.unsqueeze(-1), 1.0)
        if diag_disconnect:
            if n_in != n_out:
                raise ValueError("diag_disconnect needs a square layer")
            mask = mask * (1.0 - torch.eye(n_in, dtype=DTYPE)).unsqueeze(-1)
        self.register_buffer("delay_mask", mask)
        self.n_in, self.n_out, self.max_delay = n_in, n_out, max_delay

    def delayed_weight(self, w_eff: torch.Tensor) -> torch.Tensor:
        """(n_in, n_out) effective weight -> (n_in, n_out * max_delay)."""
        return (w_eff.unsqueeze(-1) * self.delay_mask).reshape(self.n_in, -1)


class LSNNNet(torch.nn.Module):
    """The thesis LSNN, with the FeFET in the synapses and in the membrane.

    Architecture, unchanged from the baseline::

        input (3) --DelayedLinear--> [ LIF x100 | ALIF x60 ] --LP--> Linear(4)
                                        ^                |
                                        |_DelayedLinear__|
                                          (diag disconnected)

    What the DEVICE supplies, which is the whole point of the module:

    * ``beta`` -- membrane decay per timestep. In the thesis this is exp(-dt/tau)
      with tau = C_mem(R_off + R_on) fixed at 11.11 ms. Here it is whatever the
      FeFET's own subthreshold leak at V_leak makes it, through the DPI relation
      tau = C_mem*SS/(ln10*I_tau). The two agree at the nominal device to within
      half a percent -- 0.6033 against exp(-dt/tau) = 0.6065 at the default
      pooled timestep -- which is a CHECK on the alignment, not an input to it.
    * ``th_th`` -- firing threshold; the thesis' v_threshold.
    * ``g_min``, ``g_max``, ``sig_w`` -- the synapse's programmable conductance
      window and its programming noise, applied to both FeFET arrays.

    What the device does NOT supply: the readout. The thesis labels it a
    full-precision CMOS readout (`redesign_stage1.py`), so it is not mapped
    through the conductance window here either.

    Two departures from the baseline, both forced and both stated:

    * **Refractory period.** The thesis holds a neuron silent for 5 steps of its
      0.5556 ms timestep, i.e. 2.8 ms. This network runs on POOLED timesteps (see
      `snn.ecg.pool_time`); at the default pool of 10 that is well under one
      timestep, so there is nothing left to enforce. The counter is dropped
      rather than rounded up to one step, which would impose a 5.6 ms silence --
      twice the baseline's.
    * **ALIF adaptation.** Ported in mechanism and time constant, not in MOSFET
      constants. See `ADAPT_BETA` above.
    """

    def __init__(
        self,
        n_in: int | None = None,
        n_lif: int | None = None,
        n_alif: int | None = None,
        n_out: int | None = None,
        max_delay: int | None = None,
        dt_s: float | None = None,
        seed: int = 0,
    ):
        super().__init__()
        a = THESIS_LSNN
        self.n_in = a["n_in"] if n_in is None else n_in
        self.n_lif = a["n_lif"] if n_lif is None else n_lif
        self.n_alif = a["n_alif"] if n_alif is None else n_alif
        self.n_out = a["n_out"] if n_out is None else n_out
        self.n_hidden = self.n_lif + self.n_alif
        self.max_delay = a["max_delay"] if max_delay is None else max_delay

        self.fc1 = DelayedLinear(self.n_in, self.n_hidden, self.max_delay, seed=seed)
        self.rc = DelayedLinear(
            self.n_hidden, self.n_hidden, self.max_delay, diag_disconnect=True, seed=seed + 1
        )
        g = torch.Generator().manual_seed(seed + 2)
        self.readout = torch.nn.Parameter(
            torch.randn(self.n_hidden, self.n_out, generator=g, dtype=DTYPE)
            / math.sqrt(self.n_hidden)
        )

        # Decays that belong to the CIRCUIT rather than to the FeFET: the
        # low-pass readout filter and the adaptation capacitor. Both are computed
        # from the thesis' own time constants at whatever timestep this network
        # actually runs at, so pooling the input rescales them consistently
        # instead of silently changing what they mean.
        # 0 on the plain-LIF neurons, 1 on the adaptive ones. Multiplying by this
        # is how the mixed layer is expressed without slicing the state in two.
        mask = torch.zeros(self.n_hidden, dtype=DTYPE)
        mask[self.n_lif :] = 1.0
        self.register_buffer("alif_mask", mask)

        dt = a["dt_s"] if dt_s is None else dt_s
        self.dt_s = dt
        self.register_buffer("lp_decay", torch.tensor(decay(dt, a["tau_lp_s"]), dtype=DTYPE))
        self.register_buffer(
            "adapt_decay", torch.tensor(decay(dt, a["tau_adapt_s"]), dtype=DTYPE)
        )

    def forward(self, x, phi: dict[str, torch.Tensor], seed: int = 0,
                smooth: bool = False, spike_log: list | None = None):
        """x: (B, T, n_in). Returns (logits, mean spikes per neuron per timestep).

        `spike_log`, if given, is appended with the (B, H) spike tensor of every
        timestep. ADDED 2026-08-29 (D6) for the raster figure, and deliberately
        an out-parameter rather than a second return value: the return signature
        is what `tesseract_api` and every test call, and a figure has no business
        changing it. Default None, in which case this method is byte-identical in
        behaviour to what it was -- the flagship numbers it produced still stand.

        NOTE A DIFFERENCE FROM `LIFNet`, because it changes the scale of the
        energy term. `LIFNet` returns spikes/(B*T), which sums over every neuron
        and is therefore a whole-network spike rate, not the per-neuron figure
        the output schema advertises. This divides by the neuron count as well,
        so the number matches its own description.

        The consequence is that `spikes` here is ~160x smaller than on the
        synthetic task, so a `lambda_e` tuned against one is inert against the
        other. `lambda_e` needed picking anyway -- at 1e6 the energy term was
        9e-9 * spikes against a loss of order 1, i.e. "energy-aware" with no term
        behind it -- so it has to be set against THIS scale, not carried over.
        """
        fire = soft_heaviside if smooth else spike_fn
        beta = phi["beta"]
        g_min, g_max, sig_w = phi["g_min"], phi["g_max"], phi["sig_w"]
        th_th = phi["th_th"]

        g = torch.Generator().manual_seed(seed)
        n_in = torch.randn(self.fc1.w.shape, generator=g, dtype=DTYPE)
        n_rc = torch.randn(self.rc.w.shape, generator=g, dtype=DTYPE)
        # The device's conductance window and programming noise, on both FeFET
        # arrays. `_weights` is LIFNet's, unchanged -- including the fact that
        # g_min and g_max cancel there by design.
        w_in = self.fc1.delayed_weight(LIFNet._weights(self.fc1.w, g_min, g_max, sig_w, n_in))
        w_rc = self.rc.delayed_weight(LIFNet._weights(self.rc.w, g_min, g_max, sig_w, n_rc))

        b, t_steps, _ = x.shape
        h, d = self.n_hidden, self.max_delay

        # The INPUT projection is not recurrent -- x is given in full -- so it
        # comes out of the timestep loop entirely and runs as one (B*T, n_in) @
        # (n_in, H*D) gemm. Only the recurrent term has to stay sequential.
        # Mathematically identical, and it removes T small matmuls plus T Python
        # dispatches from the hot loop.
        x_drive = (x.reshape(b * t_steps, self.n_in) @ w_in).view(b, t_steps, h, d)

        zero_tail = torch.zeros(b, h, 1, dtype=DTYPE)
        buf = torch.zeros(b, h, d, dtype=DTYPE)  # the delay line
        u = torch.zeros(b, h, dtype=DTYPE)  # membrane
        adapt = torch.zeros(b, h, dtype=DTYPE)  # adaptation state
        lp = torch.zeros(b, h, dtype=DTYPE)  # low-pass readout filter
        s = torch.zeros(b, h, dtype=DTYPE)  # last spikes, for the recurrence
        out = torch.zeros(b, self.n_out, dtype=DTYPE)
        spikes = x.new_zeros(())

        for t in range(t_steps):
            # Both delayed arrays write into the same delay line. The head is
            # what arrives this timestep; the line then rolls forward.
            buf = buf + x_drive[:, t] + (s @ w_rc).view(b, h, d)
            drive = buf[:, :, 0]
            buf = torch.cat([buf[:, :, 1:], zero_tail], dim=2)

            u = beta * u + drive
            # Adaptive threshold on the ALIF sub-population only -- the plain LIF
            # neurons keep th_th flat, which is what makes the layer mixed. The
            # mask does that rather than a slice-and-concatenate, so `adapt`
            # stays one (B, H) tensor and the loop allocates nothing extra.
            thr = th_th * (1.0 + ADAPT_BETA * adapt)
            s = fire(u - thr)
            u = u - s * thr  # soft reset, as in LIFNet
            adapt = self.adapt_decay * adapt + s * self.alif_mask

            spikes = spikes + s.sum()
            if spike_log is not None:
                spike_log.append(s.detach().clone())
            lp = self.lp_decay * lp + (1.0 - self.lp_decay) * s
            out = out + lp @ self.readout

        return out / t_steps, spikes / (b * t_steps * h)
