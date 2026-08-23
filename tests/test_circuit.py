"""The frozen circuit constants, and the arithmetic that justifies them.

V_leak and K_syn were each computed once, on 2026-08-23, against the nominal
d=5 device. These tests re-derive both from scratch. If someone edits
config/circuit.yaml without redoing that arithmetic, this fails immediately
rather than three days later inside an optimisation that will not converge.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from diffsilicon.shared.circuit import load_circuit, sigma_vth, transduce
from diffsilicon.shared.contract import DEFAULT_VG_GRID
from diffsilicon.shared.design import nominal_theta
from diffsilicon.shared.extract import extract_foms
from diffsilicon.shared.mock_device import device_params, id_vg_curves
from diffsilicon.shared.oracle import extraction_config

CC = load_circuit()
VG = jnp.asarray(DEFAULT_VG_GRID)


def _nominal_foms():
    theta = nominal_theta(5)
    cfg = extraction_config(theta)
    c = id_vg_curves(theta, DEFAULT_VG_GRID, CC.v_ds)
    return extract_foms(VG, c[0], c[1], cfg, CC.v_ds), device_params(jnp.asarray(theta))


def test_v_leak_is_frozen_at_170_pa():
    """The definition of V_leak: the bias at which the NOMINAL device draws 170 pA."""
    foms, _ = _nominal_foms()
    assert float(foms.i_leak) == pytest.approx(170e-12, rel=1e-4)


def test_v_leak_value_is_reproducible_from_first_principles():
    _, d = _nominal_foms()
    expected = float(d["vth_fwd"]) + float(d["ss_v_per_dec"]) * np.log10(
        170e-12 / float(d["i_crit"])
    )
    assert CC.v_leak == pytest.approx(expected, abs=1e-7)


def test_dpi_time_constant_is_healthy():
    """tau ~= 2 * dt_hw. Too short and the membrane forgets between timesteps;
    too long and beta saturates at 1 and d(beta)/d(SS) vanishes -- which is exactly
    what killed the naive transducer that used the FeFET's own gate capacitance."""
    foms, _ = _nominal_foms()
    ss_v = float(foms.ss) * 1e-3
    tau = CC.c_mem * ss_v / (np.log(10) * float(foms.i_leak))
    assert 1.5 * CC.dt_hw < tau < 3.0 * CC.dt_hw
    x = CC.dt_hw * np.log(10) * float(foms.i_leak) / (CC.c_mem * ss_v)
    assert np.exp(-x) == pytest.approx(0.6033, abs=2e-3)


def test_beta_channel_is_well_conditioned():
    """Both of beta's inputs come from the solver, and both must actually move it."""
    foms, d = _nominal_foms()
    ss_v = float(foms.ss) * 1e-3
    i_tau = float(foms.i_leak)
    x = CC.dt_hw * np.log(10) * i_tau / (CC.c_mem * ss_v)
    beta = np.exp(-x)
    dbeta_dss = beta * x / ss_v
    dbeta_ditau = -beta * x / i_tau
    assert dbeta_dss * 0.010 > 0.02  # a 10 mV/dec SS change must move beta > 0.02
    assert abs(dbeta_ditau * 0.10 * i_tau) > 0.02  # so must a 10% I_tau change


def test_k_syn_gives_five_spikes_to_fire():
    foms, d = _nominal_foms()
    phi = transduce(foms, CC, float(d["L_g"]), float(d["W"]))
    assert float(phi.th_th) == pytest.approx(5.0, rel=1e-3)


def test_the_network_actually_fires_at_the_nominal_operating_point():
    """th_th is only meaningful if a neuron can reach it. At th_th = 20 the whole
    network was silent -- loss pinned at ln(4), every logit zero -- while still
    returning a plausible-looking surrogate gradient. Assert liveness directly."""
    import torch

    from diffsilicon.snn.lif import LIFNet, synthetic_batch

    foms, d = _nominal_foms()
    phi = transduce(foms, CC, float(d["L_g"]), float(d["W"]))
    tphi = {
        k: torch.tensor(float(v), dtype=torch.float32)
        for k, v in zip(("beta", "g_min", "g_max", "th_th", "sig_w"), phi, strict=True)
    }
    x, _ = synthetic_batch(32, 24, 16, 4, seed=0)
    logits, spikes = LIFNet(16, 24, 4, seed=0)(x, tphi, seed=0)
    assert float(spikes) > 0.0, "no neuron fired: the device-to-classifier channel is dead"
    assert float(logits.std()) > 0.0, "all logits identical: the network cannot discriminate"


def test_k_syn_does_not_touch_sigma_w():
    """K_syn scales the conductance path uniformly, so it must cancel out of sig_w.
    If it ever stopped cancelling, a circuit-design knob would be silently moving a
    variability prediction."""
    foms, d = _nominal_foms()
    a = transduce(foms, CC, float(d["L_g"]), float(d["W"]))
    b = transduce(foms, CC._replace(k_syn=CC.k_syn * 100.0), float(d["L_g"]), float(d["W"]))
    assert float(a.sig_w) == pytest.approx(float(b.sig_w), rel=1e-12)
    assert float(a.beta) == pytest.approx(float(b.beta), rel=1e-12)


def test_sigma_vth_matches_the_published_arithmetic():
    """63 mV Pelgrom + 40 mV domain, in quadrature, at W=100nm L=40nm MW=0.5V."""
    s = float(sigma_vth(0.5, 0.100, 0.040, CC))
    assert s == pytest.approx(0.0746, abs=5e-4)
    pelgrom = CC.a_vth / np.sqrt(0.100 * 0.040)
    assert pelgrom == pytest.approx(0.0632, abs=5e-4)


def test_longer_gate_reduces_variability():
    """The tension that makes d=5 non-trivial: shrinking L_g wrecks sigma_Vth
    through BOTH terms, while buying density and energy."""
    s40 = float(sigma_vth(0.5, 0.100, 0.040, CC))
    s60 = float(sigma_vth(0.5, 0.100, 0.060, CC))
    assert s60 < s40
    assert s60 == pytest.approx(0.0609, abs=1e-3)


def test_assumed_coefficients_are_declared_in_config():
    """A_Vth and A_dom are assumed, not measured. They must live in the config file
    where the writeup can point at them, never inline in the code."""
    import re
    from pathlib import Path

    from diffsilicon.shared.circuit import _find_config

    text = Path(_find_config()).read_text(encoding="utf-8")
    assert re.search(r"ASSUMED", text), "config must flag its assumed coefficients"
    assert "A_Vth" in text and "A_dom" in text
