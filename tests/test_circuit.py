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


# --- the leak-bias trim (D4) -------------------------------------------------
#
# The whole argument is in config/circuit.yaml under "THE LEAK-BIAS TRIM".
# These tests pin the four properties the trim exists to provide, so that
# changing its constants cannot silently take one of them away.


def _trim_over_the_box():
    """Membrane decay over the measured span of the design box, and its slope."""
    import jax
    import jax.numpy as jnp

    from diffsilicon.shared.circuit import beta_from_dt_over_tau

    # log10 of the DPI ratio dt/tau, measured over the eight banked devices:
    # -5.70 (rand1) to +3.38 (t_fe_min). A little wider, to include the corners
    # nobody has evaluated yet.
    n = np.linspace(-7.0, 5.0, 61)
    x = 10.0**n
    beta = np.array([float(beta_from_dt_over_tau(jnp.asarray(v), CC)) for v in x])
    slope = np.array([
        float(jax.grad(lambda t: beta_from_dt_over_tau(t, CC))(jnp.asarray(v)))
        for v in x
    ])
    return x, beta, slope


def test_the_trim_keeps_the_membrane_decay_inside_its_band():
    """Untrimmed, beta comes back as EXACTLY 0 or EXACTLY 1 on most of the box.

    Measured on the eight banked devices before the trim: three at 0.0000, two
    at 1.0000, with derivatives of -0.0e+00, -1.4e-121 and -2.5e-147. That is a
    design knob with no slope, over most of its own range.
    """
    _, beta, _ = _trim_over_the_box()
    assert beta.min() >= CC.beta_trim_lo - 1e-9
    assert beta.max() <= CC.beta_trim_hi + 1e-9


def test_the_trim_never_kills_the_slope_anywhere_in_the_box():
    """A logistic and not a clip, precisely so that this holds at the corners.

    A clip is flat outside its range, so it would hand back a derivative of
    exactly zero on the devices that most need one, and it would put a kink at
    each edge for the optimiser to fall into.
    """
    x, _, slope = _trim_over_the_box()
    # In the log domain, which is where the trim is defined and where the scale
    # is meaningful. d(beta)/d(log10 x) = slope * x * ln(10).
    dlog = np.abs(slope) * x * np.log(10.0)
    assert np.all(dlog > 1e-4), f"flat spot: min |dbeta/dlog10(dt/tau)| = {dlog.min():.3e}"


def test_a_leakier_device_always_gives_a_leakier_neuron():
    """Strictly decreasing. The trim absorbs the SPREAD; it must not reorder the
    devices, or the optimiser would be told a leakier device is slower."""
    _, beta, _ = _trim_over_the_box()
    assert np.all(np.diff(beta) < 0.0)


def test_the_nominal_device_lands_in_the_middle_of_the_band():
    """leak_trim_centre is the nominal d=4 device's own dt/tau, so the reference
    device sits at the middle by construction rather than by assertion."""
    import jax.numpy as jnp

    from diffsilicon.shared.circuit import beta_from_dt_over_tau

    x_nom = 10.0**CC.leak_trim_centre
    beta = float(beta_from_dt_over_tau(jnp.asarray(x_nom), CC))
    mid = 0.5 * (CC.beta_trim_lo + CC.beta_trim_hi)
    # The band is set in decades of dt/tau, not in beta, so the midpoint in beta
    # is close to but not exactly the arithmetic mean. 0.04 is generous.
    assert abs(beta - mid) < 0.04, f"nominal beta {beta:.4f} against band mid {mid:.4f}"


def test_the_trim_can_be_switched_off_and_gives_back_the_old_behaviour():
    """So a pre-D4 run can be replayed exactly, and so the trim's effect can
    always be measured against its own absence rather than argued about."""
    import jax.numpy as jnp

    from diffsilicon.shared.circuit import beta_from_dt_over_tau

    off = CC._replace(leak_trim_enable=0.0)
    for n in (-5.0, -1.38, 0.0, 2.0):
        x = jnp.asarray(10.0**n)
        assert float(beta_from_dt_over_tau(x, off)) == pytest.approx(
            float(np.exp(-(10.0**n))), rel=1e-12, abs=1e-300
        )


def test_transduce_and_the_pipeline_agree_on_the_membrane_decay():
    """Two code paths compute beta -- the NamedTuple one in shared.circuit and
    the dict one in pipeline.transduce_jax. They were independent until D4,
    which is exactly how a change gets applied to one and not the other."""
    import jax.numpy as jnp

    from diffsilicon.pipeline import transduce_jax
    from diffsilicon.shared.design import nominal_theta

    foms, d = _nominal_foms()
    a = transduce(foms, CC, float(d["L_g"]), float(d["W"]))
    y = {k: jnp.asarray(float(getattr(foms, k)))
         for k in ("ss", "vth_fwd", "vth_rev", "i_leak", "g_lo", "g_hi", "dg_dvth")}
    b = transduce_jax(y, jnp.asarray(nominal_theta(4)), CC)
    assert float(a.beta) == pytest.approx(float(b["beta"]), rel=1e-12)


# --- the synapse-mirror trim (D4) --------------------------------------------


def test_the_firing_threshold_stays_where_the_circuit_was_designed_for_it():
    """th_th goes as 1/g_max, and g_max is four times bigger at the leaky corner
    of the box than at nominal. So the frozen K_syn, which exists solely to make
    th_th a sane spikes-to-fire number, stops doing its job out there.

    Measured before the trim: th_th fell to 1.22, i.e. a neuron fires on roughly
    its first input spike, and the network's gradient with respect to it came
    back at 1.8e6 against -999 at the nominal device.
    """
    import jax.numpy as jnp

    from diffsilicon.shared.circuit import th_th_trimmed

    # The raw span measured over the eight banked devices, widened a little.
    for raw in (0.5, 1.0, 1.22, 2.43, 3.76, 5.62, 12.0, 50.0):
        out = float(th_th_trimmed(jnp.asarray(raw), CC))
        assert CC.th_th_trim_lo - 1e-9 <= out <= CC.th_th_trim_hi + 1e-9


def test_the_frozen_five_spikes_to_fire_survives_the_trim():
    """The band is geometric about th_th_nominal = 5.0, so a device that already
    needs five spikes to fire still needs exactly five. Nothing frozen moves."""
    import math

    import jax.numpy as jnp

    from diffsilicon.shared.circuit import th_th_trimmed

    # rel=1e-12, not 1e-4. At 1e-4 this passed while the band's geometric centre
    # sat at 4.99998 -- both constants had been written to five decimal places --
    # and the real failure surfaced two files away, in
    # test_tier_a_pipeline.py::test_transducer_produces_a_healthy_operating_point,
    # on a device whose raw threshold is 5.000006. A guard that tolerates the bug
    # it exists to catch is not a guard.
    assert float(th_th_trimmed(jnp.asarray(5.0), CC)) == pytest.approx(5.0, rel=1e-12)
    assert math.isclose(math.sqrt(CC.th_th_trim_lo * CC.th_th_trim_hi), 5.0, rel_tol=1e-12)
    assert math.isclose(CC.th_trim_centre, math.log10(5.0), rel_tol=1e-12)


def test_the_firing_threshold_trim_keeps_devices_in_order_and_alive():
    """Strictly increasing, and never flat. Same requirement as the leak trim:
    a clip would give exactly zero slope on the devices that most need one."""
    import jax
    import jax.numpy as jnp

    from diffsilicon.shared.circuit import th_th_trimmed

    raw = np.logspace(-0.5, 1.8, 40)
    out = np.array([float(th_th_trimmed(jnp.asarray(v), CC)) for v in raw])
    slope = np.array([
        float(jax.grad(lambda t: th_th_trimmed(t, CC))(jnp.asarray(v))) for v in raw
    ])
    assert np.all(np.diff(out) > 0.0)
    dlog = slope * raw * np.log(10.0)
    assert np.all(dlog > 1e-3), f"flat spot: min slope in decades = {dlog.min():.3e}"


def test_both_trims_can_be_switched_off_together():
    """So a pre-D4 run can be replayed exactly, and so each trim's effect can be
    measured against its own absence rather than argued about."""
    import jax.numpy as jnp

    from diffsilicon.shared.circuit import beta_from_dt_over_tau, th_th_trimmed

    off = CC._replace(leak_trim_enable=0.0, th_trim_enable=0.0)
    assert float(th_th_trimmed(jnp.asarray(1.22), off)) == pytest.approx(1.22)
    assert float(beta_from_dt_over_tau(jnp.asarray(0.5), off)) == pytest.approx(
        float(np.exp(-0.5))
    )
