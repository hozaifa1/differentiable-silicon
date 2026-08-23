"""The composed objective, and the gradient that crosses three AD regimes.

    J(theta) = F(H(G(theta)))  +  lambda_E * E(H(G(theta)))  +  lambda_R * R(theta)

    G: R^D -> R^7   commercial solver, no adjoint      (adjoint-shim over an oracle)
    H: R^7 -> R^5   DPI transducer, pure JAX, exact    (shared.circuit.transduce)
    F: R^5 -> R     surrogate-gradient LIF, PyTorch    (snn-lif-ecg)

The reverse sweep is three hops and each one crosses a boundary:

1. **PyTorch -> wire -> JAX.** T4's vjp is torch.autograd.grad(L, phi); the
   cotangent leaves as five float64s and JAX picks it up with no shared tape.
2. **JAX, exact and free.** (dH/dy)^T applied by jax.vjp, in R^7.
3. **Manufactured adjoint.** T3's vjp is J^T gbar_y with J in R^(7xD), and it
   costs ZERO solver calls whenever the local model is fresh.

Not every path to theta goes through the solver: L_g and W are design variables
that H uses directly in the Pelgrom area term, so part of dJ/dtheta is exact JAX
and part is manufactured. Both are real; V2 checks the composition, not the pieces.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from tesseract_jax import apply_tesseract

from .shared.circuit import CircuitConfig, sigma_vth
from .shared.contract import DEFAULT_VG_GRID, DIFFERENTIABLE_OUTPUTS
from .shared.design import get_design
from .snn.lif import PHI_KEYS

jax.config.update("jax_enable_x64", True)

__all__ = ["oracle_call", "transduce_jax", "composed_loss", "box_project"]


def oracle_call(shim, theta, vg_grid=None, vds_lin=0.05, vds_sat=0.80):
    """G: run the solver through the shim. Differentiable via the shim's VJP."""
    payload = {
        "theta": theta,
        "vg_grid": jnp.asarray(DEFAULT_VG_GRID if vg_grid is None else vg_grid),
        "vds_lin": vds_lin,
        "vds_sat": vds_sat,
    }
    return apply_tesseract(shim, payload)


def _geometry_jax(theta):
    """(W, L_g) in nm, in JAX, so the direct theta -> sigma_Vth path stays differentiable."""
    spec = get_design(int(theta.shape[-1]))
    lo, hi = jnp.asarray(spec.lo), jnp.asarray(spec.hi)
    phys = lo + theta * (hi - lo)
    names = spec.names
    w = phys[names.index("W_dev")] if "W_dev" in names else jnp.asarray(100.0)
    lg = phys[names.index("L_g")] if "L_g" in names else jnp.asarray(40.0)
    return w, lg


def transduce_jax(y: dict, theta, cfg: CircuitConfig) -> dict:
    """H: seven FoMs (plus theta's own geometry) -> the five SNN hyperparameters."""
    ss_v = y["ss"] * 1e-3
    i_tau = y["i_leak"]
    g_min, g_max = y["g_lo"], y["g_hi"]
    mw = y["vth_fwd"] - y["vth_rev"]

    beta = jnp.exp(-cfg.dt_hw * jnp.log(10.0) * i_tau / (cfg.c_mem * ss_v))
    th_th = cfg.c_mem * cfg.v_spk / (cfg.k_syn * g_max * cfg.v_ds * cfg.dt_hw)

    w_nm, lg_nm = _geometry_jax(theta)
    s_vth = sigma_vth(mw, w_nm * 1e-3, lg_nm * 1e-3, cfg)
    sig_w = y["dg_dvth"] * s_vth / (g_max - g_min)

    return {"beta": beta, "g_min": g_min, "g_max": g_max, "th_th": th_th, "sig_w": sig_w}


def composed_loss(
    shim,
    snn,
    theta,
    cfg: CircuitConfig,
    lambda_e: float = 0.0,
    lambda_r: float = 0.0,
    seed: int = 0,
    batch: int = 32,
):
    """J(theta), with every forward value coming from the configured oracle."""
    y = oracle_call(shim, theta)
    phi = transduce_jax(y, theta, cfg)
    out = apply_tesseract(snn, {**{k: phi[k] for k in PHI_KEYS}, "seed": seed, "batch": batch})

    loss = out["loss"]
    if lambda_e:
        # E ~ C_mem V_spk^2 E[spikes]: the energy the membrane actually spends.
        loss = loss + lambda_e * cfg.c_mem * cfg.v_spk**2 * out["spikes"]
    if lambda_r:
        # Keep theta off the faces of the box; projection handles the hard bound.
        loss = loss + lambda_r * jnp.sum((theta - 0.5) ** 2)
    return loss


def box_project(theta):
    """Theta lives in [0,1]^D by the frozen contract. Project every step."""
    return jnp.clip(theta, 0.0, 1.0)


def jacobian_rows() -> tuple[str, ...]:
    return DIFFERENTIABLE_OUTPUTS


def as_numpy(tree):
    return jax.tree.map(lambda x: np.asarray(x), tree)
