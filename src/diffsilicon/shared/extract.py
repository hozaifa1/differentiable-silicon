"""SMOOTH figure-of-merit extraction from an Id-Vg double sweep.

Why this file exists
--------------------
Threshold-crossing V_th and max-slope SS on a 96-point grid are argmax / search
operations. Their derivative with respect to a process parameter is piecewise
constant and kinks every time the crossing migrates one grid cell. That
staircase -- not Newton tolerance -- is what dominates finite-difference error in
this pipeline, and it is invisible until you difference the oracle.

Everything below is a smooth reduction over the WHOLE curve: soft Gaussian
weights in log-current (centred on a decade, never on an index), weighted least
squares, and a weighted local quadratic. There is no argmax, no branch on data,
no interpolation search, no index arithmetic. Written in JAX so the extraction
itself is differentiable and lives INSIDE the oracle, i.e. smoothing happens
before differencing.

All functions here are pure and jit-safe.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

__all__ = ["ExtractionConfig", "FoMs", "extract_foms", "extract_branch"]

# Softening constant for log10 of a current that may legitimately touch zero in
# a solver dump. log10(sqrt(I^2 + FLOOR^2)) is smooth everywhere and agrees with
# log10(I) to machine precision for any current this problem ever sees.
_I_FLOOR = 1e-20


class ExtractionConfig(NamedTuple):
    """Every tunable of the extraction, in one place, so V1/G4 can sweep them."""

    i_ref: float = 1e-10  # A -- centre of the subthreshold soft window
    sigma_dec: float = 0.6  # decades -- width of that window; see _wls_subthreshold
    v_read: float = 0.60  # V -- read bias for the conductance fit
    sigma_v: float = 0.025  # V -- width of the local-polynomial window; see _wlq_conductance
    poly_order: int = 5  # degree of that local polynomial
    v_leak: float = 0.30  # V -- leak-device bias; FROZEN, see config/circuit.yaml
    i_crit_per_wl: float = 100e-9  # A -- constant-current V_th criterion, x W/L_g
    w_dev_nm: float = 100.0  # nm
    l_g_nm: float = 40.0  # nm


class FoMs(NamedTuple):
    """The seven differentiable figures of merit. Order matches OracleOutput."""

    ss: jnp.ndarray  # mV/dec
    vth_fwd: jnp.ndarray  # V
    vth_rev: jnp.ndarray  # V
    i_leak: jnp.ndarray  # A
    g_lo: jnp.ndarray  # S
    g_hi: jnp.ndarray  # S
    dg_dvth: jnp.ndarray  # S/V, positive-by-convention magnitude


def _log10_soft(i: jnp.ndarray) -> jnp.ndarray:
    """Smooth log10 of a current, finite and differentiable at I = 0."""
    return 0.5 * jnp.log10(i * i + _I_FLOOR * _I_FLOOR)


def _softmax_weights(z: jnp.ndarray) -> jnp.ndarray:
    """Normalise a vector of non-negative Gaussian kernel values, stably."""
    z = z - jnp.max(z)  # max over data is a reduction, not a selection on theta:
    # it shifts every element identically, so it cancels
    # exactly after normalisation and is derivative-free.
    e = jnp.exp(z)
    return e / jnp.sum(e)


def _wls_subthreshold(
    vg: jnp.ndarray, idv: jnp.ndarray, cfg: ExtractionConfig
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Weighted least squares of log10(Id) on V_g over the soft subthreshold window.

    Returns (SS in V/dec, weighted mean V_g, weighted mean log10 Id).
    """
    lg10 = _log10_soft(idv)
    l_ref = jnp.log10(cfg.i_ref)

    # Soft window centred on a DECADE of current, not on a grid index.
    #
    # sigma_dec = 0.6, not the 1.0 first drafted. A real Id-Vg curve leaves the
    # log-linear regime about 4.4 decades above 1e-10 A, and at sigma_dec = 1.0
    # the Gaussian tail reaches far enough into that knee to bias the slope.
    # Measured over 120 random points of the d=5 box: worst-case SS error falls
    # from 0.28% to 0.016%, worst-case V_th error from 0.68 mV to 0.05 mV, and
    # worst-case I_leak error -- which amplifies the SS error over however many
    # decades separate V_leak from the window -- from 1.72% to 0.15%.
    w = _softmax_weights(-((lg10 - l_ref) ** 2) / (2.0 * cfg.sigma_dec**2))

    vbar = jnp.sum(w * vg)
    lbar = jnp.sum(w * lg10)
    dv = vg - vbar
    dl = lg10 - lbar

    s_vv = jnp.sum(w * dv * dv)
    s_vl = jnp.sum(w * dv * dl)

    # slope [dec/V] = s_vl / s_vv  ->  SS [V/dec] = s_vv / s_vl
    ss_v_per_dec = s_vv / s_vl
    return ss_v_per_dec, vbar, lbar


def _wlq_conductance(
    vg: jnp.ndarray, idv: jnp.ndarray, cfg: ExtractionConfig, vds: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Weighted local-polynomial fit of log10(Id) near V_read.

    Three choices here were each forced by measurement on D1, not assumed:

    * **Log current, not linear current.** Near V_read the erased branch is often
      still exponential, and a polynomial in linear Id fitted across a window
      spanning two e-folds is badly wrong. In log space the same curve is a
      straight line and the fit is exact for a pure exponential.
    * **Degree 5, not 2.** The spec called for a local quadratic. A quadratic's
      slope estimate carries an O(sigma^2) bias from the third derivative of the
      curve, and the subthreshold-to-on knee sits right where V_read does: the
      measured worst-case error on dg/dV_th was 14% at degree 2 and 0.26% at
      degree 5, over 200 random points of the d=5 design box. Degrees 3 and 4
      each remove one more bias term; 5 is where every FoM lands under 0.5%.
    * **Scaled abscissa u = (V - V_read)/sigma_v.** With raw volts the degree-5
      normal matrix has condition number ~1e12; in u it is ~5e3.

    The window is narrow (sigma_v = 25 mV, slightly under one grid spacing), so
    the obvious worry is that the fit becomes grid-locked and re-introduces the
    staircase this module exists to prevent. Measured: injecting 1e-6 relative
    noise into the curve moves dg/dV_th by 0.01%, and the G4 staircase metric
    halves exactly under grid refinement (40 -> 80 -> 160 sweep points give
    0.476 -> 0.246 -> 0.125), which is the O(h) signature of ordinary curvature
    with no staircase underneath it.

    Returns (g at V_read [S], dId/dV_g at V_read [A/V]).
    """
    u = (vg - cfg.v_read) / cfg.sigma_v
    w = _softmax_weights(-(u**2) / 2.0)
    y = _log10_soft(idv)

    X = jnp.stack([u**k for k in range(cfg.poly_order + 1)], axis=1)  # (N, p+1)
    XtW = X.T * w  # (p+1, N)
    A = XtW @ X
    b = XtW @ y

    # Tikhonov floor keeps the solve well-posed if the window ever collapses; on
    # the scaled abscissa the diagonal is O(1), so 1e-14 is inert.
    A = A + jnp.eye(A.shape[0], dtype=A.dtype) * 1e-14
    a = jnp.linalg.solve(A, b)

    id_at_read = jnp.power(10.0, a[0])
    did_dvg = id_at_read * jnp.log(10.0) * a[1] / cfg.sigma_v
    return id_at_read / vds, did_dvg


def extract_branch(
    vg: jnp.ndarray, idv: jnp.ndarray, cfg: ExtractionConfig, vds: float
) -> dict[str, jnp.ndarray]:
    """Extract SS, V_th, I_leak, g and dId/dVg from a SINGLE sweep branch."""
    ss_v, vbar, lbar = _wls_subthreshold(vg, idv, cfg)

    i_crit = cfg.i_crit_per_wl * cfg.w_dev_nm / cfg.l_g_nm
    vth = vbar + ss_v * (jnp.log10(i_crit) - lbar)

    # Extrapolate the SAME fitted line down to the leak bias. Using the fit rather
    # than the nearest sample is what makes I_leak differentiable in theta.
    i_leak = jnp.power(10.0, lbar + (cfg.v_leak - vbar) / ss_v)

    g_read, did_dvg = _wlq_conductance(vg, idv, cfg, vds)

    return {
        "ss_v_per_dec": ss_v,
        "vth": vth,
        "i_leak": i_leak,
        "g_read": g_read,
        "did_dvg": did_dvg,
    }


def extract_foms(
    vg: jnp.ndarray,
    id_fwd: jnp.ndarray,
    id_rev: jnp.ndarray,
    cfg: ExtractionConfig,
    vds: float,
) -> FoMs:
    """The seven FoMs from a hysteretic Id-Vg double sweep.

    Branch convention (counterclockwise FeFET, fixed here and never revisited):
    the FORWARD (up) sweep sees the erased, high-V_th state; the REVERSE (down)
    sweep sees the programmed, low-V_th state. Hence g_lo comes from forward and
    g_hi from reverse, and the memory window is vth_fwd - vth_rev > 0.

    SS and I_leak are both taken from the forward branch, because the DPI leak
    device is held in one fixed state and the time constant tau = C*SS/(ln10*I_tau)
    is only meaningful if its two inputs describe the same device state.
    """
    fwd = extract_branch(vg, id_fwd, cfg, vds)
    rev = extract_branch(vg, id_rev, cfg, vds)

    # |dg/dV_th| under the translational-shift approximation Id(Vg; Vth + d) =
    # Id(Vg - d), i.e. dg/dV_th = -(1/V_ds) dId/dV_g. Reported as a positive
    # sensitivity: the sign is fixed by physics, and carrying it would only put an
    # abs() -- a kink -- inside the sigma_w formula. Averaged over both branches
    # because a programmed weight lives anywhere between them.
    dg_dvth = 0.5 * (fwd["did_dvg"] + rev["did_dvg"]) / vds

    return FoMs(
        ss=fwd["ss_v_per_dec"] * 1e3,  # mV/dec
        vth_fwd=fwd["vth"],
        vth_rev=rev["vth"],
        i_leak=fwd["i_leak"],
        g_lo=fwd["g_read"],
        g_hi=rev["g_read"],
        dg_dvth=dg_dvth,
    )
