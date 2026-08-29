"""SMOOTH figure-of-merit extraction from an Id-Vg double sweep.

Why this file exists
--------------------
Threshold-crossing V_th and max-slope SS on a 96-point grid are argmax / search
operations. Their derivative with respect to a process parameter is piecewise
constant and kinks every time the crossing migrates one grid cell. That
staircase -- not Newton tolerance -- is what dominates finite-difference error in
this pipeline, and it is invisible until you difference the oracle.

Everything below is a smooth reduction over the WHOLE curve: soft weights built
from the local log-slope (never from an index), weighted least squares, and a
weighted local polynomial. There is no argmax, no branch on data, no
interpolation search, no index arithmetic. Written in JAX so the extraction
itself is differentiable and lives INSIDE the oracle, i.e. smoothing happens
before differencing.

REWRITTEN 2026-08-24 (D3) for the widened [-3.5, 1.5] V window
--------------------------------------------------------------
The D1 version centred its subthreshold window on a FIXED current, I_ref = 1e-10
A. That worked while the sweep was [-1.2, 1.4] V, where 1e-10 A occurred once per
branch and always on the turn-on. On the widened window it is wrong twice over,
and it fails loudly: measured on the commercial solver's own curves it returned
SS = 9618 mV/dec and a memory window of 30 V, on curves that are themselves fine.

Both failures are the same mistake -- a current level is not a place on a curve:

1. **The floor sits within one decade of I_ref.** The erased branch spends
   roughly -3.5 V to -0.5 V between 1.1e-11 and 2.3e-11 A. That is 0.6 to 1.0
   decades from 1e-10, so at sigma_dec = 0.6 the Gaussian gives fifty-odd
   near-flat points as much weight as the handful on the real turn-on.
2. **Part of that floor has NEGATIVE slope.** It is the ambipolar / GIDL tail,
   and it falls from -10.64 to -10.95 decades as V_g rises. The regression
   therefore averages a falling branch against a rising one, the covariance
   s_vl collapses towards zero, and SS = s_vv / s_vl explodes. That is where
   9618 mV/dec came from, and why the number was large rather than merely wrong.

The fix is to stop naming a current and name the PROPERTY instead. Subthreshold
is the steep, rising, log-linear stretch of the branch, so the window is now
built from the local log-slope: points are weighted by exp(s / t_slope), which
suppresses both the flat floor and the falling tail automatically and needs no
prior knowledge of where either sits. The window then re-centres on the current
that steep region actually occupies, so the fit stays local without a hard-coded
decade. Nothing here selects a point -- a softmax over a smooth slope is smooth,
and the weights migrate continuously as the curve moves with theta.

One consequence worth stating: weighting by steepness makes SS the MINIMUM
subthreshold swing, which is the textbook definition and the one the calibration
this project is aligned with reports (45-75 mV/dec).

I_leak is likewise no longer extrapolated. See `extract_branch`.

RAISED THE MEASUREMENT FLOOR 2026-08-27 (D4)
--------------------------------------------
`_I_FLOOR` went from 1e-20 to 1e-16. That one line is what made the objective
reproducible. At 1e-20 the extraction was reading slopes out of the deep-off
region of the sweep, where the solver reports ~1e-19 A and the answer is
numerical drift rather than a current -- and a design change of ten femtometres
moved the memory window by up to 0.239 V. The whole argument, and the two
measured bounds that fix the value, are on `_I_FLOOR` below.

All functions here are pure and jit-safe.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

__all__ = ["ExtractionConfig", "FoMs", "extract_foms", "extract_branch"]

# THE MEASUREMENT FLOOR. Currents below this carry no information and the
# extraction must not read any out of them.
#
# RAISED 2026-08-27 (D4) from 1e-20 to 1e-16, and this is the single change that
# made the objective reproducible. Do not lower it without re-running the
# measurement below.
#
# WHAT IT WAS FOR. Originally this was only a softening constant, so that
# log10(sqrt(I^2 + FLOOR^2)) stays finite and differentiable if a solver dump
# puts a current at exactly zero. At 1e-20 it agreed with log10(I) to machine
# precision "for any current this problem ever sees" -- which was true, and was
# exactly the problem: it faithfully preserved numbers that mean nothing.
#
# WHAT WENT WRONG. The sweep runs to -3.50 V, where the device is fully off and
# the solver reports currents around 1e-19 A. That is not a measurement and it
# is not even arithmetic: the electron density there is ~1e-2 cm^-3 against 1e20
# in the contacts, which is sixty decades of dynamic range in one matrix against
# the sixteen that double precision carries. Whatever comes back is drift.
#
# Measured, at the design point where the D4 flagship stalled. Nudging the design
# vector by 1e-12 in normalised units -- about TEN FEMTOMETRES of film thickness,
# a hundredth of the width of an atom:
#
#   * every curve point that moved had |I| <= 4.5e-19 A. NOTHING above that
#     moved at all, at any point on either branch.
#   * with the floor at 1e-20 the reverse-branch threshold jumped 23 mV, the
#     memory window jumped by up to 0.239 V across the banked devices, and the
#     loss jumped by 0.0056 -- three times the entire descent the flagship
#     achieved. The steepness-weighted subthreshold window is free to put weight
#     on the deep-off region, because at 1e-20 that region still has a shape and
#     therefore a slope.
#   * I_leak, g_lo, g_hi and dg_dvth moved by 1e-16 or less. The instability was
#     ENTIRELY in the two threshold voltages, and almost entirely in the reverse
#     one.
#
# HOW THIS VALUE WAS CHOSEN. Both bounds are measured, on seven banked devices:
#
#   worst memory-window jump under a 1e-12 nudge, and the systematic drift the
#   floor itself puts into I_leak:
#
#       floor     worst |d MW|      I_leak drift
#       1e-20      2.392e-01 V         --
#       1e-18      3.100e-06 V       3.8e-04
#       1e-17      4.491e-06 V       1.1e-03
#       1e-16      1.028e-06 V       4.9e-03
#       1e-15      6.267e-08 V       3.6e-02
#
#   * ABOVE the noise. 1e-16 is two hundred times the 4.5e-19 A ceiling where the
#     drift lives, which is why the jump collapses by five orders of magnitude.
#   * BELOW the signal. The smallest genuine leak current anywhere in the design
#     box is 3.1e-15 A (rand1), which is thirty-one times this floor. Going one
#     decade higher, to 1e-15, buys another factor of sixteen in stability and
#     costs 3.6% of I_leak on those devices -- and I_leak sets the membrane
#     decay, so that is not a trade worth making.
#   * And it is still a hundred times below what any real parameter analyser can
#     measure, so nothing a measurement could have seen is being discarded.
#
# A remaining 1e-6 V of jitter in a memory window of ~0.5 V is two parts in a
# million, against a loss signal of ~2e-3. That is no longer the limiting term.
_I_FLOOR_DEFAULT = 1e-16


class ExtractionConfig(NamedTuple):
    """Every tunable of the extraction, in one place, so V1/G4 can sweep them."""

    t_slope: float = 2.0  # dec/V -- softness of the steepness window; see _wls_subthreshold
    sigma_dec: float = 1.2  # decades -- width of that window; see _wls_subthreshold
    v_read: float = 0.60  # V -- read bias for the conductance fit
    sigma_v: float = 0.025  # V -- width of the local-polynomial window; see _wlq_at
    sigma_v_grid_frac: float = 1.0  # x grid spacing -- floor under sigma_v; see _local_window
    poly_order: int = 9  # degree of that local polynomial; see _wlq_at
    #: A -- the measurement floor. See `_I_FLOOR_DEFAULT` for the whole argument
    #: and the two measured bounds that fix it. It is a field rather than a
    #: constant because the two things it trades off pull in opposite
    #: directions, and one caller needs the other end: the analytic accuracy
    #: test in tests/test_extract.py checks the extraction's ARITHMETIC against
    #: a closed-form reference whose leak current runs down to ~2e-16 A, which
    #: is below the floor by construction. That test measures a different
    #: property -- exactness on a clean curve -- and must set its own floor.
    #: Nothing that touches a real solver curve should.
    i_floor: float = _I_FLOOR_DEFAULT
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
    #: 1.0 if BOTH thresholds were found inside the swept window, else 0.0.
    #: Defaulted so the seven-positional-argument form still constructs.
    vth_in_range: jnp.ndarray = jnp.asarray(1.0)


def _log10_soft(i: jnp.ndarray, i_floor: float = _I_FLOOR_DEFAULT) -> jnp.ndarray:
    """Smooth log10 of a current, finite and differentiable at I = 0, and FLAT
    below the measurement floor.

    The parameter is `i_floor` and not `floor` on purpose:
    `test_no_argmax_or_threshold_crossing_in_the_source` bans the bare name
    `floor` from this module, because `jnp.floor` would put a staircase in the
    derivative. The guard is deliberately crude and it is right to be -- a
    variable named `floor` is not worth weakening it for.
    """
    return 0.5 * jnp.log10(i * i + i_floor * i_floor)


def _softmax_weights(z: jnp.ndarray) -> jnp.ndarray:
    """Normalise a vector of non-negative Gaussian kernel values, stably."""
    z = z - jnp.max(z)  # max over data is a reduction, not a selection on theta:
    # it shifts every element identically, so it cancels
    # exactly after normalisation and is derivative-free.
    e = jnp.exp(z)
    return e / jnp.sum(e)


def _log_slope(vg: jnp.ndarray, lg10: jnp.ndarray) -> jnp.ndarray:
    """d(log10 Id)/dV_g along the sweep, in dec/V.

    Central differences inside, one-sided at the two ends. The sweep grid is
    fixed by the frozen contract and does not depend on theta, so this is a
    CONSTANT linear operator applied to the curve: smooth in theta by
    construction, with no selection and no index arithmetic that could move.
    """
    d = (lg10[1:] - lg10[:-1]) / (vg[1:] - vg[:-1])
    return jnp.concatenate([d[:1], 0.5 * (d[1:] + d[:-1]), d[-1:]])


def _subthreshold_weights(
    vg: jnp.ndarray, lg10: jnp.ndarray, cfg: ExtractionConfig
) -> jnp.ndarray:
    """Soft weights selecting the steep, rising, log-linear part of a branch.

    Three terms, combined in one softmax exponent so the normalisation is done
    once and stably:

    * ``s / t_slope`` -- prefer steep. This is what excludes the leakage floor
      and the falling ambipolar tail, neither of which a current-level window can
      tell apart from the turn-on. t_slope = 2 dec/V against subthreshold slopes
      of 10-25 dec/V and parasitic slopes under 2 dec/V, so the discrimination is
      several e-folds per point while the weights still vary continuously as the
      curve shifts.
    * ``-(l - l_ref)^2 / 2 sigma_dec^2`` -- and stay local in current, around
      ``l_ref``, the steepness-weighted mean log-current. The centre is COMPUTED,
      not named: it follows the branch instead of asserting where the branch is.
      Without this term a stray steep point in the on-region would be free to
      join the fit; with it the fit spans the couple of decades either side of
      the steepest stretch, which is what "the subthreshold region" means.
    * ``-(v - v_ref)^2 / 2 sigma_v_sub^2`` -- and local in VOLTAGE as well.

    The voltage term is not redundant, and leaving it out is a bug that survives
    inspection of the weights. Suppressing a point is not the same as removing
    it, because weighted least squares gives a point LEVERAGE proportional to
    (v - vbar)^2. The far end of a 5 V sweep sits four volts from the fit centre,
    so its leverage is ~1600x that of a point one grid step away, and a weight of
    1e-5 does not begin to pay for that. Measured on the commercial solver's
    erased branch: the weights were textbook -- 0.41, 0.27, 0.21 on the three
    steepest points and under 1e-3 everywhere else -- and SS still came out at
    237 mV/dec instead of 69, because fifty all-but-unweighted floor points
    between -3.5 V and -0.5 V dominated s_vv anyway. The current-domain Gaussian
    cannot fix this on its own: the floor is only ~2.7 decades below l_ref, which
    is well inside sigma_dec.

    Its width is not a new tunable. sigma_dec is a width in decades, and the
    branch's own soft peak slope s_ref converts it to volts:

        sigma_v_sub = sigma_dec / s_ref

    so both Gaussians describe the SAME window, once in each axis, and the window
    narrows automatically on a steep device and widens on a shallow one.
    """
    s = _log_slope(vg, lg10)
    w_s = _softmax_weights(s / cfg.t_slope)
    l_ref = jnp.sum(w_s * lg10)
    v_ref = jnp.sum(w_s * vg)
    # Soft peak slope, dec/V. Floored well below any real subthreshold slope so
    # that a dead branch -- one with no turn-on anywhere in the window -- widens
    # to the whole sweep instead of dividing by zero.
    s_ref = jnp.sqrt(jnp.sum(w_s * s) ** 2 + 1.0)
    sigma_v_sub = cfg.sigma_dec / s_ref
    return _softmax_weights(
        s / cfg.t_slope
        - ((lg10 - l_ref) ** 2) / (2.0 * cfg.sigma_dec**2)
        - ((vg - v_ref) ** 2) / (2.0 * sigma_v_sub**2)
    )


def _wls_subthreshold(
    vg: jnp.ndarray, idv: jnp.ndarray, cfg: ExtractionConfig
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Weighted least squares of log10(Id) on V_g over the soft subthreshold window.

    Returns (SS in V/dec, weighted mean V_g, weighted mean log10 Id).
    """
    lg10 = _log10_soft(idv, cfg.i_floor)
    w = _subthreshold_weights(vg, lg10, cfg)

    vbar = jnp.sum(w * vg)
    lbar = jnp.sum(w * lg10)
    dv = vg - vbar
    dl = lg10 - lbar

    s_vv = jnp.sum(w * dv * dv)
    s_vl = jnp.sum(w * dv * dl)

    # slope [dec/V] = s_vl / s_vv  ->  SS [V/dec] = s_vv / s_vl
    ss_v_per_dec = s_vv / s_vl
    return ss_v_per_dec, vbar, lbar


def _local_window(vg: jnp.ndarray, cfg: ExtractionConfig) -> jnp.ndarray:
    """Effective width of the local-polynomial window, in volts.

    ADDED 2026-08-24 (D3), and it is the second thing the widened window broke.
    sigma_v = 25 mV was chosen on D1 as "slightly under one grid spacing", when
    96 points spanned 2.6 V and a spacing was 27.4 mV. The same 96 points now
    span 5.0 V, so a spacing is 52.6 mV and the fixed 25 mV window is HALF a
    grid cell: three points carry any weight at all, and a degree-5 polynomial
    has six coefficients. The Tikhonov term kept it from blowing up, which is
    exactly why this was silent rather than loud.

    So the width is now the D1 value and a grid-derived floor, in quadrature.
    The grid is a frozen contract constant, not data -- it does not depend on
    theta -- so this changes the window's SIZE without making it data-dependent,
    and smoothness in theta is untouched.
    """
    dv = (vg[-1] - vg[0]) / (vg.shape[0] - 1)
    return jnp.sqrt(cfg.sigma_v**2 + (cfg.sigma_v_grid_frac * dv) ** 2)


def _wlq_at(
    vg: jnp.ndarray, idv: jnp.ndarray, v_centre, cfg: ExtractionConfig
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Weighted local-polynomial fit of log10(Id) near an arbitrary bias.

    Three choices here were each forced by measurement on D1, not assumed:

    * **Log current, not linear current.** Near V_read the erased branch is often
      still exponential, and a polynomial in linear Id fitted across a window
      spanning two e-folds is badly wrong. In log space the same curve is a
      straight line and the fit is exact for a pure exponential.
    * **Degree 9, not 2, and not the 5 that D1 settled on.** A quadratic's slope
      estimate carries an O(sigma^2) bias from the third derivative of the curve,
      and the subthreshold-to-on knee sits right where V_read does: the measured
      worst-case error on dg/dV_th was 14% at degree 2 and 0.26% at degree 5,
      over 200 random points of the d=5 design box. Each extra degree removes one
      more bias term.

      The degree went 5 -> 9 on D3 because `_local_window` had to widen: the
      widened sweep doubled the grid spacing, the window had to grow to keep
      enough points under it, and a wider window needs more degrees to cover the
      same knee. Measured over 40 random d=5 points at the new width, worst-case
      dg/dV_th error is 1.74% at degree 5, 0.90% at 7 and 0.44% at 9.

      Degree is only safe while the window is WIDER than a grid spacing. At the
      old fixed 25 mV width, degree 9 on the new grid is ten coefficients over
      three weighted points and it detonates: injecting 1e-6 relative noise into
      a real solver curve moves I_leak by 4.9e-2, against 7.6e-7 at the width
      actually used. That is why the width is set from the grid rather than
      guessed, and why these two constants may not be moved independently.
    * **Scaled abscissa u = (V - V_read)/sigma_v.** With raw volts the degree-5
      normal matrix has condition number ~1e12; in u it is ~5e3.

    The window is comparable to one grid spacing, so the obvious worry is that the
    fit becomes grid-locked and re-introduces the staircase this module exists to
    prevent. Measured on the commercial solver's own curves: injecting 1e-6
    relative noise moves every FoM by under 1e-6 except dg/dV_th, which moves by
    8e-5, and the G4 staircase metric
    halves exactly under grid refinement (40 -> 80 -> 160 sweep points give
    0.476 -> 0.246 -> 0.125), which is the O(h) signature of ordinary curvature
    with no staircase underneath it.

    Returns (Id at v_centre [A], dId/dV_g at v_centre [A/V]).
    """
    sigma = _local_window(vg, cfg)
    u = (vg - v_centre) / sigma
    w = _softmax_weights(-(u**2) / 2.0)
    y = _log10_soft(idv, cfg.i_floor)

    X = jnp.stack([u**k for k in range(cfg.poly_order + 1)], axis=1)  # (N, p+1)
    XtW = X.T * w  # (p+1, N)
    A = XtW @ X
    b = XtW @ y

    # Tikhonov floor keeps the solve well-posed if the window ever collapses; on
    # the scaled abscissa the diagonal is O(1), so 1e-14 is inert.
    A = A + jnp.eye(A.shape[0], dtype=A.dtype) * 1e-14
    a = jnp.linalg.solve(A, b)

    id_at_centre = jnp.power(10.0, a[0])
    did_dvg = id_at_centre * jnp.log(10.0) * a[1] / sigma
    return id_at_centre, did_dvg


def extract_branch(
    vg: jnp.ndarray, idv: jnp.ndarray, cfg: ExtractionConfig, vds: float
) -> dict[str, jnp.ndarray]:
    """Extract SS, V_th, I_leak, g and dId/dVg from a SINGLE sweep branch."""
    ss_v, vbar, lbar = _wls_subthreshold(vg, idv, cfg)

    i_crit = cfg.i_crit_per_wl * cfg.w_dev_nm / cfg.l_g_nm
    vth = vbar + ss_v * (jnp.log10(i_crit) - lbar)

    # THE THRESHOLD MUST LIE INSIDE THE VOLTAGES WE ACTUALLY SWEPT.
    #
    # The line above is an extrapolation from the fit centre out to the critical
    # current. On a device with a real subthreshold region that lands inside the
    # sweep and is exactly what you want. On a device that never switches off it
    # does not, and nothing used to stop it: measured 2026-08-26, one device in
    # eight returned vth_fwd = +4.16 V from a sweep ending at +1.50 V, giving a
    # memory window of 4.44 V -- larger than the entire 5 V sweep could support.
    # That device reads SS = 431 mV/dec, i.e. it has no off-region at all, so
    # there was no threshold in there to find and the fit answered from a region
    # where no current was ever measured.
    #
    # This is NOT the D3 bug (a fixed current level landing in the leakage
    # floor); that one is fixed and this does not touch it. It is the separate
    # question of whether the answer is inside the data.
    #
    # THE VALUE IS LEFT EXACTLY AS IT IS. Only a FLAG is raised.
    #
    # Clamping the threshold to the sweep was tried first and is wrong here.
    # `test_no_argmax_or_threshold_crossing_in_the_source` bans clip/where/argmax
    # from this module on purpose: every one of them reintroduces a
    # piecewise-constant derivative in theta, which is the exact property this
    # whole module exists to avoid. A clamp would have been flat in theta outside
    # the window and would have put a kink in the path for every device, to
    # sanitise the few that are already being thrown away.
    #
    # So the flag carries the whole job. `run_oracle` folds it into `converged`,
    # and a design point that is not converged is REJECTED upstream rather than
    # patched up here. Nothing downstream should ever consume a threshold from a
    # point whose flag is 0, so the raw value costs nothing and the gradient path
    # stays smooth for every point that IS usable.
    v_lo = jnp.min(vg)
    v_hi = jnp.max(vg)
    vth_in_range = ((vth >= v_lo) & (vth <= v_hi)).astype(vth.dtype)

    # I_leak is READ OFF THE CURVE at V_leak with the same local-polynomial fit
    # used at V_read, not extrapolated down the subthreshold line.
    #
    # CHANGED 2026-08-24 (D3). The extrapolation was exact on the analytic mock,
    # whose branches are two straight lines and nothing else, and it is wrong by
    # four orders of magnitude on a real device, which has a leakage floor the
    # line knows nothing about. Measured on the commercial solver's erased
    # branch: the fitted subthreshold line reaches 1.4e-15 A at V_leak = 0.246 V
    # while the device actually draws 4.4e-11 A there, because it bottomed out on
    # its ambipolar floor two volts earlier. I_leak sets the DPI leak current and
    # hence beta, so a four-decade error there is not cosmetic.
    #
    # This was hidden until now: the old extraction reported SS = 9618 mV/dec on
    # these curves, i.e. an almost flat line, so extrapolating it barely moved
    # and I_leak came out plausible for entirely the wrong reason. Fixing SS is
    # what exposes it.
    #
    # A local fit is still smooth and still differentiable in theta -- it is the
    # same machinery, at a different centre -- so nothing about the gradient path
    # changes. It is only no longer an extrapolation.
    i_leak, _ = _wlq_at(vg, idv, cfg.v_leak, cfg)

    id_read, did_dvg = _wlq_at(vg, idv, cfg.v_read, cfg)
    g_read = id_read / vds

    return {
        "ss_v_per_dec": ss_v,
        "vth": vth,
        "vth_in_range": vth_in_range,
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
        # BOTH branches have to be inside the sweep. A memory window is a
        # DIFFERENCE of two thresholds, so one bad branch is enough to ruin it.
        vth_in_range=fwd["vth_in_range"] * rev["vth_in_range"],
    )
