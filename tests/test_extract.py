"""Extraction accuracy against a closed-form reference, and derivative continuity.

The mock device is built so that its subthreshold branch is exactly log-linear,
which makes SS, V_th and I_leak analytically known rather than merely
self-consistent. That is what gives the sub-percent assertions below any force:
they compare the extraction against arithmetic, not against itself.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from diffsilicon.shared.circuit import load_circuit
from diffsilicon.shared.contract import DEFAULT_VG_GRID
from diffsilicon.shared.design import nominal_theta
from diffsilicon.shared.extract import extract_foms
from diffsilicon.shared.mock_device import analytic_foms, device_params, id_vg_curves
from diffsilicon.shared.oracle import extraction_config

CC = load_circuit()
VG = jnp.asarray(DEFAULT_VG_GRID)

# Per-FoM tolerances, all measured on D1 over 150 random points of the d=5 box.
# V_th is held to an ABSOLUTE tolerance: it legitimately passes through zero, and
# a relative bound on a quantity that crosses zero is not a statement about
# accuracy. Everything else is relative.
REL_TOL = {"ss": 0.5, "i_leak": 0.5, "g_lo": 0.5, "g_hi": 0.5, "dg_dvth": 0.5}
VTH_ABS_TOL_V = 1e-3


def _extract(theta, vds=None):
    vds = CC.v_ds if vds is None else vds
    cfg = extraction_config(theta)
    curves = id_vg_curves(theta, DEFAULT_VG_GRID, vds)
    return extract_foms(VG, curves[0], curves[1], cfg, vds), analytic_foms(theta, cfg, vds)


@pytest.mark.parametrize("D", [3, 5, 12])
def test_nominal_matches_analytic(D):
    got, ref = _extract(nominal_theta(D))
    for k, tol in REL_TOL.items():
        err = abs(float(getattr(got, k)) - float(ref[k])) / abs(float(ref[k])) * 100
        assert err < tol, f"{k}: {err:.4f}% >= {tol}%"
    for k in ("vth_fwd", "vth_rev"):
        assert abs(float(getattr(got, k)) - float(ref[k])) < VTH_ABS_TOL_V


def test_accuracy_across_the_whole_design_box():
    rng = np.random.default_rng(0)
    worst = dict.fromkeys(REL_TOL, 0.0)
    worst_vth = 0.0
    for _ in range(40):
        got, ref = _extract(rng.random(5))
        for k in REL_TOL:
            worst[k] = max(worst[k], abs(float(getattr(got, k)) - float(ref[k]))
                           / abs(float(ref[k])) * 100)
        for k in ("vth_fwd", "vth_rev"):
            worst_vth = max(worst_vth, abs(float(getattr(got, k)) - float(ref[k])))
    for k, tol in REL_TOL.items():
        assert worst[k] < tol, f"{k}: worst {worst[k]:.4f}% >= {tol}%"
    assert worst_vth < VTH_ABS_TOL_V, f"worst V_th error {worst_vth * 1e3:.4f} mV"


def test_extraction_is_insensitive_to_solver_noise():
    """A converged Newton solve is good to ~1e-6 relative. Extraction must not care."""
    rng = np.random.default_rng(7)
    theta = nominal_theta(5)
    cfg = extraction_config(theta)
    curves = np.asarray(id_vg_curves(theta, DEFAULT_VG_GRID, CC.v_ds))
    base = extract_foms(VG, jnp.asarray(curves[0]), jnp.asarray(curves[1]), cfg, CC.v_ds)
    noisy = curves * (1.0 + 1e-6 * rng.standard_normal(curves.shape))
    pert = extract_foms(VG, jnp.asarray(noisy[0]), jnp.asarray(noisy[1]), cfg, CC.v_ds)
    for k in ("ss", "i_leak", "g_lo", "g_hi", "dg_dvth"):
        shift = abs(float(getattr(pert, k)) - float(getattr(base, k))) / abs(float(getattr(base, k)))
        assert shift < 1e-3, f"{k} moved {shift:.2e} for 1e-6 curve noise"


def test_no_argmax_or_threshold_crossing_in_the_source():
    """Guard the property the whole module exists for, not just its consequences.

    Every one of these would reintroduce a piecewise-constant derivative in theta:
    a grid-cell selection that migrates as the curve moves.
    """
    import ast
    import inspect

    from diffsilicon.shared import extract

    # Parse rather than grep: this module's docstrings talk ABOUT argmax at
    # length, and prose explaining why something is absent must not read as its
    # presence.
    tree = ast.parse(inspect.getsource(extract))
    used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    used |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    banned = {
        "argmax", "argmin", "argsort", "searchsorted", "interp", "where",
        "nonzero", "take", "clip", "round", "floor", "ceil",
    }
    offenders = sorted(banned & used)
    assert not offenders, f"{offenders} in extract.py would break smoothness in theta"


def test_memory_window_sign_convention():
    """Forward = erased = high V_th, so MW = vth_fwd - vth_rev > 0 and g_lo < g_hi."""
    got, _ = _extract(nominal_theta(5))
    assert float(got.vth_fwd) > float(got.vth_rev)
    assert float(got.g_lo) < float(got.g_hi)
    assert float(got.vth_fwd - got.vth_rev) > 0.1  # the G5 memory-window criterion


def test_i_crit_scales_with_w_over_lg():
    """The constant-current criterion is 100 nA * W/L_g, so V_th must track L_g."""
    d = device_params(jnp.asarray(nominal_theta(5)))
    assert float(d["i_crit"]) == pytest.approx(100e-9 * float(d["W"]) / float(d["L_g"]))
