"""G4 -- the extraction-smoothness gate. Threshold: max 2nd diff / mean 1st diff < 0.15.

Why this gate matters more than it looks: argmax / threshold-crossing extraction
has a derivative in theta that is piecewise constant and kinks every time the
crossing migrates one grid cell. That staircase dominates finite-difference error
long before Newton tolerance does, and it is invisible unless you difference the
oracle -- which is exactly what the shim spends its whole budget doing.

The mock is constructed so that SS and both thresholds are EXACTLY LINEAR in
t_fe. Any second difference in the extracted values is therefore extraction
artefact and nothing else, which makes this a clean test rather than a
measurement of the mock's curvature.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from diffsilicon.shared.circuit import load_circuit
from diffsilicon.shared.contract import DEFAULT_VG_GRID
from diffsilicon.shared.design import nominal_theta
from diffsilicon.shared.extract import extract_foms
from diffsilicon.shared.mock_device import id_vg_curves
from diffsilicon.shared.oracle import extraction_config

CC = load_circuit()
VG = jnp.asarray(DEFAULT_VG_GRID)
G4_THRESHOLD = 0.15


def staircase_metric(values) -> float:
    v = np.asarray(values, dtype=np.float64)
    d1 = np.diff(v)
    d2 = np.diff(d1)
    denom = np.mean(np.abs(d1))
    if denom == 0.0:
        return 0.0
    return float(np.max(np.abs(d2)) / denom)


def sweep(param_index: int, n: int = 40, D: int = 5):
    base = nominal_theta(D)
    rows = []
    for t in np.linspace(0.05, 0.95, n):
        theta = base.copy()
        theta[param_index] = t
        cfg = extraction_config(theta)
        c = id_vg_curves(theta, DEFAULT_VG_GRID, CC.v_ds)
        rows.append(extract_foms(VG, c[0], c[1], cfg, CC.v_ds))
    return rows


@pytest.mark.parametrize("fom", ["ss", "vth_fwd", "vth_rev"])
def test_g4_forty_point_t_fe_sweep(fom):
    """The gate as specified: 40 points in t_fe, SS and V_th."""
    rows = sweep(0, n=40)
    m = staircase_metric([float(getattr(r, fom)) for r in rows])
    assert m < G4_THRESHOLD, f"G4 FAILED for {fom}: {m:.6f} >= {G4_THRESHOLD}"


@pytest.mark.parametrize("param_index,fom", [(3, "ss"), (4, "ss"), (3, "vth_fwd")])
def test_g4_holds_for_other_parameters(param_index, fom):
    """t_fe leaves SS almost flat by construction; L_g and N_ch are the real test."""
    rows = sweep(param_index, n=40)
    m = staircase_metric([float(getattr(r, fom)) for r in rows])
    assert m < G4_THRESHOLD, f"G4 FAILED for {fom} vs param {param_index}: {m:.6f}"


ALL_FOMS = ("ss", "vth_fwd", "vth_rev", "i_leak", "g_lo", "g_hi", "dg_dvth")


@pytest.mark.parametrize("fom", ALL_FOMS)
def test_metric_is_curvature_limited_not_staircased(fom):
    """The decisive continuity test: refine the sweep and watch the metric.

    For a smooth function the metric is h * f'' / f', so halving the step halves
    it. A staircase -- an extraction whose derivative jumps when a grid cell
    migrates -- does not shrink at all, because the jump is the same size however
    finely you sample theta around it.

    This replaces the naive "max second difference is small" check, which cannot
    tell curvature from a kink: g_lo and I_leak vary by two orders of magnitude
    across a t_fe sweep and dg/dV_th has a turning point in the middle of it, so
    any bound on a raw or slope-normalised second difference flags healthy
    behaviour and misses the thing it was written to catch.
    """
    m40 = staircase_metric([float(getattr(r, fom)) for r in sweep(0, n=40)])
    if m40 < 1e-4:
        return  # exactly linear in t_fe by construction; nothing left to refine
    m80 = staircase_metric([float(getattr(r, fom)) for r in sweep(0, n=80)])
    ratio = m80 / m40
    assert 0.40 < ratio < 0.62, (
        f"{fom}: metric went {m40:.4f} -> {m80:.4f} (ratio {ratio:.3f}) when the "
        f"sweep step halved. Smooth curvature halves; a staircase does not."
    )
