"""Regression tests for the D3 extraction rewrite.

The bug these exist to prevent was not subtle in its symptoms -- SS = 9618
mV/dec, memory window 30 V -- but it was completely invisible in the unit tests,
because the analytic mock has no leakage floor and no ambipolar tail, so its
curves never present the situation that broke the fit.

So the fixture here is a curve shaped like a REAL FeFET on the widened window:
a flat off-state floor, a DESCENDING ambipolar tail on the far negative side,
and a steep turn-on. It is built analytically, not copied from solver output,
because the solver output in question is the user's calibrated device and this
repository is public.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from diffsilicon.shared.contract import DEFAULT_VG_GRID
from diffsilicon.shared.extract import ExtractionConfig, extract_foms

VG = jnp.asarray(DEFAULT_VG_GRID)
CFG = ExtractionConfig(v_leak=0.246391250)


def branch(vth, ss_mv_per_dec, i_floor=1.5e-11, i_on=4.0e-5, v_dip=-0.7, ambipolar=0.35):
    """A FeFET branch with the three features the widened window exposes.

    * subthreshold: log-linear at `ss_mv_per_dec`, crossing 1e-10 A near `vth`
    * off-state: bottoms out on a floor at `i_floor`
    * ambipolar tail: the floor RISES again towards negative V_g, so the branch
      descends as V_g increases over part of the sweep -- which is what made the
      old fixed-current window average a falling branch against a rising one.
    * on-state: saturates at `i_on`
    """
    vg = np.asarray(DEFAULT_VG_GRID)
    ss_v = ss_mv_per_dec * 1e-3
    sub = 10.0 ** (np.log10(1e-10) + (vg - vth) / ss_v)
    tail = i_floor * 10.0 ** (ambipolar * np.maximum(v_dip - vg, 0.0))
    return 1.0 / (1.0 / (sub + i_floor + tail) + 1.0 / i_on)


def test_the_9618_case_is_fixed():
    """SS and the memory window on a real-shaped widened-window curve.

    The numbers asserted are the ones the fixture is built with, so this is a
    statement about the extraction and not about the fixture: an SS of 68 mV/dec
    must come back as 68 mV/dec even though most of the sweep is floor.
    """
    fwd = branch(vth=+0.75, ss_mv_per_dec=68.0)
    rev = branch(vth=-1.35, ss_mv_per_dec=50.0)
    f = extract_foms(VG, jnp.asarray(fwd), jnp.asarray(rev), CFG, 0.05)

    assert float(f.ss) == pytest.approx(68.0, rel=0.10), (
        f"SS came back as {float(f.ss):.1f} mV/dec on a branch built at 68. "
        "This is the D3 failure mode: a window centred on a fixed current level "
        "lands on the leakage floor and the fit averages the falling ambipolar "
        "tail against the rising turn-on."
    )
    mw = float(f.vth_fwd - f.vth_rev)
    assert 1.8 < mw < 2.4, f"memory window {mw:.2f} V, expected about 2.1"


def test_i_leak_is_the_floor_not_an_extrapolation():
    """On a device with a floor, I_leak at V_leak IS the floor.

    Extrapolating the subthreshold line down to V_leak gives ~1e-15 A here. The
    device actually draws its floor current, four decades more, and I_leak sets
    the DPI leak that fixes beta -- so this is not a cosmetic difference.
    """
    fwd = branch(vth=+0.75, ss_mv_per_dec=68.0, i_floor=1.5e-11)
    rev = branch(vth=-1.35, ss_mv_per_dec=50.0)
    f = extract_foms(VG, jnp.asarray(fwd), jnp.asarray(rev), CFG, 0.05)

    i = int(np.argmin(np.abs(np.asarray(DEFAULT_VG_GRID) - CFG.v_leak)))
    on_curve = float(fwd[i])
    assert float(f.i_leak) == pytest.approx(on_curve, rel=0.25), (
        f"I_leak {float(f.i_leak):.3e} against a curve value of {on_curve:.3e} at V_leak"
    )
    assert float(f.i_leak) > 1e-12, "I_leak collapsed to the extrapolated line"


def test_the_falling_ambipolar_tail_is_excluded():
    """Making the tail steeper must not move SS.

    The tail is a parasitic conduction path outside the operating window. If it
    can move the reported subthreshold swing, the swing is not being measured on
    the subthreshold region.
    """
    rev = branch(vth=-1.35, ss_mv_per_dec=50.0)
    out = []
    for ambipolar in (0.0, 0.35, 0.9):
        fwd = branch(vth=+0.75, ss_mv_per_dec=68.0, ambipolar=ambipolar)
        out.append(float(extract_foms(VG, jnp.asarray(fwd), jnp.asarray(rev), CFG, 0.05).ss))
    spread = (max(out) - min(out)) / np.mean(out)
    assert spread < 0.02, f"SS moved {spread * 100:.1f}% with the ambipolar tail: {out}"


def test_the_off_state_floor_is_excluded():
    """Raising or lowering the floor by a decade must not move SS or V_th."""
    rev = branch(vth=-1.35, ss_mv_per_dec=50.0)
    ss, vth = [], []
    for i_floor in (1.5e-12, 1.5e-11, 1.5e-10):
        fwd = branch(vth=+0.75, ss_mv_per_dec=68.0, i_floor=i_floor)
        f = extract_foms(VG, jnp.asarray(fwd), jnp.asarray(rev), CFG, 0.05)
        ss.append(float(f.ss))
        vth.append(float(f.vth_fwd))
    assert (max(ss) - min(ss)) / np.mean(ss) < 0.05, f"SS tracked the floor: {ss}"
    assert max(vth) - min(vth) < 0.05, f"V_th tracked the floor: {vth}"


def test_the_local_window_covers_more_than_one_grid_cell():
    """The degree-9 fit needs support, and the widened sweep doubled the spacing.

    This is the constant that quietly broke when the window widened: 25 mV was
    under one grid spacing at 96 points over 2.6 V, and is half a spacing at 96
    points over 5.0 V. Ten polynomial coefficients over three weighted points is
    not a fit, and the failure is silent because the Tikhonov term absorbs it.
    """
    from diffsilicon.shared.extract import _local_window

    dv = float(DEFAULT_VG_GRID[1] - DEFAULT_VG_GRID[0])
    assert float(_local_window(VG, CFG)) > dv, (
        "the local-polynomial window is narrower than one grid cell"
    )


def test_extraction_is_smooth_in_a_shifting_threshold():
    """The whole module exists to keep derivatives free of grid staircases.

    V_th is swept in steps far finer than the 52.6 mV grid spacing. If any part
    of the extraction selected a grid cell, dV_th(extracted)/dV_th(true) would
    show a visible staircase; here it must stay close to 1.
    """
    rev = branch(vth=-1.35, ss_mv_per_dec=50.0)
    vths = np.linspace(0.70, 0.80, 41)
    got = np.array(
        [
            float(
                extract_foms(
                    VG, jnp.asarray(branch(vth=v, ss_mv_per_dec=68.0)),
                    jnp.asarray(rev), CFG, 0.05,
                ).vth_fwd
            )
            for v in vths
        ]
    )
    slope = np.diff(got) / np.diff(vths)
    assert np.all(slope > 0.85), f"V_th response stalls: min slope {slope.min():.3f}"
    assert np.all(slope < 1.15), f"V_th response jumps: max slope {slope.max():.3f}"
