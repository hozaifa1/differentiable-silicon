"""G5 -- the DEVSIM FeFET oracle: Miller algebra, branch sign, memory window.

The cheap tests here are pure algebra and template rendering and run in CI on
every push. The one that costs a solver call is marked `slow`: a full double
sweep is 96 x 2 bias points and takes 36-40 s.

Why the branch-sign test is the one that matters. Everything downstream reads
`MW = vth_fwd - vth_rev` and takes `g_lo` from the forward branch and `g_hi` from
the reverse. Invert that convention and the memory window comes out negative,
`sigma_w` picks up a sign, the transducer keeps producing finite plausible
numbers, and nothing fails until someone reads a figure. The build spec HAD it
inverted; D1 corrected it, and this is the test that keeps it corrected.
"""

import math
import pathlib

import numpy as np
import pytest

from diffsilicon.oracle_devsim import (
    EPS_FE_BG,
    FE_ACTIVE_FRACTION,
    MU_N_EFF,
    PHI_M_GATE,
    fefet_params,
)
from diffsilicon.shared.contract import DEFAULT_VG_GRID, NVG, make_oracle_input
from diffsilicon.shared.design import get_design, nominal_theta


def _has_calibration() -> bool:
    """Is there a device on this machine to render a deck against?

    The calibration and the deck are one person's unpublished thesis work and are
    gitignored, so CI has neither and never will. Tests that only need algebra or
    a unit conversion must run everywhere; tests that need an actual device skip.
    Skipping is the honest outcome here -- the alternative is a suite that is
    green on one laptop and red everywhere else, which is what happened.
    """
    from diffsilicon.t1_driver import T1Config

    t1 = pathlib.Path(__file__).resolve().parents[1] / "t1"
    return (
        not T1Config().missing_calibration()
        and (t1 / "sdevice_fefet_idvg.cmd").is_file()
        and (t1 / "sdevice_fefet_idvg.par").is_file()
    )


needs_calibration = pytest.mark.skipif(
    not _has_calibration(),
    reason="no T1 device calibration or deck on this machine (both are gitignored)",
)


# ---------------------------------------------------------------- Miller algebra
def test_miller_delta_puts_the_branch_through_minus_pr_at_zero_field():
    """delta is DEFINED by tanh(Ec/2delta) = Pr/Ps, so that the saturated branch
    passes through P = -Pr at E = 0 and reaches P = 0 at E = Ec. If this identity
    ever stops holding, the model still runs and still hystereses -- it just
    stops being Miller's."""
    for D in (3, 5, 12):
        p = fefet_params(nominal_theta(D))
        assert p.ps > p.pr > 0.0
        assert math.isclose(math.tanh(p.ec / (2.0 * p.delta)), p.pr / p.ps, rel_tol=1e-12)


def test_gamma_squares_the_loop_and_stays_physical():
    """Gamma is the d=12 minor-loop saturation factor: r = Pr/Ps must stay in
    (0, 1) at both ends of its box, or delta is undefined."""
    spec = get_design(12)
    for g in (0.0, 1.0):
        theta = nominal_theta(12)
        theta[spec.names.index("Gamma")] = g
        p = fefet_params(theta)
        r = p.pr / p.ps
        assert 0.0 < r < 1.0
        assert p.delta > 0.0


def test_defaults_match_the_mock_so_a_d3_point_is_the_same_device():
    """A d=3 theta still has an L_g and an N_ch. They have to be the numbers the
    mock uses, or V4 measures a difference in defaults instead of in physics."""
    p3 = fefet_params(nominal_theta(3))
    p5 = fefet_params(nominal_theta(5))
    assert p3.l_g == pytest.approx(p5.l_g)
    assert p3.n_ch == pytest.approx(p5.n_ch)
    assert p3.l_g / 1e-7 == pytest.approx(40.0)
    assert p3.n_ch == pytest.approx(1e17)
    assert p3.w_dev / 1e-7 == pytest.approx(100.0)


def test_the_three_chosen_constants_are_in_physical_ranges():
    """Not a tuning test -- a tripwire. Each of these is a scalar someone could
    quietly adjust to make a plot look better, so each is pinned to the range its
    physical justification supports."""
    assert 80.0 <= MU_N_EFF <= 300.0, "inversion-layer mobility under a high-k stack"
    assert 4.1 <= PHI_M_GATE <= 5.2, "a real metal gate work function, eV"
    assert 0.0 < FE_ACTIVE_FRACTION < 0.5, "switching-active, unscreened film fraction"
    assert 20.0 <= EPS_FE_BG <= 45.0, "background relative permittivity of HZO"


# ------------------------------------------------------------- the T1 deck tokens
@needs_calibration
def test_the_t1_deck_renders_for_every_design_vector():
    from pathlib import Path

    from diffsilicon.t1_driver import content_tag, deck_values, render_template

    t1 = Path(__file__).resolve().parents[1] / "t1"
    cmd = (t1 / "sdevice_fefet_idvg.cmd").read_text(encoding="utf-8")
    par = (t1 / "sdevice_fefet_idvg.par").read_text(encoding="utf-8")
    for D in (3, 4, 5, 12):
        v = deck_values(make_oracle_input(nominal_theta(D)))
        tag = "ds_" + content_tag(v)
        v.update(plot=f"{tag}_des.plt", tdrdat=f"{tag}_des.tdr",
                 log=f"{tag}_des.log", par=f"{tag}_des.par")
        render_template(cmd, v)  # raises on an unresolved token
        render_template(par, v)


@needs_calibration
def test_the_fixed_slab_remap_preserves_the_coercive_voltage():
    """eps and Ec scale by RECIPROCAL factors. Using one for both renders fine,
    runs fine, and silently multiplies the memory window by (t_slab/t_fe)^2.
    The invariant that catches it: Ec_eff * t_slab == Ec * t_fe."""
    from diffsilicon.t1_driver import T1Config, deck_values

    cfg = T1Config()
    t_slab_cm = cfg.t_fe_slab_nm * 1e-7
    for t_fe_n in (0.0, 0.5, 1.0):
        theta = nominal_theta(3)
        theta[0] = t_fe_n
        v = deck_values(make_oracle_input(theta), cfg)
        p = fefet_params(theta)
        # coercive VOLTAGE preserved: Ec_eff * t_slab == Ec * t_fe.
        # FC is the token the deck actually receives, so assert on that rather
        # than on a bookkeeping copy of it -- a remap that is computed and then
        # not shipped is the failure this is guarding against.
        assert float(v["FC"]) * t_slab_cm == pytest.approx(p.ec * p.t_fe, rel=1e-9)
        # capacitance per area preserved: eps_eff / t_slab == eps_bg / t_fe.
        #
        # The base is the COMMERCIAL deck's own permittivity from
        # calibration.local.json, NOT oracle_devsim.EPS_FE_BG. Those are two
        # different solvers' films and using T2's constant here would silently
        # re-fit T1 away from the hysteresis it was calibrated against.
        assert float(v["EPS_FE_EFF"]) / t_slab_cm == pytest.approx(
            cfg.eps_fe_bg / p.t_fe, rel=1e-9
        )

    # And at the calibrated thickness the remap is the identity, so nothing
    # about the fitted device moves when t_fe happens to equal the slab.
    theta = nominal_theta(3)
    theta[0] = (cfg.t_fe_slab_nm - 5.0) / 10.0  # t_fe = t_fe_slab_nm
    v = deck_values(make_oracle_input(theta), cfg)
    assert float(v["EPS_FE_EFF"]) == pytest.approx(cfg.eps_fe_bg, rel=1e-9)
    assert float(v["t_fe_remap_k"]) == pytest.approx(1.0, rel=1e-9)


# ------------------------------------------------------------------- the solve
@pytest.mark.slow
@pytest.mark.needs_devsim
def test_g5_memory_window_and_branch_sign():
    """G5 as specified: MW = vth_fwd - vth_rev > 0.1 V on the frozen grid.

    Runs the oracle out of process (the module always does; DEVSIM and torch
    cannot share one) and extracts with the frozen extractor, so this is the same
    number the gate was stated on, not a re-derivation of it.
    """
    pytest.importorskip("devsim")
    import jax.numpy as jnp

    from diffsilicon.oracle_devsim import id_vg_curves
    from diffsilicon.shared.extract import extract_foms
    from diffsilicon.shared.oracle import extraction_config

    inp = make_oracle_input(nominal_theta(3))
    curves = np.asarray(id_vg_curves(inp))

    assert curves.shape == (2, NVG)
    assert np.all(np.isfinite(curves))
    assert np.all(curves >= 0.0)

    foms = extract_foms(
        jnp.asarray(DEFAULT_VG_GRID), jnp.asarray(curves[0]), jnp.asarray(curves[1]),
        extraction_config(inp.theta), float(inp.vds_lin),
    )
    mw = float(foms.vth_fwd) - float(foms.vth_rev)
    assert mw > 0.1, f"G5: memory window {mw:.4f} V"
    # The sign convention, stated as an assertion rather than as a comment.
    assert float(foms.g_hi) > float(foms.g_lo), "reverse (programmed) must be the ON state"
    assert 50.0 < float(foms.ss) < 200.0, f"SS {float(foms.ss):.1f} mV/dec is not a MOSFET"


# ------------------------------------------------------- the mesh-rebuild path
def test_the_mesh_template_takes_all_four_design_variables():
    """The whole point: gate length, doping and interfacial thickness reach sde.

    Without this the commercial solver sees only t_fe, and three of the four d=4
    Jacobian columns are identically zero.
    """
    from pathlib import Path

    tmpl = Path(__file__).resolve().parents[1] / "t1" / "sde_fefet_mesh.cmd"
    if not tmpl.is_file():
        pytest.skip("mesh template is gitignored and not on this machine")
    text = tmpl.read_text(encoding="utf-8")
    for token in ("@L_GATE@", "@T_OX@", "@T_FE@", "@N_SUB@", "@MESH@"):
        assert token in text, f"{token} missing from the mesh template"


def test_mesh_values_are_in_microns():
    """sde works in microns; the design vector is nm and cm^-3.

    A thousand-fold slip here builds a device a thousand times too big, and it
    still meshes, still solves, and still returns a curve.
    """
    from diffsilicon.t1_driver import mesh_values

    v = mesh_values(make_oracle_input(nominal_theta(4)))
    assert float(v["L_GATE"]) == pytest.approx(0.040)  # 40 nm
    assert float(v["T_FE"]) == pytest.approx(0.007)  # 7 nm, the d=4 nominal
    assert float(v["T_OX"]) == pytest.approx(0.001)  # 1 nm
    assert float(v["N_SUB"]) == pytest.approx(1e17)  # cm^-3, NOT converted


def test_the_mesh_deck_renders_with_no_leftover_tokens():
    from pathlib import Path

    from diffsilicon.t1_driver import mesh_values, render_template

    tmpl = Path(__file__).resolve().parents[1] / "t1" / "sde_fefet_mesh.cmd"
    if not tmpl.is_file():
        pytest.skip("mesh template is gitignored and not on this machine")
    v = {**mesh_values(make_oracle_input(nominal_theta(4))), "MESH": "ds_test_msh"}
    render_template(tmpl.read_text(encoding="utf-8"), v)  # raises on a leftover


@needs_calibration
def test_rebuilding_the_mesh_turns_the_remap_OFF():
    """The double-count trap, and it is silent.

    If the mesh is built at the film's true thickness AND the fixed-slab remap is
    applied on top, the thickness is counted twice -- once in the geometry, once
    in the material -- and the memory window comes out wrong by (t_fe/t_slab)^2
    on a deck that renders and runs perfectly.
    """
    from diffsilicon.t1_driver import T1Config, deck_values

    cfg = T1Config()
    theta = nominal_theta(4)
    theta[0] = 1.0  # t_fe = 15 nm, far from the 7 nm slab, so k is nowhere near 1

    remapped = deck_values(make_oracle_input(theta), cfg, mesh_is_exact=False)
    exact = deck_values(make_oracle_input(theta), cfg, mesh_is_exact=True)

    assert float(remapped["t_fe_remap_k"]) != pytest.approx(1.0)
    assert float(exact["t_fe_remap_k"]) == pytest.approx(1.0)
    # On an exact mesh the deck gets the film's OWN coercive field and
    # permittivity, untouched.
    p = fefet_params(theta)
    assert float(exact["FC"]) == pytest.approx(p.ec, rel=1e-9)
    assert float(exact["EPS_FE_EFF"]) == pytest.approx(cfg.eps_fe_bg, rel=1e-9)
    assert float(remapped["FC"]) != pytest.approx(p.ec, rel=1e-6)


def test_rebuild_is_off_by_default():
    """An unattended overnight run must not be the first thing to try this."""
    import os

    from diffsilicon.t1_driver import T1Config

    assert os.environ.get("T1_REBUILD_MESH", "0") != "1" or T1Config().rebuild_mesh
    if os.environ.get("T1_REBUILD_MESH", "0") != "1":
        assert T1Config().rebuild_mesh is False


def test_the_baseline_geometry_refuses_to_be_guessed():
    """The control case needs the REAL geometry cfg.grid was built at.

    It is not known, and the code says so rather than defaulting to something
    plausible. A plausible wrong baseline would make a rebuilt mesh look
    validated when it is not.
    """
    from diffsilicon.t1_driver import T1Config, baseline_mesh_values

    cfg = T1Config()
    if cfg.l_gate_slab_nm and cfg.t_il_slab_nm and cfg.n_ch_slab:
        v = baseline_mesh_values(cfg)
        assert float(v["T_FE"]) == pytest.approx(cfg.t_fe_slab_nm * 1e-3)
    else:
        with pytest.raises(ValueError, match="missing"):
            baseline_mesh_values(cfg)
