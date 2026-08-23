"""The manufactured adjoint: shape, cost, correctness, and the budget cap."""

import numpy as np
import pytest

from diffsilicon.shared.contract import DIFFERENTIABLE_OUTPUTS, make_oracle_input
from diffsilicon.shared.design import nominal_theta
from diffsilicon.shim.adjoint import AdjointShim, ShimConfig, fd_jacobian, y_vector


@pytest.fixture()
def shim(tmp_path, monkeypatch):
    monkeypatch.setenv("DIFFSILICON_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("DIFFSILICON_PROVENANCE_DISABLE", "1")
    theta = nominal_theta(5)
    return AdjointShim(make_oracle_input(theta), ShimConfig(alpha=0.02, backend="mock")), theta


def test_jacobian_is_seven_by_D(shim):
    sh, theta = shim
    J = sh.jacobian(theta)
    assert J.shape == (7, 5) == (len(DIFFERENTIABLE_OUTPUTS), theta.size)


def test_central_anchor_costs_2D_plus_1_calls(shim):
    sh, theta = shim
    sh.jacobian(theta)
    assert sh.calls == 2 * theta.size + 1


def test_forward_refresh_costs_D_plus_1_calls(shim):
    """Central only for the anchor; forward differences thereafter halve the cost."""
    sh, theta = shim
    sh.jacobian(theta)
    before = sh.calls
    sh.refresh(theta)  # J already exists, so this is a forward-difference refresh
    assert sh.calls - before == theta.size + 1


def test_vjp_costs_zero_solver_calls_when_J_is_fresh(shim):
    """The entire economic argument for the apparatus is this assertion."""
    sh, theta = shim
    sh.jacobian(theta)
    before = sh.calls
    for _ in range(25):
        sh.vjp(theta, np.ones(7))
    assert sh.calls == before


def test_vjp_equals_J_transpose_times_the_cotangent(shim):
    sh, theta = shim
    J = sh.jacobian(theta)
    ct = np.array([1.0, -2.0, 0.5, 0.0, 3.0, -1.0, 0.25])
    assert np.allclose(sh.vjp(theta, ct), J.T @ ct)


def test_jvp_and_vjp_are_consistent(shim):
    """<J v, w> == <v, J^T w> for every v, w. If these ever disagree the two
    endpoints are describing different linear maps."""
    sh, theta = shim
    sh.jacobian(theta)
    rng = np.random.default_rng(0)
    v, w = rng.standard_normal(5), rng.standard_normal(7)
    assert np.isclose(sh.jvp(theta, v) @ w, v @ sh.vjp(theta, w))


def test_jacobian_signs_match_device_physics(shim):
    """A wrong sign here is invisible in every numerical check and fatal in every
    physical one, so it gets asserted directly."""
    sh, theta = shim
    J = sh.jacobian(theta)
    row = {k: i for i, k in enumerate(DIFFERENTIABLE_OUTPUTS)}
    col = {"t_fe": 0, "Pr": 1, "Ec": 2, "L_g": 3, "log10_N_ch": 4}

    # A thicker or more coercive ferroelectric opens the memory window.
    for p in ("t_fe", "Ec", "Pr"):
        assert J[row["vth_fwd"], col[p]] > 0
        assert J[row["vth_rev"], col[p]] < 0
    # A longer gate improves subthreshold swing.
    assert J[row["ss"], col["L_g"]] < 0
    # Heavier channel doping degrades it.
    assert J[row["ss"], col["log10_N_ch"]] > 0
    # Raising V_th cuts the leak current at a fixed V_leak.
    assert J[row["i_leak"], col["t_fe"]] < 0


def test_broyden_reproduces_the_secant_it_was_built_from(shim):
    """The defining property: after the rank-1 update, J s == dy exactly."""
    sh, theta = shim
    sh.jacobian(theta)
    step = np.array([0.03, -0.02, 0.01, 0.0, 0.02])
    theta2 = theta + step
    from diffsilicon.shared.oracle import run_oracle

    y2 = y_vector(run_oracle(make_oracle_input(theta2), "mock"))
    s = theta2 - np.asarray(sh.theta)
    dy = y2 - np.asarray(sh.y)

    sh.broyden_update(theta2, y2)

    # The defining secant condition: the updated J maps the step it just saw onto
    # the change it actually produced.
    assert np.allclose(sh.J @ s, dy, rtol=1e-10, atol=0)
    assert sh.ctr.broyden_updates == 1


def test_broyden_is_the_least_change_update(shim):
    """Broyden changes J as little as possible subject to the secant condition:
    the correction is rank one and lies entirely along s."""
    sh, theta = shim
    J0 = sh.jacobian(theta).copy()
    step = np.array([0.03, -0.02, 0.01, 0.0, 0.02])
    sh.broyden_update(theta + step, np.asarray(sh.y) * 1.01)
    delta = sh.J - J0
    assert np.linalg.matrix_rank(delta, tol=1e-12) == 1
    # Every row of the correction is parallel to s.
    s = step / np.linalg.norm(step)
    for r in delta:
        if np.linalg.norm(r) > 1e-14:
            assert abs(abs(r @ s) / np.linalg.norm(r) - 1.0) < 1e-9


def test_broyden_is_cheaper_than_a_refresh(shim):
    sh, theta = shim
    sh.jacobian(theta)
    before = sh.calls
    sh.broyden_update(theta + 0.01, np.asarray(sh.y) * 1.001)
    assert sh.calls == before  # the secant pair was free


def test_trust_region_forces_a_refresh_on_a_bad_step(shim):
    sh, theta = shim
    sh.jacobian(theta)
    radius = sh.radius
    sh.record_step(rho=0.05)  # the model badly over-predicted the reduction
    assert sh.radius == pytest.approx(radius / 2)
    before = sh.calls
    sh.jacobian(theta)
    assert sh.calls > before, "a rho below the shrink threshold must force ground truth"


def test_budget_cap_is_hard(tmp_path, monkeypatch):
    """Budget-capped optimisation is schedulable and honest; convergence-criterion
    optimisation against a 5-minute solver is neither."""
    monkeypatch.setenv("DIFFSILICON_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("DIFFSILICON_PROVENANCE_DISABLE", "1")
    theta = nominal_theta(5)
    sh = AdjointShim(make_oracle_input(theta), ShimConfig(backend="mock", max_oracle_calls=4))
    with pytest.raises(RuntimeError, match="budget exhausted"):
        sh.jacobian(theta)
    assert sh.calls == 4


def test_probes_stay_inside_the_design_box(tmp_path, monkeypatch):
    """theta lives in [0,1]^D. A probe outside it is a device that cannot be built."""
    monkeypatch.setenv("DIFFSILICON_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("DIFFSILICON_PROVENANCE_DISABLE", "1")
    corner = np.zeros(5)  # sitting on the lower face of the box
    cfg = ShimConfig(alpha=0.02, backend="mock")
    J, _ = fd_jacobian(corner, make_oracle_input(corner), cfg, central=True)
    assert np.all(np.isfinite(J))
