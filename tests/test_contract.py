"""The contract is frozen. These tests exist to make un-freezing it noisy."""

import numpy as np
import pytest

from diffsilicon.shared.contract import (
    DEFAULT_VG_GRID,
    DIFFERENTIABLE_OUTPUTS,
    NVG,
    OracleInput,
    OracleOutput,
    make_oracle_input,
)
from diffsilicon.shared.design import DESIGN_VECTORS, denormalise, get_design, normalise

FROZEN_INPUT_FIELDS = ("theta", "vg_grid", "vds_lin", "vds_sat")
FROZEN_OUTPUT_FIELDS = (
    "ss", "vth_fwd", "vth_rev", "i_leak", "g_lo", "g_hi", "dg_dvth",
    "id_vg", "converged", "solver_seconds",
)


def test_input_fields_frozen():
    assert tuple(OracleInput.model_fields) == FROZEN_INPUT_FIELDS


def test_output_fields_frozen():
    assert tuple(OracleOutput.model_fields) == FROZEN_OUTPUT_FIELDS


def test_jacobian_row_order_frozen():
    # This order fixes the row order of the 7 x D Jacobian in the shim, the
    # cotangent layout on the wire, and the column order of every cached record.
    assert DIFFERENTIABLE_OUTPUTS == (
        "ss", "vth_fwd", "vth_rev", "i_leak", "g_lo", "g_hi", "dg_dvth"
    )
    assert len(DIFFERENTIABLE_OUTPUTS) == 7


def test_sweep_grid_frozen():
    assert NVG == 96
    assert DEFAULT_VG_GRID.shape == (96,)
    assert DEFAULT_VG_GRID[0] == pytest.approx(-1.20)
    assert DEFAULT_VG_GRID[-1] == pytest.approx(1.40)


@pytest.mark.parametrize("D", [3, 5, 12])
def test_design_vectors_round_trip(D):
    spec = get_design(D)
    assert spec.D == D
    rng = np.random.default_rng(D)
    t = rng.random(D)
    assert np.allclose(normalise(denormalise(t, spec), spec), t)


def test_design_vectors_are_nested():
    # d=5 must extend d=3 and d=12 must extend d=5, in order. If they ever stop
    # being prefixes of each other, a d=3 cache entry silently means something
    # different from the first three columns of a d=5 one.
    n3, n5, n12 = (get_design(d).names for d in (3, 5, 12))
    assert n5[: len(n3)] == n3
    assert n12[: len(n5)] == n5


def test_theta_dimension_is_validated():
    with pytest.raises(ValueError, match="dimension 4"):
        make_oracle_input(np.zeros(4))


def test_valid_dimensions_accepted():
    for D in DESIGN_VECTORS:
        assert make_oracle_input(np.full(D, 0.5)).theta.shape == (D,)
