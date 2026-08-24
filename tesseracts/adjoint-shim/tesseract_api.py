# SPDX-License-Identifier: Apache-2.0
"""adjoint-shim (T3) -- retrofit a validated VJP onto a solver that has none.

Publishes the SAME frozen contract as `sentaurus-fefet` and `devsim-fefet`, and
adds `jacobian`, `jacobian_vector_product` and `vector_jacobian_product` on top
of it. Which solver sits underneath is decided by ORACLE_BACKEND / ORACLE_URL and
by nothing else in this file.

`apply` PROXIES. It calls the oracle and returns what the oracle returned, so the
forward pass is never a surrogate; only the derivative is manufactured, from
directed finite-difference probes of that same solver, with a trust region
forcing ground-truth refreshes. `vector_jacobian_product` costs ZERO solver calls
whenever the local model is fresh -- which is the entire economic argument for
the apparatus, and it is measured in V7 rather than asserted here.

This is also the reusable artifact: `mosaic/benchmarks/problems/thermal_mesh/
exclusions.py` excludes deal.II from every gradient experiment with a categorical
"the C++ solver ships no AD path" label. This Tesseract is exactly the thing that
fills that cell.
"""

import sys
from pathlib import Path

# Locally the package lives at <repo>/src/diffsilicon; inside a built Tesseract,
# build_config.package_data drops the same tree beside this file at
# /tesseract/diffsilicon. Note the `parents` slice has to be bounded: in the
# container this file sits two levels from the filesystem root, and indexing past
# it raises IndexError -- which surfaces only as "Could not load module from
# /tesseract/tesseract_api.py" during `tesseract build`.
_HERE = Path(__file__).resolve()
_CANDIDATES = [_HERE.parent, *(p / "src" for p in _HERE.parents)]
for _cand in _CANDIDATES:
    if (_cand / "diffsilicon").is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))
        break

import numpy as np  # noqa: E402
from tesseract_core.runtime import ShapeDType  # noqa: E402

from diffsilicon.shared.contract import (  # noqa: E402
    DIFFERENTIABLE_OUTPUTS,
    NVG,
    OracleInput,
    OracleOutput,
)
from diffsilicon.shared.oracle import run_oracle  # noqa: E402
from diffsilicon.shim.adjoint import AdjointShim, shim_for, y_vector  # noqa: E402


class InputSchema(OracleInput):
    """The frozen contract, unchanged. Subclassed rather than aliased only so the
    generated OpenAPI component is named InputSchema, which tesseract-jax requires."""


class OutputSchema(OracleOutput):
    """The frozen contract, unchanged. See InputSchema."""

def _shim_for(inputs: InputSchema) -> AdjointShim:
    """The per-grid shim, from the registry in diffsilicon.shim.adjoint.

    The registry lives in the library rather than in this file so that an
    in-process orchestrator holds the SAME object these endpoints do and can feed
    it the measured trust-region rho. Served in a container it has no such handle
    and the shim runs on its own staleness rule; both paths are supported and
    neither is required by the other.
    """
    return shim_for(inputs)


def apply(inputs: InputSchema) -> OutputSchema:
    """Straight through to the oracle. No surrogate, no interpolation, no cache trick.

    The shim is TOLD the result on the way past. A forward evaluation is a true
    (theta, y) pair that the caller already paid for, and handing it to the local
    model is what lets Broyden keep J alive between refreshes for free.
    """
    out = run_oracle(inputs)
    _shim_for(inputs).observe(np.asarray(inputs.theta, dtype=np.float64), y_vector(out))
    return OutputSchema(**dict(out))


def abstract_eval(abstract_inputs):
    scalar = ShapeDType(shape=(), dtype="float64")
    return {
        "ss": scalar,
        "vth_fwd": scalar,
        "vth_rev": scalar,
        "i_leak": scalar,
        "g_lo": scalar,
        "g_hi": scalar,
        "dg_dvth": scalar,
        "id_vg": ShapeDType(shape=(2, NVG), dtype="float64"),
        "converged": scalar,
        "solver_seconds": scalar,
    }


def _rows(keys) -> list[int]:
    return [i for i, k in enumerate(DIFFERENTIABLE_OUTPUTS) if k in keys]


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    """The 7 x D Jacobian. Only `theta` is differentiable, so jac_inputs is {'theta'}."""
    if jac_inputs - {"theta"}:
        raise ValueError(f"only 'theta' is differentiable; got {sorted(jac_inputs)}")
    J = _shim_for(inputs).jacobian(np.asarray(inputs.theta, dtype=np.float64))
    return {k: {"theta": J[i]} for i, k in enumerate(DIFFERENTIABLE_OUTPUTS) if k in jac_outputs}


def jacobian_vector_product(
    inputs: InputSchema, jvp_inputs: set[str], jvp_outputs: set[str], tangent_vector
):
    if jvp_inputs - {"theta"}:
        raise ValueError(f"only 'theta' is differentiable; got {sorted(jvp_inputs)}")
    shim = _shim_for(inputs)
    tan = np.asarray(tangent_vector["theta"], dtype=np.float64).ravel()
    full = shim.jvp(np.asarray(inputs.theta, dtype=np.float64), tan)
    return {k: full[i] for i, k in enumerate(DIFFERENTIABLE_OUTPUTS) if k in jvp_outputs}


def vector_jacobian_product(
    inputs: InputSchema, vjp_inputs: set[str], vjp_outputs: set[str], cotangent_vector
):
    """gbar_theta = J^T gbar_y. Zero solver calls when J is fresh."""
    if vjp_inputs - {"theta"}:
        raise ValueError(f"only 'theta' is differentiable; got {sorted(vjp_inputs)}")
    shim = _shim_for(inputs)
    ct = np.zeros(len(DIFFERENTIABLE_OUTPUTS), dtype=np.float64)
    for i, k in enumerate(DIFFERENTIABLE_OUTPUTS):
        if k in vjp_outputs:
            ct[i] = float(np.asarray(cotangent_vector[k]))
    return {"theta": shim.vjp(np.asarray(inputs.theta, dtype=np.float64), ct)}
