# SPDX-License-Identifier: Apache-2.0
"""sentaurus-fefet (T1) -- the flagship: a closed-source commercial TCAD solver.

Synopsys Sentaurus 2023.12 (`sde` + `sdevice`, Preisach ferroelectric), Fortran
and C++, driven from csh on a CentOS 7 host that has Python 2.7.5 and nothing
else on it. No AD path exists and none ever will.

This Tesseract runs LOCALLY on Windows and reaches the solver over plink/pscp.
It does not serve on the Sentaurus host, and nothing was installed there. See
docs/T1_CONTAINER.md for why the flagship is the one Tesseract that runs
uncontainerised, and t1/Dockerfile for the container that exists anyway.

Measured: one sdevice run is 306 s, exit status 0. There is exactly ONE sdevice
license, so calls serialise.

No gradients, by design.
"""

import sys
from pathlib import Path

for _cand in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[2] / "src"):
    if (_cand / "diffsilicon").is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from tesseract_core.runtime import ShapeDType  # noqa: E402

from diffsilicon.shared.contract import NVG, OracleInput, OracleOutput  # noqa: E402
from diffsilicon.shared.oracle import run_oracle  # noqa: E402


class InputSchema(OracleInput):
    """The frozen contract, unchanged. Subclassed rather than aliased only so the
    generated OpenAPI component is named InputSchema, which tesseract-jax requires."""


class OutputSchema(OracleOutput):
    """The frozen contract, unchanged. See InputSchema."""

BACKEND = "sentaurus"


def apply(inputs: InputSchema) -> OutputSchema:
    # dict() keeps the raw arrays; the runtime validates against this module's
    # OutputSchema, which is the frozen contract under a different __name__.
    return OutputSchema(**dict(run_oracle(inputs, BACKEND)))


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
