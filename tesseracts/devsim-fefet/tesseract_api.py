# SPDX-License-Identifier: Apache-2.0
"""devsim-fefet (T2) -- the Apache-2.0 reproducible twin.

DEVSIM 2.10, an Apache-2.0 C++ finite-volume drift-diffusion core, with a
clean-room Miller ferroelectric gate (Miller & McWhorter, J. Appl. Phys. 72,
5999, 1992). QS-Devsim is deliberately NOT used: non-commercial licence plus
patent CN 113297818 B, both incompatible with the Apache-2.0 this ships under.

Byte-identical schema to `sentaurus-fefet`. That is the entire point -- swapping
the closed-source commercial solver for this one is one environment variable,
and nothing downstream can tell the difference except the `backend` string.

No gradients. `adjoint-shim` supplies those.
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

from tesseract_core.runtime import ShapeDType  # noqa: E402

from diffsilicon.shared.contract import NVG, OracleInput, OracleOutput  # noqa: E402
from diffsilicon.shared.oracle import run_oracle  # noqa: E402


class InputSchema(OracleInput):
    """The frozen contract, unchanged. Subclassed rather than aliased only so the
    generated OpenAPI component is named InputSchema, which tesseract-jax requires."""


class OutputSchema(OracleOutput):
    """The frozen contract, unchanged. See InputSchema."""

BACKEND = "devsim"


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
