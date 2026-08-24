"""THE FROZEN CONTRACT.

Every Tesseract in this project, both oracles, the replay cache, the T1 remote
driver, the orchestrator and every test import these two schemas and nothing
else. Freezing this on day 1 is what makes "swapping the closed-source
commercial solver for the Apache-2.0 one is one environment variable" true
rather than aspirational: T1 and T2 are byte-identical in schema, so nothing
downstream can tell them apart except the `backend` provenance string.

FROZEN 2026-08-23, with ONE amendment on D2: the sweep window widened from
[-1.20, 1.40] to [-3.50, 1.50] because the real device's memory window does not
fit in 2.6 V. Fields may not be renamed, reordered, retyped, or removed. Note the
grid is part of the cache key by construction -- it is inside the hashed inputs --
so every cached result from the old window stops matching automatically.

The Jacobian this contract implies is 7 x D, not 192 x D. No efficiency claim is
made about output dimensionality -- a scalar loss over D design variables needs
D+1 forward-difference calls and always did. The apparatus is justified
empirically in V7, not rhetorically.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import BaseModel, Field, model_validator
from tesseract_core.runtime import Array, Differentiable, Float64

from .design import DESIGN_VECTORS

__all__ = [
    "NVG",
    "DEFAULT_VG_GRID",
    "DIFFERENTIABLE_OUTPUTS",
    "OracleInput",
    "OracleOutput",
    "make_oracle_input",
]

# --- Fixed sweep grid ----------------------------------------------------------
NVG = 96
# WIDENED 2026-08-24 (D2), from [-1.20, 1.40].
#
# The old window was sized against the ANALYTIC MOCK, whose memory window maxes
# out near 1.2 V. The real device does not behave like that. Measured on
# Sentaurus with the user's own calibrated deck at the nominal design point: the
# programmed state draws 9.8e-06 A at -1.2 V and never switches off anywhere in
# the old window, while the erased state sits at 1.3e-11 A until +0.77 V. Three
# million to one, and only ONE of the two thresholds visible. At Pr = 15 uC/cm2
# the film puts ~9e13 cm^-2 into the channel, about a hundred times what it takes
# to inver it, so the window is volts wide and 2.6 V could never contain it.
#
# [-3.5, 1.5] is the range the user's own read sweep already covers, so it costs
# nothing on the Sentaurus side. V_read = 0.60 and V_leak = 0.246391250 both stay
# inside it, so the frozen circuit constants are untouched.
VG_MIN = -3.50
VG_MAX = 1.50
DEFAULT_VG_GRID = np.linspace(VG_MIN, VG_MAX, NVG, dtype=np.float64)

# Order is load-bearing: it fixes the row order of the 7 x D Jacobian everywhere.
DIFFERENTIABLE_OUTPUTS = (
    "ss",
    "vth_fwd",
    "vth_rev",
    "i_leak",
    "g_lo",
    "g_hi",
    "dg_dvth",
)


class OracleInput(BaseModel):
    """One device evaluation request. `theta` is ALWAYS normalised to [0, 1]^D."""

    theta: Differentiable[Array[(None,), Float64]] = Field(
        description="Design vector, normalised to [0,1]. D must be one of 3, 5, 12; "
        "see diffsilicon.shared.design."
    )
    vg_grid: Array[(NVG,), Float64] = Field(
        description=f"Gate-voltage sweep points, V. Fixed at {NVG} points. Non-differentiable."
    )
    vds_lin: Float64 = Field(default=0.05, description="Linear-region drain bias, V.")
    vds_sat: Float64 = Field(default=0.80, description="Saturation-region drain bias, V.")

    @model_validator(mode="after")
    def _validate_theta(self) -> Self:
        # During abstract_eval this field is a ShapeDType, not an array, so ask
        # for .shape before falling back to materialising anything.
        shape = getattr(self.theta, "shape", None)
        if shape is None:
            shape = np.shape(self.theta)
        if len(shape) == 0:
            return self
        d = int(shape[-1])
        if d not in DESIGN_VECTORS:
            raise ValueError(
                f"theta has dimension {d}; the frozen design vectors are "
                f"{sorted(DESIGN_VECTORS)}"
            )
        return self


class OracleOutput(BaseModel):
    """Seven smooth figures of merit plus the raw curve and provenance.

    The seven differentiable fields are all produced by
    `diffsilicon.shared.extract.extract_foms`, i.e. smoothing happens INSIDE the
    oracle, before any differencing can see a staircase.
    """

    # --- differentiable: the 7 smooth FoMs. J is 7 x D. ---
    ss: Differentiable[Float64] = Field(
        description="Subthreshold swing, mV/dec. Soft-weighted LS on log10(Id), forward branch."
    )
    vth_fwd: Differentiable[Float64] = Field(
        description="Threshold voltage of the forward (erased, high-Vth) branch, V."
    )
    vth_rev: Differentiable[Float64] = Field(
        description="Threshold voltage of the reverse (programmed, low-Vth) branch, V."
    )
    i_leak: Differentiable[Float64] = Field(
        description="Drain current at V_leak, A, from the fitted subthreshold line."
    )
    g_lo: Differentiable[Float64] = Field(
        description="Id_fwd(V_read)/V_ds, S. The low-conductance synaptic state."
    )
    g_hi: Differentiable[Float64] = Field(
        description="Id_rev(V_read)/V_ds, S. The high-conductance synaptic state."
    )
    dg_dvth: Differentiable[Float64] = Field(
        description="|dg/dV_th| at V_read, S/V, from a local quadratic. Positive by convention."
    )

    # --- non-differentiable ---
    id_vg: Array[(2, NVG), Float64] = Field(
        description="Raw double sweep: row 0 forward, row 1 reverse. Amps."
    )
    converged: Float64 = Field(
        description="1.0 if the underlying solver converged, else 0.0."
    )
    solver_seconds: Float64 = Field(description="Wall-clock seconds inside the solver.")

    # NOTE ON PROVENANCE. `backend` and `content_hash` are deliberately NOT fields
    # here. Every leaf of a Tesseract output that crosses into JAX has to be an
    # array -- tesseract_jax.apply_tesseract raises
    # `TypeError: string indices must be integers` on any non-array leaf -- so a
    # string field in this schema would make the whole pipeline undifferentiable.
    # (That crash is an upstream bug and a two-line fix; see docs/UPSTREAM.md.)
    #
    # Nothing is lost. Both values are recorded, per call, in the content-addressed
    # cache record AND appended to results/runs/provenance.jsonl. An append-only
    # on-disk log is better evidence than a return field anyway: it is what answers
    # "was the forward pass ever a surrogate", and a return field could not.


def make_oracle_input(
    theta: np.ndarray,
    vg_grid: np.ndarray | None = None,
    vds_lin: float = 0.05,
    vds_sat: float = 0.80,
) -> OracleInput:
    """Convenience constructor that supplies the fixed sweep grid."""
    return OracleInput(
        theta=np.asarray(theta, dtype=np.float64),
        vg_grid=DEFAULT_VG_GRID.copy() if vg_grid is None else np.asarray(vg_grid, np.float64),
        vds_lin=vds_lin,
        vds_sat=vds_sat,
    )
