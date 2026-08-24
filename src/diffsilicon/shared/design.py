"""Design-vector definitions for the FeFET process space.

Dependency-free on purpose: imported by the frozen contract, by both oracles, by
the T1 remote driver, and by tests. `theta` is ALWAYS normalised to [0, 1]^D on
the wire; physical units exist only inside an oracle.

FROZEN 2026-08-23 (D1), with ONE amendment on D2: the Pr ceiling moved 25 -> 40
because the real calibrated film sits at 32 and was outside the box. Adding a new
design vector is allowed. Changing a name, an order, or a bound is not free: the
cache and the provenance log are keyed on NORMALISED theta, so the same key means
a different physical device after a bound moves. `shared.cache.cache_key` now
folds the box definition into the key, so stale entries stop matching instead of
being served for the wrong device. Move a bound only with that in mind.

RECALIBRATED 2026-08-24 (D3): d=4 is now the design vector
----------------------------------------------------------
`Pr` and `Ec` are no longer things this project optimises. They are locked to the
measured HZO film -- see `shared.material` for the numbers and for why. The
design vector is the FABRICATION knobs instead, the four quantities a process
engineer can actually be asked for:

    t_fe   ferroelectric thickness
    L_g    gate length
    N_ch   channel doping
    t_IL   interfacial-layer thickness

The old d=3, d=5 and d=12 vectors are KEPT, not deleted. They are what every
cached result, every banked run in `results/`, and the D2 cross-check were
computed against, and throwing them away would silently invalidate that evidence
rather than supersede it. They are marked instead: `tunes_locked_material`
returns True for each of them, and `diffsilicon.optimise` refuses to descend on
one. So they can still be REPLAYED and compared; they just cannot be optimised.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ThetaSpec",
    "DESIGN_VECTORS",
    "DEFAULT_D",
    "LOCKED_MATERIAL",
    "tunes_locked_material",
    "get_design",
    "denormalise",
    "normalise",
    "nominal_theta",
]


@dataclass(frozen=True)
class Param:
    """One physical design parameter."""

    name: str
    lo: float
    hi: float
    unit: str
    nominal: float  # physical units; must lie in [lo, hi]

    def __post_init__(self) -> None:
        if not (self.lo <= self.nominal <= self.hi):
            raise ValueError(f"{self.name}: nominal {self.nominal} outside [{self.lo}, {self.hi}]")


@dataclass(frozen=True)
class ThetaSpec:
    """An ordered design vector."""

    label: str
    params: tuple[Param, ...]

    @property
    def D(self) -> int:
        return len(self.params)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params)

    @property
    def lo(self) -> np.ndarray:
        return np.array([p.lo for p in self.params], dtype=np.float64)

    @property
    def hi(self) -> np.ndarray:
        return np.array([p.hi for p in self.params], dtype=np.float64)

    @property
    def units(self) -> tuple[str, ...]:
        return tuple(p.unit for p in self.params)


# --- d=3: mesh-invariant material-only branch (the safe branch) -----------------
_D3 = (
    Param("t_fe", 5.0, 15.0, "nm", 10.0),
    # WIDENED 2026-08-24 (D2), from [5, 25]. The measured film this project is
    # aimed at sits above the old ceiling, so the search could never reach it.
    Param("Pr", 5.0, 40.0, "uC/cm2", 15.0),
    Param("Ec", 0.8, 2.0, "MV/cm", 1.2),
)

# --- d=5: + geometry / channel doping ------------------------------------------
_D5 = _D3 + (
    Param("L_g", 20.0, 60.0, "nm", 40.0),
    Param("log10_N_ch", 16.0, 18.0, "log10(cm^-3)", 17.0),
)

# --- d=4: THE DESIGN VECTOR (D3 recalibration) ---------------------------------
# The four fabrication knobs, with Pr and Ec locked to the calibrated film.
#
# Bounds are carried over unchanged from the d=5 and d=12 entries below, so a
# d=4 point and a d=5 point at the same t_fe mean the same physical thickness.
#
# t_fe nominal is 7.0 nm, not the 10.0 the older vectors used. Seven is the
# thickness of the calibrated device: `t1/calibration.local.json` builds its mesh
# at t_fe_slab_nm = 7.0, and the thesis' own device study settled on 7 nm. It
# also makes the Sentaurus fixed-slab remap exact at the nominal point rather
# than merely small -- `deck_values` reports t_fe_snap_error_nm = 0 there.
_D4 = (
    Param("t_fe", 5.0, 15.0, "nm", 7.0),
    Param("L_g", 20.0, 60.0, "nm", 40.0),
    Param("log10_N_ch", 16.0, 18.0, "log10(cm^-3)", 17.0),
    Param("t_IL", 0.5, 2.0, "nm", 1.0),
)

# --- d=12: the headline experiment (DEVSIM only) --------------------------------
_D12 = _D5 + (
    Param("t_IL", 0.5, 2.0, "nm", 1.0),
    Param("log10_N_sd", 19.0, 21.0, "log10(cm^-3)", 20.0),
    Param("x_ov", 0.0, 5.0, "nm", 2.0),
    Param("log10_N_halo", 17.0, 19.0, "log10(cm^-3)", 18.0),
    Param("x_halo", 0.0, 15.0, "nm", 7.5),
    Param("Gamma", 0.0, 1.0, "-", 0.5),  # Miller minor-loop saturation factor
    Param("W_dev", 50.0, 500.0, "nm", 100.0),
)

DESIGN_VECTORS: dict[int, ThetaSpec] = {
    3: ThetaSpec("d3-material", _D3),
    4: ThetaSpec("d4-fabrication", _D4),
    5: ThetaSpec("d5-material+geometry", _D5),
    12: ThetaSpec("d12-full-process", _D12),
}

# Material constants of the calibrated HZO film. A design vector that exposes
# either of these is a LEGACY vector: it may be evaluated and replayed, but it
# may not be optimised. See `shared.material` and `tunes_locked_material`.
LOCKED_MATERIAL = ("Pr", "Ec")

#: The design vector this project optimises after the D3 recalibration.
DEFAULT_D = 4


def tunes_locked_material(spec: ThetaSpec | int) -> tuple[str, ...]:
    """The locked material constants this design vector would let an optimiser move.

    Empty tuple means the vector is safe to descend on. Anything else is a
    legacy vector from before the D3 recalibration.
    """
    if isinstance(spec, int):
        spec = get_design(spec)
    return tuple(n for n in spec.names if n in LOCKED_MATERIAL)


def get_design(D: int) -> ThetaSpec:
    if D not in DESIGN_VECTORS:
        raise ValueError(f"No design vector of dimension {D}; have {sorted(DESIGN_VECTORS)}")
    return DESIGN_VECTORS[D]


def denormalise(theta_n: np.ndarray, spec: ThetaSpec | None = None) -> np.ndarray:
    """[0,1]^D -> physical units."""
    theta_n = np.asarray(theta_n, dtype=np.float64)
    spec = spec or get_design(theta_n.shape[-1])
    return spec.lo + theta_n * (spec.hi - spec.lo)


def normalise(theta_phys: np.ndarray, spec: ThetaSpec | None = None) -> np.ndarray:
    """Physical units -> [0,1]^D."""
    theta_phys = np.asarray(theta_phys, dtype=np.float64)
    spec = spec or get_design(theta_phys.shape[-1])
    return (theta_phys - spec.lo) / (spec.hi - spec.lo)


def nominal_theta(D: int) -> np.ndarray:
    """The nominal device, normalised. V_leak is frozen against exactly this point."""
    spec = get_design(D)
    return normalise(np.array([p.nominal for p in spec.params], dtype=np.float64), spec)
