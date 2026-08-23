"""Design-vector definitions for the FeFET process space.

Dependency-free on purpose: imported by the frozen contract, by both oracles, by
the T1 remote driver, and by tests. `theta` is ALWAYS normalised to [0, 1]^D on
the wire; physical units exist only inside an oracle.

FROZEN 2026-08-23 (D1). Adding a new design vector is allowed; changing the
name, order, or bounds of an existing one is not -- the replay cache is keyed on
the physical values these bounds produce.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ThetaSpec",
    "DESIGN_VECTORS",
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
    Param("Pr", 5.0, 25.0, "uC/cm2", 15.0),
    Param("Ec", 0.8, 2.0, "MV/cm", 1.2),
)

# --- d=5: + geometry / channel doping ------------------------------------------
_D5 = _D3 + (
    Param("L_g", 20.0, 60.0, "nm", 40.0),
    Param("log10_N_ch", 16.0, 18.0, "log10(cm^-3)", 17.0),
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
    5: ThetaSpec("d5-material+geometry", _D5),
    12: ThetaSpec("d12-full-process", _D12),
}


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
