"""THE LOCKED HZO MATERIAL CONSTANTS.

These three numbers are not design variables. They are the output of a
calibration campaign against measured hysteresis, and they are the reason this
project's device is a real device rather than a plausible one.

Provenance
----------
Locked node ``cal_n16`` of the GAA-FeFET thesis calibration, reported in
``Simulations/Calibration/HYSTERESIS_CALIBRATION_RESULT.md`` under "Calibrated
parameters & the real-world physics extracted":

    FE: P_r, P_s, F_c = 32, 40 uC/cm^2, 1.4 MV/cm
    "calibrated HZO Preisach to Liao's real loop"

That fit reproduces Liao 2022 Fig. 7 branch by branch: memory window 1.296 V
against Liao's quoted 1.30 V at a constant-current criterion of 1e-8 A per
nanosheet, V_th,PGM = -0.934 V, V_th,ERS = +0.362 V, I_on = 4.8 uA/sheet,
I_on/I_off ~ 1.8e7, SS = 45-75 mV/dec.

Why they are LOCKED (2026-08-24, D3 recalibration)
--------------------------------------------------
Until now `Pr` and `Ec` were entries in the design vector and the optimiser was
free to move them. That is the wrong problem. Remanent polarization and coercive
field are properties OF A MATERIAL: you get them by depositing a different film
and re-running the calibration, not by asking a fab for a different value the way
you can ask for a different gate length. Optimising over them produces a device
that cannot be built and, worse, a memory window that can be improved by fiat --
which is exactly the knob a judge would reach for first.

So the design vector is now the FABRICATION knobs, the things a process engineer
can actually specify: ferroelectric thickness, gate length, channel doping and
interfacial-layer thickness. See `design.LOCKED_MATERIAL` and the d=4 vector.

The lock is enforced, not merely documented: `diffsilicon.optimise` refuses to
run on a design vector that exposes either of these. See
`design.tunes_locked_material`.

Units are the ones the design vector used, so that anything that reads a locked
value and anything that reads a swept value are speaking the same language.
"""

from __future__ import annotations

__all__ = ["PR_UC_CM2", "PS_UC_CM2", "EC_MV_CM", "SQUARENESS", "HZO_CALIBRATION"]

PR_UC_CM2 = 32.0  # uC/cm^2 -- remanent polarization
PS_UC_CM2 = 40.0  # uC/cm^2 -- saturation polarization
EC_MV_CM = 1.4  # MV/cm -- coercive field

# Loop squareness Pr/Ps = 0.80, taken from the calibrated pair rather than from
# the Miller `Gamma` knob. Gamma is a modelling parameter of the d=12 vector; on
# the locked material the squareness is measured, so it is not free either.
SQUARENESS = PR_UC_CM2 / PS_UC_CM2

HZO_CALIBRATION = {
    "node": "cal_n16",
    "source": "Simulations/Calibration/HYSTERESIS_CALIBRATION_RESULT.md",
    "reference": "Liao et al. 2022, Fig. 7 quasi-static Id-Vg hysteresis, +-3.5 V, V_DS = 0.2 V",
    "Pr_uC_cm2": PR_UC_CM2,
    "Ps_uC_cm2": PS_UC_CM2,
    "Ec_MV_cm": EC_MV_CM,
    "MW_V": 1.296,
    "i_ref_criterion_A_per_sheet": 1e-8,
    "SS_mV_per_dec": (45.0, 75.0),
}
