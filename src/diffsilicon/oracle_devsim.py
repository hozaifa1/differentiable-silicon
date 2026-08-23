"""T2 open oracle: DEVSIM + a clean-room Miller ferroelectric gate. Built on D2.

Clean-room from Miller & McWhorter, J. Appl. Phys. 72, 5999 (1992). QS-Devsim
cannot be used here: it is non-commercial-licensed and covered by patent
CN 113297818 B, both incompatible with the Apache-2.0 this repository ships under.

The module exists on D1 so that the T2 Tesseract image builds and passes
`tesseract-runtime check` in CI today -- G3 is gated on the container existing,
not on the physics being finished.
"""

from __future__ import annotations

from .shared.contract import OracleInput

__all__ = ["id_vg_curves"]


def id_vg_curves(inputs: OracleInput):
    raise NotImplementedError(
        "The DEVSIM FeFET oracle lands on D2 (gate G5, 20:00). Until then use "
        "ORACLE_BACKEND=mock for the Tier A pipeline test, or ORACLE_BACKEND=replay "
        "against results/cache/."
    )
