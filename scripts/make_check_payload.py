"""Emit a payload for `tesseract-runtime check-gradients` (V3, run in CI).

V3 is the organizers' own gradient checker pointed at our endpoints. Using their
tool rather than only our own tests is the point: it is an independent
implementation of the same question.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diffsilicon.shared.contract import DEFAULT_VG_GRID  # noqa: E402
from diffsilicon.shared.design import nominal_theta  # noqa: E402


def oracle_payload(D: int = 5) -> dict:
    return {
        "inputs": {
            "theta": nominal_theta(D).tolist(),
            "vg_grid": DEFAULT_VG_GRID.tolist(),
            "vds_lin": 0.05,
            "vds_sat": 0.80,
        }
    }


def snn_payload() -> dict:
    # The nominal operating point, from config/circuit.yaml and the nominal device.
    return {
        "inputs": {
            "beta": 0.6033579,
            "g_min": 2.5592765e-05,
            "g_max": 2.0014193e-04,
            "th_th": 5.0,
            "sig_w": 0.2273107,
            "seed": 0,
            "batch": 32,
            # Gradient-check the smooth relaxation, which is the function the
            # surrogate is the exact derivative of. Finite-differencing a Heaviside
            # yields either 0 or one spike-flip, so a hard forward cannot be
            # gradient-checked by anyone, and pretending otherwise would be theatre.
            "smooth_spikes": True,
        }
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snn", action="store_true", help="payload for snn-lif-ecg instead")
    ap.add_argument("-D", type=int, default=5)
    args = ap.parse_args()
    json.dump(snn_payload() if args.snn else oracle_payload(args.D), sys.stdout)
