"""Backend dispatch: one entry point every Tesseract calls to evaluate a device.

    ORACLE_BACKEND = mock | devsim | sentaurus | replay | url

`url` forwards to a served Tesseract at ORACLE_URL and is how the shim reaches a
containerised oracle. Because T1 and T2 publish the identical frozen schema, the
same shim, transducer, network and optimiser run against either one with nothing
changed but that variable.

This function is the seam the whole project is built around. T1 and T2 differ
only in which branch of it runs; nothing downstream -- not the shim, not the
transducer, not the network, not the optimiser -- can tell them apart except by
reading the `backend` string that comes back. That is what makes "swapping the
closed-source commercial solver for the Apache-2.0 one is one environment
variable" a fact about the code rather than a claim in a README.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .cache import CacheStore, cache_key, content_hash, encode_output
from .circuit import load_circuit
from .contract import OracleInput, OracleOutput
from .design import get_design
from .extract import ExtractionConfig, extract_foms

__all__ = ["device_geometry", "extraction_config", "run_oracle", "VALID_BACKENDS"]

VALID_BACKENDS = ("mock", "devsim", "sentaurus", "replay", "url")


_PROVENANCE_LOG = Path(
    os.environ.get(
        "DIFFSILICON_PROVENANCE_LOG",
        Path(__file__).resolve().parents[3] / "results" / "runs" / "provenance.jsonl",
    )
)


def _from_record(rec: dict) -> OracleOutput:
    fields = set(OracleOutput.model_fields)
    kwargs = {k: v for k, v in rec.items() if k in fields and k != "id_vg"}
    return OracleOutput(**kwargs, id_vg=np.asarray(rec["id_vg"], dtype=np.float64))


def _log_provenance(rec: dict) -> None:
    """Append backend + content hash for EVERY forward evaluation, cache hits included.

    This file, not any return value, is the answer to "is the forward pass a
    surrogate". It is append-only and it records the backend that actually
    produced each number and the hash of the inputs it was produced at.
    """
    if os.environ.get("DIFFSILICON_PROVENANCE_DISABLE") == "1":
        return
    line = {
        "t": time.time(),
        "backend": rec.get("backend"),
        "hash": rec.get("content_hash"),
        "solver_seconds": rec.get("solver_seconds"),
        "converged": rec.get("converged"),
        "ss": rec.get("ss"),
    }
    # Same reason CacheStore.put tolerates an unwritable root: inside a built
    # Tesseract this path resolves under the filesystem root, which the container
    # user cannot create. An audit trail that cannot be written must not take the
    # solve down with it. DIFFSILICON_PROVENANCE_LOG points it somewhere writable.
    try:
        _PROVENANCE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_PROVENANCE_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, separators=(",", ":")) + "\n")
    except OSError:
        return


def device_geometry(theta_n) -> tuple[float, float]:
    """(W, L_g) in nm for this design point, filling in the frozen defaults.

    Shared by every backend so that the constant-current V_th criterion
    I_crit = 100 nA * W / L_g means the same thing on all of them. If the mock
    and the solver disagreed about W/L, their thresholds would be offset by a
    constant and V4 (cross-solver sign agreement) would be measuring that offset
    instead of the physics.
    """
    theta_n = np.asarray(theta_n, dtype=np.float64).ravel()
    spec = get_design(int(theta_n.shape[0]))
    phys = dict(zip(spec.names, spec.lo + theta_n * (spec.hi - spec.lo), strict=True))
    cc = load_circuit()
    return float(phys.get("W_dev", cc.w_dev_nm)), float(phys.get("L_g", 40.0))


def extraction_config(theta_n) -> ExtractionConfig:
    cc = load_circuit()
    w, lg = device_geometry(theta_n)
    return ExtractionConfig(
        v_read=cc.v_read,
        v_leak=cc.v_leak,
        i_crit_per_wl=cc.i_crit_per_wl,
        w_dev_nm=w,
        l_g_nm=lg,
    )


def _mock_curves(inputs: OracleInput):
    from .mock_device import id_vg_curves

    return id_vg_curves(np.asarray(inputs.theta), np.asarray(inputs.vg_grid), inputs.vds_lin)


def _devsim_curves(inputs: OracleInput):
    from ..oracle_devsim import id_vg_curves as devsim_curves  # built on D2

    return devsim_curves(inputs)


def _sentaurus_curves(inputs: OracleInput):
    from ..t1_driver import id_vg_curves as sentaurus_curves

    return sentaurus_curves(inputs)


_CURVE_BACKENDS = {
    "mock": _mock_curves,
    "devsim": _devsim_curves,
    "sentaurus": _sentaurus_curves,
}


def run_oracle(inputs: OracleInput, backend: str | None = None) -> OracleOutput:
    backend = backend or os.environ.get("ORACLE_BACKEND", "mock")
    if backend not in VALID_BACKENDS:
        raise ValueError(f"ORACLE_BACKEND={backend!r}; expected one of {VALID_BACKENDS}")

    key = cache_key(inputs, backend)
    input_hash = content_hash(inputs)

    if backend == "url":
        # Forward to a served Tesseract. The remote publishes the same frozen
        # OracleOutput, so nothing here needs to know which solver it is.
        from tesseract_core import Tesseract

        url = os.environ.get("ORACLE_URL")
        if not url:
            raise ValueError("ORACLE_BACKEND=url requires ORACLE_URL to be set.")
        with Tesseract.from_url(url) as t:
            rec = t.apply(
                {
                    "theta": np.asarray(inputs.theta, dtype=np.float64),
                    "vg_grid": np.asarray(inputs.vg_grid, dtype=np.float64),
                    "vds_lin": float(inputs.vds_lin),
                    "vds_sat": float(inputs.vds_sat),
                }
            )
        out = _from_record(rec)
        # The record that comes back over the wire holds numpy arrays, and every
        # other backend reaches the cache through `encode_output`. This branch
        # did not, so the first real Tier B call died on
        # `TypeError: Object of type ndarray is not JSON serializable` -- after
        # the container had already solved, which is the expensive part.
        #
        # It also could not have stamped the record correctly if it had worked:
        # a served Tesseract cannot report which solver it is, because `backend`
        # is a string and every output leaf has to be an array to cross into JAX
        # (docs/UPSTREAM.md, item 1). So the stamp is the thing that IS known
        # here -- the URL the value came from -- and ORACLE_URL_BACKEND names the
        # solver when the operator knows it. The directory stays `url/` either
        # way, so a value that came over a wire is never filed as one that did
        # not.
        label = os.environ.get("ORACLE_URL_BACKEND") or f"url:{url}"
        stamped = encode_output(out, label, input_hash)
        CacheStore("url").put(key, stamped)
        _log_provenance(stamped)
        return out

    if backend == "replay":
        src = os.environ.get("ORACLE_REPLAY_SOURCE", "sentaurus")
        rec = CacheStore(src).get(cache_key(inputs, src))
        if rec is None:
            raise KeyError(
                f"replay: no cached {src} result for {cache_key(inputs, src)[:16]}... . "
                f"The replay cache only "
                f"covers design points this project actually evaluated; it is a reproduction "
                f"path, not a surrogate."
            )
        _log_provenance(rec)
        return _from_record(rec)

    store = CacheStore(backend)
    if os.environ.get("DIFFSILICON_CACHE_DISABLE") != "1":
        rec = store.get(key)
        if rec is not None:
            _log_provenance(rec)
            return _from_record(rec)

    t0 = time.perf_counter()
    curves = _CURVE_BACKENDS[backend](inputs)
    solver_seconds = time.perf_counter() - t0

    curves = jnp.asarray(curves, dtype=jnp.float64)
    cfg = extraction_config(inputs.theta)
    foms = extract_foms(
        jnp.asarray(np.asarray(inputs.vg_grid), dtype=jnp.float64),
        curves[0],
        curves[1],
        cfg,
        float(inputs.vds_lin),
    )

    out = OracleOutput(
        ss=float(foms.ss),
        vth_fwd=float(foms.vth_fwd),
        vth_rev=float(foms.vth_rev),
        i_leak=float(foms.i_leak),
        g_lo=float(foms.g_lo),
        g_hi=float(foms.g_hi),
        dg_dvth=float(foms.dg_dvth),
        id_vg=np.asarray(curves, dtype=np.float64),
        # "Converged" now means two things, because two different failures
        # produce a number that looks perfectly usable. The solver has to have
        # returned finite currents, AND the extraction has to have found both
        # thresholds INSIDE the voltages actually swept. A threshold read off
        # the extrapolation past the end of the sweep is not a measurement, and
        # before 2026-08-27 nothing said so -- see shared/extract.py.
        converged=float(
            np.all(np.isfinite(np.asarray(curves))) and float(foms.vth_in_range) > 0.5
        ),
        solver_seconds=solver_seconds,
    )

    # Populate as a side effect of EVERY run. Tier C depends on this line.
    rec = encode_output(out, backend=backend, content_hash=input_hash)
    store.put(key, rec)
    _log_provenance(rec)
    return out
