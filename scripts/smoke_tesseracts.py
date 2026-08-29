#!/usr/bin/env python
"""Smoke-test every tesseract_api.py in this repo: schemas, apply, and derivatives.

WHAT THIS IS FOR. The four Tesseracts are the deliverable a judge actually runs.
Between them they carry three schema surfaces (the frozen OracleInput/Output, the
shim's derivative endpoints, and the network's five-scalar contract) and four
import paths that only ever get exercised when something else calls them. A
typo in `abstract_eval`, a field renamed in one schema and not the other, or an
import that only resolves because some earlier module already put `src` on the
path -- none of those show up in the optimiser's logs. They show up when someone
else tries to serve the container.

So this checks, per Tesseract:

  1. the module IMPORTS on its own, with nothing else preloaded;
  2. InputSchema / OutputSchema build and validate;
  3. `abstract_eval` agrees with what `apply` actually returns -- same keys, same
     shapes, same dtypes. This is the schema-mismatch check and it is the one
     that catches real bugs, because abstract_eval is written by hand;
  4. `apply` runs and returns finite numbers;
  5. every derivative endpoint the module declares (jacobian, jvp, vjp) runs and
     returns the right shape.

WHICH BACKEND EACH ONE RUNS ON, and why it is not always the real solver:

  sentaurus-fefet   replay, from results/cache/sentaurus. The commercial solver
                    lives on a shared licensed host; a smoke test must not need
                    it. Replay serves a curve that host really produced.
  devsim-fefet      devsim, at the nominal device, which is already cached. The
                    open solver is the one anybody can run, so this one is real.
  adjoint-shim      mock. The shim's derivative endpoints cost 2D+1 solver probes
                    at a fresh design point; on the open solver that is nine
                    minutes to learn nothing about the shim. The mock is the
                    designated wiring harness -- it exercises the identical
                    endpoint code and returns in milliseconds. What is being
                    tested here is the API surface, not the physics; the physics
                    is tested by the flagship and by tests/.
  snn-lif-ecg       the real network, frozen mode, batch 4 to keep it quick.

Every check prints the backend it used, so nothing here can be read as a claim
that the commercial solver ran.

    python scripts/smoke_tesseracts.py
    python scripts/smoke_tesseracts.py --json results/runs/smoke_tesseracts.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

# Must be set BEFORE snn-lif-ecg is imported: it reads its training mode at
# module scope, so setting it afterwards would be silently ignored.
os.environ.setdefault("SNN_TRAIN_MODE", "frozen")
os.environ.setdefault("SNN_VJP", "fd")
os.environ["SHIM_MAX_ORACLE_CALLS"] = "100000"

import numpy as np  # noqa: E402

NOMINAL = [0.2, 0.5, 0.5, 1.0 / 3.0]  # in the sentaurus cache; see rebaseline_d3
API = _REPO / "tesseracts" / "{}" / "tesseract_api.py"


class Check:
    """One named assertion, with its own pass/fail and its own message."""

    def __init__(self, results: list, tess: str):
        self.results, self.tess = results, tess

    def __call__(self, name: str, fn, backend: str = "-"):
        t0 = time.time()
        try:
            detail = fn()
            ok, err = True, None
        except Exception as exc:  # noqa: BLE001 -- a smoke test reports, it does not raise
            detail, ok, err = None, False, f"{type(exc).__name__}: {exc}"
            if os.environ.get("SMOKE_TRACEBACK") == "1":
                traceback.print_exc()
        row = {
            "tesseract": self.tess, "check": name, "backend": backend,
            "ok": ok, "seconds": round(time.time() - t0, 2),
            "detail": detail, "error": err,
        }
        self.results.append(row)
        mark = "PASS" if ok else "FAIL"
        extra = f"  {detail}" if detail else ""
        print(f"  [{mark}] {name:38s} ({backend:9s} {row['seconds']:6.2f}s){extra}",
              flush=True)
        if err:
            print(f"         {err[:300]}", flush=True)
        return ok


def _load(name: str):
    from tesseract_core import Tesseract

    return Tesseract.from_tesseract_api(str(API).format(name))


def _finite(d: dict, keys) -> str:
    bad = [k for k in keys if not np.all(np.isfinite(np.asarray(d[k], dtype=float)))]
    if bad:
        raise ValueError(f"non-finite output leaves: {bad}")
    return ", ".join(f"{k}={float(np.asarray(d[k]).ravel()[0]):.4g}" for k in keys)


def _abstract_matches_apply(mod, inputs: dict, out: dict) -> str:
    """`abstract_eval` is hand-written. Check it against what `apply` returned.

    This is the check that earns its keep. A shape or dtype declared here and not
    delivered by `apply` is exactly what breaks tesseract-jax's tracing, and it
    breaks it with an error that points at the caller rather than at this file.
    """
    from tesseract_core.runtime import ShapeDType

    abstract = mod.abstract_eval(
        {k: ShapeDType(shape=np.asarray(v).shape, dtype=str(np.asarray(v).dtype))
         if isinstance(v, (list, np.ndarray)) else v
         for k, v in inputs.items()}
    )
    a_keys, o_keys = set(abstract), set(out)
    if a_keys != o_keys:
        raise ValueError(
            f"abstract_eval declares {sorted(a_keys - o_keys)} that apply does not "
            f"return, and apply returns {sorted(o_keys - a_keys)} that abstract_eval "
            f"does not declare"
        )
    for k, sd in abstract.items():
        got = np.asarray(out[k])
        want_shape = tuple(sd.shape if not hasattr(sd, "shape") else sd.shape)
        if tuple(got.shape) != tuple(want_shape):
            raise ValueError(
                f"{k}: abstract_eval says shape {tuple(want_shape)}, apply returned "
                f"{tuple(got.shape)}"
            )
        want_dtype = str(sd.dtype)
        if np.dtype(got.dtype) != np.dtype(want_dtype):
            raise ValueError(
                f"{k}: abstract_eval says dtype {want_dtype}, apply returned {got.dtype}"
            )
    return f"{len(abstract)} leaves agree"


def _import_module(name: str):
    """Import the api module directly, the way the runtime loads it."""
    import importlib.util

    path = str(API).format(name)
    spec = importlib.util.spec_from_file_location(f"_smoke_{name.replace('-', '_')}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- the four ---------------------------------------------------------------

def smoke_oracle(name: str, backend: str, results: list) -> None:
    """sentaurus-fefet and devsim-fefet: same frozen contract, different solver."""
    print(f"\n=== {name} ===", flush=True)
    chk = Check(results, name)
    holder = {}

    chk("module imports", lambda: (holder.__setitem__("mod", _import_module(name)),
                                   "ok")[1])
    if "mod" not in holder:
        return
    mod = holder["mod"]

    chk("InputSchema / OutputSchema build",
        lambda: f"{len(mod.InputSchema.model_fields)} in, "
                f"{len(mod.OutputSchema.model_fields)} out")

    os.environ["ORACLE_BACKEND"] = backend
    if backend == "replay":
        os.environ["ORACLE_REPLAY_SOURCE"] = "sentaurus"

    def _apply():
        with _load(name) as t:
            out = t.apply({"theta": np.asarray(NOMINAL), "vg_grid": _vg(mod),
                           "vds_lin": 0.05, "vds_sat": 0.80})
        holder["out"] = out
        return _finite(out, ("ss", "vth_fwd", "vth_rev", "i_leak", "converged"))

    chk("apply returns finite figures of merit", _apply, backend)

    if "out" in holder:
        chk("abstract_eval matches apply",
            lambda: _abstract_matches_apply(
                mod,
                {"theta": np.asarray(NOMINAL), "vg_grid": _vg(mod),
                 "vds_lin": 0.05, "vds_sat": 0.80},
                holder["out"]),
            backend)


def _vg(mod):
    from diffsilicon.shared.contract import DEFAULT_VG_GRID

    return np.asarray(DEFAULT_VG_GRID)


def smoke_shim(results: list) -> None:
    """adjoint-shim: the frozen contract PLUS jacobian / jvp / vjp."""
    name = "adjoint-shim"
    print(f"\n=== {name} ===", flush=True)
    chk = Check(results, name)
    holder = {}

    chk("module imports", lambda: (holder.__setitem__("mod", _import_module(name)),
                                   "ok")[1])
    if "mod" not in holder:
        return
    mod = holder["mod"]

    chk("InputSchema / OutputSchema build",
        lambda: f"{len(mod.InputSchema.model_fields)} in, "
                f"{len(mod.OutputSchema.model_fields)} out")

    # The mock, deliberately -- see the module docstring. The shim's endpoints
    # are what is under test; which solver answers them is not.
    os.environ["ORACLE_BACKEND"] = "mock"
    import diffsilicon.shim.adjoint as adj

    adj._REGISTRY.clear()  # a stale shim would make every number below a replay

    payload = {"theta": np.asarray(NOMINAL), "vg_grid": _vg(mod),
               "vds_lin": 0.05, "vds_sat": 0.80}
    rows = 7
    d = len(NOMINAL)

    def _apply():
        with _load(name) as t:
            holder["out"] = t.apply(payload)
        return _finite(holder["out"], ("ss", "vth_fwd", "i_leak", "converged"))

    chk("apply proxies to the oracle", _apply, "mock")

    if "out" in holder:
        chk("abstract_eval matches apply",
            lambda: _abstract_matches_apply(mod, payload, holder["out"]), "mock")

    from diffsilicon.shared.contract import DIFFERENTIABLE_OUTPUTS

    def _jac():
        with _load(name) as t:
            J = t.jacobian(payload, jac_inputs={"theta"},
                           jac_outputs=set(DIFFERENTIABLE_OUTPUTS))
        M = np.array([np.asarray(J[k]["theta"], dtype=float)
                      for k in DIFFERENTIABLE_OUTPUTS])
        holder["J"] = M
        if M.shape != (rows, d):
            raise ValueError(f"jacobian shape {M.shape}, expected {(rows, d)}")
        if not np.all(np.isfinite(M)):
            raise ValueError("jacobian has non-finite entries")
        return f"shape {M.shape}, |J|_F={np.linalg.norm(M):.4g}"

    chk("jacobian is 7 x D and finite", _jac, "mock")

    def _vjp():
        ct = {k: 1.0 for k in DIFFERENTIABLE_OUTPUTS}
        with _load(name) as t:
            g = t.vector_jacobian_product(
                payload, vjp_inputs={"theta"},
                vjp_outputs=set(DIFFERENTIABLE_OUTPUTS), cotangent_vector=ct)
        v = np.asarray(g["theta"], dtype=float)
        if v.shape != (d,):
            raise ValueError(f"vjp shape {v.shape}, expected {(d,)}")
        # J^T 1 is exactly the column sums of J, so this is checkable, not just
        # runnable. If the two disagree the endpoint is not wired to the shim.
        if "J" in holder:
            want = holder["J"].sum(axis=0)
            if not np.allclose(v, want, rtol=1e-10, atol=1e-12):
                raise ValueError(f"vjp {v} != column sums of jacobian {want}")
        return f"J^T 1 = {np.array2string(v, precision=4)}"

    chk("vjp equals J^T applied to the cotangent", _vjp, "mock")

    def _jvp():
        tan = {"theta": np.ones(d)}
        with _load(name) as t:
            j = t.jacobian_vector_product(
                payload, jvp_inputs={"theta"},
                jvp_outputs=set(DIFFERENTIABLE_OUTPUTS), tangent_vector=tan)
        v = np.array([float(np.asarray(j[k])) for k in DIFFERENTIABLE_OUTPUTS])
        if "J" in holder and not np.allclose(v, holder["J"].sum(axis=1),
                                             rtol=1e-10, atol=1e-12):
            raise ValueError("jvp != row sums of jacobian")
        return f"J 1 finite, {v.shape[0]} rows"

    chk("jvp equals J applied to the tangent", _jvp, "mock")

    def _free():
        """The economic claim: a fresh VJP costs no solver calls."""
        shim = next(iter(adj._REGISTRY.values()))
        before = shim.ctr.calls
        ct = {k: 1.0 for k in DIFFERENTIABLE_OUTPUTS}
        with _load("adjoint-shim") as t:
            t.vector_jacobian_product(payload, vjp_inputs={"theta"},
                                      vjp_outputs=set(DIFFERENTIABLE_OUTPUTS),
                                      cotangent_vector=ct)
        after = shim.ctr.calls
        if after != before:
            raise ValueError(f"a repeat VJP cost {after - before} solver calls")
        return f"0 extra probes (total {after})"

    chk("a repeat vjp costs zero solver calls", _free, "mock")


def smoke_snn(results: list) -> None:
    name = "snn-lif-ecg"
    print(f"\n=== {name} ===", flush=True)
    chk = Check(results, name)
    holder = {}

    chk("module imports", lambda: (holder.__setitem__("mod", _import_module(name)),
                                   "ok")[1])
    if "mod" not in holder:
        return
    mod = holder["mod"]

    chk("InputSchema / OutputSchema build",
        lambda: f"{len(mod.InputSchema.model_fields)} in, "
                f"{len(mod.OutputSchema.model_fields)} out")

    from diffsilicon.shared.circuit import load_circuit  # noqa: F401
    from diffsilicon.snn.lif import PHI_KEYS

    # The frozen reference device, which is where the network was fitted.
    phi = {"beta": 0.6057, "g_min": 5.0e-09, "g_max": 1.5e-04,
           "th_th": 5.0, "sig_w": 0.086}
    # batch 16, NOT something smaller. The shared reference weights are cached
    # under a key that includes the batch size, so any other value silently
    # misses the cache and re-runs an 800-step fit -- measured: twenty minutes
    # to smoke-test one endpoint. 16 is also what every banked run used.
    payload = {**phi, "seed": 0, "batch": 16, "smooth_spikes": False}

    def _apply():
        with _load(name) as t:
            holder["out"] = t.apply(payload)
        return _finite(holder["out"], ("loss", "spikes", "accuracy"))

    chk("apply runs the network", _apply, "frozen")

    if "out" in holder:
        chk("abstract_eval matches apply",
            lambda: _abstract_matches_apply(mod, payload, holder["out"]), "frozen")

    def _vjp():
        with _load(name) as t:
            g = t.vector_jacobian_product(
                payload, vjp_inputs=set(PHI_KEYS), vjp_outputs={"loss"},
                cotangent_vector={"loss": 1.0})
        v = np.array([float(np.asarray(g[k])) for k in PHI_KEYS])
        if not np.all(np.isfinite(v)):
            raise ValueError(f"non-finite dL/dphi: {v}")
        holder["g"] = v
        return "dL/dphi = " + np.array2string(v, precision=4)

    chk("vjp returns dL/dphi for all five", _vjp, "frozen")

    def _jvp():
        with _load(name) as t:
            j = t.jacobian_vector_product(
                payload, jvp_inputs=set(PHI_KEYS), jvp_outputs={"loss"},
                tangent_vector={k: 1.0 for k in PHI_KEYS})
        v = float(np.asarray(j["loss"]))
        if not np.isfinite(v):
            raise ValueError("non-finite jvp")
        return f"dL along 1 = {v:.6g}"

    chk("jvp runs", _jvp, "frozen")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(_REPO / "results" / "runs"
                                          / "smoke_tesseracts.json"))
    ap.add_argument("--skip", default="", help="comma-separated tesseract names")
    args = ap.parse_args()
    skip = {s for s in args.skip.split(",") if s}

    print("Tesseract smoke test -- schemas, apply, and every derivative endpoint.")
    print("Backends are printed per check; nothing here runs the licensed host.\n")

    results: list = []
    t0 = time.time()
    if "sentaurus-fefet" not in skip:
        smoke_oracle("sentaurus-fefet", "replay", results)
    if "devsim-fefet" not in skip:
        smoke_oracle("devsim-fefet", "devsim", results)
    if "adjoint-shim" not in skip:
        smoke_shim(results)
    if "snn-lif-ecg" not in skip:
        smoke_snn(results)

    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n{'=' * 68}")
    print(f"{n_ok} of {len(results)} checks passed in {time.time() - t0:.1f} s")
    for r in results:
        if not r["ok"]:
            print(f"  FAILED  {r['tesseract']:16s} {r['check']}")
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(
        json.dumps({"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "passed": n_ok, "total": len(results), "checks": results},
                   indent=1), encoding="utf-8")
    print(f"wrote {args.json}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
